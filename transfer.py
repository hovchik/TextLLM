"""Move a trained model to another PC and keep training it there.

On the old PC (stop training first, or let it reach a save):
    python transfer.py export --run runs/my-model              -> my-model.textllm.zip
    python transfer.py export --run runs/my-model --with-data  # also the training data, to continue
                                                               # on exactly the same text
On the new PC (inside its own copy of this repo, after run.bat / run.sh set it up once):
    python transfer.py import my-model.textllm.zip
then press Resume on the model's page in the Studio, or run the command that `import` prints.

What goes in the bundle:
  * the whole run folder: weights (ckpt.pt or adapter/), optimizer state and step (ckpt.pt or
    train_state.pt), training settings (train_cmd.json), meta.json, charts (metrics*.jsonl),
    eval history and the fast GGUF copy if there is one. A fine-tune's folder also holds its
    tokenized examples (examples.pt), so it always continues on exactly the same data;
  * from scratch: the tokenizer (as <run>/tokenizer.json). Its numbers must match the weights, so
    never train a new tokenizer for a moved model;
  * --with-data (from scratch): the train.bin + val.bin the run trains on. On import they go into
    the run's own folder (runs/<name>/data/bin) and the run is pointed at them, so they never mix
    with other data on the new PC. Without it, the new PC needs its own data/bin encoded with the
    same tokenizer.
The fine-tune's base model (e.g. Qwen/Qwen3-0.6B) is not included: the new PC downloads it once.
"""
import argparse, json, os, shutil, sys, tempfile, time, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
sys.path.insert(0, ROOT)
from core import commands, paths as core_paths  # noqa: E402
FORMAT = 1
STORED = (".pt", ".bin", ".gguf", ".safetensors", ".zip")  # already dense; zipping them only costs time


def _meta(run_dir):
    p = os.path.join(run_dir, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"{run_dir} is not a trained model (no meta.json)")
    return json.load(open(p))


def export(run_dir, out=None, with_data=False, log=print):
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir.rstrip("/\\"))
    meta = _meta(run_dir)
    out = out or os.path.join(os.getcwd(), f"{name}.textllm.zip")
    files = []  # (path on disk, path in zip)
    for dp, dns, fns in os.walk(run_dir):
        dns[:] = [d for d in dns if not d.endswith(".tmp")]
        for f in fns:
            if f.endswith(".tmp") or f == "log.txt":
                continue
            p = os.path.join(dp, f)
            files.append((p, "run/" + os.path.relpath(p, run_dir).replace(os.sep, "/")))
    if meta["type"] == "scratch" and not os.path.exists(os.path.join(run_dir, "tokenizer.json")):
        from chat_engine import tokenizer_path

        files.append((tokenizer_path(run_dir, meta), "run/tokenizer.json"))
    if with_data and meta["type"] == "scratch":
        d = _data_dir(run_dir)
        if not core_paths.inside(run_dir, d):  # data inside the run folder is already packed
            for f in ("train.bin", "val.bin"):
                if os.path.exists(os.path.join(d, f)):
                    files.append((os.path.join(d, f), "data/bin/" + f))
    manifest = {"format": FORMAT, "name": name, "type": meta["type"], "base": meta.get("base"),
                "created": time.time(), "with_data": with_data}
    tmp = out + ".part"
    total = sum(os.path.getsize(p) for p, _ in files)
    log(f"packing {len(files)} files ({total / 1e6:.1f} MB) into {out}")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        z.writestr("textllm.json", json.dumps(manifest, indent=1))
        for p, arc in files:
            comp = zipfile.ZIP_STORED if p.endswith(STORED) else zipfile.ZIP_DEFLATED
            z.write(p, arc, compress_type=comp)
    os.replace(tmp, out)
    log(f"done: {out}")
    return out


def _data_dir(run_dir):
    """Where this from-scratch run reads train.bin/val.bin from (its --data, else data/bin)."""
    cmd = _saved_cmd(run_dir) or []
    d = next((cmd[i + 1] for i, c in enumerate(cmd[:-1]) if c == "--data"), "data/bin")
    return os.path.abspath(os.path.join(ROOT, d))


def _saved_cmd(run_dir):
    p = os.path.join(run_dir, "train_cmd.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _safe_target(base, parts):
    """Join zip path parts under base, refusing anything that could land outside it
    (.., absolute paths, drive letters, backslash tricks on Windows)."""
    if not parts or any(p in ("", ".", "..") or "\\" in p or ":" in p for p in parts):
        return None
    return core_paths.under(base, *parts)


def import_bundle(zip_path, name=None, force=False, log=print):
    with zipfile.ZipFile(zip_path) as z:
        try:
            manifest = json.loads(z.read("textllm.json"))
        except KeyError:
            raise SystemExit(f"{zip_path} is not a model bundle made by transfer.py export")
        if manifest.get("format", 0) > FORMAT:
            raise SystemExit("this bundle was made by a newer version; update the code (git pull) first")
        name = name or manifest["name"]
        name = "".join(c if c.isalnum() or c in "-_." else "-" for c in name).strip(".") or "imported"
        dest = os.path.join(RUNS, name)
        if os.path.exists(dest) and not force:
            raise SystemExit(f"runs/{name} already exists: pick another --name, or --force to replace it")
        os.makedirs(RUNS, exist_ok=True)
        stage = tempfile.mkdtemp(prefix=f".{name}.", dir=RUNS)
        try:
            has_data = False
            for info in z.infolist():
                parts = info.filename.split("/")
                if info.is_dir():
                    continue
                if parts[0] == "run":
                    target = _safe_target(stage, parts[1:])
                elif parts[:2] == ["data", "bin"] and len(parts) == 3:
                    # into the run's own folder, never over this PC's data/bin (which may come
                    # from a different tokenizer)
                    target = _safe_target(os.path.join(stage, "data", "bin"), parts[2:])
                    has_data = True
                else:
                    continue
                if target is None:
                    raise SystemExit(f"refusing unsafe path in bundle: {info.filename!r}")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(target + ".part", target)
            meta_p = os.path.join(stage, "meta.json")
            meta = json.load(open(meta_p))
            if meta["type"] == "scratch":
                meta["tok"] = "tokenizer.json"  # the run's own copy; the old PC's path means nothing here
                json.dump(meta, open(meta_p, "w"))
            cmd = _saved_cmd(stage)
            if cmd is not None:
                try:
                    commands.check(cmd)
                except commands.BadCommand as e:
                    log(f"Note: the bundle's saved training command was dropped ({e}); Resume will use defaults.")
                    cmd = None
                    os.remove(os.path.join(stage, "train_cmd.json"))
            if cmd is not None:
                if "--out" in cmd:
                    cmd[cmd.index("--out") + 1] = f"runs/{name}"
                if has_data:
                    rel = f"runs/{name}/data/bin"
                    if "--data" in cmd:
                        cmd[cmd.index("--data") + 1] = rel
                    else:
                        cmd[1:1] = ["--data", rel]
                json.dump(cmd, open(os.path.join(stage, "train_cmd.json"), "w"))
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.replace(stage, dest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    log(f"imported as runs/{name}")
    if manifest["type"] == "scratch" and not os.path.exists(os.path.join(_data_dir(dest), "train.bin")):
        log("Note: to continue training, this PC needs the training data. Export again with the "
            "training data included, or copy data/bin/ from the old PC (don't retrain the tokenizer).")
    elif manifest["type"] == "scratch" and _data_dir(dest) == os.path.join(ROOT, "data", "bin"):
        mine = os.path.join(ROOT, "tokenizer", "bpe.json")
        if os.path.exists(mine) and open(mine, "rb").read() != open(os.path.join(dest, "tokenizer.json"), "rb").read():
            log("Warning: this PC's data/bin was made with a different tokenizer than this model. "
                "Export again with the training data included before resuming.")
    log("continue training: press Resume on the model's page in the Studio, or run\n  "
        + " ".join(resume_cmd(dest)))
    return name


def resume_cmd(run_dir):
    """The command that continues this run (its original settings plus --resume)."""
    rel = os.path.relpath(run_dir, ROOT).replace(os.sep, "/")
    p = os.path.join(run_dir, "train_cmd.json")
    if os.path.exists(p):
        cmd = commands.check(json.load(open(p)))
        cmd = [rel if i > 0 and cmd[i - 1] == "--out" else c for i, c in enumerate(cmd)]
    else:
        meta = _meta(run_dir)
        if meta["type"] == "scratch":
            cmd = ["train.py", "--out", rel, "--preset", meta.get("preset", "cpu")]
        else:
            cmd = ["finetune/finetune_lora.py", "--out", rel, "--base", meta["base"],
                   "--lora-r", str(meta.get("lora_r", 8))]
            if meta.get("instructions"):
                cmd += ["--instructions", "data/instructions/train.jsonl"]
    if "--resume" not in cmd:
        cmd.append("--resume")
    return ["python"] + cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="pack a run into one .zip")
    e.add_argument("--run", required=True, help="runs/<name>")
    e.add_argument("--out", help="zip file to write (default: <name>.textllm.zip here)")
    e.add_argument("--with-data", action="store_true",
                   help="from scratch: include train.bin/val.bin (fine-tunes always carry their examples)")
    i = sub.add_parser("import", help="unpack a .zip made by export into runs/")
    i.add_argument("zip")
    i.add_argument("--name", help="save under this name instead of the original")
    i.add_argument("--force", action="store_true", help="replace an existing run with the same name")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.run, a.out, a.with_data)
    else:
        import_bundle(a.zip, a.name, a.force)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
