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
    eval history and the fast GGUF copy if there is one;
  * from scratch: the tokenizer (as <run>/tokenizer.json). Its numbers must match the weights, so
    never train a new tokenizer for a moved model;
  * --with-data: data/bin/train.bin + val.bin (from scratch) or data/instructions/ (fine-tune).
    Without it, the new PC needs its own data/bin (built with the same tokenizer) to continue a
    from-scratch run; a fine-tune rebuilds its examples from whatever data is there.
The fine-tune's base model (e.g. Qwen/Qwen3-0.6B) is not included: the new PC downloads it once.
"""
import argparse, json, os, shutil, sys, tempfile, time, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
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
    if with_data:
        if meta["type"] == "scratch":
            for f in ("train.bin", "val.bin"):
                p = os.path.join(ROOT, "data", "bin", f)
                if os.path.exists(p):
                    files.append((p, "data/bin/" + f))
        else:
            d = os.path.join(ROOT, "data", "instructions")
            for dp, _, fns in os.walk(d):
                for f in fns:
                    p = os.path.join(dp, f)
                    files.append((p, "data/instructions/" + os.path.relpath(p, d).replace(os.sep, "/")))
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
            for info in z.infolist():
                parts = info.filename.split("/")
                if info.is_dir() or ".." in parts or info.filename.startswith("/") or ":" in parts[0]:
                    continue
                if parts[0] == "run":
                    target = os.path.join(stage, *parts[1:])
                elif parts[0] == "data" and len(parts) >= 3 and parts[1] in ("bin", "instructions"):
                    target = os.path.join(ROOT, *parts)
                    if os.path.exists(target) and not force:
                        log(f"kept your existing {'/'.join(parts)} (use --force to replace it)")
                        continue
                else:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(target + ".part", target)
            meta_p = os.path.join(stage, "meta.json")
            meta = json.load(open(meta_p))
            if meta["type"] == "scratch":
                meta["tok"] = "tokenizer.json"  # the run's own copy; the old PC's path means nothing here
                json.dump(meta, open(meta_p, "w"))
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.replace(stage, dest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    log(f"imported as runs/{name}")
    if manifest["type"] == "scratch" and not os.path.exists(os.path.join(ROOT, "data", "bin", "train.bin")):
        log("note: to continue training, this PC needs data/bin/train.bin + val.bin. Export with "
            "--with-data, or copy data/bin/ from the old PC (don't retrain the tokenizer).")
    log("continue training: press Resume on the model's page in the Studio, or run\n  "
        + " ".join(resume_cmd(dest)))
    return name


def resume_cmd(run_dir):
    """The command that continues this run (its original settings plus --resume)."""
    rel = os.path.relpath(run_dir, ROOT).replace(os.sep, "/")
    p = os.path.join(run_dir, "train_cmd.json")
    if os.path.exists(p):
        cmd = json.load(open(p))
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
    e.add_argument("--with-data", action="store_true", help="include the training data too")
    i = sub.add_parser("import", help="unpack a .zip made by export into runs/")
    i.add_argument("zip")
    i.add_argument("--name", help="save under this name instead of the original")
    i.add_argument("--force", action="store_true", help="replace an existing run / data with the same name")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.run, a.out, a.with_data)
    else:
        import_bundle(a.zip, a.name, a.force)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
