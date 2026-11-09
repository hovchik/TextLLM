"""Retrain an already-trained model on other data, as a NEW model. The original stays as it is.

  # from-scratch GPT: continue its weights on the news folder, 2000 steps
  python retrain.py --from runs/my-writer --out runs/my-writer-news --source data/raw/news --steps 2000
  # several corpus folders plus your own files and links (those are read, trained on, then deleted)
  python retrain.py --from runs/my-writer --out runs/my-writer-v2 --source data/raw/gutenberg,data/raw/custom \
      --files book.pdf --urls https://example.com/story
  # a fine-tuned (LoRA) model: continue its adapter on the same base model
  python retrain.py --from runs/qwen-writer --out runs/qwen-writer-v2 --source data/raw/custom --steps 300
  # stopped half way? continue the new model where it stopped
  python retrain.py --resume --out runs/my-writer-news

How it works
  * from scratch: the new model starts from the old weights and keeps the old tokenizer. The new text is
    encoded with that tokenizer (so the token ids always match), some of the old training data is mixed
    in (--mix) so it does not forget everything else, and training starts a fresh learning-rate schedule.
    --data-bin uses an already encoded folder (e.g. data/bin after re-encoding the corpus) instead; that
    is only allowed when it was encoded with the same tokenizer as the model.
  * LoRA: the new adapter starts from the old one (same base model, same rank) and is trained on
    continuation examples cut from the new text (code examples for a copilot).
  Learn.py (Training → Your texts) changes a model in place; this script never does.
"""
import argparse, json, math, os, shutil, sys, tempfile, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import graceful  # noqa: E402
from data.ingest import EOT, PDF_OPTS, ingest, utf8_stdout  # noqa: E402
from graceful import run_child  # noqa: E402

DATA_DIR = "retrain-data"   # <new run>/retrain-data: the encoded text, kept so Resume works


def say(m):
    print(m, flush=True)


def rel(p):
    """Paths under the repo are saved relative to it, so the run folder can move to another PC."""
    p = os.path.abspath(p)
    return os.path.relpath(p, ROOT).replace(os.sep, "/") if p.startswith(ROOT + os.sep) else p


def load_meta(run):
    p = os.path.join(run, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"{run} has no meta.json, so it is not a trained model")
    return json.load(open(p))


def is_model(run):
    return any(os.path.exists(os.path.join(run, f)) for f in ("ckpt.pt", "adapter", "meta.json"))


def same_file(a, b):
    try:
        return open(a, "rb").read() == open(b, "rb").read()
    except OSError:
        return False


def source_dirs(spec):
    """'a,b' or ['a', 'b,c'] → existing folders that hold .txt shards."""
    import glob

    out = []
    for item in spec or []:
        for d in [x.strip() for x in item.split(",") if x.strip()]:
            d = d if os.path.isabs(d) else os.path.join(ROOT, d)
            if not os.path.isdir(d):
                raise SystemExit(f"no folder {d}")
            if not glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
                raise SystemExit(f"{d} has no .txt text in it")
            out.append(d)
    return out


# ---------------------------------------------------------------- from scratch
def scratch_old_data(src, meta):
    """The folder the original model was trained on and the shared tokenizer that encodes it."""
    data = "data/bin-code" if meta.get("task") == "code" else "data/bin"
    try:
        cmd = json.load(open(os.path.join(src, "train_cmd.json")))
        if "--data" in cmd:
            data = cmd[cmd.index("--data") + 1]
    except (OSError, ValueError, IndexError):
        pass
    data = data if os.path.isabs(data) else os.path.join(ROOT, data)
    return data, shared_tokenizer(data, meta)


def shared_tokenizer(bin_dir, meta):
    """The tokenizer a shared data folder was encoded with (data/bin → tokenizer/bpe.json)."""
    code = meta.get("task") == "code" or os.path.basename(os.path.normpath(bin_dir)) == "bin-code"
    return os.path.join(ROOT, "tokenizer", "bpe-code.json" if code else "bpe.json")


def check_tokens(arr, vocab, what):
    if len(arr) and int(arr.max()) >= vocab:
        raise SystemExit(f"{what} has token id {int(arr.max())}, but the model only knows {vocab} tokens: "
                         "it was encoded with a different tokenizer. Use --source with the text folders instead.")


def build_scratch_data(a, src, meta, cfg, tok_p, sources, private_dir, bindir):
    """Encode the new text with the model's own tokenizer, mix in some old data, write train/val.bin."""
    import numpy as np

    from tokenizer.train_tokenizer import encode

    block, vocab = cfg["block_size"], cfg["vocab_size"]
    fim = 0.5 if meta.get("task") == "code" else 0.0
    trains, vals = [], []
    tmp = tempfile.mkdtemp(prefix="retrain_enc_")
    try:
        if a.data_bin:
            d = a.data_bin if os.path.isabs(a.data_bin) else os.path.join(ROOT, a.data_bin)
            enc_tok = a.data_tok or shared_tokenizer(d, meta)
            if not same_file(enc_tok, tok_p):
                raise SystemExit(
                    f"{rel(d)} was encoded with {rel(enc_tok)}, which is not this model's tokenizer "
                    f"(the tokenizer was rebuilt after {os.path.basename(src)} was trained, or the model came from "
                    "another PC). Its token ids would mean different words to the model. Retrain from the text "
                    "folders instead (--source data/raw/...); they are encoded with the model's own tokenizer.")
            for s, lst in (("train", trains), ("val", vals)):
                p = os.path.join(d, f"{s}.bin")
                if os.path.exists(p):
                    arr = np.memmap(p, dtype=np.uint16, mode="r")
                    check_tokens(arr[: 1 << 20], vocab, rel(p))
                    lst.append(arr)
            say(f"using {rel(d)}: {sum(len(t) for t in trains):,} tokens (same tokenizer as the model)")
        for i, d in enumerate(sources + ([private_dir] if private_dir else [])):
            out = os.path.join(tmp, str(i))
            say(f"encoding {rel(d)} with the model's own tokenizer")
            encode(d, tok_p, out, a.val_frac, fim_rate=fim, seed=int(time.time()))
            t = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)
            v = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16)
            trains.append(t)
            vals.append(v)
        new = sum(len(t) for t in trains)
        if new < block + 2:
            raise SystemExit(f"the new data is too short to train on: {new:,} tokens, need at least {block + 2:,}")
        parts = list(trains)
        old_dir, old_tok = scratch_old_data(src, meta)
        old_p = os.path.join(old_dir, "train.bin")
        if a.mix > 0 and os.path.exists(old_p):
            if not same_file(old_tok, tok_p):
                say(f"not mixing in {rel(old_dir)}: it is encoded with a different tokenizer than this model's "
                    "(the model may forget more of what it knew; add the old text folders with --source to keep it)")
            elif not (a.data_bin and os.path.samefile(old_dir, a.data_bin if os.path.isabs(a.data_bin)
                                                          else os.path.join(ROOT, a.data_bin))):
                old = np.memmap(old_p, dtype=np.uint16, mode="r")
                n = min(len(old) - 1, int(new * a.mix))
                if n > block:
                    s = int(np.random.default_rng().integers(0, len(old) - n))
                    parts.append(np.array(old[s:s + n]))
                    say(f"mixing in {n:,} tokens of what {os.path.basename(src)} learned before ({rel(old_dir)})")
        os.makedirs(bindir, exist_ok=True)
        with open(os.path.join(bindir, "train.bin"), "wb") as f:
            for p in parts:
                np.asarray(p).tofile(f)
        val = np.concatenate([np.asarray(v) for v in vals]) if vals else np.zeros(0, np.uint16)
        if len(val) < block + 2:  # a small text: "val" = how well it now knows the new text
            val = np.concatenate([np.asarray(t[:200_000]) for t in trains])[:200_000]
        val.tofile(os.path.join(bindir, "val.bin"))
        return new, sum(len(p) for p in parts)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scratch(a, src, meta, sources, private_dir, work):
    import torch

    from chat_engine import tokenizer_path
    from core import safeload

    ck = safeload.load(os.path.join(src, "ckpt.pt"))
    cfg = ck["config"]
    tok_p = tokenizer_path(src, meta)
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(tok_p, os.path.join(a.out, "tokenizer.json"))
    tok_p = os.path.join(a.out, "tokenizer.json")
    # Only private texts (your files and links): the encoded copy goes in the temp folder and is deleted.
    keep = private_dir is None
    bindir = os.path.join(a.out, DATA_DIR) if keep else os.path.join(work, "bin")
    new, total = build_scratch_data(a, src, meta, cfg, tok_p, sources, private_dir, bindir)
    tokens_per_step = a.batch * a.grad_accum * cfg["block_size"]
    steps = a.steps or max(10, math.ceil(total * a.passes / tokens_per_step))
    lr = a.lr or 3e-4
    # The weights of the original, as step 0 of the new run: no optimizer state, no sampling position,
    # so train.py --resume starts a fresh schedule (warm-up, then cosine to --min-lr) on the new data.
    torch.save({"model": ck["model"], "config": cfg, "step": 0, "best_val": float("inf"), "scaler": None,
                "args": {"retrained_from": os.path.basename(src)}}, os.path.join(a.out, "ckpt.pt"))
    del ck
    write_meta(a, src, meta, sources, private_dir, steps=steps, new_tokens=new)
    say(f"{new:,} new tokens ({total:,} with the old data mixed in) → {steps} steps at lr {lr:g}")
    cmd = ["train.py", "--data", rel(bindir), "--out", rel(a.out), "--preset", meta.get("preset", "cpu"),
           "--vocab", str(cfg["vocab_size"]), "--max-steps", str(steps), "--batch", str(a.batch),
           "--grad-accum", str(a.grad_accum), "--lr", str(lr), "--min-lr", str(a.min_lr or lr / 10),
           "--warmup", str(a.warmup or max(1, min(200, steps // 20))),
           "--eval-every", str(max(1, steps // 20)), "--eval-iters", "10", "--device", a.device]
    return cmd, keep


# ---------------------------------------------------------------- LoRA
def lora(a, src, meta, sources, private_dir):
    ad = os.path.join(src, "adapter")
    if not os.path.isdir(ad):
        raise SystemExit(f"{os.path.basename(src)} has no LoRA adapter yet (train it first)")
    dirs = sources + ([private_dir] if private_dir else [])
    if a.data_bin:
        raise SystemExit("--data-bin is for from-scratch models; give LoRA models text folders with --source")
    os.makedirs(a.out, exist_ok=True)
    steps = a.steps or 200
    n = steps * a.batch * a.grad_accum  # about one pass over the examples
    raw = ",".join(rel(d) for d in dirs)
    cmd = ["finetune/finetune_lora.py", "--base", meta["base"], "--out", rel(a.out), "--init-adapter", rel(ad),
           "--seq", str(a.seq), "--batch", str(a.batch), "--grad-accum", str(a.grad_accum),
           "--max-steps", str(steps), "--lr", str(a.lr or 1e-4), "--eval-every", str(max(1, steps // 10)),
           "--save-every", str(max(1, steps // 5)), "--min-doc-chars", str(a.min_doc_chars),
           "--pairs-per-doc", str(a.pairs_per_doc), "--device", a.device]
    if meta.get("task") == "code":  # --examples 0: no book examples, only code from the new folders
        cmd += ["--examples", "0", "--code", raw, "--code-examples", str(n), "--fim-rate", "0.5"]
    else:
        cmd += ["--raw", raw, "--examples", str(n)]
    if a.gguf:
        cmd.append("--gguf")
    keep = private_dir is None
    if not keep:
        cmd.append("--private-texts")
    m = write_meta(a, src, meta, sources, private_dir, steps=steps)
    # finetune_lora.py rewrites meta.json on every save; it carries the style profile over from the
    # adapter it starts from, and this keeps the "retrained from" note (short: it is a command argument)
    note = dict(m["retrained"])
    if len(note["sources"]) > 8:  # keep the argument short (and valid JSON): list a few, count the rest
        note["sources"] = note["sources"][:8] + [f"... and {len(note['sources']) - 8} more"]
    while len(json.dumps(note, ensure_ascii=False)) > 3000 and note["sources"]:
        note["sources"] = note["sources"][:-1]
    cmd += ["--meta-extra", json.dumps({"retrained": note}, ensure_ascii=False)]
    say(f"continuing the adapter of {os.path.basename(src)} ({meta['base']}) on about {n} examples → {steps} steps")
    return cmd, keep


def write_meta(a, src, meta, sources, private_dir, **info):
    """The new model's meta.json: what it is (copied from the original) and where it came from."""
    m = {k: v for k, v in meta.items() if k not in ("gguf", "gguf_quant", "retrained")}
    m["retrained"] = {"from": os.path.basename(src), "time": int(time.time()),
                      "sources": [rel(d) for d in sources] + (["data you gave (deleted after training)"]
                                                              if private_dir else []),
                      **({"data_bin": rel(a.data_bin)} if a.data_bin else {}),
                      "resumable": private_dir is None, **info}
    if m.get("type") == "scratch":
        m["tok"] = "tokenizer.json"
        if not m.get("tokenizer"):    # models from before tokenizer versions: read it from the file
            from tokenizer.quality import describe

            m["tokenizer"] = describe(os.path.join(src, "tokenizer.json"))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(m, f)
    return m


# ---------------------------------------------------------------- main
def resume(out):
    from core import commands

    p = os.path.join(out, "train_cmd.json")
    if not os.path.exists(p):
        raise SystemExit(f"{out} cannot be resumed: it has no saved training command (a retrain on your own "
                         "files and links keeps no copy of them, so start that one again)")
    cmd = commands.check(json.load(open(p)))
    cmd = [rel(out) if i > 0 and cmd[i - 1] == "--out" else c for i, c in enumerate(cmd)]
    return cmd + (["--resume"] if "--resume" not in cmd else [])


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", help="runs/<name> of the trained model to start from (left unchanged)")
    ap.add_argument("--out", required=True, help="runs/<new name> for the retrained model")
    ap.add_argument("--resume", action="store_true", help="continue a stopped retrain in --out")
    ap.add_argument("--force", action="store_true", help="replace a model that already exists in --out")
    g = ap.add_argument_group("new data (any mix)")
    g.add_argument("--source", nargs="*", default=[], help="text folders, e.g. data/raw/news data/raw/custom")
    g.add_argument("--data-bin", help="from scratch only: an encoded folder (train.bin/val.bin), e.g. data/bin")
    g.add_argument("--data-tok", help="the tokenizer --data-bin was encoded with (default: tokenizer/bpe.json)")
    g.add_argument("--files", nargs="*", default=[], help="your PDF/Word/TXT/MD/HTML files or folders")
    g.add_argument("--urls", nargs="*", default=[])
    g.add_argument("--url-list", help="a text file with one link per line")
    g.add_argument("--delete-originals", action="store_true", help="delete --files after reading them")
    g.add_argument("--ocr", choices=["auto", "off"], default="auto")
    g.add_argument("--ocr-lang", default="auto")
    t = ap.add_argument_group("training")
    t.add_argument("--steps", type=int, default=0,
                   help="training steps (0 = from scratch: read the new data --passes times; LoRA: 200)")
    t.add_argument("--passes", type=float, default=1.0, help="from scratch with --steps 0: reads of the data")
    t.add_argument("--lr", type=float, default=0, help="0 = 3e-4 from scratch, 1e-4 LoRA")
    t.add_argument("--min-lr", type=float, default=0, help="from scratch: end of the schedule (0 = lr/10)")
    t.add_argument("--warmup", type=int, default=0, help="from scratch: warm-up steps (0 = 5%% of the steps)")
    t.add_argument("--mix", type=float, default=0.5,
                   help="from scratch: old training tokens mixed in per new token (0 = only the new data)")
    t.add_argument("--val-frac", type=float, default=0.01, help="from scratch: share of new text held out")
    t.add_argument("--batch", type=int, default=0, help="0 = 16 from scratch, 2 LoRA")
    t.add_argument("--grad-accum", type=int, default=0, help="0 = 4 from scratch, 8 LoRA")
    t.add_argument("--seq", type=int, default=512, help="LoRA: tokens per example")
    t.add_argument("--min-doc-chars", type=int, default=500, help="LoRA: skip shorter documents")
    t.add_argument("--pairs-per-doc", type=int, default=8, help="LoRA: max examples from one document")
    t.add_argument("--gguf", action="store_true", help="LoRA: make the fast CPU copy afterwards")
    t.add_argument("--device", default="auto")
    return ap.parse_args(argv)


def prepare(a, work):
    """Check the request, gather the data and return (command, keep_for_resume). Runs nothing."""
    a.out = os.path.abspath(a.out)
    if not a.src:
        raise SystemExit("--from is required (the model to retrain)")
    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        raise SystemExit(f"no model at {src}")
    if os.path.normcase(src) == os.path.normcase(a.out):
        raise SystemExit("--out must be a new name: retraining never changes the original model")
    if is_model(a.out):
        if not a.force:
            raise SystemExit(f"{a.out} already holds a model. Pick another name, or --force to replace it "
                             "(--resume continues a stopped retrain)")
        for f in ("ckpt.pt", "adapter", "train_state.pt", "examples.pt", "metrics.jsonl", "train_cmd.json",
                  DATA_DIR, "gguf"):
            p = os.path.join(a.out, f)
            shutil.rmtree(p) if os.path.isdir(p) else (os.remove(p) if os.path.exists(p) else None)
    meta = load_meta(src)
    kind = meta.get("type")
    a.batch = a.batch or (16 if kind == "scratch" else 2)
    a.grad_accum = a.grad_accum or (4 if kind == "scratch" else 8)
    sources = source_dirs(a.source)
    urls = list(a.urls) + ([l.strip() for l in open(a.url_list) if l.strip()] if a.url_list else [])
    private_dir = None
    if a.files or urls:
        say("reading your files and links (they are deleted after training)")
        private_dir = os.path.join(work, "text")
        shard, n, _ = ingest(a.files, urls, private_dir, a.delete_originals, log=say)
        if not n:
            raise SystemExit("nothing could be read from your files and links (the ✗ lines above say why)")
    if not (sources or private_dir or a.data_bin):
        raise SystemExit("give the new data: --source text folders, --files/--urls, or --data-bin")
    if kind == "scratch":
        return scratch(a, src, meta, sources, private_dir, work)
    if kind == "lora":
        return lora(a, src, meta, sources, private_dir)
    raise SystemExit(f"{os.path.basename(src)} is a {kind!r} model; only from-scratch and LoRA models can be retrained")


def main(argv=None):
    graceful.install()
    a = get_args(argv)
    utf8_stdout()
    PDF_OPTS.update(ocr=a.ocr, ocr_lang=a.ocr_lang)
    if a.resume:
        cmd = resume(os.path.abspath(a.out))
        say(f"continuing the retrain in {rel(a.out)}")
        rc = run_child([sys.executable] + cmd, ROOT)
        raise SystemExit(0 if rc == 0 else f"training failed (exit {rc})")
    work = tempfile.mkdtemp(prefix="retrain_")
    try:
        cmd, keep = prepare(a, work)
        saved = os.path.join(a.out, "train_cmd.json")
        if keep:  # Resume (here or on the Runs page) runs this again with --resume
            json.dump(cmd, open(saved, "w"))
        elif os.path.exists(saved):
            os.remove(saved)
        if cmd[0] == "train.py":  # start from the copied weights in <out>/ckpt.pt (step 0, fresh optimizer)
            cmd = cmd + ["--resume"]
        mp = os.path.join(a.out, "metrics.jsonl")
        if os.path.exists(mp):
            os.remove(mp)
        say(f"== training {os.path.basename(a.out)} (from {os.path.basename(os.path.abspath(a.src))})")
        rc = run_child([sys.executable] + cmd, ROOT)
        if rc != 0:
            raise SystemExit(f"training failed (exit {rc}); {os.path.basename(a.src)} is unchanged")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"✅ done. {os.path.basename(a.out)} is ready (chat with it in the Chat tab); "
        f"{os.path.basename(os.path.abspath(a.src))} is unchanged.")


if __name__ == "__main__":
    main()
