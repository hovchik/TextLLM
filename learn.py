"""Teach an already-trained model new texts: extract → train → delete.

  python learn.py --run runs/scratch-cpu --files book.pdf essay.docx --urls https://example.com/story
  python learn.py --run runs/finetune-qwen --url-list links.txt --passes 3 --delete-originals

The downloaded/extracted text lives in a temporary folder and is deleted when training ends
(also if it fails). With --delete-originals the files you passed in are removed too.
The model keeps what it learned; the texts themselves are not stored anywhere.
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import graceful  # noqa: E402
from core import safeload  # noqa: E402
from data.ingest import EOT, PDF_OPTS, ingest, utf8_stdout  # noqa: E402
from graceful import run_child  # noqa: E402


def say(m):
    print(m, flush=True)



def last_step(run):
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    steps = [json.loads(l).get("step", 0) for l in open(p) if l.strip()]
    return max(steps) if steps else 0


def learn_scratch(a, meta, text_dir, tmp):
    import torch
    from tokenizers import Tokenizer

    from chat_engine import tokenizer_path

    ck = safeload.load(os.path.join(a.run, "ckpt.pt"))
    cfg, step = ck["config"], ck["step"]
    tok = Tokenizer.from_file(tokenizer_path(a.run, meta))
    eot = tok.token_to_id(EOT)
    docs = [d.strip() for f in sorted(os.listdir(text_dir))
            for d in open(os.path.join(text_dir, f), encoding="utf-8").read().split(EOT) if d.strip()]
    # encode_batch tokenizes on all cores; each document ends with the end-of-text token
    parts = [np.array(e.ids + [eot], dtype=np.uint16) for e in tok.encode_batch(docs)]
    new = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
    block = cfg["block_size"]
    if len(new) < block + 2:
        raise SystemExit(f"text too short: {len(new)} tokens, need at least {block + 2}")
    # Mix in some of the original corpus so the model doesn't forget everything else.
    parts = [new] * max(1, int(round(a.passes)))
    old_path = os.path.join(ROOT, "data/bin/train.bin")
    if a.mix > 0 and os.path.exists(old_path):
        old = np.memmap(old_path, dtype=np.uint16, mode="r")
        n = min(len(old) - 1, int(len(new) * a.passes * a.mix))
        if n > block:
            s = np.random.default_rng().integers(0, len(old) - n)
            parts.append(np.array(old[s:s + n]))
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir)
    np.concatenate(parts).tofile(os.path.join(bindir, "train.bin"))
    new.tofile(os.path.join(bindir, "val.bin"))  # "val" here = how well it now knows the new text
    tokens_per_step = a.batch * a.grad_accum * block
    steps = max(5, math.ceil(sum(len(p) for p in parts) / tokens_per_step))
    say(f"{len(new):,} new tokens → {steps} training steps (from step {step})")
    cmd = [sys.executable, "train.py", "--data", bindir, "--out", a.run, "--resume",
           "--preset", meta.get("preset", "cpu"), "--vocab", str(cfg["vocab_size"]),
           "--max-steps", str(step + steps), "--batch", str(a.batch), "--grad-accum", str(a.grad_accum),
           "--lr", str(a.lr or 3e-4), "--min-lr", str(a.lr or 3e-4), "--warmup", "1",
           "--eval-every", str(max(1, steps // 5)), "--eval-iters", "5"]
    return cmd


def learn_lora(a, meta, text_dir):
    seq, bs, acc = 512, 2, 8
    # finetune_lora makes ~1 example per (0.9*seq) tokens of text; estimate from characters.
    chars = sum(os.path.getsize(os.path.join(text_dir, f)) for f in os.listdir(text_dir))
    examples = max(16, chars // (4 * int(seq * 0.9)))
    steps = max(5, math.ceil(examples * a.passes / (bs * acc)))
    say(f"about {examples} examples → {steps} fine-tuning steps")
    return [sys.executable, "finetune/finetune_lora.py", "--base", meta["base"], "--raw", text_dir,
            "--out", a.run, "--init-adapter", os.path.join(a.run, "adapter"), "--examples", str(examples),
            "--pairs-per-doc", "1000", "--min-doc-chars", "500", "--seq", str(seq),
            "--batch", str(bs), "--grad-accum", str(acc), "--max-steps", str(steps),
            "--lr", str(a.lr or 1e-4),
            "--eval-every", str(max(1, steps // 5)), "--save-every", str(steps), "--step-offset", str(last_step(a.run)),
            "--private-texts"]


def main():
    graceful.install()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> of an already-trained model")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--urls", nargs="*", default=[])
    ap.add_argument("--url-list")
    ap.add_argument("--text-file", nargs="*", default=[],
                    help="ready-made shards (documents separated by <|endoftext|>, e.g. from data/crawl.py)")
    ap.add_argument("--passes", type=float, default=3, help="how many times to read the new texts")
    ap.add_argument("--mix", type=float, default=1.0, help="old-corpus tokens per new token (scratch only)")
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 (scratch) / 1e-4 (LoRA)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--delete-originals", action="store_true")
    ap.add_argument("--ocr", choices=["auto", "off"], default="auto", help="OCR scanned PDF pages (Tesseract)")
    ap.add_argument("--ocr-lang", default="auto", help="Tesseract languages, e.g. eng+hye")
    a = ap.parse_args()
    utf8_stdout()
    PDF_OPTS.update(ocr=a.ocr, ocr_lang=a.ocr_lang)
    a.run = os.path.abspath(a.run)
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    urls = list(a.urls) + ([l.strip() for l in open(a.url_list) if l.strip()] if a.url_list else [])

    tmp = tempfile.mkdtemp(prefix="learn_")
    try:
        say("== 1/3 downloading and extracting text")
        text_dir = os.path.join(tmp, "text")
        shard, n = None, 0
        if a.files or urls:
            shard, n, _ = ingest(a.files, urls, text_dir, a.delete_originals, log=say)
        for i, tf in enumerate(a.text_file):  # already clean: copy as-is, keeping document breaks
            os.makedirs(text_dir, exist_ok=True)
            dst = os.path.join(text_dir, f"given_{i}.txt")
            shutil.copyfile(tf, dst)
            n += sum(1 for d in open(dst, encoding="utf-8").read().split(EOT) if d.strip())
            shard = shard or dst
        if not shard:
            raise SystemExit("nothing could be extracted, so there is nothing to learn (the ✗ lines above say why)")
        say(f"== 2/3 training {os.path.basename(a.run)} on {n} new document(s)")
        cmd = learn_scratch(a, meta, text_dir, tmp) if meta["type"] == "scratch" else learn_lora(a, meta, text_dir)
        rc = run_child(cmd, ROOT)
        if rc != 0:
            raise SystemExit(f"training failed (exit {rc})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        say("== 3/3 removed downloaded and extracted text")
    say("✅ done. The model has learned the new texts; chat with it in the Chat tab.")


if __name__ == "__main__":
    main()
