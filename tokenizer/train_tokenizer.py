"""Train a byte-level BPE tokenizer on the raw corpus, then encode it to uint16 .bin files.

Usage:
  python tokenizer/train_tokenizer.py --raw data/raw --vocab 32000 --out tokenizer/bpe.json
  python tokenizer/train_tokenizer.py --encode --raw data/raw --tok tokenizer/bpe.json --bin data/bin
"""
import argparse, glob, os, random

import numpy as np
from tqdm import tqdm

EOT = "<|endoftext|>"


def shards(raw):
    files = sorted(glob.glob(os.path.join(raw, "**", "*.txt"), recursive=True))
    if not files:
        raise SystemExit(f"no .txt shards under {raw}")
    return files


def train(raw, vocab, out, sample_files):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    files = shards(raw)
    random.seed(0)
    if sample_files and len(files) > sample_files:
        files = random.sample(files, sample_files)  # a few GB is plenty for BPE
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train(files, trainer)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tok.save(out)
    print(f"saved {out} (vocab {tok.get_vocab_size()})")


def encode(raw, tok_path, bin_dir, val_frac):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    assert tok.get_vocab_size() < 65536, "uint16 storage needs vocab < 65536"
    eot = tok.token_to_id(EOT)
    os.makedirs(bin_dir, exist_ok=True)
    out = {s: open(os.path.join(bin_dir, f"{s}.bin"), "wb") for s in ("train", "val")}
    counts = {"train": 0, "val": 0}
    rng = random.Random(0)
    for path in tqdm(shards(raw), desc="encoding"):
        docs = [d.strip() for d in open(path, encoding="utf-8").read().split(EOT) if d.strip()]
        for enc, _ in zip(tok.encode_batch(docs), docs):
            split = "val" if rng.random() < val_frac else "train"
            arr = np.array(enc.ids + [eot], dtype=np.uint16)
            arr.tofile(out[split])
            counts[split] += len(arr)
    for f in out.values():
        f.close()
    print({k: f"{v/1e6:.1f}M tokens" for k, v in counts.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", "--tok", dest="tok", default="tokenizer/bpe.json")
    ap.add_argument("--sample-files", type=int, default=40, help="shards used to fit BPE")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--bin", default="data/bin")
    ap.add_argument("--val-frac", type=float, default=0.005)
    a = ap.parse_args()
    if a.encode:
        encode(a.raw, a.tok, a.bin, a.val_frac)
    else:
        train(a.raw, a.vocab, a.tok, a.sample_files)
