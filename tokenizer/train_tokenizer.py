"""Train a byte-level BPE tokenizer on the raw corpus, then encode it to uint16 .bin files.

Usage:
  python tokenizer/train_tokenizer.py --raw data/raw --vocab 32000 --out tokenizer/bpe.json
  python tokenizer/train_tokenizer.py --encode --raw data/raw --tok tokenizer/bpe.json --bin data/bin

Copilot (code) model: --code adds the fill-in-the-middle tokens, and --fim-rate rearranges that share
of the code documents into prefix/suffix/middle order while encoding (coding/fim.py):
  python tokenizer/train_tokenizer.py --code --raw data/code --vocab 16384 --out tokenizer/bpe-code.json
  python tokenizer/train_tokenizer.py --encode --raw data/code --tok tokenizer/bpe-code.json --bin data/bin-code --fim-rate 0.5
--raw takes several folders separated by commas (e.g. data/code,data/raw to also learn English).
"""
import argparse, glob, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

EOT = "<|endoftext|>"


def shards(raw):
    files = []
    for d in [x.strip() for x in raw.split(",") if x.strip()]:
        files += sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    if not files:
        raise SystemExit(f"no .txt shards under {raw}")
    return files


def train(raw, vocab, out, sample_files, code=False):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from coding import fim

    files = shards(raw)
    random.seed(0)
    if sample_files and len(files) > sample_files:
        files = random.sample(files, sample_files)  # a few GB is plenty for BPE
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=list(fim.SPECIAL) if code else [EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train(files, trainer)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tok.save(out)
    print(f"saved {out} (vocab {tok.get_vocab_size()})")


def encode(raw, tok_path, bin_dir, val_frac, fim_rate=0.0, seed=0):
    from tokenizers import Tokenizer

    from coding import fim
    from coding.langs import is_code_doc

    tok = Tokenizer.from_file(tok_path)
    assert tok.get_vocab_size() < 65536, "uint16 storage needs vocab < 65536"
    eot = tok.token_to_id(EOT)
    ids = fim.token_ids(tok)
    if fim_rate > 0 and not ids:
        print("this tokenizer has no fill-in-the-middle tokens (train it with --code); encoding without FIM")
        fim_rate = 0.0
    os.makedirs(bin_dir, exist_ok=True)
    out = {s: open(os.path.join(bin_dir, f"{s}.bin"), "wb") for s in ("train", "val")}
    counts = {"train": 0, "val": 0}
    rng, frng, n_fim = random.Random(0), random.Random(seed + 1), 0
    for path in tqdm(shards(raw), desc="encoding"):
        docs = [d.strip() for d in open(path, encoding="utf-8").read().split(EOT) if d.strip()]
        # each document becomes one piece, or three (prefix, suffix, middle) for a FIM example
        plan, pieces = [], []
        for d in docs:
            if fim_rate > 0 and is_code_doc(d) and frng.random() < fim_rate:
                pre, mid, suf = fim.split(d, frng)
                plan.append((len(pieces), 3))
                pieces += [pre, suf, mid]
                n_fim += 1
            else:
                plan.append((len(pieces), 1))
                pieces.append(d)
        enc = [e.ids for e in tok.encode_batch(pieces)] if pieces else []
        for start, n in plan:
            if n == 1:
                seq = enc[start]
            else:
                pre, suf, mid = enc[start:start + 3]
                seq = [ids["prefix"], *pre, ids["suffix"], *suf, ids["middle"], *mid]
            split = "val" if rng.random() < val_frac else "train"
            arr = np.array(seq + [eot], dtype=np.uint16)
            arr.tofile(out[split])
            counts[split] += len(arr)
    for f in out.values():
        f.close()
    print({k: f"{v/1e6:.1f}M tokens" for k, v in counts.items()})
    if fim_rate > 0:
        print(f"{n_fim} code files rearranged for fill-in-the-middle")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", "--tok", dest="tok", default="tokenizer/bpe.json")
    ap.add_argument("--sample-files", type=int, default=40, help="shards used to fit BPE")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--bin", default="data/bin")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--code", action="store_true", help="add the fill-in-the-middle tokens (copilot models)")
    ap.add_argument("--fim-rate", type=float, default=0.0,
                    help="--encode: share of code documents rearranged for fill-in-the-middle (0.5 is typical)")
    a = ap.parse_args()
    if a.encode:
        encode(a.raw, a.tok, a.bin, a.val_frac, max(0.0, min(1.0, a.fim_rate)))
    else:
        train(a.raw, a.vocab, a.tok, a.sample_files, a.code)
