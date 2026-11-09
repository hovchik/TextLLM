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
    from core import progress

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
        show_progress=not progress.active(),
    )
    # The files are read here instead of inside the tokenizer library, so the Studio can show how much
    # is read; the library still counts words on every core. Lines are what tok.train(files) uses too.
    sizes = [os.path.getsize(f) for f in files]
    board = progress.board()
    board.title, board.kind = "Building the tokenizer", "work"
    read = board.task("read", "Reading text", sum(sizes), "bytes")
    read.verb = "Reading"
    learn = board.task("merges", f"Learning {vocab:,} word pieces", 0, "steps")
    learn.verb = "Learning"
    read.start(note=f"{len(files)} files")
    print(f"building a {vocab:,}-piece tokenizer from {len(files)} files, {progress.fmt_bytes(sum(sizes))}")

    def lines():
        done = 0
        for i, (path, size) in enumerate(zip(files, sizes)):
            at = 0
            with open(path, encoding="utf-8") as f:
                while True:
                    batch = f.readlines(1 << 20)       # about 1 MB of lines
                    if not batch:
                        break
                    yield batch
                    at += sum(map(len, batch))         # characters; bytes for English, close enough
                    read.set(done=done + min(size, at), note=f"file {i + 1} of {len(files)}")
            done += size
            read.set(done=done)
        read.finish(note=f"{len(files)} files")
        learn.start(note="the last part; it shows no percent, usually a few minutes")

    try:
        tok.train_from_iterator(lines(), trainer)
    except BaseException:
        for t in (read, learn):
            if t.state in ("running", "waiting"):
                t.finish("failed")
        raise
    learn.finish(note=f"vocabulary {tok.get_vocab_size():,}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tok.save(out)
    print(f"saved {out} (vocab {tok.get_vocab_size()})")


BATCH_CHARS = 4_000_000   # text per tokenizer call: ~1M tokens, so memory stays small whatever the shard size


def _batches(text, fim_rate, frng, is_code_doc, fim):
    """Split one shard into batches of documents. Yields (pieces, plan, chars of the shard consumed).

    plan has one (start, n) per document: n is 1 for a plain document, or 3 (prefix, suffix, middle)
    for a fill-in-the-middle example."""
    pieces, plan, size, pos = [], [], 0, 0
    for raw_doc in text.split(EOT):
        pos += len(raw_doc) + len(EOT)
        d = raw_doc.strip()
        if not d:
            continue
        if fim_rate > 0 and is_code_doc(d) and frng.random() < fim_rate:
            pre, mid, suf = fim.split(d, frng)
            plan.append((len(pieces), 3))
            pieces += [pre, suf, mid]
        else:
            plan.append((len(pieces), 1))
            pieces.append(d)
        size += len(d)
        if size >= BATCH_CHARS:
            yield pieces, plan, min(pos, len(text))
            pieces, plan, size = [], [], 0
    if pieces:
        yield pieces, plan, len(text)


def encode(raw, tok_path, bin_dir, val_frac, fim_rate=0.0, seed=0):
    """Encode every shard to train.bin / val.bin (uint16 token ids, one <|endoftext|> after each document).

    Documents go to the tokenizer a few MB at a time (it uses every CPU core for each batch), while the
    previous batch is written, so memory stays low even for 100 MB shards. Progress (bytes of text,
    tokens, speed, time left) goes to the console and, in the Studio, to the job card."""
    from concurrent.futures import ThreadPoolExecutor

    from tokenizers import Tokenizer

    from coding import fim
    from coding.langs import is_code_doc
    from core import progress

    tok = Tokenizer.from_file(tok_path)
    assert tok.get_vocab_size() < 65536, "uint16 storage needs vocab < 65536"
    eot = tok.token_to_id(EOT)
    ids = fim.token_ids(tok)
    if fim_rate > 0 and not ids:
        print("this tokenizer has no fill-in-the-middle tokens (train it with --code); encoding without FIM")
        fim_rate = 0.0
    run = getattr(tok, "encode_batch_fast", tok.encode_batch)   # no character offsets: a bit faster
    files = shards(raw)
    sizes = [os.path.getsize(f) for f in files]
    total = sum(sizes)
    board = progress.board()
    board.title, board.kind = "Encoding the corpus", "work"
    task = board.task("encode", "Encoding the corpus", total, "bytes")
    task.verb = "Encoding"
    task.start(note=f"file 1 of {len(files)}")
    print(f"encoding {len(files)} files, {progress.fmt_bytes(total)} of text, with {os.cpu_count()} CPU threads")
    bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="encoding",
               mininterval=10 if progress.active() else 0.5)

    os.makedirs(bin_dir, exist_ok=True)
    out = {s: open(os.path.join(bin_dir, f"{s}.bin"), "wb") for s in ("train", "val")}
    counts = {"train": 0, "val": 0}
    rng, frng, n_fim = random.Random(0), random.Random(seed + 1), 0
    tail = np.array([eot], dtype=np.uint16)

    def write(pieces, plan, enc):
        nonlocal n_fim
        parts = {"train": [], "val": []}
        for start, n in plan:
            if n == 1:
                seq = enc[start].ids
            else:
                pre, suf, mid = (e.ids for e in enc[start:start + 3])
                seq = [ids["prefix"], *pre, ids["suffix"], *suf, ids["middle"], *mid]
                n_fim += 1
            split = "val" if rng.random() < val_frac else "train"
            parts[split] += [np.array(seq, dtype=np.uint16), tail]
        for split, arrs in parts.items():
            if arrs:
                a = np.concatenate(arrs)
                a.tofile(out[split])
                counts[split] += len(a)

    done_before = 0
    try:
        with ThreadPoolExecutor(1) as pool:      # the tokenizer runs the next batch while this one is written
            for i, (path, size) in enumerate(zip(files, sizes)):
                text = open(path, encoding="utf-8").read()
                n_chars, shown = max(1, len(text)), 0
                pending = None
                for pieces, plan, pos in _batches(text, fim_rate, frng, is_code_doc, fim):
                    job = pool.submit(run, pieces)
                    if pending:
                        write(*pending[:2], pending[2].result())
                    pending = (pieces, plan, job)
                    # progress counts the batch just handed to the tokenizer; close enough at a few MB
                    now = int(size * pos / n_chars)
                    bar.update(now - shown)
                    shown = now
                    tokens = counts["train"] + counts["val"]
                    task.set(done=done_before + now, note=f"file {i + 1} of {len(files)} · {tokens / 1e6:,.1f}M tokens")
                if pending:
                    write(*pending[:2], pending[2].result())
                bar.update(size - shown)
                done_before += size
                task.set(done=done_before)
                del text
    except BaseException:
        task.finish("failed")
        raise
    finally:
        bar.close()
        for f in out.values():
            f.close()
    tokens = counts["train"] + counts["val"]
    task.finish(note=f"{len(files)} files · {tokens / 1e6:,.1f}M tokens")
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
