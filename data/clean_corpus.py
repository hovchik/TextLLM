#!/usr/bin/env python3
"""Clean the raw corpus before training: drop low-quality documents and exact or near duplicates.

    python data/clean_corpus.py                       # data/raw -> data/clean
    python data/clean_corpus.py --threshold 0.7       # stricter near-duplicate matching
    python data/clean_corpus.py --no-near             # exact duplicates and quality only (less RAM)
    python tokenizer/train_tokenizer.py --raw data/clean ...   # then train on the clean copy

Reads every data/raw/<source>/*.txt shard (documents separated by <|endoftext|>, as written by
data/build_corpus.py, the crawler and learn.py) and writes the kept documents to
data/clean/<source>/ in the same format. data/raw is never changed, so downloads can keep
resuming. A report with counts per source and per reason goes to data/clean/report.json.

Why: duplicated text (the same book from two sources, reposted news, boilerplate pages) makes a
model memorise instead of generalise, and junk (tables of numbers, menus, OCR noise) wastes
training steps. Removing both is one of the cheapest ways to get a better model from the same data.

Filters (simple versions of the ones used for Gopher / FineWeb):
  - fewer than --min-words words
  - mean word length outside 3..10 characters (gibberish, number tables)
  - less than 70% of non-space characters are letters
  - more than 30% of non-empty lines are repeats of other lines in the same document
  - exact duplicate (after lower-casing and collapsing whitespace)
  - near duplicate: MinHash LSH over word 5-grams, Jaccard >= --threshold (needs datasketch,
    installed with the AI toolkit; without it only exact duplicates are removed)
Sources are processed in alphabetical order and the first copy of a document wins.
"""
import argparse, glob, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict

SEP = "<|endoftext|>"
SHARD_BYTES = 100 * 1024 * 1024
WORD = re.compile(r"\w+", re.UNICODE)


def documents(path):
    """Yield the documents of one shard, streaming so 100 MB shards don't need 100 MB strings twice."""
    buf = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == SEP:
                doc = "".join(buf).strip()
                if doc:
                    yield doc
                buf = []
            else:
                buf.append(line)
    doc = "".join(buf).strip()
    if doc:
        yield doc


def quality_reason(doc, min_words):
    """None if the document passes, otherwise a short reason."""
    words = WORD.findall(doc)
    if len(words) < min_words:
        return "too_short"
    mean = sum(map(len, words)) / len(words)
    if not 3 <= mean <= 10:
        return "odd_word_length"
    chars = [c for c in doc if not c.isspace()]
    if chars and sum(c.isalpha() for c in chars) / len(chars) < 0.7:
        return "few_letters"
    lines = [l.strip() for l in doc.splitlines() if l.strip()]
    if len(lines) >= 5:
        counts = Counter(lines)
        repeated = sum(n for n in counts.values() if n > 1)
        if repeated / len(lines) > 0.3:
            return "repeated_lines"
    return None


def normalise(doc):
    return " ".join(doc.lower().split())


def minhash(words, num_perm, max_words):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    words = words[:max_words]  # a whole book is not needed to recognise a copy of it
    m.update_batch(" ".join(words[i:i + 5]).encode("utf-8") for i in range(max(1, len(words) - 4)))
    return m


class Writer:
    """Writes kept documents to <out>/<source>/shard_XXXXX.txt, ~100 MB each."""

    def __init__(self, out_dir):
        self.dir, self.n, self.f, self.size = out_dir, 0, None, 0
        os.makedirs(out_dir, exist_ok=True)

    def write(self, doc):
        if self.f is None or self.size >= SHARD_BYTES:
            self.close()
            self.f = open(os.path.join(self.dir, f"shard_{self.n:05d}.txt"), "w", encoding="utf-8")
            self.n, self.size = self.n + 1, 0
        data = doc + "\n" + SEP + "\n"
        self.f.write(data)
        self.size += len(data.encode("utf-8"))

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


def sources(raw):
    """{source name: [shard paths]}; folders starting with _ (crawler state) are skipped."""
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(raw, "**", "*.txt"), recursive=True)):
        rel = os.path.relpath(path, raw)
        parts = rel.split(os.sep)
        if any(p.startswith("_") for p in parts):
            continue
        out[parts[0] if len(parts) > 1 else "_root"].append(path)
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description="Remove low-quality and duplicate documents from the corpus.")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity for near duplicates")
    ap.add_argument("--num-perm", type=int, default=128, help="MinHash permutations (more = exact, slower)")
    ap.add_argument("--max-words", type=int, default=20000, help="words per document used for MinHash")
    ap.add_argument("--no-near", action="store_true", help="skip near-duplicate detection")
    ap.add_argument("--no-quality", action="store_true", help="skip the quality filters")
    a = ap.parse_args()

    srcs = sources(a.raw)
    if not srcs:
        sys.exit(f"No .txt shards under {a.raw}. Download a corpus first (data/build_corpus.py).")
    raw_real, out_real = os.path.realpath(a.raw), os.path.realpath(a.out)
    if (raw_real == out_real or out_real.startswith(raw_real + os.sep)
            or raw_real.startswith(out_real + os.sep)):
        sys.exit("--out must not be, contain, or sit inside --raw (the raw corpus is never modified).")
    # Start from an empty output: shards from an earlier run (including sources that are gone
    # now) would otherwise still be picked up by the tokenizer.
    for old in glob.glob(os.path.join(a.out, "*", "shard_*.txt")):
        os.remove(old)

    lsh = None
    if not a.no_near:
        try:
            from datasketch import MinHashLSH
            lsh = MinHashLSH(threshold=a.threshold, num_perm=a.num_perm)
        except ImportError:
            print("datasketch is not installed, so only exact duplicates are removed. "
                  "Install it with: pip install -r requirements-ai.txt", flush=True)

    seen = set()
    report = {"settings": vars(a), "sources": {}}
    total = Counter()
    start = time.time()
    for name, shards in srcs.items():
        stats = Counter()
        writer = Writer(os.path.join(a.out, name))
        for shard in shards:
            for doc in documents(shard):
                stats["in"] += 1
                reason = None if a.no_quality else quality_reason(doc, a.min_words)
                if reason is None:
                    norm = normalise(doc)
                    key = hashlib.sha1(norm.encode("utf-8")).digest()
                    if key in seen:
                        reason = "exact_duplicate"
                    else:
                        seen.add(key)
                        if lsh is not None:
                            m = minhash(norm.split(), a.num_perm, a.max_words)
                            if lsh.query(m):
                                reason = "near_duplicate"
                            else:
                                lsh.insert(f"{name}:{stats['in']}:{len(seen)}", m)
                if reason:
                    stats[reason] += 1
                else:
                    stats["kept"] += 1
                    writer.write(doc)
                if stats["in"] % 5000 == 0:
                    print(f"  {name}: {stats['in']:,} read, {stats['kept']:,} kept", flush=True)
        writer.close()
        report["sources"][name] = dict(stats)
        total.update(stats)
        dropped = stats["in"] - stats["kept"]
        print(f"{name}: kept {stats['kept']:,} of {stats['in']:,} documents "
              f"({dropped:,} removed: " + ", ".join(f"{k} {v:,}" for k, v in stats.items()
                                                   if k not in ("in", "kept")) + ")", flush=True)

    report["total"] = dict(total)
    report["seconds"] = round(time.time() - start, 1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nKept {total['kept']:,} of {total['in']:,} documents in {report['seconds']}s. "
          f"Clean corpus: {a.out}  (report: {os.path.join(a.out, 'report.json')})")


if __name__ == "__main__":
    main()
