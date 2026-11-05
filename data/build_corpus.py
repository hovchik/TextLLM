"""Build a plain-text training corpus from open sources.

Sources:
  * Project Gutenberg (public-domain books), English only.
  * FineWeb-Edu (ODC-By licensed web text) for modern prose / posts.

Output: data/raw/{gutenberg,fineweb}/shard_XXXXX.txt  (~100 MB each)

Usage:
  pip install requests datasets tqdm
  python build_corpus.py --gutenberg-max 70000 --fineweb-gb 20
"""
import argparse, csv, io, os, re, time
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
# Please use a mirror for bulk downloads (Gutenberg asks robots not to hit www).
MIRROR = "https://aleph.gutenberg.org"
SHARD_BYTES = 100 * 1024 * 1024

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)


class ShardWriter:
    def __init__(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.outdir, self.idx, self.size, self.f = outdir, 0, 0, None
        self._open()

    def _open(self):
        if self.f:
            self.f.close()
        self.f = open(os.path.join(self.outdir, f"shard_{self.idx:05d}.txt"), "w", encoding="utf-8")
        self.idx += 1
        self.size = 0

    def write(self, doc):
        data = doc.strip() + "\n<|endoftext|>\n"
        self.f.write(data)
        self.size += len(data.encode("utf-8"))
        if self.size >= SHARD_BYTES:
            self._open()

    def close(self):
        self.f.close()


def strip_gutenberg(text):
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def mirror_url(book_id):
    s = str(book_id)
    path = "/".join(s[:-1]) if len(s) > 1 else "0"
    return f"{MIRROR}/{path}/{s}/{s}-0.txt", f"{MIRROR}/{path}/{s}/{s}.txt"


def fetch_book(book_id, session):
    for url in mirror_url(book_id):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                r.encoding = r.encoding or "utf-8"
                return strip_gutenberg(r.text)
        except requests.RequestException:
            time.sleep(1)
    return None


def gutenberg(outdir, max_books, workers):
    cat = requests.get(CATALOG, timeout=60).text
    ids = [
        row["Text#"]
        for row in csv.DictReader(io.StringIO(cat))
        if row["Type"] == "Text" and row["Language"] == "en"
    ][:max_books]
    print(f"Gutenberg: {len(ids)} English books")
    w, session = ShardWriter(outdir), requests.Session()
    with ThreadPoolExecutor(workers) as pool:
        for text in tqdm(pool.map(lambda i: fetch_book(i, session), ids), total=len(ids)):
            if text and len(text) > 2000:
                w.write(text)
    w.close()


def fineweb(outdir, max_gb):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    w, budget, done = ShardWriter(outdir), max_gb * 1024**3, 0
    for ex in tqdm(ds, desc="FineWeb-Edu"):
        w.write(ex["text"])
        done += len(ex["text"])
        if done >= budget:
            break
    w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--gutenberg-max", type=int, default=70000)
    ap.add_argument("--fineweb-gb", type=float, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.gutenberg_max > 0:
        gutenberg(os.path.join(a.out, "gutenberg"), a.gutenberg_max, a.workers)
    if a.fineweb_gb > 0:
        fineweb(os.path.join(a.out, "fineweb"), a.fineweb_gb)
