"""Build a plain-text training corpus from open sources.

Sources:
  * Project Gutenberg (public-domain books), English only.
  * FineWeb-Edu (ODC-By licensed web text) for modern prose / posts.

Output: data/raw/{gutenberg,fineweb}/shard_XXXXX.txt  (~100 MB each)

Usage:
  pip install requests pyarrow tqdm
  python build_corpus.py --gutenberg-max 70000 --fineweb-gb 20

Interrupted downloads resume: run the same command again. FineWeb-Edu progress is saved in
data/raw/fineweb/progress.json after every chunk, and network drops are retried with backoff.
On a flaky connection, --fineweb-mode download fetches whole parquet files (resumable) first.
"""
import argparse, csv, io, json, os, random, re, time
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
    """Writes documents into ~100 MB shards. Can reopen a shard at a saved byte offset to resume."""

    def __init__(self, outdir, idx=0, size=0):
        os.makedirs(outdir, exist_ok=True)
        self.outdir, self.idx, self.size, self.f = outdir, idx, size, None
        path = self._path()
        if size and os.path.exists(path):
            # drop anything written after the last checkpoint, then append
            with open(path, "r+b") as f:
                f.truncate(size)
            self.f = open(path, "a", encoding="utf-8", newline="\n")
        else:
            self.size = 0
            self.f = open(path, "w", encoding="utf-8", newline="\n")

    def _path(self):
        return os.path.join(self.outdir, f"shard_{self.idx:05d}.txt")

    def write(self, doc):
        data = doc.strip() + "\n<|endoftext|>\n"
        self.f.write(data)
        self.size += len(data.encode("utf-8"))
        if self.size >= SHARD_BYTES:
            self.f.close()
            self.idx, self.size = self.idx + 1, 0
            self.f = open(self._path(), "w", encoding="utf-8", newline="\n")

    def checkpoint(self):
        """Flush to disk and return (shard index, byte size) that a resume can restart from."""
        self.f.flush()
        os.fsync(self.f.fileno())
        return self.idx, self.size

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
    marker = os.path.join(outdir, "_done.json")
    if os.path.exists(marker):
        done = json.load(open(marker)).get("max_books", 0)
        if done >= max_books:
            print(f"Gutenberg: already downloaded ({done} books requested last time), skipping. "
                  f"Delete {marker} to download again.")
            return
    elif glob_shards(outdir) and glob_shards(os.path.join(os.path.dirname(outdir), "fineweb")):
        # an older version of this script got past Gutenberg (FineWeb shards exist) but left no marker
        print("Gutenberg: shards from an earlier run found, skipping. "
              f"Delete {outdir} to download again.")
        json.dump({"max_books": max_books}, open(marker, "w"))
        return
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
    json.dump({"max_books": max_books}, open(marker, "w"))


def glob_shards(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".txt"))


# ---------------------------------------------------------------- FineWeb-Edu
# Reads the parquet files of the sample-10BT subset straight from Hugging Face with HTTP range
# requests, one row group at a time, so a dropped connection only costs one retry of one chunk.
# Progress (file, row group, shard offset) is saved after every row group, so a crashed or
# stopped run continues where it left off.

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
FW_REPO = "HuggingFaceFW/fineweb-edu"
FW_DIR = "sample/10BT"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class NetError(IOError):
    """A transient download problem (short read, bad status) that is worth retrying."""


def retry(fn, what, tries):
    """Call fn() and retry network errors with exponential backoff (2s, 4s, ... capped at 2 min)."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except (requests.RequestException, NetError, ConnectionError, TimeoutError) as e:
            if isinstance(e, requests.HTTPError) and e.response is not None \
                    and e.response.status_code not in RETRY_STATUS:
                raise
            if attempt == tries:
                raise
            wait = min(120, 2 ** attempt) * random.uniform(0.8, 1.2)
            print(f"\n  network error while {what}: {str(e)[:200]}\n"
                  f"  retry {attempt}/{tries - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def hf_session():
    s = requests.Session()
    s.headers["User-Agent"] = "textllm-corpus-builder"
    tok = os.environ.get("HF_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    return s


def fw_files(session, tries):
    """Pin the dataset to one commit and list its parquet files with their sizes."""
    def get():
        info = session.get(f"{HF}/api/datasets/{FW_REPO}", timeout=60)
        info.raise_for_status()
        sha = info.json()["sha"]
        files = _tree(session, sha)
        if not files:
            raise NetError("no parquet files listed")
        return sha, files
    return retry(get, "listing FineWeb-Edu files", tries)


def fw_url(sha, path):
    return f"{HF}/datasets/{FW_REPO}/resolve/{sha}/{path}"


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests; every read is retried on its own."""

    def __init__(self, url, size, session, tries):
        self.url, self._size, self.s, self.tries, self.pos = url, size, session, tries, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self._size + off}[whence]
        return self.pos

    def size(self):
        return self._size

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self.pos
        n = min(n, self._size - self.pos)
        if n <= 0:
            return b""
        start = self.pos

        def get():
            r = self.s.get(self.url, headers={"Range": f"bytes={start}-{start + n - 1}"},
                           timeout=(30, 300))
            if r.status_code in RETRY_STATUS:
                raise NetError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.content
            if r.status_code == 200:          # server ignored the range
                data = data[start:start + n]
            if len(data) != n:
                raise NetError(f"short read: got {len(data)} of {n} bytes")
            return data

        data = retry(get, f"reading {self.url.rsplit('/', 1)[-1]} @ {start / 1e6:.0f} MB", self.tries)
        self.pos += n
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def download_file(url, dest, size, session, tries):
    """Download url to dest, resuming a partial .part file after any disconnect."""
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    bar = tqdm(total=size, unit="B", unit_scale=True, desc=f"download {os.path.basename(dest)}")

    def get():
        have = os.path.getsize(part) if os.path.exists(part) else 0
        bar.n = have
        bar.refresh()
        if have >= size:
            return
        r = session.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(30, 300))
        if r.status_code in RETRY_STATUS:
            raise NetError(f"HTTP {r.status_code}")
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"   # 200 = server restarted from 0
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        if os.path.getsize(part) < size:
            raise NetError("connection ended early")

    retry(get, f"downloading {os.path.basename(dest)}", tries)
    bar.close()
    if os.path.getsize(part) != size:
        raise SystemExit(f"{part}: size mismatch, delete it and run again")
    os.replace(part, dest)
    return dest


def adopt_legacy(outdir):
    """Shards from the old streaming version have no progress file. That version read the files in
    order, so the number of complete documents tells us how many rows to skip. Cut a half-written
    last document so the shards end cleanly."""
    shards = glob_shards(outdir)
    if not shards:
        return None
    sep, docs, chars = b"<|endoftext|>", 0, 0
    for name in shards:
        with open(os.path.join(outdir, name), "rb") as f:
            data = f.read()
        docs += data.count(sep)
        chars += len(data)
    last = os.path.join(outdir, shards[-1])
    with open(last, "rb") as f:
        data = f.read()
    cut = data.rfind(sep)
    cut = 0 if cut < 0 else cut + len(sep)
    with open(last, "r+b") as f:
        f.truncate(cut)
        if cut:                               # end on a newline even if it was never written
            f.seek(cut)
            f.write(b"\n")
            cut += 1
    chars -= len(data) - cut
    print(f"FineWeb-Edu: continuing an earlier run ({docs:,} documents, {chars / 1e9:.2f} GB already saved)")
    return {"sha": None, "file": None, "row_group": 0, "skip_rows": docs, "chars": chars,
            "shard": len(shards) - 1, "shard_bytes": cut}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def fineweb(outdir, max_gb, mode="stream", tries=30, cache="data/cache/fineweb", keep_parquet=False):
    import pyarrow.parquet as pq

    os.makedirs(outdir, exist_ok=True)
    state_path = os.path.join(outdir, "progress.json")
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        print(f"FineWeb-Edu: resuming at {st['chars'] / 1e9:.2f} GB "
              f"({st['file'] or 'start'}, row group {st['row_group']})")
    else:
        st = adopt_legacy(outdir) or {"sha": None, "file": None, "row_group": 0, "skip_rows": 0,
                                       "chars": 0, "shard": 0, "shard_bytes": 0}
    budget = int(max_gb * 1024**3)
    if st.get("end"):
        print(f"FineWeb-Edu: the whole dataset is already saved ({st['chars'] / 1e9:.2f} GB)")
        return
    if st["chars"] >= budget:
        print(f"FineWeb-Edu: already have {st['chars'] / 1e9:.2f} GB, nothing to do")
        return

    session = hf_session()
    sha, files = fw_files(session, tries)
    if st["sha"] and st["sha"] != sha:
        print(f"FineWeb-Edu: dataset updated since last run, keeping the old version {st['sha'][:8]}")
        sha = st["sha"]
        files = retry(lambda: _tree(session, sha), "listing files", tries)
    st["sha"] = sha
    names = [p for p, _ in files]
    start = names.index(st["file"]) if st["file"] in names else 0

    w = ShardWriter(outdir, st["shard"], st["shard_bytes"])
    bar = tqdm(total=budget, initial=st["chars"], unit="B", unit_scale=True, desc="FineWeb-Edu")
    try:
        for path, size in files[start:]:
            url = fw_url(sha, path)
            local = None
            if mode == "download":
                local = download_file(url, os.path.join(cache, sha[:8], path), size, session, tries)
                src = local
            else:
                src = HttpFile(url, size, session, tries)
            pf = retry(lambda: pq.ParquetFile(src), f"opening {path}", tries)
            try:
                rg0 = st["row_group"] if st["file"] == path else 0
                st["file"] = path
                for rg in range(rg0, pf.num_row_groups):
                    rows = pf.metadata.row_group(rg).num_rows
                    if st["skip_rows"] >= rows:           # already saved by an earlier run
                        st["skip_rows"] -= rows
                        continue
                    texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
                    used = st["skip_rows"]
                    for t in texts[used:]:
                        w.write(t)
                        used += 1
                        st["chars"] += len(t)
                        bar.update(len(t))
                        if st["chars"] >= budget:
                            break
                    st["shard"], st["shard_bytes"] = w.checkpoint()
                    if used < rows:                       # budget reached mid-chunk: remember where
                        st["row_group"], st["skip_rows"] = rg, used
                    else:
                        st["row_group"], st["skip_rows"] = rg + 1, 0
                    save_state(state_path, st)
                    if st["chars"] >= budget:
                        return
                if path != names[-1]:
                    st["file"], st["row_group"] = names[names.index(path) + 1], 0
                else:
                    st["end"] = True
                save_state(state_path, st)
            finally:
                pf.close()        # Windows can't delete a file that is still open
            if local and not keep_parquet:
                os.remove(local)
        print("FineWeb-Edu: reached the end of the dataset")
    finally:
        bar.close()
        w.close()


def _tree(session, sha):
    tree = session.get(f"{HF}/api/datasets/{FW_REPO}/tree/{sha}/{FW_DIR}", timeout=60)
    tree.raise_for_status()
    return sorted((f["path"], f["size"]) for f in tree.json()
                  if f.get("type") == "file" and f["path"].endswith(".parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--gutenberg-max", type=int, default=70000)
    ap.add_argument("--fineweb-gb", type=float, default=20)
    ap.add_argument("--fineweb-mode", choices=["stream", "download"], default="stream",
                    help="stream: read parquet chunks over HTTP (default). download: save whole "
                         "parquet files to --cache first (about 2 GB each, resumable), then read them.")
    ap.add_argument("--cache", default="data/cache/fineweb", help="where --fineweb-mode download keeps files")
    ap.add_argument("--keep-parquet", action="store_true", help="keep downloaded parquet files after use")
    ap.add_argument("--retries", type=int, default=30,
                    help="attempts per network request (backoff up to 2 min, so 30 rides out ~45 min offline)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.gutenberg_max > 0:
        gutenberg(os.path.join(a.out, "gutenberg"), a.gutenberg_max, a.workers)
    if a.fineweb_gb > 0:
        fineweb(os.path.join(a.out, "fineweb"), a.fineweb_gb, a.fineweb_mode, a.retries,
                a.cache, a.keep_parquet)
