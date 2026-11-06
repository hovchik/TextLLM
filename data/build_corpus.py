"""Build a plain-text training corpus from open, rights-cleared sources.

Every source writes to its own folder, data/raw/<source>/shard_XXXXX.txt (~100 MB each). A source
is skipped when its size is 0 (Gutenberg and FineWeb-Edu have defaults, the rest start at 0):

  source             flag                    what it is                                      license
  -----------------  ----------------------  ----------------------------------------------  -----------------
  gutenberg          --gutenberg-max N       Project Gutenberg, English books                 public domain (US)
  pg19               --pg19-books N          DeepMind PG-19: 28k Gutenberg books before 1919  Apache 2.0 / PD
  standard_ebooks    --standard-ebooks N     Standard Ebooks: proofread public-domain books   CC0
  common_corpus      --common-corpus-gb G    Pleias Common Corpus, "Open Culture" part only   public domain
  archive            --archive-books N       Internet Archive scans published before 1931     public domain (US)
  cosmopedia         --cosmopedia-gb G       Synthetic stories and textbooks (Mixtral)        Apache 2.0
  fineweb            --fineweb-gb G          FineWeb-Edu educational web text                 ODC-By

Left out on purpose: BookCorpus and BookCorpusOpen (copyrighted Smashwords books copied without
permission), The Pile (contains Books3, taken down after a DMCA notice), the RedPajama book subset
(PG-19 + Books3), Open Library dumps (catalog metadata, no book text). The Dolma book subset is
Project Gutenberg again, so use --gutenberg-max / --pg19-books instead.

Usage:
  pip install requests pyarrow tqdm
  python data/build_corpus.py --gutenberg-max 3000 --pg19-books 2000 --cosmopedia-gb 0.5 --fineweb-gb 0.3

Everything resumes: run the same command again after a crash, a closed laptop or a dropped
connection. Each source keeps data/raw/<source>/progress.json, saved after every small chunk, and
network errors are retried with backoff (--retries). Books that Gutenberg already saved are
listed in gutenberg/ids.list (not .txt, so the tokenizer never reads it) and PG-19 skips them.
"""
import argparse, csv, html, io, json, os, posixpath, random, re, sys, time, zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import quote, urljoin

import requests
from tqdm import tqdm

CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
# Please use a mirror for bulk downloads (Gutenberg asks robots not to hit www).
MIRROR = "https://aleph.gutenberg.org"
# The robot harvest endpoint is Gutenberg's own sanctioned way to bulk download: it lists zip files
# on the mirror, a page at a time.
HARVEST = "https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
SHARD_BYTES = 100 * 1024 * 1024
UA = "textllm-corpus-builder (+https://github.com/hovchik/textllm)"

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


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


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, path)


class Progress:
    """progress.json plus a shard writer for one source, so any source resumes where it stopped.

    limit is a number of books (unit="books") or a number of bytes of text (unit="chars")."""

    def __init__(self, outdir, label, limit, unit="chars", **cursor):
        os.makedirs(outdir, exist_ok=True)
        self.outdir, self.label, self.limit, self.unit = outdir, label, limit, unit
        self.path = os.path.join(outdir, "progress.json")
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.st = json.load(f)
            self.st.setdefault("docs", 0)
        else:
            self.st = {"chars": 0, "docs": 0, "shard": 0, "shard_bytes": 0, **cursor}
        self.w = self.bar = None

    def count(self):
        return self.st["docs"] if self.unit == "books" else self.st["chars"]

    def _fmt(self, n):
        return f"{n:,} books" if self.unit == "books" else f"{n / 1e9:.2f} GB"

    def full(self):
        return self.count() >= self.limit

    def finished(self):
        """True (with a message) if there is nothing left to do."""
        if self.st.get("end"):
            print(f"{self.label}: the whole source is already saved ({self._fmt(self.count())})")
            return True
        if self.full():
            print(f"{self.label}: already have {self._fmt(self.count())}, nothing to do")
            return True
        if self.count():
            print(f"{self.label}: resuming at {self._fmt(self.count())}")
        return False

    def __enter__(self):
        self.w = ShardWriter(self.outdir, self.st["shard"], self.st["shard_bytes"])
        books = self.unit == "books"
        self.bar = tqdm(total=self.limit, initial=self.count(), desc=self.label,
                        unit="book" if books else "B", unit_scale=not books)
        return self

    def __exit__(self, *exc):
        self.bar.close()
        self.w.close()

    def write(self, text):
        self.w.write(text)
        self.st["chars"] += len(text)
        self.st["docs"] += 1
        self.bar.update(1 if self.unit == "books" else len(text))

    def save(self, **cursor):
        self.st.update(cursor)
        self.st["shard"], self.st["shard_bytes"] = self.w.checkpoint()
        save_state(self.path, self.st)

    def end(self, **cursor):
        self.save(end=True, **cursor)
        print(f"{self.label}: reached the end of the source ({self._fmt(self.count())})")


# ---------------------------------------------------------------- networking helpers

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


def session(token_env=None):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    tok = os.environ.get(token_env) if token_env else None
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    return s


def get(sess, url, tries, what=None, missing_ok=False, **kw):
    """GET with retries. Returns None for 404/403/410 when missing_ok, so one missing file is skipped."""
    def go():
        r = sess.get(url, timeout=(30, 300), **kw)
        if r.status_code in RETRY_STATUS:
            raise NetError(f"HTTP {r.status_code}")
        if missing_ok and r.status_code in (401, 403, 404, 410):
            return None
        r.raise_for_status()
        return r
    return retry(go, what or f"fetching {url.rsplit('/', 1)[-1][:60]}", tries)


def in_chunks(items, start, size):
    """Yield (next_position, chunk) so the caller can save progress after each chunk."""
    for i in range(start, len(items), size):
        yield i + size, items[i:i + size]


def run_list(p, items, fetch, pool, chunk, min_len=2000, on_saved=None):
    """Fetch items in order, a chunk at a time in parallel, saving progress after every chunk."""
    for i in range(p.st.get("pos", 0), len(items), chunk):
        part, saved = items[i:i + chunk], []
        nxt = i + len(part)
        for j, (item, text) in enumerate(zip(part, pool.map(fetch, part))):
            if text and len(text) > min_len:
                p.write(text)
                saved.append(item)
                if p.full():
                    nxt = i + j + 1          # a bigger limit later continues right after this book
                    break
        p.save(pos=nxt)
        if on_saved:
            on_saved(saved)
        if p.full():
            return
    p.end()


def tidy(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def glob_shards(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".txt"))


# ---------------------------------------------------------------- Project Gutenberg

def strip_gutenberg(text):
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return tidy(text)


def mirror_url(book_id):
    s = str(book_id)
    path = "/".join(s[:-1]) if len(s) > 1 else "0"
    return f"{MIRROR}/{path}/{s}/{s}-0.txt", f"{MIRROR}/{path}/{s}/{s}.txt"


def fetch_book(book_id, sess, tries=3):
    for url in mirror_url(book_id):
        try:
            r = get(sess, url, tries, missing_ok=True)
            if r is not None and len(r.content) > 5000:
                r.encoding = r.encoding or "utf-8"
                return strip_gutenberg(r.text)
        except requests.RequestException:
            time.sleep(1)
    return None


def catalog_ids(sess, tries):
    cat = get(sess, CATALOG, tries, "downloading the Gutenberg catalog").text
    return [row["Text#"] for row in csv.DictReader(io.StringIO(cat))
            if row["Type"] == "Text" and row["Language"] == "en"]


def record_ids(outdir, ids):
    if ids:
        with open(os.path.join(outdir, "ids.list"), "a") as f:
            f.write("".join(f"{i}\n" for i in ids))


def gutenberg_ids(outdir, tries):
    """Gutenberg ebook numbers already saved under outdir, so other sources can skip them."""
    ids = set()
    path = os.path.join(outdir, "ids.list")
    if os.path.exists(path):
        ids |= {line.strip() for line in open(path) if line.strip()}
    marker = os.path.join(outdir, "_done.json")
    if os.path.exists(marker):          # finished by an older version that didn't write ids.list
        n = json.load(open(marker)).get("max_books", 0)
        try:
            ids |= set(catalog_ids(session(), tries)[:n])
        except requests.RequestException as e:
            print(f"  (couldn't read the Gutenberg catalog to skip duplicates: {e})")
    return ids


def gutenberg(outdir, max_books, workers, method="mirror", tries=30):
    marker = os.path.join(outdir, "_done.json")
    if not os.path.exists(os.path.join(outdir, "progress.json")):
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
    p = Progress(outdir, "Gutenberg", max_books, unit="books", method=method, pos=0, page=None)
    if p.finished():
        return
    if p.st.get("method", "mirror") != method:
        print(f"Gutenberg: continuing with --gutenberg-method {p.st['method']} like the earlier run")
        method = p.st["method"]
    (gutenberg_harvest if method == "harvest" else gutenberg_mirror)(p, workers, tries)


def gutenberg_mirror(p, workers, tries):
    sess = session()
    ids = catalog_ids(sess, tries)
    print(f"Gutenberg: {len(ids):,} English books in the catalog")
    with p, ThreadPoolExecutor(workers) as pool:
        run_list(p, ids, lambda i: fetch_book(i, sess), pool, workers * 4,
                 on_saved=lambda saved: record_ids(p.outdir, saved))


ZIP_RE = re.compile(r"/(\d+)(-\d)?\.zip$")


def harvest_page(sess, url, tries):
    page = get(sess, url, tries, "reading a Gutenberg harvest page").text
    zips = [html.unescape(h) for h in re.findall(r'href="([^"]+\.zip)"', page)]
    nxt = re.search(r'href="([^"]*harvest\?[^"]*offset=[^"]*)"[^>]*>\s*Next', page, re.I)
    return zips, urljoin(url, html.unescape(nxt.group(1))) if nxt else None


def fetch_zip_text(sess, url, tries):
    r = get(sess, url.replace("http://", "https://", 1), tries, missing_ok=True)
    if r is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".txt")), None)
            if not name:
                return None
            raw = z.read(name)
    except zipfile.BadZipFile:
        return None
    enc = "latin-1" if re.search(r"-8\.zip$", url) else "utf-8"
    try:
        return strip_gutenberg(raw.decode(enc))
    except UnicodeDecodeError:
        return strip_gutenberg(raw.decode("latin-1"))


def gutenberg_harvest(p, workers, tries):
    sess = session()
    done = gutenberg_ids(p.outdir, tries)
    url = p.st.get("page") or HARVEST
    with p, ThreadPoolExecutor(workers) as pool:
        while url:
            zips, nxt = harvest_page(sess, url, tries)
            best = {}                # one file per book: prefer UTF-8 (-0), then Latin-1 (-8), then ASCII
            rank = {"-0": 0, "-8": 1, None: 2}
            for z in zips:
                m = ZIP_RE.search(z)
                if m and m.group(1) not in done:
                    bid, r = m.group(1), rank.get(m.group(2), 3)
                    if bid not in best or r < best[bid][0]:
                        best[bid] = (r, z)
            todo = [(bid, z) for bid, (_, z) in best.items()]
            for _, chunk in in_chunks(todo, 0, workers * 4):
                saved = []
                for (bid, _), text in zip(chunk, pool.map(lambda t: fetch_zip_text(sess, t[1], tries), chunk)):
                    if text and len(text) > 2000 and not p.full():
                        p.write(text)
                        saved.append(bid)
                    done.add(bid)
                p.save(page=url)         # re-reading this page on resume skips the saved books
                record_ids(p.outdir, saved)
                if p.full():
                    return
            url = nxt
            p.save(page=url)
        p.end()


# ---------------------------------------------------------------- PG-19 (DeepMind)
# The same files as huggingface.co/datasets/deepmind/pg19, read straight from DeepMind's public
# bucket: plain text, one Gutenberg book per file, named by its Gutenberg ebook number.

PG19_BUCKET = "deepmind-gutenberg"


def pg19_files(sess, tries):
    names = []
    for split in ("train", "validation", "test"):
        token = None
        while True:
            params = {"prefix": f"{split}/", "fields": "items(name),nextPageToken", "maxResults": 1000}
            if token:
                params["pageToken"] = token
            js = get(sess, f"https://storage.googleapis.com/storage/v1/b/{PG19_BUCKET}/o", tries,
                     "listing PG-19 books", params=params).json()
            names += [i["name"] for i in js.get("items", []) if i["name"].endswith(".txt")]
            token = js.get("nextPageToken")
            if not token:
                break
    order = {"train": 0, "validation": 1, "test": 2}
    key = lambda n: (order.get(n.split("/")[0], 3), int(re.sub(r"\D", "", n.rsplit("/", 1)[-1]) or 0))
    return sorted(names, key=key)


def pg19(outdir, max_books, workers, tries, gutenberg_dir):
    p = Progress(outdir, "PG-19", max_books, unit="books", pos=0)
    if p.finished():
        return
    sess = session()
    names = pg19_files(sess, tries)
    skip = gutenberg_ids(gutenberg_dir, tries)
    print(f"PG-19: {len(names):,} books" + (", skipping the ones Gutenberg already saved" if skip else ""))

    def fetch(name):
        if name.rsplit("/", 1)[-1][:-4] in skip:
            return None
        r = get(sess, f"https://storage.googleapis.com/{PG19_BUCKET}/{quote(name)}", tries, missing_ok=True)
        return tidy(r.content.decode("utf-8", "replace")) if r is not None else None

    with p, ThreadPoolExecutor(workers) as pool:
        run_list(p, names, fetch, pool, workers * 4)


# ---------------------------------------------------------------- Standard Ebooks
# Standard Ebooks' editions are CC0. Two ways in:
#  * STANDARD_EBOOKS_EMAIL set (Patrons Circle members): the OPDS/Atom feed and its epub files.
#  * otherwise: the public GitHub repositories (github.com/standardebooks), one per book.

SE_FEED = "https://standardebooks.org/feeds/atom/all"
SE_SKIP = {"titlepage", "halftitlepage", "imprint", "colophon", "uncopyright", "toc", "loi",
           "endnotes", "copyright"}


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr",
             "section", "article", "header", "footer", "hr", "table", "dt", "dd"}

    def __init__(self):
        super().__init__()
        self.out, self.hide = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.hide += 1
        elif tag in self.BLOCK:
            self.out.append("\n\n" if tag not in ("br", "li", "tr") else "\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.hide = max(0, self.hide - 1)
        elif tag in self.BLOCK and tag != "br":
            self.out.append("\n\n" if tag not in ("br", "li", "tr") else "\n")

    def handle_data(self, data):
        if not self.hide:
            self.out.append(re.sub(r"\s+", " ", data))


def html_text(markup):
    p = _Text()
    p.feed(markup if isinstance(markup, str) else markup.decode("utf-8", "replace"))
    text = "".join(p.out)
    text = re.sub(r" *\n *", "\n", text)
    return tidy(text)


def opf_text(opf, read):
    """Book text in reading order from an epub package file; read(href) returns a file's bytes."""
    root = ET.fromstring(opf)
    items = {i.get("id"): i.get("href") for i in root.findall(".//{*}item")
             if "html" in (i.get("media-type") or "")}
    parts = []
    for ref in root.findall(".//{*}itemref"):
        href = items.get(ref.get("idref"))
        if not href or posixpath.splitext(posixpath.basename(href))[0] in SE_SKIP:
            continue
        data = read(href)
        if data:
            parts.append(html_text(data))
    return "\n\n".join(p for p in parts if p)


def epub_text(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        cont = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = cont.find(".//{*}rootfile").get("full-path")
        base = posixpath.dirname(opf_path)
        names = set(z.namelist())

        def read(href):
            n = posixpath.normpath(posixpath.join(base, href))
            return z.read(n) if n in names else None
        return opf_text(z.read(opf_path), read)


def se_books_feed(sess, email, tries):
    r = get(sess, SE_FEED, tries, "reading the Standard Ebooks feed", auth=(email, ""), missing_ok=True)
    if r is None or "xml" not in r.headers.get("Content-Type", "xml"):
        raise SystemExit("Standard Ebooks: the feed refused STANDARD_EBOOKS_EMAIL. Feeds need a Patrons "
                         "Circle membership; unset STANDARD_EBOOKS_EMAIL to use the GitHub copies instead.")
    books = []
    for e in ET.fromstring(r.content).iter("{http://www.w3.org/2005/Atom}entry"):
        bid = e.findtext("{http://www.w3.org/2005/Atom}id")
        epubs = [l.get("href") for l in e.iter("{http://www.w3.org/2005/Atom}link")
                 if l.get("type") == "application/epub+zip"]
        plain = [h for h in epubs if "advanced" not in h and not h.endswith(".kepub.epub")]
        if bid and (plain or epubs):
            books.append({"id": bid, "epub": urljoin(SE_FEED, (plain or epubs)[0])})
    return books


def se_books_github(sess, tries):
    books, page = [], 1
    while True:
        r = get(sess, "https://api.github.com/orgs/standardebooks/repos", tries,
                "listing Standard Ebooks repositories", params={"per_page": 100, "page": page})
        repos = r.json()
        if not repos:
            break
        # book repositories are named author_title; the rest are tools and the website
        books += [{"id": x["name"], "branch": x.get("default_branch") or "master"}
                  for x in repos if "_" in x["name"] and not x.get("archived")]
        page += 1
    return sorted(books, key=lambda b: b["id"])


def standard_ebooks(outdir, max_books, workers, tries):
    p = Progress(outdir, "Standard Ebooks", max_books, unit="books", pos=0)
    if p.finished():
        return
    email = os.environ.get("STANDARD_EBOOKS_EMAIL", "").strip()
    sess = session("GITHUB_TOKEN" if not email else None)
    listing = os.path.join(outdir, "books.json")       # keep the list so a resume sees the same order
    if os.path.exists(listing):
        books = json.load(open(listing))
    else:
        books = se_books_feed(sess, email, tries) if email else se_books_github(sess, tries)
        save_state(listing, books)
    print(f"Standard Ebooks: {len(books):,} books ({'Patrons feed' if email else 'GitHub'})")

    def fetch(b):
        try:
            if "epub" in b:
                r = get(sess, b["epub"], tries, missing_ok=True, auth=(email, ""))
                return epub_text(r.content) if r is not None else None
            raw = f"https://raw.githubusercontent.com/standardebooks/{b['id']}/{b['branch']}/src/epub/"
            opf = get(sess, raw + "content.opf", tries, missing_ok=True)
            if opf is None:
                return None
            read = lambda href: getattr(get(sess, raw + quote(href), tries, missing_ok=True), "content", None)
            return opf_text(opf.content, read)
        except (ET.ParseError, zipfile.BadZipFile, KeyError, AttributeError) as e:
            print(f"  skipping {b['id']}: {e}")
            return None

    with p, ThreadPoolExecutor(min(workers, 4)) as pool:
        run_list(p, books, fetch, pool, 8)


# ---------------------------------------------------------------- Internet Archive
# Only scanned books published before 1931 (public domain in the US since 1 Jan 2026) from the
# library-digitization collections, never the lending library. The text is OCR, so it is noisier
# than Gutenberg; keep this a small share of the mix.

IA_QUERY = ("mediatype:texts AND date:[1000-01-01 TO 1930-12-31] AND language:(eng OR English) "
            "AND collection:(americana OR toronto OR europeanlibraries) "
            "AND NOT collection:(inlibrary OR printdisabled OR lendinglibrary)")
IA_SCRAPE = "https://archive.org/services/search/v1/scrape"


def clean_ocr(text):
    text = text.replace("\r\n", "\n").replace("\f", "\n\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                 # words split across lines
    paras = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in para.split("\n")]
        lines = [l for l in lines if l and not re.fullmatch(r"[\d\W]{1,8}", l)]   # page numbers, specks
        if lines:
            paras.append(" ".join(lines))
    text = "\n\n".join(p for p in paras if len(p) > 3)
    letters = sum(c.isalpha() for c in text)
    return text if text and letters / len(text) > 0.7 else None


def archive(outdir, max_books, workers, tries, query=IA_QUERY):
    p = Progress(outdir, "Internet Archive", max_books, unit="books", cursor=None, done=[])
    if p.finished():
        return
    sess = session()
    done = set(p.st.get("done", []))

    def page(cursor):
        params = {"q": query, "fields": "identifier", "count": 1000}
        if cursor:
            params["cursor"] = cursor
        return get(sess, IA_SCRAPE, tries, "searching the Internet Archive", params=params).json()

    def fetch(ident):
        r = get(sess, f"https://archive.org/download/{quote(ident)}/{quote(ident)}_djvu.txt", tries,
                missing_ok=True)
        return clean_ocr(r.content.decode("utf-8", "replace")) if r is not None else None

    cursor = p.st.get("cursor")
    with p, ThreadPoolExecutor(min(workers, 4)) as pool:   # be gentle with the Archive
        while True:
            try:
                js = page(cursor)
            except requests.HTTPError:
                if not cursor:
                    raise
                print("Internet Archive: the saved search position expired, starting the list again "
                      "(books already saved are skipped)")
                cursor = None
                js = page(None)
            ids = [i["identifier"] for i in js.get("items", []) if i["identifier"] not in done]
            for _, chunk in in_chunks(ids, 0, 8):
                for ident, text in zip(chunk, pool.map(fetch, chunk)):
                    if p.full():
                        break                     # leave the rest for a bigger limit later
                    if text and len(text) > 5000:
                        p.write(text)
                    done.add(ident)
                p.save(cursor=cursor, done=sorted(done))
                if p.full():
                    return
            cursor = js.get("cursor")
            if not cursor:
                break
            p.save(cursor=cursor, done=sorted(done))
        p.end()


# ---------------------------------------------------------------- Hugging Face parquet datasets
# Reads parquet files straight from Hugging Face with HTTP range requests, one row group at a time,
# so a dropped connection only costs one retry of one chunk. Progress (file, row group, shard
# offset) is saved after every row group. Used for FineWeb-Edu, Cosmopedia and Common Corpus.

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
FW_REPO = "HuggingFaceFW/fineweb-edu"
FW_DIR = "sample/10BT"
COSMO_REPO = "HuggingFaceTB/cosmopedia"
COSMO_SUBSETS = ["stories", "wikihow", "openstax", "khanacademy", "stanford",
                 "web_samples_v1", "web_samples_v2", "auto_math_text"]
CC_REPO = "PleIAs/common_corpus"


def hf_session():
    return session("HF_TOKEN")


def _tree(sess, repo, sha, d):
    url = f"{HF}/api/datasets/{repo}/tree/{sha}" + (f"/{d}" if d else "")
    params, out = {"recursive": "true"}, []
    while url:
        r = sess.get(url, params=params, timeout=60)
        r.raise_for_status()
        out += [(f["path"], f["size"]) for f in r.json()
                if f.get("type") == "file" and f["path"].endswith(".parquet")]
        url, params = r.links.get("next", {}).get("url"), None     # large repos are paged
    return sorted(out)


def hf_files(sess, repo, dirs, tries, sha=None):
    """Pin the dataset to one commit and list its parquet files (in dirs order) with their sizes."""
    def go():
        nonlocal sha
        if not sha:
            info = sess.get(f"{HF}/api/datasets/{repo}", timeout=60)
            info.raise_for_status()
            sha = info.json()["sha"]
        files = [f for d in dirs for f in _tree(sess, repo, sha, d)]
        if not files:
            raise NetError("no parquet files listed")
        return sha, files
    return retry(go, f"listing {repo} files", tries)


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests; every read is retried on its own."""

    def __init__(self, url, size, sess, tries):
        self.url, self._size, self.s, self.tries, self.pos = url, size, sess, tries, 0

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

        def go():
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

        data = retry(go, f"reading {self.url.rsplit('/', 1)[-1]} @ {start / 1e6:.0f} MB", self.tries)
        self.pos += n
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def download_file(url, dest, size, sess, tries):
    """Download url to dest, resuming a partial .part file after any disconnect."""
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    bar = tqdm(total=size, unit="B", unit_scale=True, desc=f"download {os.path.basename(dest)}")

    def go():
        have = os.path.getsize(part) if os.path.exists(part) else 0
        bar.n = have
        bar.refresh()
        if have >= size:
            return
        r = sess.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(30, 300))
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

    retry(go, f"downloading {os.path.basename(dest)}", tries)
    bar.close()
    if os.path.getsize(part) != size:
        raise SystemExit(f"{part}: size mismatch, delete it and run again")
    os.replace(part, dest)
    return dest


def adopt_legacy(outdir):
    """FineWeb shards from the old streaming version have no progress file. That version read the
    files in order, so the number of complete documents tells us how many rows to skip. Cut a
    half-written last document so the shards end cleanly."""
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
    return {"sha": None, "file": None, "row_group": 0, "skip_rows": docs, "chars": chars, "docs": docs,
            "shard": len(shards) - 1, "shard_bytes": cut}


def hf_parquet(outdir, label, repo, dirs, max_gb, mode="stream", tries=30, cache="data/cache",
               keep_parquet=False, column="text", filters=None, legacy=False, cache_name=None):
    """Stream the `column` text of a Hugging Face parquet dataset into shards, up to max_gb.

    filters maps column name -> predicate(value); a row is kept only if every predicate is true.
    Filter columns are read first, so row groups with no matching rows never download their text."""
    import pyarrow.parquet as pq

    p = Progress(outdir, label, int(max_gb * 1024**3), sha=None, file=None, row_group=0, skip_rows=0)
    if legacy and not os.path.exists(p.path):
        p.st.update(adopt_legacy(outdir) or {})
    if p.st.setdefault("dirs", dirs) != dirs:
        raise SystemExit(f"{label}: {outdir} was downloaded from {p.st['dirs']}, not {dirs}. "
                         f"Use another --out folder or delete {p.path} to start over.")
    if p.finished():
        return
    filters = filters or {}
    st = p.st

    sess = hf_session()
    sha, files = hf_files(sess, repo, dirs, tries)
    if st["sha"] and st["sha"] != sha:
        print(f"{label}: dataset updated since last run, keeping the old version {st['sha'][:8]}")
        sha, files = hf_files(sess, repo, dirs, tries, st["sha"])
    st["sha"] = sha
    names = [f for f, _ in files]
    start = names.index(st["file"]) if st["file"] in names else 0

    with p:
        for path, size in files[start:]:
            url = f"{HF}/datasets/{repo}/resolve/{sha}/{path}"
            local = None
            if mode == "download":
                local = download_file(url, os.path.join(cache, cache_name or repo.split("/")[-1], sha[:8], path),
                                      size, sess, tries)
                src = local
            else:
                src = HttpFile(url, size, sess, tries)
            pf = retry(lambda: pq.ParquetFile(src), f"opening {path}", tries)
            try:
                missing = [c for c in [column, *filters] if c not in pf.schema_arrow.names]
                if missing:
                    raise SystemExit(f"{label}: {path} has no column(s) {missing}; columns are "
                                     f"{pf.schema_arrow.names}. The dataset layout changed, please report it.")
                rg0 = st["row_group"] if st["file"] == path else 0
                st["file"] = path
                for rg in range(rg0, pf.num_row_groups):
                    rows = pf.metadata.row_group(rg).num_rows
                    if st["skip_rows"] >= rows:           # already saved by an earlier run
                        st["skip_rows"] -= rows
                        continue
                    used = st["skip_rows"]
                    keep = range(used, rows)
                    if filters:
                        cols = pf.read_row_group(rg, columns=list(filters))
                        vals = {c: cols.column(c).to_pylist() for c in filters}
                        keep = [i for i in keep if all(f(vals[c][i]) for c, f in filters.items())]
                    texts = pf.read_row_group(rg, columns=[column]).column(column).to_pylist() if keep else []
                    for i in keep:
                        t = texts[i]
                        used = i + 1
                        if t:
                            p.write(t)
                        if p.full():
                            break
                    if not p.full():
                        used = rows
                    if used < rows:                       # budget reached mid-chunk: remember where
                        p.save(row_group=rg, skip_rows=used)
                    else:
                        p.save(row_group=rg + 1, skip_rows=0)
                    if p.full():
                        return
                if path != names[-1]:
                    p.save(file=names[names.index(path) + 1], row_group=0)
                else:
                    p.st["end"] = True
                    p.save()
            finally:
                pf.close()        # Windows can't delete a file that is still open
            if local and not keep_parquet:
                os.remove(local)
        print(f"{label}: reached the end of the dataset")


def fineweb(outdir, max_gb, mode="stream", tries=30, cache="data/cache", keep_parquet=False):
    hf_parquet(outdir, "FineWeb-Edu", FW_REPO, [FW_DIR], max_gb, mode, tries, cache, keep_parquet,
               legacy=True, cache_name="fineweb")


def cosmopedia(outdir, max_gb, subsets, mode="stream", tries=30, cache="data/cache", keep_parquet=False):
    bad = [s for s in subsets if s not in COSMO_SUBSETS]
    if bad:
        raise SystemExit(f"unknown Cosmopedia subset(s) {bad}; choose from {COSMO_SUBSETS}")
    # one folder and progress file per subset, so changing the selection later never mixes cursors
    for sub in subsets:
        hf_parquet(os.path.join(outdir, sub), f"Cosmopedia {sub}", COSMO_REPO, [f"data/{sub}"],
                   max_gb / len(subsets), mode, tries, cache, keep_parquet)


def _lang_ok(lang):
    return lambda v: lang == "all" or (isinstance(v, str) and v.strip().lower()[:2] == lang[:2])


def _open_culture(v):
    """Common Corpus is split into Open Culture (public-domain books, newspapers), Open Government,
    Open Science, Open Source (code), Open Web and Open Semantic. Only keep Open Culture."""
    return isinstance(v, str) and "culture" in v.lower()


def common_corpus(outdir, max_gb, lang="en", mode="stream", tries=30, cache="data/cache",
                  keep_parquet=False):
    hf_parquet(outdir, "Common Corpus", CC_REPO, [""], max_gb, mode, tries, cache, keep_parquet,
               filters={"open_type": _open_culture, "language": _lang_ok(lang)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw")
    src = ap.add_argument_group("sources (0 = skip)")
    src.add_argument("--gutenberg-max", type=int, default=70000, help="Project Gutenberg books")
    src.add_argument("--gutenberg-method", choices=["mirror", "harvest"], default="mirror",
                     help="mirror: catalog order from aleph.gutenberg.org (default). harvest: Gutenberg's "
                          "robot harvest pages (zip files, the officially recommended way for bulk downloads)")
    src.add_argument("--pg19-books", type=int, default=0, help="DeepMind PG-19 books (28,752 in total)")
    src.add_argument("--standard-ebooks", type=int, default=0,
                     help="Standard Ebooks books (~1,400). Set STANDARD_EBOOKS_EMAIL to use the Patrons "
                          "feed; otherwise the books are read from their GitHub repositories (set "
                          "GITHUB_TOKEN to avoid GitHub's 60 requests/hour limit when listing)")
    src.add_argument("--common-corpus-gb", type=float, default=0, help="Common Corpus, Open Culture part")
    src.add_argument("--common-corpus-lang", default="en", help="language code to keep, or 'all'")
    src.add_argument("--archive-books", type=int, default=0, help="Internet Archive books published before 1931")
    src.add_argument("--cosmopedia-gb", type=float, default=0,
                     help="Cosmopedia synthetic text, split evenly over the chosen subsets")
    src.add_argument("--cosmopedia-subsets", default="stories",
                     help=f"comma-separated, from: {','.join(COSMO_SUBSETS)}")
    src.add_argument("--fineweb-gb", type=float, default=20, help="FineWeb-Edu web text")
    net = ap.add_argument_group("downloading")
    net.add_argument("--fineweb-mode", "--hf-mode", dest="hf_mode", choices=["stream", "download"],
                     default="stream",
                     help="Hugging Face sources. stream: read parquet chunks over HTTP (default). download: "
                          "save whole parquet files to --cache first (resumable), then read them.")
    net.add_argument("--cache", default="data/cache", help="where --hf-mode download keeps files")
    net.add_argument("--keep-parquet", action="store_true", help="keep downloaded parquet files after use")
    net.add_argument("--retries", type=int, default=30,
                     help="attempts per network request (backoff up to 2 min, so 30 rides out ~45 min offline)")
    net.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = lambda name: os.path.join(a.out, name)
    hf = dict(mode=a.hf_mode, tries=a.retries, cache=a.cache, keep_parquet=a.keep_parquet)
    try:
        if a.gutenberg_max > 0:
            gutenberg(out("gutenberg"), a.gutenberg_max, a.workers, a.gutenberg_method, a.retries)
        if a.pg19_books > 0:
            pg19(out("pg19"), a.pg19_books, a.workers, a.retries, out("gutenberg"))
        if a.standard_ebooks > 0:
            standard_ebooks(out("standard_ebooks"), a.standard_ebooks, a.workers, a.retries)
        if a.common_corpus_gb > 0:
            common_corpus(out("common_corpus"), a.common_corpus_gb, a.common_corpus_lang, **hf)
        if a.archive_books > 0:
            archive(out("archive"), a.archive_books, a.workers, a.retries)
        if a.cosmopedia_gb > 0:
            subsets = [s.strip() for s in a.cosmopedia_subsets.split(",") if s.strip()]
            cosmopedia(out("cosmopedia"), a.cosmopedia_gb, subsets, **hf)
        if a.fineweb_gb > 0:
            fineweb(out("fineweb"), a.fineweb_gb, **hf)
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run the same command again to continue where it left off.")
