"""News and web crawler: fetch new articles from RSS/Atom feeds and web pages into the corpus.

  python data/crawl.py                               # crawl every enabled source once
  python data/crawl.py --learn runs/my-model         # ...then teach that model the new articles
  python data/crawl.py --list                        # show sources and their licenses
  python data/crawl.py --add https://example.org/feed.xml --license cc-by --name "Example"

The Studio runs this on a schedule (Training → News & web). Sources live in runs/_crawler/sources.json.

Licenses. The project only trains on public-domain or rights-cleared text, and most news is
copyrighted. Sources therefore carry a license, and a source marked "own-rights" or "unknown"
is skipped unless an admin ticked "I have the right to use this" (rights_confirmed). The presets
are permissive: Wikinews and Global Voices (CC BY) and NASA (US government, public domain).
CC BY requires attribution, so every article's URL, title, source and license is kept in
data/raw/news/attribution.jsonl.

Politeness. robots.txt is obeyed (including Crawl-delay), each host gets at most one request
every --delay seconds (default 3), and requests identify the crawler with a User-Agent.

Resuming. Every URL's state is kept in runs/_crawler/crawl.db: found URLs wait in a queue, so a
stopped or crashed run picks up where it left off. Downloads are written to a .part file and
continued with HTTP Range requests after a dropped connection, with retries and backoff.

Output: data/raw/news/<source>/<YYYY-MM>.txt, documents separated by <|endoftext|> like the rest
of data/raw/, so the tokenizer, train.py and finetune_lora.py pick them up. Articles not yet
taught to a model also go to runs/_crawler/pending.txt for --learn.
"""
import argparse, hashlib, json, os, re, sqlite3, subprocess, sys, time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.ingest import EOT, clean, from_docx, from_html, from_pdf  # noqa: E402

STATE = os.path.join(ROOT, "runs", "_crawler")
SOURCES_PATH = os.path.join(STATE, "sources.json")
DB_PATH = os.path.join(STATE, "crawl.db")
PENDING = os.path.join(STATE, "pending.txt")
PARTIAL = os.path.join(STATE, "partial")
OUT = os.path.join(ROOT, "data", "raw", "news")
ATTRIBUTION = os.path.join(OUT, "attribution.jsonl")
AGENT = "textllm-crawler"
UA = f"{AGENT}/1.0 (+https://github.com/hovchik/textllm; training a small writing model)"
MAX_BYTES = 15_000_000

LICENSES = {
    "public-domain": {"label": "Public domain", "ok": True,
                      "note": "e.g. works of the US federal government (NASA, NIH, NOAA)"},
    "cc0": {"label": "CC0", "ok": True, "note": "no conditions"},
    "cc-by": {"label": "CC BY", "ok": True, "note": "attribution is kept in attribution.jsonl"},
    "cc-by-sa": {"label": "CC BY-SA", "ok": True,
                 "note": "share-alike: if you publish the model, use a compatible license"},
    "own-rights": {"label": "I have the rights", "ok": False,
                   "note": "your own site, or written permission from the owner"},
    "unknown": {"label": "Copyrighted / unknown", "ok": False,
                "note": "most news sites. Only with permission; you take responsibility"},
}
# Not offered as licenses because they forbid it: CC BY-ND / CC BY-NC-ND (no derivatives),
# and sites whose terms forbid text and data mining.

PRESETS = [
    {"id": "wikinews-en", "name": "Wikinews (English)", "kind": "feed", "license": "cc-by",
     "url": "https://en.wikinews.org/w/index.php?title=Special:NewsFeed&feed=atom&categories=Published"
            "&notcategories=No%20publish%7CArchived%7CAutoArchived%7Cdisputed&namespace=0&count=30"
            "&ordermethod=categoryadd&stablepages=only",
     "home": "https://en.wikinews.org", "credit": "Wikinews contributors, CC BY 2.5"},
    {"id": "globalvoices", "name": "Global Voices", "kind": "feed", "license": "cc-by",
     "url": "https://globalvoices.org/feed/", "home": "https://globalvoices.org",
     "credit": "Global Voices, CC BY 3.0"},
    {"id": "nasa-news", "name": "NASA news releases", "kind": "feed", "license": "public-domain",
     "url": "https://www.nasa.gov/news-release/feed/", "home": "https://www.nasa.gov",
     "credit": "NASA (US government work)"},
]

TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|cmpid$|ocid$)", re.I)
NOT_ARTICLE = re.compile(r"\.(jpe?g|png|gif|webp|svg|mp[34]|mov|avi|zip|gz|exe|css|js|ico|woff2?)$"
                         r"|/(tag|tags|category|categories|author|authors|login|signin|signup|register|search"
                         r"|account|subscribe|privacy|terms|contact|about|feed|rss)(/|$)", re.I)


def say(m):
    print(m, flush=True)


def run_child(cmd, cwd):
    """subprocess.call, except that on Ctrl-C / Stop the child gets time to save and exit
    (subprocess.call would kill it at once, and a grandchild trainer would be left running)."""
    p = subprocess.Popen(cmd, cwd=cwd)
    try:
        return p.wait()
    except KeyboardInterrupt:
        # Ctrl-C and Studio's Stop already reach the whole process group, so the child has the
        # signal too: wait for it (ignoring repeated interrupts), and only kill it if it hangs.
        end = time.time() + 120
        while p.poll() is None and time.time() < end:
            try:
                p.wait(max(0.1, end - time.time()))
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                pass
        if p.poll() is None:
            p.kill()
            p.wait()
        raise


# ---------------------------------------------------------------- sources
def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "source"


def load_sources():
    if not os.path.exists(SOURCES_PATH):
        save_sources([{**p, "enabled": True, "max_items": 30, "created": time.time()} for p in PRESETS])
    return json.load(open(SOURCES_PATH, encoding="utf-8"))


def save_sources(srcs):
    os.makedirs(STATE, exist_ok=True)
    tmp = SOURCES_PATH + ".tmp"
    json.dump(srcs, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, SOURCES_PATH)


def check_source(s):
    """Validate and normalise a source dict (from the CLI or the Studio)."""
    url = (s.get("url") or "").strip()
    if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
        raise ValueError("The link must start with http:// or https://")
    if s.get("kind", "feed") not in ("feed", "page", "links"):
        raise ValueError("Kind must be feed, page or links.")
    if s.get("license") not in LICENSES:
        raise ValueError("Choose a license: " + ", ".join(LICENSES))
    if not LICENSES[s["license"]]["ok"] and not s.get("rights_confirmed"):
        raise ValueError("For this license, confirm that you have the right to use these texts.")
    name = (s.get("name") or urlsplit(url).netloc).strip()[:80]
    return {"id": s.get("id") or slug(name), "name": name, "url": url, "kind": s.get("kind", "feed"),
            "license": s["license"], "rights_confirmed": bool(s.get("rights_confirmed")),
            "enabled": bool(s.get("enabled", True)), "max_items": max(1, min(500, int(s.get("max_items") or 30))),
            "home": s.get("home") or f"{urlsplit(url).scheme}://{urlsplit(url).netloc}",
            "credit": (s.get("credit") or name).strip()[:200], "created": s.get("created") or time.time()}


def add_source(s):
    srcs = load_sources()
    s = check_source(s)
    ids = {x["id"] for x in srcs}
    base, i = s["id"], 2
    while s["id"] in ids:
        s["id"], i = f"{base}-{i}", i + 1
    if any(x["url"] == s["url"] for x in srcs):
        raise ValueError("That link is already a source.")
    save_sources(srcs + [s])
    return s


def update_source(sid, changes):
    srcs = load_sources()
    for i, s in enumerate(srcs):
        if s["id"] == sid:
            merged = {**s, **{k: v for k, v in changes.items() if k != "id"}}
            if "license" in changes and "rights_confirmed" not in changes and changes["license"] != s["license"]:
                merged["rights_confirmed"] = False  # a new license needs a fresh confirmation
            srcs[i] = check_source({**merged, "id": sid})
            save_sources(srcs)
            return srcs[i]
    raise LookupError("No such source.")


def delete_source(sid):
    srcs = load_sources()
    if not any(s["id"] == sid for s in srcs):
        raise LookupError("No such source.")
    save_sources([s for s in srcs if s["id"] != sid])


def allowed(s):
    return s.get("enabled") and (LICENSES.get(s.get("license"), {}).get("ok") or s.get("rights_confirmed"))


# ---------------------------------------------------------------- state
def db():
    os.makedirs(STATE, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS items (
        url TEXT PRIMARY KEY, source TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',   -- queued | done | skipped | failed
        title TEXT, published TEXT, feed_text TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, error TEXT,
        chars INTEGER, text_hash TEXT, found REAL NOT NULL, fetched REAL);
    CREATE INDEX IF NOT EXISTS idx_items_src ON items(source, status);
    CREATE TABLE IF NOT EXISTS hashes (hash TEXT PRIMARY KEY, url TEXT);
    CREATE TABLE IF NOT EXISTS feeds (source TEXT PRIMARY KEY, etag TEXT, modified TEXT, checked REAL);
    CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL, ended REAL,
        added INTEGER, failed INTEGER, learned TEXT);
    """)
    return c


def stats():
    """Numbers for the Studio page."""
    if not os.path.exists(DB_PATH):
        return {"sources": {}, "total": 0, "queued": 0, "pending": 0, "recent": [], "runs": []}
    with db() as c:
        per = {}
        for r in c.execute("SELECT source, status, COUNT(*) n, SUM(chars) ch, MAX(fetched) last "
                           "FROM items GROUP BY source, status"):
            d = per.setdefault(r["source"], {"done": 0, "queued": 0, "failed": 0, "skipped": 0, "chars": 0, "last": None})
            d[r["status"]] = r["n"]
            if r["status"] == "done":
                d["chars"], d["last"] = r["ch"] or 0, r["last"]
        recent = [dict(r) for r in c.execute("SELECT url, source, title, chars, fetched FROM items "
                                             "WHERE status='done' ORDER BY fetched DESC LIMIT 25")]
        runs = [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 10")]
    return {"sources": per, "total": sum(d["done"] for d in per.values()),
            "queued": sum(d["queued"] for d in per.values()), "pending": pending_count(),
            "recent": recent, "runs": runs}


def pending_count():
    if not os.path.exists(PENDING):
        return 0
    with open(PENDING, encoding="utf-8") as f:
        return f.read().count(EOT)


# ---------------------------------------------------------------- polite, resumable HTTP
def norm_url(u):
    p = urlsplit(u.strip())
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not TRACKING.match(k)]
    path = p.path or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, urlencode(q), ""))


class Fetcher:
    def __init__(self, delay=3.0, retries=4, timeout=30):
        import requests

        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en;q=1, *;q=0.5"})
        self.delay, self.retries, self.timeout = delay, retries, timeout
        self.robots, self.last, self.dead = {}, {}, set()
        os.makedirs(PARTIAL, exist_ok=True)

    def _host(self, url):
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}"

    def _wait(self, host):
        rp = self.robots.get(host)
        gap = max(self.delay, float((rp.crawl_delay(AGENT) if rp else None) or 0))
        sleep = self.last.get(host, 0) + gap - time.time()
        if sleep > 0:
            time.sleep(sleep)
        self.last[host] = time.time()

    def can_fetch(self, url):
        host = self._host(url)
        if host in self.dead:
            return False
        if host not in self.robots:
            rp = RobotFileParser()
            try:
                r = self._request(host + "/robots.txt", host)
                if r.status_code in (401, 403):
                    rp.disallow_all = True       # robots.txt that needs a login: stay out
                elif r.status_code >= 400:
                    rp.allow_all = True          # no robots.txt: everything is allowed
                else:
                    rp.parse(r.text.splitlines())
            except Exception as e:
                say(f"  ! {host} unreachable ({type(e).__name__}); skipping it this run")
                self.dead.add(host)
                return False
            self.robots[host] = rp
        return self.robots[host].can_fetch(AGENT, url)

    def _request(self, url, host, headers=None, stream=False):
        """GET with backoff: 2, 4, 8, 16 s on dropped connections, timeouts, 429 and 5xx."""
        import requests

        for attempt in range(self.retries + 1):
            self._wait(host)
            try:
                r = self.s.get(url, headers=headers or {}, timeout=self.timeout, stream=stream, allow_redirects=True)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    ra = r.headers.get("Retry-After", "")
                    time.sleep(min(300, int(ra)) if ra.isdigit() else 2 ** (attempt + 1))
                    continue
                return r
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt == self.retries:
                    raise
                time.sleep(2 ** (attempt + 1))

    def get(self, url, headers=None):
        """Download url, resuming a half-finished download from its .part file. Returns
        (status, content bytes, content type, final url, response headers)."""
        import requests

        host = self._host(url)
        part = os.path.join(PARTIAL, hashlib.sha1(url.encode()).hexdigest() + ".part")
        meta_p = part + ".json"
        for attempt in range(self.retries + 1):
            have = os.path.getsize(part) if os.path.exists(part) else 0
            # identity: a Range offset counts bytes on the wire, and requests would write
            # decompressed gzip bytes to the .part file, so resuming would start at the wrong place
            h = {"Accept-Encoding": "identity", **(headers or {})}
            meta = json.load(open(meta_p)) if have and os.path.exists(meta_p) else {}
            if have and meta.get("validator"):
                h.update({"Range": f"bytes={have}-", "If-Range": meta["validator"]})
            try:
                r = self._request(url, host, h, stream=True)
                if r.status_code == 304:
                    return 304, b"", "", r.url, r.headers
                if r.status_code >= 400:
                    r.close()
                    return r.status_code, b"", "", r.url, r.headers
                mode = "ab" if r.status_code == 206 else "wb"
                validator = r.headers.get("ETag") or r.headers.get("Last-Modified")
                if r.headers.get("Accept-Ranges", "").lower() == "bytes" and validator:
                    json.dump({"validator": validator}, open(meta_p, "w"))
                size = have if mode == "ab" else 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                        size += len(chunk)
                        if size > MAX_BYTES:
                            r.close()
                            raise ValueError(f"larger than {MAX_BYTES // 1_000_000} MB, skipped")
                data = open(part, "rb").read()
                for p in (part, meta_p):
                    if os.path.exists(p):
                        os.remove(p)
                return 200, data, r.headers.get("Content-Type", "").lower(), r.url, r.headers
            except ValueError:
                for p in (part, meta_p):
                    if os.path.exists(p):
                        os.remove(p)
                raise
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt == self.retries:
                    raise  # the .part file stays, so the next run continues the download
                say(f"  connection dropped, resuming from {os.path.getsize(part) if os.path.exists(part) else 0:,} bytes")
                time.sleep(2 ** (attempt + 1))


# ---------------------------------------------------------------- discovery
def _tag(e):
    return e.tag.rsplit("}", 1)[-1].lower()


def _child(e, *names):
    """First child with one of these tag names, tried in order (an Element with no children is falsy,
    so never chain these with `or`)."""
    for n in names:
        for c in e:
            if _tag(c) == n:
                return c
    return None


def _date(s):
    if not s:
        return None
    try:
        return parsedate_to_datetime(s).isoformat()
    except (TypeError, ValueError):
        return s.strip()[:40]


def parse_feed(data, base):
    """RSS 2.0, RSS 1.0 (RDF) or Atom → [{url, title, published, text}]."""
    root = ET.fromstring(data)
    out = []
    for e in root.iter():
        t = _tag(e)
        if t not in ("item", "entry"):
            continue
        link = None
        if t == "entry":
            for l in e:
                if _tag(l) == "link" and l.get("rel", "alternate") == "alternate" and l.get("href"):
                    link = l.get("href")
                    break
        else:
            le = _child(e, "link")
            link = (le.text or "").strip() if le is not None else None
            if not link:
                g = _child(e, "guid")
                if g is not None and (g.text or "").startswith("http") and g.get("isPermaLink", "true") != "false":
                    link = g.text.strip()
        if not link:
            continue
        title = _child(e, "title")
        date = _child(e, "pubdate", "published", "updated", "date")
        body = _child(e, "encoded", "content", "description", "summary")
        out.append({"url": urljoin(base, link), "title": (title.text or "").strip() if title is not None else "",
                    "published": _date(date.text) if date is not None else None,
                    "text": (body.text or "") if body is not None else ""})
    return out


def page_links(html, base, limit):
    """Article-looking links on the same site as base."""
    from bs4 import BeautifulSoup

    host = urlsplit(base).netloc.lower()
    seen, out = set(), []
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        u = norm_url(urljoin(base, a["href"]))
        p = urlsplit(u)
        if p.scheme not in ("http", "https") or p.netloc.lower() != host or u in seen or u == norm_url(base):
            continue
        if NOT_ARTICLE.search(p.path) or len(p.path) < 12 or not re.search(r"[-_]|\d{4}", p.path):
            continue
        seen.add(u)
        out.append({"url": u, "title": a.get_text(" ", strip=True)[:200], "published": None, "text": ""})
        if len(out) >= limit:
            break
    return out


def discover(f, c, s):
    """Queue new URLs from one source. Returns how many were new."""
    found = []
    if s["kind"] == "page":
        found = [{"url": s["url"], "title": s["name"], "published": None, "text": ""}]
    elif f.can_fetch(s["url"]):
        h = {}
        row = c.execute("SELECT * FROM feeds WHERE source=?", (s["id"],)).fetchone() if s["kind"] == "feed" else None
        if row and row["etag"]:
            h["If-None-Match"] = row["etag"]
        if row and row["modified"]:
            h["If-Modified-Since"] = row["modified"]
        code, data, ctype, final, headers = f.get(s["url"], h)
        if code == 304:
            say("  feed unchanged since last check")
        elif code >= 400:
            say(f"  ! {s['url']}: HTTP {code}")
        elif s["kind"] == "feed":
            try:
                found = parse_feed(data, final)
            except ET.ParseError as e:
                say(f"  ! not a valid RSS/Atom feed ({e}). Use kind 'links' for a normal web page.")
            c.execute("INSERT INTO feeds(source, etag, modified, checked) VALUES(?,?,?,?) ON CONFLICT(source) DO "
                      "UPDATE SET etag=excluded.etag, modified=excluded.modified, checked=excluded.checked",
                      (s["id"], headers.get("ETag"), headers.get("Last-Modified"), time.time()))
        else:
            found = page_links(data.decode("utf-8", "replace"), final, s["max_items"] * 3)
    else:
        say(f"  robots.txt does not allow {s['url']}")
    new = 0
    for it in found:
        cur = c.execute("INSERT OR IGNORE INTO items(url, source, title, published, feed_text, found) "
                        "VALUES(?,?,?,?,?,?)", (norm_url(it["url"]), s["id"], it["title"][:300], it["published"],
                                                it["text"][:400_000] or None, time.time()))
        new += cur.rowcount
    c.commit()
    return new


# ---------------------------------------------------------------- extraction
def extract(data, ctype, url):
    name = urlsplit(url).path.lower()
    if "pdf" in ctype or name.endswith(".pdf"):
        p = os.path.join(PARTIAL, "x.pdf")
        open(p, "wb").write(data)
        try:
            return from_pdf(p)
        finally:
            os.remove(p)
    if "wordprocessingml" in ctype or name.endswith(".docx"):
        p = os.path.join(PARTIAL, "x.docx")
        open(p, "wb").write(data)
        try:
            return from_docx(p)
        finally:
            os.remove(p)
    text = data.decode("utf-8", "replace")
    if "html" in ctype or "xml" in ctype or "<html" in text[:3000].lower():
        return from_html(text, url)
    return text


def text_hash(text):
    t = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(t.encode()).hexdigest()


def _append(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(doc)
        f.flush()
        os.fsync(f.fileno())


def fetch_items(f, c, s, min_chars, limit):
    added = failed = 0
    rows = c.execute("SELECT * FROM items WHERE source=? AND status IN ('queued','failed') AND attempts < 3 "
                     "ORDER BY status DESC, found LIMIT ?", (s["id"], limit)).fetchall()
    for r in rows:
        url = r["url"]
        try:
            text = clean(from_html(r["feed_text"], url) or "") if r["feed_text"] else ""
            if len(text) >= max(min_chars, 1500):
                pass  # the feed already has the full article
            elif not f.can_fetch(url):
                c.execute("UPDATE items SET status='skipped', error='robots.txt' WHERE url=?", (url,))
                c.commit()
                continue
            else:
                code, data, ctype, final, _ = f.get(url)
                if code >= 400:
                    raise ValueError(f"HTTP {code}")
                text = extract(data, ctype, final)
            text = clean(text or "")
            if len(text) < min_chars:
                c.execute("UPDATE items SET status='skipped', error=?, fetched=? WHERE url=?",
                          (f"only {len(text)} characters", time.time(), url))
                c.commit()
                continue
            h = text_hash(text)
            if c.execute("SELECT 1 FROM hashes WHERE hash=?", (h,)).fetchone():
                c.execute("UPDATE items SET status='skipped', error='duplicate', fetched=? WHERE url=?",
                          (time.time(), url))
                c.commit()
                continue
            title = r["title"] or ""
            doc = (f"{title}\n\n{text}" if title and not text.startswith(title) else text) + f"\n{EOT}\n"
            _append(os.path.join(OUT, s["id"], time.strftime("%Y-%m") + ".txt"), doc)
            _append(PENDING, doc)
            _append(ATTRIBUTION, json.dumps({"url": url, "title": title, "source": s["name"], "license": s["license"],
                                             "credit": s.get("credit"), "published": r["published"],
                                             "fetched": time.strftime("%Y-%m-%d")}, ensure_ascii=False) + "\n")
            c.execute("INSERT OR IGNORE INTO hashes(hash, url) VALUES(?,?)", (h, url))
            c.execute("UPDATE items SET status='done', chars=?, text_hash=?, fetched=?, error=NULL WHERE url=?",
                      (len(text), h, time.time(), url))
            c.commit()
            added += 1
            say(f"  ✓ {title[:70] or url} ({len(text):,} chars)")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed += 1
            c.execute("UPDATE items SET status='failed', attempts=attempts+1, error=?, fetched=? WHERE url=?",
                      (f"{type(e).__name__}: {e}"[:300], time.time(), url))
            c.commit()
            say(f"  ✗ {url}: {type(e).__name__}: {str(e)[:120]}")
    return added, failed


# ---------------------------------------------------------------- learning
def learn(run, min_docs, passes):
    n = pending_count()
    if n < min_docs:
        say(f"{n} new article(s) waiting to be learned; training starts at {min_docs}.")
        return None
    say(f"== teaching {os.path.basename(run)} {n} new article(s)")
    batch = PENDING + f".{int(time.time())}.learning"
    os.replace(PENDING, batch)  # new articles found while training go to a fresh pending.txt
    try:
        rc = run_child([sys.executable, "learn.py", "--run", run, "--text-file", batch, "--passes", str(passes)], ROOT)
    except KeyboardInterrupt:
        rc = 130
    if rc != 0:
        # put the batch back in front of anything new, so nothing is lost
        rest = open(PENDING, encoding="utf-8").read() if os.path.exists(PENDING) else ""
        with open(PENDING, "w", encoding="utf-8") as f:
            f.write(open(batch, encoding="utf-8").read() + rest)
        os.remove(batch)
        raise SystemExit(f"training {'stopped' if rc == 130 else f'failed (exit {rc})'}; "
                         "the articles stay queued for the next run")
    os.remove(batch)
    return os.path.basename(run)


def recover_batches():
    """A previous learn step that was killed leaves *.learning files: queue them again."""
    if not os.path.isdir(STATE):
        return
    for fn in sorted(os.listdir(STATE)):
        if fn.startswith("pending.txt.") and fn.endswith(".learning"):
            p = os.path.join(STATE, fn)
            rest = open(PENDING, encoding="utf-8").read() if os.path.exists(PENDING) else ""
            with open(PENDING, "w", encoding="utf-8") as f:
                f.write(open(p, encoding="utf-8").read() + rest)
            os.remove(p)


def crawl(only=None, delay=3.0, min_chars=600, learn_run=None, min_docs=20, passes=2):
    recover_batches()
    srcs = [s for s in load_sources() if allowed(s) and (not only or s["id"] in only)]
    skipped = [s["name"] for s in load_sources() if s.get("enabled") and not allowed(s)]
    if skipped:
        say("Skipped (license not confirmed): " + ", ".join(skipped))
    if not srcs:
        say("No enabled sources. Add or enable one under Training → News & web.")
    f, c = Fetcher(delay=delay), db()
    run_id = c.execute("INSERT INTO runs(started, added, failed) VALUES(?,0,0)", (time.time(),)).lastrowid
    c.commit()
    added = failed = 0
    try:
        for s in srcs:
            say(f"== {s['name']} ({LICENSES[s['license']]['label']}) {s['url']}")
            try:
                say(f"  {discover(f, c, s)} new link(s)")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                say(f"  ! could not read the source: {type(e).__name__}: {str(e)[:150]}")
            a, b = fetch_items(f, c, s, min_chars, s.get("max_items", 30))
            added, failed = added + a, failed + b
            c.execute("UPDATE runs SET added=?, failed=? WHERE id=?", (added, failed, run_id))
            c.commit()
    except KeyboardInterrupt:
        say("stopped. Found links stay queued and continue next run.")
        c.execute("UPDATE runs SET ended=? WHERE id=?", (time.time(), run_id))
        c.commit()
        raise SystemExit(130)
    queued = c.execute("SELECT COUNT(*) FROM items WHERE status='queued'").fetchone()[0]
    say(f"added {added} article(s), {failed} failed (retried next run), {queued} still queued")
    learned = None
    try:
        if learn_run:
            learned = learn(learn_run, min_docs, passes)
    finally:
        c.execute("UPDATE runs SET ended=?, learned=? WHERE id=?", (time.time(), learned, run_id))
        c.commit()
        c.close()
    say("✅ crawl finished")
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="*", help="only these source ids")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between requests to one site")
    ap.add_argument("--min-chars", type=int, default=600, help="skip pages with less text than this")
    ap.add_argument("--learn", help="runs/<name>: teach this model the new articles afterwards")
    ap.add_argument("--min-docs", type=int, default=20, help="only train once this many new articles are waiting")
    ap.add_argument("--passes", type=float, default=2)
    ap.add_argument("--list", action="store_true", help="show the sources and exit")
    ap.add_argument("--add", metavar="URL", help="add a source")
    ap.add_argument("--name")
    ap.add_argument("--kind", default="feed", choices=["feed", "page", "links"])
    ap.add_argument("--license", choices=list(LICENSES))
    ap.add_argument("--i-have-rights", action="store_true", help="confirm rights for own-rights/unknown sources")
    a = ap.parse_args()
    if a.add:
        s = add_source({"url": a.add, "name": a.name, "kind": a.kind, "license": a.license,
                        "rights_confirmed": a.i_have_rights})
        say(f"added {s['id']}")
        return
    if a.list:
        for s in load_sources():
            say(f"{'on ' if allowed(s) else 'off'}  {s['id']:<20} {s['kind']:<5} {LICENSES[s['license']]['label']:<22} {s['url']}")
        return
    crawl(a.source, a.delay, a.min_chars, os.path.abspath(a.learn) if a.learn else None, a.min_docs, a.passes)


if __name__ == "__main__":
    main()
