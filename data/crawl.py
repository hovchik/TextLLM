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

Learning is checked. Before --learn trains the model its weights are copied aside; afterwards a
quick evaluation (evaluate.py --quick) runs and, if the score fell by more than --max-drop points,
the old weights are put back. --no-gate skips this.

Social networks: sources of kind "mastodon" or "bluesky" read public posts through those networks'
public APIs (see data/social.py for links, rights and personal-data handling). Each one is off until
an admin adds it and confirms the server's terms allow it.

Languages: --languages en,hy keeps only articles in those languages (data/lang.py guesses).

Without the Studio: --install-schedule 6 registers the crawl with Windows Task Scheduler (or
cron) every 6 hours; --uninstall-schedule removes it. Only one crawl runs at a time, and the
learn step waits while the Studio is running another job.
"""
import argparse, hashlib, json, os, re, sqlite3, subprocess, sys, time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import graceful  # noqa: E402
from data.ingest import EOT, clean, from_docx, from_html, from_pdf  # noqa: E402
from data import lang as langmod  # noqa: E402
from data import social  # noqa: E402
from evaluate import counts  # noqa: E402
from graceful import run_child  # noqa: E402

STATE = os.path.join(ROOT, "runs", "_crawler")
SOURCES_PATH = os.path.join(STATE, "sources.json")
DB_PATH = os.path.join(STATE, "crawl.db")
PENDING = os.path.join(STATE, "pending.txt")
PARTIAL = os.path.join(STATE, "partial")
LOCK = os.path.join(STATE, "lock")                              # one crawl at a time
JOB_LOCK = os.path.join(ROOT, "runs", "_job.lock")              # written by the Studio while a job runs
TASK_NAME = "TextLLM news crawler"                              # Windows Task Scheduler / crontab marker
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
    "cc-stated": {"label": "Only openly licensed authors", "ok": True, "social": True,
                  "note": "social sources: keeps a post only if its author's profile states CC0, CC BY or CC BY-SA"},
}
SOCIAL_LICENSES = ("cc-stated", "own-rights", "unknown")   # a hashtag or timeline has no license of its own
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
    kind = s.get("kind", "feed")
    if kind not in ("feed", "page", "links") + social.KINDS:
        raise ValueError("Kind must be feed, page, links, mastodon or bluesky.")
    why = social.not_supported(url)
    if why:
        raise ValueError(why)
    if s.get("license") not in LICENSES:
        raise ValueError("Choose a license: " + ", ".join(LICENSES))
    from core import net

    net.check_url(url, dns=False)  # no local addresses (core/net.py); the crawl checks the resolved address too
    if kind in social.KINDS:
        social.parse_target(kind, url)   # raises a helpful ValueError for a link it can't read
        if s["license"] not in SOCIAL_LICENSES:
            raise ValueError("A social source mixes many authors, so choose 'Only openly licensed authors', "
                             "'I have the rights' or 'Copyrighted / unknown'.")
        if not s.get("terms_confirmed"):
            raise ValueError("Confirm that this server's terms allow downloading public posts for AI training.")
    elif LICENSES[s["license"]].get("social"):
        raise ValueError("That license is only for Mastodon and Bluesky sources.")
    if not LICENSES[s["license"]]["ok"] and not s.get("rights_confirmed"):
        raise ValueError("For this license, confirm that you have the right to use these texts.")
    name = (s.get("name") or urlsplit(url).netloc).strip()[:80]
    out = {"id": s.get("id") or slug(name), "name": name, "url": url, "kind": kind,
           "license": s["license"], "rights_confirmed": bool(s.get("rights_confirmed")),
           "enabled": bool(s.get("enabled", True)), "max_items": max(1, min(500, int(s.get("max_items") or 30))),
           "home": s.get("home") or f"{urlsplit(url).scheme}://{urlsplit(url).netloc}",
           "credit": (s.get("credit") or name).strip()[:200], "created": s.get("created") or time.time()}
    if kind in social.KINDS:
        out["terms_confirmed"] = True
        out["min_chars"] = max(20, min(5000, int(s.get("min_chars") or 120)))
    return out


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
            if "url" in changes and "terms_confirmed" not in changes and changes["url"] != s["url"]:
                merged["terms_confirmed"] = False   # another server needs its terms checked again
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
    if s.get("kind") in social.KINDS and not s.get("terms_confirmed"):
        return False
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
    for table, col, typ in (("items", "lang", "TEXT"), ("items", "excerpt", "TEXT"),
                            ("runs", "score_before", "REAL"), ("runs", "score_after", "REAL"), ("runs", "verdict", "TEXT")):
        if col not in [r["name"] for r in c.execute(f"PRAGMA table_info({table})")]:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
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
        recent = [dict(r) for r in c.execute("SELECT url, source, title, chars, lang, fetched FROM items "
                                             "WHERE status='done' ORDER BY fetched DESC LIMIT 25")]
        runs = [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 10")]
        langs = {r["lang"] or "?": r["n"] for r in c.execute("SELECT lang, COUNT(*) n FROM items WHERE status='done' "
                                                            "GROUP BY lang ORDER BY n DESC")}
        skipped_lang = c.execute("SELECT COUNT(*) FROM items WHERE status='skipped' AND error LIKE 'language %'").fetchone()[0]
    return {"sources": per, "total": sum(d["done"] for d in per.values()),
            "queued": sum(d["queued"] for d in per.values()), "pending": pending_count(),
            "recent": recent, "runs": runs, "languages": langs, "skipped_language": skipped_lang,
            "lock": crawl_lock_info(), "schedule": schedule_status()}


def recent_articles(query="", n=5, days=7):
    """Recent articles for the "news" chat format, best keyword matches first."""
    if not os.path.exists(DB_PATH):
        return []
    words = {w for w in re.findall(r"[^\W\d_]{3,}", (query or "").lower())}
    with db() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT url, source, title, excerpt, lang, published, fetched FROM items WHERE status='done' "
            "AND excerpt IS NOT NULL AND fetched > ? ORDER BY fetched DESC LIMIT 400", (time.time() - days * 86400,))]
    names = {s["id"]: s["name"] for s in load_sources()}
    now = time.time()
    for r in rows:
        text = f"{r['title']} {r['excerpt']}".lower()
        hits = sum(text.count(w) for w in words)
        r["score"] = hits * 10 + (1 - min(1, (now - r["fetched"]) / (days * 86400)))
        r["source_name"] = names.get(r["source"], r["source"])
    rows.sort(key=lambda r: -r["score"])
    return rows[:n]


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

        from core import net

        for attempt in range(self.retries + 1):
            self._wait(host)
            try:
                # address check on the link, every redirect and the connection itself (core/net.py)
                r = net.get(url, headers=headers, timeout=self.timeout, stream=True, session=self.s)
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


def save_text(c, s, url, title, text, published, languages=(), lang=None, extra=None):
    """Write one cleaned document to the corpus, pending.txt and attribution.jsonl, and mark its item
    done. Returns its language, or False when it was skipped (other language, duplicate)."""
    lang = lang or langmod.detect(text)
    if languages and lang not in languages:
        c.execute("UPDATE items SET status='skipped', error=?, lang=?, fetched=? WHERE url=?",
                  (f"language {lang or 'unknown'}", lang, time.time(), url))
        c.commit()
        return False
    h = text_hash(text)
    if c.execute("SELECT 1 FROM hashes WHERE hash=?", (h,)).fetchone():
        c.execute("UPDATE items SET status='skipped', error='duplicate', fetched=? WHERE url=?", (time.time(), url))
        c.commit()
        return False
    doc = (f"{title}\n\n{text}" if title and not text.startswith(title) else text) + f"\n{EOT}\n"
    _append(os.path.join(OUT, s["id"], time.strftime("%Y-%m") + ".txt"), doc)
    _append(PENDING, doc)
    _append(ATTRIBUTION, json.dumps({"url": url, "title": title, "source": s["name"], "license": s["license"],
                                     "credit": s.get("credit"), "published": published,
                                     "fetched": time.strftime("%Y-%m-%d"), **(extra or {})}, ensure_ascii=False) + "\n")
    c.execute("INSERT OR IGNORE INTO hashes(hash, url) VALUES(?,?)", (h, url))
    c.execute("UPDATE items SET status='done', chars=?, text_hash=?, fetched=?, error=NULL, lang=?, excerpt=? "
              "WHERE url=?", (len(text), h, time.time(), lang, text[:1500], url))
    c.commit()
    return lang


def fetch_social(f, c, s, languages=()):
    """Read a Mastodon or Bluesky source: posts come with their text, so there is no second download.
    Returns (added, skipped)."""
    added = skipped = failed = 0
    for p in social.posts(f, s):
        url = p["url"]
        old = c.execute("SELECT status, attempts FROM items WHERE url=?", (url,)).fetchone()
        if old and (old["status"] in ("done", "skipped") or old["attempts"] >= 3):
            continue   # decided in an earlier run; queued/failed ones (a save that broke) are retried
        if not old:
            c.execute("INSERT INTO items(url, source, title, published, found) VALUES(?,?,?,?,?)",
                      (url, s["id"], "", p["published"], time.time()))
        reason = p["skip"] or (f"only {len(p['text'])} characters" if len(p["text"]) < s.get("min_chars", 120) else None)
        if reason:
            c.execute("UPDATE items SET status='skipped', error=?, fetched=? WHERE url=?", (reason, time.time(), url))
            c.commit()
            skipped += 1
            continue
        lang = p["lang"] if p["lang"] and len(p["text"]) < 400 else None   # short posts: trust the author's tag
        extra = {"author": p["author"], "license": p["license"]} if p["license"] != s["license"] else {"author": p["author"]}
        try:
            saved = save_text(c, s, url, "", p["text"], p["published"], languages, lang, extra)
        except OSError as e:   # e.g. disk full: keep the post retryable instead of losing it
            c.execute("UPDATE items SET status='failed', attempts=attempts+1, error=?, fetched=? WHERE url=?",
                      (f"{type(e).__name__}: {e}"[:300], time.time(), url))
            c.commit()
            failed += 1
            say(f"  ✗ {url}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if saved is False:
            skipped += 1
            continue
        added += 1
        say(f"  ✓ {p['text'][:70].replace(chr(10), ' ')} ({len(p['text']):,} chars, {p['license']})")
    if skipped:
        say(f"  {skipped} post(s) skipped (replies, boosts, opt-outs, no open license, too short, duplicates)")
    return added, failed


def fetch_items(f, c, s, min_chars, limit, languages=()):
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
            title = r["title"] or ""
            lang = save_text(c, s, url, title, text, r["published"], languages)
            if lang is False:
                continue
            added += 1
            say(f"  ✓ {title[:70] or url} ({len(text):,} chars, {langmod.name(lang)})")
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
WEIGHT_FILES = ("ckpt.pt", "adapter", "train_state.pt", "examples.pt", "meta.json")   # what a learn step changes


def _requeue(batch):
    """Put a learning batch back in front of anything new, so nothing is lost."""
    rest = open(PENDING, encoding="utf-8").read() if os.path.exists(PENDING) else ""
    with open(PENDING, "w", encoding="utf-8") as f:
        f.write(open(batch, encoding="utf-8").read() + rest)
    os.remove(batch)


def backup_weights(run):
    """Copy the model's weights aside, so a learn step that makes it worse can be undone."""
    import shutil

    bk = os.path.join(run, "_before_learn")
    shutil.rmtree(bk, ignore_errors=True)
    os.makedirs(bk)
    for name in WEIGHT_FILES:
        p = os.path.join(run, name)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(bk, name))
        elif os.path.exists(p):
            shutil.copy2(p, os.path.join(bk, name))
    return bk


def restore_weights(run, bk):
    import shutil

    for name in WEIGHT_FILES:
        src, dst = os.path.join(bk, name), os.path.join(run, name)
        if not os.path.exists(src):
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)
    for stale in ("gguf",):  # a quantized copy of the undone weights would be out of date
        shutil.rmtree(os.path.join(run, stale), ignore_errors=True)
    shutil.rmtree(bk, ignore_errors=True)


def quick_score(run, since=0):
    """Latest complete quick evaluation score of the run, made after `since`."""
    h = os.path.join(run, "eval", "history.jsonl")
    if not os.path.exists(h):
        return None
    rows = [r for r in (json.loads(l) for l in open(h) if l.strip())
            if r.get("quick") and counts(r) and r["time"] >= since]
    return rows[-1]["score"] if rows else None


def reject_evals(run, since):
    """Mark evaluations made after `since` as rejected: they scored weights that were undone."""
    h = os.path.join(run, "eval", "history.jsonl")
    if not os.path.exists(h):
        return
    rows = [json.loads(l) for l in open(h) if l.strip()]
    for r in rows:
        if r["time"] >= since:
            r["rejected"] = True
    tmp = h + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(tmp, h)


def evaluate_quick(run):
    say(f"== quick check of {os.path.basename(run)} (8 prompts)")
    t = time.time()
    rc = run_child([sys.executable, "evaluate.py", "--run", run, "--quick"], ROOT)
    if rc != 0:
        say(f"  ! evaluation failed (exit {rc})")
        return None
    return quick_score(run, since=t - 1)


def _weights_time(run):
    ts = [os.path.getmtime(os.path.join(run, n)) for n in WEIGHT_FILES if os.path.exists(os.path.join(run, n))]
    return max(ts) if ts else 0


def learn(run, min_docs, passes, gate=True, max_drop=5.0, c=None, run_id=None):
    """Teach the model the waiting articles. With gate=True a quick evaluation runs after
    learning, and if the score fell by more than max_drop points the old weights are restored."""
    n = pending_count()
    if n < min_docs:
        say(f"{n} new article(s) waiting to be learned; training starts at {min_docs}.")
        return None
    if studio_busy():
        say("The Studio is running another job; the articles wait for the next run.")
        return None
    stale = os.path.join(run, "_before_learn")
    if os.path.isdir(stale):  # a previous learn step was killed before it was decided: undo it
        say("a previous learn step never finished (power cut or kill): restoring the weights from before it")
        reject_evals(run, os.path.getmtime(stale))
        restore_weights(run, stale)
    before = quick_score(run, since=_weights_time(run)) if gate else None
    if gate and before is None:
        say("No quick check of the current weights yet, so one runs first (the baseline).")
        before = evaluate_quick(run)
    say(f"== teaching {os.path.basename(run)} {n} new article(s)")
    with job_lock():
        verdict, after = _learn_checked(run, passes, gate, max_drop, before)
    if c is not None and run_id is not None:
        c.execute("UPDATE runs SET score_before=?, score_after=?, verdict=? WHERE id=?", (before, after, verdict, run_id))
        c.commit()
    return os.path.basename(run) + ("" if verdict in ("learned", "kept") else f" ({verdict})")


def _learn_checked(run, passes, gate, max_drop, before):
    """Train, then (with the gate) evaluate and maybe undo. The article batch and the weight backup
    stay until the whole step is decided, so a Stop or crash anywhere in it restores the old weights
    and requeues the articles."""
    import shutil

    started = time.time()
    bk = backup_weights(run) if gate else None
    batch = PENDING + f".{int(started)}.learning"
    os.replace(PENDING, batch)  # new articles found while training go to a fresh pending.txt

    def undo(reason):
        _requeue(batch)
        if bk and os.path.isdir(bk):
            say("restoring the weights from before this learn step")
            restore_weights(run, bk)
            reject_evals(run, started)
        raise SystemExit(f"{reason}; the articles stay queued for the next run")

    try:
        rc = run_child([sys.executable, "learn.py", "--run", run, "--text-file", batch, "--passes", str(passes)], ROOT)
    except KeyboardInterrupt:
        rc = 130
    if rc != 0:
        undo("training stopped" if rc == 130 else f"training failed (exit {rc})")
    if not gate:
        os.remove(batch)
        return "learned", None
    try:
        after = evaluate_quick(run)
    except (KeyboardInterrupt, Exception) as e:
        undo("stopped during the quick check" if isinstance(e, KeyboardInterrupt) else f"quick check failed ({e})")
    verdict = "kept"
    if before is not None and after is not None:
        say(f"== quick-check score {before} → {after}")
        if after < before - max_drop:
            say(f"  the model got worse by more than {max_drop} points: undoing this learn step "
                "(the articles stay in the corpus but are not retried)")
            restore_weights(run, bk)
            reject_evals(run, started)  # the score of the undone weights must not become the next baseline
            verdict = "rolled back"
    else:
        say("  no comparable scores, keeping the new weights")
    shutil.rmtree(bk, ignore_errors=True)
    os.remove(batch)
    return verdict, after


# ---------------------------------------------------------------- locks and the OS schedule
def _alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_lock(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def crawl_lock_info():
    info = _read_lock(LOCK)
    return info if info and _alive(info.get("pid")) else None


def take_lock():
    os.makedirs(STATE, exist_ok=True)
    other = crawl_lock_info()
    if other and other["pid"] != os.getpid():
        raise SystemExit(f"another crawl is already running (pid {other['pid']}, started "
                         f"{time.strftime('%H:%M', time.localtime(other['started']))}); nothing to do")
    json.dump({"pid": os.getpid(), "started": time.time()}, open(LOCK, "w"))


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def studio_busy():
    """True when the Studio is running a job other than this crawl (so training must wait)."""
    info = _read_lock(JOB_LOCK)
    if not info or not _alive(info.get("pid")):
        return False
    return info["pid"] not in (os.getpid(), os.getppid())


class job_lock:
    """Hold runs/_job.lock while this crawl trains, so the Studio refuses to start a second job on
    the same model. When the Studio itself started this crawl, the lock is already ours."""

    def __enter__(self):
        info = _read_lock(JOB_LOCK)
        self.mine = not (info and _alive(info.get("pid")) and info["pid"] in (os.getpid(), os.getppid()))
        if self.mine:
            os.makedirs(os.path.dirname(JOB_LOCK), exist_ok=True)
            json.dump({"pid": os.getpid(), "kind": "crawl", "name": "news & web (scheduled)", "started": time.time()},
                      open(JOB_LOCK, "w"))
        return self

    def __exit__(self, *exc):
        if self.mine:
            try:
                os.remove(JOB_LOCK)
            except OSError:
                pass


def crawl_command(learn=None, min_docs=20, passes=2, languages="", gate=True):
    cmd = [sys.executable, os.path.join(ROOT, "data", "crawl.py")]
    if learn:
        cmd += ["--learn", learn, "--min-docs", str(min_docs), "--passes", str(passes)]
        if not gate:
            cmd.append("--no-gate")
    if languages:
        cmd += ["--languages", languages]
    return cmd


def schedule_status():
    """Is the crawl registered with the OS scheduler (runs without the Studio)?"""
    try:
        if os.name == "nt":
            r = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"], capture_output=True, text=True)
            return {"installed": r.returncode == 0, "kind": "Task Scheduler", "detail": r.stdout.strip()[-400:] if r.returncode == 0 else ""}
        r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if TASK_NAME in l), "")
        return {"installed": bool(line), "kind": "cron", "detail": line}
    except FileNotFoundError:
        return {"installed": False, "kind": "none", "detail": "no scheduler found"}


def install_schedule(every_hours=6, learn=None, min_docs=20, passes=2, languages="", gate=True):
    """Register the crawl with Windows Task Scheduler or cron, so it runs when the Studio is closed."""
    log = os.path.join(STATE, "log.txt")
    os.makedirs(STATE, exist_ok=True)
    cmd = crawl_command(learn, min_docs, passes, languages, gate)
    q = lambda s: f'"{s}"' if " " in s or os.name == "nt" else s
    hours = max(1, int(round(every_hours)))
    if os.name == "nt":
        tr = "cmd /c " + " ".join(q(c) for c in cmd) + f" >> {q(log)} 2>&1"
        args = ["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/SC", "HOURLY", "/MO", str(hours), "/TR", tr]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip() or "schtasks failed")
        return f"Task Scheduler runs the crawl every {hours} h, also when the Studio is closed."
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = [l for l in (r.stdout.splitlines() if r.returncode == 0 else []) if TASK_NAME not in l]
    sched = f"0 */{hours} * * *" if hours < 24 else "0 3 * * *"
    lines.append(f"{sched} cd {q(ROOT)} && " + " ".join(q(c) for c in cmd) + f" >> {q(log)} 2>&1  # {TASK_NAME}")
    r = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or "crontab failed")
    return f"cron runs the crawl every {hours} h, also when the Studio is closed."


def uninstall_schedule():
    if os.name == "nt":
        r = subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK_NAME], capture_output=True, text=True)
        if r.returncode != 0 and "cannot find" not in (r.stderr + r.stdout).lower():
            raise RuntimeError((r.stderr or r.stdout).strip())
        return "Removed from Task Scheduler."
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = [l for l in (r.stdout.splitlines() if r.returncode == 0 else []) if TASK_NAME not in l]
    subprocess.run(["crontab", "-"], input="\n".join(lines) + ("\n" if lines else ""), capture_output=True, text=True)
    return "Removed from cron."


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


def crawl(only=None, delay=3.0, min_chars=600, learn_run=None, min_docs=20, passes=2, languages=(),
          gate=True, max_drop=5.0):
    take_lock()
    try:
        return _crawl(only, delay, min_chars, learn_run, min_docs, passes, tuple(languages), gate, max_drop)
    finally:
        release_lock()


def _crawl(only, delay, min_chars, learn_run, min_docs, passes, languages, gate, max_drop):
    recover_batches()
    if languages:
        say("keeping only: " + ", ".join(langmod.name(l) for l in languages))
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
            if s["kind"] in social.KINDS:
                try:
                    a, b = fetch_social(f, c, s, languages)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    a, b = 0, 1
                    say(f"  ! could not read the source: {type(e).__name__}: {str(e)[:200]}")
                added, failed = added + a, failed + b
                c.execute("UPDATE runs SET added=?, failed=? WHERE id=?", (added, failed, run_id))
                c.commit()
                continue
            try:
                say(f"  {discover(f, c, s)} new link(s)")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                say(f"  ! could not read the source: {type(e).__name__}: {str(e)[:150]}")
            a, b = fetch_items(f, c, s, min_chars, s.get("max_items", 30), languages)
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
            learned = learn(learn_run, min_docs, passes, gate, max_drop, c, run_id)
    finally:
        c.execute("UPDATE runs SET ended=?, learned=? WHERE id=?", (time.time(), learned, run_id))
        c.commit()
        c.close()
    say("✅ crawl finished")
    return added


def main():
    graceful.install()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="*", help="only these source ids")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between requests to one site")
    ap.add_argument("--min-chars", type=int, default=600, help="skip pages with less text than this")
    ap.add_argument("--learn", help="runs/<name>: teach this model the new articles afterwards")
    ap.add_argument("--min-docs", type=int, default=20, help="only train once this many new articles are waiting")
    ap.add_argument("--passes", type=float, default=2)
    ap.add_argument("--languages", default="", help="keep only these languages, e.g. en,hy (default: all)")
    ap.add_argument("--no-gate", action="store_true", help="don't run a quick check after learning / never undo")
    ap.add_argument("--max-drop", type=float, default=5.0, help="undo the learn step if the score falls by more")
    ap.add_argument("--install-schedule", type=float, metavar="HOURS",
                    help="register with Task Scheduler (Windows) or cron so the crawl runs when the Studio is closed")
    ap.add_argument("--uninstall-schedule", action="store_true")
    ap.add_argument("--list", action="store_true", help="show the sources and exit")
    ap.add_argument("--add", metavar="URL", help="add a source")
    ap.add_argument("--name")
    ap.add_argument("--kind", default="feed", choices=["feed", "page", "links", "mastodon", "bluesky"])
    ap.add_argument("--license", choices=list(LICENSES))
    ap.add_argument("--i-have-rights", action="store_true", help="confirm rights for own-rights/unknown sources")
    ap.add_argument("--terms-ok", action="store_true",
                    help="social sources: confirm the server's terms allow downloading public posts for AI training")
    ap.add_argument("--min-post-chars", type=int, default=120, help="social sources: skip shorter posts")
    a = ap.parse_args()
    if a.add:
        s = add_source({"url": a.add, "name": a.name, "kind": a.kind, "license": a.license,
                        "rights_confirmed": a.i_have_rights, "terms_confirmed": a.terms_ok,
                        "min_chars": a.min_post_chars})
        say(f"added {s['id']}")
        return
    if a.install_schedule:
        say(install_schedule(a.install_schedule, os.path.abspath(a.learn) if a.learn else None, a.min_docs, a.passes,
                             a.languages, not a.no_gate))
        return
    if a.uninstall_schedule:
        say(uninstall_schedule())
        return
    if a.list:
        for s in load_sources():
            say(f"{'on ' if allowed(s) else 'off'}  {s['id']:<20} {s['kind']:<8} {LICENSES[s['license']]['label']:<22} {s['url']}")
        return
    langs = [l.strip().lower() for l in a.languages.split(",") if l.strip()]
    crawl(a.source, a.delay, a.min_chars, os.path.abspath(a.learn) if a.learn else None, a.min_docs, a.passes,
          langs, not a.no_gate, a.max_drop)


if __name__ == "__main__":
    main()
