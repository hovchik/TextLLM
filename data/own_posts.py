"""Import one person's posts for a style profile: their own data exports, Pages they manage, and
Mastodon / Bluesky profiles.

  python data/own_posts.py --profile me --name "Me" --x-archive twitter-2026.zip --i-own-these
  python data/own_posts.py --profile me --facebook-export facebook-me.zip --i-own-these
  python data/own_posts.py --profile me --instagram-export instagram-me.zip --i-own-these
  python data/own_posts.py --profile me --mastodon-archive archive.zip --i-own-these
  python data/own_posts.py --profile me --meta-page 1234567890 --page-token EAAB... --i-own-these
  python data/own_posts.py --profile me --profile-link https://bsky.app/profile/me.bsky.social --i-own-these
  python data/own_posts.py --profile poet --profile-link https://example.social/@poet   # CC-licensed author

Where posts may come from, and why:
  * Your own data export (X "Download an archive of your data", Facebook / Instagram "Download your
    information" as JSON, Mastodon "Export / Request your archive"). Your posts are yours, and the
    export is the platform's own way to hand them to you. The .zip is read in place and deleted
    after the import when it was uploaded through the Studio.
  * A Facebook Page or Instagram professional account you manage, through the Meta Graph API with
    a Page access token. Meta only gives that token to people who manage the Page.
  * A Mastodon or Bluesky profile: yours (--i-own-these), or anyone whose profile states CC0 / CC BY /
    CC BY-SA (same check as the news & web crawler, see data/social.py).
Not possible: other people's Facebook, Instagram, Threads or X accounts. Meta has no API for that and
forbids scraping; the X API is paid and its developer terms forbid training models on X content
(even through your own key), so X posts come only from your own archive.

What is kept: original posts with text. Reposts/retweets, replies (unless --keep-replies) and empty
media posts are dropped. Links, @mentions, e-mail addresses and phone numbers are removed (social.scrub).
Posts are stored in runs/_styles/<profile>/posts.jsonl; studio/styles.py trains a model on them.
"""
import argparse, hashlib, html, io, json, os, re, sys, time, zipfile
from email.utils import parsedate_to_datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import social  # noqa: E402

STYLES = os.path.join(ROOT, "runs", "_styles")
GRAPH = "https://graph.facebook.com/v21.0/"
MIN_CHARS = 20
KINDS = {"x-archive": "X archive", "facebook-export": "Facebook export", "instagram-export": "Instagram export",
         "mastodon-archive": "Mastodon archive", "meta-page": "Facebook Page (Graph API)",
         "instagram-account": "Instagram professional account (Graph API)", "profile-link": "Mastodon / Bluesky profile"}
NOT_POSSIBLE = (
    "Other people's Facebook, Instagram, Threads or X accounts can't be imported. Meta has no API for "
    "reading other people's posts and its terms forbid scraping; X's API is paid and its developer terms "
    "forbid training models on X content. Import your own export, a Page you manage, or a Mastodon / "
    "Bluesky profile (yours, or one whose author states a CC license).")


def say(m):
    print(m, flush=True)


# ---------------------------------------------------------------- helpers
def _post(text, created=None, kind="post", reply=False):
    return {"text": text or "", "created": created, "kind": kind, "reply": bool(reply)}


def _iso(ts):
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return None


def fix_meta_text(s):
    """Facebook/Instagram exports write UTF-8 bytes as \\u00XX escapes ("cafÃ©"): undo that."""
    if not isinstance(s, str):
        return ""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


class Archive:
    """A .zip or an unpacked folder, read without extracting everything (X archives hold GBs of media)."""

    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
        if not self.zip and not os.path.isdir(path):
            raise ValueError(f"{os.path.basename(path)} is neither a .zip nor a folder")

    def names(self):
        if self.zip:
            return [n for n in self.zip.namelist() if not n.endswith("/")]
        out = []
        for d, _, fs in os.walk(self.path):
            out += [os.path.relpath(os.path.join(d, f), self.path).replace(os.sep, "/") for f in fs]
        return out

    def read(self, name):
        if self.zip:
            with self.zip.open(name) as f:
                return f.read().decode("utf-8", "replace")
        return open(os.path.join(self.path, name), encoding="utf-8", errors="replace").read()

    def find(self, pattern):
        rx = re.compile(pattern, re.I)
        return sorted(n for n in self.names() if rx.search(n))


# ---------------------------------------------------------------- exports
def from_x_archive(path):
    """X/Twitter "Download an archive of your data": data/tweets.js (+ tweets-partN.js, note-tweet.js)."""
    a = Archive(path)
    files = a.find(r"(^|/)data/(tweets?|tweets-part\d+)\.js$")
    if not files:
        raise ValueError("No data/tweets.js in this file. Is it the archive from X → Settings → Your account → "
                         "Download an archive of your data?")
    notes = {}
    for nf in a.find(r"(^|/)data/note-tweet\.js$"):   # long posts: full text lives here
        for it in _ytd(a.read(nf)):
            n = it.get("noteTweet") or {}
            core = (n.get("core") or {}).get("text")
            if core:
                notes[core[:60]] = core
    out = []
    for f in files:
        for it in _ytd(a.read(f)):
            t = it.get("tweet") or it
            text = html.unescape(t.get("full_text") or t.get("text") or "")
            if text.startswith("RT @") or t.get("retweeted"):
                continue
            text = next((v for k, v in notes.items() if text.startswith(k[:40])), text)
            try:
                created = parsedate_to_datetime(t.get("created_at")).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (TypeError, ValueError):
                created = None
            out.append(_post(text, created, "x_post", t.get("in_reply_to_status_id_str") or t.get("in_reply_to_status_id")))
    return out


def _ytd(js):
    """window.YTD.tweets.part0 = [ ... ]  →  the list."""
    i = js.find("[")
    return json.loads(js[i:]) if i >= 0 else []


def from_facebook_export(path):
    """Facebook "Download your information" (JSON): your_posts*.json."""
    a = Archive(path)
    files = a.find(r"(^|/)posts/your_posts[^/]*\.json$") or a.find(r"your_posts[^/]*\.json$")
    if not files:
        if a.find(r"your_posts[^/]*\.html$"):
            raise ValueError("This export is in HTML. Request it again with Format: JSON.")
        raise ValueError("No posts/your_posts*.json in this file. Use Facebook → Settings → Download your "
                         "information, with Format: JSON and at least 'Posts' selected.")
    out = []
    for f in files:
        rows = json.loads(a.read(f))
        rows = rows if isinstance(rows, list) else rows.get("status_updates_v2") or []
        for r in rows:
            text = "\n".join(fix_meta_text(d.get("post")) for d in r.get("data") or [] if d.get("post"))
            out.append(_post(text, _iso(r.get("timestamp")), "facebook_post"))
    return out


def from_instagram_export(path):
    """Instagram "Download your information" (JSON): content/posts_*.json and reels.json captions."""
    a = Archive(path)
    files = a.find(r"(^|/)(content|media)/(posts_\d+|reels)\.json$")
    if not files:
        raise ValueError("No content/posts_1.json in this file. Use Instagram → Accounts Center → Your "
                         "information and permissions → Download your information, Format: JSON.")
    out = []
    for f in files:
        data = json.loads(a.read(f))
        rows = data if isinstance(data, list) else data.get("ig_reels_media") or []
        for r in rows:
            media = r.get("media") or []
            title = r.get("title") or next((m.get("title") for m in media if m.get("title")), "")
            ts = r.get("creation_timestamp") or next((m.get("creation_timestamp") for m in media), None)
            out.append(_post(fix_meta_text(title), _iso(ts), "instagram_caption"))
    return out


def from_mastodon_archive(path):
    """Mastodon "Export → Request your archive": outbox.json (ActivityPub)."""
    a = Archive(path)
    files = a.find(r"(^|/)outbox\.json$")
    if not files:
        raise ValueError("No outbox.json in this file. Use Mastodon → Preferences → Import and export → "
                         "Request your archive.")
    out = []
    for it in json.loads(a.read(files[0])).get("orderedItems") or []:
        o = it.get("object")
        if it.get("type") != "Create" or not isinstance(o, dict) or o.get("type") != "Note":
            continue   # Announce = boost
        public = "https://www.w3.org/ns/activitystreams#Public"
        if public not in (o.get("to") or []) + (o.get("cc") or []) or o.get("sensitive") or o.get("summary"):
            continue
        out.append(_post(social.html_text(o.get("content")), o.get("published"), "post", o.get("inReplyTo")))
    return out


# ---------------------------------------------------------------- APIs
def _get_json(url, params=None):
    import requests

    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "textllm-style-import/1.0"})
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400:
        msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        raise ValueError(f"HTTP {r.status_code}: {msg or r.text[:200]}")
    return data


def from_meta_page(page_id, token, limit=2000):
    """Posts of a Facebook Page you manage (Graph API, Page access token with pages_read_engagement)."""
    page = _get_json(GRAPH + page_id, {"fields": "name", "access_token": token})
    say(f"  Page: {page.get('name', page_id)}")
    url, params, out = GRAPH + f"{page_id}/posts", {"fields": "message,created_time", "limit": 100,
                                                    "access_token": token}, []
    while url and len(out) < limit:
        data = _get_json(url, params)
        out += [_post(p.get("message", ""), p.get("created_time"), "facebook_post") for p in data.get("data") or []]
        url, params = (data.get("paging") or {}).get("next"), None   # "next" already carries the token
    return out


def from_instagram_account(ig_user_id, token, limit=2000):
    """Captions of an Instagram professional account linked to a Page you manage (Graph API)."""
    url, params, out = GRAPH + f"{ig_user_id}/media", {"fields": "caption,timestamp", "limit": 100,
                                                       "access_token": token}, []
    while url and len(out) < limit:
        data = _get_json(url, params)
        out += [_post(p.get("caption", ""), p.get("timestamp"), "instagram_caption") for p in data.get("data") or []]
        url, params = (data.get("paging") or {}).get("next"), None
    return out


def from_profile_link(url, mine, limit=500, fetcher=None, replies=False):
    """A Mastodon or Bluesky profile through the public APIs. Not yours: only if its bio states CC0/BY/BY-SA."""
    host = url.split("/")[2].lower() if "//" in url else ""
    kind = "bluesky" if host in ("bsky.app", "www.bsky.app") else "mastodon"
    t = social.parse_target(kind, url)
    if t["api"] not in ("account", "author"):
        raise ValueError("Use a profile link: https://server/@user (Mastodon) or https://bsky.app/profile/handle")
    if fetcher is None:
        from data.crawl import Fetcher
        fetcher = Fetcher(delay=1.0)
    s = {"kind": kind, "url": url, "license": "own-rights" if mine else "cc-stated", "max_items": limit}
    out, refused = [], 0
    for p in social.posts(fetcher, s, replies):
        if p["skip"] in ("author states no open license", "author opted out"):
            refused += 1
            continue
        if p["skip"] in (None, "reply"):
            out.append(_post(p["text"], p["published"], "post", p["skip"] == "reply"))
    if refused and not out:
        raise ValueError("This author's profile doesn't state CC0, CC BY or CC BY-SA (or opts out of bots), so "
                         "their posts can't be used. If it is your own account, tick 'These are my posts'.")
    return out


# ---------------------------------------------------------------- profiles
def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:40] or "profile"


def profile_dir(pid):
    if not re.fullmatch(r"[a-z0-9-]{1,40}", pid or ""):
        raise ValueError("bad profile id")
    return os.path.join(STYLES, pid)


def load_profile(pid):
    p = os.path.join(profile_dir(pid), "meta.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def list_profiles():
    if not os.path.isdir(STYLES):
        return []
    out = [load_profile(d) for d in sorted(os.listdir(STYLES)) if os.path.isdir(os.path.join(STYLES, d))]
    return [p for p in out if p]


def load_posts(pid):
    p = os.path.join(profile_dir(pid), "posts.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def clean_posts(posts, keep_replies=False):
    """Drop replies/empty posts, remove personal data, de-duplicate. Returns (kept, dropped counts)."""
    kept, seen, dropped = [], set(), {"reply": 0, "too short": 0, "duplicate": 0}
    for p in posts:
        if p["reply"] and not keep_replies:
            dropped["reply"] += 1
            continue
        text = social.scrub(p["text"])
        if len(text) < MIN_CHARS:
            dropped["too short"] += 1
            continue
        h = hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()
        if h in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(h)
        kept.append({"text": text, "created": p["created"], "kind": p["kind"], "hash": h})
    return kept, dropped


def add_posts(pid, name, kind, posts, mine, keep_replies=False, label=None):
    """Add imported posts to a profile (created if new). Returns a summary dict."""
    if kind != "profile-link" and not mine:
        raise ValueError("Confirm that these are your own posts (or that you have the author's written permission).")
    d = profile_dir(pid)
    meta = load_profile(pid) or {"id": pid, "name": (name or pid).strip()[:80], "created": time.time(),
                                 "sources": [], "model": None}
    if name:
        meta["name"] = name.strip()[:80]
    kept, dropped = clean_posts(posts, keep_replies)
    old = load_posts(pid)
    have = {p["hash"] for p in old}
    new = [p for p in kept if p["hash"] not in have]
    dropped["duplicate"] += len(kept) - len(new)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "posts.jsonl"), "a", encoding="utf-8") as f:
        for p in new:
            f.write(json.dumps({**p, "source": kind}, ensure_ascii=False) + "\n")
    meta["sources"].append({"kind": kind, "label": label or KINDS.get(kind, kind), "found": len(posts),
                            "added": len(new), "dropped": dropped, "own": bool(mine), "time": time.time()})
    meta["posts"] = len(old) + len(new)
    tmp = os.path.join(d, "meta.json.tmp")
    json.dump(meta, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(d, "meta.json"))
    return {"profile": meta, "found": len(posts), "added": len(new), "dropped": dropped}


def import_source(pid, name, kind, mine=False, path=None, url=None, page_id=None, token=None, keep_replies=False):
    if kind == "x-archive":
        posts = from_x_archive(path)
    elif kind == "facebook-export":
        posts = from_facebook_export(path)
    elif kind == "instagram-export":
        posts = from_instagram_export(path)
    elif kind == "mastodon-archive":
        posts = from_mastodon_archive(path)
    elif kind == "meta-page":
        if not (page_id and token):
            raise ValueError("Give the Page ID and a Page access token.")
        posts = from_meta_page(page_id.strip(), token.strip())
    elif kind == "instagram-account":
        if not (page_id and token):
            raise ValueError("Give the Instagram account ID and an access token.")
        posts = from_instagram_account(page_id.strip(), token.strip())
    elif kind == "profile-link":
        why = social.not_supported(url or "")
        if why:
            raise ValueError(why + " " + NOT_POSSIBLE)
        posts = from_profile_link(url, mine, replies=keep_replies)
    else:
        raise ValueError(f"unknown source {kind}")
    return add_posts(pid, name, kind, posts, mine, keep_replies, label=url if kind == "profile-link" else None)


def delete_profile(pid):
    import shutil

    d = profile_dir(pid)
    if not os.path.isdir(d):
        raise LookupError("No such profile.")
    shutil.rmtree(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="profile id, e.g. me")
    ap.add_argument("--name", help="display name, e.g. 'Hovhannes'")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--x-archive")
    g.add_argument("--facebook-export")
    g.add_argument("--instagram-export")
    g.add_argument("--mastodon-archive")
    g.add_argument("--meta-page", metavar="PAGE_ID")
    g.add_argument("--instagram-account", metavar="IG_USER_ID")
    g.add_argument("--profile-link", metavar="URL")
    ap.add_argument("--page-token", default=os.environ.get("META_PAGE_TOKEN"), help="or set META_PAGE_TOKEN")
    ap.add_argument("--i-own-these", action="store_true", help="these are my posts (or I have written permission)")
    ap.add_argument("--keep-replies", action="store_true")
    a = ap.parse_args()
    kind, val = next((k.replace("_", "-"), v) for k, v in vars(a).items()
                     if k in ("x_archive", "facebook_export", "instagram_export", "mastodon_archive", "meta_page",
                              "instagram_account", "profile_link") and v)
    path = val if kind.endswith(("archive", "export")) else None
    r = import_source(slug(a.profile), a.name, kind, a.i_own_these, path=path, url=val if kind == "profile-link" else None,
                      page_id=val if kind in ("meta-page", "instagram-account") else None, token=a.page_token,
                      keep_replies=a.keep_replies)
    say(f"{r['found']} posts found, {r['added']} added to '{r['profile']['name']}' "
        f"({r['profile']['posts']} in total). Dropped: " + ", ".join(f"{v} {k}" for k, v in r["dropped"].items() if v))


if __name__ == "__main__":
    main()
