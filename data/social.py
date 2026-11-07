"""Public posts from social networks (Mastodon and Bluesky) as a crawler source kind.

Used by data/crawl.py for sources of kind "mastodon" or "bluesky". A source's link says what to read:

  Mastodon  https://<server>/tags/<hashtag>        posts with a hashtag, as that server sees them
            https://<server>/@<user>                one account's public posts (no replies or boosts)
            https://<server>/public/local           the server's local public timeline
  Bluesky   https://bsky.app/profile/<handle>       one account's posts (no replies or reposts)
            https://bsky.app/profile/<handle>/feed/<name>   a custom feed
            https://bsky.app/hashtag/<tag>          posts with a hashtag
            https://bsky.app/search?q=<words>       a post search

Both are read through their public, documented APIs without logging in: the Mastodon REST API
(/api/v1/...) and the Bluesky AppView (public.api.bsky.app/xrpc/...). Nothing is scraped from web pages.

Rights. A public post is still its author's copyright, so a social source is only allowed with
  - license "cc-stated": a post is kept only when its author's profile states CC0, CC BY or CC BY-SA
    (NC and ND licenses are refused); the license found is recorded with the post, or
  - license "own-rights" / "unknown" with rights_confirmed: your own account, or permission.
And always: the admin confirms the server's terms allow it (terms_confirmed). mastodon.social and
mastodon.online (run by Mastodon gGmbH) forbid collecting posts for LLM training, so they are refused.

Skipped always: non-public posts, replies, boosts/reposts, posts with a content warning or marked
sensitive, accounts that opt out (Mastodon "noindex", #nobot / #noai / #notraining / #nosearch in the
profile; Bluesky "!no-unauthenticated"), and anything shorter than the source's min_chars.

Personal data. The training text never contains the author: @mentions become "@someone", links are
removed, e-mail addresses and phone numbers are replaced. The post's link, author handle and license
are kept only in data/raw/news/attribution.jsonl, because CC BY requires attribution.
"""
import html as htmlmod
import re
import time
from urllib.parse import parse_qs, quote, urlsplit

KINDS = ("mastodon", "bluesky")
BSKY_API = "https://public.api.bsky.app/xrpc/"
# Servers whose terms forbid collecting posts for LLM training (mastodon.social terms, July 2025).
FORBIDDEN_SERVERS = {"mastodon.social", "mastodon.online"}
# Networks without a usable route: say why instead of trying.
NOT_SUPPORTED = {
    "reddit.com": "Reddit's Data API terms forbid using Reddit content to train AI models without a separate "
                  "agreement with Reddit, so it is not offered.",
    "x.com": "X (Twitter) only offers paid API access, and its terms forbid using the content to train models.",
    "twitter.com": "X (Twitter) only offers paid API access, and its terms forbid using the content to train models.",
    "facebook.com": "Facebook has no API for public posts, and its terms forbid automated collection.",
    "instagram.com": "Instagram has no API for public posts, and its terms forbid automated collection.",
    "threads.net": "Threads has no public read API for other people's posts, and Meta's terms forbid scraping.",
    "threads.com": "Threads has no public read API for other people's posts, and Meta's terms forbid scraping.",
    "linkedin.com": "LinkedIn's terms forbid scraping and its API does not give access to public posts.",
    "tiktok.com": "TikTok's research API is only for approved academic research, not for training models.",
}
OPT_OUT = re.compile(r"#(nobot|noai|notraining|nosearch|noindex|nobots|noarchive)\b", re.I)


def not_supported(url):
    host = urlsplit(url).netloc.lower().split(":")[0]
    for h, why in NOT_SUPPORTED.items():
        if host == h or host.endswith("." + h):
            return why
    return None


# ---------------------------------------------------------------- what a source link means
def parse_target(kind, url):
    """Source link → {"api": ..., ...}. Raises ValueError with a helpful message."""
    p = urlsplit(url.strip())
    host, parts = p.netloc.lower(), [x for x in p.path.split("/") if x]
    if kind == "mastodon":
        if host in FORBIDDEN_SERVERS:
            raise ValueError(f"{host}'s terms forbid collecting posts for AI training. Choose another server "
                             "whose terms allow it, or your own server.")
        if len(parts) >= 2 and parts[0] == "tags":
            return {"api": "tag", "host": host, "tag": parts[1].lstrip("#")}
        if parts and parts[0].startswith("@"):
            return {"api": "account", "host": host, "acct": parts[0][1:]}
        if parts[:1] == ["public"]:
            return {"api": "public", "host": host, "local": parts[1:2] == ["local"]}
        raise ValueError("For Mastodon use a link like https://server/tags/poetry, https://server/@user "
                         "or https://server/public/local")
    if kind == "bluesky":
        if host not in ("bsky.app", "www.bsky.app"):
            raise ValueError("For Bluesky use a bsky.app link, e.g. https://bsky.app/profile/alice.bsky.social")
        if len(parts) >= 4 and parts[0] == "profile" and parts[2] == "feed":
            return {"api": "feed", "actor": parts[1], "feed": parts[3]}
        if len(parts) >= 2 and parts[0] == "profile":
            return {"api": "author", "actor": parts[1]}
        if len(parts) >= 2 and parts[0] == "hashtag":
            return {"api": "search", "q": "#" + parts[1].lstrip("#")}
        if parts[:1] == ["search"] and parse_qs(p.query).get("q"):
            return {"api": "search", "q": parse_qs(p.query)["q"][0]}
        raise ValueError("For Bluesky use https://bsky.app/profile/<handle>, .../profile/<handle>/feed/<name>, "
                         "https://bsky.app/hashtag/<tag> or https://bsky.app/search?q=<words>")
    raise ValueError(f"not a social source kind: {kind}")


# ---------------------------------------------------------------- licenses stated in a profile
_CC_URL = re.compile(r"creativecommons\.org/(?:licenses/([a-z-]+)|publicdomain/(zero))", re.I)
_CC_ABBR = re.compile(r"\bCC[\s-]?(0|ZERO|Zero)\b|\bCC[\s-]?(BY(?:[\s-](?:SA|NC|ND))*)\b")   # upper case only


def stated_license(text):
    """'cc0', 'cc-by', 'cc-by-sa' or None for a profile text. NC / ND licenses → None (not allowed)."""
    found = []
    for m in _CC_URL.finditer(text or ""):
        found.append("0" if m.group(2) else m.group(1))
    for m in _CC_ABBR.finditer(text or ""):
        found.append("0" if m.group(1) else m.group(2))
    best = None
    for code in found:
        bits = code.lower().replace(" ", "-").split("-")
        if "nc" in bits or "nd" in bits:
            return None   # a restrictive license anywhere in the profile: stay out
        lic = "cc0" if code == "0" else "cc-by-sa" if "sa" in bits else "cc-by"
        best = best or lic
    return best


# ---------------------------------------------------------------- personal data
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
MENTION = re.compile(r"(?<![\w@])@[\w.-]+(?:@[\w.-]+)?\w")
PHONE = re.compile(r"(?<!\w)\+?\d[\d ()./-]{7,}\d(?!\w)")


def _phone(m):
    """Looks like a phone number (9+ digits, not a date, year range or plain number)?"""
    x = m.group(0)
    if sum(ch.isdigit() for ch in x) < 9 or re.fullmatch(r"[\d.,]+", x) or re.fullmatch(r"\d{4}[-/.]\d\d[-/.]\d\d.*", x):
        return x
    return "[phone]"


def scrub(text):
    """Remove personal data from a post: links, e-mails, @mentions, phone numbers."""
    t = URL.sub("", text)
    t = MENTION.sub("@someone", t)   # before EMAIL: @user@server looks like an address
    t = EMAIL.sub("[email]", t)
    t = PHONE.sub(_phone, t)
    t = re.sub(r"(@someone[\s,]*){2,}", "@someone ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def html_text(h):
    """Mastodon post/profile HTML → plain text with paragraph breaks."""
    h = re.sub(r"<br\s*/?>", "\n", h or "", flags=re.I)
    h = re.sub(r"</p>\s*<p[^>]*>", "\n\n", h, flags=re.I)
    h = re.sub(r"<span class=\"invisible\">[^<]*</span>", "", h)   # Mastodon hides parts of long links
    return htmlmod.unescape(re.sub(r"<[^>]+>", "", h)).strip()


# ---------------------------------------------------------------- the APIs
def _json(f, url):
    host = "{0.scheme}://{0.netloc}".format(urlsplit(url))
    r = f._request(url, host, {"Accept": "application/json"})
    if r.status_code == 401 or r.status_code == 403:
        raise PermissionError(f"HTTP {r.status_code}: this API needs a login on this server, so it can't be read "
                              "publicly. Try another server or link.")
    if r.status_code >= 400:
        raise ValueError(f"HTTP {r.status_code} from {urlsplit(url).netloc}")
    return r.json(), r


def _mastodon_posts(f, t, limit):
    base = f"https://{t['host']}/api/v1/"
    if t["api"] == "tag":
        path = f"timelines/tag/{quote(t['tag'])}?limit=40"
    elif t["api"] == "public":
        path = "timelines/public?limit=40" + ("&local=true" if t["local"] else "")
    else:
        acc, _ = _json(f, base + "accounts/lookup?acct=" + quote(t["acct"]))
        path = f"accounts/{acc['id']}/statuses?limit=40&exclude_replies=true&exclude_reblogs=true"
    got, max_id = [], None
    while len(got) < limit:
        page, _ = _json(f, base + path + (f"&max_id={max_id}" if max_id else ""))
        if not page:
            break
        got += page
        max_id = page[-1]["id"]
        if len(page) < 20:
            break
    for st in got[:limit]:
        a = st.get("account") or {}
        bio = html_text(a.get("note")) + "\n" + "\n".join(
            f"{x.get('name', '')} {html_text(x.get('value'))}" for x in a.get("fields") or [])
        skip = None
        if st.get("reblog"):
            skip = "boost"
        elif st.get("visibility") != "public":
            skip = "not public"
        elif st.get("in_reply_to_id"):
            skip = "reply"
        elif st.get("sensitive") or st.get("spoiler_text"):
            skip = "content warning"
        elif a.get("noindex") or OPT_OUT.search(bio):
            skip = "author opted out"
        acct = a.get("acct", "")
        yield {"url": st.get("url") or st.get("uri"), "text": html_text(st.get("content")), "bio": bio,
               "author": acct if "@" in acct else f"{acct}@{t['host']}", "published": st.get("created_at"),
               "lang": st.get("language"), "skip": skip}


def _bsky_posts(f, t, limit):
    if t["api"] == "author":
        method, q = "app.bsky.feed.getAuthorFeed", f"actor={quote(t['actor'])}&filter=posts_no_replies&limit=100"
    elif t["api"] == "feed":
        did = t["actor"]
        if not did.startswith("did:"):
            did = _json(f, BSKY_API + "com.atproto.identity.resolveHandle?handle=" + quote(did))[0]["did"]
        method, q = "app.bsky.feed.getFeed", "feed=" + quote(f"at://{did}/app.bsky.feed.generator/{t['feed']}") + "&limit=100"
    else:
        method, q = "app.bsky.feed.searchPosts", f"q={quote(t['q'])}&sort=latest&limit=100"
    got, cursor = [], None
    while len(got) < limit:
        data, _ = _json(f, BSKY_API + method + "?" + q + (f"&cursor={quote(cursor)}" if cursor else ""))
        items = data.get("feed") or [{"post": p} for p in data.get("posts") or []]
        if not items:
            break
        got += items
        cursor = data.get("cursor")
        if not cursor:
            break
    got = got[:limit]
    # profiles (for the bio's license and opt-outs), 25 per request
    dids = sorted({it["post"]["author"]["did"] for it in got})
    bios = {}
    for i in range(0, len(dids), 25):
        data, _ = _json(f, BSKY_API + "app.bsky.actor.getProfiles?" + "&".join("actors=" + quote(d) for d in dids[i:i + 25]))
        bios.update({p["did"]: p.get("description") or "" for p in data.get("profiles") or []})
    for it in got:
        p, rec = it["post"], it["post"].get("record") or {}
        a = p.get("author") or {}
        labels = {l.get("val") for l in (a.get("labels") or []) + (p.get("labels") or [])}
        bio = bios.get(a.get("did"), "")
        skip = None
        if it.get("reason"):
            skip = "repost"
        elif rec.get("reply"):
            skip = "reply"
        elif "!no-unauthenticated" in labels or OPT_OUT.search(bio):
            skip = "author opted out"
        elif labels & {"porn", "sexual", "nudity", "graphic-media", "gore", "!warn"}:
            skip = "content warning"
        rkey = p["uri"].rsplit("/", 1)[-1]
        yield {"url": f"https://bsky.app/profile/{a.get('handle') or a.get('did')}/post/{rkey}",
               "text": rec.get("text", ""), "bio": bio, "author": a.get("handle") or a.get("did"),
               "published": rec.get("createdAt"), "lang": (rec.get("langs") or [None])[0], "skip": skip}


def posts(f, s):
    """Yield posts of a social source: {url, text (scrubbed), author, published, lang, license, skip}."""
    t = parse_target(s["kind"], s["url"])
    gen = _mastodon_posts if s["kind"] == "mastodon" else _bsky_posts
    for p in gen(f, t, s.get("max_items", 30)):
        if not p["url"]:
            continue
        lic = stated_license(p["bio"])
        if s["license"] == "cc-stated":
            p["license"] = lic
            if not p["skip"] and not lic:
                p["skip"] = "author states no open license"
        else:
            p["license"] = s["license"]
        p["text"] = scrub(p["text"])
        p["published"] = p["published"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        yield p
