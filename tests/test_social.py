"""Offline test of the Mastodon and Bluesky source kinds with recorded-shape API responses.

    python tests/test_social.py
"""
import json, os, sys, tempfile
from urllib.parse import urlsplit, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import crawl, social

tmp = tempfile.mkdtemp()
crawl.STATE = os.path.join(tmp, "_crawler")
crawl.SOURCES_PATH = os.path.join(crawl.STATE, "sources.json")
crawl.DB_PATH = os.path.join(crawl.STATE, "crawl.db")
crawl.PENDING = os.path.join(crawl.STATE, "pending.txt")
crawl.PARTIAL = os.path.join(crawl.STATE, "partial")
crawl.LOCK = os.path.join(crawl.STATE, "lock")
crawl.OUT = os.path.join(tmp, "news")
crawl.ATTRIBUTION = os.path.join(crawl.OUT, "attribution.jsonl")
crawl.JOB_LOCK = os.path.join(tmp, "_job.lock")

LONG = "The rain came down on the harbour all night, and the fishing boats knocked together like old friends " \
       "telling the same story. By morning the town smelled of salt and bread."


def acct(name, note, **kw):
    return {"id": name, "acct": name, "note": f"<p>{note}</p>", "fields": [], **kw}


def status(i, account, content=LONG, **kw):
    return {"id": str(100 - i), "url": f"https://writers.example/@{account['acct']}/{i}", "visibility": "public",
            "in_reply_to_id": None, "reblog": None, "sensitive": False, "spoiler_text": "", "language": "en",
            "created_at": "2026-11-06T10:00:00Z", "account": account, "content": f"<p>{content}</p>", **kw}


cc = acct("ann", "Poems. All my posts are CC BY 4.0")
nc = acct("bob", "CC BY-NC please")
none = acct("cat", "Just a cat person")
optout = acct("dan", "CC0 #nobot")
noidx = acct("eve", "CC0", noindex=True)
MASTO = [
    status(1, cc, LONG + " Write to @friend@other.example or me@mail.example, call +1 (555) 123-4567, see https://x.example/a"),
    status(2, nc), status(3, none), status(4, optout), status(5, noidx),
    status(6, cc, visibility="unlisted"), status(7, cc, in_reply_to_id="9"), status(8, cc, spoiler_text="cw"),
    status(9, cc, content="too short"), status(10, cc, reblog={"id": "x"}),
    status(11, cc),  # same text as 1 minus the scrubbed bits → different hash; dedupe check via 12
    status(12, cc),  # duplicate of 11
]

BSKY_FEED = {"feed": [
    {"post": {"uri": "at://did:plc:a/app.bsky.feed.post/3k1", "author": {"did": "did:plc:a", "handle": "ann.bsky.social", "labels": []},
              "record": {"text": LONG + " #writing", "langs": ["en"], "createdAt": "2026-11-06T10:00:00Z"}, "labels": []}},
    {"post": {"uri": "at://did:plc:b/app.bsky.feed.post/3k2", "author": {"did": "did:plc:b", "handle": "bob.bsky.social",
              "labels": [{"val": "!no-unauthenticated"}]}, "record": {"text": LONG + " b", "createdAt": "2026-11-06T10:00:00Z"}}},
    {"post": {"uri": "at://did:plc:a/app.bsky.feed.post/3k3", "author": {"did": "did:plc:a", "handle": "ann.bsky.social"},
              "record": {"text": LONG + " reply", "reply": {"parent": {}}, "createdAt": "2026-11-06T10:00:00Z"}}},
    {"post": {"uri": "at://did:plc:c/app.bsky.feed.post/3k4", "author": {"did": "did:plc:c", "handle": "cat.bsky.social"},
              "record": {"text": LONG + " c", "createdAt": "2026-11-06T10:00:00Z"}}, "reason": {"$type": "repost"}},
]}
PROFILES = {"profiles": [{"did": "did:plc:a", "description": "Stories, creativecommons.org/publicdomain/zero/1.0"},
                         {"did": "did:plc:b", "description": "CC0"}, {"did": "did:plc:c", "description": "CC BY-SA"}]}
calls = []


class R:
    def __init__(self, data, code=200):
        self.data, self.status_code, self.headers = data, code, {}

    def json(self):
        return self.data


def fake(self, url, host, headers=None, stream=False):
    calls.append(url)
    p = urlsplit(url)
    if p.netloc == "writers.example":
        if "max_id" in p.query:
            return R([])
        return R(MASTO)
    if p.netloc == "locked.example":
        return R({"error": "login"}, 401)
    if "getAuthorFeed" in p.path or "searchPosts" in p.path:
        return R(BSKY_FEED if "searchPosts" not in p.path else {"posts": [x["post"] for x in BSKY_FEED["feed"]]})
    if "getProfiles" in p.path:
        return R(PROFILES)
    raise AssertionError(url)


crawl.Fetcher._request = fake
crawl.Fetcher._wait = lambda self, host: None


def expect_error(src, needle):
    try:
        crawl.check_source(src)
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"no error for {src}")


# validation
expect_error({"url": "https://www.reddit.com/r/WritingPrompts", "kind": "feed", "license": "cc-by"}, "Reddit")
expect_error({"url": "https://x.com/someone", "kind": "links", "license": "unknown", "rights_confirmed": True}, "X (Twitter)")
expect_error({"url": "https://mastodon.social/tags/poetry", "kind": "mastodon", "license": "cc-stated", "terms_confirmed": True}, "forbid")
expect_error({"url": "https://writers.example/tags/poetry", "kind": "mastodon", "license": "cc-stated"}, "terms")
expect_error({"url": "https://writers.example/tags/poetry", "kind": "mastodon", "license": "cc-by", "terms_confirmed": True}, "mixes")
expect_error({"url": "https://writers.example/about", "kind": "mastodon", "license": "cc-stated", "terms_confirmed": True}, "link like")
expect_error({"url": "https://example.org/feed", "kind": "feed", "license": "cc-stated"}, "only for")
expect_error({"url": "https://example.org/x", "kind": "bluesky", "license": "cc-stated", "terms_confirmed": True}, "bsky.app")
assert social.parse_target("bluesky", "https://bsky.app/search?q=short+story") == {"api": "search", "q": "short story"}
assert social.parse_target("bluesky", "https://bsky.app/profile/a.b/feed/xyz")["api"] == "feed"
assert social.parse_target("mastodon", "https://h.example/@me@other.example") == {"api": "account", "host": "h.example", "acct": "me@other.example"}

crawl.save_sources([])
crawl.add_source({"url": "https://writers.example/tags/poetry", "kind": "mastodon", "license": "cc-stated",
                  "terms_confirmed": True, "name": "Masto poetry"})
crawl.add_source({"url": "https://bsky.app/profile/ann.bsky.social", "kind": "bluesky", "license": "cc-stated",
                  "terms_confirmed": True, "name": "Bsky writing"})
crawl.add_source({"url": "https://locked.example/public/local", "kind": "mastodon", "license": "own-rights",
                  "rights_confirmed": True, "terms_confirmed": True, "name": "Locked"})
# changing the server drops the terms confirmation
try:
    crawl.update_source("locked", {"url": "https://locked.example/public"})
    raise AssertionError("server change kept the terms confirmation")
except ValueError as e:
    assert "terms" in str(e)
crawl.update_source("locked", {"url": "https://locked.example/public/local", "terms_confirmed": True})

added = crawl.crawl()
docs = open(crawl.PENDING, encoding="utf-8").read().split(crawl.EOT)
docs = [d.strip() for d in docs if d.strip()]
attr = [json.loads(l) for l in open(crawl.ATTRIBUTION, encoding="utf-8")]
print("added", added)
for d in docs:
    print("---", d[:220].replace("\n", " "))
print(json.dumps(attr, indent=1)[:1500])
with crawl.db() as c:
    for r in c.execute("SELECT url, status, error FROM items ORDER BY url"):
        print(r["status"], r["error"], r["url"])

assert added == 3, added    # masto 1 and 11 (12 duplicate), bluesky ann only
assert all("@friend" not in d and "mail.example" not in d and "555" not in d and "http" not in d for d in docs)
assert "ann" not in " ".join(docs)
assert {a["license"] for a in attr} == {"cc-by", "cc0"}
assert all(a["author"] for a in attr)
errs = {r["url"].rsplit("/", 1)[-1]: r["error"] for r in crawl.db().execute("SELECT url, error FROM items WHERE status='skipped'")}
assert errs["2"] == "author states no open license" and errs["3"] == "author states no open license"
assert errs["4"] == "author opted out" and errs["5"] == "author opted out"
assert errs["6"] == "not public" and errs["7"] == "reply" and errs["8"] == "content warning"
assert errs["9"].startswith("only") and errs["10"] == "boost" and errs["12"] == "duplicate"
assert errs["3k2"] == "author opted out" and errs["3k3"] == "reply" and errs["3k4"] == "repost"

# second run: nothing new, nothing re-added
n0 = len(calls)
assert crawl.crawl() == 0
print("second run added 0, API calls", len(calls) - n0)

# review fixes: NC/ND split by any whitespace, Mastodon link HTML, retrying a post whose save failed
assert social.stated_license("CC BY\u00a0NC 4.0") is None and social.stated_license("CC BY\nND") is None
h = ('<p>Read <a href="https://example.com/private"><span class="invisible">https://</span><span>example.com/private</span></a> '
     'hi <span class="h-card"><a href="https://m.ex/@bob" class="u-url mention">@<span>bob</span></a></span> '
     '<a href="https://m.ex/tags/poem" class="mention hashtag" rel="tag">#<span>poem</span></a></p>')
assert social.scrub(social.html_text(h)) == "Read hi @someone #poem", social.scrub(social.html_text(h))

MASTO[:] = [status(20, cc, LONG + " retry me")]
real_append, boom = crawl._append, {"n": 1}
def flaky(path, doc):
    if boom["n"]:
        boom["n"] -= 1
        raise OSError("disk full")
    real_append(path, doc)
crawl._append = flaky
assert crawl.crawl() == 0          # the save fails: the post stays retryable
crawl._append = real_append
assert crawl.crawl() == 1          # ...and is added on the next run
print("ALL OK")
