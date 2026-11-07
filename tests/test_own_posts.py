"""Offline test of style-profile imports (exports, Graph API, profile links) and training pairs.

    python tests/test_own_posts.py
"""
import json, os, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import own_posts, social
from studio import styles

tmp = tempfile.mkdtemp()
own_posts.STYLES = os.path.join(tmp, "_styles")
LONG = "Baked the first sourdough of the season this morning, and the whole street smelled of it by eight."


def make_zip(name, files):
    p = os.path.join(tmp, name)
    with zipfile.ZipFile(p, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return p


# --- X archive
tweets = [{"tweet": {"full_text": LONG + " Come by @neighbour https://t.co/abc", "created_at": "Wed Oct 10 20:19:24 +0000 2018"}},
          {"tweet": {"full_text": "RT @someone: not mine at all, a retweet of a long enough text", "created_at": "Wed Oct 10 20:19:24 +0000 2018"}},
          {"tweet": {"full_text": "@friend thanks, that is a reply and should go", "in_reply_to_status_id_str": "1",
                     "created_at": "Wed Oct 10 20:19:24 +0000 2018"}},
          {"tweet": {"full_text": "ok", "created_at": "Wed Oct 10 20:19:24 +0000 2018"}},
          {"tweet": {"full_text": "Rain again &amp; again. Good day for writing letters nobody will read.",
                     "created_at": "Thu Oct 11 20:19:24 +0000 2018"}}]
xz = make_zip("x.zip", {"data/tweets.js": "window.YTD.tweets.part0 = " + json.dumps(tweets),
                        "data/tweets_media/big.jpg": "x" * 100})
r = own_posts.import_source("me", "Me", "x-archive", mine=True, path=xz)
assert r["found"] == 4 and r["added"] == 2, r       # retweet dropped at read; reply + "ok" dropped by clean
assert r["dropped"]["reply"] == 1 and r["dropped"]["too short"] == 1
posts = own_posts.load_posts("me")
assert posts[0]["text"].endswith("Come by @someone"), posts[0]["text"]
assert "&amp;" not in posts[1]["text"] and posts[0]["created"] == "2018-10-10T20:19:24Z"

# importing the same archive again adds nothing
assert own_posts.import_source("me", "Me", "x-archive", mine=True, path=xz)["added"] == 0

# not confirming ownership is refused for exports
try:
    own_posts.import_source("me", "Me", "x-archive", mine=False, path=xz)
    raise AssertionError("ownership not required")
except ValueError as e:
    assert "own posts" in str(e)

# --- Facebook export (Meta writes UTF-8 bytes as \u00XX escapes)
fb_text = "CafÃ© opening on Saturday ð\u009f\u008e\u0089 bring your friends and your appetite"
fbz = make_zip("fb.zip", {"your_facebook_activity/posts/your_posts__check_ins__photos_and_videos_1.json":
                          json.dumps([{"timestamp": 1700000000, "data": [{"post": fb_text}]},
                                      {"timestamp": 1700000001, "attachments": []}])})
r = own_posts.import_source("me", "Me", "facebook-export", mine=True, path=fbz)
assert r["added"] == 1, r
assert "Café opening on Saturday 🎉" in own_posts.load_posts("me")[-1]["text"]
try:
    own_posts.import_source("me", "Me", "facebook-export", mine=True, path=make_zip("fbh.zip", {"posts/your_posts_1.html": "<html>"}))
    raise AssertionError
except ValueError as e:
    assert "JSON" in str(e)

# --- Instagram export
igz = make_zip("ig.zip", {"your_instagram_activity/content/posts_1.json": json.dumps(
    [{"media": [{"uri": "a.jpg", "creation_timestamp": 1700000000, "title": "Morning light on the old bridge, coffee in hand #morning"}]},
     {"media": [{"uri": "b.jpg", "creation_timestamp": 1700000000, "title": ""}]}]),
    "your_instagram_activity/content/reels.json": json.dumps({"ig_reels_media": [{"media": [
        {"title": "How we shape the loaves, start to finish, in one minute", "creation_timestamp": 1700000500}]}]})})
r = own_posts.import_source("me", "Me", "instagram-export", mine=True, path=igz)
assert r["added"] == 2 and r["dropped"]["too short"] == 1, r

# --- Mastodon archive
public = "https://www.w3.org/ns/activitystreams#Public"
outbox = {"orderedItems": [
    {"type": "Create", "object": {"type": "Note", "to": [public], "published": "2026-01-01T00:00:00Z",
                                  "content": "<p>Wrote three pages today. The ending finally knows what it wants.</p>"}},
    {"type": "Announce", "object": "https://x/1"},
    {"type": "Create", "object": {"type": "Note", "to": ["https://me/followers"], "content": "<p>followers only, keep this private</p>"}},
    {"type": "Create", "object": {"type": "Note", "to": [public], "summary": "cw", "content": "<p>behind a content warning text</p>"}}]}
r = own_posts.import_source("me", "Me", "mastodon-archive", mine=True, path=make_zip("m.zip", {"outbox.json": json.dumps(outbox)}))
assert r["found"] == 1 and r["added"] == 1, r

# --- Facebook Page through the Graph API (paged)
calls = []
def fake_get(url, params=None):
    calls.append((url, params))
    if url.endswith("/123"):
        return {"name": "Our Bakery"}
    if "after=2" in url:
        return {"data": [{"message": "Second page post: rye is back on the shelf from Monday", "created_time": "2026-01-02"}]}
    return {"data": [{"message": "We open at seven tomorrow, fresh croissants for the first fifty people", "created_time": "2026-01-01"},
                     {"created_time": "2026-01-01"}], "paging": {"next": "https://graph.facebook.com/v21.0/123/posts?after=2"}}
own_posts._get_json = fake_get
r = own_posts.import_source("bakery", "Our Bakery", "meta-page", mine=True, page_id="123", token="T")
assert r["found"] == 3 and r["added"] == 2, r
assert calls[1][1]["access_token"] == "T" and calls[2][1] is None   # the "next" link carries the token itself

# --- other people's X / Meta pages are refused with the reason
for u in ("https://x.com/somebody", "https://www.facebook.com/somepage", "https://www.instagram.com/someone"):
    try:
        own_posts.import_source("other", "Other", "profile-link", url=u)
        raise AssertionError(u)
    except ValueError as e:
        assert "can't be imported" in str(e), e

# --- Mastodon / Bluesky profile links: not yours → only CC-licensed authors
class F:
    def __init__(self, bio):
        self.bio = bio
    def _request(self, url, host, headers=None, stream=False):
        class R:
            status_code, headers = 200, {}
            def __init__(s, d): s.d = d
            def json(s): return s.d
        if "lookup" in url:
            return R({"id": "1"})
        if "max_id" in url:
            return R([])
        acc = {"acct": "poet", "note": self.bio, "fields": []}
        return R([{"id": "9", "url": "https://w.example/@poet/9", "visibility": "public", "in_reply_to_id": None,
                   "reblog": None, "sensitive": False, "spoiler_text": "", "account": acc, "created_at": "2026-01-01",
                   "content": "<p>Small poems for long winters, one each morning until spring.</p>"}])
try:
    own_posts.from_profile_link("https://w.example/@poet", mine=False, fetcher=F("<p>just poems</p>"))
    raise AssertionError("unlicensed author accepted")
except ValueError as e:
    assert "CC0" in str(e)
assert len(own_posts.from_profile_link("https://w.example/@poet", mine=False, fetcher=F("<p>poems, CC BY 4.0</p>"))) == 1
assert len(own_posts.from_profile_link("https://w.example/@poet", mine=True, fetcher=F("<p>just poems</p>"))) == 1
# "keep replies" asks the API for replies; otherwise the API leaves them out
seen = []
class G(F):
    def _request(self, url, host, headers=None, stream=False):
        seen.append(url)
        return F._request(self, url, host, headers, stream)
own_posts.from_profile_link("https://w.example/@poet", mine=True, fetcher=G(""), replies=True)
assert not any("exclude_replies" in u for u in seen), seen
seen.clear()
own_posts.from_profile_link("https://w.example/@poet", mine=True, fetcher=G(""))
assert any("exclude_replies=true" in u for u in seen), seen

# --- training pairs and the command
try:
    styles.build_pairs("me", os.path.join(tmp, "p.jsonl"))
    raise AssertionError("too few posts accepted")
except ValueError as e:
    assert "at least" in str(e)
many = [own_posts._post(f"Post number {i}: notes on bread, rain and the harbour at dawn, version {i}") for i in range(40)]
own_posts.add_posts("me", "Me", "x-archive", many, mine=True)
n = styles.build_pairs("me", os.path.join(tmp, "p.jsonl"))
rows = [json.loads(l) for l in open(os.path.join(tmp, "p.jsonl"), encoding="utf-8")]
assert n == len(rows) == own_posts.load_profile("me")["posts"], (n, len(rows))
assert all(r["system"] == styles.F.system_prompt("style") for r in rows)
assert any("about:" in r["prompt"] for r in rows)
print("example pair:", json.dumps(rows[0], ensure_ascii=False)[:300])
print("keywords:", styles.keywords(LONG))

runs = os.path.join(tmp, "runs")
os.makedirs(os.path.join(runs, "ft"))
json.dump({"type": "lora", "base": "Qwen/Qwen3-0.6B"}, open(os.path.join(runs, "ft", "meta.json"), "w"))
cmd, out = styles.train_command("me", "python", runs, start_from=os.path.join(runs, "ft"))
assert out.endswith("style-me") and "--init-adapter" in cmd and cmd[cmd.index("--examples") + 1] == "0"
assert json.loads(cmd[cmd.index("--meta-extra") + 1])["style"]["name"] == "Me"
os.makedirs(os.path.join(runs, "scr"))
json.dump({"type": "scratch"}, open(os.path.join(runs, "scr", "meta.json"), "w"))
try:
    styles.train_command("me", "python", runs, start_from=os.path.join(runs, "scr"))
    raise AssertionError
except ValueError as e:
    assert "LoRA" in str(e)

own_posts.delete_profile("bakery")
assert [p["id"] for p in own_posts.list_profiles()] == ["me"]
print("ALL OK")
