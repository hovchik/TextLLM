"""Style profiles: teach a model to write like one account, from that account's imported posts.

The posts come from data/own_posts.py (runs/_styles/<id>/posts.jsonl). Training is a LoRA fine-tune
(finetune/finetune_lora.py) on pairs built here:

    system:   the "style" format's system prompt (writing/formats.json), the same one Chat uses
    user:     "Write a short post about: harbour, rain, bread"   (topic words taken from the post)
    answer:   the post itself

so the model learns to answer a topic with a post in that voice: length, tone, vocabulary, emoji
and hashtag habits. The result is the run runs/style-<id>; in Chat pick it with the format
"Write like a style profile". Starting from an existing fine-tuned run keeps its writing skills.
"""
import json, os, random, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import own_posts  # noqa: E402
from writing import formats as F  # noqa: E402

FORMAT = "style"
MIN_POSTS = 30
DEFAULT_BASE = "Qwen/Qwen3-0.6B"
KIND_WORD = {"x_post": "short post", "facebook_post": "Facebook post", "instagram_caption": "Instagram caption",
             "post": "post"}
_STOP = set("""about above after again against also been before being below between both could does doing down
during each from further have having here hers herself himself into itself just more most myself once only other
ought ours ourselves over same should some such than that their theirs them themselves then there these they this
those through under until very were what when where which while whom will with would your yours yourself
yourselves today really because going still thing things make made like know just want need every much many""".split())


def keywords(text, n=4):
    """A few topic words from a post, in order of appearance (hashtags count as words)."""
    words = re.findall(r"[^\W\d_][\w'-]{3,}", text.lower())
    out = []
    for w in words:
        w = w.strip("'-")
        if w not in _STOP and w not in out and w != "someone":
            out.append(w)
    freq = {w: words.count(w) for w in out}
    top = sorted(out, key=lambda w: (-freq[w], -len(w)))[:n]
    return [w for w in out if w in top]


def build_pairs(pid, path, seed=0):
    """Write the training pairs for a profile as JSONL (prompt, response, system). Returns their count."""
    posts = own_posts.load_posts(pid)
    if len(posts) < MIN_POSTS:
        raise ValueError(f"Only {len(posts)} posts; import at least {MIN_POSTS} to learn a style.")
    rng = random.Random(seed)
    system = F.system_prompt(FORMAT)
    n = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in posts:
            what = KIND_WORD.get(p.get("kind"), "post")
            kw = keywords(p["text"])
            req = f"Write a {what} about: {', '.join(kw)}" if kw and rng.random() > 0.15 else f"Write a {what}."
            f.write(json.dumps({"prompt": F.user_prompt(FORMAT, req), "response": p["text"], "system": system},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def run_name(pid):
    return f"style-{pid}"


def train_command(pid, py, runs_dir, start_from=None, epochs=3.0, gguf=True):
    """The finetune_lora.py command for a profile. start_from: a fine-tuned run's folder, or None
    to start from the default base model. Returns (cmd, out_dir)."""
    meta = own_posts.load_profile(pid) or {}
    if not meta:
        raise LookupError("No such profile.")
    out = os.path.join(runs_dir, run_name(pid))
    pairs = os.path.join(out, "style_pairs.jsonl")
    n = build_pairs(pid, pairs)
    base, init = DEFAULT_BASE, None
    if start_from:
        m = json.load(open(os.path.join(start_from, "meta.json")))
        if m.get("type") != "lora":
            raise ValueError("A style is learned on top of a fine-tuned model (LoRA). Pick a fine-tuned run, "
                             "or none to start from the base model.")
        base, init = m["base"], os.path.join(start_from, "adapter")
    steps = max(20, int(n * epochs / 8))   # batch 2 × grad-accum 4
    cmd = [py, "finetune/finetune_lora.py", "--out", out, "--base", base, "--extra", pairs, "--examples", "0",
           "--epochs", str(epochs), "--seq", "384", "--batch", "2", "--grad-accum", "4", "--lr", "2e-4",
           "--lora-r", "8", "--eval-every", str(max(5, steps // 10)), "--save-every", str(max(5, steps // 5)),
           "--meta-extra", json.dumps({"style": {"id": pid, "name": meta["name"], "posts": n}})]
    if init:
        cmd += ["--init-adapter", init]
    if gguf:
        cmd.append("--gguf")
    return cmd, out
