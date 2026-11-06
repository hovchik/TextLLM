"""Writing formats: system prompts, request templates, sampling and automatic checks.

The formats live in writing/formats.json so platform limits can be changed without code.
The same system prompts are used in Chat, in the instruction data (data/build_instructions.py)
and in evaluation (evaluate.py), so training and use match.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = None


def config():
    global _CFG
    if _CFG is None:
        _CFG = json.load(open(os.path.join(HERE, "formats.json"), encoding="utf-8"))
    return _CFG


def formats():
    return config()["formats"]


def get(fmt):
    return formats().get(fmt or "free") or formats()["free"]


def public_list():
    """What the web app needs to draw the format picker."""
    labels = config()["field_labels"]
    out = []
    for k, f in formats().items():
        out.append({"id": k, "label": f["label"], "group": f["group"],
                    "fields": [{"id": x, "label": labels.get(x, x)} for x in f.get("fields", [])],
                    "sampling": f["sampling"], "chars": f.get("chars"), "words": f.get("words")})
    return out


def _words_target(f, fields):
    try:
        n = int(re.sub(r"[^\d]", "", str((fields or {}).get("length", ""))) or 0)
    except ValueError:
        n = 0
    if n:
        return n
    lo, hi = f.get("words", [0, 0])
    return (lo + hi) // 2 if hi else 0


def _hashtags(f, fields):
    lo, hi = f.get("hashtags", [0, 0])
    v = str((fields or {}).get("hashtags", "")).strip()
    if v.isdigit():
        lo = hi = min(int(v), f.get("max_hashtags", 30))
    return lo, hi


def system_prompt(fmt, fields=None):
    f = get(fmt)
    lo, hi = _hashtags(f, fields)
    vals = {"target_chars": f.get("target_chars", ""), "hashtags_min": lo, "hashtags_max": hi,
            "hook_words": f.get("hook_words", ""), "words": _words_target(f, fields)}
    return f["system"].format(**vals)


def user_prompt(fmt, message, fields=None):
    """The request plus the filled-in form fields, as one user message."""
    f = get(fmt)
    labels = config()["field_labels"]
    lines = [message.strip()]
    extra = [(labels.get(k, k), str(v).strip()) for k, v in (fields or {}).items()
             if k in f.get("fields", []) and str(v).strip()]
    if extra:
        lines.append("")
        lines += [f"{k}: {v}" for k, v in extra]
    return "\n".join(lines)


def messages(fmt, message, fields=None, history=None):
    """Chat messages for a request in a format: system + history + user."""
    out = [{"role": "system", "content": system_prompt(fmt, fields)}]
    out += [m for m in (history or []) if m.get("role") in ("user", "assistant")]
    out.append({"role": "user", "content": user_prompt(fmt, message, fields)})
    return out


def outline_messages(fmt, message, fields=None):
    f = get(fmt)
    req = user_prompt(fmt, message, fields)
    return [{"role": "system", "content": "You plan writing. Answer with the plan only."},
            {"role": "user", "content": f["outline_prompt"].format(request=req)}]


def with_outline(msgs, outline):
    """Add the plan to the last user message so the writer follows it."""
    out = [dict(m) for m in msgs]
    out[-1]["content"] += f"\n\nFollow this plan:\n{outline.strip()}\n\nNow write the full text."
    return out


# ---------------------------------------------------------------- automatic checks
HASHTAG = re.compile(r"(?<!\w)#\w+")
PREAMBLE = re.compile(r"^(sure|here('| i)s|certainly|of course|okay)[^\n]{0,60}[:!.]\s*", re.I)


def clean(text):
    """Strip things small models add around a post: a 'Sure, here is...' line and wrapping quotes."""
    t = PREAMBLE.sub("", text.strip())
    if len(t) > 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t


def _distinct2(words):
    if len(words) < 3:
        return 1.0
    grams = list(zip(words, words[1:]))
    return len(set(grams)) / len(grams)


def check(fmt, text, fields=None):
    """Return {"score": 0-100, "checks": {name: bool}, "stats": {...}} for one answer."""
    f = get(fmt)
    t = text.strip()
    words = re.findall(r"\w+", t.lower())
    lines = [l for l in t.splitlines() if l.strip()]
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 20]
    tags = HASHTAG.findall(t)
    d2 = _distinct2(words)
    c = {"not_empty": len(words) >= 3,
         "no_repetition": d2 >= 0.6 and len(sents) == len(set(sents))}
    if f.get("chars"):
        c["length"] = 0 < len(t) <= f["chars"]
    if f.get("words"):
        target = _words_target(f, fields)
        c["length"] = 0.5 * target <= len(words) <= 1.6 * target
    if "hashtags" in f:
        lo, hi = _hashtags(f, fields)
        c["hashtags"] = lo <= len(tags) <= hi
    if f.get("hook_words") and lines:
        c["hook"] = len(lines[0].split()) <= f["hook_words"]
    if f.get("ends_with_question") and lines:
        body = [l for l in lines if not l.strip().startswith("#")]
        c["ends_with_question"] = bool(body) and body[-1].rstrip().endswith("?")
    if f.get("steps") == "outline":
        c["paragraphs"] = len([p for p in re.split(r"\n\s*\n", t) if p.strip()]) >= 3
    c["no_preamble"] = not PREAMBLE.match(t)
    score = round(100 * sum(c.values()) / len(c), 1)
    return {"score": score, "checks": c,
            "stats": {"chars": len(t), "words": len(words), "hashtags": len(tags), "distinct2": round(d2, 3)}}
