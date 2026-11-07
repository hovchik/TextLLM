"""Turn chat ratings into training examples."""
import json

from studio import db
from writing import formats as F

GOOD, BAD = 4, 2  # stars: >= GOOD is liked, <= BAD is disliked


def build(model):
    """Return (examples, message_ids) for one model's unused feedback.

    Each example: {"messages": [...context, user prompt], "chosen": str, "rejected": str | None}
    - liked answers and user-written corrections become "chosen" (learn to write like this)
    - disliked answers to the same prompt become "rejected" (learn to prefer chosen over this)
    """
    rows = db.q("SELECT * FROM messages WHERE role='assistant' AND model=? AND trained_in IS NULL "
                "AND (rating IS NOT NULL OR correction IS NOT NULL) ORDER BY id", (model,))
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    examples, ids = [], [r["id"] for r in rows]
    for pid, answers in by_prompt.items():
        prompt = db.q("SELECT * FROM messages WHERE id=?", (pid,), one=True)
        if not prompt:
            continue
        hist = db.history(prompt["conversation_id"], before_id=pid)
        if prompt.get("format") and prompt["format"] != "free":
            # train with the same system prompt and filled-in template Chat used
            fields = json.loads(prompt["fields"]) if prompt.get("fields") else None
            ctx = F.messages(prompt["format"], prompt["content"], fields, hist)
        else:
            ctx = hist + [{"role": "user", "content": prompt["content"]}]
        raw = None
        if F.get(prompt.get("format")).get("steps") == "fim":
            # Complete code: the model saw a raw fill-in-the-middle prompt, not a chat (feedback_train.py)
            from coding import fim

            raw = fim.psm(*fim.split_cursor(prompt["content"]))
        chosen, rejected = [], []
        for a in answers:
            if a["correction"]:
                chosen.append(a["correction"])
                if (a["rating"] or 3) < GOOD:
                    rejected.append(a["content"])
            elif a["rating"] is not None and a["rating"] >= GOOD:
                chosen.append(a["content"])
            elif a["rating"] is not None and a["rating"] <= BAD:
                rejected.append(a["content"])
        for i, c in enumerate(chosen):
            examples.append({"messages": ctx, "chosen": c, "rejected": rejected[i % len(rejected)] if rejected else None,
                             "format": prompt.get("format") or "free", **({"raw": raw} if raw else {})})
        if not chosen:
            for r in rejected:  # only negatives: still useful for preference training
                examples.append({"messages": ctx, "chosen": None, "rejected": r, "format": prompt.get("format") or "free",
                                 **({"raw": raw} if raw else {})})
    return examples, ids


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def mark_trained(ids, tag):
    for i in ids:
        db.ex("UPDATE messages SET trained_in=? WHERE id=?", (tag, i))


def export_liked(path, models=None):
    """Write liked or corrected answers as instruction examples (for data/build_instructions.py)."""
    sql = ("SELECT a.content, a.correction, a.rating, u.content prompt, u.format, u.fields FROM messages a "
           "JOIN messages u ON u.id=a.prompt_id WHERE a.role='assistant' "
           "AND (a.correction IS NOT NULL OR a.rating >= ?)")
    rows = db.q(sql, (GOOD,))
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            fmt = r["format"] or "free"
            fields = json.loads(r["fields"]) if r["fields"] else None
            f.write(json.dumps({"format": fmt, "system": F.system_prompt(fmt, fields),
                                "prompt": F.user_prompt(fmt, r["prompt"], fields),
                                "response": r["correction"] or r["content"],
                                "source": "studio-ratings", "license": "own"}, ensure_ascii=False) + "\n")
            n += 1
    return n
