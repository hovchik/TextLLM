"""Turn chat ratings into training examples."""
import json

from studio import db

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
        ctx = db.history(prompt["conversation_id"], before_id=pid) + [{"role": "user", "content": prompt["content"]}]
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
            examples.append({"messages": ctx, "chosen": c, "rejected": rejected[i % len(rejected)] if rejected else None})
        if not chosen:
            for r in rejected:  # only negatives: still useful for preference training
                examples.append({"messages": ctx, "chosen": None, "rejected": r})
    return examples, ids


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def mark_trained(ids, tag):
    for i in ids:
        db.ex("UPDATE messages SET trained_in=? WHERE id=?", (tag, i))
