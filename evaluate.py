"""Score a model on the fixed writing prompt set (writing/eval_prompts.jsonl).

    python evaluate.py --run runs/my-qwen-writer               # all 60 prompts
    python evaluate.py --run runs/my-qwen-writer --quick       # 8 prompts, a few minutes on CPU
    python evaluate.py --run runs/my-qwen-writer --formats x_post,linkedin_post
    python evaluate.py --pairs runs/new-model runs/old-model   # blind A/B pairs for the Analytics page
    python evaluate.py --pairs runs/new runs/old --judge Qwen/Qwen3-4B   # let a bigger model vote

Automatic checks (writing/formats.py): length and hashtag limits, hook length, closing question,
paragraphs, repetition, no "Sure, here is" preamble. Each answer gets 0-100; the run's score is
the mean per format and overall. Results go to runs/<run>/eval/<time>.json and history.jsonl.
If data/instructions/val.jsonl exists, the loss on it is reported per format too.
"""
import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
PROMPTS = os.path.join(ROOT, "writing", "eval_prompts.jsonl")
QUICK = ["x_post-01", "x_post-03", "threads_post-01", "linkedin_post-01", "instagram_caption-01",
         "facebook_post-01", "story-16", "essay-05"]


def say(m):
    print(m, flush=True)


def load_prompts(formats=None, quick=False, limit=0):
    ps = [json.loads(l) for l in open(PROMPTS, encoding="utf-8") if l.strip()]
    if quick:
        ps = [p for p in ps if p["id"] in QUICK]
    if formats:
        ps = [p for p in ps if p["format"] in formats]
    return ps[:limit] if limit else ps


def summarize(items):
    by = {}
    for it in items:
        by.setdefault(it["format"], []).append(it["check"]["score"])
    per = {k: round(sum(v) / len(v), 1) for k, v in by.items()}
    checks = {}
    for it in items:
        for k, ok in it["check"]["checks"].items():
            checks.setdefault(k, []).append(1 if ok else 0)
    return {"score": round(sum(i["check"]["score"] for i in items) / max(1, len(items)), 1),
            "per_format": per, "per_check": {k: round(100 * sum(v) / len(v), 1) for k, v in checks.items()},
            "n": len(items)}


def val_loss(eng):
    """Mean response loss per format on data/instructions/val.jsonl (fine-tuned models, HF backend)."""
    p = os.path.join(ROOT, "data", "instructions", "val.jsonl")
    if eng.backend != "hf" or not os.path.exists(p):
        return None
    import torch
    from writing import hf

    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()][:200]
    by = {}
    with torch.no_grad():
        for r in rows:
            msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + \
                   [{"role": "user", "content": r["prompt"]}]
            pi = eng.tok(hf.chat_text(eng.tok, msgs), add_special_tokens=False)["input_ids"]
            ri = eng.tok(r["response"] + (eng.tok.eos_token or ""), add_special_tokens=False)["input_ids"]
            ids = (pi + ri)[:1024]
            labels = ([-100] * len(pi) + ri)[:1024]
            if all(l == -100 for l in labels):
                continue
            x = torch.tensor([ids], device=eng.device)
            y = torch.tensor([labels], device=eng.device)
            by.setdefault(r.get("format", "free"), []).append(eng.model(input_ids=x, labels=y).loss.item())
    return {k: round(sum(v) / len(v), 4) for k, v in by.items()}


def run_eval(run, formats=None, quick=False, limit=0, seed=0, max_tokens=None, out_dir=None, engine=None):
    import torch
    from chat_engine import Engine

    eng = engine or Engine(run)
    prompts = load_prompts(formats, quick, limit)
    say(f"== evaluating {os.path.basename(run)} ({eng.backend}) on {len(prompts)} prompts")
    items, t0 = [], time.time()
    for i, p in enumerate(prompts, 1):
        torch.manual_seed(seed + i)
        random.seed(seed + i)
        samp = {"max_tokens": max_tokens} if max_tokens else None
        outline, final = None, None
        for ev in eng.write(p["format"], p["message"], p.get("fields"), sampling=samp):
            if ev["type"] == "outline":
                outline = ev["text"]
            if ev["type"] == "final":
                final = ev
        items.append({"id": p["id"], "format": p["format"], "message": p["message"], "fields": p.get("fields"),
                      "text": final["text"], "outline": outline, "check": final["check"]})
        say(f"[{i}/{len(prompts)}] {p['id']}: {final['check']['score']:.0f}")
    summary = summarize(items)
    summary["val_loss"] = val_loss(eng)
    partial = bool(formats or limit)  # a subset run is not comparable; the gate ignores it
    res = {"run": os.path.basename(run), "time": time.time(), "seconds": round(time.time() - t0),
           "backend": eng.backend, "quick": quick, "partial": partial, "max_tokens": max_tokens or None,
           "summary": summary, "items": items}
    d = out_dir or os.path.join(run, "eval")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json.dump(res, open(os.path.join(d, f"{stamp}.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(d, "history.jsonl"), "a") as h:
        h.write(json.dumps({"time": res["time"], "file": f"{stamp}.json", "quick": quick, "partial": partial,
                            "backend": eng.backend, "max_tokens": max_tokens or None, **summary}) + "\n")
    say(f"== score {summary['score']} · " + " · ".join(f"{k} {v}" for k, v in summary["per_format"].items()))
    return res


def latest(run, quick=None):
    d = os.path.join(run, "eval")
    h = os.path.join(d, "history.jsonl")
    if not os.path.exists(h):
        return None
    rows = [r for r in (json.loads(l) for l in open(h) if l.strip()) if not r.get("partial")]
    if quick is not None:
        rows = [r for r in rows if r.get("quick") == quick]
    return json.load(open(os.path.join(d, rows[-1]["file"]))) if rows else None


# ---------------------------------------------------------------- pairwise
RUBRIC = ("You are judging two pieces of writing for the same request. Judge fit to the request and format, "
          "readability, originality and tone. Answer with exactly one letter: A, B, or T for a tie.")


def make_judge(name):
    """A callable(request, a, b) -> 'a'|'b'|'tie' using a larger model."""
    from writing import hf

    if name.endswith(".gguf"):
        from llama_cpp import Llama

        llm = Llama(model_path=name, n_ctx=8192, n_threads=hf.physical_cores(), verbose=False)

        def ask(msgs):
            return llm.create_chat_completion(messages=msgs, max_tokens=3, temperature=0)["choices"][0]["message"]["content"]
    else:
        import torch

        tok = hf.load_tokenizer(name)
        model = hf.load_causal_lm(name).eval()

        def ask(msgs):
            ids = tok(hf.chat_text(tok, msgs), return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=3, do_sample=False)
            return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def judge(request, a, b):
        votes = []
        for x, y, flip in ((a, b, False), (b, a, True)):  # both orders, to cancel position bias
            msg = f"Request:\n{request}\n\n=== A ===\n{x}\n\n=== B ===\n{y}\n\nWhich is better? A, B or T?"
            r = ask([{"role": "system", "content": RUBRIC}, {"role": "user", "content": msg}]).strip().upper()[:1]
            v = {"A": "a", "B": "b"}.get(r, "tie")
            votes.append({"a": "b", "b": "a"}.get(v, v) if flip else v)
        return votes[0] if votes[0] == votes[1] else "tie"

    return judge


def make_pairs(run_a, run_b, judge=None, quick=False):
    """Pair the latest evals of two runs prompt by prompt and store them for blind voting."""
    from studio import db

    from chat_engine import weights_stamp

    def usable(run):  # a full-length evaluation of the run's current weights
        e = latest(run, quick)
        return e if e and not e.get("max_tokens") and e["time"] >= weights_stamp(run) else None

    ea, eb = usable(run_a), usable(run_b)
    for r, e in ((run_a, ea), (run_b, eb)):
        if not e:
            say(f"{os.path.basename(r)} has no evaluation of its current weights; running it first")
    ea = ea or run_eval(run_a, quick=quick)
    eb = eb or run_eval(run_b, quick=quick)
    bi = {i["id"]: i for i in eb["items"]}
    j = make_judge(judge) if judge else None
    n = 0
    for ia in ea["items"]:
        ib = bi.get(ia["id"])
        if not ib:
            continue
        pid = db.add_pair(ia["id"], ia["format"], ia["message"], os.path.basename(run_a), os.path.basename(run_b),
                          ia["text"], ib["text"])
        if j:
            from writing import formats as F
            v = j(F.user_prompt(ia["format"], ia["message"], ia.get("fields")), ia["text"], ib["text"])
            db.vote_pair(pid, v, f"judge:{os.path.basename(judge)}")
            say(f"{ia['id']}: {v}")
        n += 1
    say(f"== {n} pairs ready" + ("" if j else " for blind voting on the Analytics page"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--quick", action="store_true", help="8 prompts instead of 60")
    ap.add_argument("--formats", default="", help="comma-separated format ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="cap answer length (faster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", nargs=2, metavar=("NEW", "CURRENT"))
    ap.add_argument("--judge", help="HF model id or .gguf file that votes on the pairs")
    a = ap.parse_args()
    if a.pairs:
        make_pairs(*[os.path.abspath(p) for p in a.pairs], judge=a.judge, quick=a.quick)
    elif a.run:
        run_eval(os.path.abspath(a.run), [f for f in a.formats.split(",") if f] or None, a.quick, a.limit,
                 a.seed, a.max_tokens or None)
    else:
        ap.error("give --run or --pairs")


if __name__ == "__main__":
    main()
