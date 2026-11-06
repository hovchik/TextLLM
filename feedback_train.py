"""Train a model on the ratings people gave its chat answers.

  python feedback_train.py --run runs/finetune-qwen
  python feedback_train.py --run runs/scratch-cpu

Fine-tuned (LoRA) models learn three ways:
  * liked answers and your written corrections → supervised training ("write like this")
  * a liked/corrected answer vs a disliked one to the same prompt → DPO preference loss
  * a disliked answer with nothing better → pushed below the model's previous likelihood
From-scratch models only continue text, so they continue-train on prompt + liked answer.

Feedback that was used is marked so it isn't trained on twice.

--gate (on for automatic training): score the model on a quick evaluation before and after; if
any format's automatic-check score drops by more than 5 points, the previous adapter is restored
so a bad batch of ratings cannot quietly make the model worse. If the run had a fast GGUF copy,
it is rebuilt after training.
"""
import argparse, json, math, os, random, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def say(m):
    print(m, flush=True)


def last_step(run):
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    s = [json.loads(l).get("step", 0) for l in open(p) if l.strip()]
    return max(s) if s else 0


def train_lora(a, meta, examples):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from writing import hf

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hf.set_threads()
    ad = os.path.join(a.run, "adapter")
    tok = hf.load_tokenizer(ad)
    model = hf.load_causal_lm(meta["base"]).to(dev)
    model = PeftModel.from_pretrained(model, ad, is_trainable=True)
    model.train()

    def prompt_text(msgs):
        return hf.chat_text(tok, msgs)

    def encode(msgs, answer):
        p = tok(prompt_text(msgs), add_special_tokens=False)["input_ids"][-(a.seq // 2):]
        r = tok(answer + (tok.eos_token or ""), add_special_tokens=False)["input_ids"][: a.seq - len(p)]
        ids = torch.tensor([p + r], device=dev)
        labels = torch.tensor([[-100] * len(p) + r], device=dev)
        return ids, labels

    def seq_logp(m, ids, labels):
        logits = m(input_ids=ids).logits[:, :-1].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        lp = torch.gather(F.log_softmax(logits, -1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (lp * mask).sum(), mask.sum()

    def ref_logp(ids, labels):
        with torch.no_grad(), model.disable_adapter():
            return seq_logp(model, ids, labels)[0]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    offset = last_step(a.run)
    steps = max(1, math.ceil(len(examples) * a.epochs))
    mlog = open(os.path.join(a.run, "metrics.jsonl"), "a")
    rng = random.Random(0)
    order = []
    t0 = time.time()
    for step in range(1, steps + 1):
        if not order:
            order = examples[:]
            rng.shuffle(order)
        e = order.pop()
        loss, parts = 0.0, {}
        if e.get("chosen"):
            ids, lab = encode(e["messages"], e["chosen"])
            lp_c, n = seq_logp(model, ids, lab)
            sft = -lp_c / max(1, n)
            loss = loss + sft
            parts["sft"] = sft.item()
            if e.get("rejected"):
                ids_r, lab_r = encode(e["messages"], e["rejected"])
                lp_r, _ = seq_logp(model, ids_r, lab_r)
                margin = a.beta * ((lp_c - ref_logp(ids, lab)) - (lp_r - ref_logp(ids_r, lab_r)))
                dpo = -F.logsigmoid(margin)
                loss = loss + dpo
                parts["dpo"] = dpo.item()
        elif e.get("rejected"):
            ids_r, lab_r = encode(e["messages"], e["rejected"])
            lp_r, _ = seq_logp(model, ids_r, lab_r)
            neg = -F.logsigmoid(-a.beta * (lp_r - ref_logp(ids_r, lab_r)))
            loss = loss + neg
            parts["neg"] = neg.item()
        else:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        say(f"step {step}/{steps} | loss {loss.item():.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()))
        mlog.write(json.dumps({"time": time.time(), "max_steps": offset + steps, "kind": "feedback",
                               "step": offset + step, "loss": loss.item(), "lr": a.lr,
                               "tokens_per_sec": 0, **parts}) + "\n")
        mlog.flush()
    model.save_pretrained(ad)
    say(f"saved adapter ({time.time() - t0:.0f}s)")


def train_scratch(a, examples):
    tmp = tempfile.mkdtemp(prefix="feedback_")
    path = os.path.join(tmp, "feedback.txt")
    texts = [e["messages"][-1]["content"].strip() + " " + e["chosen"].strip() for e in examples if e.get("chosen")]
    if not texts:
        say("no liked answers to learn from (from-scratch models can't use dislikes alone)")
        return True
    # Liked exchanges, separated by blank lines. Very small sets are repeated so there is
    # at least one full training window of text.
    body = "\n\n".join(texts)
    while len(body) < 4000:
        body += "\n\n" + "\n\n".join(texts)
    open(path, "w", encoding="utf-8").write(body)
    rc = subprocess.call([sys.executable, "learn.py", "--run", a.run, "--files", path,
                          "--passes", str(max(1, int(a.epochs))), "--delete-originals"], cwd=ROOT)
    return rc == 0


QUICK_TOKENS = 300  # same cap before and after, so long formats are compared like for like


def quick_score(run, fresh=False):
    """Summary of a quick evaluation; reuses the last one if the weights haven't changed since."""
    import evaluate
    from chat_engine import Engine, weights_stamp

    last = evaluate.latest(run, quick=True)
    if (last and not fresh and last.get("backend") == "hf" and last["time"] >= weights_stamp(run)
            and last.get("max_tokens") == QUICK_TOKENS):
        return last["summary"]
    # Short answers keep this to a few minutes on CPU; the HF backend sees the new adapter at once.
    return evaluate.run_eval(run, quick=True, max_tokens=QUICK_TOKENS, engine=Engine(run, backend="hf"))["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", help="JSONL from studio.feedback (default: build it from the chat database)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO strength")
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--gate", action="store_true", help="undo the training if quick checks get worse")
    a = ap.parse_args()
    a.run = os.path.abspath(a.run)
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    run_name = os.path.basename(a.run)

    from studio import feedback

    if a.data:
        examples = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
        ids = []
    else:
        examples, ids = feedback.build(run_name)
    liked = sum(1 for e in examples if e.get("chosen"))
    pairs = sum(1 for e in examples if e.get("chosen") and e.get("rejected"))
    say(f"== feedback for {run_name}: {len(examples)} examples ({liked} liked/corrected, {pairs} preference pairs)")
    if not examples:
        say("nothing to train on yet: rate some answers in Chat first")
        return
    if meta["type"] == "lora":
        from chat_engine import gguf_path

        had_gguf = bool(gguf_path(a.run))
        before = quick_score(a.run) if a.gate else None
        ad, prev = os.path.join(a.run, "adapter"), os.path.join(a.run, "adapter.prev")
        shutil.rmtree(prev, ignore_errors=True)
        shutil.copytree(ad, prev)
        ok = train_lora(a, meta, examples) is None
        if ok and a.gate:
            after = quick_score(a.run, fresh=True)
            from studio.gate import check_drops

            drops = check_drops(after, before)
            if drops:
                say("⚠️ quick checks got worse (" + ", ".join(f"{k}: {c} → {n}" for k, (n, c) in drops.items()) +
                    "); restoring the previous adapter. These ratings are marked 'gate-rejected' and won't be "
                    "retried until someone rates them again.")
                shutil.rmtree(ad)
                shutil.copytree(prev, ad)
                if ids:
                    feedback.mark_trained(ids, f"gate-rejected@{time.strftime('%Y-%m-%d %H:%M')}")
                sys.exit(3)
            say(f"gate passed: quick score {before and before['score']} → {after['score']}")
        if ok and had_gguf:
            subprocess.call([sys.executable, os.path.join(ROOT, "export_gguf.py"), "--run", a.run], cwd=ROOT)
    else:
        ok = train_scratch(a, examples)
    if ok and ids:
        feedback.mark_trained(ids, f"{run_name}@{time.strftime('%Y-%m-%d %H:%M')}")
    say("✅ done. The model has learned from your ratings." if ok else "training failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
