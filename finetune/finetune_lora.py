"""Phase 2: LoRA fine-tune a small open chat model on the book corpus.

The model is taught to continue book passages in chat form ("Continue this story: ..."),
so it picks up literary style but keeps the base model's ability to follow requests.
You can also mix in your own examples (one JSON object per line: {"prompt": ..., "response": ...}).

CPU (default):  python finetune/finetune_lora.py
Colab GPU:      python finetune/finetune_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --batch 8 --seq 1024

Writes runs/<name>/adapter/ (LoRA weights + tokenizer), meta.json and metrics.jsonl.
"""
import argparse, glob, json, math, os, random, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

EOT = "<|endoftext|>"
PROMPTS = [
    "Continue this story:\n\n{p}",
    "Write the next part of this passage, keeping the same style:\n\n{p}",
    "Here is the start of a story. Continue it.\n\n{p}",
]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id or local folder")
    ap.add_argument("--raw", default="data/raw", help="folder of .txt shards (books + your own texts)")
    ap.add_argument("--extra", default=None, help="optional JSONL with prompt/response pairs")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512, help="total tokens per example")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="overrides --epochs if > 0")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--init-adapter", default=None, help="continue training an existing LoRA adapter")
    ap.add_argument("--pairs-per-doc", type=int, default=8, help="max examples taken from one document")
    ap.add_argument("--min-doc-chars", type=int, default=5000)
    ap.add_argument("--step-offset", type=int, default=0, help="added to logged steps (continued runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def fmt(tok, prompt, response=None):
    """Return the prompt text (and full text) in the model's chat format."""
    msgs = [{"role": "user", "content": prompt}]
    if tok.chat_template:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        p = f"### Request:\n{prompt}\n\n### Response:\n"
    return p if response is None else (p, p + response + (tok.eos_token or ""))


def book_pairs(tok, raw, n, seq, rng, per_doc=8, min_chars=5000):
    files = sorted(glob.glob(os.path.join(raw, "**", "*.txt"), recursive=True))
    if not files:
        raise SystemExit(f"no .txt shards under {raw} (run data/build_corpus.py first)")
    rng.shuffle(files)
    pairs, ctx, cont = [], int(seq * 0.4), int(seq * 0.5)
    # Short texts (articles, posts) get a smaller window so they still yield examples.
    for f in files:
        docs = [d for d in open(f, encoding="utf-8").read().split(EOT) if len(d.strip()) > min_chars]
        rng.shuffle(docs)
        for d in docs:
            ids = tok(d[:400_000], add_special_tokens=False)["input_ids"]
            c, k = ctx, cont
            if len(ids) <= c + k + 1:
                c, k = int(len(ids) * 0.4), int(len(ids) * 0.55)
                if c < 32:
                    continue
            for _ in range(min(per_doc, max(1, len(ids) // (c + k)))):
                i = rng.randrange(0, len(ids) - c - k)
                pairs.append((rng.choice(PROMPTS).format(p=tok.decode(ids[i:i + c]).strip()),
                              tok.decode(ids[i + c:i + c + k]).strip()))
                if len(pairs) >= n:
                    return pairs
    return pairs


def encode(tok, pairs, seq):
    out = []
    for prompt, resp in pairs:
        p, full = fmt(tok, prompt, resp)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        fi = tok(full, add_special_tokens=False)["input_ids"][:seq]
        labels = [-100] * min(len(pi), len(fi)) + fi[len(pi):]  # loss only on the response
        if any(l != -100 for l in labels):
            out.append((fi, labels))
    return out


def batches(data, bs, pad, rng):
    while True:
        rng.shuffle(data)
        for i in range(0, len(data) - bs + 1, bs):
            chunk = data[i:i + bs]
            L = max(len(x) for x, _ in chunk)
            ids = torch.full((bs, L), pad)
            lab = torch.full((bs, L), -100)
            att = torch.zeros((bs, L), dtype=torch.long)
            for j, (x, y) in enumerate(chunk):
                ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            yield ids, lab, att


def main():
    a = args()
    rng = random.Random(0)
    os.makedirs(a.out, exist_ok=True)
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a")

    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print("building examples from books...")
    pairs = book_pairs(tok, a.raw, a.examples, a.seq, rng, a.pairs_per_doc, a.min_doc_chars)
    if a.extra:
        pairs += [(j["prompt"], j["response"]) for j in map(json.loads, open(a.extra, encoding="utf-8"))]
    data = encode(tok, pairs, a.seq)
    rng.shuffle(data)
    n_val = max(4, len(data) // 50)
    val, train = data[:n_val], data[n_val:]
    print(f"{len(train)} train / {len(val)} val examples")

    dtype = torch.bfloat16 if a.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype).to(a.device)
    if a.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()

    steps = a.max_steps or max(1, int(len(train) * a.epochs / (a.batch * a.grad_accum)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, steps // 20)
    lr_at = lambda s: a.lr * min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))

    def log(**kw):
        if "step" in kw:
            kw["step"] += a.step_offset
        mlog.write(json.dumps({"time": time.time(), "max_steps": steps + a.step_offset, **kw}) + "\n")
        mlog.flush()

    @torch.no_grad()
    def evaluate():
        model.eval()
        ls = [model(input_ids=x.to(a.device), attention_mask=m.to(a.device), labels=y.to(a.device)).loss.item()
              for x, y, m in [next(batches(list(val), 1, pad, random.Random(i))) for i in range(len(val))]]
        model.train()
        return sum(ls) / len(ls)

    def save():
        model.save_pretrained(os.path.join(a.out, "adapter"))
        tok.save_pretrained(os.path.join(a.out, "adapter"))
        json.dump({"type": "lora", "base": a.base}, open(os.path.join(a.out, "meta.json"), "w"))

    log(kind="val", step=0, loss=evaluate())
    it, t0, model_tokens = batches(train, a.batch, pad, rng), time.time(), 0
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        total = 0.0
        for _ in range(a.grad_accum):
            x, y, m = next(it)
            model_tokens += int(m.sum())
            loss = model(input_ids=x.to(a.device), attention_mask=m.to(a.device), labels=y.to(a.device)).loss
            (loss / a.grad_accum).backward()
            total += loss.item() / a.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.time() - t0
        t0 = time.time()
        print(f"step {step}/{steps} | loss {total:.3f} | {dt:.1f} s/step")
        log(kind="train", step=step, loss=total, lr=lr_at(step), tokens_per_sec=model_tokens / dt)
        model_tokens = 0
        if step % a.eval_every == 0 or step == steps:
            vl = evaluate()
            print(f"== step {step} val loss {vl:.3f}")
            log(kind="val", step=step, loss=vl)
        if step % a.save_every == 0 or step == steps:
            save()
    print(f"saved adapter to {a.out}/adapter")


if __name__ == "__main__":
    main()
