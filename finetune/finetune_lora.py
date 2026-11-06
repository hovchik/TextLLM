"""Phase 2: LoRA fine-tune a small open chat model for posts, stories and essays.

Two kinds of examples are mixed:
  * writing instructions (data/build_instructions.py): a format's system prompt, a request and
    a good answer, so the model learns lengths, hooks, structure and tone per format;
  * book continuation ("Continue this story: ..."), capped by --book-share (default 20%), so it
    keeps a literary style.
You can also mix in your own examples (one JSON object per line: {"prompt": ..., "response": ...}).

CPU (default):  python finetune/finetune_lora.py --instructions data/instructions/train.jsonl --gguf
Colab GPU:      python finetune/finetune_lora.py --base Qwen/Qwen3-1.7B --batch 8 --seq 1024

CPU speed-ups: physical-core threads, bfloat16 autocast on CPUs with AVX-512 BF16 / AMX,
length-bucketed batches (little padding) and a cache of the tokenized examples.
Writes runs/<name>/adapter/ (LoRA weights + tokenizer), meta.json and metrics.jsonl;
--gguf also makes the fast quantized copy for chat (export_gguf.py).
"""
import argparse, contextlib, glob, hashlib, json, math, os, random, subprocess, sys, time

import torch
from peft import LoraConfig, get_peft_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from writing import hf  # noqa: E402

EOT = "<|endoftext|>"
PROMPTS = [
    "Continue this story:\n\n{p}",
    "Write the next part of this passage, keeping the same style:\n\n{p}",
    "Here is the start of a story. Continue it.\n\n{p}",
]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B", help="HF model id or local folder")
    ap.add_argument("--raw", default="data/raw", help="folder of .txt shards (books + your own texts)")
    ap.add_argument("--extra", default=None, help="optional JSONL with prompt/response pairs")
    ap.add_argument("--instructions", default=None,
                    help="JSONL from data/build_instructions.py (format, system, prompt, response)")
    ap.add_argument("--book-share", type=float, default=0.2,
                    help="share of book-continuation examples when --instructions is given (0 = none)")
    ap.add_argument("--gguf", action="store_true", help="make the fast quantized CPU copy after training")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512, help="total tokens per example")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0, help="overrides --epochs if > 0")
    ap.add_argument("--lora-r", type=int, default=8, help="8 for <1B and small sets; 16 for 2B+")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--init-adapter", default=None, help="continue training an existing LoRA adapter")
    ap.add_argument("--pairs-per-doc", type=int, default=8, help="max examples taken from one document")
    ap.add_argument("--min-doc-chars", type=int, default=5000)
    ap.add_argument("--step-offset", type=int, default=0, help="added to logged steps (continued runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def fmt(tok, prompt, response=None, system=None):
    """Return the prompt text (and full text) in the model's chat format (thinking off)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    p = hf.chat_text(tok, msgs)
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
    for pr in pairs:
        prompt, resp, system = (pr + (None,))[:3]
        p, full = fmt(tok, prompt, resp, system)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        fi = tok(full, add_special_tokens=False)["input_ids"][:seq]
        labels = [-100] * min(len(pi), len(fi)) + fi[len(pi):]  # loss only on the response
        if any(l != -100 for l in labels):
            out.append((fi, labels))
    return out


def batches(data, bs, pad, rng):
    """Length-bucketed batches: similar lengths together, so short posts aren't padded to essays."""
    while True:
        order = sorted(data, key=lambda x: len(x[0]) + rng.random())
        groups = [order[i:i + bs] for i in range(0, len(order) - bs + 1, bs)] or [order]
        rng.shuffle(groups)
        for chunk in groups:
            n = len(chunk)
            L = max(len(x) for x, _ in chunk)
            ids = torch.full((n, L), pad)
            lab = torch.full((n, L), -100)
            att = torch.zeros((n, L), dtype=torch.long)
            for j, (x, y) in enumerate(chunk):
                ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            yield ids, lab, att


def main():
    a = args()
    rng = random.Random(0)
    os.makedirs(a.out, exist_ok=True)
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a")

    threads = hf.set_threads()
    tok = hf.load_tokenizer(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = load_or_encode(a, tok, rng)
    rng.shuffle(data)
    n_val = max(4, len(data) // 50)
    val, train = data[:n_val], data[n_val:]
    print(f"{len(train)} train / {len(val)} val examples")

    dtype = torch.bfloat16 if a.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = hf.load_causal_lm(a.base, dtype).to(a.device)
    # On CPU the weights stay fp32 (LoRA needs fp32 master weights) and matmuls run in bf16 when
    # the CPU has native bf16 instructions; otherwise plain fp32.
    cpu_bf16 = a.device == "cpu" and hf.cpu_bf16_ok()
    autocast = (lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if cpu_bf16 else contextlib.nullcontext
    print(f"device {a.device} · {threads} threads · {'bf16 autocast' if cpu_bf16 else str(dtype).split('.')[-1]}")
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
        ls = []
        for x, y in val:
            with autocast():
                ls.append(model(input_ids=torch.tensor([x], device=a.device),
                                labels=torch.tensor([y], device=a.device)).loss.item())
        model.train()
        return sum(ls) / len(ls)

    def save():
        model.save_pretrained(os.path.join(a.out, "adapter"))
        tok.save_pretrained(os.path.join(a.out, "adapter"))
        json.dump({"type": "lora", "base": a.base, "lora_r": a.lora_r,
                   "instructions": bool(a.instructions), "book_share": a.book_share},
                  open(os.path.join(a.out, "meta.json"), "w"))

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
            with autocast():
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
    if a.gguf:
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "export_gguf.py"), "--run", a.out])
        if rc != 0:
            print("GGUF export failed; chat will use the adapter directly (slower). "
                  "Retry from the run's page with 'Make fast CPU copy'.")


def load_or_encode(a, tok, rng):
    """Build the example mix and tokenize it, cached on disk (re-tokenizing takes minutes)."""
    srcs = [a.instructions, a.extra] + sorted(glob.glob(os.path.join(a.raw, "**", "*.txt"), recursive=True))
    key = json.dumps([a.base, a.examples, a.seq, a.book_share, a.pairs_per_doc, a.min_doc_chars,
                      [(p, os.path.getmtime(p)) for p in srcs if p and os.path.exists(p)]])
    cache = os.path.join(ROOT, "data", "cache", hashlib.sha1(key.encode()).hexdigest()[:16] + ".pt")
    if os.path.exists(cache):
        print(f"using cached tokenized examples ({os.path.relpath(cache, ROOT)})")
        return torch.load(cache)
    pairs = []
    if a.instructions:
        rows = [json.loads(l) for l in open(a.instructions, encoding="utf-8") if l.strip()]
        pairs += [(r["prompt"], r["response"], r.get("system")) for r in rows]
        print(f"{len(rows)} writing instructions from {a.instructions}")
    n_books = a.examples if not a.instructions else \
        min(a.examples, int(round(len(pairs) * a.book_share / max(1e-6, 1 - a.book_share))))
    if n_books > 0 and glob.glob(os.path.join(a.raw, "**", "*.txt"), recursive=True):
        print(f"building {n_books} book-continuation examples...")
        pairs += book_pairs(tok, a.raw, n_books, a.seq, rng, a.pairs_per_doc, a.min_doc_chars)
    if a.extra:
        pairs += [(j["prompt"], j["response"], j.get("system"))
                  for j in map(json.loads, open(a.extra, encoding="utf-8")) if j]
    if not pairs:
        raise SystemExit("no training examples: give --instructions or put .txt files under --raw")
    data = encode(tok, pairs, a.seq)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(data, cache)
    return data


if __name__ == "__main__":
    main()
