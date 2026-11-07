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
Writes runs/<name>/adapter/ (LoRA weights + tokenizer), train_state.pt (optimizer, step, batch
position), examples.pt (the tokenized examples), meta.json and metrics.jsonl; --gguf also makes the fast quantized copy for chat (export_gguf.py).
Stopped, or moved to another PC? Run the same command again with --resume to continue from the last save.

Copilot (code) models: --code data/code adds examples cut from source files (data/build_code.py):
  * fill-in-the-middle (--fim-rate): <|fim_prefix|>before<|fim_suffix|>after<|fim_middle|> → the missing code,
    in the raw format code models use for editor completion;
  * plain continuation: the code so far → the next lines;
  * chat (--code-chat-share): "Complete this Python code: ..." with the Code assistant system prompt,
    so the model also helps in Chat.
  python finetune/finetune_lora.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --code data/code --examples 0 \
      --code-examples 4000 --seq 1024 --out runs/my-copilot
"""
import argparse, contextlib, glob, hashlib, json, math, os, random, shutil, subprocess, sys, time

import torch
from peft import LoraConfig, get_peft_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import devices  # noqa: E402
import graceful  # noqa: E402
from coding import fim, langs as L  # noqa: E402
from writing import formats as F  # noqa: E402
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
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/adapter + <out>/train_state.pt (optimizer, step, schedule)")
    ap.add_argument("--meta-extra", default=None, help="JSON object merged into meta.json (e.g. a style profile)")
    ap.add_argument("--private-texts", action="store_true",
                    help="learn.py: the texts are temporary, so keep no tokenized copy (no data/cache, no "
                         "examples.pt) and leave the run's resume state (train_state.pt) as it was")
    ap.add_argument("--device", default="auto", help="auto (GPU if there is one), cuda, mps, xpu or cpu")
    cg = ap.add_argument_group("copilot (code) examples")
    cg.add_argument("--code", default=None, help="folder of code shards from data/build_code.py (e.g. data/code)")
    cg.add_argument("--code-examples", type=int, default=4000, help="examples cut from the code")
    cg.add_argument("--fim-rate", type=float, default=0.5, help="share of code examples that are fill-in-the-middle")
    cg.add_argument("--code-chat-share", type=float, default=0.2,
                    help="share of code examples in chat form (Chat → Code assistant)")
    cg.add_argument("--code-langs", default="", help="only these languages, e.g. python,typescript (empty = all)")
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


def code_docs(code_dir, rng, langs):
    """Yield code documents (one source file each) from the shards under code_dir, in random order."""
    files = sorted(glob.glob(os.path.join(code_dir, "**", "*.txt"), recursive=True))
    if not files:
        raise SystemExit(f"no code under {code_dir} (collect some first: Training → Copilot, or data/build_code.py)")
    rng.shuffle(files)
    for f in files:
        docs = [d.strip("\n") for d in open(f, encoding="utf-8").read().split(EOT) if d.strip()]
        rng.shuffle(docs)
        for d in docs:
            if L.is_code_doc(d) and (not langs or L.wanted(L.doc_lang(d) or "", langs)):
                yield d


def code_window(doc, rng, max_chars):
    """A piece of a long file that fits the example: the header line plus whole lines from a random start."""
    head, _, body = doc.partition("\n")
    if len(body) <= max_chars:
        return doc
    lines = body.splitlines(keepends=True)
    i = rng.randrange(0, len(lines))
    while i > 0 and sum(map(len, lines[i:])) < max_chars // 2:
        i -= 1
    out, n = [], 0
    for ln in lines[i:]:
        if n + len(ln) > max_chars and out:
            break
        out.append(ln)
        n += len(ln)
    return head + "\n" + "".join(out)


def code_pairs(tok, code_dir, n, seq, rng, fim_rate=0.5, chat_share=0.2, langs=None, per_doc=8):
    """Examples for a copilot: fill-in-the-middle, continuation (both raw) and chat-form completion."""
    has_fim = fim.token_ids(tok) is not None
    if fim_rate > 0 and not has_fim:
        print("this base model's tokenizer has no fill-in-the-middle tokens; using continuation examples instead")
    system = F.system_prompt("code")
    max_chars = int(seq * 2.6)  # ~3 characters per token in code, keeping room for the answer
    out = []
    for doc in code_docs(code_dir, rng, langs):
        for _ in range(min(per_doc, max(1, len(doc) // max(1, max_chars)))):
            w = code_window(doc, rng, max_chars)
            r = rng.random()
            if r < chat_share:
                head, _, body = w.partition("\n")
                lines = body.splitlines(keepends=True)
                if len(lines) < 4:
                    continue
                cut = rng.randrange(max(1, len(lines) // 3), len(lines) - 1)
                k = rng.randint(1, 15)
                lang = L.doc_lang(w) or "code"
                f = L.fence(lang)
                before, nxt = "".join(lines[:cut]).rstrip("\n"), "".join(lines[cut:cut + k]).rstrip("\n")
                if not nxt.strip():
                    continue
                prompt = (f"Complete this {lang} code from {L.doc_path(w)}. Reply with only the lines that come "
                          f"next.\n\n```{f}\n{before}\n```")
                out.append((prompt, f"```{f}\n{nxt}\n```", system, False))
            elif has_fim and r < chat_share + (1 - chat_share) * fim_rate:
                pre, mid, suf = fim.split(w, rng, max_middle=max_chars // 3)
                if mid.strip():
                    out.append((fim.psm(pre, suf), mid, None, True))
            else:
                cut = rng.randrange(max(1, len(w) // 4), max(2, len(w) * 3 // 4))
                cut = w.rfind("\n", 0, cut) + 1 or cut
                if w[cut:].strip():
                    out.append((w[:cut], w[cut:cut + max_chars // 3], None, True))
            if len(out) >= n:
                return out
    return out


def encode(tok, pairs, seq):
    """pairs: (prompt, response[, system[, raw]]). raw=True examples (code completion) skip the chat
    template: the prompt is given as it is and the response ends with the end-of-text token."""
    out, end = [], None
    for pr in pairs:
        prompt, resp, system, raw = (tuple(pr) + (None, None))[:4]
        if raw:
            end = end if end is not None else fim.end_token(tok)
            p, full = prompt, prompt + resp + end
        else:
            p, full = fmt(tok, prompt, resp, system)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        fi = tok(full, add_special_tokens=False)["input_ids"][:seq]
        labels = [-100] * min(len(pi), len(fi)) + fi[len(pi):]  # loss only on the response
        if any(l != -100 for l in labels):
            out.append((fi, labels))
    return out


def batches(data, bs, pad, seed=0, skip=0):
    """Length-bucketed batches: similar lengths together, so short posts aren't padded to essays.

    Each pass over the data has its own fixed order (from seed + pass number), so a resumed run
    skips the `skip` batches it already trained on and continues with exactly the same batches."""
    epoch = 0
    while True:
        r = random.Random(seed * 100003 + epoch)
        epoch += 1
        order = sorted(data, key=lambda x: len(x[0]) + r.random())
        groups = [order[i:i + bs] for i in range(0, len(order), bs)]  # last batch may be smaller
        r.shuffle(groups)
        if skip >= len(groups):
            skip -= len(groups)
            continue
        groups, skip = groups[skip:], 0
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
    graceful.install()
    a = args()
    rng = random.Random(0)
    os.makedirs(a.out, exist_ok=True)
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a")

    threads = hf.set_threads()
    tok = hf.load_tokenizer(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ex_p = os.path.join(a.out, "examples.pt")
    if a.resume and os.path.exists(ex_p):
        # The exact examples this run was trained on (kept in the run folder, so it moves with it).
        data = torch.load(ex_p, weights_only=False)
        print(f"using this run's saved examples ({len(data)})")
    else:
        data = load_or_encode(a, tok, rng, cache=not a.private_texts)
        if not a.private_texts:
            torch.save(data, ex_p + ".tmp")
            os.replace(ex_p + ".tmp", ex_p)
    data = list(data)
    if len(data) < 2:
        raise SystemExit(f"only {len(data)} usable example(s): give more text, or a larger --seq "
                         f"(examples whose prompt alone is longer than --seq {a.seq} are dropped)")
    random.Random(1).shuffle(data)  # same train/val split every time for the same examples
    n_val = min(max(4, len(data) // 50), len(data) // 2)  # small sets keep at least half for training
    val, train = data[:n_val], data[n_val:]
    print(f"{len(train)} train / {len(val)} val examples")

    a.device = devices.pick(a.device)
    amp = devices.train_dtype(a.device)
    # GPUs with bf16 (RTX 30xx and newer) keep the frozen base weights in bf16 (half the memory).
    # Older NVIDIA GPUs keep fp32 weights and run matmuls in fp16 with loss scaling; Apple GPUs use fp32.
    dtype = torch.bfloat16 if amp == torch.bfloat16 else torch.float32
    model = hf.load_causal_lm(a.base, dtype).to(a.device)
    # On CPU the weights stay fp32 (LoRA needs fp32 master weights) and matmuls run in bf16 when
    # the CPU has native bf16 instructions; otherwise plain fp32.
    cpu_ac = devices.cpu_autocast(a.device)
    cpu_bf16 = cpu_ac is not None
    if cpu_bf16:
        autocast = lambda: cpu_ac
    elif amp == torch.float16:
        autocast = lambda: devices.autocast(a.device, amp)
    else:
        autocast = contextlib.nullcontext
    scaler = devices.grad_scaler(a.device, amp)
    prec = "bf16 autocast" if cpu_bf16 else ("fp16 autocast" if amp == torch.float16 else str(dtype).split(".")[-1])
    where = devices.describe(a.device) if a.device != "cpu" else f"{threads} threads"
    print(f"device {a.device} · {where} · {prec}")
    state_p = os.path.join(a.out, "train_state.pt")
    state = None
    if a.resume and os.path.exists(state_p) and os.path.isdir(os.path.join(a.out, "adapter")):
        state = torch.load(state_p, map_location="cpu", weights_only=False)
        a.init_adapter = os.path.join(a.out, "adapter")
    elif a.resume and os.path.isdir(os.path.join(a.out, "adapter")):
        print("no train_state.pt (made by an older version): continuing from the saved adapter weights "
              "with a fresh optimizer and schedule")
        a.init_adapter = os.path.join(a.out, "adapter")
    elif a.resume:
        print(f"nothing to resume in {a.out}; starting from step 0")
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
    start = 0
    if state is not None:
        opt.load_state_dict(state["opt"])
        for st in opt.state.values():  # saved on another device (e.g. GPU → CPU)
            for k, v in st.items():
                if torch.is_tensor(v) and k != "step":
                    st[k] = v.to(a.device)
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        start = state["step"]
        steps = max(steps, state["steps"]) if not a.max_steps else a.max_steps  # keep the same LR schedule
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed at step {start}/{steps}")
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

    def save(step):
        # Save into a temp folder and swap it in, so a crash mid-save keeps the previous adapter.
        ad, tmp = os.path.join(a.out, "adapter"), os.path.join(a.out, "adapter.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        if not a.private_texts:  # a learn step must not replace the run's own resume point
            torch.save({"opt": opt.state_dict(), "step": step, "steps": steps, "batches": step * a.grad_accum,
                        "torch_rng": torch.get_rng_state(), "base": a.base,
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None}, state_p + ".tmp")
        shutil.rmtree(ad, ignore_errors=True)
        os.replace(tmp, ad)
        if not a.private_texts:
            os.replace(state_p + ".tmp", state_p)
        write_meta(a)

    if state is None:
        log(kind="val", step=0, loss=evaluate())
    done = state.get("batches", start * a.grad_accum) if state else 0
    it, t0, model_tokens = batches(train, a.batch, pad, seed=0, skip=done), time.time(), 0
    model.train()
    for step in range(start + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        total = 0.0
        for _ in range(a.grad_accum):
            x, y, m = next(it)
            model_tokens += int(m.sum())
            with autocast():
                loss = model(input_ids=x.to(a.device), attention_mask=m.to(a.device), labels=y.to(a.device)).loss
            scaler.scale(loss / a.grad_accum).backward()
            total += loss.item() / a.grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
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
            save(step)
    print(f"saved adapter to {a.out}/adapter")
    if a.gguf:
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "export_gguf.py"), "--run", a.out])
        if rc != 0:
            print("GGUF export failed; chat will use the adapter directly (slower). "
                  "Retry from the run's page with 'Make fast CPU copy'.")


def write_meta(a):
    """Write <out>/meta.json. A run that continues its own adapter (learn.py, the crawler, Resume)
    keeps what its meta.json already says (style profile, gguf, LoRA rank, instructions); only
    --meta-extra and the base are written over it."""
    p = os.path.join(a.out, "meta.json")
    fresh = {"type": "lora", "base": a.base, "lora_r": a.lora_r,
             "instructions": bool(a.instructions), "book_share": a.book_share}
    if getattr(a, "code", None):
        fresh.update(task="code", fim=a.fim_rate > 0)
    own = os.path.join(a.out, "adapter")
    continuing = a.resume or (a.init_adapter and os.path.abspath(a.init_adapter) == os.path.abspath(own))
    old = {}
    if continuing and os.path.exists(p):
        try:
            old = json.load(open(p))
        except ValueError:
            pass
    meta = {**fresh, **old, "type": "lora", "base": a.base, **(json.loads(a.meta_extra) if a.meta_extra else {})}
    with open(p + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(p + ".tmp", p)


def load_or_encode(a, tok, rng, cache=True):
    """Build the example mix and tokenize it, cached on disk (re-tokenizing takes minutes).
    cache=False (private, temporary texts) never writes the tokenized examples to disk."""
    srcs = [a.instructions, a.extra] + sorted(glob.glob(os.path.join(a.raw, "**", "*.txt"), recursive=True))
    if getattr(a, "code", None):
        srcs += sorted(glob.glob(os.path.join(a.code, "**", "*.txt"), recursive=True))
    key = json.dumps([a.base, a.examples, a.seq, a.book_share, a.pairs_per_doc, a.min_doc_chars,
                      [(p, os.path.getmtime(p)) for p in srcs if p and os.path.exists(p)]]
                     + ([a.code, a.code_examples, a.fim_rate, a.code_chat_share, a.code_langs]
                        if getattr(a, "code", None) else []))
    path = os.path.join(ROOT, "data", "cache", hashlib.sha1(key.encode()).hexdigest()[:16] + ".pt")
    if cache and os.path.exists(path):
        print(f"using cached tokenized examples ({os.path.relpath(path, ROOT)})")
        return torch.load(path)
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
    if getattr(a, "code", None) and a.code_examples > 0:
        print(f"building {a.code_examples} code examples from {a.code}...")
        cp = code_pairs(tok, a.code, a.code_examples, a.seq, rng, a.fim_rate, a.code_chat_share,
                        L.parse_langs(a.code_langs), a.pairs_per_doc)
        print(f"{len(cp)} code examples ({sum(1 for c in cp if c[3] and c[0].startswith(fim.FIM_PREFIX))} "
              f"fill-in-the-middle, {sum(1 for c in cp if not c[3])} chat)")
        pairs += cp
    if not pairs:
        raise SystemExit("no training examples: give --instructions or --code, or put .txt files under --raw")
    data = encode(tok, pairs, a.seq)
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)
    return data


if __name__ == "__main__":
    main()
