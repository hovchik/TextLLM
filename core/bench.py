"""Micro-benchmarks for the hot paths (CPU): run before and after a change.

    python -m core.bench            # all, ~1 minute
    python -m core.bench --quick    # fewer repetitions

Each number is the best of a few repetitions, so background noise mostly cancels out.
"""
import argparse, os, random, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def best(fn, reps):
    fn()  # warm-up
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return min(out)


def bench_generate(reps):
    import torch
    from model.gpt import GPT, GPTConfig, PRESETS

    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=8192, **PRESETS["cpu"])).eval()
    ids = torch.randint(0, 8192, (1, 32))
    n = 128

    def run():
        for _ in m.stream(ids, n, 0.8, 50, 0.95):
            pass

    t = best(run, reps)
    return f"scratch generation (cpu preset, 128 tokens): {n / t:6.1f} tok/s"


def bench_sample(reps):
    import torch
    from model.gpt import sample

    torch.manual_seed(0)
    logits = torch.randn(1, 32000)
    n = 200

    def run():
        for _ in range(n):
            sample(logits, 0.8, 50, 0.95)

    t = best(run, reps)
    return f"sample() (vocab 32k, top-k 50, top-p 0.95): {t / n * 1e3:6.2f} ms/token"


def bench_loader(reps):
    import numpy as np
    from train import Loader

    d = tempfile.mkdtemp()
    p = os.path.join(d, "train.bin")
    np.random.randint(0, 8192, 4_000_000, dtype=np.uint16).tofile(p)
    ld = Loader(p, 16, 256, "cpu", 0, 1)
    n = 200

    def run():
        for _ in range(n):
            ld.next()

    t = best(run, reps)
    return f"train Loader.next() (batch 16 x 256): {n / t:6.0f} batches/s"


def bench_train_step(reps):
    import torch
    from model.gpt import GPT, GPTConfig, PRESETS
    from train import make_optimizer

    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=8192, **PRESETS["cpu"]))
    opt = make_optimizer(m, 1e-3, 0.1, "cpu")
    x = torch.randint(0, 8192, (8, 256))

    def run():
        _, loss = m(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    t = best(run, max(2, reps // 2))
    return f"train step (cpu preset, batch 8 x 256, fwd+bwd+AdamW): {t * 1e3:6.0f} ms"


def bench_encode(reps):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from finetune.finetune_lora import encode

    words = "the quick brown fox jumps over the lazy dog and writes a long story about harbours".split()
    r = random.Random(0)
    docs = [" ".join(r.choice(words) for _ in range(300)) for _ in range(60)]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    pairs = [(d[:200], d[200:], None) for d in docs] * 10

    def run():
        encode(tok, pairs, 512)

    t = best(run, reps)
    return f"finetune encode() ({len(pairs)} examples): {len(pairs) / t:6.0f} examples/s"


BENCHES = {"generate": bench_generate, "sample": bench_sample, "loader": bench_loader,
           "train": bench_train_step, "encode": bench_encode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated: " + ",".join(BENCHES))
    a = ap.parse_args()
    from writing import hf

    hf.set_threads()
    reps = 2 if a.quick else 4
    names = [n for n in a.only.split(",") if n] or list(BENCHES)
    for n in names:
        try:
            print(BENCHES[n](reps), flush=True)
        except Exception as e:  # noqa: BLE001 - a missing optional package shouldn't stop the others
            print(f"{n}: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
