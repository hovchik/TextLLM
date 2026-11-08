#!/usr/bin/env python3
"""Check which parts of the AI toolkit are installed and working on this PC.

    python stack_check.py          # table of every package, its version and what it unlocks
    python stack_check.py --quick  # versions only, skip the small smoke tests

Each package also gets a tiny offline smoke test (no downloads), so "OK" means it really works
here, not just that pip installed it. Exit code is 0 when every core package works.
"""
import argparse, importlib, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import backends  # noqa: E402

# (pip name, import name, group, what it unlocks)
PACKAGES = [
    ("torch", "torch", "core", "Tensors, autograd, GPU/CPU training"),
    ("transformers", "transformers", "core", "Load and run Qwen, Llama, Mistral, Gemma ..."),
    ("peft", "peft", "core", "LoRA / QLoRA adapters: fine-tune big models cheaply"),
    ("accelerate", "accelerate", "core", "Device placement, mixed precision, multi-GPU"),
    ("datasets", "datasets", "core", "Stream and map huge datasets from disk or the Hub"),
    ("tokenizers", "tokenizers", "core", "Fast BPE tokenizer training (from-scratch GPT)"),
    ("safetensors", "safetensors", "core", "Safe, fast model weight files"),
    ("cryptography", "cryptography", "core", "Open AES-encrypted PDFs (most protected e-books)"),
    ("pdfminer.six", "pdfminer", "core", "Second PDF reader when pypdf gets little text"),
    ("trl", "trl", "ai", "SFT, DPO, ORPO, KTO, GRPO trainers (instruction + preference tuning)"),
    ("lm-eval", "lm_eval", "ai", "Standard benchmarks: HellaSwag, ARC, MMLU, ..."),
    ("sentence-transformers", "sentence_transformers", "ai", "Embeddings for search, RAG, semantic dedup"),
    ("faiss-cpu", "faiss", "ai", "Vector index: nearest-neighbour search over millions of texts"),
    ("datasketch", "datasketch", "ai", "MinHash LSH near-duplicate removal (data/clean_corpus.py)"),
    ("rapidfuzz", "rapidfuzz", "ai", "Fast fuzzy string matching"),
    ("gguf", "gguf", "ai", "Read/write GGUF files for llama.cpp"),
    ("tensorboard", "tensorboard", "ai", "Loss curves in the browser: tensorboard --logdir runs"),
    ("pytesseract", "pytesseract", "ai", "OCR for scanned PDFs (also needs the Tesseract program)"),
] + [(pip, imp, grp, what) for imp, pip, grp, what, _ in backends.ENGINES if grp == "fast"] + [
    ("bitsandbytes", "bitsandbytes", "gpu", "4/8-bit loading (QLoRA) and 8-bit optimizers"),
    ("liger-kernel", "liger_kernel", "gpu", "Fused Triton kernels: faster, leaner training (Linux)"),
]


def smoke(name):
    """A tiny offline test per package. Returns a short note or raises."""
    if name == "torch":
        import torch
        x = torch.randn(64, 64, requires_grad=True)
        (x @ x).sum().backward()
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if dev == "cuda":
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return f"device: {dev}"
    if name == "transformers":
        from transformers import AutoConfig  # noqa: F401
        return ""
    if name == "peft":
        from peft import LoraConfig
        LoraConfig(r=8, target_modules=["q_proj"])
        return ""
    if name == "trl":
        from trl import SFTConfig, DPOConfig, SFTTrainer, DPOTrainer  # noqa: F401
        return ""
    if name == "faiss-cpu":
        import faiss, numpy as np
        xb = np.random.rand(200, 32).astype("float32")
        index = faiss.IndexFlatL2(32)
        index.add(xb)
        _, ids = index.search(xb[:3], 1)
        assert list(ids[:, 0]) == [0, 1, 2]
        return ""
    if name == "datasketch":
        from datasketch import MinHash
        a, b = MinHash(num_perm=64), MinHash(num_perm=64)
        for w in "the quick brown fox jumps over the lazy dog".split():
            a.update(w.encode())
        for w in "the quick brown fox jumped over the lazy dog".split():
            b.update(w.encode())
        return f"similarity test {a.jaccard(b):.2f}"
    if name == "rapidfuzz":
        from rapidfuzz import fuzz
        assert fuzz.ratio("story", "stories") > 50
        return ""
    if name == "bitsandbytes":
        import torch, bitsandbytes  # noqa: F401
        return "4-bit ready" if torch.cuda.is_available() else "installed (needs an NVIDIA GPU to help)"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="versions only, no smoke tests")
    a = ap.parse_args()

    print(f"Python {sys.version.split()[0]} on {sys.platform}\n")
    print(f"{'package':<22} {'group':<5} {'version':<12} {'status':<8} what it unlocks")
    print("-" * 100)
    core_ok, missing = True, []
    for pip_name, mod, group, unlocks in PACKAGES:
        ver = backends.version(pip_name)
        status, note = "missing", ""
        if ver:
            try:
                t = time.time()
                importlib.import_module(mod)
                note = "" if a.quick else smoke(pip_name)
                status = "OK"
                if not a.quick and time.time() - t > 20:
                    note = (note + " (slow import)").strip()
            except Exception as e:  # installed but broken here (e.g. Triton without a GPU)
                status, note = "error", f"{type(e).__name__}: {str(e).splitlines()[0][:60]}" if str(e) else type(e).__name__
        if status != "OK":
            if group == "core":
                core_ok = False
            missing.append(pip_name)
        line = f"{pip_name:<22} {group:<5} {ver or '-':<12} {status:<8} {unlocks}"
        print(line + (f"  [{note}]" if note else ""))

    from data import pdftext

    exe = pdftext.tesseract_cmd()
    print(f"\nTesseract OCR program: {exe or 'not found (scanned PDFs are skipped; see README)'}")
    print()
    if missing:
        print("Not working here: " + ", ".join(missing))
        print("  core  -> run.bat --reinstall   (or pip install -r requirements.txt)")
        print("  ai    -> pip install -r requirements-ai.txt")
        print("  gpu   -> pip install -r requirements-gpu.txt   (only useful with an NVIDIA GPU)")
        print("  fast  -> run.bat installs a prebuilt wheel; or pip install llama-cpp-python")
    else:
        print("Everything is installed and working.")
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
