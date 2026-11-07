"""Which fast engines are installed. The project runs on pure Python + PyTorch everywhere; these
compiled libraries make parts of it much faster when they are present:

    tokenizers    (Rust)   BPE training and encoding, used for every from-scratch model     [required]
    llama.cpp     (C++)    4-bit GGUF chat on the CPU, 5-10x faster than PyTorch for fine-tunes [optional]
    faiss         (C++)    vector index for near-duplicate search and retrieval            [optional]
    ctranslate2   (C++)    int8 CPU inference for Transformer models                      [optional]
    onnxruntime   (C++)    graph-optimised CPU/GPU inference                              [optional]

Studio → Settings shows this table; `python stack_check.py` prints it (its "fast" group).
"""
import functools
import importlib.metadata as md
import importlib.util

ENGINES = [
    # (import name, pip name, group as in stack_check.py, what it speeds up, how to get it)
    ("tokenizers", "tokenizers", "core", "Tokenizer training and encoding (Rust)", "installed with requirements.txt"),
    ("llama_cpp", "llama-cpp-python", "fast", "Fast 4-bit GGUF chat and code completion on the CPU",
     "pip install llama-cpp-python  (needs a C++ compiler, or use a prebuilt wheel)"),
    ("faiss", "faiss-cpu", "ai", "Near-duplicate search and retrieval over millions of texts", "pip install faiss-cpu"),
    ("ctranslate2", "ctranslate2", "fast", "int8 CPU inference for exported Transformer models", "pip install ctranslate2"),
    ("onnxruntime", "onnxruntime", "fast", "Graph-optimised inference for exported models", "pip install onnxruntime"),
    ("bitsandbytes", "bitsandbytes", "gpu", "4/8-bit model loading (QLoRA) on NVIDIA GPUs", "run.bat --gpu"),
]


def version(pip_name):
    try:
        return md.version(pip_name)
    except md.PackageNotFoundError:
        return None


def available(import_name):
    """Installed or not, without importing it: some of these load their own OpenMP runtime, which
    can clash with PyTorch's inside the Studio process."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


@functools.lru_cache(maxsize=1)
def report():
    """[{name, pip, installed, version, speeds_up, install}] for every engine. Cached: what is
    installed doesn't change while the process runs."""
    out = []
    for imp, pip, _group, what, how in ENGINES:
        ok = available(imp)
        out.append({"name": imp.replace("_", "-"), "pip": pip, "installed": ok,
                    "version": version(pip) if ok else None, "speeds_up": what, "install": how})
    return out
