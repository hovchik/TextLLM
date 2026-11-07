"""Shared helpers for Hugging Face models on CPU: loading, chat templates, threads, precision."""
import os


def physical_cores():
    """Physical CPU cores (hyperthreads slow matmuls down). Falls back to logical/2."""
    try:
        seen = set()
        phys = core = None
        for line in open("/proc/cpuinfo"):
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
                seen.add((phys, core))
        if seen:
            return len(seen)
    except OSError:
        pass
    n = os.cpu_count() or 2
    return max(1, n // 2) if n > 2 else n


def set_threads():
    import torch

    n = int(os.environ.get("TEXTLLM_THREADS", 0)) or physical_cores()
    torch.set_num_threads(n)
    return n


def cpu_bf16_ok():
    """True when this CPU runs bfloat16 matmuls natively (AVX-512 BF16 or AMX)."""
    if os.environ.get("TEXTLLM_BF16") == "0":
        return False
    try:
        flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
        return "avx512_bf16" in flags or "amx_bf16" in flags
    except (OSError, StopIteration):
        return False


def load_causal_lm(name, dtype=None):
    """Load a text model. Multimodal checkpoints (Qwen3.5, Gemma 4) fall back to their
    image-text class, which still takes plain text input."""
    import torch
    from transformers import AutoModelForCausalLM

    kw = {dtype_kwarg(): dtype or torch.float32}
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError) as e:
        last = e
    for cls in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        try:
            import transformers

            return getattr(transformers, cls).from_pretrained(name, **kw)
        except (AttributeError, ValueError, KeyError) as e:
            last = e
    raise last


def dtype_kwarg():
    """from_pretrained's precision argument: "dtype" since transformers 4.56, "torch_dtype" before
    (older versions silently ignore "dtype" and load fp32, twice the memory)."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_tokenizer(name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return getattr(tok, "tokenizer", tok)  # a processor wraps the tokenizer


def chat_text(tok, msgs, add_generation_prompt=True):
    """Render chat messages with the model's template, thinking mode off.

    Templates without a system role get the system text folded into the first user turn."""
    if not tok.chat_template:
        sys_ = "".join(m["content"] + "\n\n" for m in msgs if m["role"] == "system")
        body = "".join(f"### {m['role'].title()}:\n{m['content']}\n\n" for m in msgs if m["role"] != "system")
        return sys_ + body + ("### Assistant:\n" if add_generation_prompt else "")
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       enable_thinking=False)
    except Exception:
        if msgs and msgs[0]["role"] == "system":
            rest = [dict(m) for m in msgs[1:]]
            if rest:
                rest[0]["content"] = msgs[0]["content"] + "\n\n" + rest[0]["content"]
            return tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=add_generation_prompt)
        raise


def repetition_from_presence(pp):
    """HF generate has no presence penalty; map it onto a mild repetition penalty."""
    return 1.0 + 0.05 * max(0.0, float(pp or 0))
