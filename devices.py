"""Pick the fastest device this PC has, and the matching precision.

Order: NVIDIA GPU (CUDA, also AMD on Linux via ROCm builds), Intel GPU (XPU), Apple GPU (MPS), CPU.
Override with --device on any script, or the TEXTLLM_DEVICE environment variable (cpu, cuda, cuda:1, mps, xpu).

Checkpoints are always loaded with map_location="cpu" and then moved, so a run saved on a GPU
continues on a CPU and the other way round.
"""
import os
from contextlib import nullcontext

import core.quiet  # noqa: F401  mutes "triton not found" before torch is imported


def _mps_ok(torch):
    b = getattr(torch.backends, "mps", None)
    return bool(b and b.is_available())


def _xpu_ok(torch):
    x = getattr(torch, "xpu", None)
    try:
        return bool(x and x.is_available())
    except Exception:
        return False


def pick(pref=None):
    """Return a torch device string. pref: None/"auto" or an explicit device."""
    import torch

    pref = (pref or os.environ.get("TEXTLLM_DEVICE") or "auto").strip().lower()
    if pref != "auto":
        ok = {"cuda": torch.cuda.is_available(), "mps": _mps_ok(torch), "xpu": _xpu_ok(torch), "cpu": True}
        if ok.get(pref.split(":")[0], False):
            return pref
        print(f"[device] {pref} was asked for but is not available here; using auto-detect instead")
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_ok(torch):
        return "xpu"
    if _mps_ok(torch):
        return "mps"
    return "cpu"


def kind(dev):
    return str(dev).split(":")[0]


def train_dtype(dev):
    """Mixed-precision type for training: bf16 where the GPU has it, fp16 on older NVIDIA GPUs
    (GTX 10/16, RTX 20 series; needs a GradScaler), fp32 on Apple and CPU."""
    import torch

    k = kind(dev)
    if k == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if k == "xpu":
        return torch.bfloat16
    return torch.float32


def infer_dtype(dev):
    """Weight type for chatting/generation on a GPU (half the memory of fp32); None = keep fp32."""
    import torch

    k = kind(dev)
    if k == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if k in ("mps", "xpu"):
        return torch.float16
    return None


def autocast(dev, dtype):
    import torch

    if kind(dev) in ("cuda", "xpu") and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(kind(dev), dtype=dtype)
    return nullcontext()


def cpu_autocast(dev):
    """bf16 autocast for training on a CPU with native bf16 matmuls (AVX-512 BF16 / AMX), about
    1.7x faster there; weights and optimizer stay fp32. None elsewhere. TEXTLLM_BF16=0 turns it off."""
    import torch

    from writing import hf

    if kind(dev) == "cpu" and hf.cpu_bf16_ok():
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return None


def grad_scaler(dev, dtype):
    """A GradScaler that only does something for fp16 (bf16/fp32 don't need loss scaling)."""
    import torch

    on = kind(dev) == "cuda" and dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=on)


def describe(dev):
    """Human-readable name, e.g. 'NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB)'."""
    import torch

    k = kind(dev)
    try:
        if k == "cuda":
            i = torch.device(dev).index or 0
            p = torch.cuda.get_device_properties(i)
            return f"{p.name} ({p.total_memory / 2**30:.1f} GB)"
        if k == "xpu":
            return torch.xpu.get_device_name(0)
        if k == "mps":
            return "Apple GPU (Metal)"
    except Exception:
        pass
    return f"CPU ({os.cpu_count() or '?'} threads)"


def info():
    """For the Studio status bar: what will be used and why."""
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "name": "CPU", "gpu": False, "hint": "PyTorch is not installed"}
    dev = pick()
    out = {"device": kind(dev), "name": describe(dev), "gpu": kind(dev) != "cpu", "torch": torch.__version__,
           "cuda_build": bool(torch.version.cuda)}
    if not out["gpu"] and not torch.version.cuda and _nvidia_present():
        out["hint"] = "An NVIDIA GPU is present but this PyTorch is the CPU build. Run run.bat / run.sh again to switch."
    return out


def available():
    """Devices this PC can use, for the Studio's device picker: [{"id": "cuda:0", "name": ...}, ...]."""
    out = [{"id": "auto", "name": "Automatic (fastest available)"}, {"id": "cpu", "name": "CPU"}]
    try:
        import torch
    except ImportError:
        return out
    try:
        for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            out.append({"id": f"cuda:{i}" if i else "cuda", "name": describe(f"cuda:{i}")})
    except Exception:
        pass
    if _xpu_ok(torch):
        out.append({"id": "xpu", "name": describe("xpu")})
    if _mps_ok(torch):
        out.append({"id": "mps", "name": describe("mps")})
    return out


def _nvidia_present():
    import shutil, subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
