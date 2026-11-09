"""Silence harmless library warnings that only clutter the console.

- "triton not found; flop counting will not work for triton kernels": PyTorch's CUDA build logs
  this on Windows (triton has no Windows wheels) whenever torch.utils.flop_counter is imported.
  Nothing here counts flops, so it means nothing for training or chat.

Filters are added to the named loggers before torch creates them, and logging keeps the same
logger object, so this works whether it runs before or after `import torch`.
"""
import logging
import os

_MUTED = {"torch.utils.flop_counter": ("triton not found",)}


class _Drop(logging.Filter):
    def __init__(self, needles):
        super().__init__()
        self.needles = needles

    def filter(self, record):
        msg = record.getMessage()
        return not any(n in msg for n in self.needles)


def install():
    for name, needles in _MUTED.items():
        lg = logging.getLogger(name)
        if not any(isinstance(f, _Drop) for f in lg.filters):
            lg.addFilter(_Drop(needles))


# Driver debug switches (oneDNN verbose, Level Zero loader trace, SYCL/UR trace). Left on in the
# environment they print a line for every GPU operation and slow training down a lot on Intel GPUs.
# TEXTLLM_GPU_DEBUG=1 keeps them.
GPU_DEBUG_VARS = ("ONEDNN_VERBOSE", "DNNL_VERBOSE", "ZE_ENABLE_LOADER_DEBUG_TRACE", "ZE_ENABLE_VALIDATION_LAYER",
                  "ZE_ENABLE_TRACING_LAYER", "SYCL_PI_TRACE", "SYCL_UR_TRACE", "UR_LOG_LEVEL_ZERO", "UR_ENABLE_LAYERS")


def gpu_debug_off(env=os.environ):
    """Turn the driver debug switches off in `env` (os.environ by default); returns the names removed."""
    if env.get("TEXTLLM_GPU_DEBUG") == "1":
        return []
    gone = [k for k in GPU_DEBUG_VARS if k in env]
    for k in gone:
        env.pop(k, None)
    return gone


install()
_GPU_DEBUG_REMOVED = gpu_debug_off()
