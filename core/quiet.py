"""Silence harmless library warnings that only clutter the console.

- "triton not found; flop counting will not work for triton kernels": PyTorch's CUDA build logs
  this on Windows (triton has no Windows wheels) whenever torch.utils.flop_counter is imported.
  Nothing here counts flops, so it means nothing for training or chat.

Filters are added to the named loggers before torch creates them, and logging keeps the same
logger object, so this works whether it runs before or after `import torch`.
"""
import logging

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


install()
