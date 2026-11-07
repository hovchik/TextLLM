"""Load checkpoints as data only.

torch.load() unpickles by default, and a pickle can run any code while loading. Model bundles move
between PCs (transfer.py, cluster workers, downloads), so every checkpoint, optimizer state and
example cache is loaded with weights_only=True: tensors, numbers, strings, lists, dicts and nothing
else. Our own files only ever contain those, so nothing is lost.
"""
import io
import os
import pickle


def load(path, map_location="cpu"):
    """torch.load(path) restricted to plain data. Raises ValueError with a clear message when a file
    holds anything else (a checkpoint made by other software, or a tampered one)."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        name = os.path.basename(path) if isinstance(path, (str, os.PathLike)) else "the data"
        raise ValueError(f"{name} holds more than tensors and numbers, so it was not loaded. "
                         "Only checkpoints made by this project are supported.") from e


def load_bytes(data, map_location="cpu"):
    """Same for an in-memory buffer (cluster weight transfers)."""
    return load(io.BytesIO(data), map_location)
