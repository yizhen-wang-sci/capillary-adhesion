"""Helper functions to copy between CuPy and NumPy."""
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def get_xp(*arrays):
    """Numpy or cupy, whichever the arrays live in."""
    return cp.get_array_module(*arrays) if cp is not None else np


def to_host(a):
    """Copy to host (cpu)."""
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def like(a, ref):
    """Move `a` onto the device that `ref` lives on."""
    return get_xp(ref).asarray(a)
