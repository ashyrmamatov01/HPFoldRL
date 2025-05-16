"""Utility helpers shared across the project."""
from __future__ import annotations
import random
import numpy as np


def set_seed(seed: int | None = None) -> int:
    """Set PY + NumPy RNGs and return the resolved seed."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    return seed


import inspect

def extract_params(obj):
    """
    Return a dict of all public attributes of obj whose values are int, float or bool.
    """
    nums = {}
    for name, val in vars(obj).items():
        # skip private / dunder attributes
        if name.startswith("_"):
            continue
        # keep only plain numbers or booleans
        if isinstance(val, (int, float, bool)):
            nums[name] = val
    return nums