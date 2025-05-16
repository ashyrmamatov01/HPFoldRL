"""Legal move sets for 2-D and 3-D lattice HP folding."""
from __future__ import annotations
from enum import Enum
import numpy as np


class Move2D(Enum):
    """Relative moves on a square lattice."""
    LEFT = (-1, 0)
    FORWARD = (0, 1)
    RIGHT = (1, 0)

    @classmethod
    def as_array(cls) -> np.ndarray:
        return np.array([m.value for m in cls])


class Move3D(Enum):
    # """Absolute moves on a cubic lattice."""
    POS_X = ( 1,  0,  0)
    NEG_X = (-1,  0,  0)
    POS_Y = ( 0,  1,  0)
    NEG_Y = ( 0, -1,  0)
    POS_Z = ( 0,  0,  1)
    NEG_Z = ( 0,  0, -1)

    """Relative moves on a cubic lattice."""
    # FORWARD

    @classmethod
    def as_array(cls) -> np.ndarray:
        return np.array([m.value for m in cls])
