from __future__ import annotations

from gymnasium.spaces import MultiDiscrete
import numpy as np


class MaskedMultiDiscreteCpp(MultiDiscrete):
    def __init__(self, space_cpp):
        super().__init__(space_cpp.nvec, dtype=np.int64)
        self._space_cpp = space_cpp

    def set_mask(self, mask: np.ndarray):
        if mask.shape != tuple(self.nvec):
            raise ValueError(f"Mask shape {mask.shape} must match {self.nvec}")
        if mask.dtype != bool:
            raise ValueError("Mask must be a boolean array.")
        self._space_cpp.set_mask(mask.flatten())

    def sample(self):
        return self._space_cpp.sample()

    @property
    def mask(self):
        return self._space_cpp.mask
