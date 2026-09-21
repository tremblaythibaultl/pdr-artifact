from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MatVecBackend(ABC):
    @abstractmethod
    def matvec(self, M: np.ndarray, x: np.ndarray) -> np.ndarray:
        ...


class PlaintextBackend(MatVecBackend):
    def matvec(self, M: np.ndarray, x: np.ndarray) -> np.ndarray:
        return M.astype(np.int64, copy=False) @ x.astype(np.int64, copy=False)
