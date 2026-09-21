from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model: str = "BAAI/bge-large-en-v1.5"
    n: int = 1024
    bits: int = 8
    dataset: str = "scifact"
    k: int = 10
    # Percentile of |X| that maps to qmax (entries beyond it saturate). None = max-abs.
    clip: float | None = None
