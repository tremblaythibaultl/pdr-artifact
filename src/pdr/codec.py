from __future__ import annotations

import numpy as np


def encode_doc(text: str, ell: int) -> np.ndarray:
    raw = text.encode("utf-8")
    if len(raw) > ell:
        raise ValueError(f"document of {len(raw)} bytes exceeds row length {ell}")
    row = np.zeros(ell, dtype=np.int64)
    row[: len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    return row


def decode_doc(row: np.ndarray) -> str:
    arr = np.asarray(row, dtype=np.int64)
    nz = np.nonzero(arr)[0]
    end = int(nz[-1]) + 1 if nz.size else 0
    return bytes(arr[:end].astype(np.uint8).tolist()).decode("utf-8", errors="strict")


def build_D(docs: list[str]) -> tuple[np.ndarray, int]:
    ell = max((len(d.encode("utf-8")) for d in docs), default=0)
    D = np.zeros((len(docs), ell), dtype=np.int64)
    for i, d in enumerate(docs):
        D[i] = encode_doc(d, ell)
    return D, ell
