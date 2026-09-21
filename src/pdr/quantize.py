from __future__ import annotations

import numpy as np


def symmetric_scale(X: np.ndarray, bits: int) -> float:
    """Max-abs scale: the single largest coordinate maps to qmax (no saturation)."""
    qmax = (1 << (bits - 1)) - 1
    amax = float(np.max(np.abs(X)))
    if amax == 0.0:
        return 1.0
    return amax / qmax


def clipped_scale(X: np.ndarray, bits: int, percentile: float) -> float:
    """Clipped scale: the given percentile of |X| maps to qmax; larger entries saturate.

    Max-abs scaling lets one outlier coordinate set the step for every other
    coordinate. Clipping at, e.g., the 99th percentile shrinks the step by 2-4x
    for BERT-family encoders at the cost of saturating ~1% of entries. Entries
    still lie in [-qmax, qmax], so the bound B = 2^(b-1) and every cryptographic
    constraint that depends on it are unchanged.
    """
    qmax = (1 << (bits - 1)) - 1
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    p = float(np.percentile(np.abs(X), percentile))
    if p == 0.0:
        return 1.0
    return p / qmax


def quantize(
    X: np.ndarray,
    bits: int = 8,
    scale: float | None = None,
    clip_percentile: float | None = None,
) -> tuple[np.ndarray, float]:
    """Symmetric scalar quantization to b-bit signed integers.

    scale=None picks the scale from X: max-abs by default, or the clipped rule
    when clip_percentile is given (100 is equivalent to max-abs). The result is
    always clipped to [-qmax, qmax], which is a no-op under max-abs and the
    saturation step under the clipped rule.
    """
    if scale is None:
        if clip_percentile is None or clip_percentile >= 100.0:
            scale = symmetric_scale(X, bits)
        else:
            scale = clipped_scale(X, bits, clip_percentile)
    qmax = (1 << (bits - 1)) - 1
    Xi = np.round(X / scale).clip(-qmax, qmax).astype(np.int64)
    return Xi, scale
