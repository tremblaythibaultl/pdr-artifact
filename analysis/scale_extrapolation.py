"""Does the margin shrink with corpus size, and where does b=4 (clipped) break?

For each query keep its float top-1 document and subsample the remaining
documents at fractions of the corpus. The gap between the top-1 and the best
of the rest shrinks as more distractors compete; for light-tailed scores the
maximum of m draws grows like sqrt(ln m), but the measured decay is much steeper, so we
fit a power law ratio ~ m^-alpha on the upper half of the range and extrapolate
to 1e5 and 1e6 documents (an in-distribution extrapolation; the scale
run measures it). Agreement is the fraction of queries
whose clipped-quantized argmax over the subset is still the float top-1.

Run: python analysis/scale_extrapolation.py (needs the cached embeddings in .pdr-cache/)
"""
from __future__ import annotations

from math import erf, log, sqrt
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / ".pdr-cache"
DS = [("scifact", 5183, 300), ("nfcorpus", 3633, 323)]
ENC = [("arctic", "Snowflake_snowflake-arctic-embed-l-v2.0", True, "__p2d5312", ""),
       ("cohere", "embed-english-v3.0", True, "", ""),
       ("bge", "BAAI_bge-large-en-v1.5", False, "", ""),
       ("mxbai", "mixedbread-ai_mxbai-embed-large-v1", False, "__p9c844c", ""),
       ("e5", "intfloat_e5-large-v2", False, "__p2d5312", "__p537364"),
       ("gte", "thenlper_gte-large", False, "", "")]
FRACS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
REPS = 4
P = 99


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def quant(X, s, qmax):
    return np.round(X / s).clip(-qmax, qmax).astype(np.int64)


for bits in (4, 3):
    qmax = (1 << (bits - 1)) - 1
    print(f"\n===== clipped p{P}, b={bits}: gap-to-noise ratio and top-1 agreement vs corpus size =====")
    print(f"{'corpus':<9}{'encoder':<8}{'lowbit':>6} | " + "".join(f"m={int(m):>6}      " for m in [5183/32, 5183/16, 5183/8, 5183/4, 5183/2, 5183]) + "| alpha | ratio@1e5 ratio@1e6 | pred agr@1e5 @1e6")
    for ds, m_full, nq in DS:
        for key, stem, lb, qt, dt in ENC:
            A = np.load(CACHE / f"{ds}__{stem}__corpus__n{m_full}{dt}.npy").astype(np.float64)
            X = np.load(CACHE / f"{ds}__{stem}__queries__n{nq}{qt}.npy").astype(np.float64)
            sA = np.percentile(np.abs(A), P) / qmax
            sx = (np.percentile(np.abs(X), P, axis=1) / qmax)[:, None]
            A_int, X_int = quant(A, sA, qmax), quant(X, sx, qmax)
            S = X @ A.T
            Sq = (A_int @ X_int.T).T
            top1 = S.argmax(1)
            noise = np.sqrt((sA**2 * (X**2).sum(1) + sx[:, 0] ** 2) / 12.0)
            rng = np.random.default_rng(0)
            ms, ratios, agrs = [], [], []
            for f in FRACS:
                r_acc, a_acc = [], []
                for rep in range(REPS if f < 1 else 1):
                    keep = rng.random(m_full) < f
                    gaps, agree = [], []
                    for qi in range(nq):
                        k = keep.copy(); k[top1[qi]] = True
                        idx = np.flatnonzero(k)
                        s = S[qi, idx]; sq = Sq[qi, idx]
                        order = np.argsort(-s)
                        gaps.append(s[order[0]] - s[order[1]])
                        agree.append(idx[np.argmax(sq)] == top1[qi])
                    r_acc.append(np.median(np.array(gaps) / noise)); a_acc.append(np.mean(agree))
                ms.append(f * m_full); ratios.append(np.mean(r_acc)); agrs.append(np.mean(a_acc))
            # The measured decay is far steeper than the sqrt(ln m) of light-tailed order
            # statistics: fit a power law ratio ~ m^-alpha on the upper half of the range.
            sel = slice(2, None)
            alpha, loga = np.polyfit(np.log(ms[sel]), np.log(ratios[sel]), 1)
            alpha = -alpha
            def ext(m): return float(np.exp(loga) * m ** (-alpha))
            r5, r6 = ext(1e5), ext(1e6)
            print(f"{ds:<9}{key:<8}{'yes' if lb else 'no':>6} | " + "".join(f"{r:5.2f}/{g:.2f}  " for r, g in zip(ratios, agrs)) + f"| {alpha:5.2f} | {r5:8.2f} {r6:9.2f} | {phi(r5/sqrt(2)):9.2f} {phi(r6/sqrt(2)):6.2f}")
