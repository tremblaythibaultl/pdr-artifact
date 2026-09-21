"""Required bit-width versus corpus size, from the scale sweeps (scale/run_scale.sbatch).

Reads results/scale/scale__nq__<model>__clip99.csv (one per encoder) and
prints, per encoder and corpus size: top-1 agreement at each b (and the b=8
ceiling, which is what near-duplicate passages cap it at), nDCG@10 retained,
the margin statistic, its per-decade decay exponent, and the smallest b that
meets two criteria: nDCG retained >= 99% (the top-k endpoint) and top-1
agreement within 1 point of the b=8 ceiling (the exact top-1 endpoint).

Run: python analysis/scale_analysis.py
"""
from __future__ import annotations

import csv
import glob
from collections import defaultdict
from math import log10
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results" / "scale"
NDCG_OK = 99.0
TOP1_SLACK = 0.01


def load(path: Path):
    d = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            d[int(r["m"])][r["bits"]] = r
    return d


for path in sorted(glob.glob(str(RES / "scale__nq__*__clip99.csv"))):
    d = load(Path(path))
    model = next(iter(next(iter(d.values())).values()))["model"]
    bits = [b for b in ("3", "4", "5", "6", "8")]
    print(f"\n===== {model} =====")
    print(f"{'m':>9} | " + "".join(f"{'top1@'+b:>8}" for b in bits) + f" | {'tol@4':>6}{'tie%':>6} | " +
          "".join(f"{'ret@'+b:>7}" for b in bits) + f" | {'gap/n@4':>8}{'alpha':>6} | {'b_ndcg':>6}{'b_top1':>6}")
    prev = None
    for m in sorted(d):
        row = d[m]
        top1 = {b: float(row[b]["top1_agreement"]) for b in bits if b in row}
        ret = {b: float(row[b]["ndcg_retained_pct"]) for b in bits if b in row}
        ratio4 = float(row["4"]["median_gap_over_noise"]) if "4" in row else float("nan")
        alpha = "" if prev is None else f"{log10(prev[1] / ratio4) / log10(m / prev[0]):.2f}"
        ceiling = top1.get("8", max(top1.values()))
        b_ndcg = next((b for b in bits if ret.get(b, 0) >= NDCG_OK), ">8")
        b_top1 = next((b for b in bits if top1.get(b, 0) >= ceiling - TOP1_SLACK), ">8")
        tol4 = row["4"].get("top1_agreement_tol", "")
        tie = row["4"].get("frac_near_tie", "")
        print(f"{m:>9} | " + "".join(f"{top1[b]:8.3f}" for b in bits if b in top1) +
              f" | {float(tol4) if tol4 else float('nan'):6.3f}{100*float(tie) if tie else float('nan'):6.1f} | " +
              "".join(f"{ret[b]:7.2f}" for b in bits if b in ret) + f" | {ratio4:8.2f}{alpha:>6} | {b_ndcg:>6}{b_top1:>6}")
        prev = (m, ratio4)
print("\nb_ndcg: smallest b with nDCG@10 retained >= 99%.  b_top1: smallest b with top-1 agreement within 1 pt of the b=8 ceiling.")
print("alpha: per-decade decay exponent of the b=4 margin statistic between consecutive sizes.")

# --- does the margin statistic still predict agreement at scale? ---
import numpy as np
from math import erf, sqrt
xs, ys, ceil = [], [], []
for path in sorted(glob.glob(str(RES / "scale__nq__*__clip99.csv"))):
    d = load(Path(path))
    for m in d:
        c = float(d[m]["8"]["top1_agreement"])
        for b in ("3", "4", "5", "6", "8"):
            xs.append(float(d[m][b]["median_gap_over_noise"])); ys.append(float(d[m][b]["top1_agreement"])); ceil.append(c)
xs, ys, ceil = np.array(xs), np.array(ys), np.array(ceil)
rho = np.corrcoef(xs.argsort().argsort(), ys.argsort().argsort())[0, 1]
rho_rel = np.corrcoef(xs.argsort().argsort(), (ys / ceil).argsort().argsort())[0, 1]
print(f"\nMechanism at scale: Spearman(median gap/noise, top-1 agreement) over {len(xs)} (encoder, m, b) points = {rho:.3f}; "
      f"against agreement relative to the b=8 ceiling = {rho_rel:.3f}")
for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]:
    sel = (xs >= lo) & (xs < hi)
    if sel.any():
        print(f"  gap/noise in [{lo:>2},{hi if hi < 1e9 else 'inf':>3}): n={sel.sum():2d}  top-1 {ys[sel].min():.3f}-{ys[sel].max():.3f}  "
              f"relative to ceiling {(ys[sel]/ceil[sel]).min():.3f}-{(ys[sel]/ceil[sel]).max():.3f}")

# --- LaTeX body of tab:scale-full (appendix): nDCG@10 retained at b=3 and b=4, smallest b at >= 99% ---
NAMES = {"Snowflake/snowflake-arctic-embed-l-v2.0": r"\texttt{arctic-embed-l-v2.0}",
         "BAAI/bge-large-en-v1.5": r"\texttt{bge-large-en-v1.5}",
         "mixedbread-ai/mxbai-embed-large-v1": r"\texttt{mxbai-embed-large-v1}",
         "intfloat/e5-large-v2": r"\texttt{e5-large-v2}",
         "thenlper/gte-large": r"\texttt{gte-large}"}
ORDER = list(NAMES)  # the quantization-aware encoder first, then the conventional four
MSIZE = {10000: "$10^4$", 100000: "$10^5$", 1000000: "$10^6$", 2681468: r"$2.68{\cdot}10^6$"}
by_model = {}
for path in glob.glob(str(RES / "scale__nq__*__clip99.csv")):
    d = load(Path(path))
    by_model[next(iter(next(iter(d.values())).values()))["model"]] = d
print("\n% body of tab:scale-full")
present = [mo for mo in ORDER if mo in by_model]
for i, mo in enumerate(present):
    d = by_model[mo]
    for j, m in enumerate(sorted(d)):
        ret = {b: float(d[m][b]["ndcg_retained_pct"]) for b in ("3", "4", "5", "6", "8")}
        need = next((b for b in ("3", "4", "5", "6", "8") if ret[b] >= NDCG_OK), ">8")
        lead = NAMES[mo] if j == 0 else ""
        print(f"{lead} & {MSIZE[m]} & ${ret['3']:.1f}$ & ${ret['4']:.1f}$ & ${need}$ \\\\")
    if i < len(present) - 1:
        print(r"\midrule")
