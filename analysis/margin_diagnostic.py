"""Why does one encoder survive b-bit quantization and another not?

Diagnostic run on the cached float embeddings only (no protocol, no reader).
For each (dataset, model) it reports the two encoder properties that set the
quantization step and the score margin, and a Gaussian-noise prediction of
top-1 agreement at bit-width b, to compare against the measured top1_agreement
in results/<dataset>__<model>__k10.csv.

Mechanism tested: symmetric scalar quantization rounds every coordinate to a
grid of step s = max|X| / (2^(b-1)-1). Rounding both sides perturbs each score
by noise of std ~ sqrt((s_A^2 ||x||^2 + s_x^2 ||a||^2)/12). The float top-1
survives when the gap to the runner-up exceeds that noise, so the predictor is
median over queries of gap / noise. Both terms are encoder properties: the gap
is set by how spread the similarity distribution is, the step by the largest
coordinate the encoder emits.

Usage: python analysis/margin_diagnostic.py [--bits 4] [--cache .pdr-cache]
(the max-abs sweeps and the cached embeddings must be present).
"""
from __future__ import annotations

import argparse
import csv
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (dataset, label) -> (corpus cache file, query cache file). Prefix-tagged
# files are the model-card-prescribed instruction variants (see pdr.embed).
FILES = {
    ("scifact", "mxbai"): ("scifact__mixedbread-ai_mxbai-embed-large-v1__corpus__n5183.npy", "scifact__mixedbread-ai_mxbai-embed-large-v1__queries__n300__p9c844c.npy"),
    ("scifact", "bge"): ("scifact__BAAI_bge-large-en-v1.5__corpus__n5183.npy", "scifact__BAAI_bge-large-en-v1.5__queries__n300.npy"),
    ("scifact", "bge+pfx"): ("scifact__BAAI_bge-large-en-v1.5__corpus__n5183.npy", "scifact__BAAI_bge-large-en-v1.5__queries__n300__p9c844c.npy"),
    ("scifact", "e5"): ("scifact__intfloat_e5-large-v2__corpus__n5183__p537364.npy", "scifact__intfloat_e5-large-v2__queries__n300__p2d5312.npy"),
    ("scifact", "gte"): ("scifact__thenlper_gte-large__corpus__n5183.npy", "scifact__thenlper_gte-large__queries__n300.npy"),
    ("scifact", "cohere"): ("scifact__embed-english-v3.0__corpus__n5183.npy", "scifact__embed-english-v3.0__queries__n300.npy"),
    ("scifact", "bert"): ("scifact__bert-base-uncased__corpus__n5183.npy", "scifact__bert-base-uncased__queries__n300.npy"),
    ("scifact", "arctic"): ("scifact__Snowflake_snowflake-arctic-embed-l-v2.0__corpus__n5183.npy", "scifact__Snowflake_snowflake-arctic-embed-l-v2.0__queries__n300__p2d5312.npy"),
    ("nfcorpus", "mxbai"): ("nfcorpus__mixedbread-ai_mxbai-embed-large-v1__corpus__n3633.npy", "nfcorpus__mixedbread-ai_mxbai-embed-large-v1__queries__n323__p9c844c.npy"),
    ("nfcorpus", "bge"): ("nfcorpus__BAAI_bge-large-en-v1.5__corpus__n3633.npy", "nfcorpus__BAAI_bge-large-en-v1.5__queries__n323.npy"),
    ("nfcorpus", "bge+pfx"): ("nfcorpus__BAAI_bge-large-en-v1.5__corpus__n3633.npy", "nfcorpus__BAAI_bge-large-en-v1.5__queries__n323__p9c844c.npy"),
    ("nfcorpus", "e5"): ("nfcorpus__intfloat_e5-large-v2__corpus__n3633__p537364.npy", "nfcorpus__intfloat_e5-large-v2__queries__n323__p2d5312.npy"),
    ("nfcorpus", "gte"): ("nfcorpus__thenlper_gte-large__corpus__n3633.npy", "nfcorpus__thenlper_gte-large__queries__n323.npy"),
    ("nfcorpus", "cohere"): ("nfcorpus__embed-english-v3.0__corpus__n3633.npy", "nfcorpus__embed-english-v3.0__queries__n323.npy"),
    ("nfcorpus", "bert"): ("nfcorpus__bert-base-uncased__corpus__n3633.npy", "nfcorpus__bert-base-uncased__queries__n323.npy"),
    ("nfcorpus", "arctic"): ("nfcorpus__Snowflake_snowflake-arctic-embed-l-v2.0__corpus__n3633.npy", "nfcorpus__Snowflake_snowflake-arctic-embed-l-v2.0__queries__n323__p2d5312.npy"),
}

# Which results file holds the measured sweep for each label.
RESULTS = {
    "mxbai": "results/{ds}__mixedbread-ai_mxbai-embed-large-v1__k10.csv",
    "bge": "results/{ds}__BAAI_bge-large-en-v1.5__k10.csv",
    "bge+pfx": "results/bge_prefixed/{ds}__BAAI_bge-large-en-v1.5__k10.csv",
    "e5": "results/{ds}__intfloat_e5-large-v2__k10.csv",
    "gte": "results/{ds}__thenlper_gte-large__k10.csv",
    "cohere": "results/{ds}__embed-english-v3.0__k10.csv",
    "bert": "results/{ds}__bert-base-uncased__k10.csv",
    "arctic": "results/{ds}__Snowflake_snowflake-arctic-embed-l-v2.0__k10.csv",
}


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def measured(ds: str, label: str, bits: int) -> tuple[float, float, float]:
    path = ROOT / RESULTS[label].format(ds=ds)
    with path.open() as f:
        for r in csv.DictReader(f):
            if int(r["bits"]) == bits:
                fl = float(r["ndcg_float"])
                return float(r["top1_agreement"]), fl, 100 * (fl - float(r["ndcg_quantized"])) / fl
    raise KeyError((ds, label, bits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--cache", default=str(ROOT / ".pdr-cache"))
    args = ap.parse_args()
    qmax = (1 << (args.bits - 1)) - 1

    hdr = (f"{'dataset':<9}{'model':<8}{'max|A|':>7}{'max/std':>8}{'top-dim%':>9}"
           f"{'std(s)':>8}{'med gap':>8}{'gap/noise':>10}{'pred top1':>10}{'meas top1':>10}{'loss%':>7}")
    print(hdr)
    xs, ys = [], []
    for (ds, label), (fa, fq) in FILES.items():
        A = np.load(Path(args.cache) / fa).astype(np.float64)
        X = np.load(Path(args.cache) / fq).astype(np.float64)
        S = X @ A.T
        srt = -np.sort(-S, axis=1)
        gap = srt[:, 0] - srt[:, 1]
        s_a = np.abs(A).max() / qmax
        s_x = np.abs(X).max(axis=1) / qmax
        noise = np.sqrt((s_a**2 * (X**2).sum(1) + s_x**2 * 1.0) / 12.0)
        ratio = gap / noise
        pred = float(np.mean([phi(r / sqrt(2)) for r in ratio]))
        energy = (A**2).sum(axis=0)
        t1, _, loss = measured(ds, label, args.bits)
        print(f"{ds:<9}{label:<8}{np.abs(A).max():7.3f}{np.abs(A).max()/A.std():8.1f}{100*energy.max()/energy.sum():9.2f}"
              f"{np.median(S.std(1)):8.4f}{np.median(gap):8.4f}{np.median(ratio):10.2f}{pred:10.3f}{t1:10.3f}{loss:7.1f}")
        xs.append(np.median(ratio)); ys.append(t1)
    xs, ys = np.array(xs), np.array(ys)
    rho = np.corrcoef(xs.argsort().argsort(), ys.argsort().argsort())[0, 1]
    print(f"\nSpearman(median gap/noise, measured top-1 agreement at b={args.bits}) over {len(xs)} pairs: {rho:.3f}")


if __name__ == "__main__":
    main()
