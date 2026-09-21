r"""Numbers and LaTeX for the encoder-suite appendix and section 5.2.

Reads the sweep CSVs (clipped, --clip 99, the paper's quantizer; and max-abs,
the appendix baseline) plus the cached float embeddings, prints every figure
the body text quotes, and writes the LaTeX bodies of the appendix tables so
nothing is transcribed by hand. Run:
    python analysis/suite_tables.py --clip 99 [--chunks]
"""
from __future__ import annotations

import argparse
import csv
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CACHE = ROOT / ".pdr-cache"
DS = [("scifact", "SciFact", 5183, 300), ("nfcorpus", "NFCorpus", 3633, 323)]
# --chunks: the corpora chunked by scale/chunk_corpus.py, ranked per chunk and scored per document
# (pdr.cli sweep --collapse, results stem suffix __doc). m is the chunk count of the embedding cache.
DS_CHUNKS = [("scifact_chunks", "SciFact", 7510, 300), ("nfcorpus_chunks", "NFCorpus", 5761, 323)]
# key, stem, display, low-bit-trained, query cache tag, doc cache tag
ENC = [
    ("arctic", "Snowflake_snowflake-arctic-embed-l-v2.0", "arctic-embed-l-v2.0", True, "__p2d5312", ""),
    ("cohere", "embed-english-v3.0", "Cohere embed-v3", True, "", ""),
    ("bge", "BAAI_bge-large-en-v1.5", "bge-large-en-v1.5", False, "__p9c844c", ""),
    ("mxbai", "mixedbread-ai_mxbai-embed-large-v1", "mxbai-embed-large-v1", False, "__p9c844c", ""),
    ("e5", "intfloat_e5-large-v2", "e5-large-v2", False, "__p2d5312", "__p537364"),
    ("gte", "thenlper_gte-large", "gte-large", False, "", ""),
]


def load(ds, key, stem, clip, doc=False):
    if clip is None:
        # max-abs baseline; BGE with its card prefix lives in results/bge_prefixed/
        path = (RES / "bge_prefixed" if key == "bge" else RES) / f"{ds}__{stem}__k10.csv"
    else:
        path = RES / f"{ds}__{stem}__k10__clip{clip:g}{'__doc' if doc else ''}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return {int(r["bits"]): r for r in csv.DictReader(f)}


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def margin(ds, key, stem, m, nq, qtag, dtag, bits, clip):
    A = np.load(CACHE / f"{ds}__{stem}__corpus__n{m}{dtag}.npy").astype(np.float64)
    X = np.load(CACHE / f"{ds}__{stem}__queries__n{nq}{qtag}.npy").astype(np.float64)
    S = X @ A.T
    srt = -np.sort(-S, axis=1)
    gap = srt[:, 0] - srt[:, 1]
    qmax = (1 << (bits - 1)) - 1
    if clip is None:
        s_a = np.abs(A).max() / qmax
        s_x = np.abs(X).max(axis=1) / qmax
    else:
        s_a = np.percentile(np.abs(A), clip) / qmax
        s_x = np.percentile(np.abs(X), clip, axis=1) / qmax
    noise = np.sqrt((s_a**2 * (X**2).sum(1) + s_x**2) / 12.0)
    ratio = gap / noise
    energy = (A**2).sum(axis=0)
    return dict(maxabs=float(np.abs(A).max()), topdim=100 * energy.max() / energy.sum(),
                gap=float(np.median(gap)), ratio=float(np.median(ratio)),
                pred=float(np.mean([phi(r / sqrt(2)) for r in ratio])), step=s_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--chunks", action="store_true", help="chunked corpora, ranked per chunk and scored per document")
    args = ap.parse_args()
    global DS
    if args.chunks:
        DS = DS_CHUNKS
    C = {(ds, k): load(ds, k, stem, args.clip, doc=args.chunks) for ds, *_ in DS for k, stem, *_ in ENC}
    M = {(ds, k): load(ds.replace("_chunks", ""), k, stem, None) for ds, *_ in DS for k, stem, *_ in ENC}
    missing = [(ds, k) for (ds, k), v in C.items() if v is None]
    if missing:
        print("missing sweeps:", missing)
        ENC[:] = [e for e in ENC if all(C.get((ds, e[0])) for ds, *_ in DS)]

    def ret(d, b):
        return 100 * float(d[b]["ndcg_quantized"]) / float(d[b]["ndcg_float"])

    print(f"=== body numbers, clipped p{args.clip:g} (max-abs in brackets) ===")
    for ds, name, m, nq in DS:
        print(f"-- {name}")
        fl = [float(C[(ds, k)][8]["ndcg_float"]) for k, *_ in ENC]
        print(f"   float nDCG@10 range over six trained retrievers: {min(fl):.3f}-{max(fl):.3f} (spread {max(fl)-min(fl):.3f})")
        for k, stem, disp, lb, *_ in ENC:
            c, mm = C[(ds, k)], M[(ds, k)]
            print(f"  {disp:<24} float={float(c[8]['ndcg_float']):.3f} | ret b3={ret(c,3):5.1f} [{ret(mm,3):5.1f}] b4={ret(c,4):5.1f} [{ret(mm,4):5.1f}] "
                  f"b8={ret(c,8):5.1f} | top1 b3={float(c[3]['top1_agreement']):.3f} [{float(mm[3]['top1_agreement']):.3f}] "
                  f"b4={float(c[4]['top1_agreement']):.3f} [{float(mm[4]['top1_agreement']):.3f}] b8={float(c[8]['top1_agreement']):.3f} "
                  f"| |dNDCG@8|={abs(float(c[8]['ndcg_quantized'])-float(c[8]['ndcg_float'])):.4f}")
        for grp, keys in (("low-bit pair", ["arctic", "cohere"]), ("conventional four", ["bge", "mxbai", "e5", "gte"])):
            for b in (4, 3):
                r = [100 - ret(C[(ds, k)], b) for k in keys]; t = [float(C[(ds, k)][b]["top1_agreement"]) for k in keys]
                print(f"   {grp:<18} b={b}: loss {min(r):.1f}-{max(r):.1f}%  top-1 {min(t):.2f}-{max(t):.2f}")
    print("-- arctic / Cohere / BGE SciFact nDCG@10 for Table 3 (b=2..8, float), clipped:")
    for k in ("bge", "cohere", "arctic"):
        d = C[(DS[0][0], k)]
        print(f"   {k:<7}", [f"{float(d[b]['ndcg_quantized']):.3f}" for b in range(2, 9)], f"float={float(d[8]['ndcg_float']):.3f}")

    print("\n=== LaTeX: tab:quality-sweep body (clipped; nDCG@10 retained, %) ===")
    for ds, name, *_ in DS:
        print(r"\multicolumn{7}{@{}l}{\emph{" + name + r"}}\\")
        for b in range(2, 9):
            lab = f"$\\mathbf{{{b}}}$" if b == 4 else f"${b}$"
            print(lab + " & " + " & ".join(f"${ret(C[(ds, k)], b):.1f}$" for k, *_ in ENC) + r" \\")
    if args.chunks:
        print("\n(margin statistic skipped for --chunks: it is defined on whole-abstract scores)")
        return

    print("\n=== margin statistic, clipped, 12 pairs ===")
    rows = []
    for bits in (4, 3):
        xs, ys = [], []
        for ds, name, m, nq in DS:
            for k, stem, disp, lb, qt, dt in ENC:
                st = margin(ds, k, stem, m, nq, qt, dt, bits, args.clip)
                meas = float(C[(ds, k)][bits]["top1_agreement"])
                xs.append(st["ratio"]); ys.append(meas)
                if bits == 4:
                    rows.append((name, disp, st, meas))
        xs, ys = np.array(xs), np.array(ys)
        print(f"b={bits}: Spearman(median gap/noise, top-1) = {np.corrcoef(xs.argsort().argsort(), ys.argsort().argsort())[0,1]:.3f}")

    print("\n=== LaTeX: tab:suite body (clipped) ===")
    for k, stem, disp, lb, *_ in ENC:
        cells = [r"\texttt{" + disp + "}", r"\cmark" if lb else r"\xmark"]
        for ds, *_ in DS:
            d = C[(ds, k)]
            cells += [f"${float(d[8]['ndcg_float']):.3f}$", f"${ret(d,4):.1f}$", f"${ret(d,3):.1f}$",
                      f"${float(d[4]['top1_agreement']):.2f}$", f"${float(d[3]['top1_agreement']):.2f}$"]
        print(" & ".join(cells) + r" \\")

    print("\n=== LaTeX: tab:quantizer body (max-abs -> clipped; retained% / top-1 at b=4 and b=3; one block per corpus) ===")
    for i, (ds, dsname, *_) in enumerate(DS):
        if i:
            print(r"\midrule")
        print(r"\multicolumn{5}{@{}l}{\emph{" + dsname + r"}}\\")
        for k, stem, disp, lb, *_ in ENC:
            cells = [r"\texttt{" + disp + "}"]
            c, mm = C[(ds, k)], M[(ds, k)]
            for b in (4, 3):
                cells += [f"${ret(mm,b):.1f} \\to {ret(c,b):.1f}$", f"${float(mm[b]['top1_agreement']):.2f} \\to {float(c[b]['top1_agreement']):.2f}$"]
            print(" & ".join(cells) + r" \\")

    print("\n=== LaTeX: tab:margin body (clipped, b=4) ===")
    cur = None
    for name, disp, st, meas in rows:
        if name != cur:
            if cur is not None:
                print(r"\midrule")
            cur = name
        print(" & ".join([r"\texttt{" + disp + "}", name, f"${st['maxabs']:.3f}$", f"${st['step']:.4f}$", f"${st['topdim']:.1f}$",
                          f"${st['gap']:.4f}$", f"${st['ratio']:.2f}$", f"${st['pred']:.2f}$", f"${meas:.2f}$"]) + r" \\")


if __name__ == "__main__":
    main()
