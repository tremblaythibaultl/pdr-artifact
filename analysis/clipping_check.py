"""Does the encoder ordering survive a better quantizer?

The sweep quantizes with a global max-abs scale, so the single largest
coordinate of the corpus matrix sets the step for every other coordinate. That
is the most outlier-sensitive rule possible. Here we clip instead: the scale is
the p-th percentile of |X| (over all entries of A, per vector for a query) and
values beyond it saturate at +-qmax. Entries stay in [-qmax, qmax], so B, p and
every Round-1 constraint are unchanged; only the quantization map differs.

Scoring mirrors pdr.eval.evaluate's Round 1 exactly (int64 matvec, same
argsort), and the max-abs rule must reproduce results/<ds>__<model>__k10.csv.

Run: python analysis/clipping_check.py (reads the max-abs sweeps in results/ and the cached
embeddings in .pdr-cache/).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pdr.data import corpus_to_lists, load_beir, queries_to_lists
from pdr.eval import _argsort_desc, ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]
RES, CACHE, DATA = ROOT / "results", ROOT / ".pdr-cache", ROOT / ".beir-data"
DS = [("scifact", "SciFact"), ("nfcorpus", "NFCorpus")]
# key, stem, display, low-bit-trained, query tag, doc tag
ENC = [
    ("arctic", "Snowflake_snowflake-arctic-embed-l-v2.0", "arctic-l-v2.0", True, "__p2d5312", ""),
    ("cohere", "embed-english-v3.0", "Cohere v3", True, "", ""),
    ("bge", "BAAI_bge-large-en-v1.5", "bge-large", False, "", ""),
    ("mxbai", "mixedbread-ai_mxbai-embed-large-v1", "mxbai-large", False, "__p9c844c", ""),
    ("e5", "intfloat_e5-large-v2", "e5-large-v2", False, "__p2d5312", "__p537364"),
    ("gte", "thenlper_gte-large", "gte-large", False, "", ""),
    ("bert", "bert-base-uncased", "BERT-base", False, "", ""),
]
RULES = ["maxabs", "p99.9", "p99", "p95"]
BITS = [3, 4, 5]
K = 10


def scale_of(X: np.ndarray, rule: str, qmax: int, axis=None) -> np.ndarray | float:
    a = np.abs(X)
    if rule == "maxabs":
        s = a.max(axis=axis)
    else:
        s = np.percentile(a, float(rule[1:]), axis=axis)
    return np.maximum(s, 1e-12) / qmax


def quant(X: np.ndarray, scale, qmax: int) -> np.ndarray:
    return np.round(X / scale).clip(-qmax, qmax).astype(np.int64)


def sweep_ref(ds: str, stem: str, bits: int) -> tuple[float, float, float]:
    with (RES / f"{ds}__{stem}__k10.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["bits"]) == bits:
                return float(r["ndcg_quantized"]), float(r["ndcg_float"]), float(r["top1_agreement"])
    raise KeyError


def main() -> None:
    out = {}
    for ds, name in DS:
        corpus, queries, qrels = load_beir(ds, DATA, split="test")
        doc_ids, _ = corpus_to_lists(corpus)
        qids, _ = queries_to_lists(queries)
        for key, stem, disp, lb, qtag, dtag in ENC:
            A = np.load(CACHE / f"{ds}__{stem}__corpus__n{len(doc_ids)}{dtag}.npy")
            X = np.load(CACHE / f"{ds}__{stem}__queries__n{len(qids)}{qtag}.npy")
            S_f = np.stack([A @ x for x in X])  # per-query float32 matvec, as in pdr.eval.evaluate
            for bits in BITS:
                qmax = (1 << (bits - 1)) - 1
                for rule in RULES:
                    A_int = quant(A, scale_of(A, rule, qmax), qmax)
                    clipped = float(np.mean(np.abs(A_int) == qmax)) if rule != "maxabs" else float(np.mean(np.abs(A / scale_of(A, rule, qmax)) > qmax - 0.5))
                    sx = scale_of(X, rule, qmax, axis=1)[:, None]
                    X_int = quant(X, sx, qmax)
                    S_q = (A_int @ X_int.T).T  # (nq, m) int64, same product as server.round1
                    nd_q, nd_f, agree, n = [], [], 0, 0
                    for qi, qid in enumerate(qids):
                        qrel = qrels.get(qid)
                        if not qrel:
                            continue
                        n += 1
                        oq, of = _argsort_desc(S_q[qi]), _argsort_desc(S_f[qi])
                        agree += oq[0] == of[0]
                        nd_q.append(ndcg_at_k(qrel, [doc_ids[i] for i in oq[:K]], K))
                        nd_f.append(ndcg_at_k(qrel, [doc_ids[i] for i in of[:K]], K))
                    out[(ds, key, bits, rule)] = (100 * np.mean(nd_q) / np.mean(nd_f), agree / n, clipped)
            # sanity: max-abs must reproduce the sweep
            for bits in BITS:
                rq, rf, rt = sweep_ref(ds, stem, bits)
                got = out[(ds, key, bits, "maxabs")]
                nq = sum(1 for q in qids if qrels.get(q))
                assert abs(got[0] - 100 * rq / rf) < 0.1 and abs(got[1] - rt) <= 1.0 / nq + 1e-9, (ds, key, bits, got, rq / rf, rt)
    print("max-abs rule reproduces the sweep (nDCG retained within 0.1 pt, top-1 within one query).\n")

    for bits in BITS:
        print(f"===== b = {bits}: nDCG@10 retained (%) / top-1 agreement, by scale rule =====")
        hdr = f"{'corpus':<9}{'encoder':<14}{'lowbit':>6} |" + "".join(f"{r:>18}" for r in RULES) + f" | {'clipped@p99':>11}"
        print(hdr)
        for ds, name in DS:
            for key, stem, disp, lb, *_ in ENC:
                cells = "".join(f"{out[(ds,key,bits,r)][0]:8.1f} /{out[(ds,key,bits,r)][1]:6.3f}  " for r in RULES)
                print(f"{name:<9}{disp:<14}{'yes' if lb else 'no':>6} |{cells} | {100*out[(ds,key,bits,'p99')][2]:10.2f}%")
        print()

    # Persist every cell so the paper can be built from it.
    with (RES / "clipping_check__k10.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "low_bit_trained", "bits", "scale_rule", "ndcg_retained_pct", "top1_agreement", "clipped_fraction"])
        for (ds, key, bits, rule), (ret, t1, cl) in sorted(out.items()):
            stem = [e[1] for e in ENC if e[0] == key][0]
            w.writerow([ds, stem, int([e[3] for e in ENC if e[0] == key][0]), bits, rule, f"{ret:.3f}", f"{t1:.6f}", f"{cl:.6f}"])
    print(f"wrote {RES / 'clipping_check__k10.csv'}\n")

    # Headline comparison at b=4: best clipped rule per encoder vs max-abs, and gap between groups
    print("===== b=4 summary: max-abs vs best percentile rule (nDCG retained / top-1) =====")
    for ds, name in DS:
        for key, stem, disp, lb, *_ in ENC:
            base = out[(ds, key, 4, "maxabs")]
            best_rule = max(RULES[1:], key=lambda r: out[(ds, key, 4, r)][1])
            best = out[(ds, key, 4, best_rule)]
            print(f"{name:<9}{disp:<14}{'yes' if lb else 'no':>4} | max-abs {base[0]:5.1f}/{base[1]:.3f} -> {best_rule:<5} {best[0]:5.1f}/{best[1]:.3f}  (top-1 {best[1]-base[1]:+.3f})")


if __name__ == "__main__":
    main()
