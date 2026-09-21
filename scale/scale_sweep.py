#!/usr/bin/env python3
"""Retrieval quality versus corpus size under clipped b-bit quantization.

Round 1 only (no document matrix). For nested corpus subsets that always
contain every relevant document of the query set, and for each bit-width,
computes on the GPU: nDCG@k and Recall@k of the float and the quantized
ranking, top-1 agreement between them, and the margin statistic (median
top-1/top-2 float gap divided by the rounding noise). Quantized scores are
computed in float32, which is exact for b <= 8 because every partial sum is an
integer below 2^24; a spot check against int64 numpy asserts it.

Self-contained (numpy, torch). Run:
  python scale_sweep.py --emb <dir from embed_beir.py> --data <BEIR dir> \
      --sizes 10000 100000 1000000 all --bits 3 4 5 6 8 --clip 99 --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


def ndcg_at_k(qrel: dict[str, int], ranking: list[str], k: int) -> float:
    def dcg(g):
        return float(sum(s / np.log2(i + 2) for i, s in enumerate(g)))
    gains = [qrel.get(d, 0) for d in ranking[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    return 0.0 if sum(ideal) == 0 else dcg(gains) / dcg(ideal)


def recall_at_k(qrel: dict[str, int], ranking: list[str], k: int) -> float:
    rel = {d for d, s in qrel.items() if s > 0}
    return 0.0 if not rel else len(set(ranking[:k]) & rel) / len(rel)


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for qid, did, score in r:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def topk_lowest_index_ties(S: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k per row, ties broken toward the lower index (deterministic)."""
    vals, idx = torch.topk(S, k, dim=1)
    # torch.topk tie order is unspecified; re-sort each row by (-value, index)
    order = torch.argsort(idx, dim=1, stable=True)
    vals, idx = torch.gather(vals, 1, order), torch.gather(idx, 1, order)
    order = torch.argsort(-vals, dim=1, stable=True)
    return torch.gather(vals, 1, order), torch.gather(idx, 1, order)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sizes", nargs="+", default=["10000", "100000", "1000000", "all"])
    ap.add_argument("--bits", nargs="+", type=int, default=[3, 4, 5, 6, 8])
    ap.add_argument("--clip", type=float, default=99.0, help="percentile for the scale; 100 = max-abs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query-chunk", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = pick_device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False  # exactness of the integer products
        torch.backends.cudnn.allow_tf32 = False
    emb = Path(args.emb)
    meta = json.load((emb / "meta.json").open())
    doc_ids = json.load((emb / "doc_ids.json").open())
    q_ids_all = json.load((emb / "query_ids.json").open())
    qrels = load_qrels(Path(args.data) / "qrels" / "test.tsv")
    q_keep = [i for i, q in enumerate(q_ids_all) if q in qrels]
    q_ids = [q_ids_all[i] for i in q_keep]
    X = torch.from_numpy(np.load(emb / "queries.f16.npy")[q_keep].astype(np.float32)).to(dev)
    A_np = np.load(emb / "corpus.f16.npy", mmap_mode="r")
    m_all, n = A_np.shape
    print(f"[scale] {meta['model']}: m={m_all} n={n} queries with qrels={len(q_ids)} device={dev}", flush=True)

    doc_pos = {d: i for i, d in enumerate(doc_ids)}
    gold = sorted({doc_pos[d] for q in q_ids for d in qrels[q] if d in doc_pos})
    rng = np.random.default_rng(args.seed)
    non_gold = np.setdiff1d(np.arange(m_all), np.array(gold), assume_unique=True)
    perm = rng.permutation(non_gold)
    print(f"[scale] gold docs={len(gold)}; subsets always contain them", flush=True)

    sizes = []
    for s in args.sizes:
        m = m_all if s == "all" else int(s)
        if m > m_all:
            print(f"[scale] size {m} > corpus {m_all}, skipping")
            continue
        sizes.append(m)
    sizes = sorted(set(sizes))

    rows = []
    t0 = time.time()
    for m in sizes:
        idx = np.sort(np.concatenate([np.array(gold), perm[: max(0, m - len(gold))]]))
        ids_sub = [doc_ids[i] for i in idx]
        A = torch.from_numpy(np.ascontiguousarray(A_np[idx]).astype(np.float32)).to(dev)  # (m, n)
        # float reference: top-k and top-2 gap
        f_top, f_gap, f_val = [], [], []
        for c in range(0, X.shape[0], args.query_chunk):
            S = X[c:c + args.query_chunk] @ A.T
            v, i = topk_lowest_index_ties(S, args.k)
            f_top.append(i.cpu()); f_gap.append((v[:, 0] - v[:, 1]).cpu()); f_val.append(v[:, 0].cpu())
        f_top = torch.cat(f_top).numpy(); f_gap = torch.cat(f_gap).numpy(); f_val = torch.cat(f_val).numpy()
        TOL = 1e-3  # cosine; a quantized top-1 this close to the float top-1 is a near-duplicate passage
        nd_f = np.array([ndcg_at_k(qrels[q], [ids_sub[j] for j in f_top[qi]], args.k) for qi, q in enumerate(q_ids)])
        rc_f = np.array([recall_at_k(qrels[q], [ids_sub[j] for j in f_top[qi]], args.k) for qi, q in enumerate(q_ids)])
        rows.append(dict(model=meta["model"], m=m, bits="float", n_queries=len(q_ids), ndcg_q=nd_f.mean(), ndcg_f=nd_f.mean(),
                         ndcg_retained_pct=100.0, recall_q=rc_f.mean(), recall_f=rc_f.mean(), top1_agreement=1.0,
                         top1_agreement_tol=1.0, frac_near_tie=float(np.mean(f_gap < TOL)),
                         median_gap=float(np.median(f_gap)), median_gap_over_noise="", frac_gap_gt_noise="", s_A=""))
        for b in args.bits:
            qmax = (1 << (b - 1)) - 1
            if args.clip < 100:
                g = torch.Generator(device="cpu").manual_seed(args.seed)
                rows_s = torch.randint(0, A.shape[0], (min(A.shape[0], 4096),), generator=g).to(dev)
                sA = torch.quantile(A[rows_s].abs().flatten(), args.clip / 100).item() / qmax
            else:
                sA = A.abs().max().item() / qmax
            A_int = (A / sA).round_().clamp_(-qmax, qmax)  # one extra (m, n) float32 buffer, in place
            sx = (torch.quantile(X.abs(), args.clip / 100, dim=1) if args.clip < 100 else X.abs().max(dim=1).values) / qmax
            X_int = torch.round(X / sx[:, None]).clamp_(-qmax, qmax)
            noise = torch.sqrt((sA ** 2 + sx ** 2) / 12.0).cpu().numpy()  # unit-norm vectors
            q_top, q_top_fval = [], []
            for c in range(0, X.shape[0], args.query_chunk):
                Sq = X_int[c:c + args.query_chunk] @ A_int.T
                if c == 0 and m == sizes[0] and b == args.bits[0]:
                    # exactness spot check against int64 numpy on the first 8 queries
                    ref = X_int[:8].cpu().numpy().astype(np.int64) @ A_int.cpu().numpy().astype(np.int64).T
                    assert np.array_equal(Sq[:8].cpu().numpy().astype(np.int64), ref), "float32 product not exact"
                _, i = topk_lowest_index_ties(Sq, args.k)
                q_top.append(i.cpu())
                # float score of the quantized top-1, for the near-duplicate-tolerant agreement
                q_top_fval.append((X[c:c + args.query_chunk] * A[i[:, 0]]).sum(1).cpu())
            q_top = torch.cat(q_top).numpy(); q_top_fval = torch.cat(q_top_fval).numpy()
            nd_q = np.array([ndcg_at_k(qrels[q], [ids_sub[j] for j in q_top[qi]], args.k) for qi, q in enumerate(q_ids)])
            rc_q = np.array([recall_at_k(qrels[q], [ids_sub[j] for j in q_top[qi]], args.k) for qi, q in enumerate(q_ids)])
            ratio = f_gap / noise
            rows.append(dict(model=meta["model"], m=m, bits=b, n_queries=len(q_ids), ndcg_q=nd_q.mean(), ndcg_f=nd_f.mean(),
                             ndcg_retained_pct=100 * nd_q.mean() / nd_f.mean(), recall_q=rc_q.mean(), recall_f=rc_f.mean(),
                             top1_agreement=float(np.mean(q_top[:, 0] == f_top[:, 0])),
                             top1_agreement_tol=float(np.mean((q_top[:, 0] == f_top[:, 0]) | (f_val - q_top_fval < TOL))),
                             frac_near_tie=float(np.mean(f_gap < TOL)),
                             median_gap=float(np.median(f_gap)), median_gap_over_noise=float(np.median(ratio)),
                             frac_gap_gt_noise=float(np.mean(ratio > 1)), s_A=sA))
            r = rows[-1]
            print(f"[scale] m={m:>8} b={b} retained={r['ndcg_retained_pct']:6.2f}% top1={r['top1_agreement']:.3f} "
                  f"top1_tol={r['top1_agreement_tol']:.3f} near_tie={r['frac_near_tie']:.3f} "
                  f"gap/noise={r['median_gap_over_noise']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            del A_int, X_int
        del A
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[scale] wrote {args.out} ({len(rows)} rows so far)", flush=True)


if __name__ == "__main__":
    main()
