#!/usr/bin/env python3
"""Import embed_beir.py output into the pdr CLI cache so `pdr.cli sweep` runs on cluster embeddings.

The CLI keys its cache by (dataset, model, kind, n, instruction prefix) and orders rows as the BEIR
loader does (test-split queries only), so rows are re-indexed by id here. Embeddings are float16 on
disk and normalised; they are written back as float32.

Usage: python scale/import_emb_cache.py --emb <embed_beir.py output dir> --dataset scifact_chunks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pdr.cli import _cache_path
from pdr.data import corpus_to_lists, load_beir, queries_to_lists
from pdr.embed import resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default=".beir-data")
    ap.add_argument("--cache-dir", default=".pdr-cache")
    args = ap.parse_args()
    emb = Path(args.emb)
    meta = json.load((emb / "meta.json").open())
    model = meta["model"]; spec = resolve(model)
    assert meta.get("query_prefix", "") == spec.query_prefix and meta.get("doc_prefix", "") == spec.doc_prefix, \
        f"prefix mismatch: cluster {meta.get('query_prefix')!r}/{meta.get('doc_prefix')!r} vs CLI {spec.query_prefix!r}/{spec.doc_prefix!r}"
    corpus, queries, _ = load_beir(args.dataset, Path(args.data_root))
    doc_ids, _ = corpus_to_lists(corpus); qids, _ = queries_to_lists(queries)
    A = np.load(emb / "corpus.f16.npy").astype(np.float32); Q = np.load(emb / "queries.f16.npy").astype(np.float32)
    dpos = {d: i for i, d in enumerate(json.load((emb / "doc_ids.json").open()))}
    qpos = {q: i for i, q in enumerate(json.load((emb / "query_ids.json").open()))}
    A = A[[dpos[d] for d in doc_ids]]; Q = Q[[qpos[q] for q in qids]]
    for X, name in ((A, "corpus"), (Q, "queries")):
        nrm = np.linalg.norm(X, axis=1)
        assert abs(nrm.mean() - 1) < 1e-2, f"{name} embeddings not unit-norm (mean {nrm.mean():.4f})"
    cache = Path(args.cache_dir)
    pa = _cache_path(cache, args.dataset, model, "corpus", len(doc_ids), spec.doc_prefix)
    pq = _cache_path(cache, args.dataset, model, "queries", len(qids), spec.query_prefix)
    cache.mkdir(exist_ok=True); np.save(pa, A); np.save(pq, Q)
    print(f"wrote {pa.name} {A.shape} and {pq.name} {Q.shape}")


if __name__ == "__main__":
    main()
