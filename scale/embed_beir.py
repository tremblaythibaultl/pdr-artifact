#!/usr/bin/env python3
"""Embed a BEIR-format corpus and its queries with one encoder; save float16.

Self-contained (numpy, torch, sentence-transformers only) so it runs unchanged
on a laptop and on an offline compute node. Prefixes are the ones each
model card prescribes. The corpus is encoded in chunks that are written as
they finish, so a killed job resumes where it stopped.

Outputs under <out>/<model with / replaced by _>/:
  corpus.f16.npy, queries.f16.npy, doc_ids.json, query_ids.json, meta.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PREFIX = {  # model id -> (query prefix, document prefix)
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "mixedbread-ai/mxbai-embed-large-v1": ("Represent this sentence for searching relevant passages: ", ""),
    "Snowflake/snowflake-arctic-embed-l-v2.0": ("query: ", ""),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "thenlper/gte-large": ("", ""),
}


def read_jsonl(path: Path, limit: int | None = None):
    ids, texts = [], []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            ids.append(r["_id"])
            texts.append((r.get("title", "") + " " + r.get("text", "")).strip())
    return ids, texts


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with corpus.jsonl and queries.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=200_000, help="corpus rows per checkpoint file")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N docs")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    qp, dp = PREFIX.get(args.model, ("", ""))
    out = Path(args.out) / args.model.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "corpus.f16.npy").exists() and (out / "queries.f16.npy").exists():
        print(f"[embed] {out} already complete, skipping")
        return

    data = Path(args.data)
    doc_ids, docs = read_jsonl(data / "corpus.jsonl", args.limit)
    q_ids, queries = read_jsonl(data / "queries.jsonl")
    print(f"[embed] {args.model}: {len(docs)} docs, {len(queries)} queries, prefixes q={qp!r} d={dp!r}", flush=True)

    device = pick_device()
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq
    if device == "cuda":
        model.half()
    print(f"[embed] device={device} max_seq_length={model.max_seq_length}", flush=True)

    def enc(texts, prefix):
        if prefix:
            texts = [prefix + t for t in texts]
        e = model.encode(texts, batch_size=args.batch, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        return e.astype(np.float16)

    t0 = time.time()
    Q = enc(queries, qp)
    np.save(out / "queries.f16.npy", Q)
    json.dump(q_ids, (out / "query_ids.json").open("w"))
    print(f"[embed] queries done {Q.shape} in {time.time()-t0:.0f}s", flush=True)

    parts = []
    for start in range(0, len(docs), args.chunk):
        pf = out / f"corpus.part{start // args.chunk:04d}.f16.npy"
        if pf.exists():
            parts.append(np.load(pf))
            print(f"[embed] resume: {pf.name}", flush=True)
            continue
        t1 = time.time()
        E = enc(docs[start:start + args.chunk], dp)
        np.save(pf, E)
        parts.append(E)
        done = min(start + args.chunk, len(docs))
        rate = E.shape[0] / (time.time() - t1)
        print(f"[embed] corpus {done}/{len(docs)} ({rate:.0f} docs/s, elapsed {time.time()-t0:.0f}s)", flush=True)
    A = np.concatenate(parts, axis=0)
    np.save(out / "corpus.f16.npy", A)
    json.dump(doc_ids, (out / "doc_ids.json").open("w"))
    for pf in out.glob("corpus.part*.f16.npy"):
        pf.unlink()
    json.dump({"model": args.model, "query_prefix": qp, "doc_prefix": dp, "max_seq_length": args.max_seq,
               "n_docs": int(A.shape[0]), "n_queries": int(Q.shape[0]), "dim": int(A.shape[1]),
               "device": device, "seconds": round(time.time() - t0)}, (out / "meta.json").open("w"), indent=2)
    print(f"[embed] corpus done {A.shape} in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
