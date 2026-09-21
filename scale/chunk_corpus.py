#!/usr/bin/env python3
"""Chunk a BEIR corpus into fixed-byte passages for the Round-2 length study.

Each document is split at sentence boundaries into chunks of at most --max-bytes bytes of UTF-8
(title included, since Round 2 returns the title with the excerpt), 1,536 bytes by default, i.e.
512 plaintext coefficients of 24 bits. Chunk ids are "<doc_id>#<i>". The qrels are expanded so that
every chunk of a relevant document is relevant with the document's score, which is what the
gold-in-top-k statistic of build_jobs.py and the BEIR loader expect. Queries are copied unchanged.

Usage: python scale/chunk_corpus.py --data .beir-data/scifact --out data/scifact_chunks [--max-bytes 1536]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics as st
from pathlib import Path

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def hard_split(s: str, budget: int) -> list[str]:
    """Split one over-long sentence at whitespace so every piece fits the byte budget."""
    out, cur = [], ""
    for w in s.split():
        cand = (cur + " " + w).strip()
        if len(cand.encode()) > budget and cur:
            out.append(cur); cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def chunk_doc(title: str, text: str, max_bytes: int) -> list[str]:
    head = len((title + "\n").encode())
    budget = max(max_bytes - head, 64)
    chunks, cur = [], ""
    for s in split_sentences(text):
        pieces = [s] if len(s.encode()) <= budget else hard_split(s, budget)
        for p in pieces:
            cand = (cur + " " + p).strip()
            if len(cand.encode()) > budget and cur:
                chunks.append(cur); cur = p
            else:
                cur = cand
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=1536)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    (out / "qrels").mkdir(parents=True, exist_ok=True)

    n_chunks, sizes, per_doc = {}, [], []
    with (out / "corpus.jsonl").open("w") as f:
        for line in (data / "corpus.jsonl").open():
            r = json.loads(line)
            title, text = r.get("title", ""), r.get("text", "")
            chunks = chunk_doc(title, text, args.max_bytes)
            n_chunks[r["_id"]] = len(chunks); per_doc.append(len(chunks))
            for i, c in enumerate(chunks):
                sizes.append(len((title + "\n" + c).encode()))
                f.write(json.dumps({"_id": f"{r['_id']}#{i}", "title": title, "text": c}) + "\n")
    shutil.copy(data / "queries.jsonl", out / "queries.jsonl")
    for split in ("test", "train", "dev"):
        src = data / "qrels" / f"{split}.tsv"
        if not src.exists():
            continue
        with src.open() as fi, (out / "qrels" / f"{split}.tsv").open("w", newline="") as fo:
            rd = csv.reader(fi, delimiter="\t"); w = csv.writer(fo, delimiter="\t", lineterminator="\n")
            w.writerow(next(rd))
            for qid, did, score in rd:
                for i in range(n_chunks.get(did, 0)):
                    w.writerow([qid, f"{did}#{i}", score])
    stats = {"source": str(data), "max_bytes": args.max_bytes, "docs": len(per_doc), "chunks": len(sizes),
             "chunks_per_doc_mean": round(st.mean(per_doc), 3), "chunks_per_doc_max": max(per_doc),
             "docs_single_chunk": sum(1 for c in per_doc if c == 1),
             "chunk_bytes_mean": round(st.mean(sizes)), "chunk_bytes_max": max(sizes)}
    json.dump(stats, (out / "stats.json").open("w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
