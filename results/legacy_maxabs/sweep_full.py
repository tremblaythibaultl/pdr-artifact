"""Full SciFact sweep: 300 claims, k in {1,5}, 6 bit-width conditions.

Emits deduplicated judging jobs per k (same hash scheme as the pilot, so
existing pilot verdicts stay valid as a cache for k=5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]  # repository root; the pdr package must be installed

from pdr.data import load_beir, corpus_to_lists, queries_to_lists
from pdr.quantize import quantize

OUT = Path(__file__).parent
KS = [5, 1]
BITS = [8, 6, 4, 3, 2]

corpus, queries, qrels = load_beir("scifact", REPO / ".beir-data", split="test")
doc_ids, _ = corpus_to_lists(corpus)
qids, qtexts = queries_to_lists(queries)
qtext_by_id = dict(zip(qids, qtexts))

A = np.load(REPO / ".pdr-cache/scifact__BAAI_bge-large-en-v1.5__corpus__n5183.npy")
X = np.load(REPO / ".pdr-cache/scifact__BAAI_bge-large-en-v1.5__queries__n300.npy")

labels = {}
with open(REPO / ".beir-data/scifact/queries.jsonl") as f:
    for line in f:
        row = json.loads(line)
        meta = row.get("metadata") or {}
        lab = {e["label"] for ev in meta.values() for e in ev}
        labels[row["_id"]] = ("CONTRADICT" if "CONTRADICT" in lab
                              else "SUPPORT" if "SUPPORT" in lab else "NOINFO")

KMAX = max(KS)

def top_kmax(cond):
    if cond == "float":
        S = X @ A.T
        return np.argsort(-S, axis=1)[:, :KMAX]
    b = int(cond)
    A_int, _ = quantize(A, bits=b)
    tops = np.empty((X.shape[0], KMAX), dtype=np.int64)
    for i in range(X.shape[0]):
        x_int, _ = quantize(X[i], bits=b)
        tops[i] = np.argsort(-(A_int @ x_int))[:KMAX]
    return tops

conds = ["float"] + [str(b) for b in BITS]
tops = {c: top_kmax(c) for c in conds}

gold_docs = {q: set(qrels.get(q, {})) for q in qids}

def doc_text(j: int) -> str:
    d = corpus[doc_ids[j]]
    return (d.get("title", "") + "\n" + d.get("text", "")).strip()

for K in KS:
    stats, jobs, mapping = {}, {}, []
    for cond in conds:
        T = tops[cond][:, :K]
        ov, hit = [], []
        for i, q in enumerate(qids):
            got = {doc_ids[j] for j in T[i]}
            ref = {doc_ids[j] for j in tops["float"][i, :K]}
            ov.append(len(got & ref) / K)
            hit.append(bool(got & gold_docs[q]))
            ctx = "\n\n".join(f"[Abstract {r+1}]\n{doc_text(j)}" for r, j in enumerate(T[i]))
            h = hashlib.sha1((q + "||" + ctx).encode()).hexdigest()[:16]
            if h not in jobs:
                jobs[h] = (
                    "You are verifying a scientific claim against retrieved abstracts.\n\n"
                    f"Claim: {qtext_by_id[q]}\n\n"
                    f"Retrieved abstracts:\n{ctx}\n\n"
                    "Based ONLY on these abstracts, does the evidence SUPPORT the claim, "
                    "CONTRADICT the claim, or is there NOINFO (no abstract is relevant "
                    "enough to decide)?\nAnswer with exactly one word: SUPPORT, CONTRADICT, or NOINFO."
                )
            mapping.append({"qid": q, "cond": cond, "job": h, "gold": labels[q]})
        stats[cond] = {"topk_overlap_vs_float": float(np.mean(ov)),
                       "gold_in_topk": float(np.mean(hit))}
    jd = OUT / f"jobs_k{K}"
    jd.mkdir(exist_ok=True)
    for h, prompt in jobs.items():
        p = jd / f"{h}.txt"
        if not p.exists():
            p.write_text(prompt)
    json.dump(mapping, open(OUT / f"mapping_k{K}.json", "w"), indent=0)
    json.dump(stats, open(OUT / f"retrieval_stats_k{K}.json", "w"), indent=2)
    print(f"k={K}: {len(jobs)} unique jobs, {len(mapping)} pairs")
    print(json.dumps(stats, indent=2))
