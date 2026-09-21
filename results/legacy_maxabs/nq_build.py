"""NQ mini-corpus endpoint: sample 100 answerable test queries, build a
gold+distractor corpus, embed with BGE-large, run the bit-width sweep, and
emit extractive-QA judging jobs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import string
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]  # repository root; the pdr package must be installed

from pdr.embed import encode
from pdr.quantize import quantize

HERE = Path(__file__).parent
NQ = HERE / "nq" / "nq"
K = 5
BITS = [8, 6, 4, 3, 2]
N_QUERIES = 100
N_DISTRACTORS = 15000
SEED = 0

def norm_q(s: str) -> str:
    s = s.lower().strip().rstrip("?").strip()
    return re.sub(r"\s+", " ", s.translate(str.maketrans("", "", string.punctuation)))

# NQ-open answers keyed by normalized question.
answers = {}
with open(HERE / "NQ-open.dev.jsonl") as f:
    for line in f:
        row = json.loads(line)
        answers.setdefault(norm_q(row["question"]), []).extend(row["answer"])

# BEIR NQ test queries + qrels.
queries = {}
with open(NQ / "queries.jsonl") as f:
    for line in f:
        row = json.loads(line)
        queries[row["_id"]] = row["text"]
qrels = {}
with open(NQ / "qrels" / "test.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for qid, did, score in reader:
        if int(score) > 0:
            qrels.setdefault(qid, set()).add(did)

matched = [q for q in qrels if norm_q(queries[q]) in answers]
print(f"BEIR NQ test queries with qrels: {len(qrels)}; matched to NQ-open answers: {len(matched)}")

rng = np.random.default_rng(SEED)
sample = sorted(rng.choice(sorted(matched), size=N_QUERIES, replace=False))
gold_ids = set().union(*(qrels[q] for q in sample))
print(f"sampled {len(sample)} queries, {len(gold_ids)} gold docs")

# Stream corpus: collect golds; reservoir-sample distractors. Cached on disk.
docs_cache = HERE / "nq_docs.jsonl"
if docs_cache.exists():
    docs = [json.loads(line) for line in open(docs_cache)]
    print(f"loaded cached mini-corpus: {len(docs)} docs")
else:
    gold_docs, reservoir, seen = {}, [], 0
    with open(NQ / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            did = row["_id"]
            if did in gold_ids:
                gold_docs[did] = row
                continue
            seen += 1
            if len(reservoir) < N_DISTRACTORS:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, seen))
                if j < N_DISTRACTORS:
                    reservoir[j] = row
    print(f"golds found: {len(gold_docs)}/{len(gold_ids)}; distractors: {len(reservoir)}")
    docs = list(gold_docs.values()) + reservoir
    with open(docs_cache, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
doc_ids = [d["_id"] for d in docs]
texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in docs]

emb_c = HERE / "nq_corpus_emb.npy"
emb_q = HERE / "nq_query_emb.npy"
if emb_c.exists():
    A = np.load(emb_c)
else:
    A = encode("BAAI/bge-large-en-v1.5", texts, normalize=True)
    np.save(emb_c, A)
if emb_q.exists():
    X = np.load(emb_q)
else:
    X = encode("BAAI/bge-large-en-v1.5", [queries[q] for q in sample],
               normalize=True, is_query=True)
    np.save(emb_q, X)
print("embeddings:", A.shape, X.shape)

conds = ["float"] + [str(b) for b in BITS]
def topk(cond):
    if cond == "float":
        return np.argsort(-(X @ A.T), axis=1)[:, :K]
    A_int, _ = quantize(A, bits=int(cond))
    T = np.empty((X.shape[0], K), dtype=np.int64)
    for i in range(X.shape[0]):
        x_int, _ = quantize(X[i], bits=int(cond))
        T[i] = np.argsort(-(A_int @ x_int))[:K]
    return T

tops = {c: topk(c) for c in conds}

def doc_prompt(j: int) -> str:
    d = docs[j]
    return (d.get("title", "") + "\n" + d.get("text", "")).strip()

stats, jobs, mapping = {}, {}, []
for cond in conds:
    T = tops[cond]
    ov, hit = [], []
    for i, q in enumerate(sample):
        got = {doc_ids[j] for j in T[i]}
        ref = {doc_ids[j] for j in tops["float"][i]}
        ov.append(len(got & ref) / K)
        hit.append(bool(got & qrels[q]))
        ctx = "\n\n".join(f"[Passage {r+1}]\n{doc_prompt(j)}" for r, j in enumerate(T[i]))
        h = hashlib.sha1((q + "||" + ctx).encode()).hexdigest()[:16]
        if h not in jobs:
            jobs[h] = (
                "Answer the question using ONLY the passages below.\n\n"
                f"Question: {queries[q]}\n\n"
                f"{ctx}\n\n"
                "If the passages contain the answer, reply with just the answer "
                "(a few words, no sentence). If they do not contain it, reply "
                "with exactly: UNKNOWN"
            )
        mapping.append({"qid": q, "cond": cond, "job": h,
                        "gold": answers[norm_q(queries[q])]})
    stats[cond] = {"topk_overlap_vs_float": float(np.mean(ov)),
                   "gold_in_topk": float(np.mean(hit))}

jd = HERE / "jobs_nq"
jd.mkdir(exist_ok=True)
for h, prompt in jobs.items():
    p = jd / f"{h}.txt"
    if not p.exists():
        p.write_text(prompt)
json.dump(mapping, open(HERE / "mapping_nq.json", "w"), indent=0)
json.dump(stats, open(HERE / "retrieval_stats_nq.json", "w"), indent=2)
print(f"{len(jobs)} unique jobs, {len(mapping)} pairs")
print(json.dumps(stats, indent=2))
