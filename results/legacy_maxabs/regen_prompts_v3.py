"""Regenerate SciFact k=5 prompts with a v2 template.

Job hashes are sha1(qid || context) and so are unchanged by the template;
mapping_k5.json therefore still applies to the v2 job files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]  # repository root; the pdr package must be installed
from pdr.data import load_beir, corpus_to_lists, queries_to_lists
from pdr.quantize import quantize

HERE = Path(__file__).parent
K = 5
BITS = [8, 6, 4, 3, 2]

TEMPLATE = """You are assessing a scientific claim against abstracts retrieved from the literature.

Decide which of these three labels applies. Read all three definitions before answering.

SUPPORT    - at least one abstract presents findings consistent with the claim, i.e. the
             evidence indicates the claim is TRUE.
CONTRADICT - at least one abstract presents findings inconsistent with the claim, i.e. the
             evidence indicates the claim is FALSE. This includes abstracts reporting the
             opposite direction of an effect, no effect where the claim asserts one, or a
             result that refutes or negates the claim.
NOINFO     - none of the abstracts is relevant enough to judge the claim either way.

All three labels are common in this task. Two mistakes are equally bad: answering SUPPORT for
an abstract that actually reports the opposite result, and answering SUPPORT or CONTRADICT for
an abstract that is merely on the same broad topic without addressing this specific claim.

Claim: {claim}

Retrieved abstracts:
{ctx}

Reason in two short steps, then answer.
Step 1: Does any abstract directly address this specific claim, or are they only loosely
related to the topic? If none directly addresses it, the answer is NOINFO.
Step 2: If one does address it, state whether its findings agree with the claim (SUPPORT) or
report the opposite (CONTRADICT).

Then, on a new line, write exactly:
ANSWER: <SUPPORT or CONTRADICT or NOINFO>"""

corpus, queries, qrels = load_beir("scifact", REPO / ".beir-data", split="test")
doc_ids, _ = corpus_to_lists(corpus)
qids, qtexts = queries_to_lists(queries)
qtext_by_id = dict(zip(qids, qtexts))

A = np.load(REPO / ".pdr-cache/scifact__BAAI_bge-large-en-v1.5__corpus__n5183.npy")
X = np.load(REPO / ".pdr-cache/scifact__BAAI_bge-large-en-v1.5__queries__n300.npy")


def topk(cond):
    if cond == "float":
        return np.argsort(-(X @ A.T), axis=1)[:, :K]
    A_int, _ = quantize(A, bits=int(cond))
    T = np.empty((X.shape[0], K), dtype=np.int64)
    for i in range(X.shape[0]):
        x_int, _ = quantize(X[i], bits=int(cond))
        T[i] = np.argsort(-(A_int @ x_int))[:K]
    return T


def doc_text(j):
    d = corpus[doc_ids[j]]
    return (d.get("title", "") + "\n" + d.get("text", "")).strip()


out = HERE / "jobs_k5_v3"
out.mkdir(exist_ok=True)
written = 0
for cond in ["float"] + [str(b) for b in BITS]:
    T = topk(cond)
    for i, q in enumerate(qids):
        ctx = "\n\n".join(f"[Abstract {r+1}]\n{doc_text(j)}" for r, j in enumerate(T[i]))
        h = hashlib.sha1((q + "||" + ctx).encode()).hexdigest()[:16]
        p = out / f"{h}.txt"
        if not p.exists():
            p.write_text(TEMPLATE.format(claim=qtext_by_id[q], ctx=ctx))
            written += 1
print(f"wrote {written} v2 prompt files to {out}")
