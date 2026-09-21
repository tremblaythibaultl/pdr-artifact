#!/usr/bin/env python3
"""Build reader prompt sets for one (endpoint, encoder) under clipped quantization.

For each condition in --conds (float and bit-widths), retrieves the top-k
passages for every query in the subset with the clipped quantizer (Round 1,
same scoring as scale_sweep.py), writes one prompt file per distinct context
(hash of qid || context, so identical contexts across conditions are one job),
a mapping of (qid, cond) -> job and gold, and retrieval statistics per
condition (top-k overlap with float, gold-in-top-k).

Endpoints:
  scifact : one-word claim verification; gold label from queries.jsonl metadata.
  qa      : extractive QA (answer or UNKNOWN); gold answers from --answers json
            (NQ) or from queries.jsonl metadata.exact_answer (BioASQ).

Usage (cluster or laptop):
  python build_jobs.py --emb <embed_beir.py output dir> --data <BEIR dir> --endpoint qa \
      --subset nq_subset.json --k 5 --conds float 8 6 4 3 2 --clip 99 --out jobs/nq_bge
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCIFACT_TEMPLATE = (
    "You are verifying a scientific claim against retrieved abstracts.\n\n"
    "Claim: {claim}\n\n"
    "Retrieved abstracts:\n{ctx}\n\n"
    "Based ONLY on these abstracts, does the evidence SUPPORT the claim, "
    "CONTRADICT the claim, or is there NOINFO (no abstract is relevant "
    "enough to decide)?\nAnswer with exactly one word: SUPPORT, CONTRADICT, or NOINFO."
)
QA_TEMPLATE = (
    "Answer the question using ONLY the passages below.\n\n"
    "Question: {question}\n\n"
    "{ctx}\n\n"
    "If the passages contain the answer, reply with just the answer "
    "(a few words, no sentence). If they do not contain it, reply "
    "with exactly: UNKNOWN"
)
# Reference conditions without retrieval. "closed": the reader sees no passages (closed book);
# "gold": it sees the labelled relevant passages (the qrels), at most k of them, in place of the retrieved ones.
SCIFACT_CLOSED_TEMPLATE = (
    "You are verifying a scientific claim from your own knowledge, without any retrieved abstracts.\n\n"
    "Claim: {claim}\n\n"
    "Does the scientific evidence you know of SUPPORT the claim, CONTRADICT the claim, "
    "or is there NOINFO (you cannot decide)?\nAnswer with exactly one word: SUPPORT, CONTRADICT, or NOINFO."
)
QA_CLOSED_TEMPLATE = (
    "Answer the question from your own knowledge, without any passages.\n\n"
    "Question: {question}\n\n"
    "If you know the answer, reply with just the answer (a few words, no sentence). "
    "If you do not, reply with exactly: UNKNOWN"
)
EXTRA_CONDS = ("closed", "gold")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for qid, did, score in r:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def scifact_label(meta: dict) -> str:
    labels = {e["label"] for ev in (meta or {}).values() for e in ev}
    return "CONTRADICT" if "CONTRADICT" in labels else "SUPPORT" if "SUPPORT" in labels else "NOINFO"


def topk_lowest_index_ties(S: torch.Tensor, k: int) -> torch.Tensor:
    vals, idx = torch.topk(S, k, dim=1)
    order = torch.argsort(idx, dim=1, stable=True)
    vals, idx = torch.gather(vals, 1, order), torch.gather(idx, 1, order)
    order = torch.argsort(-vals, dim=1, stable=True)
    return torch.gather(idx, 1, order)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", choices=["scifact", "qa"], required=True)
    ap.add_argument("--subset", default=None, help="json: list of qids, or {qid: {question, answers}}")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--conds", nargs="+", default=["float", "8", "6", "4", "3", "2"])
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunked", action="store_true", help="corpus is chunk_corpus.py output: label contexts as abstract excerpts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                       "mps" if args.device == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
                       "cpu" if args.device == "auto" else args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    emb, data, out = Path(args.emb), Path(args.data), Path(args.out)
    meta = json.load((emb / "meta.json").open())
    doc_ids = json.load((emb / "doc_ids.json").open())
    q_ids_all = json.load((emb / "query_ids.json").open())
    qrels = load_qrels(data / "qrels" / "test.tsv")
    qmeta = {}
    for line in (data / "queries.jsonl").open():
        r = json.loads(line); qmeta[r["_id"]] = r

    subset_answers = None
    if args.subset:
        sub = json.load(open(args.subset))
        if isinstance(sub, dict):
            subset_answers = sub; keep = sorted(sub)
        else:
            keep = sorted(sub)
    else:
        keep = sorted(q for q in q_ids_all if q in qrels)
    qpos = {q: i for i, q in enumerate(q_ids_all)}
    keep = [q for q in keep if q in qpos]
    X = torch.from_numpy(np.load(emb / "queries.f16.npy")[[qpos[q] for q in keep]].astype(np.float32)).to(dev)
    A = torch.from_numpy(np.load(emb / "corpus.f16.npy").astype(np.float32)).to(dev)
    m = A.shape[0]
    print(f"[jobs] {args.endpoint} {meta['model']}: m={m} queries={len(keep)} k={args.k} conds={args.conds} clip={args.clip} device={dev}", flush=True)

    def retrieve(cond: str) -> np.ndarray:
        if cond == "float":
            S = X @ A.T
            return topk_lowest_index_ties(S, args.k).cpu().numpy()
        b = int(cond); qmax = (1 << (b - 1)) - 1
        if args.clip < 100:
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            rows = torch.randint(0, m, (min(m, 4096),), generator=g).to(dev) if m > 4096 else torch.arange(m, device=dev)
            sA = torch.quantile(A[rows].abs().flatten(), args.clip / 100).item() / qmax
            sx = torch.quantile(X.abs(), args.clip / 100, dim=1) / qmax
        else:
            sA = A.abs().max().item() / qmax
            sx = X.abs().max(dim=1).values / qmax
        A_int = (A / sA).round_().clamp_(-qmax, qmax)
        X_int = (X / sx[:, None]).round_().clamp_(-qmax, qmax)
        S = X_int @ A_int.T
        del A_int
        return topk_lowest_index_ties(S, args.k).cpu().numpy()

    tops = {c: retrieve(c) for c in args.conds if c not in EXTRA_CONDS}
    needed = sorted({int(j) for T in tops.values() for row in T for j in row})
    need_ids = {doc_ids[j] for j in needed}
    gold_ids = {}
    if "gold" in args.conds:
        for q in keep:
            gold_ids[q] = [d for d, _ in sorted(qrels.get(q, {}).items(), key=lambda t: -t[1])][: args.k]
            need_ids.update(gold_ids[q])
    texts = {}
    for line in (data / "corpus.jsonl").open():
        r = json.loads(line)
        if r["_id"] in need_ids:
            texts[r["_id"]] = (r.get("title", "") + "\n" + r.get("text", "")).strip()
    assert len(texts) == len(need_ids), "missing passage texts"

    jobs_dir = out / "jobs"; jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs, mapping, stats = {}, [], {}
    label = ("Abstract excerpt" if args.chunked else "Abstract") if args.endpoint == "scifact" else "Passage"
    for cond in args.conds:
        T = tops.get(cond); ov, hit = [], []
        for qi, q in enumerate(keep):
            question = subset_answers[q]["question"] if subset_answers else qmeta[q]["text"]
            if cond == "closed":
                ctx = ""
                hit.append(False)
            elif cond == "gold":
                ids = gold_ids[q]
                ctx = "\n\n".join(f"[{label} {r+1}]\n{texts[d]}" for r, d in enumerate(ids)) or "(none)"
                hit.append(bool(ids))
            else:
                got = {doc_ids[j] for j in T[qi]}
                if "float" in tops:
                    ref = {doc_ids[j] for j in tops["float"][qi]}
                    ov.append(len(got & ref) / args.k)
                hit.append(bool(got & set(qrels.get(q, {}))))
                ctx = "\n\n".join(f"[{label} {r+1}]\n{texts[doc_ids[j]]}" for r, j in enumerate(T[qi]))
            h = hashlib.sha1((q + "||" + cond * (cond == "closed") + "||" + ctx).encode()).hexdigest()[:16]
            if h not in jobs:
                if cond == "closed":
                    jobs[h] = (SCIFACT_CLOSED_TEMPLATE.format(claim=qmeta[q]["text"]) if args.endpoint == "scifact"
                               else QA_CLOSED_TEMPLATE.format(question=question))
                elif args.endpoint == "scifact":
                    jobs[h] = SCIFACT_TEMPLATE.format(claim=qmeta[q]["text"], ctx=ctx)
                else:
                    jobs[h] = QA_TEMPLATE.format(question=question, ctx=ctx)
            if args.endpoint == "scifact":
                gold = scifact_label(qmeta[q].get("metadata"))
            else:
                gold = subset_answers[q]["answers"] if subset_answers else qmeta[q]["metadata"]["exact_answer"]
            mapping.append({"qid": q, "cond": cond, "job": h, "gold": gold})
        stats[cond] = {"topk_overlap_vs_float": float(np.mean(ov)) if ov else float("nan"), "gold_in_topk": float(np.mean(hit))}
        print(f"[jobs] cond={cond:>6} overlap@{args.k}={stats[cond]['topk_overlap_vs_float']:.3f} gold@{args.k}={stats[cond]['gold_in_topk']:.3f}", flush=True)
    for h, prompt in jobs.items():
        (jobs_dir / f"{h}.txt").write_text(prompt)
    json.dump(mapping, (out / "mapping.json").open("w"), indent=0)
    json.dump(stats, (out / "retrieval_stats.json").open("w"), indent=2)
    json.dump({"endpoint": args.endpoint, "model": meta["model"], "k": args.k, "conds": args.conds, "clip": args.clip,
               "n_queries": len(keep), "n_jobs": len(jobs), "chunked": args.chunked, "template": "one-word" if args.endpoint == "scifact" else "qa-or-unknown"},
              (out / "meta.json").open("w"), indent=2)
    print(f"[jobs] {len(jobs)} unique prompts, {len(mapping)} (qid, cond) pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
