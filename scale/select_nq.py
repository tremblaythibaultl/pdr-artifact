#!/usr/bin/env python3
"""Select the NQ downstream query set: the original 100 sampled qids plus a
seeded draw of further BEIR-NQ test queries whose question matches an NQ-open
answer list, up to --n in total. Writes <out> as {qid: {"question", "answers"}}.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm_q(s: str) -> str:
    s = s.lower().strip().rstrip("?").strip()
    return re.sub(r"\s+", " ", s.translate(str.maketrans("", "", string.punctuation)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "data/nq_meta/queries.jsonl"))
    ap.add_argument("--qrels", default=str(ROOT / "data/nq_meta/qrels_test.tsv"))
    ap.add_argument("--nq-open", default=str(ROOT / "data/nq_meta/NQ-open.dev.jsonl"))
    ap.add_argument("--original", default=str(ROOT / "results/legacy_maxabs/mapping_nq.json"),
                    help="mapping of the naive-quantizer baseline run, whose 100 queries are kept")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/nq_meta/nq_subset.json"))
    args = ap.parse_args()

    answers = {}
    for line in open(args.nq_open):
        r = json.loads(line)
        answers.setdefault(norm_q(r["question"]), []).extend(r["answer"])
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(args.queries)}
    with_qrels = set()
    for i, line in enumerate(open(args.qrels)):
        if i == 0:
            continue
        qid, did, score = line.rstrip("\n").split("\t")
        if int(score) > 0:
            with_qrels.add(qid)
    matched = sorted(q for q in with_qrels if norm_q(queries[q]) in answers)
    original = sorted({r["qid"] for r in json.load(open(args.original))})
    assert all(q in matched for q in original), "original sample not reproducible"
    rng = random.Random(args.seed)
    extra = rng.sample([q for q in matched if q not in original], args.n - len(original))
    chosen = sorted(set(original) | set(extra))
    out = {q: {"question": queries[q], "answers": sorted(set(answers[norm_q(queries[q])]))} for q in chosen}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"BEIR-NQ test queries with qrels: {len(with_qrels)}; matched to NQ-open answers: {len(matched)}; "
          f"original: {len(original)}; selected: {len(chosen)} -> {args.out}")


if __name__ == "__main__":
    main()
