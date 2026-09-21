#!/usr/bin/env python3
"""Build the BioASQ factoid endpoint in BEIR format.

Passage unit: the snippets of one (question, document) pair, de-duplicated and
concatenated. Corpus: every such passage from every BioASQ-QA question (all four
types), so the distractors are topically plausible biomedical text. Queries: a
seeded sample of factoid questions with at most --max-gold gold documents and a
non-empty exact_answer (the answer synonyms go into queries.jsonl metadata).
qrels: a question's own passages.

Writes <out>/{corpus.jsonl, queries.jsonl, qrels/test.tsv, stats.json}.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/bioasq_raw/training11b.json"),
                    help="BioASQ-QA training11b.json (registration required at bioasq.org)")
    ap.add_argument("--out", default=str(ROOT / "data/bioasq"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-gold", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = json.load(open(args.raw))["questions"]
    out = Path(args.out); (out / "qrels").mkdir(parents=True, exist_ok=True)

    # passages for every question of every type
    corpus, per_q = {}, defaultdict(list)
    for q in qs:
        by_doc = defaultdict(list)
        for s in q.get("snippets", []):
            t = " ".join(s["text"].split())
            if t and t not in by_doc[s["document"]]:
                by_doc[s["document"]].append(t)
        for doc, texts in by_doc.items():
            pid = f"{q['id']}_{doc.rsplit('/', 1)[-1]}"
            corpus[pid] = {"_id": pid, "title": "", "text": " ".join(texts)}
            per_q[q["id"]].append(pid)

    eligible = [q for q in qs if q["type"] == "factoid" and q.get("exact_answer")
                and 1 <= len(per_q[q["id"]]) <= args.max_gold]
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(eligible, min(args.n, len(eligible))), key=lambda q: q["id"])

    with (out / "corpus.jsonl").open("w") as f:
        for pid in sorted(corpus):
            f.write(json.dumps(corpus[pid]) + "\n")
    with (out / "queries.jsonl").open("w") as f, (out / "qrels" / "test.tsv").open("w") as g:
        g.write("query-id\tcorpus-id\tscore\n")
        for q in sample:
            ans = q["exact_answer"]
            ans = [a for sub in ans for a in (sub if isinstance(sub, list) else [sub])]  # flatten defensively
            f.write(json.dumps({"_id": q["id"], "text": q["body"], "metadata": {"exact_answer": ans, "type": q["type"]}}) + "\n")
            for pid in per_q[q["id"]]:
                g.write(f"{q['id']}\t{pid}\t1\n")

    lens = [len(c["text"]) for c in corpus.values()]
    stats = {"n_passages": len(corpus), "passage_chars_median": statistics.median(lens), "passage_chars_mean": round(statistics.mean(lens)),
             "n_eligible_factoid": len(eligible), "n_sampled": len(sample), "max_gold": args.max_gold,
             "gold_per_query": {str(k): sum(1 for q in sample if len(per_q[q["id"]]) == k) for k in range(1, args.max_gold + 1)},
             "answers_per_query_median": statistics.median(len(q["exact_answer"]) for q in sample)}
    json.dump(stats, (out / "stats.json").open("w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
