"""Aggregate all scale-up runs: SciFact (2 readers x 2 k) + NQ.

Headline metric: accuracy on evidence-bearing claims (SciFact) / answer
containment (NQ). Paired exact McNemar test vs the float condition.
"""
from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["float", "8", "6", "4", "3", "2"]

def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)

def read_verdict(outdir: str, h: str):
    """Parse a verdict. Output style depends on the prompt that produced it:
    the one-word prompt yields a bare label (first match is the verdict); the
    two-step prompt yields reasoning then 'ANSWER: <label>', and reasoning
    truncated before that line is unparseable rather than guessable."""
    p = HERE / outdir / f"{h}.txt"
    if not p.exists() or not p.read_text().strip():
        return None
    t = p.read_text().upper()
    if outdir.endswith("_v3"):
        m = re.search(r"ANSWER:\s*\**\s*(SUPPORT|CONTRADICT|NOINFO)", t)
        return m.group(1) if m else "UNPARSED"
    hits = re.findall(r"\b(SUPPORT|CONTRADICT|NOINFO)\b", t)
    return hits[0] if hits else "UNPARSED"

def scifact_run(name: str, mapping_file: str, outdir: str, stats_file: str):
    mapping = json.load(open(HERE / mapping_file))
    rstats = json.load(open(HERE / stats_file))
    by_cond = defaultdict(dict)
    for row in mapping:
        by_cond[row["cond"]][row["qid"]] = (read_verdict(outdir, row["job"]), row["gold"])
    print(f"\n=== SciFact {name} ===")
    print(f"{'cond':>6} {'n':>4} {'acc_all':>8} {'acc_evid':>9} {'acc_noinfo':>10} "
          f"{'flip':>6} {'McNemar_p':>10} {'gold@k':>7}")
    out = {}
    for cond in ORDER:
        rows = by_cond[cond]
        ok = {q: (v, g) for q, (v, g) in rows.items() if v not in (None, "UNPARSED")}
        acc_all = sum(v == g for v, g in ok.values()) / max(len(ok), 1)
        ev = {q: vg for q, vg in ok.items() if vg[1] != "NOINFO"}
        acc_ev = sum(v == g for v, g in ev.values()) / max(len(ev), 1)
        ni = {q: vg for q, vg in ok.items() if vg[1] == "NOINFO"}
        acc_ni = sum(v == g for v, g in ni.values()) / max(len(ni), 1)
        # paired vs float on correctness of evidence-bearing claims
        b = c = flips = 0
        for q, (v, g) in ev.items():
            fv = by_cond["float"].get(q, (None,))[0]
            if fv in (None, "UNPARSED"):
                continue
            fc, qc = fv == g, v == g
            b += fc and not qc
            c += qc and not fc
        for q, (v, g) in ok.items():
            fv = by_cond["float"].get(q, (None,))[0]
            flips += fv not in (None, "UNPARSED") and v != fv
        p = mcnemar_exact(b, c)
        print(f"{cond:>6} {len(ok):>4} {acc_all:>8.3f} {acc_ev:>9.3f} {acc_ni:>10.3f} "
              f"{flips/max(len(ok),1):>6.3f} {p:>10.4f} {rstats[cond]['gold_in_topk']:>7.3f}")
        out[cond] = {"n": len(ok), "acc_all": acc_all, "acc_evidence": acc_ev,
                     "acc_noinfo": acc_ni, "mcnemar_p_evidence_vs_float": p,
                     "discordant": [b, c], **rstats[cond]}
    return out

def norm_ans(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def nq_run(outdir="out_nq", label="Haiku"):
    mapping = json.load(open(HERE / "mapping_nq.json"))
    rstats = json.load(open(HERE / "retrieval_stats_nq.json"))
    responses = {}
    for row in mapping:
        h = row["job"]
        if h not in responses:
            p = HERE / outdir / f"{h}.txt"
            responses[h] = p.read_text().strip() if p.exists() else None
    by_cond = defaultdict(dict)
    for row in mapping:
        by_cond[row["cond"]][row["qid"]] = (responses[row["job"]], row["gold"])
    print(f"\n=== NQ mini-corpus ({label}, k=5) ===")
    print(f"{'cond':>6} {'n':>4} {'acc':>6} {'unknown':>8} {'McNemar_p':>10} {'gold@k':>7}")
    out = {}
    def correct(resp, golds):
        if resp is None:
            return None
        r = norm_ans(resp)
        return any(norm_ans(g) in r for g in golds)
    for cond in ORDER:
        rows = by_cond[cond]
        ok = {q: (correct(v, g), v) for q, (v, g) in rows.items() if v is not None}
        acc = sum(1 for c_, _ in ok.values() if c_) / max(len(ok), 1)
        unk = sum(1 for _, v in ok.values() if "UNKNOWN" in v.upper()) / max(len(ok), 1)
        b = c = 0
        for q, (qc, _) in ok.items():
            fresp, fgold = by_cond["float"].get(q, (None, None))
            fc = correct(fresp, fgold)
            if fc is None:
                continue
            b += fc and not qc
            c += qc and not fc
        p = mcnemar_exact(b, c)
        print(f"{cond:>6} {len(ok):>4} {acc:>6.3f} {unk:>8.3f} {p:>10.4f} "
              f"{rstats[cond]['gold_in_topk']:>7.3f}")
        out[cond] = {"n": len(ok), "acc": acc, "unknown_rate": unk,
                     "mcnemar_p_vs_float": p, "discordant": [b, c], **rstats[cond]}
    return out

results = {}
for name, mp, od, st in [
    ("haiku k=5", "mapping_k5.json", "out_haiku_k5", "retrieval_stats_k5.json"),
    ("haiku k=1", "mapping_k1.json", "out_haiku_k1", "retrieval_stats_k1.json"),
    ("sonnet k=5", "mapping_k5.json", "out_sonnet_k5", "retrieval_stats_k5.json"),
    ("llama3.1-8b k=5", "mapping_k5.json", "out_llama_k5", "retrieval_stats_k5.json"),
    ("llama3.1-8b-v3prompt k=5", "mapping_k5.json", "out_llama_k5_v3", "retrieval_stats_k5.json"),
]:
    if (HERE / od).exists() and any((HERE / od).iterdir()):
        results[f"scifact_{name.replace(' ', '_').replace('=', '')}"] = scifact_run(name, mp, od, st)
for od, label, key in [("out_nq", "Haiku", "nq_haiku_k5"),
                       ("out_llama_nq", "Llama-3.1-8B", "nq_llama_k5")]:
    if (HERE / od).exists() and any((HERE / od).iterdir()):
        results[key] = nq_run(od, label)

json.dump(results, open(HERE / "final_results_full.json", "w"), indent=2)
