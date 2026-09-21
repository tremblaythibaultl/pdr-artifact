"""Score reader outputs for the clipped downstream runs.

Layout (from scale/build_jobs.py and scale/run_readers.py):
  <jobs_root>/<dir>/mapping.json            [{qid, cond, job, gold}]
  <jobs_root>/<dir>/retrieval_stats.json     {cond: {topk_overlap_vs_float, gold_in_topk}}
  <out_root>/out_<reader>_<dir>/<job>.txt    reader response

SciFact: accuracy on evidence-bearing claims (gold != NOINFO) and on all claims,
NOINFO accuracy, flips vs float; QA: normalised answer containment, UNKNOWN rate.
Both: exact McNemar vs the float condition on paired per-query correctness,
a paired 95% CI on the accuracy difference (normal approximation on the paired
differences, as in the paper) and TOST equivalence p-values at +-5 and +-3 points.
Writes <out_root>/downstream_clip99.json.

Run: python analysis/downstream_analysis.py [--jobs-root results/jobs] [--out-root results/out]
(extract results/reader_outputs.tar.gz into results/out/ first).
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from math import comb, erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["closed", "gold", "float", "8", "6", "4", "3", "2"]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def norm_ans(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def paired_ci(diffs: list[int]) -> tuple[float, float]:
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(diffs) / n
    sd = sqrt(sum((d - m) ** 2 for d in diffs) / max(n - 1, 1))
    h = 1.96 * sd / sqrt(n)
    return (100 * (m - h), 100 * (m + h))


def _phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def tost(diffs: list[int], delta_pts: float) -> float:
    """Two one-sided tests of equivalence on the paired per-query differences.

    H0: |mean difference| >= delta (in points); H1: the accuracies are equivalent
    within +-delta. Returns the larger of the two one-sided p-values (normal
    approximation on the paired differences, matching paired_ci). p < 0.05 is
    the same event as the 90% CI lying inside [-delta, +delta].
    """
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = 100 * sum(diffs) / n
    sd = 100 * sqrt(sum((100 * d - m) ** 2 for d in diffs) / (n - 1)) / 100
    se = sd / sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < delta_pts else 1.0
    p_lower = 1 - _phi((m + delta_pts) / se)   # H0a: mean <= -delta
    p_upper = 1 - _phi((delta_pts - m) / se)   # H0b: mean >= +delta
    return max(p_lower, p_upper)


def score_dir(jobs_dir: Path, out_dir: Path, endpoint: str) -> dict:
    mapping = json.load((jobs_dir / "mapping.json").open())
    rstats = json.load((jobs_dir / "retrieval_stats.json").open())
    # reference conditions (closed book, gold context) built separately into <jobs_root>_extra/<dir>
    extra = jobs_dir.parent.with_name(jobs_dir.parent.name + "_extra") / jobs_dir.name
    if (extra / "mapping.json").exists():
        mapping += json.load((extra / "mapping.json").open())
        rstats.update(json.load((extra / "retrieval_stats.json").open()))
    resp = {}
    for r in mapping:
        if r["job"] not in resp:
            p = out_dir / f"{r['job']}.txt"
            resp[r["job"]] = p.read_text().strip() if p.exists() and p.read_text().strip() else None
    by_cond = defaultdict(dict)
    for r in mapping:
        by_cond[r["cond"]][r["qid"]] = (resp[r["job"]], r["gold"])

    def correct(v, gold):
        if v is None:
            return None
        if endpoint == "scifact":
            return v.strip().upper() == gold
        return any(norm_ans(g) in norm_ans(v) for g in gold)

    out = {}
    for cond in [c for c in ORDER if c in by_cond]:
        rows = by_cond[cond]
        ok = {q: (correct(v, g), v, g) for q, (v, g) in rows.items() if v is not None}
        n = len(ok)
        if endpoint == "scifact":
            ev = {q: t for q, t in ok.items() if t[2] != "NOINFO"}
            ni = {q: t for q, t in ok.items() if t[2] == "NOINFO"}
            acc = sum(t[0] for t in ev.values()) / max(len(ev), 1)
            extra = {"acc_all": sum(t[0] for t in ok.values()) / max(n, 1),
                     "acc_noinfo": sum(t[0] for t in ni.values()) / max(len(ni), 1),
                     "n_evidence": len(ev), "unparsed": sum(1 for q in rows if resp[[r for r in mapping if r["qid"] == q and r["cond"] == cond][0]["job"]] is None)}
            paired = ev
        else:
            acc = sum(t[0] for t in ok.values()) / max(n, 1)
            extra = {"unknown_rate": sum(1 for t in ok.values() if "UNKNOWN" in t[1].upper()) / max(n, 1)}
            paired = ok
        b = c = 0; diffs = []
        for q, t in paired.items():
            f = by_cond["float"].get(q)
            fc = correct(f[0], f[1]) if f else None
            if fc is None:
                continue
            b += fc and not t[0]; c += t[0] and not fc
            diffs.append(int(t[0]) - int(fc))
        lo, hi = paired_ci(diffs)
        out[cond] = {"n": n, "acc": acc, **extra, "mcnemar_p_vs_float": mcnemar_exact(b, c), "discordant": [b, c],
                     "diff_pts": 100 * sum(diffs) / max(len(diffs), 1), "ci95": [lo, hi],
                     "tost_p_5pts": tost(diffs, 5.0), "tost_p_3pts": tost(diffs, 3.0), **rstats.get(cond, {})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-root", default=str(ROOT / "results" / "jobs"))
    ap.add_argument("--out-root", default=str(ROOT / "results" / "out"))
    args = ap.parse_args()
    jobs_root, out_root = Path(args.jobs_root), Path(args.out_root)
    results = {}
    for od in sorted(out_root.glob("out_*")):
        reader, d = od.name[4:].split("_", 1)
        # reader names contain no underscore-separated endpoint prefix; the jobs dir is the suffix
        cand = [j for j in jobs_root.iterdir() if od.name.endswith(j.name)]
        if not cand:
            continue
        jd = cand[0]; reader = od.name[4:-len(jd.name) - 1]
        meta = json.load((jd / "meta.json").open())
        res = score_dir(jd, od, meta["endpoint"])
        results[f"{reader}__{jd.name}"] = {"reader": reader, "jobs": jd.name, "endpoint": meta["endpoint"], "model": meta["model"], "k": meta["k"], "conds": res}
        print(f"\n=== {reader} | {jd.name} ({meta['endpoint']}, {meta['model']}, k={meta['k']}) ===")
        print(f"{'cond':>6} {'n':>4} {'acc':>6} {'diff':>6} {'CI95':>16} {'p':>7} {'disc':>7} {'ovl@k':>6} {'gold@k':>6}" + ("  acc_all noinfo" if meta["endpoint"] == "scifact" else "  unknown"))
        for cond, r in res.items():
            tail = f"  {r['acc_all']:.3f} {r['acc_noinfo']:.3f}" if meta["endpoint"] == "scifact" else f"  {r['unknown_rate']:.3f}"
            print(f"{cond:>6} {r['n']:>4} {r['acc']:.3f} {r['diff_pts']:+6.1f} [{r['ci95'][0]:+6.1f},{r['ci95'][1]:+6.1f}] {r['mcnemar_p_vs_float']:7.4f} {str(r['discordant']):>7} {r.get('topk_overlap_vs_float', float('nan')):6.3f} {r.get('gold_in_topk', float('nan')):6.3f}{tail}")
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(results, (out_root / "downstream_clip99.json").open("w"), indent=1)
    print(f"\nwrote {out_root / 'downstream_clip99.json'} ({len(results)} runs)")


if __name__ == "__main__":
    main()
