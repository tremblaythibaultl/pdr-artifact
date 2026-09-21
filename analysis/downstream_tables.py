"""LaTeX for the clipped downstream results (from downstream_analysis.py's JSON).

Prints (a) the body table for one reader (--reader, default phi-4): rows are
bit-widths, columns are endpoint x encoder, bold marks a significant drop from
float (exact McNemar p < 0.05); (b) one appendix block per reader with the
paired difference and 95% CI at b=4, b=3 and b=2 for every endpoint x encoder.
Run: python analysis/downstream_tables.py [--reader phi-4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["closed", "gold", "float", "8", "6", "4", "3", "2"]
SETS = [("scifact_k5", "SciFact $k{=}5$"), ("scifact_k1", "SciFact $k{=}1$"), ("bioasq_k5", "BioASQ"), ("nq_k5", "NQ ($2.68$M)")]
# Chunked SciFact (scale/chunk_corpus.py, 1,536-byte chunks): the retrieval unit study of Section 5.4.
CHUNK_SETS = [("scifact_chunks_k5", "SciFact chunks $k{=}5$"), ("scifact_chunks_k1", "SciFact chunks $k{=}1$")]
ENC = [("bge", "BGE"), ("arctic", "arctic")]
READER_NAME = {"phi-4": "phi-4 (14B)", "olmo-2-1124-7b": "OLMo-2 (7B)", "qwen2.5-72b": "Qwen2.5 (72B)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "results" / "out" / "downstream_clip99.json"))
    ap.add_argument("--reader", default="phi-4")
    args = ap.parse_args()
    R = json.load(open(args.json))
    readers = sorted({v["reader"] for v in R.values()})
    print(f"readers present: {readers}\n")

    def cell(reader, s, e, cond, fmt="{:.3f}"):
        key = f"{reader}__{s}_{e}_clip99"
        if key not in R or cond not in R[key]["conds"]:
            return "---"
        r = R[key]["conds"][cond]
        v = fmt.format(r["acc"])
        return f"$\\mathbf{{{v}}}$" if r["mcnemar_p_vs_float"] < 0.05 and r["diff_pts"] < 0 else f"${v}$"

    print(f"=== body table, reader {args.reader} ===")
    print("$b$ & " + " & ".join(f"{sn} {en}" for s, sn in SETS for e, en in ENC) + r" \\")
    for cond in ORDER:
        lab = {"closed": "closed book", "gold": "gold context", "float": "float"}.get(cond, f"${cond}$")
        row = [lab] + [cell(args.reader, s, e, cond) for s, _ in SETS for e, _ in ENC]
        if cond == "float":
            print(r"\midrule")
        print(" & ".join(row) + r" \\")

    print("\n=== appendix: paired difference (points) and 95% CI vs float, per reader ===")
    for reader in readers:
        print(f"--- {READER_NAME.get(reader, reader)} ---")
        for s, sn in SETS:
            for e, en in ENC:
                key = f"{reader}__{s}_{e}_clip99"
                if key not in R:
                    continue
                c = R[key]["conds"]
                cells = []
                for b in ("4", "3", "2"):
                    r = c[b]; lo, hi = r["ci95"]
                    star = r"$^{*}$" if r["mcnemar_p_vs_float"] < 0.05 else ""
                    eq = r"$^{\dagger}$" if r.get("tost_p_5pts", 1.0) < 0.05 else ""
                    cells.append(f"${r['diff_pts']:+.1f}$ $[{lo:+.1f}, {hi:+.1f}]${star}{eq}")
                print(f"{sn} & {en} & ${c['float']['acc']:.3f}$ & " + " & ".join(cells) + r" \\")

    print("\n=== chunked SciFact vs abstracts (reader %s): float accuracy and paired difference vs float at b=4,3,2 ===" % args.reader)
    for (s_abs, _), (s_ch, sn) in zip(SETS[:2], CHUNK_SETS):
        for e, en in ENC:
            ka, kc = f"{args.reader}__{s_abs}_{e}_clip99", f"{args.reader}__{s_ch}_{e}_clip99"
            if ka not in R or kc not in R:
                continue
            ca, cc = R[ka]["conds"], R[kc]["conds"]
            def cells(c):
                out = []
                for b in ("4", "3", "2"):
                    r = c[b]; lo, hi = r["ci95"]
                    star = r"$^{*}$" if r["mcnemar_p_vs_float"] < 0.05 and r["diff_pts"] < 0 else ""
                    eq = r"$^{\dagger}$" if r.get("tost_p_5pts", 1.0) < 0.05 else ""
                    out.append(f"${r['diff_pts']:+.1f}$ $[{lo:+.1f}, {hi:+.1f}]${star}{eq}")
                return out
            print(f"{sn.replace(' chunks','')} & {en} & abstracts & ${ca['float']['acc']:.3f}$ & " + " & ".join(cells(ca)) + r" \\")
            print(f" & & chunks & ${cc['float']['acc']:.3f}$ & " + " & ".join(cells(cc)) + r" \\")
            print(f"%   gold@k float abstracts {ca['float'].get('gold_in_topk', float('nan')):.3f} chunks {cc['float'].get('gold_in_topk', float('nan')):.3f}; at b=3 {ca['3'].get('gold_in_topk', float('nan')):.3f} / {cc['3'].get('gold_in_topk', float('nan')):.3f}; at b=2 {ca['2'].get('gold_in_topk', float('nan')):.3f} / {cc['2'].get('gold_in_topk', float('nan')):.3f}")

    print("\n=== equivalence within +-5 points (TOST p<0.05), all readers ===")
    for b in ("4", "3", "2"):
        tot = eq5 = eq3 = 0
        for reader in readers:
            for s, _ in SETS:
                for e, _ in ENC:
                    key = f"{reader}__{s}_{e}_clip99"
                    if key not in R:
                        continue
                    r = R[key]["conds"][b]; tot += 1
                    eq5 += r.get("tost_p_5pts", 1.0) < 0.05; eq3 += r.get("tost_p_3pts", 1.0) < 0.05
        print(f"b={b}: {eq5}/{tot} equivalent within 5 pts, {eq3}/{tot} within 3 pts")
    for reader in readers:
        for b in ("4", "3"):
            tot = eq5 = 0
            for s, _ in SETS:
                for e, _ in ENC:
                    key = f"{reader}__{s}_{e}_clip99"
                    if key in R:
                        tot += 1; eq5 += R[key]["conds"][b].get("tost_p_5pts", 1.0) < 0.05
            print(f"  {reader:<16} b={b}: {eq5}/{tot} within 5 pts")

    print("\n=== summary: max significant-drop bit-width per (reader, set, encoder) ===")
    for reader in readers:
        for s, sn in SETS:
            for e, en in ENC:
                key = f"{reader}__{s}_{e}_clip99"
                if key not in R:
                    continue
                c = R[key]["conds"]
                sig = [b for b in ("8", "6", "4", "3", "2") if c[b]["mcnemar_p_vs_float"] < 0.05 and c[b]["diff_pts"] < 0]
                worst = min((c[b]["ci95"][0] for b in ("4", "3")), default=float("nan"))
                print(f"{reader:<16} {s:<11} {e:<7} float={c['float']['acc']:.3f}  sig. drops at b in {sig or 'none'};  CI lower bound over b in {{4,3}}: {worst:+.1f} pts")


if __name__ == "__main__":
    main()
