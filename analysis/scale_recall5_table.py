"""Recall@5 retained under the clipped quantizer versus corpus size, from the k=5 scale sweeps
(scale/run_scale.sbatch with K=5): results/scale/scale__nq__<model>__clip99__k5.csv.
Prints the LaTeX body of tab:scale-recall5 (appendix) and a console summary."""
import csv, glob
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "scale"
NAMES = {"BAAI/bge-large-en-v1.5": r"\texttt{bge-large-en-v1.5}",
         "Snowflake/snowflake-arctic-embed-l-v2.0": r"\texttt{arctic-embed-l-v2.0}",
         "mixedbread-ai/mxbai-embed-large-v1": r"\texttt{mxbai-embed-large-v1}",
         "intfloat/e5-large-v2": r"\texttt{e5-large-v2}",
         "thenlper/gte-large": r"\texttt{gte-large}"}
# the quantization-aware encoder first, then the conventional four
ORDER = ["Snowflake/snowflake-arctic-embed-l-v2.0", "BAAI/bge-large-en-v1.5",
         "mixedbread-ai/mxbai-embed-large-v1", "intfloat/e5-large-v2", "thenlper/gte-large"]
BITS = ["2", "3", "4"]

def msize(m):
    return {"10000": "$10^4$", "100000": "$10^5$", "1000000": "$10^6$", "2681468": r"$2.68{\cdot}10^6$"}[m]

data = {}
for p in glob.glob(str(RES / "scale__nq__*__clip99__k5.csv")):
    for r in csv.DictReader(open(p)):
        data.setdefault(r["model"], {}).setdefault(r["m"], {})[r["bits"]] = r

print(r"\begin{tabular}{@{}ll c ccc@{}}")
print(r"\toprule")
print(r"& & & \multicolumn{3}{c}{Recall@5 retained (\%)} \\")
print(r"\cmidrule(lr){4-6}")
print(r"Encoder & $m$ & Recall@5 (float) & $b{=}2$ & $b{=}3$ & $b{=}4$ \\")
print(r"\midrule")
ORDER = [m for m in ORDER if m in data]
for i, model in enumerate(ORDER):
    rows = data[model]
    for j, m in enumerate(sorted(rows, key=int)):
        f = float(rows[m]["float"]["recall_f"])
        cells = [f"${100*float(rows[m][b]['recall_q'])/f:.1f}$" for b in BITS]
        lead = NAMES[model] if j == 0 else ""
        print(f"{lead} & {msize(m)} & ${f:.3f}$ & " + " & ".join(cells) + r" \\")
    if i < len(ORDER) - 1:
        print(r"\midrule")
print(r"\bottomrule")
print(r"\end{tabular}")

print("\n% summary: min retained over sizes, per encoder and b")
for model in ORDER:
    rows = data[model]
    for b in BITS:
        vals = [100*float(rows[m][b]["recall_q"])/float(rows[m]["float"]["recall_f"]) for m in rows]
        print(f"% {model} b={b}: min {min(vals):.1f} max {max(vals):.1f}")

# --- does the gap to the quantization-aware encoder widen with m? ---
QAT = "Snowflake/snowflake-arctic-embed-l-v2.0"
def retained(model, m, b):
    r = data[model][m]
    return 100 * float(r[b]["recall_q"]) / float(r["float"]["recall_f"])
sizes = sorted(data[QAT], key=int)
print("\n% gap: arctic minus encoder, retained Recall@5 (points), per size")
for b in ("2", "3"):
    print(f"% b={b:<2} {'encoder':42s} " + " ".join(f"{msize(m).strip('$'):>14s}" for m in sizes))
    for model in ORDER:
        if model == QAT:
            continue
        gaps = [retained(QAT, m, b) - retained(model, m, b) for m in sizes]
        print(f"%      {model:42s} " + " ".join(f"{g:14.1f}" for g in gaps))
print("\n% min retained over the five encoders, per size and b")
for b in BITS:
    for m in sizes:
        vals = {model: retained(model, m, b) for model in ORDER}
        lo = min(vals, key=vals.get)
        print(f"% b={b} m={m:>8}: min {vals[lo]:.1f} ({lo})")
