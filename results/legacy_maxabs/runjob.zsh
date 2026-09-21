#!/bin/zsh
# Answer one prompt file with a hosted model through the Claude command line, skipping existing
# outputs. Usage: runjob.zsh <model> <outdir> <jobfile>   (e.g. haiku out_haiku_k5 jobs_k5/<hash>.txt)
set -u
cd "$(dirname "$0")"
model="$1"; outdir="$2"; f="$3"
h="${f##*/}"; h="${h%.txt}"
out="${outdir}/${h}.txt"
mkdir -p "$outdir"
[[ -s "$out" ]] && exit 0
for attempt in 1 2 3 4; do
  if claude -p --model "$model" < "$f" > "${out}.tmp" 2>/dev/null && [[ -s "${out}.tmp" ]]; then
    mv "${out}.tmp" "$out"; exit 0
  fi
  rm -f "${out}.tmp"
  sleep $((attempt * 5))
done
exit 0
