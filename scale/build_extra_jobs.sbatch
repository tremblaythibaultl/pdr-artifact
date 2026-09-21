#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --job-name=pdr-extrajobs
#SBATCH --output=%x-%j.out
# Reference rows of the downstream table: closed book and gold context, for every abstract-level
# endpoint x encoder (same directory names as jobs/, under jobs_extra/), then jobs_extra.tgz for
# run_readers.sbatch (TARBALL=jobs_extra.tgz JOBSDIR=jobs_extra). Neither condition depends on the
# encoder; both directories get them so that the scorer merges them into every existing run.
# Needs the embeddings of build_jobs.sbatch. Submit from scale/:
#   PDR_PROJ=/path/to/project sbatch --account=<account> build_extra_jobs.sbatch
set -euo pipefail
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
CODE=${PDR_CODE:-$SLURM_SUBMIT_DIR}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
module load StdEnv/2023 python/3.11 cuda/12.6
virtualenv --no-download "$SLURM_TMPDIR/venv" && source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch numpy -q
D="$SLURM_TMPDIR/data"; mkdir -p "$D/nq"
cp -r "$PROJ/data/scifact" "$PROJ/data/bioasq" "$D/"
cp "$PROJ/data/nq/corpus.jsonl" "$PROJ/data/nq/queries.jsonl" "$PROJ/data/nq/nq_subset.json" "$D/nq/"; cp -r "$PROJ/data/nq/qrels" "$D/nq/"
declare -A SHORT=( ["BAAI/bge-large-en-v1.5"]="bge" ["Snowflake/snowflake-arctic-embed-l-v2.0"]="arctic" )
for MODEL in "BAAI/bge-large-en-v1.5" "Snowflake/snowflake-arctic-embed-l-v2.0"; do
  SAFE=${MODEL//\//_}; ENC=${SHORT[$MODEL]}
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/scifact/$SAFE" --data "$D/scifact" --endpoint scifact --k 5 --clip 99 --conds closed gold --out "$PROJ/jobs_extra/scifact_k5_${ENC}_clip99"
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/scifact/$SAFE" --data "$D/scifact" --endpoint scifact --k 1 --clip 99 --conds closed gold --out "$PROJ/jobs_extra/scifact_k1_${ENC}_clip99"
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/bioasq/$SAFE" --data "$D/bioasq" --endpoint qa --k 5 --clip 99 --conds closed gold --out "$PROJ/jobs_extra/bioasq_k5_${ENC}_clip99"
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/nq/$SAFE" --data "$D/nq" --endpoint qa --subset "$D/nq/nq_subset.json" --k 5 --clip 99 --conds closed gold --out "$PROJ/jobs_extra/nq_k5_${ENC}_clip99"
done
cd "$PROJ" && tar czf jobs_extra.tgz jobs_extra && echo "=== $(date) done: $(ls jobs_extra | tr '\n' ' ') ===" && du -sh jobs_extra.tgz
