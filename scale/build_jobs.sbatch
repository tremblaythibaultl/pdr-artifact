#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=pdr-jobs
#SBATCH --output=%x-%j.out
# Embed SciFact and BioASQ with both downstream encoders, then build every reader prompt set
# (clipped p99) for SciFact k=5 and k=1, NQ over the full 2.68M corpus (k=5) and BioASQ (k=5).
# Needs $PDR_PROJ/data/{scifact,bioasq,nq} (BEIR format; nq/nq_subset.json from select_nq.py) and
# $PDR_PROJ/emb/nq/ from run_scale.sbatch. Submit from scale/:
#   PDR_PROJ=/path/to/project sbatch --account=<account> build_jobs.sbatch
# Output: $PDR_PROJ/jobs/<endpoint>_k<k>_<enc>_clip99/ and jobs_clip99.tgz for run_readers.sbatch.
set -euo pipefail
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
CODE=${PDR_CODE:-$SLURM_SUBMIT_DIR}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load StdEnv/2023 python/3.11 cuda/12.6
virtualenv --no-download "$SLURM_TMPDIR/venv" && source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch sentence_transformers numpy huggingface_hub requests tqdm scikit_learn scipy Pillow typing_extensions -q

D="$SLURM_TMPDIR/data"; mkdir -p "$D/nq"
cp -r "$PROJ/data/scifact" "$PROJ/data/bioasq" "$D/"
cp "$PROJ/data/nq/corpus.jsonl" "$PROJ/data/nq/queries.jsonl" "$PROJ/data/nq/nq_subset.json" "$D/nq/"; cp -r "$PROJ/data/nq/qrels" "$D/nq/"

declare -A SHORT=( ["BAAI/bge-large-en-v1.5"]="bge" ["Snowflake/snowflake-arctic-embed-l-v2.0"]="arctic" )
for MODEL in "BAAI/bge-large-en-v1.5" "Snowflake/snowflake-arctic-embed-l-v2.0"; do
  SAFE=${MODEL//\//_}; ENC=${SHORT[$MODEL]}
  for DS in scifact bioasq; do
    echo "=== $(date) embed $DS $MODEL ==="
    python "$CODE/embed_beir.py" --data "$D/$DS" --model "$MODEL" --out "$PROJ/emb/$DS" --batch 256 --max-seq 512
  done
  echo "=== $(date) build jobs $MODEL ==="
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/scifact/$SAFE" --data "$D/scifact" --endpoint scifact --k 5 --clip 99 --out "$PROJ/jobs/scifact_k5_${ENC}_clip99"
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/scifact/$SAFE" --data "$D/scifact" --endpoint scifact --k 1 --clip 99 --out "$PROJ/jobs/scifact_k1_${ENC}_clip99"
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/bioasq/$SAFE" --data "$D/bioasq" --endpoint qa --k 5 --clip 99 --out "$PROJ/jobs/bioasq_k5_${ENC}_clip99"
  python "$CODE/build_jobs.py" --emb "$PROJ/emb/nq/$SAFE" --data "$D/nq" --endpoint qa --subset "$D/nq/nq_subset.json" --k 5 --clip 99 --out "$PROJ/jobs/nq_k5_${ENC}_clip99"
done
cd "$PROJ" && tar czf jobs_clip99.tgz jobs && echo "=== $(date) done: $(ls jobs | tr '\n' ' ') ===" && du -sh jobs_clip99.tgz
