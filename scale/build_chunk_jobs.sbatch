#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=pdr-chunkjobs
#SBATCH --output=%x-%j.out
# Chunked SciFact (chunk_corpus.py output staged to $PDR_PROJ/data/scifact_chunks): embed the chunk
# corpus with both downstream encoders, build the SciFact k=5 and k=1 prompt sets (clipped p99), tar
# them as jobs_chunks.tgz for run_readers.sbatch (TARBALL=jobs_chunks.tgz JOBSDIR=jobs_chunks).
# Submit from scale/:  PDR_PROJ=/path/to/project sbatch --account=<account> build_chunk_jobs.sbatch
set -euo pipefail
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
CODE=${PDR_CODE:-$SLURM_SUBMIT_DIR}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
module load StdEnv/2023 python/3.11 cuda/12.6
virtualenv --no-download "$SLURM_TMPDIR/venv" && source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch sentence_transformers numpy huggingface_hub requests tqdm scikit_learn scipy Pillow typing_extensions -q
D="$SLURM_TMPDIR/data"; mkdir -p "$D"; cp -r "$PROJ/data/scifact_chunks" "$D/"
declare -A SHORT=( ["BAAI/bge-large-en-v1.5"]="bge" ["Snowflake/snowflake-arctic-embed-l-v2.0"]="arctic" )
for MODEL in "BAAI/bge-large-en-v1.5" "Snowflake/snowflake-arctic-embed-l-v2.0"; do
  SAFE=${MODEL//\//_}; ENC=${SHORT[$MODEL]}
  echo "=== $(date) embed scifact_chunks $MODEL ==="
  python "$CODE/embed_beir.py" --data "$D/scifact_chunks" --model "$MODEL" --out "$PROJ/emb/scifact_chunks" --batch 256 --max-seq 512
  for K in 5 1; do
    python "$CODE/build_jobs.py" --emb "$PROJ/emb/scifact_chunks/$SAFE" --data "$D/scifact_chunks" --endpoint scifact --k $K --clip 99 --chunked --out "$PROJ/jobs_chunks/scifact_chunks_k${K}_${ENC}_clip99"
  done
done
cd "$PROJ" && tar czf jobs_chunks.tgz jobs_chunks && echo "=== $(date) done: $(ls jobs_chunks | tr '\n' ' ') ===" && du -sh jobs_chunks.tgz
