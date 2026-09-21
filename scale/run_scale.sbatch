#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --job-name=pdr-scale
#SBATCH --output=%x-%A_%a.out
# Retrieval quality at scale on BEIR-NQ (2.68M passages): one array task per encoder, embed the
# corpus then run the clipped scale sweep (scale_sweep.py). Submit from scale/ once
# stage_cluster.sh has populated $PDR_PROJ:
#   PDR_PROJ=/path/to/project sbatch --account=<account> run_scale.sbatch
#   K=5 BITS="2 3 4 5 6 8" PDR_PROJ=... sbatch --account=<account> run_scale.sbatch   # Recall@5 sweep
#   MODELS="mixedbread-ai/mxbai-embed-large-v1 intfloat/e5-large-v2" PDR_PROJ=... sbatch --array=0-1 ...
# Outputs: $PDR_PROJ/emb/nq/<model>/ and $PDR_PROJ/results/scale__nq__<model>__clip99[__k5].csv.
# The compute nodes we used had no internet access: models come from the HF cache staged by
# stage_cluster.sh and wheels from the cluster's local wheelhouse (pip --no-index). Adapt the
# module and pip lines to your site. One task took about 35 minutes on one H100.
set -euo pipefail
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
CODE=${PDR_CODE:-$SLURM_SUBMIT_DIR}
K=${K:-10}; BITS=${BITS:-"3 4 5 6 8"}
read -ra MODELS <<< "${MODELS:-BAAI/bge-large-en-v1.5 Snowflake/snowflake-arctic-embed-l-v2.0 thenlper/gte-large}"
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SAFE=${MODEL//\//_}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load StdEnv/2023 python/3.11 cuda/12.6
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch sentence_transformers numpy huggingface_hub requests tqdm scikit_learn scipy Pillow typing_extensions -q
python -c "import torch, sentence_transformers as st; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'st', st.__version__)"

mkdir -p "$SLURM_TMPDIR/data" "$PROJ/results"
cp "$PROJ/data/nq/corpus.jsonl" "$PROJ/data/nq/queries.jsonl" "$SLURM_TMPDIR/data/"
cp -r "$PROJ/data/nq/qrels" "$SLURM_TMPDIR/data/"

echo "=== $(date) embed $MODEL ==="
python "$CODE/embed_beir.py" --data "$SLURM_TMPDIR/data" --model "$MODEL" --out "$PROJ/emb/nq" --batch 256 --max-seq 512
SUFFIX=""; [ "$K" != 10 ] && SUFFIX="__k$K"
echo "=== $(date) scale sweep $MODEL (k=$K, bits $BITS) ==="
python "$CODE/scale_sweep.py" --emb "$PROJ/emb/nq/$SAFE" --data "$SLURM_TMPDIR/data" \
    --sizes 10000 100000 1000000 all --bits $BITS --clip 99 --k "$K" \
    --out "$PROJ/results/scale__nq__${SAFE}__clip99${SUFFIX}.csv"
echo "=== $(date) done ==="
