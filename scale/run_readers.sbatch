#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=pdr-readers
#SBATCH --output=%x-%j.out
# One open-weight reader over every prompt set, with vLLM (run_readers.py). Submit from scale/:
#   PDR_PROJ=... sbatch --account=<account> --gres=gpu:h100:1 --export=ALL,MODEL=microsoft/phi-4,TP=1 run_readers.sbatch
#   PDR_PROJ=... sbatch --account=<account> --gres=gpu:h100:4 --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,TP=4,EXTRA=--enforce-eager run_readers.sbatch
# Optional: EXTRA="--limit 50" for a smoke test; DIRS="scifact_k5_bge_clip99" to restrict;
# TARBALL=jobs_chunks.tgz JOBSDIR=jobs_chunks for a prompt set other than jobs_clip99.tgz;
# MAXLEN=4096 for OLMo-2. Outputs: $PDR_PROJ/out/out_<reader>_<set>/<hash>.txt.
# The readers must be in the HF cache beforehand (stage_readers.sh).
set -euo pipefail
: "${MODEL:?set MODEL}"; TP=${TP:-1}; EXTRA=${EXTRA:-}; DIRS=${DIRS:-}; MAXLEN=${MAXLEN:-8192}; TARBALL=${TARBALL:-jobs_clip99.tgz}; JOBSDIR=${JOBSDIR:-jobs}
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
CODE=${PDR_CODE:-$SLURM_SUBMIT_DIR}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=true
# vLLM install. We used vLLM 0.25.0 from the cluster wheelhouse, which pins torch 2.11.0 and
# torchvision 0.26.0 (CUDA 12.9) and whose OpenCV dependency does not resolve offline; the working
# recipe was a venv assembled once with --no-deps (torch, torchvision, vllm, then the packages
# `pip check` asks for, without opencv and torchaudio) and frozen to vllm_requirements.txt. With
# internet access, `pip install vllm` in a fresh venv is enough.
module load StdEnv/2023 python/3.11 cuda/12.9
virtualenv --no-download "$SLURM_TMPDIR/venv" && source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip -q
if [ -f "$CODE/vllm_requirements.txt" ]; then
  pip install --no-index --no-deps -r "$CODE/vllm_requirements.txt" -q
else
  pip install --no-index vllm -q
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count())"
# stage prompts to node-local disk (many small files)
tar xzf "$PROJ/$TARBALL" -C "$SLURM_TMPDIR" && ls "$SLURM_TMPDIR/$JOBSDIR"
echo "=== $(date) $MODEL ==="
python "$CODE/run_readers.py" --model "$MODEL" --jobs-root "$SLURM_TMPDIR/$JOBSDIR" --out-root "$PROJ/out" --tp "$TP" --max-model-len "$MAXLEN" $EXTRA ${DIRS:+--dirs $DIRS}
echo "=== $(date) done ==="
