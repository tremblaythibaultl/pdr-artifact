#!/bin/bash
# Login-node staging for the reader pass: vLLM from the wheelhouse into the staging venv (install
# test) and the open-weight readers into the HF cache. Idempotent. Usage:
#   PDR_PROJ=/path/to/project bash stage_readers.sh
set -euo pipefail
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
VENV=${PDR_VENV:-$PROJ/venv-stage}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache}
export READERS=${READERS:-"microsoft/phi-4 allenai/OLMo-2-1124-7B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-72B-Instruct"}
module load StdEnv/2023 python/3.11
source "$VENV/bin/activate"
echo "=== $(date) vllm install test ==="
pip install --no-index vllm -q 2>&1 | tail -3 || echo "VLLM INSTALL FAILED (see run_readers.sbatch for the recipe we used)"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" || echo "VLLM IMPORT FAILED"
echo "=== $(date) downloading readers ==="
python - <<'PY'
import os
from huggingface_hub import snapshot_download, model_info
for m in os.environ["READERS"].split():
    info = model_info(m)
    print("gated:", m, info.gated, flush=True)
    p = snapshot_download(m, ignore_patterns=["*.bin", "*.pth", "*.gguf", "*.onnx", "original/*", "consolidated*"])
    print("staged", m, "->", p, flush=True)
PY
du -sh "$HF_HOME"/hub/models--* 2>/dev/null
echo "=== $(date) READERS STAGED ==="
