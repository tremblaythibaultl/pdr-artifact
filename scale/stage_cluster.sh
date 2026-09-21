#!/bin/bash
# Login-node staging (internet access): a venv from the cluster wheelhouse, BEIR-NQ and the five
# encoders into the HF cache, then an offline load test. Idempotent. Usage:
#   PDR_PROJ=/path/to/project bash stage_cluster.sh
# Layout created under $PDR_PROJ: data/ emb/ results/ logs/ hf_cache/ (HF_HOME overrides the last).
set -euo pipefail
PROJ=${PDR_PROJ:?set PDR_PROJ to the project directory on the cluster}
VENV=${PDR_VENV:-$PROJ/venv-stage}
export HF_HOME=${HF_HOME:-$PROJ/hf_cache}
export MODELS=${MODELS:-"BAAI/bge-large-en-v1.5 Snowflake/snowflake-arctic-embed-l-v2.0 thenlper/gte-large mixedbread-ai/mxbai-embed-large-v1 intfloat/e5-large-v2"}
mkdir -p "$PROJ"/{data,emb,results,logs} "$HF_HOME"
module load StdEnv/2023 python/3.11
[ -x "$VENV/bin/python" ] || virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch sentence_transformers numpy huggingface_hub requests tqdm scikit_learn scipy Pillow typing_extensions -q
python -c "import torch, sentence_transformers as st, transformers, huggingface_hub as hh; print('torch', torch.__version__, 'st', st.__version__, 'transformers', transformers.__version__, 'hub', hh.__version__)"
cd "$PROJ/data"
if [ ! -f nq/corpus.jsonl ]; then
  echo "=== $(date) downloading BEIR nq.zip ==="
  wget -q https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip -O nq.zip
  unzip -q -o nq.zip && rm -f nq.zip
fi
echo "=== $(date) nq: $(wc -l < nq/corpus.jsonl) docs, $(wc -l < nq/queries.jsonl) queries, $(wc -l < nq/qrels/test.tsv) qrel lines ==="
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for m in os.environ["MODELS"].split():
    p = snapshot_download(m, ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.bin", "*.h5", "*.ot", "*.msgpack", "*.ckpt"])
    print("staged", m, "->", p, flush=True)
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python - <<'PY'
import os
from sentence_transformers import SentenceTransformer
for m in os.environ["MODELS"].split():
    e = SentenceTransformer(m, device="cpu").encode(["offline load test"], normalize_embeddings=True)
    print("offline load OK", m, e.shape, flush=True)
PY
du -sh "$HF_HOME" "$PROJ/data/nq"
echo "=== $(date) STAGING COMPLETE ==="
