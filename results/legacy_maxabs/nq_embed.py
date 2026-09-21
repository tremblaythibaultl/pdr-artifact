"""Embed the NQ mini-corpus with BGE-large, checkpointed every chunk."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

device = sys.argv[1] if len(sys.argv) > 1 else None
CHUNK = 1000

docs = [json.loads(line) for line in open(HERE / "nq_docs.jsonl")]
texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in docs]
print(f"{len(texts)} docs", flush=True)

from sentence_transformers import SentenceTransformer  # noqa: E402
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
print(f"model on {model.device}", flush=True)

parts_dir = HERE / "nq_emb_parts"
parts_dir.mkdir(exist_ok=True)
for start in range(0, len(texts), CHUNK):
    p = parts_dir / f"part_{start:06d}.npy"
    if p.exists():
        continue
    emb = model.encode(texts[start:start + CHUNK], batch_size=32,
                       normalize_embeddings=True, show_progress_bar=False)
    np.save(p, np.asarray(emb, dtype=np.float32))
    print(f"chunk {start} done", flush=True)

parts = sorted(parts_dir.glob("part_*.npy"))
A = np.concatenate([np.load(p) for p in parts])
assert A.shape[0] == len(texts)
np.save(HERE / "nq_corpus_emb.npy", A)
print("corpus embeddings saved:", A.shape)
