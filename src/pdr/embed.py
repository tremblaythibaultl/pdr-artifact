from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Cohere caps a single /embed request at 96 inputs.
_COHERE_BATCH = 96
# Trial keys enforce a per-minute token limit; on a 429 we wait out the window
# (the API gives no Retry-After here) and retry, rather than failing the run.
_COHERE_RATELIMIT_WAIT = 60.0
_COHERE_MAX_RETRIES = 15


@dataclass(frozen=True)
class ModelSpec:
    model_id: str  # provider-specific identifier (HF repo or Cohere model name)
    backend: str  # "sentence-transformers" | "cohere"
    dim: int
    # Asymmetric-retrieval prefixes, verbatim from each model card. Getting these
    # wrong silently costs float quality, so they are declared here rather than
    # left to the caller. The Cohere backend uses input_type instead and ignores
    # both fields.
    query_prefix: str = ""
    doc_prefix: str = ""
    # Cap on tokens per text. None keeps the checkpoint's own default. Set it for
    # long-context encoders (arctic-l-v2.0 defaults to 8192) so a padded batch of
    # abstracts fits in memory and so every encoder in a comparison sees the same
    # 512-token window, which is what the BERT-family models use anyway.
    max_seq_length: int | None = None


# Canonical models used in the experiments. Unregistered ids fall back to the
# sentence-transformers backend (see `resolve`), so any HF repo keeps working.
# The four 1024-dim encoders of the quantization-aware suite carry the prefixes
# their cards prescribe:
#   bge-large-en-v1.5 / mxbai-embed-large-v1: query instruction, no passage prefix
#     (mxbai's is also declared in its config_sentence_transformers.json prompts).
#   e5-large-v2: "query: " / "passage: " on both sides.
#   gte-large: no prefix on either side.
#   snowflake-arctic-embed-l-v2.0: "query: " on queries, no passage prefix.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

REGISTRY: dict[str, ModelSpec] = {
    "BAAI/bge-large-en-v1.5": ModelSpec(
        "BAAI/bge-large-en-v1.5", "sentence-transformers", 1024, query_prefix=_BGE_QUERY_INSTRUCTION
    ),
    "mixedbread-ai/mxbai-embed-large-v1": ModelSpec(
        "mixedbread-ai/mxbai-embed-large-v1",
        "sentence-transformers",
        1024,
        query_prefix=_BGE_QUERY_INSTRUCTION,
    ),
    "intfloat/e5-large-v2": ModelSpec(
        "intfloat/e5-large-v2",
        "sentence-transformers",
        1024,
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    "thenlper/gte-large": ModelSpec("thenlper/gte-large", "sentence-transformers", 1024),
    # The one open 1024-dim encoder whose card claims *quantization-aware embedding
    # training* (with MRL); XLM-R backbone, CLS pooling, "query: " prefix per its
    # config_sentence_transformers.json prompts, no passage prefix.
    "Snowflake/snowflake-arctic-embed-l-v2.0": ModelSpec(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "sentence-transformers",
        1024,
        query_prefix="query: ",
        max_seq_length=512,
    ),
    "bert-base-uncased": ModelSpec("bert-base-uncased", "sentence-transformers", 768),
    "embed-english-v3.0": ModelSpec("embed-english-v3.0", "cohere", 1024),
}


def resolve(model_name: str) -> ModelSpec:
    if model_name in REGISTRY:
        return REGISTRY[model_name]
    return ModelSpec(model_name, "sentence-transformers", -1)


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return emb / norms


@lru_cache(maxsize=4)
def _load_st(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def _encode_st(
    model_id: str,
    texts: list[str],
    normalize: bool,
    prefix: str = "",
    max_seq_length: int | None = None,
) -> np.ndarray:
    model = _load_st(model_id)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    if prefix:
        texts = [prefix + t for t in texts]
    emb = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


@lru_cache(maxsize=1)
def _cohere_client():
    try:
        import cohere
    except ImportError as e:  # pragma: no cover - exercised via import error path
        raise RuntimeError(
            "Cohere models require the 'cohere' package: pip install -e '.[cohere]'"
        ) from e
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Cohere models require the COHERE_API_KEY environment variable.")
    return cohere.ClientV2(api_key=api_key)


def _cohere_embed(client, batch: list[str], model_id: str, input_type: str):
    import cohere

    for attempt in range(_COHERE_MAX_RETRIES + 1):
        try:
            return client.embed(
                texts=batch,
                model=model_id,
                input_type=input_type,
                embedding_types=["float"],
                truncate="END",
            )
        except cohere.TooManyRequestsError:
            if attempt == _COHERE_MAX_RETRIES:
                raise
            print(
                f"[cohere] rate limited; waiting {_COHERE_RATELIMIT_WAIT:.0f}s "
                f"(retry {attempt + 1}/{_COHERE_MAX_RETRIES})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(_COHERE_RATELIMIT_WAIT)


def _encode_cohere(model_id: str, texts: list[str], normalize: bool, is_query: bool) -> np.ndarray:
    client = _cohere_client()
    input_type = "search_query" if is_query else "search_document"
    out: list[list[float]] = []
    for start in range(0, len(texts), _COHERE_BATCH):
        batch = texts[start : start + _COHERE_BATCH]
        resp = _cohere_embed(client, batch, model_id, input_type)
        out.extend(resp.embeddings.float)
    emb = np.asarray(out, dtype=np.float32)
    return _l2_normalize(emb) if normalize else emb


def encode(
    model_name: str,
    texts: list[str],
    normalize: bool = True,
    is_query: bool = False,
) -> np.ndarray:
    spec = resolve(model_name)
    if spec.backend == "cohere":
        return _encode_cohere(spec.model_id, texts, normalize, is_query)
    prefix = spec.query_prefix if is_query else spec.doc_prefix
    return _encode_st(spec.model_id, texts, normalize, prefix, spec.max_seq_length)
