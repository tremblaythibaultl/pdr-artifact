import numpy as np
import pytest

from pdr import embed


def test_resolve_registered_models():
    assert embed.resolve("BAAI/bge-large-en-v1.5").backend == "sentence-transformers"
    assert embed.resolve("bert-base-uncased").backend == "sentence-transformers"
    cohere = embed.resolve("embed-english-v3.0")
    assert cohere.backend == "cohere"
    assert cohere.model_id == "embed-english-v3.0"


def test_resolve_unknown_falls_back_to_st():
    spec = embed.resolve("some/other-hf-model")
    assert spec.backend == "sentence-transformers"
    assert spec.model_id == "some/other-hf-model"
    assert spec.dim == -1


class _FakeEmbeddings:
    def __init__(self, vecs):
        self.float = vecs


class _FakeResp:
    def __init__(self, vecs):
        self.embeddings = _FakeEmbeddings(vecs)


class _FakeClient:
    def __init__(self):
        self.calls = []

    def embed(self, *, texts, model, input_type, embedding_types, truncate):
        self.calls.append({"n": len(texts), "input_type": input_type, "model": model})
        # Non-unit vectors so normalization is observable.
        return _FakeResp([[float(len(t)) + 1.0, 2.0, 3.0] for t in texts])


@pytest.fixture
def fake_cohere(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(embed, "_cohere_client", lambda: client)
    return client


def test_cohere_batches_and_sets_input_type(fake_cohere):
    texts = [f"text number {i}" for i in range(200)]
    emb = embed.encode("embed-english-v3.0", texts, normalize=False, is_query=True)

    assert emb.shape == (200, 3)
    assert [c["n"] for c in fake_cohere.calls] == [96, 96, 8]  # 96-input cap
    assert {c["input_type"] for c in fake_cohere.calls} == {"search_query"}
    assert {c["model"] for c in fake_cohere.calls} == {"embed-english-v3.0"}


def test_cohere_document_input_type(fake_cohere):
    embed.encode("embed-english-v3.0", ["a doc"], normalize=False, is_query=False)
    assert fake_cohere.calls[0]["input_type"] == "search_document"


def test_cohere_normalizes_when_requested(fake_cohere):
    emb = embed.encode("embed-english-v3.0", ["a", "bb", "ccc"], normalize=True)
    norms = np.linalg.norm(emb, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
