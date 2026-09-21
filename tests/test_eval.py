import numpy as np

from pdr.eval import evaluate, ndcg_at_k, recall_at_k


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def test_ndcg_perfect_ranking():
    qrel = {"d1": 2, "d2": 1, "d3": 0}
    rank = ["d1", "d2", "d3"]
    assert ndcg_at_k(qrel, rank, k=3) == 1.0


def test_ndcg_handles_missing_docs():
    qrel = {"d1": 1}
    assert ndcg_at_k(qrel, ["dX", "d1"], k=2) > 0.0


def test_ndcg_no_relevant():
    qrel = {"d1": 0, "d2": 0}
    assert ndcg_at_k(qrel, ["d1", "d2"], k=2) == 0.0


def test_recall_at_k():
    qrel = {"d1": 1, "d2": 1, "d3": 0}
    assert recall_at_k(qrel, ["d1", "d2", "d3"], k=2) == 1.0
    assert recall_at_k(qrel, ["d3", "dX"], k=2) == 0.0


def test_evaluate_end_to_end_synthetic():
    rng = np.random.default_rng(5)
    m, n, q = 60, 32, 15

    corpus_emb = _normalize(rng.standard_normal((m, n)).astype(np.float32))
    targets = rng.integers(0, m, size=q)
    query_emb = _normalize(corpus_emb[targets] + 0.1 * rng.standard_normal((q, n)).astype(np.float32))

    doc_ids = [f"d{i}" for i in range(m)]
    query_ids = [f"q{i}" for i in range(q)]
    qrels = {f"q{i}": {f"d{int(targets[i])}": 1} for i in range(q)}
    docs_text = [f"doc {i}" for i in range(m)]

    out = evaluate(
        corpus_emb=corpus_emb,
        query_emb=query_emb,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        docs_text=docs_text,
        bits=8,
        k=10,
    )
    assert out["n_queries"] == q
    assert out["bits"] == 8
    assert out["top1_agreement"] >= 0.9
    assert out["ndcg@k_quantized"] >= 0.9 * out["ndcg@k_float"]
