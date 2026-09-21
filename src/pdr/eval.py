from __future__ import annotations

import numpy as np

from pdr.codec import build_D
from pdr.protocol import Client, make_server
from pdr.quantize import quantize


def ndcg_at_k(qrel: dict[str, int], ranking: list[str], k: int = 10) -> float:
    def dcg(scores: list[int]) -> float:
        return float(sum(s / np.log2(i + 2) for i, s in enumerate(scores)))

    gains = [qrel.get(d, 0) for d in ranking[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    if sum(ideal) == 0:
        return 0.0
    return dcg(gains) / dcg(ideal)


def recall_at_k(qrel: dict[str, int], ranking: list[str], k: int = 10) -> float:
    relevant = {d for d, s in qrel.items() if s > 0}
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def _argsort_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores)[::-1]


class ParentCollapse:
    """Rank documents from chunk scores: a document's score is the best of its chunks.

    Chunk ids are ``<doc>#<i>`` (scale/chunk_corpus.py); ``parent_sep`` splits them. Scores over
    chunks are reduced to one per document with ``np.maximum.reduceat`` on a chunk order grouped by
    document, so rankings, top-1 agreement and the metrics are all at document level while the
    protocol still scores every chunk.
    """

    def __init__(self, doc_ids: list[str], parent_sep: str = "#"):
        parents = [d.rsplit(parent_sep, 1)[0] for d in doc_ids]
        uniq = sorted(set(parents))
        index = {d: i for i, d in enumerate(uniq)}
        group = np.array([index[p] for p in parents])
        self.perm = np.argsort(group, kind="stable")
        sorted_group = group[self.perm]
        self.starts = np.flatnonzero(np.r_[True, sorted_group[1:] != sorted_group[:-1]])
        self.ids = [uniq[g] for g in sorted_group[self.starts]]
        self.n_chunks, self.n_docs = len(doc_ids), len(uniq)

    def scores(self, chunk_scores: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(chunk_scores[self.perm], self.starts)

    def qrels(self, qrels: dict[str, dict[str, int]], parent_sep: str = "#") -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for q, rel in qrels.items():
            for d, sc in rel.items():
                pd = d.rsplit(parent_sep, 1)[0]
                out.setdefault(q, {})[pd] = max(sc, out.get(q, {}).get(pd, 0))
        return out


def evaluate_float(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    collapse: ParentCollapse | None = None,
) -> dict:
    """Float-only retrieval quality: no quantization, no protocol, no D matrix.

    Used by the encoder screen, whose question is only whether a set of models
    lands at comparable floating-point quality before any bit-width sweep is run.
    """
    scores = corpus_emb @ query_emb.T  # (n_docs, n_queries)
    if collapse is not None:
        qrels = collapse.qrels(qrels)
        doc_ids = collapse.ids

    ndcg, recall = [], []
    for qi, qid in enumerate(query_ids):
        qrel = qrels.get(qid)
        if not qrel:
            continue
        col = scores[:, qi] if collapse is None else collapse.scores(scores[:, qi])
        order = _argsort_desc(col)[:k]
        ranking = [doc_ids[i] for i in order]
        ndcg.append(ndcg_at_k(qrel, ranking, k))
        recall.append(recall_at_k(qrel, ranking, k))

    n = len(ndcg)
    return {
        "n_queries": n,
        "k": k,
        "ndcg@k_float": float(np.mean(ndcg)) if n else 0.0,
        "recall@k_float": float(np.mean(recall)) if n else 0.0,
    }


def evaluate(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    docs_text: list[str],
    bits: int = 2,
    k: int = 10,
    clip: float | None = None,
    collapse: ParentCollapse | None = None,
) -> dict:
    A_int, s_A = quantize(corpus_emb, bits=bits, clip_percentile=clip)
    if collapse is not None:
        qrels = collapse.qrels(qrels)
        rank_ids = collapse.ids
    else:
        rank_ids = doc_ids
    D_int, _ = build_D(docs_text)
    server = make_server(A_int, D_int)
    client = Client()

    ndcg_q, recall_q = [], []
    ndcg_f, recall_f = [], []
    top1_agree = 0
    n_evaluated = 0
    for qi, x_float in enumerate(query_emb):
        qid = query_ids[qi]
        qrel = qrels.get(qid)
        if not qrel:
            continue
        n_evaluated += 1

        x_int, _ = quantize(x_float, bits=bits, clip_percentile=clip)
        c1 = client.round1_query(x_int)
        scores_q = server.round1(c1)
        if collapse is not None:
            scores_q = collapse.scores(scores_q)
        order_q = _argsort_desc(scores_q)
        rank_q = [rank_ids[i] for i in order_q[:k]]

        scores_f = corpus_emb @ x_float
        if collapse is not None:
            scores_f = collapse.scores(scores_f)
        order_f = _argsort_desc(scores_f)
        rank_f = [rank_ids[i] for i in order_f[:k]]

        if order_q[0] == order_f[0]:
            top1_agree += 1
        ndcg_q.append(ndcg_at_k(qrel, rank_q, k))
        recall_q.append(recall_at_k(qrel, rank_q, k))
        ndcg_f.append(ndcg_at_k(qrel, rank_f, k))
        recall_f.append(recall_at_k(qrel, rank_f, k))

    return {
        "n_queries": n_evaluated,
        "bits": bits,
        "k": k,
        "clip": clip,
        "collapsed_to_documents": collapse is not None,
        "n_docs": collapse.n_docs if collapse is not None else len(doc_ids),
        "s_A": float(s_A),
        "ndcg@k_quantized": float(np.mean(ndcg_q)) if ndcg_q else 0.0,
        "recall@k_quantized": float(np.mean(recall_q)) if recall_q else 0.0,
        "ndcg@k_float": float(np.mean(ndcg_f)) if ndcg_f else 0.0,
        "recall@k_float": float(np.mean(recall_f)) if recall_f else 0.0,
        "top1_agreement": top1_agree / n_evaluated if n_evaluated else 0.0,
    }
