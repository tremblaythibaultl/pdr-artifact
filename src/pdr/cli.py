from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from pdr.config import Config
from pdr.data import corpus_to_lists, load_beir, queries_to_lists
from pdr.embed import REGISTRY, encode, resolve
from pdr.eval import ParentCollapse, evaluate, evaluate_float

# Canonical model ids (others fall back to the sentence-transformers backend).
_MODEL_HELP = "embedding model id; canonical: " + ", ".join(REGISTRY) + " (any HF repo also works)"

# Order of columns written to the plot-ready CSV. Keys map evaluate()'s output
# (with the "@k" metric names flattened) plus dataset/model provenance.
CSV_COLUMNS = [
    "dataset",
    "model",
    "k",
    "bits",
    "n_queries",
    "s_A",
    "ndcg_quantized",
    "recall_quantized",
    "ndcg_float",
    "recall_float",
    "top1_agreement",
]


# Columns of the float-only encoder screen (gate before any bit-width sweep).
SCREEN_COLUMNS = [
    "dataset",
    "model",
    "dim",
    "k",
    "n_queries",
    "ndcg_float",
    "recall_float",
    "query_prefix",
    "doc_prefix",
]


def _cache_path(
    cache_dir: Path, dataset: str, model: str, kind: str, n: int, prefix: str = ""
) -> Path:
    """Cache file for one (dataset, model, kind) encode.

    The instruction prefix is part of the key: two runs of the same model with
    different prefixes are different embeddings. Empty prefixes add no tag, so
    caches written before prefixes existed stay valid wherever the prefix is
    empty (and the pre-instruction BGE query cache stays on disk under its old
    name, which is what the published BGE numbers were computed from).
    """
    safe_model = model.replace("/", "_")
    tag = "__p" + hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:6] if prefix else ""
    return cache_dir / f"{dataset}__{safe_model}__{kind}__n{n}{tag}.npy"


def _cached_encode(model: str, texts: list[str], cache_file: Path, is_query: bool = False) -> np.ndarray:
    if cache_file.exists():
        emb = np.load(cache_file)
        if emb.shape[0] != len(texts):
            raise RuntimeError(
                f"cache {cache_file} has {emb.shape[0]} rows but {len(texts)} texts were requested; "
                "delete the stale cache and re-run."
            )
        return emb
    emb = encode(model, texts, normalize=True, is_query=is_query)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    return emb


def _load_data_and_embeddings(args: argparse.Namespace, cfg: Config) -> dict:
    """Load BEIR corpus/queries/qrels and the (cached) embeddings once.

    Shared by `eval` and `sweep` so a bit-width sweep pays the load + encode
    cost a single time.
    """
    cache_dir = Path(args.cache_dir)
    data_root = Path(args.data_root)

    corpus, queries, qrels = load_beir(cfg.dataset, data_root)
    doc_ids, docs_text = corpus_to_lists(corpus)
    qids, q_text = queries_to_lists(queries)

    if args.limit_corpus:
        doc_ids = doc_ids[: args.limit_corpus]
        docs_text = docs_text[: args.limit_corpus]
    if args.limit_queries:
        qids = qids[: args.limit_queries]
        q_text = q_text[: args.limit_queries]

    spec = resolve(cfg.model)
    corpus_emb = _cached_encode(
        cfg.model,
        docs_text,
        _cache_path(
            cache_dir, cfg.dataset, cfg.model, "corpus", len(docs_text), spec.doc_prefix
        ),
    )
    query_emb = _cached_encode(
        cfg.model,
        q_text,
        _cache_path(
            cache_dir, cfg.dataset, cfg.model, "queries", len(q_text), spec.query_prefix
        ),
        is_query=True,
    )

    return {
        "corpus_emb": corpus_emb,
        "query_emb": query_emb,
        "doc_ids": doc_ids,
        "query_ids": qids,
        "qrels": qrels,
        "docs_text": docs_text,
    }


def _flatten_metrics(metrics: dict, dataset: str, model: str) -> dict:
    """Flatten evaluate()'s output into a CSV-friendly record with provenance."""
    return {
        "dataset": dataset,
        "model": model,
        "k": metrics["k"],
        "bits": metrics["bits"],
        "n_queries": metrics["n_queries"],
        "s_A": metrics["s_A"],
        "ndcg_quantized": metrics["ndcg@k_quantized"],
        "recall_quantized": metrics["recall@k_quantized"],
        "ndcg_float": metrics["ndcg@k_float"],
        "recall_float": metrics["recall@k_float"],
        "top1_agreement": metrics["top1_agreement"],
    }


def results_stem(dataset: str, model: str, k: int, clip: float | None = None, collapse: bool = False) -> str:
    """File stem for one sweep. Clipped sweeps get a __clip<p> suffix so the
    max-abs files are never overwritten (they are the appendix baseline); sweeps
    that rank chunks but score documents (--collapse) get a __doc suffix."""
    stem = f"{dataset}__{model.replace('/', '_')}__k{k}"
    if clip is not None and clip < 100:
        stem += f"__clip{clip:g}"
    if collapse:
        stem += "__doc"
    return stem


def _write_results(
    results_dir: Path, dataset: str, model: str, k: int, new_records: list[dict], clip: float | None = None,
    collapse: bool = False,
) -> Path:
    """Upsert records (keyed by bits) into per-(dataset, model, k[, clip]) CSV + JSON files.

    Re-running the sweep with different bit-widths accumulates rows rather than
    clobbering; re-running with the same bits overwrites that row.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = results_stem(dataset, model, k, clip, collapse)
    csv_path = results_dir / f"{stem}.csv"
    json_path = results_dir / f"{stem}.json"

    by_bits: dict[int, dict] = {}
    if json_path.exists():
        existing = json.loads(json_path.read_text())
        for rec in existing.get("records", []):
            by_bits[rec["bits"]] = rec
    for rec in new_records:
        by_bits[rec["bits"]] = rec

    records = [by_bits[b] for b in sorted(by_bits)]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow({c: rec[c] for c in CSV_COLUMNS})

    payload = {
        "dataset": dataset,
        "model": model,
        "k": k,
        "clip_percentile": clip,
        "collapsed_to_documents": collapse,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return csv_path


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = Config(model=args.model, dataset=args.dataset, bits=args.bits, k=args.k, clip=args.clip)
    data = _load_data_and_embeddings(args, cfg)
    collapse = ParentCollapse(data["doc_ids"]) if args.collapse else None

    metrics = evaluate(**data, bits=cfg.bits, k=cfg.k, clip=cfg.clip, collapse=collapse)
    print(json.dumps(metrics, indent=2))


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = Config(model=args.model, dataset=args.dataset, bits=args.bits[0], k=args.k, clip=args.clip)
    data = _load_data_and_embeddings(args, cfg)
    collapse = ParentCollapse(data["doc_ids"]) if args.collapse else None
    if collapse is not None:
        print(f"[sweep] ranking {collapse.n_chunks} chunks, scoring {collapse.n_docs} documents", flush=True)

    records = []
    for bits in args.bits:
        print(f"[sweep] evaluating bits={bits} clip={cfg.clip} ...", flush=True)
        metrics = evaluate(**data, bits=bits, k=cfg.k, clip=cfg.clip, collapse=collapse)
        print(json.dumps(metrics, indent=2))
        records.append(_flatten_metrics(metrics, cfg.dataset, cfg.model))

    csv_path = _write_results(Path(args.results_dir), cfg.dataset, cfg.model, cfg.k, records, clip=cfg.clip,
                              collapse=args.collapse)
    print(f"[sweep] wrote {csv_path} (+ .json) for bits={args.bits}")


def _write_screen(results_dir: Path, k: int, new_records: list[dict]) -> Path:
    """Upsert screen rows (keyed by dataset+model) into results/float_screen__k<k>.*."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"float_screen__k{k}"
    csv_path = results_dir / f"{stem}.csv"
    json_path = results_dir / f"{stem}.json"

    by_key: dict[tuple[str, str], dict] = {}
    if json_path.exists():
        existing = json.loads(json_path.read_text())
        for rec in existing.get("records", []):
            by_key[(rec["dataset"], rec["model"])] = rec
    for rec in new_records:
        by_key[(rec["dataset"], rec["model"])] = rec

    records = [by_key[key] for key in sorted(by_key)]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCREEN_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow({c: rec[c] for c in SCREEN_COLUMNS})

    payload = {
        "k": k,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return csv_path


def cmd_screen(args: argparse.Namespace) -> None:
    """Float-only quality screen across models and datasets.

    Answers one question: do these encoders land at comparable floating-point
    nDCG@k? Runs no quantization and no protocol, so it is cheap enough to gate
    a bit-width sweep on.
    """
    records = []
    for dataset in args.datasets:
        for model in args.models:
            cfg = Config(model=model, dataset=dataset, bits=8, k=args.k)
            spec = resolve(model)
            print(f"[screen] {dataset} / {model} ...", flush=True)
            data = _load_data_and_embeddings(args, cfg)
            metrics = evaluate_float(
                corpus_emb=data["corpus_emb"],
                query_emb=data["query_emb"],
                doc_ids=data["doc_ids"],
                query_ids=data["query_ids"],
                qrels=data["qrels"],
                k=cfg.k,
            )
            rec = {
                "dataset": dataset,
                "model": model,
                "dim": int(data["corpus_emb"].shape[1]),
                "k": metrics["k"],
                "n_queries": metrics["n_queries"],
                "ndcg_float": metrics["ndcg@k_float"],
                "recall_float": metrics["recall@k_float"],
                "query_prefix": spec.query_prefix,
                "doc_prefix": spec.doc_prefix,
            }
            print(json.dumps(rec, indent=2), flush=True)
            records.append(rec)

    csv_path = _write_screen(Path(args.results_dir), args.k, records)
    print(f"[screen] wrote {csv_path} (+ .json)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pdr")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Run end-to-end evaluation on a BEIR dataset.")
    p_eval.add_argument("--dataset", default="scifact")
    p_eval.add_argument("--model", default="BAAI/bge-large-en-v1.5", help=_MODEL_HELP)
    p_eval.add_argument("--bits", type=int, default=8)
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.add_argument("--data-root", default=".beir-data")
    p_eval.add_argument("--cache-dir", default=".pdr-cache")
    p_eval.add_argument("--limit-corpus", type=int, default=None)
    p_eval.add_argument("--limit-queries", type=int, default=None)
    p_eval.add_argument("--collapse", action="store_true", help="see sweep --collapse")
    p_eval.add_argument("--clip", type=float, default=None, help="see sweep --clip")
    p_eval.set_defaults(func=cmd_eval)

    p_sweep = sub.add_parser(
        "sweep", help="Run evaluation over several bit-widths and store plot-ready CSV/JSON."
    )
    p_sweep.add_argument("--dataset", default="scifact")
    p_sweep.add_argument("--model", default="BAAI/bge-large-en-v1.5", help=_MODEL_HELP)
    p_sweep.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    p_sweep.add_argument("--k", type=int, default=10)
    p_sweep.add_argument("--data-root", default=".beir-data")
    p_sweep.add_argument("--cache-dir", default=".pdr-cache")
    p_sweep.add_argument("--results-dir", default="results")
    p_sweep.add_argument("--limit-corpus", type=int, default=None)
    p_sweep.add_argument("--limit-queries", type=int, default=None)
    p_sweep.add_argument(
        "--collapse", action="store_true",
        help="chunk corpus (ids <doc>#<i>): rank chunks, score documents by their best chunk, evaluate against document qrels",
    )
    p_sweep.add_argument(
        "--clip", type=float, default=None,
        help="percentile of |X| mapped to qmax (entries beyond saturate); omit for max-abs",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_screen = sub.add_parser(
        "screen",
        help="Float-only quality screen over models x datasets (no quantization).",
    )
    p_screen.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    p_screen.add_argument(
        "--models",
        nargs="+",
        default=[
            "mixedbread-ai/mxbai-embed-large-v1",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2",
            "thenlper/gte-large",
        ],
        help=_MODEL_HELP,
    )
    p_screen.add_argument("--k", type=int, default=10)
    p_screen.add_argument("--data-root", default=".beir-data")
    p_screen.add_argument("--cache-dir", default=".pdr-cache")
    p_screen.add_argument("--results-dir", default="results")
    p_screen.add_argument("--limit-corpus", type=int, default=None)
    p_screen.add_argument("--limit-queries", type=int, default=None)
    p_screen.set_defaults(func=cmd_screen)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
