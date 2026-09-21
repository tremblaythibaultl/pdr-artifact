from __future__ import annotations

from pathlib import Path

# Corpora built by scale/chunk_corpus.py and scale/build_bioasq.py ship in data/ at the
# repository root. They are not BEIR downloads, so load_beir looks there before asking
# the BEIR server for a dataset it does not find under data_root.
_LOCAL_DATA = Path(__file__).resolve().parents[2] / "data"


def load_beir(dataset: str, data_root: Path, split: str = "test"):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    target_dir = data_root / dataset
    if not target_dir.exists() and (_LOCAL_DATA / dataset).exists():
        target_dir = _LOCAL_DATA / dataset
    if not target_dir.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        util.download_and_unzip(url, str(data_root))
    corpus, queries, qrels = GenericDataLoader(data_folder=str(target_dir)).load(split=split)
    return corpus, queries, qrels


def corpus_to_lists(corpus: dict) -> tuple[list[str], list[str]]:
    doc_ids = list(corpus.keys())
    texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]
    return doc_ids, texts


def queries_to_lists(queries: dict) -> tuple[list[str], list[str]]:
    qids = list(queries.keys())
    texts = [queries[q] for q in qids]
    return qids, texts
