"""Embedding index (sentence-transformers) over past-flagged cases, with
FAISS cosine-similarity lookup surfaced as supporting evidence for a new
verdict.

Unlike the VLM (src/eval, src/training), a sentence-embedding model is small
enough (all-MiniLM-L6-v2: ~90MB, 384-dim) to actually load and run on this
project's RAM-constrained local dev machine — confirmed by loading it and
encoding real text locally before writing this module. So build_index/query
below are exercised for real in local dev, not deferred to Kaggle; only the
model DOWNLOAD (first call, ~90s once, then HF-cached) is excluded from the
fast automatic test suite, same reasoning field_tamper.py's EasyOCR path
already established (tests/test_pipeline_smoke.py's module docstring).
"""

import json
from pathlib import Path

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger("case_index")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_loaded_model_name = None


def _get_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model {model_name}")
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
    return _model


def case_text(case: dict) -> str:
    """Builds the text actually embedded for a case — combines whatever
    fields are present (document type, verdict, tier, and the model's own
    natural-language explanation) so retrieval surfaces cases similar in WHY
    they were flagged, not merely that they were. Falls back to case_id alone
    if none of the richer fields are present, rather than embedding an empty
    string.
    """
    parts = []
    if case.get("document_code"):
        parts.append(case["document_code"])
    if case.get("tamper_verdict"):
        parts.append(f"verdict: {case['tamper_verdict']}")
    if case.get("tier"):
        parts.append(f"tier: {case['tier']}")
    if case.get("explanation"):
        parts.append(case["explanation"])
    return " | ".join(parts) if parts else str(case.get("case_id", ""))


def build_index(cases: list[dict], model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """cases: list of dicts, each should have a "case_id" plus whatever
    case_text() can use. Returns an in-memory index; see save_index/load_index
    for persistence.
    """
    if not cases:
        raise ValueError("build_index requires at least one case")
    model = _get_model(model_name)
    texts = [case_text(c) for c in cases]
    embeddings = model.encode(texts, normalize_embeddings=True)  # normalized -> inner product == cosine similarity
    return {"model_name": model_name, "cases": cases, "embeddings": np.asarray(embeddings, dtype="float32")}


def save_index(index: dict, out_path: str) -> None:
    """Persists as a .npz (embeddings) + sidecar .json (cases + model_name).
    No FAISS-specific serialization needed at this project's scale (dozens to
    low hundreds of flagged cases) — FAISS is used in-memory at query time via
    IndexFlatIP, a one-line exact cosine-similarity search, rather than
    hand-rolling one or paying for an ANN index this data doesn't need.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path.with_suffix(".npz"), embeddings=index["embeddings"])
    sidecar = {"model_name": index["model_name"], "cases": index["cases"]}
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    logger.info(f"Saved index ({len(index['cases'])} cases) to {out_path.with_suffix('.npz')} / .json")


def load_index(path: str) -> dict:
    path = Path(path)
    embeddings = np.load(str(path.with_suffix(".npz")))["embeddings"]
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return {"model_name": sidecar["model_name"], "cases": sidecar["cases"], "embeddings": embeddings}


def query(index: dict, query_text: str, top_k: int = 3) -> list[dict]:
    """Returns the top_k most similar cases to query_text, each case dict
    annotated with its cosine similarity score, highest first.
    """
    import faiss

    model = _get_model(index["model_name"])
    query_embedding = np.asarray(model.encode([query_text], normalize_embeddings=True), dtype="float32")

    dim = index["embeddings"].shape[1]
    faiss_index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    faiss_index.add(index["embeddings"])
    top_k = min(top_k, len(index["cases"]))
    scores, indices = faiss_index.search(query_embedding, top_k)

    return [{**index["cases"][idx], "similarity": float(score)} for idx, score in zip(indices[0], scores[0])]
