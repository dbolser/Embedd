"""Embed abstracts into a shared vector space.

Two interchangeable backends behind one interface:
  * local  -> a PubMedBERT-based sentence-transformer (default; free, reproducible)
  * openai -> text-embedding-3-large (cross-check; needs OPENAI_API_KEY)

Both return L2-normalized vectors so cosine similarity == dot product.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import config as C


def load_corpus() -> list[dict]:
    """Load and de-duplicate all raw JSONL records by PMID.

    A PMID may appear in both a field file and the background; we keep the
    field-labeled copy so ground-truth founders are never masked as background.
    """
    records: dict[str, dict] = {}
    files = sorted(C.RAW.glob("*.jsonl"))
    # load background first so field records overwrite it on collision
    files = sorted(files, key=lambda p: p.name != "background.jsonl")
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pmid = rec["pmid"]
            if pmid in records and records[pmid]["field"] != "background":
                continue  # keep existing field label
            records[pmid] = rec
    return list(records.values())


def _texts(records: list[dict]) -> list[str]:
    """Title + abstract is the unit of meaning we embed."""
    return [f"{r['title']} {r['abstract']}".strip() for r in records]


def embed_local(records: list[dict], model_name: str | None = None,
                batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_name = model_name or C.LOCAL_MODEL
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        _texts(records),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_openai(records: list[dict], model_name: str | None = None,
                 batch_size: int = 256) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = model_name or C.OPENAI_MODEL
    texts = [t[:30000] for t in _texts(records)]  # stay well under token limit
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=chunk)
        out.extend([d.embedding for d in resp.data])
        print(f"  openai embed {i+len(chunk)}/{len(texts)}", flush=True)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def build(backend: str = "local") -> tuple[np.ndarray, list[dict]]:
    """Embed the whole corpus, persist vectors + aligned metadata, return both."""
    records = load_corpus()
    records.sort(key=lambda r: (r["year"], r["pmid"]))  # stable, time-ordered
    if backend == "local":
        vecs = embed_local(records)
        tag = "local"
    elif backend == "openai":
        vecs = embed_openai(records)
        tag = "openai"
    else:
        raise ValueError(backend)

    np.save(C.EMBEDDINGS / f"vecs_{tag}.npy", vecs)
    meta_path = C.EMBEDDINGS / f"meta_{tag}.jsonl"
    with meta_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps({k: r[k] for k in (
                "pmid", "year", "title", "field", "is_founder",
                "first_author", "journal")}) + "\n")
    print(f"saved {vecs.shape} vectors -> {tag}")
    return vecs, records


def load_embeddings(tag: str = "local") -> tuple[np.ndarray, list[dict]]:
    vecs = np.load(C.EMBEDDINGS / f"vecs_{tag}.npy")
    meta = [json.loads(l) for l in (C.EMBEDDINGS / f"meta_{tag}.jsonl").read_text().splitlines()]
    return vecs, meta
