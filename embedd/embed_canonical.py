"""Embed the one canonical corpus with any backend, aligned row-for-row.

All backends write a checkpointed memmap (+ done bitmap) so a multi-hour run
survives interruption and resumes. The shared meta (data/embeddings/
meta_canonical.jsonl) is written once; every vecs_canonical_<backend>.npy lines
up index-for-index with it.

Backends: 'pubbert' (local PubMedBERT, free), 'titan' (Bedrock), 'openai'.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config as C

DIMS = {"pubbert": 768, "titan": 1024, "openai": 1536}
OPENAI_MODEL = "text-embedding-3-small"  # "ada-class" modern default (1536-d)


def load_canonical() -> list[dict]:
    path = C.DATA / "canonical" / "corpus.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: (r["year"], r["pmid"]))  # must match build order
    return recs


def _texts(recs):
    return [f"{r['title']} {r['abstract']}".strip() for r in recs]


def write_meta(recs):
    out = C.EMBEDDINGS / "meta_canonical.jsonl"
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps({k: r.get(k) for k in (
                "pmid", "year", "title", "field", "sources",
                "first_author", "journal")}) + "\n")
    return out


def _memmap(backend, N):
    dim = DIMS[backend]
    wd = C.EMBEDDINGS / f"canon_{backend}_ckpt"
    wd.mkdir(parents=True, exist_ok=True)
    vpath, dpath = wd / "vecs.f32", wd / "done.npy"
    vecs = np.memmap(vpath, dtype=np.float32,
                     mode=("r+" if vpath.exists() else "w+"), shape=(N, dim))
    done = np.load(dpath) if dpath.exists() else np.zeros(N, dtype=bool)
    return vecs, done, dpath


def _flush(vecs, done, dpath, n, N, t0, fails):
    vecs.flush(); np.save(dpath, done)
    print(f"  {n}/{N}  {n/(time.time()-t0+1e-9):.0f}/s  fails={fails}", flush=True)


def embed_pubbert(texts, N):
    from sentence_transformers import SentenceTransformer
    vecs, done, dpath = _memmap("pubbert", N)
    todo = np.where(~done)[0]
    if not len(todo):
        return np.asarray(vecs)
    model = SentenceTransformer(C.LOCAL_MODEL)
    t0 = time.time(); chunk = 2000
    for s in range(0, len(todo), chunk):
        idx = todo[s:s + chunk]
        emb = model.encode([texts[i] for i in idx], batch_size=64,
                           normalize_embeddings=True, convert_to_numpy=True)
        vecs[idx] = emb.astype(np.float32)
        done[idx] = True
        _flush(vecs, done, dpath, int(done.sum()), N, t0, 0)
    return np.asarray(vecs)


def embed_openai(texts, N):
    from openai import OpenAI
    client = OpenAI()
    vecs, done, dpath = _memmap("openai", N)
    todo = np.where(~done)[0]
    if not len(todo):
        return np.asarray(vecs)
    # OpenAI caps embeddings requests at 300k tokens; 500 abstracts ~ 165k, safe.
    t0 = time.time(); batch = 500; fails = 0
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        chunk = [texts[i][:24000] or " " for i in idx]
        try:
            resp = client.embeddings.create(model=OPENAI_MODEL, input=chunk)
            arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            vecs[idx] = arr
            done[idx] = True
        except Exception as e:  # noqa: BLE001
            fails += len(idx)
            print(f"  openai batch fail @{s}: {type(e).__name__}", flush=True)
            time.sleep(3)
        if s % (batch * 10) == 0:
            _flush(vecs, done, dpath, int(done.sum()), N, t0, fails)
    _flush(vecs, done, dpath, int(done.sum()), N, t0, fails)
    return np.asarray(vecs)


def embed_titan(texts, N, workers=48):
    from . import embed_bedrock
    vecs, done, dpath = _memmap("titan", N)
    todo = np.where(~done)[0]
    if not len(todo):
        return np.asarray(vecs)
    local = threading.local(); lock = threading.Lock()
    cnt = {"n": int(done.sum()), "fail": 0, "t0": time.time()}

    def work(i):
        if getattr(local, "cl", None) is None:
            local.cl = embed_bedrock._client()
        v = embed_bedrock._embed_one(local.cl, texts[i])
        with lock:
            if v is None:
                cnt["fail"] += 1
            else:
                vecs[i] = v; done[i] = True; cnt["n"] += 1
                if cnt["n"] % 2000 == 0:
                    _flush(vecs, done, dpath, cnt["n"], N, cnt["t0"], cnt["fail"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    _flush(vecs, done, dpath, int(done.sum()), N, cnt["t0"], cnt["fail"])
    return np.asarray(vecs)


def build(backend: str):
    recs = load_canonical()
    N = len(recs)
    write_meta(recs)
    texts = _texts(recs)
    print(f"embedding {N} canonical abstracts with {backend}", flush=True)
    fn = {"pubbert": embed_pubbert, "openai": embed_openai, "titan": embed_titan}[backend]
    vecs = fn(texts, N)
    np.save(C.EMBEDDINGS / f"vecs_canonical_{backend}.npy", np.asarray(vecs))
    print(f"saved vecs_canonical_{backend}.npy {np.asarray(vecs).shape}", flush=True)
