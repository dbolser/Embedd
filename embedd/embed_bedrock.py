"""Embed abstracts via AWS Bedrock Titan Text Embeddings V2.

Titan takes one text per invoke, so we thread many calls and checkpoint to a
memmap + done-bitmap on disk. The run is fully resumable: rerun and it skips
indices already embedded. Handles Bedrock throttling with exponential backoff.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REGION = "eu-west-2"
MODEL = "amazon.titan-embed-text-v2:0"
DIM = 1024


def _client():
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(retries={"max_attempts": 0},
                                      read_timeout=30, connect_timeout=10))


def _embed_one(client, text: str, max_retry: int = 6):
    body = json.dumps({"inputText": text[:40000], "dimensions": DIM,
                       "normalize": True})
    delay = 0.5
    for attempt in range(max_retry):
        try:
            resp = client.invoke_model(modelId=MODEL, body=body,
                                       contentType="application/json",
                                       accept="application/json")
            return np.asarray(json.loads(resp["body"].read())["embedding"],
                              dtype=np.float32)
        except Exception as e:  # noqa: BLE001 - throttling / transient
            name = type(e).__name__
            if "Throttl" in name or "TooMany" in name or "Timeout" in name or attempt < 2:
                time.sleep(delay); delay = min(delay * 2, 8.0)
                continue
            if attempt == max_retry - 1:
                return None
            time.sleep(delay); delay = min(delay * 2, 8.0)
    return None


def embed_titan(texts: list[str], workdir: Path, workers: int = 12,
                flush_every: int = 2000) -> np.ndarray:
    """Embed texts with checkpointing under workdir. Resumable."""
    workdir.mkdir(parents=True, exist_ok=True)
    N = len(texts)
    vecs_path = workdir / "vecs.f32.memmap"
    done_path = workdir / "done.npy"
    vecs = np.memmap(vecs_path, dtype=np.float32, mode=("r+" if vecs_path.exists() else "w+"),
                     shape=(N, DIM))
    done = np.load(done_path) if done_path.exists() else np.zeros(N, dtype=bool)
    todo = np.where(~done)[0]
    print(f"titan: {N} texts, {len(todo)} remaining", flush=True)
    if len(todo) == 0:
        return np.asarray(vecs)

    local = threading.local()
    lock = threading.Lock()
    counter = {"n": int(done.sum()), "fail": 0, "t0": time.time()}

    def worker(i):
        if getattr(local, "client", None) is None:
            local.client = _client()
        v = _embed_one(local.client, texts[i])
        if v is None:
            with lock:
                counter["fail"] += 1
            return
        vecs[i] = v
        with lock:
            done[i] = True
            counter["n"] += 1
            if counter["n"] % flush_every == 0:
                vecs.flush(); np.save(done_path, done)
                rate = counter["n"] / (time.time() - counter["t0"] + 1e-9)
                print(f"  {counter['n']}/{N}  {rate:.0f}/s  fails={counter['fail']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(worker, todo))
    vecs.flush(); np.save(done_path, done)
    print(f"titan done: {int(done.sum())}/{N} embedded, {counter['fail']} fails", flush=True)
    return np.asarray(vecs)
