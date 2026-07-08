"""Titan-embed the broad corpus (random + any focused topic corpora). Resumable
& checkpointed. Saves vecs_broad.npy + meta_broad.jsonl in data/embeddings/."""
from __future__ import annotations

import json

import numpy as np

from embedd import config as C
from embedd import embed_bedrock

RAW_BROAD = C.DATA / "raw_broad"


def load_broad():
    records: dict[str, dict] = {}
    # topic files win over random on PMID collision
    files = sorted(RAW_BROAD.glob("*.jsonl"),
                   key=lambda p: p.name == "random.jsonl")
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["pmid"] in records and records[r["pmid"]]["field"] != "random":
                continue
            records[r["pmid"]] = r
    recs = list(records.values())
    recs.sort(key=lambda r: (r["year"], r["pmid"]))
    return recs


def main():
    recs = load_broad()
    import collections
    print("broad corpus:", len(recs),
          dict(collections.Counter(r["field"] for r in recs)), flush=True)
    texts = [f"{r['title']} {r['abstract']}" for r in recs]

    vecs = embed_bedrock.embed_titan(texts, C.EMBEDDINGS / "titan_broad_ckpt",
                                     workers=48)
    np.save(C.EMBEDDINGS / "vecs_broad.npy", np.asarray(vecs))
    with (C.EMBEDDINGS / "meta_broad.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps({k: r.get(k) for k in (
                "pmid", "year", "title", "field", "is_founder",
                "first_author", "journal")}) + "\n")
    print(f"saved vecs_broad.npy {np.asarray(vecs).shape}", flush=True)


if __name__ == "__main__":
    main()
