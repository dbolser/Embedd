"""Embed only the outcome-blind random control corpus, tagged 'random'."""
from __future__ import annotations

import json

import numpy as np

from embedd import config as C
from embedd import embed


def main():
    records = [json.loads(l) for l in (C.RAW / "random.jsonl").read_text().splitlines() if l.strip()]
    records.sort(key=lambda r: (r["year"], r["pmid"]))
    print(f"embedding {len(records)} random abstracts ...", flush=True)
    vecs = embed.embed_local(records)
    np.save(C.EMBEDDINGS / "vecs_random.npy", vecs)
    with (C.EMBEDDINGS / "meta_random.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps({k: r.get(k) for k in (
                "pmid", "year", "title", "field", "is_founder",
                "first_author", "journal")}) + "\n")
    print(f"saved {vecs.shape} -> random", flush=True)


if __name__ == "__main__":
    main()
