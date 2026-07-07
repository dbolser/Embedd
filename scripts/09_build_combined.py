"""Concatenate the field corpus and the random control corpus into one realistic
'PubMed-like' set (tag 'combined') where the exploding fields are a small
minority diluted in outcome-blind random abstracts. De-duplicates by PMID,
keeping the field/background label over the generic 'random' label.
"""
from __future__ import annotations

import json

import numpy as np

from embedd import config as C
from embedd import embed


def main():
    Ef, mf = embed.load_embeddings("local")
    Er, mr = embed.load_embeddings("random")
    print(f"field={Ef.shape}  random={Er.shape}")

    seen = {m["pmid"]: i for i, m in enumerate(mf)}  # field wins on collision
    rows_E = [Ef]
    rows_m = list(mf)
    keep_idx = []
    for i, m in enumerate(mr):
        if m["pmid"] in seen:
            continue
        seen[m["pmid"]] = True
        keep_idx.append(i)
        rows_m.append(m)
    E = np.vstack([Ef, Er[keep_idx]]).astype(np.float32)
    print(f"combined={E.shape}  (dropped {len(mr)-len(keep_idx)} random dups)")

    np.save(C.EMBEDDINGS / "vecs_combined.npy", E)
    with (C.EMBEDDINGS / "meta_combined.jsonl").open("w") as fh:
        for m in rows_m:
            fh.write(json.dumps(m) + "\n")
    print("saved combined embeddings")


if __name__ == "__main__":
    main()
