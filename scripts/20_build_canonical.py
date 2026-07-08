"""Build ONE canonical corpus = the de-duplicated union of every abstract we have
fetched (data/raw + data/raw_broad). Persist it with a stable row order so that
each embedding backend (PubMedBERT / Titan / OpenAI) embeds the SAME rows in the
SAME order, and all three vector sets align index-for-index.

Writes data/canonical/corpus.jsonl (full records) and prints provenance.
"""
from __future__ import annotations

import json
from collections import Counter

from embedd import config as C

OUT = C.DATA / "canonical"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    records: dict[str, dict] = {}
    provenance: dict[str, set] = {}
    dirs = [C.RAW, C.DATA / "raw_broad"]
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")):
            src = f.stem
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                pmid = r["pmid"]
                provenance.setdefault(pmid, set()).add(src)
                # keep the richest copy; prefer a named field over random/background
                if pmid not in records or (
                    records[pmid].get("field") in ("random", "background")
                    and r.get("field") not in ("random", "background")):
                    records[pmid] = r

    recs = list(records.values())
    recs.sort(key=lambda r: (r["year"], r["pmid"]))  # stable, deterministic
    for r in recs:
        r["sources"] = sorted(provenance[r["pmid"]])

    out = OUT / "corpus.jsonl"
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    print(f"canonical corpus: {len(recs)} unique abstracts -> {out}")
    print("years:", min(r["year"] for r in recs), "-", max(r["year"] for r in recs))
    src_counts = Counter(s for r in recs for s in r["sources"])
    print("provenance (abstracts touching each source):")
    for s, n in src_counts.most_common():
        print(f"  {s:16s} {n:7d}")


if __name__ == "__main__":
    main()
