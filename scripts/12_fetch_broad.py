"""Broad corpus for the concept-relation / moves-of-science analysis:
  * ~200k outcome-blind random abstracts (breadth), sampled by date only, plus
  * every paper for any focused topic defined in `topics.local.json` (untracked),
    each as {"name": "<pubmed query>"} -- so no specific domains live in the repo.

Written to data/raw_broad/*.jsonl so it stays separate from the pilot corpus.
Resumable: each file is skipped if already complete.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from Bio import Entrez

from embedd import config as C
from embedd import fetch

Entrez.email = C.ENTREZ_EMAIL
OUT = C.DATA / "raw_broad"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_TARGET = 200_000
BROAD = '(hasabstract[text] AND English[lang] AND "journal article"[pt])'
QUARTERS = [("01", "03"), ("04", "06"), ("07", "09"), ("10", "12")]

# Focused topic queries are kept out of the repo; define them in an untracked
# topics.local.json at the project root, e.g. {"topic_a": "<pubmed query>"}.
_topics_file = C.ROOT / "topics.local.json"
TOPICS = json.loads(_topics_file.read_text()) if _topics_file.exists() else {}


def esearch_window(term, retmax, sort="pub_date", datestr=None):
    q = term if datestr is None else f'({term}) AND ({datestr})'
    h = fetch._retry(Entrez.esearch, db="pubmed", term=q, retmax=retmax, sort=sort)
    rec = Entrez.read(h); h.close(); time.sleep(0.34)
    return list(rec["IdList"])


def fetch_topic_all(name, query):
    out = OUT / f"{name}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[{name}] exists, skipping", flush=True)
        return
    seen, n, t0 = set(), 0, time.time()
    with out.open("w") as fh:
        for year in range(C.YEAR_MIN, C.YEAR_MAX + 1):
            ds = f'"{year}/01/01"[pdat] : "{year}/12/31"[pdat]'
            pmids = [p for p in esearch_window(query, 9999, datestr=ds) if p not in seen]
            for rec in fetch.efetch_records(pmids):
                if rec["pmid"] in seen:
                    continue
                seen.add(rec["pmid"]); rec["field"] = name; rec["is_founder"] = False
                fh.write(json.dumps(rec) + "\n"); n += 1
            print(f"[{name}] {year}: {n:6d} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{name}] DONE {n} records", flush=True)


def fetch_random_big():
    out = OUT / "random.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print("[random] exists, skipping", flush=True)
        return
    per_q = RANDOM_TARGET // ((C.YEAR_MAX - C.YEAR_MIN + 1) * 4) + 1
    seen, n, t0 = set(), 0, time.time()
    with out.open("w") as fh:
        for year in range(C.YEAR_MIN, C.YEAR_MAX + 1):
            for mmin, mmax in QUARTERS:
                ds = f'"{year}/{mmin}/01"[pdat] : "{year}/{mmax}/28"[pdat]'
                pmids = [p for p in esearch_window(BROAD, per_q, datestr=ds)
                         if p not in seen]
                for rec in fetch.efetch_records(pmids):
                    if rec["pmid"] in seen:
                        continue
                    seen.add(rec["pmid"]); rec["field"] = "random"; rec["is_founder"] = False
                    fh.write(json.dumps(rec) + "\n"); n += 1
            print(f"[random] {year}: {n:6d} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[random] DONE {n} records", flush=True)


def main():
    for name, q in TOPICS.items():
        fetch_topic_all(name, q)
    fetch_random_big()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
