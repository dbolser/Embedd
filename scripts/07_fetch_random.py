"""Outcome-blind control corpus: a temporally-spread random PubMed sample, with
NO field selection. Used to test whether the forecasting signal survives when
explosive regions are rare rather than hand-picked. Sampling is by date only
(never by novelty/topic), so it cannot bake in the result.
"""
from __future__ import annotations

import json
import time

from Bio import Entrez

from embedd import config as C
from embedd import fetch

Entrez.email = C.ENTREZ_EMAIL
BROAD = '(hasabstract[text] AND English[lang] AND "journal article"[pt])'
PER_QUARTER = 350  # ~1400/year, spread across 4 quarters to avoid early-year bias
QUARTERS = [("01", "03"), ("04", "06"), ("07", "09"), ("10", "12")]


def esearch_window(year, mmin, mmax, retmax):
    term = f'{BROAD} AND ("{year}/{mmin}/01"[pdat] : "{year}/{mmax}/28"[pdat])'
    h = fetch._retry(Entrez.esearch, db="pubmed", term=term, retmax=retmax,
                     sort="pub_date")
    rec = Entrez.read(h); h.close(); time.sleep(0.34)
    return list(rec["IdList"])


def main():
    out = C.RAW / "random.jsonl"
    seen = set()
    n = 0
    t0 = time.time()
    with out.open("w") as fh:
        for year in range(C.YEAR_MIN, C.YEAR_MAX + 1):
            got = 0
            for mmin, mmax in QUARTERS:
                pmids = [p for p in esearch_window(year, mmin, mmax, PER_QUARTER)
                         if p not in seen]
                for rec in fetch.efetch_records(pmids):
                    if rec["pmid"] in seen:
                        continue
                    seen.add(rec["pmid"])
                    rec["field"] = "random"
                    rec["is_founder"] = False
                    fh.write(json.dumps(rec) + "\n")
                    n += 1; got += 1
            print(f"  [random] {year}: +{got:4d}  total={n:6d}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"DONE random: {n} records in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
