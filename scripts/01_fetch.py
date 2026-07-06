"""Fetch the full pilot corpus: all revolution fields + a background sample."""
from __future__ import annotations

import json
import time
from pathlib import Path

from embedd import config as C
from embedd import fetch


def fetch_background(out_path: Path) -> int:
    """Broad slice of PubMed across the window to represent 'the rest of space'."""
    per_year = max(1, C.BACKGROUND_SIZE // (C.YEAR_MAX - C.YEAR_MIN + 1))
    seen: set[str] = set()
    n = 0
    with out_path.open("w") as fh:
        for year in range(C.YEAR_MIN, C.YEAR_MAX + 1):
            pmids = fetch.esearch_pmids(C.BACKGROUND_QUERY, year, per_year)
            pmids = [p for p in pmids if p not in seen]
            for rec in fetch.efetch_records(pmids):
                if rec["pmid"] in seen:
                    continue
                seen.add(rec["pmid"])
                rec["field"] = "background"
                rec["is_founder"] = False
                fh.write(json.dumps(rec) + "\n")
                n += 1
            print(f"  [background] {year}: {n:5d} total", flush=True)
    return n


def main() -> None:
    t0 = time.time()
    total = 0
    for name, spec in C.FIELDS.items():
        spec = {**spec, "_name": name}
        out = C.RAW / f"{name}.jsonl"
        print(f"=== fetching field: {name} ({spec['label']}) ===", flush=True)
        n = fetch.fetch_field(name, spec, out)
        fetch.ensure_founders_present(spec, out)
        print(f"    -> {n} records ({time.time()-t0:.0f}s elapsed)", flush=True)
        total += n

    print("=== fetching background ===", flush=True)
    nb = fetch_background(C.RAW / "background.jsonl")
    total += nb
    print(f"\nDONE: {total} records in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
