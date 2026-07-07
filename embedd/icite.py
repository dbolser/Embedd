"""Fetch open citation stats from NIH iCite as a *comparison baseline only*.

These counts never enter the novelty metric — they exist so we can show that the
citation-free pioneer score agrees with citations where it should, yet flags
early foundational work that citation counts (biased toward recent, well-marketed
papers, and gameable by self-citation) rank differently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from . import config as C

ICITE_URL = "https://icite.od.nih.gov/api/pubs"


def fetch_icite(pmids: list[str], batch: int = 200) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(pmids), batch):
        chunk = pmids[i : i + batch]
        for attempt in range(4):
            try:
                r = requests.get(
                    ICITE_URL, params={"pmids": ",".join(chunk), "format": "json"},
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json().get("data", [])
                for d in data:
                    out[str(d["pmid"])] = {
                        "citation_count": d.get("citation_count"),
                        "rcr": d.get("relative_citation_ratio"),
                        "cites_per_year": d.get("citations_per_year"),
                        "nih_percentile": d.get("nih_percentile"),
                    }
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"  icite batch {i} failed: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
        print(f"  icite {min(i+batch, len(pmids))}/{len(pmids)}", flush=True)
        time.sleep(0.3)
    return out


def build(tag: str = "local") -> dict[str, dict]:
    meta = [json.loads(l) for l in (C.EMBEDDINGS / f"meta_{tag}.jsonl").read_text().splitlines()]
    pmids = [m["pmid"] for m in meta]
    stats = fetch_icite(pmids)
    out = C.EMBEDDINGS / "icite.json"
    Path(out).write_text(json.dumps(stats))
    print(f"wrote {len(stats)} icite records -> {out}")
    return stats
