"""Check PubMed hit counts for the focused topic queries (from the untracked
topics.local.json) so we can size the broad corpus before fetching. Throwaway."""
from __future__ import annotations

import json

from Bio import Entrez

from embedd import config as C
from embedd import fetch  # noqa: F401 (sets Entrez.email)

Entrez.email = C.ENTREZ_EMAIL

_topics_file = C.ROOT / "topics.local.json"
QUERIES = json.loads(_topics_file.read_text()) if _topics_file.exists() else {}


def count(term):
    term = f'({term}) AND hasabstract[text] AND English[lang] AND ("{C.YEAR_MIN}"[pdat] : "{C.YEAR_MAX}"[pdat])'
    h = fetch._retry(Entrez.esearch, db="pubmed", term=term, retmax=0)
    rec = Entrez.read(h); h.close()
    return int(rec["Count"])


if __name__ == "__main__":
    if not QUERIES:
        print("No topics.local.json found — define {\"name\": \"<pubmed query>\"} there.")
    for name, q in QUERIES.items():
        print(f"{count(q):8d}  {name}")
