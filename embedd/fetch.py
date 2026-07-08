"""Fetch abstracts from PubMed via NCBI Entrez.

We fetch per field, year by year, so that:
  * sparse early years (near a field's founding) are captured exhaustively, and
  * dense later years are capped, preserving the temporal "fill-in" signal
    without letting a single hot field dominate the corpus.

Each record is normalized to a flat dict and written as JSONL.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from Bio import Entrez

from . import config as C

Entrez.email = C.ENTREZ_EMAIL
if C.ENTREZ_API_KEY:
    Entrez.api_key = C.ENTREZ_API_KEY

_SLEEP = 0.12 if C.ENTREZ_API_KEY else 0.34  # respect 10/s (key) or 3/s (no key)


def _retry(fn, *args, tries=4, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - Entrez raises many transient errors
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def esearch_pmids(query: str, year: int, retmax: int) -> list[str]:
    """Return up to `retmax` PMIDs for `query` restricted to publication `year`."""
    term = f"({query}) AND {year}[pdat]"
    handle = _retry(
        Entrez.esearch, db="pubmed", term=term, retmax=retmax, sort="relevance"
    )
    rec = Entrez.read(handle)
    handle.close()
    time.sleep(_SLEEP)
    return list(rec["IdList"])


def _text(elem) -> str:
    """Flatten an Entrez element (possibly with markup children) to a string."""
    if elem is None:
        return ""
    if isinstance(elem, str):
        return str(elem)
    # StringElement / list-ish
    try:
        return "".join(_text(x) for x in elem)
    except TypeError:
        return str(elem)


def _parse_article(art) -> dict | None:
    try:
        medline = art["MedlineCitation"]
        pmid = str(medline["PMID"])
        article = medline["Article"]
        title = _text(article.get("ArticleTitle", ""))

        # abstract: may be several labeled sections
        abstract_parts = []
        abst = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abst, (str,)):
            abstract_parts.append(_text(abst))
        else:
            for seg in abst:
                label = seg.attributes.get("Label") if hasattr(seg, "attributes") else None
                txt = _text(seg)
                abstract_parts.append(f"{label}: {txt}" if label else txt)
        abstract = " ".join(p for p in abstract_parts if p).strip()
        if not abstract:
            return None

        # year: prefer article date, fall back to journal issue / medline date
        year = None
        for path in (
            article.get("ArticleDate", [{}]),
            [article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})],
        ):
            if not path:
                continue
            entry = path[0] if isinstance(path, list) else path
            y = entry.get("Year")
            if y and str(y).isdigit():
                year = int(y)
                break
        if year is None:
            return None

        journal = _text(article.get("Journal", {}).get("Title", ""))

        # first author
        authors = article.get("AuthorList", [])
        first_author = ""
        if authors:
            a0 = authors[0]
            first_author = f"{_text(a0.get('LastName',''))} {_text(a0.get('Initials',''))}".strip()

        # MeSH terms
        mesh = []
        for mh in medline.get("MeshHeadingList", []):
            mesh.append(_text(mh.get("DescriptorName", "")))

        return {
            "pmid": pmid,
            "year": year,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "first_author": first_author,
            "mesh": mesh,
        }
    except Exception:  # noqa: BLE001 - skip malformed records
        return None


def _efetch_read(chunk: list[str], tries: int = 5):
    """efetch + read as one retried unit. Network hiccups (IncompleteRead) can
    fire during read, not just the request, so both must be inside the retry.
    Returns None if the batch keeps failing (caller skips it rather than crash)."""
    last = None
    for i in range(tries):
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(chunk),
                                   rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records
        except Exception as e:  # noqa: BLE001 - IncompleteRead, HTTP, parse errors
            last = e
            time.sleep(2.0 * (i + 1))
    print(f"  [efetch] skipping batch of {len(chunk)} after {tries} tries: "
          f"{type(last).__name__}", flush=True)
    return None


def efetch_records(pmids: list[str], batch: int = 200):
    """Yield normalized records for a list of PMIDs. Never raises on network
    errors; a persistently failing batch is skipped."""
    for i in range(0, len(pmids), batch):
        chunk = pmids[i : i + batch]
        records = _efetch_read(chunk)
        if records is None:
            continue
        for art in records.get("PubmedArticle", []):
            rec = _parse_article(art)
            if rec is not None:
                yield rec
        time.sleep(_SLEEP)


def fetch_field(name: str, spec: dict, out_path: Path) -> int:
    """Fetch one field across all years; write JSONL. Returns record count."""
    seen: set[str] = set()
    n = 0
    with out_path.open("w") as fh:
        for year in range(C.YEAR_MIN, C.YEAR_MAX + 1):
            pmids = esearch_pmids(spec["query"], year, C.PER_YEAR_CAP)
            pmids = [p for p in pmids if p not in seen]
            for rec in efetch_records(pmids):
                if rec["pmid"] in seen:
                    continue
                seen.add(rec["pmid"])
                rec["field"] = name
                rec["is_founder"] = rec["pmid"] in spec.get("founders", {})
                fh.write(json.dumps(rec) + "\n")
                n += 1
            print(f"  [{name}] {year}: {len(pmids):4d} pmids, {n:5d} total", flush=True)
            if n >= C.PER_FIELD_CAP:
                break
    return n


def ensure_founders_present(spec: dict, out_path: Path) -> None:
    """Guarantee ground-truth founder PMIDs are in the file (they may be too old
    or rank below the relevance cap in their year)."""
    founders = spec.get("founders", {})
    if not founders:
        return
    have = set()
    for line in out_path.read_text().splitlines():
        have.add(json.loads(line)["pmid"])
    missing = [p for p in founders if p not in have]
    if not missing:
        return
    with out_path.open("a") as fh:
        for rec in efetch_records(missing):
            rec["field"] = spec.get("_name", "")
            rec["is_founder"] = True
            fh.write(json.dumps(rec) + "\n")
            print(f"  [+founder] {rec['pmid']} ({rec['year']})", flush=True)
