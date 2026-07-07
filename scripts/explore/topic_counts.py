"""Check PubMed hit counts for candidate PDE / ALS queries so we can size the
broad corpus before fetching. Throwaway."""
from __future__ import annotations

from Bio import Entrez

from embedd import config as C
from embedd import fetch  # noqa: F401 (sets Entrez.email)

Entrez.email = C.ENTREZ_EMAIL

QUERIES = {
    "PDE phosphodiesterase (MeSH+tiab)":
        '"Phosphoric Diester Hydrolases"[MeSH] OR "phosphodiesterase"[tiab] OR "phosphodiesterases"[tiab]',
    "PDE acronym only":
        '"PDE"[tiab]',
    "ALS (MeSH)": '"Amyotrophic Lateral Sclerosis"[MeSH]',
    "ALS (MeSH+tiab+MND)":
        '"Amyotrophic Lateral Sclerosis"[MeSH] OR "amyotrophic lateral sclerosis"[tiab] OR "motor neuron disease"[tiab]',
}


def count(term):
    term = f'({term}) AND hasabstract[text] AND English[lang] AND ("{C.YEAR_MIN}"[pdat] : "{C.YEAR_MAX}"[pdat])'
    h = fetch._retry(Entrez.esearch, db="pubmed", term=term, retmax=0)
    rec = Entrez.read(h); h.close()
    return int(rec["Count"])


if __name__ == "__main__":
    for name, q in QUERIES.items():
        print(f"{count(q):8d}  {name}")
