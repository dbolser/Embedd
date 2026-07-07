"""Central configuration: paths, corpus definition, and ground-truth pioneers.

The corpus is built around a handful of biomedical "revolutions" that each have
famous, datable founding papers. That gives us ground truth: a good novelty
metric should rank those founding papers near the top of their field.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from .env into the environment (no dependency)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EMBEDDINGS = DATA / "embeddings"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
for _p in (RAW, PROCESSED, EMBEDDINGS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --- NCBI Entrez -------------------------------------------------------------
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "dan.bolser@outsee.co.uk")
ENTREZ_API_KEY = os.environ.get("NCBI_API_KEY")  # optional; raises rate limit to 10/s

# 30-year window (today = 2026). Founding papers of several revolutions fall
# inside this window, letting us watch each region fill in over time.
YEAR_MIN = 1995
YEAR_MAX = 2025  # last complete year

# --- corpus: known-revolution fields ----------------------------------------
# Each field is a PubMed query designed to capture the founding work AND the
# subsequent explosion. `founders` lists ground-truth PMIDs of landmark papers
# that a good pioneer metric should surface.
FIELDS: dict[str, dict] = {
    "crispr": {
        "label": "CRISPR gene editing",
        "query": '("CRISPR"[tiab] OR "Cas9"[tiab] OR "CRISPR-Cas"[tiab])',
        "founders": {
            "22745249": "Jinek 2012 Science - programmable dual-RNA guided DNA endonuclease",
            "23287718": "Cong 2013 Science - multiplex genome engineering CRISPR/Cas",
            "23287722": "Mali 2013 Science - RNA-guided human genome engineering",
        },
    },
    "rnai": {
        "label": "RNA interference",
        "query": '("RNA interference"[tiab] OR "RNAi"[tiab] OR "small interfering RNA"[tiab] OR "siRNA"[tiab])',
        "founders": {
            "9486653": "Fire & Mello 1998 Nature - potent gene silencing by dsRNA in C. elegans",
            "11373684": "Elbashir 2001 Nature - 21nt siRNA duplexes silence in mammalian cells",
        },
    },
    "ipsc": {
        "label": "Induced pluripotent stem cells",
        "query": '("induced pluripotent"[tiab] OR "iPSC"[tiab] OR "iPS cell"[tiab] OR "reprogramming"[tiab] AND "pluripotent"[tiab])',
        "founders": {
            "16904174": "Takahashi & Yamanaka 2006 Cell - iPSC from mouse fibroblasts",
            "18035408": "Takahashi 2007 Cell - iPSC from adult human fibroblasts",
        },
    },
    "optogenetics": {
        "label": "Optogenetics",
        "query": '("optogenetic"[tiab] OR "channelrhodopsin"[tiab] OR "halorhodopsin"[tiab])',
        "founders": {
            "16116447": "Boyden 2005 Nat Neurosci - millisecond optical control of neural activity",
        },
    },
    "gwas": {
        "label": "Genome-wide association studies",
        "query": '("genome-wide association"[tiab] OR "GWAS"[tiab])',
        "founders": {
            "17554300": "WTCCC 2007 Nature - 14,000 cases of 7 diseases, 3,000 controls",
        },
    },
    "scrnaseq": {
        "label": "Single-cell RNA sequencing",
        "query": '("single-cell RNA"[tiab] OR "single cell RNA-seq"[tiab] OR "scRNA-seq"[tiab] OR "single-cell transcriptom*"[tiab])',
        "founders": {
            "19349980": "Tang 2009 Nat Methods - mRNA-seq of a single cell",
            "25700174": "Macosko 2015 Cell - Drop-seq highly parallel single-cell",
        },
    },
}

# Background sample: a broad slice of PubMed to represent "the rest of the
# space" so field regions are embedded among unrelated work, not in a vacuum.
BACKGROUND_QUERY = '("journal article"[pt] AND hasabstract[text] AND English[lang])'

# --- pilot sizing ------------------------------------------------------------
# Per-field cap and background size for the first end-to-end pass. Early (sparse)
# years are fetched exhaustively; dense later years are capped, to preserve the
# temporal "fill-in" signal without exploding the corpus.
PER_FIELD_CAP = 2500
BACKGROUND_SIZE = 6000
PER_YEAR_CAP = 400  # max abstracts per field per year (keeps dense years bounded)

# --- embedding ---------------------------------------------------------------
# Local biomedical/scientific sentence encoder. PubMedBERT-based model tuned for
# semantic search keeps the metric citation-free end to end. OpenAI cross-check
# is added later via the same interface.
LOCAL_MODEL = os.environ.get("EMBEDD_LOCAL_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
OPENAI_MODEL = "text-embedding-3-large"
