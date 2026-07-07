# Embedd — a citation-free novelty metric for the scientific literature

> 📄 **[Read the plain-language summary →](https://dbolser.github.io/Embedd/)**

**Question.** When a new paper lands in the embedding space of all prior work,
does it fall inside an existing region of ideas, or does it stake out empty
ground that later work rushes to fill? The second kind of paper *led the way*.
Can we score that — without ever touching citations, which can be gamed?

## The idea

Embed every abstract (title + abstract) into a shared vector space and keep its
publication year. Then, using only geometry and time:

- **Isolation at birth** — how empty was this abstract's neighborhood when it was
  published? (novelty)
- **Vanguard** — how many *later* abstracts name this one as their earliest
  content ancestor within a similarity radius τ? (attributed followership)
- **Pioneer score** — isolated at birth **and** heavily followed. This separates
  true pioneers from dead-end outliers (isolated, never followed) and from
  bandwagon work (followed, but never early).

Because it uses no citation data, the metric can't be inflated by self-citation
or citation cartels — the "reckoning" the project is named for.

## Validation

The corpus is built around biomedical revolutions with famous, datable founding
papers (CRISPR, RNAi, iPSC, optogenetics, GWAS, single-cell RNA-seq) plus a
broad PubMed background sample. A good metric should rank each field's founders
near the top *within that field*, and beat naive baselines (raw density, plain
novelty). See `embedd/config.py` for the ground-truth founder PMIDs.

## Pipeline

```
python scripts/01_fetch.py      # PubMed -> data/raw/*.jsonl
python scripts/02_embed.py local  # -> data/embeddings/{vecs,meta}_local.*
python scripts/03_analyze.py local  # metrics + founder validation -> results/
```

Embeddings default to a local PubMedBERT sentence-transformer (free,
reproducible); `02_embed.py openai` cross-checks with text-embedding-3-large.

## Layout

- `embedd/config.py` — paths, corpus fields + ground-truth founders, sizing
- `embedd/fetch.py` — NCBI Entrez fetch + XML normalization
- `embedd/embed.py` — local / OpenAI embedding backends
- `embedd/metric.py` — isolation, vanguard, pioneer score (block-wise, O(N²))
- `embedd/validate.py` — τ calibration + founder-recovery evaluation
- `scripts/` — the three pipeline stages
- `results/` — metrics tables, figures, summaries
