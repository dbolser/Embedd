# Pioneers without citations: a temporal-geometric novelty metric for the scientific literature

*Working draft. Numbers marked `[TBD]` are filled from `results/` after the analysis run.*

## Abstract

Citation-based impact metrics are lagged, field-biased, and gameable by
self-citation and citation cartels. We ask whether the *content* of a paper —
its position in the embedding space of the literature, together with its
publication date — is enough to identify foundational work as it happens. We
embed ~21k PubMed abstracts (1995–2025) spanning six biomedical revolutions with
famous, datable founding papers, plus a broad background sample. We define a
**pioneer score** from two purely geometric-temporal signals: *isolation at
birth* (how empty a paper's neighborhood was in its own year and before) and
*vanguard* (how many later papers name it as their earliest content ancestor
within a similarity radius). The score requires no citation data. It recovers
known founders at a mean within-field percentile of `[TBD]`, beating naive
density and novelty baselines, and correlates with — but is not reducible to —
citation counts. [TBD one-line headline result.]

## 1. Introduction

- The reckoning motivation: separate true pioneers from self-citation gamers and
  bandwagon riders. Citation-free by construction.
- Three archetypes a good metric must distinguish: **pioneers** (isolated at
  birth, later followed), **dead-end outliers** (isolated, never followed),
  **bandwagon** (born into a crowd, followed but not early).
- Contribution: (i) the pioneer score; (ii) a ground-truth validation design
  using known biomedical revolutions; (iii) evidence it is not a citation proxy.

## 2. Related work

- Disruption / CD index (Funk & Owens; Wu, Wang & Evans 2019) — citation-network
  based; we are citation-free and content-based.
- Novelty as atypical combination (Uzzi et al. 2013) — journal-pair z-scores;
  we use continuous embedding geometry instead of discrete co-occurrence.
- Embedding-based novelty and SPECTER/SciBERT representations.
- Sleeping beauties (Ke et al.) — delayed recognition; our vanguard term is a
  content analog of "being awoken by followers".

## 3. Data

- Corpus: six fields (CRISPR, RNAi, iPSC, optogenetics, GWAS, single-cell
  RNA-seq) fetched across 1995–2025, capped per field per year to preserve the
  temporal fill-in signal, plus a background PubMed sample. N = `[TBD]` unique
  abstracts. Ground-truth founder PMIDs listed in `config.FIELDS`.
- Embeddings: PubMedBERT sentence-transformer (primary, citation-free);
  OpenAI text-embedding-3-large (robustness cross-check).

## 4. Method

Let each abstract *i* have unit embedding *e_i* and year *t_i*. For a similarity
threshold τ (calibrated so same-field pairs exceed it and cross-field pairs do
not):

- **Isolation at birth** `I_i = 1 − max_j { e_i·e_j : t_j ≤ t_i }` — a paper with
  a close same-year or earlier twin is not isolated.
- **Content ancestry**: every paper *f* with prior similar work assigns credit
  `e_f·e_a` to its earliest such ancestor *a* (ties broken by similarity).
- **Vanguard** `V_i = Σ credit received from followers`.
- **Pioneer score** `P_i = norm(log(1+V_i)) · I_i²` — high only when isolated at
  birth *and* heavily followed.

τ calibration, complexity (block-wise O(N²)), and robustness to τ in §6.

## 5. Results

- 5.1 Founder recovery: mean/min within-field percentile under P vs baselines. `[TBD]`
- 5.2 The archetype map (isolation × vanguard); founders in the pioneer quadrant. `[TBD]`
- 5.3 Temporal fill-in per field; pioneer scores vs publication year. `[TBD]`
- 5.4 Pioneer score vs citations: correlation, and named divergences
  (under-cited pioneers; highly-cited bandwagon). `[TBD]`
- 5.5 Robustness: τ sweep, local vs OpenAI embeddings. `[TBD]`

## 6. Discussion & limitations

- Embedding model bias; year granularity; field boundary effects; the "hub"
  problem (review articles as false ancestors); corpus is revolution-enriched,
  not a uniform PubMed sample.
- The reckoning claim, honestly scoped: content-based novelty is complementary
  to, not a replacement for, expert judgement.

## 7. Conclusion
