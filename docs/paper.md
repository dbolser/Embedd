# Temporal precedence in embedding space: a citation-free signal for foundational science

*Working draft. The outcome-blind control result is marked `[control pending]`
until the random-corpus forecast completes.*

## Abstract

Citation-based impact metrics are lagged, field-biased, and gameable by
self-citation and citation cartels. We ask whether the *content* of a paper —
its position in the embedding space of the literature together with its
publication date — is enough to identify foundational work, and to do so
*early*. We embed ~21k PubMed abstracts (1995–2023) spanning six biomedical
revolutions with famous, datable founding papers (CRISPR, RNAi, iPSC,
optogenetics, GWAS, single-cell RNA-seq), plus an outcome-blind random control
corpus. We find that the intuitive signal — *isolation at birth* — does **not**
work: founding papers are not isolated in embedding space; they sit among
contemporaries. What separates them is **temporal precedence**: a founder's
semantic neighborhood arrives mostly *after* it. A precedence score
(future-facing neighborhood mass, weighted by region size) recovers the 11
ground-truth founders at a mean within-field percentile of **98.1** (min 92.7),
placing several exactly first in their field — with no citation data. Two
methodological results are essential to this working at all: (i) sentence
embeddings are strongly anisotropic and must be **mean-centered** before cosine
is discriminative, and (ii) within a keyword-selected field precedence is
confounded with "published early", so the load-bearing evidence is a **forecast
on an outcome-blind corpus**, where standing at year D we predict which
still-nascent papers explode after D at AUC ~0.65 `[control pending]`.

## 1. Introduction

The "reckoning" motivation: separate true pioneers from self-citation gamers and
bandwagon riders, using content and time alone. Three archetypes any such metric
must distinguish — pioneers (early in a region that later fills in), dead-end
outliers (early but never followed), and bandwagon work (arrives into an
existing crowd). Contributions: (i) the precedence score; (ii) two negative/
methodological findings that overturn the obvious approach (isolation fails;
anisotropy must be removed); (iii) an early-forecasting evaluation on an
outcome-blind corpus that tests prediction, not post-hoc confirmation.

## 2. Related work

- Disruption / CD index (Funk & Owens; Wu, Wang & Evans 2019): citation-network
  based; we are citation-free and content-based.
- Atypical combination novelty (Uzzi et al. 2013): journal-pair z-scores; we use
  continuous embedding geometry and time.
- Sleeping beauties (Ke et al.): delayed recognition; our precedence signal is a
  content analog of "the field arrives later".
- Anisotropy of contextual embeddings (Ethayarajh 2019; Mu & Viswanath 2018):
  motivates mean-centering / all-but-the-top-k; we confirm centering is
  necessary for any threshold-based neighborhood to be meaningful.

## 3. Data

Six fields fetched across 1995–2025, capped per field per year to preserve the
temporal fill-in signal, plus (a) a broad background sample and (b) an
outcome-blind random control corpus of 38k abstracts sampled by publication date
only. After dropping non-research editorial content (errata, corrections, news;
~26% of the naive background) and boundary-year papers, the field corpus is
15,873 research abstracts. Ground-truth founders are listed in `config.FIELDS`.
Embeddings: PubMedBERT sentence-transformer; OpenAI cross-check planned.

## 4. Method

Each abstract *i* has embedding *e_i* and year *t_i*. We **mean-center** the
embeddings (subtract the global mean, renormalize) — without this, cosine is
non-discriminative (§5.1). We pick a similarity threshold τ to target a median
of ~25 neighbors. For each *i*:

- region_size = # abstracts within τ (any time);
- precedence = (# within τ published *after* t_i) / (region_size + 1);
- **pioneer score** = precedence × normalized log(region_size).

Rejected alternatives (§5.2): *isolation at birth* (1 − max prior similarity) and
*winner-take-all vanguard* (credit to a follower's single earliest content
ancestor) both fail — the former because founders are not isolated, the latter
because it rewards the oldest tangential paper.

## 5. Results

### 5.1 Embeddings must be mean-centered
Raw PubMedBERT: same-field vs cross-field cosine 0.932 / 0.898 (separation
d=1.97); at any single threshold the neighborhood is either empty or the whole
corpus. Mean-centered: 0.258 / −0.067 (d=2.36); cross-field pairs become
near-orthogonal. All subsequent results use centered space.

### 5.2 Isolation fails; precedence works
Within-field founder percentile (tie-aware): isolation 46.2, winner-take-all
vanguard is an artifact, whereas precedence 98.0 and the full pioneer score
**98.1** (min 92.7). Corpus-wide, the top-ranked pioneers are the actual
Yamanaka iPSC papers (ranks 1–2) and early CRISPR/RNAi founders; Elbashir siRNA
and Yamanaka human-iPSC rank #1 in their fields.

### 5.3 The confound, stated plainly
Within a field we *selected because it exploded*, "precedence" ≈ "published
early", so 5.2 is a sanity check, not proof. Hence 5.4.

### 5.4 Early forecasting on outcome-blind data
Standing at year D with only ≤D information: as-of followership is predictable at
AUC ~0.91, but a size-only baseline already reaches ~0.9 (momentum). The
non-trivial test — among papers still *nascent* at D, which explode after D —
gives AUC ~0.65 on the field corpus. On the random control corpus: `[control
pending]`. If the nascent signal survives there, it is not a selection artifact.

### 5.5 Concept relations (exploratory)
Clustering (UMAP+HDBSCAN, centered space) is interpretable; GWAS spontaneously
fragments into disease-specific sub-clusters sharing a genome-wide-association
signature. Relation-vector clustering on six deep verticals mostly recovers
field identity; surfacing reusable transformations ("genome-wide-ization",
"single-cell-ization") needs corpus breadth. Deferred to future work.

## 6. Discussion & limitations

Embedding-model dependence; year granularity; the corpus is revolution-enriched;
precedence rewards being early in *any* growing region, so it measures leadership
of a trend, not correctness. The honest claim: content geometry + time gives a
citation-free, immediately-available signal that agrees with expert-known
founders and carries real (if modest) early-warning power.

## 7. Conclusion
Foundational papers are legible in embedding space — not as isolated points, but
as early occupants of regions that fill in behind them.
