# Findings — pilot run (overnight)

Short version: **the core idea works, but not the way we first framed it.** The
signal that identifies foundational papers is *temporal precedence* in embedding
space — not *isolation* — and it survives the two tests that could have killed it
(dilution and an outcome-blind forecast). Two methodological gotchas had to be
fixed first, and both are results in their own right.

## What we built
- 21k-abstract field corpus (6 revolutions with 11 datable founder papers) +
  6k background, and a **39k outcome-blind random control corpus** (sampled by
  date only, never by topic/novelty).
- Local PubMedBERT embeddings (citation-free end to end).
- Pipeline: fetch → embed → metric → validate → forecast → concept-relations.

## The idea, corrected
- **Isolation-at-birth fails.** Founders are *not* isolated in embedding space —
  they sit among contemporaries (isolation ranks them ~46th percentile, useless).
- **Temporal precedence works.** Score a paper by how much of its semantic
  neighborhood arrives *after* it, weighted by how big that neighborhood becomes.
  Founder recovery: **98.1 mean / 92.7 min** within-field percentile. Elbashir
  siRNA and Yamanaka human-iPSC rank **#1 in their field**; corpus-wide the top
  two pioneers of all 15,873 are the two Yamanaka iPSC papers.

## Two fixes that were essential (and are findings)
1. **Embeddings are anisotropic.** Raw PubMedBERT cosine is compressed so high
   (even cross-field median 0.90) that any threshold makes the whole corpus one
   neighborhood. **Mean-centering** restores discrimination (same-vs-cross
   separation d 1.97 → 2.36). Without it, "precedence" collapses to publication
   year and the whole thing is meaningless. This bit us hard (top pioneers were
   1995 editorial junk) before we caught it.
2. **Winner-take-all ancestor credit is broken**, and an argsort tie-ordering
   bug made an early version *look* like it worked (94th pct) when it didn't.
   Now using tie-aware ranks everywhere.

## The two tests that matter (answering the circularity worry)
Because we picked fields *because* they exploded, within-field recovery is
confounded ("precedence ≈ published early"). So:

- **Dilution:** bury the 11 founders among 39k random abstracts (combined ~55k).
  They still rank at **mean 99.3 corpus-wide percentile**. Not washed out.
- **Outcome-blind forecast:** standing at year D, among papers still *nascent*
  at D, predict which explode *after* D — on the random corpus (no cherry-picked
  fields): **AUC ~0.70** (0.70/0.76/0.63 at D=2011/13/15), *stronger* than on the
  fields. In the realistic combined mix (founders + 39k random, ~55k): **AUC
  ~0.74** (0.78/0.77/0.67). Low prior-density (0.73–0.80) and isolation
  (0.64–0.70) are the predictors. **This is the real result: citation-free early
  detection of foundational work, on unbiased data.**

## The reckoning, tested against real citations (iCite RCR)
Spearman(precedence, RCR) is only **+0.13 to +0.23** within field — precedence is
**not a citation proxy**. The two divergence quadrants:
- **Bandwagon (high cites, low precedence) — works well.** Dominated by *tools,
  resources, databases*: NHGRI GWAS Catalog (2089 cites), CHOPCHOP CRISPR
  toolbox, MAGeCK software. Massive citation utility, but they arrived *after*
  the conceptual foundation. Precedence cleanly separates conceptual leaders
  from infrastructure.
- **Overlooked (high precedence, low cites) — does NOT work.** Surfaces
  early-but-minor papers (guidelines, non-English reviews, incremental early
  work with 1–3 citations), not wronged giants. Being early in embedding space
  doesn't imply hidden importance; precedence has no quality signal to tell
  "early and important" from "early and forgettable". Honest negative result —
  the "recognise the overlooked pioneer" half of the reckoning needs more than
  this.

## Honest limitations
- As-of forecasting of raw followership is ~0.9 AUC but mostly *momentum* (a
  size-only baseline already ~0.9); the non-trivial lift is the nascent test.
- Precedence rewards being early in *any* growing region — it measures trend
  leadership, not correctness.
- Concept-relation analysis (your "moves of science" idea) needs corpus breadth;
  on 6 deep verticals it recovers field identity, not reusable transformations.
  GWAS auto-fragmenting into disease-specific sub-clusters is a promising hint.

## Figures (`results/figures/`)
- `quadrants_local.png` — the archetype map; all 11 founders top-right.
- `precedence_year_local.png` — founders combine early dates + high precedence.
- `fillin_local.png` — per-field temporal fill-in (founder years dashed).
- `landscape_local.png`, `concept_map_local.png` — the embedding landscape.

## Update — scaling & the "moves of science" test (session 2)

**Scaled to a broad corpus (~272k abstracts, Titan embeddings)** and ran the
concept-relation analysis properly — including the fix of UMAP-reducing the
relation vectors before HDBSCAN (they were being clustered in full 768/1024-dim,
the curse-of-dimensionality trap).

- The fix **worked as a method improvement**: relation-group coherence rose from
  ~0.35 to **~0.65–0.71**. The difference vectors do cluster cleanly now.
- But the coherent groups are **topic geography, not reusable operations** — they
  are within/between the big domains (one domain's sub-clusters pointing at
  another's), not one transformation shared across *unrelated* domains.
- Two causes: (a) the endpoint-diversity guard is **fooled by domain size** (a big
  domain fragments into many sub-clusters, trivially clearing "≥4 distinct
  src/dst"); (b) the **corpus shape is wrong** — two huge domains + 200k random
  spread thin (52% noise). Analogies need *many matched medium domains*.
- **Verdict:** reusable "moves of science" do **not** fall out of unsupervised
  difference-clustering, even at scale with the right reduction. Next tool is the
  *seeded* probe (`concepts.analogy_map`) over a corpus of matched domain pairs.

**Now building a canonical dataset** (`docs/TODO.md`): 294,053 unique abstracts,
embedded row-aligned with PubMedBERT + Titan + OpenAI, so we can test whether the
precedence/forecast signal is robust across embedding models.

## Suggested next steps
See `docs/TODO.md` for the full roadmap. Headline: cross-model robustness of the
precedence signal on the canonical dataset (the paper's spine); seeded analogy
probes for the moves idea; beat the momentum baseline; patents; ANN for >1M.
