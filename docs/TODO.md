# Embedd — session handoff & TODO

Picking up in a new session. This is the current state and the roadmap.

## Where we are (one paragraph)

We have a **citation-free novelty signal** — *temporal precedence* (a paper's
semantic neighbourhood arrives after it) — that recovers known founding papers,
survives dilution, and forecasts which nascent papers will be followed, on
outcome-blind data. Two methodological findings were essential: embeddings must
be **mean-centered** (anisotropy), and ranking must be **tie-aware**. We then
tested the **"moves of science"** idea (do concept-cluster differences cluster
into reusable transformations?) at scale and got an honest **negative**:
unsupervised difference-clustering recovers *topic geography*, not reusable
operations. We are now standardising the data.

## The canonical dataset (the current task)

One corpus, three embedding spaces, aligned row-for-row.

- **Corpus:** `data/canonical/corpus.jsonl` — **294,053** unique abstracts
  (de-duplicated union of the pilot fields, the two random pulls, and the two
  focused topic domains defined in the untracked `topics.local.json`). Built by
  `scripts/20_build_canonical.py`. Stable order = sort by (year, pmid).
- **Embeddings** (all via `scripts/21_embed_canonical.py <backend>`, checkpointed
  & resumable to `data/embeddings/canon_<backend>_ckpt/`, final
  `data/embeddings/vecs_canonical_<backend>.npy` + shared `meta_canonical.jsonl`):

  | backend  | model                     | dim  | status when handed off |
  |----------|---------------------------|------|------------------------|
  | pubbert  | S-PubMedBert-MS-MARCO     | 768  | running (~6 h, CPU)    |
  | titan    | amazon.titan-embed-text-v2| 1024 | running (~30 min)      |
  | openai   | text-embedding-3-small    | 1536 | running (~18 min)      |

  **First thing next session:** check they finished —
  `ls -la data/embeddings/vecs_canonical_*.npy` and tail
  `results/embed_canon_*.log`. Any that died: just rerun
  `python scripts/21_embed_canonical.py <backend>` (resumes from checkpoint).
  Note ~0.1–0.2% of rows may be zero (failed API calls) — filter by row norm,
  as `scripts/14_moves.py: load_centered` does.
  Cost so far: Titan + OpenAI ≈ $3 total.

## Next experiments (roughly in priority order)

1. **Cross-model robustness.** Re-run founder recovery (`03_analyze`) and the
   outcome-blind forecast (`10_forecast_compare`, `predict.py`) on the canonical
   corpus under each of the three embeddings. Does temporal precedence hold up
   across models? This is the cleanest paper-strengthening result and the whole
   point of building the aligned dataset. (Adapt the loaders to read
   `vecs_canonical_<backend>` + `meta_canonical`, mean-center, `choose_tau`.)
2. **Model agreement.** How much do the three spaces agree on the precedence
   ranking / neighbour graphs? Rank correlation of pioneer scores across models;
   nearest-neighbour overlap. Disagreement localises where the signal is model
   artifact vs real.
3. **Moves of science, done right.** The unsupervised approach failed for known
   reasons (topic geography dominates; the endpoint-diversity guard is fooled by
   big domains fragmenting into sub-clusters; corpus was 2 giants + thin noise).
   Next: (a) use the **seeded** probe `concepts.analogy_map` — name a move from
   one known pair, test whether it generalises; (b) tighten the diversity guard
   to require sources/destinations from *different super-domains*; (c) if pursuing
   seriously, build a corpus of **matched domain pairs** (a field and its "before"
   form) rather than deep verticals in random noise.
4. **Beat the momentum baseline.** The forecast is mostly "big-now → big-later".
   Predict who *outgrows* their trajectory (residual target in `predict.py`) —
   the genuine early-detection claim. Currently weak (~0.5–0.6); see if a better
   embedding or feature set helps.
5. **analogy_map dataset.** Persist the analogy probes/results as a dataset to
   fold into the coherent data collection (user's plan).
6. **Patent extension.** Apply precedence to patent prior-art (user has repos) —
   cleaner "before" semantics + economic ground truth.
7. **Scale.** >1M abstracts needs ANN (faiss/hnsw) instead of the O(N²)
   block-wise neighbour computation in `metric.py` / `predict.py`.

## Open pitfalls we've mapped (so we don't re-fall)

- Sentence embeddings are anisotropic → **always mean-center** before cosine.
- **Tie-ordering** in argsort silently fakes good results → tie-aware ranks.
- Corpus **selection confounds** within-field recovery → outcome-blind control.
- **Window boundary** (earliest years) look artificially isolated → censor.
- Editorial **junk** (errata/news) forms false hub clusters → filter.
- HDBSCAN on **full-dim** relation vectors collapses → UMAP-reduce first.
- Difference-clustering finds **topic geography**, not analogies.
- OpenAI embeddings: **300k tokens/request** cap → batch ≤ 500 abstracts.

## Housekeeping

- Work on branch `analysis-pilot` (PR #1). `main` has the public Pages summary
  (https://dbolser.github.io/Embedd/) and README link.
- Focused-domain names are kept out of the repo (`topics.local.json` is
  gitignored); broad/canonical result artifacts are gitignored (they contain
  domain terms). Deferred by user choice: the `analysis-pilot` *history* still
  holds one large data blob + one commit message naming the domains — full purge
  needs a force-push if ever wanted.
- Data (`data/**`), embeddings, and `.env` are all gitignored.
