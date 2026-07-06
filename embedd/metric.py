"""Content-based, citation-free novelty metrics in embedding space.

Core idea: give every later abstract a way to name its *content ancestor* — the
earliest prior abstract that already occupies its region of meaning. A pioneer
is an abstract that (a) had little or no prior work near it when published
(isolated at birth) yet (b) is later named as the content ancestor of many
followers (high vanguard). This separates pioneers from dead-end outliers
(isolated but never followed) and from bandwagon work (followed but not early).

Everything here uses only embeddings + publication years. No citations, so the
metric cannot be inflated by self-citation or citation cartels.
"""
from __future__ import annotations

import numpy as np


def _sim_block(E: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Cosine similarities of rows [lo:hi] against all rows (vectors are unit-norm)."""
    return E[lo:hi] @ E.T


def _clear_self(mask: np.ndarray, lo: int) -> None:
    """Zero the self-match (row bi corresponds to global column lo+bi)."""
    b = mask.shape[0]
    idx = np.arange(b)
    cols = idx + lo
    valid = cols < mask.shape[1]
    mask[idx[valid], cols[valid]] = False


def compute_metrics(
    E: np.ndarray,
    years: np.ndarray,
    tau: float = 0.7,
    block: int = 512,
) -> dict[str, np.ndarray]:
    """Compute per-abstract novelty metrics.

    Parameters
    ----------
    E : (N, d) L2-normalized embeddings.
    years : (N,) integer publication years.
    tau : cosine-similarity threshold defining "same region of meaning".

    Returns a dict of (N,) arrays:
      prior_count      : # abstracts within tau published strictly before
      birth_count      : # abstracts within tau published the same year or earlier
                         (density at birth; year is our finest temporal resolution)
      future_count     : # abstracts within tau published strictly after
      isolation        : 1 - max cosine sim to any same-year-or-earlier abstract
                         (novelty at birth; a paper with same-year twins is not isolated)
      nearest_prior_yr : year of the most-similar strictly-prior abstract (NaN if none)
      vanguard         : credit mass from followers that name this abstract their
                         earliest content ancestor (followership, attributed)
      n_descendants    : # followers that name this abstract their earliest ancestor
      is_seed          : True if this abstract had no prior work within tau
      pioneer          : combined score (see pioneer_score)
    """
    N = E.shape[0]
    years = np.asarray(years)

    prior_count = np.zeros(N, dtype=np.int32)
    birth_count = np.zeros(N, dtype=np.int32)
    future_count = np.zeros(N, dtype=np.int32)
    isolation = np.ones(N, dtype=np.float32)
    nearest_prior_yr = np.full(N, np.nan, dtype=np.float32)
    vanguard = np.zeros(N, dtype=np.float64)
    n_descendants = np.zeros(N, dtype=np.int32)

    for lo in range(0, N, block):
        hi = min(lo + block, N)
        S = _sim_block(E, lo, hi)  # (b, N)
        yb = years[lo:hi][:, None]  # (b,1)

        within = S >= tau
        _clear_self(within, lo)  # exclude self
        prior_mask = within & (years[None, :] < yb)     # strictly earlier
        birth_mask = within & (years[None, :] <= yb)    # same year or earlier
        future_mask = within & (years[None, :] > yb)

        prior_count[lo:hi] = prior_mask.sum(1)
        birth_count[lo:hi] = birth_mask.sum(1)
        future_count[lo:hi] = future_mask.sum(1)

        # isolation: 1 - best similarity to any same-year-or-earlier abstract.
        # Not thresholded by tau (we want the true nearest), but self-excluded.
        le_mask = years[None, :] <= yb
        _clear_self(le_mask, lo)
        S_birth = np.where(le_mask, S, -1.0)
        best_birth = S_birth.max(1)
        has_birth = best_birth > -1.0
        isolation[lo:hi] = np.where(has_birth, 1.0 - np.clip(best_birth, -1, 1), 1.0)
        # year of nearest strictly-prior abstract (for lineage reporting)
        S_prior = np.where(years[None, :] < yb, S, -1.0)
        best_prior = S_prior.max(1)
        has_prior = best_prior > -1.0
        arg = S_prior.argmax(1)
        nearest_prior_yr[lo:hi] = np.where(has_prior, years[arg], np.nan)

        # --- credit assignment: each f in this block names its earliest ancestor
        for bi in range(hi - lo):
            f = lo + bi
            anc = np.where(prior_mask[bi])[0]  # prior similar abstracts (ancestors)
            if anc.size == 0:
                continue
            anc_years = years[anc]
            earliest = anc_years.min()
            cand = anc[anc_years == earliest]
            # among the earliest, credit the single most similar one
            best = cand[np.argmax(S[bi, cand])]
            vanguard[best] += float(S[bi, best])
            n_descendants[best] += 1

    is_seed = prior_count == 0
    out = {
        "prior_count": prior_count,
        "birth_count": birth_count,
        "future_count": future_count,
        "isolation": isolation,
        "nearest_prior_yr": nearest_prior_yr,
        "vanguard": vanguard.astype(np.float32),
        "n_descendants": n_descendants,
        "is_seed": is_seed,
    }
    out["pioneer"] = pioneer_score(out)
    return out


def pioneer_score(m: dict[str, np.ndarray]) -> np.ndarray:
    """Combine birth-isolation and attributed followership into one score.

    A pioneer is isolated at birth AND heavily followed. We multiply a
    saturating followership term by an isolation gate so that neither pure
    outliers (no followers) nor bandwagon work (no isolation) score highly.
    """
    isolation = m["isolation"].astype(np.float64)
    vanguard = m["vanguard"].astype(np.float64)
    # saturating followership (log) keeps a few mega-hubs from dominating
    follow = np.log1p(vanguard)
    follow = follow / (follow.max() + 1e-12)
    # isolation gate in [0,1]; emphasize genuinely isolated births
    gate = isolation ** 2
    return (follow * gate).astype(np.float32)
