"""Validation: does the metric recover known pioneers, and beat naive baselines?

Ground truth = the founder PMIDs in config.FIELDS. A good novelty metric should
rank each field's founders near the top *within that field*, and should do so
better than raw prior/future density or citation-free naive novelty.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def calibrate_tau(E: np.ndarray, meta: list[dict], n: int = 4000,
                  seed_offset: int = 0) -> dict:
    """Estimate a similarity threshold separating same-field from cross-field pairs.

    Returns summary percentiles; we pick tau near the point that best separates
    within-field (should be > tau) from background/cross-field (should be < tau).
    """
    fields = np.array([m["field"] for m in meta])
    idx = np.arange(len(meta))
    # deterministic pseudo-sample without Math.random: stride the array
    take = idx[(idx * 2654435761 + seed_offset) % max(1, len(idx) // n + 1) == 0][:n]
    same, cross = [], []
    for a in range(len(take)):
        i = take[a]
        j = take[(a + 1) % len(take)]
        if i == j:
            continue
        s = float(E[i] @ E[j])
        if fields[i] == fields[j] and fields[i] not in ("background", "random"):
            same.append(s)
        else:
            cross.append(s)
    same, cross = np.array(same), np.array(cross)
    return {
        "same_field_median": float(np.median(same)) if same.size else float("nan"),
        "same_field_p25": float(np.percentile(same, 25)) if same.size else float("nan"),
        "cross_field_median": float(np.median(cross)) if cross.size else float("nan"),
        "cross_field_p90": float(np.percentile(cross, 90)) if cross.size else float("nan"),
        "n_same": int(same.size),
        "n_cross": int(cross.size),
    }


def choose_tau(E: np.ndarray, target_median: int = 25, sample: int = 2500) -> float:
    """Pick a cosine threshold so the median abstract has ~target_median
    within-tau neighbors. Robust to the absolute similarity scale (which varies
    with model and with mean-centering), unlike a hard-coded tau."""
    n = min(sample, E.shape[0])
    idx = np.linspace(0, E.shape[0] - 1, n).astype(int)
    S = E[idx] @ E.T
    S[np.arange(n), idx] = -1.0           # exclude self
    lo, hi = 0.0, 0.95
    for _ in range(30):
        mid = (lo + hi) / 2
        med = np.median((S >= mid).sum(1))
        if med > target_median:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def founder_percentiles(scores: np.ndarray, meta: list[dict]) -> list[dict]:
    """For each founder, its tie-aware percentile within its own field (100=top).

    Uses average ranks so that a block of tied scores (e.g. many papers at
    vanguard=0) cannot spuriously push a founder up or down -- an earlier
    argsort-based version was fooled exactly this way.
    """
    from scipy.stats import rankdata

    pmid_to_i = {m["pmid"]: i for i, m in enumerate(meta)}
    fields = np.array([m["field"] for m in meta])
    rows = []
    for field, spec in C.FIELDS.items():
        sel = np.where(fields == field)[0]
        n = len(sel)
        if n < 2:
            continue
        ranks = rankdata(scores[sel])  # 1..n, higher score -> higher rank
        local = {int(gi): k for k, gi in enumerate(sel)}
        for pmid, desc in spec.get("founders", {}).items():
            i = pmid_to_i.get(pmid)
            if i is None or i not in local:
                rows.append({"field": field, "pmid": pmid, "desc": desc,
                             "found": False})
                continue
            k = local[i]
            pct = 100 * (ranks[k] - 1) / (n - 1)
            rows.append({
                "field": field, "pmid": pmid, "desc": desc, "found": True,
                "rank": int(n - ranks[k] + 1), "n_field": n,
                "percentile": round(float(pct), 2),
            })
    return rows


def metric_comparison(metrics: dict, meta: list[dict]) -> dict:
    """Mean founder percentile under each candidate metric (higher = better)."""
    candidates = {
        "pioneer": metrics["pioneer"],                       # precedence x log size
        "precedence": metrics["precedence"].astype(float),
        "future_count": metrics["future_count"].astype(float),
        "neg_prior_count": -metrics["prior_count"].astype(float),
        "isolation": metrics["isolation"].astype(float),     # fails: founders not isolated
        "vanguard": metrics["vanguard"].astype(float),       # fails: winner-take-all artifact
    }
    summary = {}
    for name, sc in candidates.items():
        rows = [r for r in founder_percentiles(sc, meta) if r.get("found")]
        pcts = [r["percentile"] for r in rows]
        summary[name] = {
            "mean_percentile": round(float(np.mean(pcts)), 2) if pcts else None,
            "min_percentile": round(float(np.min(pcts)), 2) if pcts else None,
            "n_founders": len(pcts),
        }
    return summary
