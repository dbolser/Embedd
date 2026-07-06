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
        if fields[i] == fields[j] and fields[i] != "background":
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


def rank_within_field(scores: np.ndarray, meta: list[dict], field: str) -> np.ndarray:
    """Indices of `field` abstracts sorted by score descending."""
    idx = np.array([i for i, m in enumerate(meta) if m["field"] == field])
    return idx[np.argsort(-scores[idx])]


def founder_percentiles(scores: np.ndarray, meta: list[dict]) -> list[dict]:
    """For each founder, its percentile rank among its own field (100 = top)."""
    pmid_to_i = {m["pmid"]: i for i, m in enumerate(meta)}
    rows = []
    for field, spec in C.FIELDS.items():
        order = rank_within_field(scores, meta, field)
        n = len(order)
        pos = {int(gi): r for r, gi in enumerate(order)}  # 0 = best
        for pmid, desc in spec.get("founders", {}).items():
            i = pmid_to_i.get(pmid)
            if i is None or i not in pos:
                rows.append({"field": field, "pmid": pmid, "desc": desc,
                             "found": False})
                continue
            r = pos[i]
            rows.append({
                "field": field, "pmid": pmid, "desc": desc, "found": True,
                "rank": r + 1, "n_field": n,
                "percentile": round(100 * (1 - r / max(1, n - 1)), 2),
            })
    return rows


def metric_comparison(metrics: dict, meta: list[dict]) -> dict:
    """Mean founder percentile under each candidate metric (higher = better)."""
    candidates = {
        "pioneer": metrics["pioneer"],
        "isolation_only": metrics["isolation"],
        "future_count": metrics["future_count"].astype(float),
        "vanguard": metrics["vanguard"],
        "n_descendants": metrics["n_descendants"].astype(float),
        "prior_count_inv": -metrics["prior_count"].astype(float),
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
