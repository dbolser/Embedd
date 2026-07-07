"""Diagnose embedding anisotropy and test fixes.

PubMedBERT cosines are compressed high (everything looks similar), which makes a
global similarity threshold degenerate. We compare raw vs mean-centered vs
whitened geometry on how well they SEPARATE same-field from cross-field pairs,
and report what tau (or k) yields sensible neighborhood sizes.

Throwaway exploration.
"""
from __future__ import annotations

import numpy as np

from embedd import config as C
from embedd import embed


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def pair_stats(E, meta, n=6000):
    fields = np.array([m["field"] for m in meta])
    rng = np.arange(len(meta))
    a = rng[(rng * 2654435761) % (max(1, len(rng)//n)) == 0][:n]
    same, cross = [], []
    for k in range(len(a) - 1):
        i, j = a[k], a[k + 1]
        s = float(E[i] @ E[j])
        if fields[i] == fields[j] and fields[i] not in ("background", "random"):
            same.append(s)
        else:
            cross.append(s)
    same, cross = np.array(same), np.array(cross)
    sep = (same.mean() - cross.mean()) / (0.5 * (same.std() + cross.std()) + 1e-9)
    return {"same_med": round(float(np.median(same)), 3),
            "cross_med": round(float(np.median(cross)), 3),
            "same_mean": round(float(same.mean()), 3),
            "cross_mean": round(float(cross.mean()), 3),
            "separation_d": round(float(sep), 3),
            "n_same": len(same), "n_cross": len(cross)}


def neighborhood_sizes(E, taus):
    """Median # neighbors above each tau, on a subsample (via one block)."""
    sub = E[:2000]
    S = sub @ E.T
    out = {}
    for t in taus:
        cnt = (S >= t).sum(1) - 1
        out[t] = int(np.median(cnt))
    return out


def whiten(E, k=1):
    """Remove global mean + top-k principal directions (all-but-the-top-k)."""
    mu = E.mean(0, keepdims=True)
    Ec = E - mu
    U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
    Ew = Ec - (Ec @ Vt[:k].T) @ Vt[:k]
    return unit(Ew)


def main():
    E, meta = embed.load_clean("local")
    print("== RAW ==")
    print("  pairs:", pair_stats(E, meta))
    print("  median neighbors:", neighborhood_sizes(E, [0.85, 0.9, 0.93, 0.95, 0.97]))

    Ec = unit(E - E.mean(0, keepdims=True))
    print("== MEAN-CENTERED ==")
    print("  pairs:", pair_stats(Ec, meta))
    print("  median neighbors:", neighborhood_sizes(Ec, [0.3, 0.4, 0.5, 0.6, 0.7]))

    Ew = whiten(E, k=1)
    print("== WHITENED (drop top-1) ==")
    print("  pairs:", pair_stats(Ew, meta))
    print("  median neighbors:", neighborhood_sizes(Ew, [0.3, 0.4, 0.5, 0.6, 0.7]))


if __name__ == "__main__":
    main()
