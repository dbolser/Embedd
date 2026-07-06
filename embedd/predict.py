"""Can we predict pioneers *early*, before the region fills in?

This is the load-bearing experiment. For every abstract we split time at its own
publication date plus a short warm-up window w:

  * EARLY features use only abstracts published up to  t_i + w  (available then).
  * LATE label uses only abstracts published after     t_i + w  (the future).

If an early signal (isolation at birth + the first w years of local dynamics)
predicts which papers get heavily followed *later*, we are forecasting
foundational work, not confirming it after the fact -- and doing so years before
citations could reveal it.

Everything is derived from one artifact: for each abstract, a histogram of how
many within-tau neighbors it has in each calendar year.
"""
from __future__ import annotations

import numpy as np


def neighbor_year_hist(E: np.ndarray, years: np.ndarray, tau: float = 0.7,
                       block: int = 512):
    """Return (H, year_axis, isolation).

    H[i, y] = # abstracts within cosine tau of i published in calendar year
    year_axis[y] (self excluded). isolation[i] = 1 - max sim to any same-year-
    or-earlier abstract.
    """
    N = E.shape[0]
    year_axis = np.arange(years.min(), years.max() + 1)
    yidx = {int(y): k for k, y in enumerate(year_axis)}
    col_year = np.array([yidx[int(y)] for y in years])  # (N,) year-bin of each abstract
    onehot = np.zeros((N, len(year_axis)), dtype=np.float32)
    onehot[np.arange(N), col_year] = 1.0

    H = np.zeros((N, len(year_axis)), dtype=np.float32)
    isolation = np.ones(N, dtype=np.float32)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        S = E[lo:hi] @ E.T                      # (b, N)
        within = (S >= tau).astype(np.float32)
        # exclude self
        for bi in range(hi - lo):
            within[bi, lo + bi] = 0.0
        H[lo:hi] = within @ onehot              # (b, Y)
        # isolation: nearest same-year-or-earlier neighbor
        yb = years[lo:hi][:, None]
        le = years[None, :] <= yb
        for bi in range(hi - lo):
            le[bi, lo + bi] = False
        S_le = np.where(le, S, -1.0)
        best = S_le.max(1)
        has = best > -1.0
        isolation[lo:hi] = np.where(has, 1.0 - np.clip(best, -1, 1), 1.0)
    return H, year_axis, isolation


def early_features_and_labels(H, year_axis, isolation, years,
                              warmup: int = 2, min_followers: int = 5,
                              top_frac: float = 0.10, require_year_max=None):
    """Split each abstract's neighbor history at t_i + warmup.

    Returns a dict of arrays over the *eligible* abstracts (those with at least
    `warmup` years of observable future inside the corpus window):
      X            : early feature matrix
      feat_names   : names of the feature columns
      y_follow     : late follower count (label, continuous)
      y_top        : binary label, top `top_frac` by late followers
      idx          : indices into the corpus of eligible abstracts
    """
    N = H.shape[0]
    ymax = int(year_axis.max()) if require_year_max is None else require_year_max
    yr_to_k = {int(y): k for k, y in enumerate(year_axis)}

    feats, labels, idx = [], [], []
    for i in range(N):
        t = int(years[i])
        # need at least `warmup` observable years after publication
        if t + warmup > ymax:
            continue
        kt = yr_to_k[t]
        k_end_early = yr_to_k[t + warmup]
        # EARLY: neighbors published in [t, t+warmup]  (contemporaneous emergence)
        early_same = H[i, kt:k_end_early + 1].sum()
        # prior neighbors (before t): density at birth
        prior = H[i, :kt].sum()
        # early growth: neighbors appearing in the warm-up window
        growth = H[i, kt + 1:k_end_early + 1].sum()
        # LATE: neighbors published strictly after t+warmup (the future)
        late = H[i, k_end_early + 1:].sum()

        feats.append([
            isolation[i],          # novelty at birth
            prior,                 # prior density (low for pioneers)
            early_same,            # contemporaneous crowd
            growth,                # early follower influx in warm-up
            growth / (early_same + 1.0),  # early acceleration
        ])
        labels.append(late)
        idx.append(i)

    X = np.array(feats, dtype=np.float32)
    y_follow = np.array(labels, dtype=np.float32)
    idx = np.array(idx)
    feat_names = ["isolation", "prior_density", "early_crowd",
                  "early_growth", "early_acceleration"]
    if len(y_follow):
        thresh = np.quantile(y_follow, 1 - top_frac)
        y_top = (y_follow >= max(thresh, min_followers)).astype(int)
    else:
        y_top = np.array([], dtype=int)
    return {"X": X, "feat_names": feat_names, "y_follow": y_follow,
            "y_top": y_top, "idx": idx}


def evaluate_prediction(data: dict, seed: int = 0):
    """Time-agnostic AUC of each early feature alone, plus a simple combined
    logistic model under a temporal split (train older half, test newer half)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    X, y = data["X"], data["y_top"]
    names = data["feat_names"]
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "per_feature_auc": {}}
    if y.sum() == 0 or y.sum() == len(y):
        return out
    for c, name in enumerate(names):
        try:
            out["per_feature_auc"][name] = round(float(roc_auc_score(y, X[:, c])), 3)
        except ValueError:
            out["per_feature_auc"][name] = None

    # combined model, temporal holdout by corpus order (idx is time-sorted)
    order = np.argsort(data["idx"])
    Xs, ys = X[order], y[order]
    cut = len(ys) // 2
    if ys[:cut].sum() and ys[cut:].sum():
        sc = StandardScaler().fit(Xs[:cut])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(sc.transform(Xs[:cut]), ys[:cut])
        p = clf.predict_proba(sc.transform(Xs[cut:]))[:, 1]
        out["combined_temporal_auc"] = round(float(roc_auc_score(ys[cut:], p)), 3)
        out["combined_temporal_ap"] = round(float(average_precision_score(ys[cut:], p)), 3)
        out["coef"] = {n: round(float(w), 3) for n, w in zip(names, clf.coef_[0])}
        out["base_rate"] = round(float(ys[cut:].mean()), 3)
    return out
