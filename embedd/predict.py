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


def features_as_of(H, year_axis, isolation, years, meta, D: int,
                   min_pub: int, max_pub: int, field_only: bool = False,
                   top_frac: float = 0.10):
    """'Standing in year D' forecast setup with no leakage.

    For each paper published in [min_pub, max_pub] (all with pub <= D), compute
    features from abstracts up to and including year D, and the label from
    abstracts published strictly after D. This is a genuine forecast: predict the
    future (>D) from the present (<=D).
    """
    yr_to_k = {int(y): k for k, y in enumerate(year_axis)}
    kD = yr_to_k[D]
    feats, late, idx = [], [], []
    for i in range(len(years)):
        t = int(years[i])
        if t < min_pub or t > max_pub or t > D:
            continue
        if field_only and meta[i]["field"] == "background":
            continue
        kt = yr_to_k[t]
        prior = float(H[i, :kt].sum())               # neighbors before publication
        early = float(H[i, kt:kD + 1].sum())         # neighbors from pub..D (current size)
        growth = float(H[i, kt + 1:kD + 1].sum())    # neighbors accrued after pub, by D
        late_ct = float(H[i, kD + 1:].sum())         # FUTURE neighbors (label)
        feats.append([
            isolation[i], prior, early, growth,
            growth / (early + 1.0),        # early acceleration
            float(D - t),                  # age at decision (control)
        ])
        late.append(late_ct)
        idx.append(i)
    X = np.array(feats, dtype=np.float32)
    late = np.array(late, dtype=np.float32)
    idx = np.array(idx)
    names = ["isolation", "prior_density", "current_size", "growth",
             "acceleration", "age"]
    if len(late):
        thr = np.quantile(late, 1 - top_frac)
        y_top = (late >= max(thr, 1)).astype(int)
    else:
        y_top = np.array([], dtype=int)
    return {"X": X, "feat_names": names, "y_follow": late, "y_top": y_top,
            "idx": idx, "D": D}


def evaluate_asof(data: dict, seed: int = 0):
    """Cross-validated AUC: full model vs a size-only baseline, plus the key
    test -- does the metric predict who OUTGROWS their current size?"""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y, names = data["X"], data["y_top"], data["feat_names"]
    out = {"D": data["D"], "n": int(len(y)), "n_pos": int(y.sum()),
           "base_rate": round(float(y.mean()), 4) if len(y) else None}
    if y.sum() < 10 or y.sum() > len(y) - 10:
        out["note"] = "too few positives/negatives"
        return out

    for c, name in enumerate(names):
        s = X[:, c]
        auc = roc_auc_score(y, s)
        out.setdefault("per_feature_auc", {})[name] = round(float(max(auc, 1 - auc)), 3)

    def cv_auc(cols):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        p = cross_val_predict(clf, X[:, cols], y, cv=5, method="predict_proba")[:, 1]
        return round(float(roc_auc_score(y, p)), 3)

    size_col = [names.index("current_size"), names.index("age")]
    all_cols = list(range(len(names)))
    out["auc_size_baseline"] = cv_auc(size_col)
    out["auc_full_model"] = cv_auc(all_cols)

    # residual test: regress late-followers on current_size+age, ask whether
    # isolation/acceleration predict who beats that expectation.
    from sklearn.metrics import roc_auc_score as _auc
    late = data["y_follow"]
    base = LinearRegression().fit(
        np.log1p(X[:, size_col]), np.log1p(late))
    resid = np.log1p(late) - base.predict(np.log1p(X[:, size_col]))
    over = (resid >= np.quantile(resid, 1 - 0.10)).astype(int)
    for name in ("isolation", "acceleration"):
        c = names.index(name)
        a = _auc(over, X[:, c])
        out.setdefault("residual_auc", {})[name] = round(float(max(a, 1 - a)), 3)
    return out


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
