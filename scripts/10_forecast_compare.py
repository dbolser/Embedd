"""Run the identical early-prediction experiment on a given corpus tag and print
both the as-of-D forecast and the nascent early-detection test. Compare the AUCs
across tags (local / random / combined) to see whether the forecasting signal is
real or an artifact of selecting exploding fields.

Usage: python scripts/10_forecast_compare.py [tag] [Dyears...]
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from embedd import config as C
from embedd import embed, predict, validate


def nascent_test(H, yaxis, iso, years, meta, D):
    data = predict.features_as_of(H, yaxis, iso, years, meta, D=D,
                                  min_pub=D - 3, max_pub=D, field_only=False)
    X, names, late = data["X"], data["feat_names"], data["y_follow"]
    cs = X[:, names.index("current_size")]
    nas = cs <= np.median(cs)
    Xn, laten = X[nas], late[nas]
    if len(laten) < 50:
        return None
    y = (laten >= max(np.quantile(laten, 0.9), 1)).astype(int)
    if y.sum() < 10:
        return None
    cols = [names.index(n) for n in ("isolation", "prior_density", "growth", "acceleration")]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, Xn[:, cols], y, cv=5, method="predict_proba")[:, 1]

    def fauc(name):
        a = roc_auc_score(y, Xn[:, names.index(name)])
        return round(max(a, 1 - a), 3)

    return {"n_nascent": int(nas.sum()), "n_pos": int(y.sum()),
            "iso": fauc("isolation"), "prior_density": fauc("prior_density"),
            "growth": fauc("growth"), "acceleration": fauc("acceleration"),
            "combined_geom": round(float(roc_auc_score(y, p)), 3)}


def main(tag="combined", Ds=(2011, 2013, 2015)):
    E, meta = embed.load_clean(tag)
    years = np.array([m["year"] for m in meta])
    tau = validate.choose_tau(E, target_median=25)
    print(f"[{tag}] N={len(meta)}  tau={tau} (mean-centered)")
    H, yaxis, iso = predict.neighbor_year_hist(E, years, tau=tau)

    out = {"tag": tag, "N": len(meta), "tau": tau, "asof": {}, "nascent": {}}
    for D in Ds:
        data = predict.features_as_of(H, yaxis, iso, years, meta, D=D,
                                      min_pub=D - 3, max_pub=D, field_only=False)
        ev = predict.evaluate_asof(data)
        out["asof"][D] = ev
        nt = nascent_test(H, yaxis, iso, years, meta, D)
        out["nascent"][D] = nt
        sb = ev.get("auc_size_baseline"); fm = ev.get("auc_full_model")
        print(f"  D={D}  asof: size-only={sb} full={fm}  n={ev['n']} pos={ev['n_pos']}")
        if nt:
            print(f"         nascent: combined_geom={nt['combined_geom']} "
                  f"prior_dens={nt['prior_density']} iso={nt['iso']} "
                  f"accel={nt['acceleration']}  (n_nascent={nt['n_nascent']}, pos={nt['n_pos']})")
    (C.RESULTS / f"forecast_{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote results/forecast_{tag}.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = args[0] if args else "combined"
    Ds = tuple(int(x) for x in args[1:]) if len(args) > 1 else (2011, 2013, 2015)
    main(tag, Ds)
