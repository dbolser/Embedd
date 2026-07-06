"""Early-prediction experiment: can we forecast future followership from the
signal available in a paper's first `warmup` years? Usage: 06_predict.py [local]"""
from __future__ import annotations

import json
import sys

import numpy as np

from embedd import config as C
from embedd import embed, predict, validate


def main(tag="local"):
    E, meta = embed.load_embeddings(tag)
    years = np.array([m["year"] for m in meta])
    cal = validate.calibrate_tau(E, meta)
    lo, hi = cal["cross_field_p90"], cal["same_field_p25"]
    tau = float(np.clip((lo + hi) / 2, 0.5, 0.85)) if hi > lo else 0.7
    print(f"tau={tau:.3f}")

    H, year_axis, isolation = predict.neighbor_year_hist(E, years, tau=tau)

    results = {}
    for warmup in (1, 2, 3):
        data = predict.early_features_and_labels(H, year_axis, isolation, years,
                                                 warmup=warmup)
        ev = predict.evaluate_prediction(data)
        results[f"warmup_{warmup}"] = ev
        print(f"\n=== warmup={warmup}y  (n={ev['n']}, positives={ev['n_pos']}) ===")
        print("  per-feature AUC:", json.dumps(ev.get("per_feature_auc", {})))
        if "combined_temporal_auc" in ev:
            print(f"  combined temporal-holdout AUC = {ev['combined_temporal_auc']} "
                  f"(AP={ev['combined_temporal_ap']}, base rate={ev['base_rate']})")
            print("  coefficients:", json.dumps(ev["coef"]))

    (C.RESULTS / f"prediction_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote results/prediction_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
