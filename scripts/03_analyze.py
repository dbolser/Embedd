"""Compute metrics, calibrate tau, validate against founders, write results."""
from __future__ import annotations

import json

import numpy as np

from embedd import config as C
from embedd import embed, metric, validate


def main(tag: str = "local") -> None:
    E, meta = embed.load_clean(tag)
    years = np.array([m["year"] for m in meta])
    print(f"loaded {E.shape} embeddings, years {years.min()}-{years.max()}")

    tau = validate.choose_tau(E, target_median=25)
    print(f"chosen tau = {tau} (mean-centered, ~25 median neighbors)")

    m = metric.compute_metrics(E, years, tau=tau)
    comp = validate.metric_comparison(m, meta)
    print("\n=== metric comparison (mean founder percentile) ===")
    print(json.dumps(comp, indent=2))

    print("\n=== founder ranks under pioneer score ===")
    for r in validate.founder_percentiles(m["pioneer"], meta):
        if r.get("found"):
            print(f"  {r['field']:12s} p{r['percentile']:6.2f}  "
                  f"(rank {r['rank']}/{r['n_field']})  {r['desc'][:60]}")
        else:
            print(f"  {r['field']:12s}  MISSING  {r['pmid']}")

    # persist full per-abstract table
    out = C.RESULTS / f"metrics_{tag}.jsonl"
    with out.open("w") as fh:
        for i, mm in enumerate(meta):
            row = {**mm, "tau": tau}
            for k in ("prior_count", "future_count", "isolation", "vanguard",
                      "n_descendants", "is_seed", "pioneer"):
                v = m[k][i]
                row[k] = float(v) if isinstance(v, (np.floating, float)) else int(v)
            fh.write(json.dumps(row) + "\n")
    with (C.RESULTS / f"summary_{tag}.json").open("w") as fh:
        json.dump({"tau": tau, "calibration": cal, "comparison": comp,
                   "founders": validate.founder_percentiles(m["pioneer"], meta)},
                  fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
