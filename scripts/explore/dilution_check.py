"""Dilution test: when the ground-truth founders are embedded among tens of
thousands of outcome-blind random abstracts, does the precedence score still
rank them at the top -- within their field AND corpus-wide?

If corpus-wide founder percentiles stay high under dilution, the metric is not
merely re-sorting a hand-picked field. Throwaway exploration.
Usage: python scripts/explore/dilution_check.py [tag]
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import rankdata

from embedd import config as C
from embedd import embed, metric, validate


def main(tag="combined"):
    E, meta = embed.load_clean(tag)
    years = np.array([m["year"] for m in meta])
    fields = np.array([m["field"] for m in meta])
    tau = validate.choose_tau(E, target_median=25)
    print(f"[{tag}] N={len(meta)}  tau={tau}")
    m = metric.compute_metrics(E, years, tau=tau)
    pio = m["pioneer"]

    # corpus-wide tie-aware percentile
    ranks_all = rankdata(pio)
    n_all = len(pio)
    pmid_to_i = {mm["pmid"]: i for i, mm in enumerate(meta)}

    print(f"\n{'field':11s} {'pmid':9s} {'field_pct':>9s} {'corpus_pct':>10s} "
          f"{'region':>6s} {'prec':>4s}  title")
    fps = []
    for field, spec in C.FIELDS.items():
        sel = np.where(fields == field)[0]
        if len(sel) < 2:
            continue
        franks = rankdata(pio[sel])
        local = {int(gi): k for k, gi in enumerate(sel)}
        for pmid, desc in spec["founders"].items():
            i = pmid_to_i.get(pmid)
            if i is None:
                print(f"  {field:11s} {pmid:9s}   MISSING")
                continue
            fpct = 100 * (franks[local[i]] - 1) / (len(sel) - 1)
            cpct = 100 * (ranks_all[i] - 1) / (n_all - 1)
            fps.append((fpct, cpct))
            print(f"{field:11s} {pmid:9s} {fpct:9.2f} {cpct:10.2f} "
                  f"{int(m['region_size'][i]):6d} {m['precedence'][i]:4.2f}  {desc[:34]}")
    fps = np.array(fps)
    print(f"\nmean field pct = {fps[:,0].mean():.1f}  |  "
          f"mean corpus-wide pct = {fps[:,1].mean():.1f}  (n={len(fps)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "combined")
