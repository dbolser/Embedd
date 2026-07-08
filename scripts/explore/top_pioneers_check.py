"""Sanity check: does the precedence-based pioneer score surface real early,
field-leading work (boundary-censored, non-trivial region) rather than junk?

Throwaway exploration -- promote useful bits into embedd/ later.
Usage: python scripts/explore/top_pioneers_check.py [tag] [topn]
"""
from __future__ import annotations

import sys

import numpy as np

from embedd import config as C
from embedd import embed, metric, validate


def main(tag="local", topn=18, min_region=15):
    E, meta = embed.load_clean(tag)
    years = np.array([m["year"] for m in meta])
    tau = validate.choose_tau(E, target_median=25)
    print(f"tau={tau} (mean-centered, target median 25 neighbors)")
    m = metric.compute_metrics(E, years, tau=tau)
    scoreable = ((years >= C.YEAR_MIN + 3) & (years <= C.YEAR_MAX - 1)
                 & (m["region_size"] >= min_region))
    sc = np.where(scoreable, m["pioneer"], -1.0)
    order = np.argsort(-sc)[:topn]
    print("rank  yr  field        prec  region  pioneer  title")
    for r, i in enumerate(order, 1):
        mm = meta[i]
        flag = "*F*" if mm["is_founder"] else "   "
        print(f"{r:2d} {flag} {mm['year']} {mm['field']:11s} "
              f"{m['precedence'][i]:.2f} {int(m['region_size'][i]):5d} "
              f"{m['pioneer'][i]:.3f}  {mm['title'][:52]}")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "local"
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    main(tag, topn)
