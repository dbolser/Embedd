"""The 'reckoning' demonstrator: where does citation-free precedence agree with,
and where does it diverge from, actual citations?

We compare the pioneer (precedence) score against NIH iCite's Relative Citation
Ratio (RCR, field- and time-normalized) within each field. Two divergences are
the interesting story:
  * OVERLOOKED  -- high precedence, modest citations (early work the crowd under-cites)
  * BANDWAGON   -- high citations, low precedence (rode an existing wave)

Usage: python scripts/11_citation_compare.py [tag]
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

from embedd import config as C
from embedd import embed, metric, validate


def pct(x):
    r = rankdata(x)
    return 100 * (r - 1) / (len(r) - 1)


def main(tag="local"):
    E, meta = embed.load_clean(tag)
    years = np.array([m["year"] for m in meta])
    tau = validate.choose_tau(E, target_median=25)
    m = metric.compute_metrics(E, years, tau=tau)
    pioneer = m["pioneer"]

    icite = json.loads((C.EMBEDDINGS / "icite.json").read_text())
    rcr = np.array([(icite.get(mm["pmid"], {}) or {}).get("rcr") or np.nan
                    for mm in meta])
    cites = np.array([(icite.get(mm["pmid"], {}) or {}).get("citation_count") or 0
                      for mm in meta])
    fields = np.array([mm["field"] for mm in meta])

    print(f"tag={tag} tau={tau}\n")
    overlooked, bandwagon = [], []
    for f in C.FIELDS:
        sel = np.where((fields == f) & np.isfinite(rcr))[0]
        if len(sel) < 30:
            continue
        pp = pct(pioneer[sel])
        rp = pct(rcr[sel])
        rho, _ = spearmanr(pioneer[sel], rcr[sel])
        print(f"{f:12s} n={len(sel):4d}  spearman(pioneer, RCR)={rho:+.3f}")
        gap = pp - rp
        for k in np.argsort(-gap)[:3]:
            i = sel[k]
            overlooked.append((gap[k], f, meta[i], pp[k], rp[k], int(cites[i]),
                               float(rcr[i])))
        for k in np.argsort(gap)[:3]:
            i = sel[k]
            bandwagon.append((-gap[k], f, meta[i], pp[k], rp[k], int(cites[i]),
                              float(rcr[i])))

    def show(title, rows):
        print(f"\n=== {title} ===")
        for gap, f, mm, pp_, rp_, ct, r in sorted(rows, key=lambda x: -x[0])[:10]:
            print(f"  [{f:11s} {mm['year']}] pioneer_pct={pp_:5.1f} "
                  f"rcr_pct={rp_:5.1f} cites={ct:5d} rcr={r:5.1f}  "
                  f"{'*F*' if mm['is_founder'] else '   '} {mm['title'][:48]}")

    show("OVERLOOKED: high precedence, low citations", overlooked)
    show("BANDWAGON: high citations, low precedence", bandwagon)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
