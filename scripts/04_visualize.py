"""Figures: the embedding landscape, the archetype quadrants, temporal fill-in,
and top-pioneer tables. Usage: python scripts/04_visualize.py [local]"""
from __future__ import annotations

import json
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from embedd import config as C  # noqa: E402
from embedd import embed, metric, validate  # noqa: E402

FIELD_COLORS = {
    "crispr": "#e6194b", "rnai": "#3cb44b", "ipsc": "#4363d8",
    "optogenetics": "#f58231", "gwas": "#911eb4", "scrnaseq": "#008080",
    "background": "#cfcfcf",
}


def project_2d(E: np.ndarray) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                            random_state=42)
        return reducer.fit_transform(E)
    except Exception as e:  # noqa: BLE001
        print(f"umap unavailable ({e}); using PCA", flush=True)
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(E)


def fig_landscape(E2, meta, m, tag):
    fields = np.array([mm["field"] for mm in meta])
    founder = np.array([mm["is_founder"] for mm in meta])
    fig, ax = plt.subplots(figsize=(10, 8))
    # background first
    bg = fields == "background"
    ax.scatter(E2[bg, 0], E2[bg, 1], s=3, c=FIELD_COLORS["background"],
               alpha=0.4, linewidths=0, label="background")
    for f, col in FIELD_COLORS.items():
        if f == "background":
            continue
        sel = (fields == f) & ~founder
        ax.scatter(E2[sel, 0], E2[sel, 1], s=5, c=col, alpha=0.6,
                   linewidths=0, label=C.FIELDS[f]["label"])
    fs = founder
    ax.scatter(E2[fs, 0], E2[fs, 1], s=220, marker="*", c="black",
               edgecolors="white", linewidths=1.2, zorder=5, label="founders")
    ax.set_title("Abstract embedding landscape (founders starred)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"landscape_{tag}.png", dpi=140)
    plt.close(fig)


def fig_quadrants(m, meta, tag):
    """Archetype map: temporal precedence vs region size. Pioneers are top-right
    (a large region that arrives mostly after them); bandwagon work is bottom
    (low precedence); tiny corners are left."""
    prec = m["precedence"]
    size = np.log1p(m["region_size"].astype(float))
    founder = np.array([mm["is_founder"] for mm in meta])
    field = np.array([mm["field"] for mm in meta])
    fig, ax = plt.subplots(figsize=(9, 7))
    real = ~np.isin(field, ["background", "random"])
    ax.scatter(prec[real & ~founder], size[real & ~founder], s=6, c="#9aa",
               alpha=0.4, linewidths=0, label="field abstracts")
    other = np.isin(field, ["background", "random"])
    ax.scatter(prec[other], size[other], s=4, c="#e5e5e5", alpha=0.3,
               linewidths=0, label="background/random")
    for i in np.where(founder)[0]:
        ax.scatter(prec[i], size[i], s=180, marker="*",
                   c=FIELD_COLORS.get(field[i], "#000"),
                   edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_xlabel("temporal precedence  (fraction of region published after)")
    ax.set_ylabel("region size  log(1 + neighbors)")
    ax.set_title("Archetype map: pioneers lead a large region that fills in behind them")
    ax.text(0.98, 0.98, "PIONEERS", ha="right", va="top", transform=ax.transAxes,
            color="black", fontsize=11, fontweight="bold")
    ax.text(0.02, 0.98, "derivative in\na big field", ha="left", va="top",
            transform=ax.transAxes, color="#06c", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"quadrants_{tag}.png", dpi=140)
    plt.close(fig)


def fig_precedence_year(m, meta, tag):
    """Precedence vs publication year; founders starred. Shows founders combine
    early dates with high precedence."""
    prec = m["precedence"]
    years = np.array([mm["year"] for mm in meta])
    founder = np.array([mm["is_founder"] for mm in meta])
    field = np.array([mm["field"] for mm in meta])
    real = ~np.isin(field, ["background", "random"])
    fig, ax = plt.subplots(figsize=(10, 6))
    jitter = ((np.arange(len(years)) % 7) - 3) / 12.0
    ax.scatter(years[real & ~founder] + jitter[real & ~founder], prec[real & ~founder],
               s=5, c="#bbb", alpha=0.4, linewidths=0)
    for i in np.where(founder)[0]:
        ax.scatter(years[i], prec[i], s=170, marker="*",
                   c=FIELD_COLORS.get(field[i], "#000"), edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(field[i], (years[i], prec[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("publication year"); ax.set_ylabel("temporal precedence")
    ax.set_title("Founders combine early dates with future-facing neighborhoods")
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"precedence_year_{tag}.png", dpi=140)
    plt.close(fig)


def fig_fillin(E, meta, m, tag, tau):
    """For each field, the count of abstracts per year, with founder years marked —
    the 'region fills in behind the pioneer' picture."""
    years = np.array([mm["year"] for mm in meta])
    field = np.array([mm["field"] for mm in meta])
    founder = np.array([mm["is_founder"] for mm in meta])
    fields = [f for f in C.FIELDS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, f in zip(axes.ravel(), fields):
        sel = field == f
        ys = years[sel]
        yr = np.arange(C.YEAR_MIN, C.YEAR_MAX + 1)
        counts = [(ys == y).sum() for y in yr]
        ax.bar(yr, counts, color=FIELD_COLORS[f], alpha=0.7)
        for i in np.where(sel & founder)[0]:
            ax.axvline(meta[i]["year"], color="black", ls="--", lw=1)
        ax.set_title(C.FIELDS[f]["label"], fontsize=10)
        ax.set_ylabel("abstracts/yr", fontsize=8)
    fig.suptitle("Temporal fill-in per field (dashed = founder year)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fillin_{tag}.png", dpi=140)
    plt.close(fig)


def top_pioneers_table(m, meta, tag, topn=25):
    # Boundary censoring: papers within BOUNDARY years of the window start have
    # no observable prior context (artificially isolated) and become spurious
    # ancestors of everything after them; drop them from the ranking.
    boundary = 3
    years = np.array([mm["year"] for mm in meta])
    scoreable = (years >= C.YEAR_MIN + boundary) & (years <= C.YEAR_MAX - 1)
    score = np.where(scoreable, m["pioneer"], -np.inf)
    order = np.argsort(-score)
    rows = []
    for rank, i in enumerate(order[:topn], 1):
        mm = meta[i]
        rows.append({
            "rank": rank, "pioneer": round(float(m["pioneer"][i]), 4),
            "isolation": round(float(m["isolation"][i]), 3),
            "vanguard": round(float(m["vanguard"][i]), 1),
            "n_desc": int(m["n_descendants"][i]),
            "year": mm["year"], "field": mm["field"],
            "is_founder": mm["is_founder"], "pmid": mm["pmid"],
            "title": mm["title"][:80],
        })
    with (C.RESULTS / f"top_pioneers_{tag}.json").open("w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n=== top {topn} pioneers (corpus-wide, {tag}) ===")
    for r in rows:
        flag = " *FOUNDER*" if r["is_founder"] else ""
        print(f"  {r['rank']:2d}. [{r['field']:11s} {r['year']}] "
              f"P={r['pioneer']:.3f} desc={r['n_desc']:4d}{flag}  {r['title'][:60]}")
    return rows


def main(tag="local"):
    E, meta = embed.load_clean(tag)
    years = np.array([mm["year"] for mm in meta])
    tau = validate.choose_tau(E, target_median=25)
    print(f"tau={tau} (mean-centered)")
    m = metric.compute_metrics(E, years, tau=tau)

    top_pioneers_table(m, meta, tag)
    print("projecting to 2D ...", flush=True)
    E2 = project_2d(E)
    fig_landscape(E2, meta, m, tag)
    fig_quadrants(m, meta, tag)
    fig_precedence_year(m, meta, tag)
    fig_fillin(E, meta, m, tag, tau)
    print(f"figures -> {C.FIGURES}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
