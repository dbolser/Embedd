"""Concept clustering + recurring-relation discovery.
Usage: python scripts/05_concepts.py [local]"""
from __future__ import annotations

import json
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from embedd import config as C  # noqa: E402
from embedd import concepts, embed  # noqa: E402


def fig_concept_map(Z2, labels, clusters, tag):
    fig, ax = plt.subplots(figsize=(12, 10))
    noise = labels == -1
    ax.scatter(Z2[noise, 0], Z2[noise, 1], s=2, c="#e0e0e0", alpha=0.3, linewidths=0)
    rng = np.linspace(0, 1, max(2, len(clusters)))
    cmap = plt.cm.nipy_spectral
    for r, cl in enumerate(clusters):
        m = cl.members
        ax.scatter(Z2[m, 0], Z2[m, 1], s=5, color=cmap(rng[r]), alpha=0.6, linewidths=0)
        cx, cy = Z2[m, 0].mean(), Z2[m, 1].mean()
        ax.text(cx, cy, str(cl.cid), fontsize=8, fontweight="bold",
                ha="center", va="center")
    ax.set_title(f"Concept clusters ({len(clusters)} clusters, {tag})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"concept_map_{tag}.png", dpi=140)
    plt.close(fig)


def main(tag="local"):
    E, meta = embed.load_clean(tag)
    print(f"loaded {E.shape}")
    labels, Z = concepts.cluster_concepts(E)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"clusters={n_clusters}  noise={n_noise} ({100*n_noise/len(labels):.1f}%)")

    clusters = concepts.build_clusters(E, labels)
    concepts.label_clusters(clusters, meta, E)

    print("\n=== concept clusters ===")
    for cl in clusters:
        fields = ",".join(f"{k}:{v}" for k, v in cl.top_fields.items())
        print(f"  c{cl.cid:3d} (n={cl.size:4d}) [{fields}]  {', '.join(cl.terms[:8])}")

    # relations
    pairs, diffs, units = concepts.relation_vectors(clusters)
    rlabels, groups = concepts.cluster_relations(pairs, units)
    diverse = [g for g in groups if g["diverse"]]
    print(f"\n=== relation groups: {len(groups)} total, {len(diverse)} endpoint-diverse ===")
    described = []
    for g in diverse:
        d = concepts.describe_relation(g, pairs, clusters, meta)
        described.append(d)
        print(f"\n  R{d['rid']} size={d['size']} coh={d['coherence']} "
              f"src={d['n_src']} dst={d['n_dst']}")
        print(f"    move-terms: {', '.join(d['move_terms'][:10])}")
        for ex in d["examples"][:4]:
            print(f"    [{','.join(ex['from']['terms'][:3])}] -> "
                  f"[{','.join(ex['to']['terms'][:3])}]")

    # save 2D projection for the figure (reuse first 2 UMAP comps via a fresh 2D fit)
    import umap
    Z2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric="cosine", random_state=42).fit_transform(E)
    fig_concept_map(Z2, labels, clusters, tag)

    out = {
        "n_clusters": n_clusters, "n_noise": n_noise,
        "clusters": [{"cid": cl.cid, "size": cl.size, "fields": cl.top_fields,
                      "terms": cl.terms, "exemplars": cl.exemplars} for cl in clusters],
        "relations": described,
    }
    (C.RESULTS / f"concepts_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/concepts_{tag}.json and figure")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
