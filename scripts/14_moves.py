"""Hunt for reusable 'moves of science' in the broad Titan-embedded corpus.

Compares two views of the cluster-to-cluster relation vectors:
  * DIRECTION clustering (normalized) — recurring directions, ignores magnitude.
  * OFFSET clustering (raw c_B - c_A) — recurring *offsets*; a tight offset
    cluster IS a set of analogous pairs (parallelogram c_B-c_A = c_D-c_C).
Reports endpoint-diverse, coherent groups and characterizes each move's terms.

Usage: python scripts/14_moves.py [tag]   (default tag: broad)
"""
from __future__ import annotations

import json
import sys

import numpy as np

from embedd import config as C
from embedd import concepts, embed


def load_centered(tag):
    E = np.load(C.EMBEDDINGS / f"vecs_{tag}.npy")
    meta = [json.loads(l) for l in (C.EMBEDDINGS / f"meta_{tag}.jsonl").read_text().splitlines()]
    # drop junk, mean-center (reuse the pilot cleaner via titles/lengths if present)
    keep = embed.clean_mask(meta)
    E, meta = E[keep], [m for m, k in zip(meta, keep) if k]
    E = embed.center_unit(E)
    return E, meta


def report_groups(title, pairs, vecs, clusters, meta):
    rlabels, groups = concepts.cluster_relations(pairs, vecs, min_cluster_size=6,
                                                 min_samples=3, min_distinct=4)
    diverse = [g for g in groups if g["diverse"]]
    print(f"\n########## {title}: {len(groups)} groups, {len(diverse)} diverse ##########")
    for g in diverse[:12]:
        d = concepts.describe_relation(g, pairs, clusters, meta)
        print(f"\n  R{d['rid']} size={d['size']} coh={d['coherence']} "
              f"src={d['n_src']} dst={d['n_dst']}")
        print(f"    move-terms: {', '.join(d['move_terms'][:10])}")
        for ex in d["examples"][:5]:
            print(f"      [{','.join(ex['from']['terms'][:3])}] -> "
                  f"[{','.join(ex['to']['terms'][:3])}]")
    return diverse


def main(tag="broad"):
    E, meta = load_centered(tag)
    print(f"[{tag}] N={len(meta)} (cleaned, centered)")
    labels, Z = concepts.cluster_concepts(E, min_cluster_size=60, min_samples=15)
    nC = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"clusters={nC} noise={(labels==-1).sum()} ({100*(labels==-1).mean():.1f}%)")
    clusters = concepts.build_clusters(E, labels)
    concepts.label_clusters(clusters, meta, E)

    pairs, diffs, units = concepts.relation_vectors(clusters)
    report_groups("DIRECTION (normalized)", pairs, units, clusters, meta)
    report_groups("OFFSET (raw c_B - c_A)", pairs, diffs, clusters, meta)

    # persist cluster labels for interactive seeded probes later
    out = {"n_clusters": nC,
           "clusters": [{"cid": cl.cid, "size": cl.size, "fields": cl.top_fields,
                         "terms": cl.terms, "exemplars": cl.exemplars}
                        for cl in clusters]}
    (C.RESULTS / f"moves_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/moves_{tag}.json ({nC} clusters)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "broad")
