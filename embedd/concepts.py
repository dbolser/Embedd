"""Concept clustering and the *relations between concepts* in embedding space.

Two stages:
  1. Cluster the abstract embeddings into concept clusters (UMAP -> HDBSCAN).
     Label each cluster by its enriched terms and representative titles.
  2. Treat every ordered pair of concept centroids as a directed "relation
     vector" d = c_j - c_i, and ask whether these relation vectors themselves
     cluster. A recurring relation is a *direction* shared by many *disjoint*
     concept pairs (B-A ~ D-C), e.g. a "genome-wide-ization" or "single-cell-
     ization" move that recurs across unrelated fields.

The key guard against a trivial result: a group of relation vectors only counts
as a real recurring relation if its member pairs have DIVERSE endpoints (not all
sharing one source or destination cluster) -- otherwise we've just rediscovered
"vectors that point at cluster j".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    cid: int
    members: np.ndarray            # indices into the corpus
    centroid: np.ndarray          # unit-normalized, in original embedding space
    size: int
    terms: list[str] = field(default_factory=list)
    top_fields: dict = field(default_factory=dict)
    exemplars: list[str] = field(default_factory=list)


def cluster_concepts(E: np.ndarray, n_components: int = 10,
                     n_neighbors: int = 30, min_cluster_size: int = 40,
                     min_samples: int = 10, seed: int = 42):
    """UMAP -> HDBSCAN. Cluster in a reduced space, but keep centroids in the
    ORIGINAL embedding space so relation vectors live in the semantic space."""
    import umap
    import hdbscan

    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=0.0, metric="cosine", random_state=seed)
    Z = reducer.fit_transform(E)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric="euclidean")
    labels = clusterer.fit_predict(Z)
    return labels, Z


def build_clusters(E: np.ndarray, labels: np.ndarray) -> list[Cluster]:
    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1:  # HDBSCAN noise
            continue
        members = np.where(labels == cid)[0]
        c = E[members].mean(0)
        c /= np.linalg.norm(c) + 1e-12
        clusters.append(Cluster(cid=int(cid), members=members,
                                centroid=c.astype(np.float32), size=len(members)))
    return clusters


def label_clusters(clusters: list[Cluster], meta: list[dict], E: np.ndarray,
                   top_terms: int = 12, top_ex: int = 4) -> None:
    """Attach enriched terms (TF-IDF vs rest), field mix, and exemplar titles."""
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [f"{meta[i]['title']} {meta[i].get('journal','')}" for i in range(len(meta))]
    # cluster-level documents for term enrichment
    cl_docs = []
    for cl in clusters:
        cl_docs.append(" ".join(docs[i] for i in cl.members))
    vec = TfidfVectorizer(max_features=8000, stop_words="english",
                          ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(cl_docs)
    vocab = np.array(vec.get_feature_names_out())
    for r, cl in enumerate(clusters):
        row = X[r].toarray().ravel()
        cl.terms = vocab[np.argsort(-row)[:top_terms]].tolist()
        fields = Counter(meta[i]["field"] for i in cl.members)
        cl.top_fields = dict(fields.most_common(4))
        # exemplars: members closest to centroid
        sims = E[cl.members] @ cl.centroid
        best = cl.members[np.argsort(-sims)[:top_ex]]
        cl.exemplars = [meta[i]["title"][:90] for i in best]


def relation_vectors(clusters: list[Cluster]):
    """All ordered pairs -> directed relation vectors and their unit directions."""
    C = np.stack([cl.centroid for cl in clusters])  # (K, d), unit rows
    K = len(clusters)
    pairs = []
    diffs = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            d = C[j] - C[i]
            pairs.append((i, j))
            diffs.append(d)
    diffs = np.array(diffs, dtype=np.float32)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12
    units = diffs / norms
    return np.array(pairs), diffs, units


def analogy_map(clusters: list[Cluster], src_i: int, dst_i: int,
                tol_frac: float = 0.5):
    """Seeded analogy probe. Given a move v = c[dst] - c[src], find every cluster
    C whose image C+v lands nearest to a real, distinct cluster D (within
    tol_frac * |v|). Returns list of (C_idx, D_idx, dist/|v|) — the pairs the
    move 'explains'. This is the word2vec-style test: does one transformation
    generalize across domains?"""
    from scipy.spatial.distance import cdist

    C = np.stack([cl.centroid for cl in clusters])
    v = C[dst_i] - C[src_i]
    vn = np.linalg.norm(v) + 1e-12
    pred = C + v
    Dp = cdist(pred, C)
    np.fill_diagonal(Dp, np.inf)
    out = []
    for a in range(len(clusters)):
        b = int(Dp[a].argmin())
        rel = Dp[a, b] / vn
        if b != a and rel < tol_frac:
            out.append((a, b, round(float(rel), 3)))
    return sorted(out, key=lambda x: x[2])


def cluster_relations(pairs: np.ndarray, units: np.ndarray,
                      min_cluster_size: int = 6, min_samples: int = 3,
                      min_distinct: int = 4, reduce_dims: int | None = 15):
    """Cluster relation vectors. Return groups that pass the endpoint-diversity
    guard (>= min_distinct distinct sources AND destinations).

    HDBSCAN on the full 768/1024-dim vectors suffers the curse of dimensionality
    (distances concentrate, everything collapses into one blob). So by default we
    UMAP-reduce the relation vectors to `reduce_dims` first, then cluster there;
    coherence is still measured in the original space so it stays meaningful.
    """
    import hdbscan

    if reduce_dims and units.shape[1] > reduce_dims and len(units) > reduce_dims + 2:
        import umap
        space = umap.UMAP(n_components=reduce_dims,
                          n_neighbors=min(30, len(units) - 1),
                          min_dist=0.0, metric="cosine",
                          random_state=42).fit_transform(units)
    else:
        space = units
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples, metric="euclidean")
    rlabels = clusterer.fit_predict(space)
    groups = []
    for rid in sorted(set(rlabels)):
        if rid == -1:
            continue
        idx = np.where(rlabels == rid)[0]
        srcs = set(pairs[idx, 0].tolist())
        dsts = set(pairs[idx, 1].tolist())
        diverse = len(srcs) >= min_distinct and len(dsts) >= min_distinct
        # coherence: mean pairwise cosine of the directions in this group
        U = units[idx]
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        coh = float((U @ U.T).mean())
        groups.append({
            "rid": int(rid), "size": int(len(idx)),
            "member_pair_idx": idx,
            "n_src": len(srcs), "n_dst": len(dsts),
            "diverse": bool(diverse), "coherence": round(coh, 3),
        })
    groups.sort(key=lambda g: (-g["diverse"], -g["coherence"]))
    return rlabels, groups


def describe_relation(group: dict, pairs: np.ndarray, clusters: list[Cluster],
                      meta: list[dict], max_examples: int = 6):
    """Characterize a relation group: example A->B pairs, and terms that are
    enriched in destinations vs sources across the group (the 'move')."""
    from collections import Counter

    idx = group["member_pair_idx"]
    examples = []
    src_terms, dst_terms = Counter(), Counter()
    for k in idx:
        i, j = pairs[k]
        a, b = clusters[i], clusters[j]
        examples.append({
            "from": {"cid": a.cid, "terms": a.terms[:5]},
            "to": {"cid": b.cid, "terms": b.terms[:5]},
        })
        for t in a.terms:
            src_terms[t] += 1
        for t in b.terms:
            dst_terms[t] += 1
    # terms distinctively on the destination side of the move
    move_terms = [(t, dst_terms[t] - src_terms.get(t, 0))
                  for t in dst_terms if dst_terms[t] - src_terms.get(t, 0) > 0]
    move_terms.sort(key=lambda x: -x[1])
    return {
        "rid": group["rid"], "size": group["size"],
        "coherence": group["coherence"],
        "n_src": group["n_src"], "n_dst": group["n_dst"],
        "move_terms": [t for t, _ in move_terms[:12]],
        "examples": examples[:max_examples],
    }
