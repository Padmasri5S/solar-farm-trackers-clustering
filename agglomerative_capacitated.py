"""
Agglomerative, capacity-aware clustering for solar tracker rows.
- Pass 1: merge nearest clusters while total <= 240 (with adjacency + distance threshold)
- Pass 2: merge adjacent clusters while total <= 480 (dual PCU)
- Lineage preserved; visualization and summary included

Why agglomerative?
Direct capacity control at merge time: we only merge two clusters if the combined weight ≤ capacity (240 first pass, 480 second pass).
Spatial coherence: merging is driven by nearest centroids (or graph adjacency), so groups remain compact and contiguous.
Auditability: each merge is a clear, local decision—easy to trace and explain downstream (important for a real-world PCU design review).

Design choices (with sensible defaults)
Adjacency
kNN graph (default): connect each tracker to its k=6 nearest neighbors—mirrors what you did for the Louvain graph build. [surbanajur...epoint.com]
MST (optional): use the minimum spanning tree to enforce strict contiguity (no cross‑gaps); handy where rows form chains.

Merge priority (tie‑breaks)
Primary: centroid distance (nearest first)
Secondary: utilization gain toward the target (i.e., prefer pairs whose combined total is closest to 240 in pass‑1 and 480 in pass‑2)

Distance thresholds
Pass‑1 (≤240): don’t merge if centroid distance > 60 m (tuneable), to avoid spanning across road corridors or long gaps.
Pass‑2 (≤480): allow a bit more reach, e.g., 80 m, to enable sensible dual‑PCU merges while still staying local.

Lineage & integrity
Persist full label lineage: singleton → ≤240 → ≤480, so downstream joins/QA are deterministic (same approach you used in your baseline scripts)
Inputs:
  - CSV with columns: LocationX (Easting), LocationY (Northing), Weighting
This script reads your CSV, runs pass‑1 (≤240) and pass‑2 (≤480), writes lineage, and renders a compact visualization with centroids + legend.
It defaults to kNN adjacency; set ADJACENCY_MODE="mst" to switch.
  """

import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import heapq

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "X_EL_SOLAR PANELS_2025-11-27_1354.csv"  # X_EL_SOLAR PANELS_2025-11-27_1354.csv
ADJACENCY_MODE = "knn"       # "knn" or "mst"
K_NEIGHBORS = 6              # for kNN graph
DTHRESH_240 = 60.0           # max centroid distance to consider merge in pass-1
DTHRESH_480 = 80.0           # max centroid distance to consider merge in pass-2
TARGET_1 = 240
TARGET_2 = 480
RANDOM_STATE = 42

# ----------------------------
# Utilities
# ----------------------------
def format_axes_numeric(ax, thousands=True):
    from matplotlib.ticker import FuncFormatter
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    if thousands:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}"))
    ax.grid(True, alpha=0.3, linestyle='--')

def build_knn_edges(X, k=6):
    nn = NearestNeighbors(n_neighbors=min(k+1, len(X)), metric="euclidean")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X)
    # each row i has neighbors idxs[i][1:] (skip itself at [0])
    edges = set()
    for i in range(len(X)):
        for j in idxs[i][1:]:
            a, b = sorted((i, j))
            edges.add((a, b))
    return list(edges)

def build_mst_edges(X):
    # naive Prim's algorithm (sufficient for ~2,500 points)
    n = len(X)
    remaining = set(range(n))
    current = 0
    remaining.remove(current)
    edges = []
    # precompute distance matrix on demand
    while remaining:
        # find nearest node to the current set
        best = None
        best_dist = np.inf
        for u in list(remaining):
            # nearest to any in the grown tree; optimize with KDTree if needed
            d = cdist(X[[u]], X[list(set(range(n)) - remaining)])
            m = float(np.min(d))
            if m < best_dist:
                best_dist = m
                best = u
        # connect best to its nearest in the grown set
        grown = list(set(range(n)) - remaining)
        d_to_grown = cdist(X[[best]], X[grown]).ravel()
        v = grown[int(np.argmin(d_to_grown))]
        edges.append(tuple(sorted((best, v))))
        remaining.remove(best)
    return edges

def cluster_centroid(X, ids):
    pts = X[ids]
    return np.mean(pts, axis=0)

def compute_totals(weights, clusters):
    return [np.sum(weights[list(c)]) for c in clusters]

def init_singletons(n):
    return [set([i]) for i in range(n)]

def adjacency_from_edges(edges, n):
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    return adj

def merge_priority(c1, c2, centroids, totals, target):
    # smaller centroid distance first, then closer to target (higher utilization)
    d = np.linalg.norm(centroids[c1] - centroids[c2])
    util_gap = abs(target - (totals[c1] + totals[c2]))
    return (d, util_gap)

# ----------------------------
# Agglomerative pass (generic)
# ----------------------------
def agglomerate_capacity(X, weights, target, edges, dthresh):
    """
    Start from singletons; merge adjacent clusters while:
      - centroid distance <= dthresh
      - combined weight <= target
    """
    n = len(X)
    # mapping from point -> current cluster id
    clusters = init_singletons(n)
    cid = list(range(n))  # cluster id per point
    alive = set(range(n))

    # precompute point centroids (initially themselves)
    centroids = {i: X[[i]].mean(axis=0) for i in range(n)}
    totals = {i: float(weights[i]) for i in range(n)}

    # adjacency on cluster ids (start as point adjacency)
    base_adj = adjacency_from_edges(edges, n)

    # build candidate heap: (distance, util_gap, a, b)
    heap = []
    seen_pairs = set()
    for a, b in edges:
        if a == b: continue
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs: 
            continue
        seen_pairs.add(pair)
        d = np.linalg.norm(centroids[a] - centroids[b])
        if d <= dthresh and (totals[a] + totals[b] <= target):
            util_gap = abs(target - (totals[a] + totals[b]))
            heapq.heappush(heap, (d, util_gap, a, b))

    # union-find via representative dict
    rep = {i: i for i in range(n)}
    def find(u):
        while rep[u] != u:
            rep[u] = rep[rep[u]]
            u = rep[u]
        return u
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return ra
        # attach rb -> ra
        rep[rb] = ra
        return ra

    # iterative merging
    while heap:
        d, util_gap, a, b = heapq.heappop(heap)
        ra, rb = find(a), find(b)
        if ra == rb: 
            continue
        # recompute feasibility with current totals/centroids
        if np.linalg.norm(centroids[ra] - centroids[rb]) > dthresh:
            continue
        if totals[ra] + totals[rb] > target:
            continue

        # merge rb into ra
        new_id = union(ra, rb)
        alive.discard(rb)
        # update members
        clusters[new_id] |= clusters[rb]
        clusters[rb].clear()

        # update totals/centroids
        totals[new_id] = totals[ra] + totals[rb]
        centroids[new_id] = cluster_centroid(X, list(clusters[new_id]))

        # push new candidate merges for neighbors of new_id
        neighbors = (base_adj[ra] | base_adj[rb])  # point-level adjacency
        # map neighbors to current reps
        neighbor_reps = set(find(u) for u in neighbors if u in alive)
        for nb in neighbor_reps:
            if nb == new_id: 
                continue
            pair = tuple(sorted((new_id, nb)))
            if pair in seen_pairs: 
                continue
            seen_pairs.add(pair)
            if totals[new_id] + totals[nb] <= target:
                d2 = np.linalg.norm(centroids[new_id] - centroids[nb])
                if d2 <= dthresh:
                    util_gap2 = abs(target - (totals[new_id] + totals[nb]))
                    heapq.heappush(heap, (d2, util_gap2, new_id, nb))

    # finalize clusters & labels
    labels = np.full(n, -1, dtype=int)
    idmap = {}
    next_id = 0
    final_clusters = []
    for i in range(n):
        ri = find(i)
        if ri not in idmap:
            idmap[ri] = next_id
            final_clusters.append(clusters[ri])
            next_id += 1
        labels[i] = idmap[ri]

    # compute totals list
    totals_list = [np.sum(weights[list(c)]) for c in final_clusters]
    return labels, final_clusters, totals_list

# ----------------------------
# Main
# ----------------------------
def main():
    # Load
    df = pd.read_csv(CSV_PATH).rename(columns={"LocationX":"Easting","LocationY":"Northing","Weighting":"Weighting"})
    df["Weighting"] = df["Weighting"].astype(float)
    # collapse exact duplicates
    df = df.groupby(["Easting","Northing"], as_index=False)["Weighting"].sum()

    X = df[["Easting","Northing"]].values
    w = df["Weighting"].values
    n = len(df)

    # Build adjacency
    if ADJACENCY_MODE == "mst":
        edges = build_mst_edges(X)    # strict contiguity
    else:
        edges = build_knn_edges(X, k=K_NEIGHBORS)  # default

    # PASS 1: ≤240
    labels_240, clusters_240, totals_240 = agglomerate_capacity(X, w, TARGET_1, edges, DTHRESH_240)
    violations_240 = [t for t in totals_240 if t > TARGET_1]
    print(f"\n=== Pass 1 (≤{TARGET_1}) ===")
    print(f"Clusters: {len(clusters_240)} | Violations: {len(violations_240)}")
    print(f"Utilization: mean={np.mean(totals_240):.2f}/{TARGET_1} ({np.mean(totals_240)/TARGET_1*100:.1f}%)")

    # PASS 2: ≤480 — adjacency merges of cluster reps
    # To reuse the same engine, approximate reps by taking cluster centroids and building a kNN graph on reps
    reps_centroids = np.array([cluster_centroid(X, list(c)) for c in clusters_240])
    reps_weights   = np.array(totals_240)
    if ADJACENCY_MODE == "mst":
        rep_edges = build_mst_edges(reps_centroids)
    else:
        rep_edges = build_knn_edges(reps_centroids, k=min(K_NEIGHBORS, len(reps_centroids)-1))

    labels_480_local, clusters_480_local, totals_480 = agglomerate_capacity(reps_centroids, reps_weights, TARGET_2, rep_edges, DTHRESH_480)

    # Map final 480 labels back to points
    # labels_240 -> a rep index r; labels_480_local[r] -> final 480 id
    final_480_ids = np.array([labels_480_local[lab240] for lab240 in labels_240])
    violations_480 = [t for t in totals_480 if t > TARGET_2]
    print(f"\n=== Pass 2 (≤{TARGET_2}) ===")
    print(f"Clusters: {len(totals_480)} | Violations: {len(violations_480)}")
    print(f"Utilization: mean={np.mean(totals_480):.2f}/{TARGET_2} ({np.mean(totals_480)/TARGET_2*100:.1f}%)")

    # Lineage table
    df_out = df.copy()
    df_out["label_240"] = labels_240
    df_out["label_480"] = final_480_ids
    df_out.to_csv("agglomerative_lineage.csv", index=False)

    # Visualization: merged ≤480 clusters
    unique_480 = np.unique(final_480_ids)
    cmap = cm.get_cmap("tab20", len(unique_480))
    colors = {c: cmap(i) for i, c in enumerate(unique_480)}

    plt.figure(figsize=(9,7))
    for c in unique_480:
        idx = np.where(final_480_ids == c)[0]
        plt.scatter(X[idx,0], X[idx,1], s=8, color=colors[c])
        # annotate centroid + total
        ccent = np.mean(X[idx], axis=0)
        ctot  = float(np.sum(w[idx]))
        plt.scatter([ccent[0]],[ccent[1]], marker='x', s=40, color=colors[c])
        plt.text(ccent[0], ccent[1], f"{ctot:.0f}", fontsize=8, ha='left', va='bottom')

    plt.xlabel("Easting"); plt.ylabel("Northing")
    plt.title(f"Agglomerative clustering with capacity (≤{TARGET_2})")
    format_axes_numeric(plt.gca(), thousands=True)
    plt.tight_layout()
    plt.savefig("agglomerative_pcu_480.png", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()