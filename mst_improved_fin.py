
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MST-Based Capacitated Clustering for Solar Tracker Rows (Zone-Aware)
====================================================================

Features:
- Preprocess points from row midpoints to road-end nearest the access road
- Build MST and cluster with capacity ≤240 (contiguity guaranteed)
- Zone-aware adjacency merges up to ≤480 *only* in 'dual' PCU zones
  • Greedy, target-hitting merge score favors ~480 and short centroid distances
  • Optional road-crossing rejection for merges
  • Detailed logging to console + CSV for traceability
- Subdivide final clusters to ≤16 for combiner/string management
- Validation, summary stats, and visualization outputs
- Exports a single CSV with assignments at all levels

Author: Padmasri SRINIVAS + Copilot
Date: 2025-12-05
"""

# ============================================================================
# Imports
# ============================================================================
import os
import math
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Point, LineString, Polygon
from shapely import ops as shapely_ops
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
# Input/Output
CSV_PATH      = "X_EL_SOLAR PANELS_2025-11-27_1354.csv"   # <-- replace with your input CSV
OUTPUT_CSV    = "mst_clustered_output.csv"                 # clusters export
MERGE_LOG_CSV = "merge_decisions_log.csv"                  # traceability log

# Capacity targets
TARGET_MAX     = 240   # first pass capacity
MERGE_MAX      = 480   # second pass capacity
SUBDIVIDE_MAX  = 16    # final subdivision capacity

# Road geometry (example) -- replace with site access road polyline
ROAD_GEOMETRY = LineString([
    (681000, 6407000),  # Start point
    (681600, 6408600)   # End point
])

# Row geometry assumptions
ROW_LENGTH     = 100
ROW_ORIENTATION = 'auto'   # 'auto' | 'horizontal' | 'vertical' | angle in degrees

# Zone polygons (placeholder examples) -- replace with real zones
ZONE_POLYGONS = {
    1: Polygon([(681000, 6407000), (681600, 6407000), (681600, 6407600), (681000, 6407600)]),
    2: Polygon([(681000, 6407600), (681600, 6407600), (681600, 6408600), (681000, 6408600)]),
}

# PCU mode per zone: 'single' (≤240 only) or 'dual' (merges to ≤480 allowed)
ZONE_MODES = {
    1: 'dual',
    2: 'single'
}

# Target-hitting merge tuning
MERGE_ALPHA_DIST   = 1.0   # weight on centroid distance
MERGE_BETA_TARGET  = 2.0   # weight on gap to 480
MERGE_MAX_DISTANCE = 150.0 # max centroid distance to consider merging (meters)
REJECT_ROAD_CROSS  = True  # reject merges where centroid segment crosses the road

# Plot settings
SAVE_PLOTS = True

# ============================================================================
# Utility: Row orientation & road-end adjustment
# ============================================================================
def estimate_row_orientation(df, sample_size=200):
    """Estimate predominant row orientation using nearest-neighbor angles."""
    from scipy.spatial.distance import cdist
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    X = sample[['Easting', 'Northing']].values

    d = cdist(X, X)
    np.fill_diagonal(d, np.inf)
    nearest = np.argmin(d, axis=1)

    angles = []
    for i, j in enumerate(nearest):
        dx = X[j, 0] - X[i, 0]
        dy = X[j, 1] - X[i, 1]
        ang = np.degrees(np.arctan2(dy, dx))
        angles.append(ang)

    mean_angle = np.mean(np.array(angles))
    if -30 <= mean_angle <= 30 or 150 <= abs(mean_angle) <= 180:
        return 'horizontal'
    elif 60 <= mean_angle <= 120 or -120 <= mean_angle <= -60:
        return 'vertical'
    else:
        return float(mean_angle)

def adjust_to_road_end(df, road_geometry, row_length=ROW_LENGTH, orientation='auto'):
    """Shift midpoints to the row-end nearest the access road."""
    df = df.copy()
    if orientation == 'auto':
        orientation = estimate_row_orientation(df)
        print(f"[PRE] Auto-detected row orientation: {orientation}")

    if orientation == 'horizontal':
        direction = np.array([1.0, 0.0])
    elif orientation == 'vertical':
        direction = np.array([0.0, 1.0])
    else:
        angle_rad = float(orientation) * np.pi / 180.0
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    adjusted = []
    half_len = row_length / 2.0

    for _, row in df.iterrows():
        mid = np.array([row['Easting'], row['Northing']])
        end1 = mid + direction * half_len
        end2 = mid - direction * half_len
        p1, p2 = Point(end1[0], end1[1]), Point(end2[0], end2[1])
        d1, d2 = p1.distance(road_geometry), p2.distance(road_geometry)
        chosen = end1 if d1 < d2 else end2
        adjusted.append(chosen)

    df['Easting']  = [p[0] for p in adjusted]
    df['Northing'] = [p[1] for p in adjusted]
    return df

# ============================================================================
# Zone assignment
# ============================================================================
def assign_zones_by_polygons(coordinates, zone_polygons):
    """Assign zone_id per node based on polygon containment; -1 if none."""
    zone_ids = []
    for (x, y) in coordinates:
        p = Point(x, y)
        assigned = -1
        for zid, poly in zone_polygons.items():
            if poly.contains(p):
                assigned = zid
                break
        zone_ids.append(assigned)
    return zone_ids

def cluster_zone_id(cluster_nodes, node_zone_ids):
    """Set cluster zone via majority of member nodes."""
    zones = [node_zone_ids[n] for n in cluster_nodes if node_zone_ids[n] != -1]
    if not zones:
        return -1
    # majority zone
    return max(set(zones), key=zones.count)

# ============================================================================
# MST construction
# ============================================================================
def build_mst(coordinates):
    """Build MST from coordinates; return as NetworkX graph with distance weights."""
    print("[MST] Computing distance matrix...")
    dist = squareform(pdist(coordinates, metric='euclidean'))

    print("[MST] Building minimum spanning tree (Prim's)...")
    mst_s = minimum_spanning_tree(dist)

    G = nx.Graph()
    for i in range(len(coordinates)):
        G.add_node(i, pos=(coordinates[i, 0], coordinates[i, 1]))

    coo = mst_s.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        G.add_edge(i, j, weight=float(w))

    print(f"[MST] Created graph with {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

# ============================================================================
# Pass 1: MST-based clustering ≤240
# ============================================================================
def mst_capacitated_clustering(G, weights, capacity=TARGET_MAX):
    """DFS traversal; cut when adding next node would exceed capacity."""
    print(f"[P1] Clustering with capacity ≤{capacity}...")
    clusters = []
    visited = set()
    node_to_cluster = {}

    for start in G.nodes():
        if start in visited:
            continue

        stack = [(start, None)]
        current = []
        current_w = 0.0
        cluster_id = len(clusters)

        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue

            w = float(weights[node])
            if current_w + w <= capacity:
                current.append(node)
                current_w += w
                visited.add(node)
                node_to_cluster[node] = cluster_id
                for nbr in G.neighbors(node):
                    if nbr not in visited:
                        stack.append((nbr, node))
            else:
                if current:
                    clusters.append({'nodes': current, 'weight': current_w})
                current   = [node]
                current_w = w
                visited.add(node)
                cluster_id = len(clusters)
                node_to_cluster[node] = cluster_id
                for nbr in G.neighbors(node):
                    if nbr not in visited:
                        stack.append((nbr, node))

        if current:
            clusters.append({'nodes': current, 'weight': current_w})

    print(f"[P1] Created {len(clusters)} clusters at ≤{capacity}")
    return clusters, node_to_cluster

# ============================================================================
# Helpers: centroids, scoring, road-crossing rejection
# ============================================================================
def centroid_of_nodes(coords, nodes):
    pts = coords[nodes]
    return np.mean(pts, axis=0)

def target_hitting_score(weight_a, weight_b, centroid_a, centroid_b,
                         alpha=MERGE_ALPHA_DIST, beta=MERGE_BETA_TARGET,
                         target=MERGE_MAX, max_dist=MERGE_MAX_DISTANCE):
    """
    Lower score is better. Penalize long distances and deviation from target.
    """
    dist = np.linalg.norm(centroid_a - centroid_b)
    if dist > max_dist:
        return np.inf
    combined = weight_a + weight_b
    if combined > target:
        return np.inf
    gap = abs(target - combined)
    return alpha * dist + beta * gap

def segment_crosses_road(ca, cb, road_line: LineString):
    """Check whether the centroid segment intersects the access road polyline."""
    seg = LineString([tuple(ca), tuple(cb)])
    return seg.crosses(road_line) or seg.intersects(road_line)

# ============================================================================
# Pass 2: Zone-aware, target-hitting adjacency merges ≤480
# ============================================================================
def merge_to_480_zone_aware(
    G, clusters_240, node_to_cluster_240, coordinates,
    node_zone_ids, zone_modes,
    target=MERGE_MAX, alpha=MERGE_ALPHA_DIST, beta=MERGE_BETA_TARGET,
    max_dist=MERGE_MAX_DISTANCE, reject_road_cross=REJECT_ROAD_CROSS,
    road_line=ROAD_GEOMETRY, merge_log_csv=MERGE_LOG_CSV
):
    """
    Build cluster adjacency from MST edges and greedily merge adjacent pairs
    ONLY in 'dual' zones, ranking candidates by target-hitting score.

    Returns:
      final_clusters, new_node_to_cluster
    """
    print(f"[P2] Zone-aware merging to ≤{target} (dual zones only)...")

    # Prepare logging CSV
    with open(merge_log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "merge_step", "cluster_a", "cluster_b",
            "weight_a_before", "weight_b_before",
            "combined_weight_after", "centroid_distance_m",
            "score", "zone_a", "zone_b", "road_cross_rejected"
        ])

    C = len(clusters_240)
    c_weight   = np.array([c['weight'] for c in clusters_240], dtype=float)
    c_nodes    = [list(c['nodes']) for c in clusters_240]
    c_centroid = [centroid_of_nodes(coordinates, c['nodes']) for c in clusters_240]
    c_zone     = [cluster_zone_id(c['nodes'], node_zone_ids) for c in clusters_240]

    # Build adjacency from MST edges
    c_adj = {i: set() for i in range(C)}
    for u, v in G.edges():
        cu, cv = node_to_cluster_240[u], node_to_cluster_240[v]
        if cu != cv:
            c_adj[cu].add(cv)
            c_adj[cv].add(cu)

    # Union-Find
    parent = {i: i for i in range(C)}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False

    def dual_zone(cluster_idx):
        zid = c_zone[cluster_idx]
        return (zid != -1) and (zone_modes.get(zid, 'single') == 'dual')

    step = 0
    improved = True
    while improved:
        improved = False
        best = (np.inf, None, None, None)  # score, a, b, info

        # Evaluate all adjacency pairs
        for i in range(C):
            ri = find(i)
            if not dual_zone(ri):
                continue
            for j in list(c_adj[i]):
                rj = find(j)
                if ri == rj or not dual_zone(rj):
                    continue

                wi, wj = c_weight[ri], c_weight[rj]
                ci, cj = c_centroid[ri], c_centroid[rj]

                # Road-cross rejection (optional)
                road_reject = False
                if reject_road_cross and segment_crosses_road(ci, cj, road_line):
                    road_reject = True
                    continue  # don't even score this pair

                s = target_hitting_score(wi, wj, ci, cj,
                                         alpha=alpha, beta=beta,
                                         target=target, max_dist=max_dist)
                if s < best[0]:
                    best = (s, ri, rj, {"wi": wi, "wj": wj, "ci": ci, "cj": cj, "road_reject": road_reject})

        if best[1] is not None:
            s, a, b, info = best
            combined = c_weight[a] + c_weight[b]
            if combined <= target and dual_zone(a) and dual_zone(b):
                # Perform merge
                union(a, b)
                # Update representative 'a'
                prev_a, prev_b = c_weight[a], c_weight[b]
                c_weight[a]    = combined
                c_nodes[a]    += c_nodes[b]
                c_centroid[a]  = centroid_of_nodes(coordinates, c_nodes[a])
                # Update adjacency
                c_adj[a] = (c_adj.get(a, set()) | c_adj.get(b, set())) - {a, b}

                # Logging
                step += 1
                dist_ab = np.linalg.norm(info["ci"] - info["cj"])
                with open(merge_log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        step, int(a), int(b),
                        f"{prev_a:.2f}", f"{prev_b:.2f}",
                        f"{combined:.2f}", f"{dist_ab:.2f}",
                        f"{s:.2f}", int(c_zone[a]), int(c_zone[b]),
                        info["road_reject"]
                    ])

                print(f"[P2] MERGE #{step}: C{a} + C{b} | "
                      f"w={combined:.2f} (prev {prev_a:.2f}+{prev_b:.2f}) "
                      f"dist={dist_ab:.1f}m score={s:.2f} zone={c_zone[a]}")

                improved = True

    # Build final clusters from representatives
    root_to_cluster = {}
    for i in range(C):
        r = find(i)
        if r not in root_to_cluster:
            root_to_cluster[r] = {'nodes': [], 'weight': 0.0}
        root_to_cluster[r]['nodes'] += clusters_240[i]['nodes']
        root_to_cluster[r]['weight'] += clusters_240[i]['weight']

    final_clusters = list(root_to_cluster.values())

    # Remap nodes → new cluster ids
    new_node_to_cluster = {}
    for new_id, cl in enumerate(final_clusters):
        for n in cl['nodes']:
            new_node_to_cluster[n] = new_id

    print(f"[P2] Zone-aware merging complete → {len(final_clusters)} clusters")
    print(f"[P2] Merge log written to: {merge_log_csv}")
    return final_clusters, new_node_to_cluster

# ============================================================================
# Pass 3: Subdivide to ≤16
# ============================================================================
def subdivide_clusters(clusters, coordinates, weights, max_sub=SUBDIVIDE_MAX):
    """Greedy spatial packing inside each cluster to meet ≤16 weight."""
    print(f"[P3] Subdividing clusters to ≤{max_sub}...")
    all_subclusters = []
    node_to_sub = {}

    for cid, cl in enumerate(clusters):
        nodes = cl['nodes']
        total_w = cl['weight']
        if total_w <= max_sub:
            # Already fine
            all_subclusters.append({'nodes': nodes, 'weight': total_w, 'parent': cid})
            for n in nodes:
                node_to_sub[n] = len(all_subclusters) - 1
            continue

        # Local centroid and greedy packing from center
        node_coords = coordinates[nodes]
        node_weights = weights[nodes]
        centroid = np.mean(node_coords, axis=0)
        dists = np.linalg.norm(node_coords - centroid, axis=1)
        order = np.argsort(dists)

        cur = []
        cur_w = 0.0
        for idx in order:
            n = nodes[idx]
            w = float(node_weights[idx])
            if cur_w + w > max_sub and cur:
                all_subclusters.append({'nodes': cur, 'weight': cur_w, 'parent': cid})
                for nn in cur:
                    node_to_sub[nn] = len(all_subclusters) - 1
                cur, cur_w = [], 0.0
            cur.append(n)
            cur_w += w

        if cur:
            all_subclusters.append({'nodes': cur, 'weight': cur_w, 'parent': cid})
            for nn in cur:
                node_to_sub[nn] = len(all_subclusters) - 1

    print(f"[P3] Created {len(all_subclusters)} subclusters")
    return all_subclusters, node_to_sub

# ============================================================================
# Validation & stats
# ============================================================================
def validate_clusters(clusters, max_capacity, label="Clusters"):
    weights = [c['weight'] for c in clusters]
    violations = [(i, w) for i, w in enumerate(weights) if w > max_capacity]

    print("\n" + "="*70)
    print(f"{label} - Validation Report")
    print("="*70)
    print(f"Total clusters: {len(clusters)}")
    print(f"Capacity limit: {max_capacity}")
    print(f"Violations: {len(violations)}")

    if violations:
        print("\nClusters exceeding capacity (first 10):")
        for i, w in violations[:10]:
            print(f"  Cluster {i}: {w:.2f} (excess: {w - max_capacity:.2f})")

    print("\nWeight distribution:")
    print(f"  Min:    {min(weights):.2f}")
    print(f"  Max:    {max(weights):.2f}")
    print(f"  Mean:   {np.mean(weights):.2f}")
    print(f"  Median: {np.median(weights):.2f}")
    print(f"  Total:  {sum(weights):.2f}")
    return violations

# ============================================================================
# Visualization (basic)
# ============================================================================
def plot_mst(G, coordinates, title="Minimum Spanning Tree", filename=None):
    plt.figure(figsize=(12, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=18, c='blue', alpha=0.6, zorder=2)
    for u, v in G.edges():
        x = [coordinates[u, 0], coordinates[v, 0]]
        y = [coordinates[u, 1], coordinates[v, 1]]
        plt.plot(x, y, 'k-', alpha=0.3, linewidth=0.5, zorder=1)
    plt.xlabel('Easting'); plt.ylabel('Northing'); plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def plot_clusters(coordinates, node_to_cluster, clusters, title, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    nC = len(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, min(nC, 20)))

    # Spatial
    for i, cl in enumerate(clusters):
        nodes = cl['nodes']; col = colors[i % 20]
        ax1.scatter(coordinates[nodes, 0], coordinates[nodes, 1], s=24, c=[col],
                    alpha=0.75, label=f"C{i} (w={cl['weight']:.0f})")
    ax1.set_xlabel('Easting'); ax1.set_ylabel('Northing')
    ax1.set_title(f'{title} - Spatial'); ax1.grid(True, alpha=0.3)

    # Weights
    ws = [c['weight'] for c in clusters]; ids = range(len(clusters))
    ax2.bar(ids, ws, color=colors[:len(clusters)])
    ax2.axhline(y=TARGET_MAX, color='orange', linestyle='--', linewidth=2, label=f'Target {TARGET_MAX}')
    ax2.axhline(y=MERGE_MAX,  color='red',    linestyle='--', linewidth=2, label=f'Max {MERGE_MAX}')
    ax2.set_xlabel('Cluster ID'); ax2.set_ylabel('Total Weight')
    ax2.set_title(f'{title} - Weight Dist'); ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison(X_original, X_adjusted, filename='mst_road_adjustment.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(X_original[:, 0], X_original[:, 1], s=10, alpha=0.6)
    ax1.set_title('Original Midpoints'); ax1.set_xlabel('Easting'); ax1.set_ylabel('Northing'); ax1.grid(True, alpha=0.3)
    ax2.scatter(X_adjusted[:, 0], X_adjusted[:, 1], s=10, alpha=0.6, color='green')
    ax2.set_title('Adjusted to Road Endpoints'); ax2.set_xlabel('Easting'); ax2.set_ylabel('Northing'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("MST-Based Capacitated Clustering (Zone-Aware, Target-Hitting Merges)")
    print("="*70)

    # Step 1: Load data
    print("\n[Step 1/8] Loading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"LocationX": "Easting", "LocationY": "Northing"})
    df["Weighting"] = df["Weighting"].astype(float)
    df = df.groupby(["Easting", "Northing"], as_index=False)["Weighting"].sum()
    print(f"  Loaded {len(df)} unique tracker rows")
    print(f"  Total weight: {df['Weighting'].sum():.2f}")

    X_original = df[["Easting", "Northing"]].values.copy()

    # Step 2: Road-end preprocessing
    print("\n[Step 2/8] Applying road-end preprocessing...")
    df_adj = adjust_to_road_end(df, ROAD_GEOMETRY, ROW_LENGTH, ROW_ORIENTATION)
    coordinates = df_adj[["Easting", "Northing"]].values
    weights     = df_adj["Weighting"].values
    plot_comparison(X_original, coordinates)

    # Step 3: Build MST
    print("\n[Step 3/8] Building MST...")
    G = build_mst(coordinates)
    plot_mst(G, coordinates, "Minimum Spanning Tree", "mst_tree.png")

    # Step 4: Pass 1 clustering (≤240)
    print("\n[Step 4/8] MST-based clustering (≤240)...")
    clusters_240, node_to_cluster_240 = mst_capacitated_clustering(G, weights, TARGET_MAX)
    validate_clusters(clusters_240, TARGET_MAX, "Pass 1 (≤240)")
    plot_clusters(coordinates, node_to_cluster_240, clusters_240, "Pass 1: MST Clusters ≤240", "mst_pass1_240.png")

    # Step 5: Zone assignment
    print("\n[Step 5/8] Assigning zones to nodes and clusters...")
    node_zone_ids = assign_zones_by_polygons(coordinates, ZONE_POLYGONS)
    # (Optional) print basic zone stats
    z_counts = pd.Series(node_zone_ids).value_counts(dropna=False)
    print("  Node zone counts:", dict(z_counts))

    # Step 6: Zone-aware merges to ≤480
    print("\n[Step 6/8] Zone-aware target-hitting merges (≤480)...")
    clusters_480, node_to_cluster_480 = merge_to_480_zone_aware(
        G, clusters_240, node_to_cluster_240, coordinates,
        node_zone_ids, ZONE_MODES,
        target=MERGE_MAX, alpha=MERGE_ALPHA_DIST, beta=MERGE_BETA_TARGET,
        max_dist=MERGE_MAX_DISTANCE, reject_road_cross=REJECT_ROAD_CROSS,
        road_line=ROAD_GEOMETRY, merge_log_csv=MERGE_LOG_CSV
    )
    validate_clusters(clusters_480, MERGE_MAX, "Pass 2 (≤480)")
    plot_clusters(coordinates, node_to_cluster_480, clusters_480,
                  "Pass 2: Zone-aware Merged Clusters ≤480", "mst_pass2_480.png")

    # Step 7: Subdivide to ≤16
    print("\n[Step 7/8] Subdividing clusters (≤16)...")
    subclusters, node_to_subcluster = subdivide_clusters(clusters_480, coordinates, weights, SUBDIVIDE_MAX)
    validate_clusters(subclusters, SUBDIVIDE_MAX, "Pass 3 (≤16)")
    plot_clusters(coordinates, node_to_subcluster, subclusters,
                  "Pass 3: Subdivided ≤16", "mst_pass3_16.png")

    # Step 8: Prepare & export CSV
    print("\n[Step 8/8] Preparing output data...")
    assign_240 = np.array([node_to_cluster_240[i] for i in range(len(coordinates))])
    assign_480 = np.array([node_to_cluster_480[i] for i in range(len(coordinates))])
    assign_sub = np.array([node_to_subcluster[i] for i in range(len(coordinates))])

    # parent cluster id for each subcluster
    parent_ids = np.array([subclusters[node_to_subcluster[i]]['parent'] for i in range(len(coordinates))])

    df_adj['Cluster_240']   = assign_240
    df_adj['Cluster_480']   = assign_480
    df_adj['Cluster_Parent'] = parent_ids
    df_adj['Cluster_Sub']   = assign_sub
    df_adj['Zone_Id']       = node_zone_ids

    df_adj.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*70)
    print("MST Clustering Complete!")
    print("="*70)
    print("Output files:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - mst_tree.png")
    print(f"  - mst_road_adjustment.png")
    print(f"  - mst_pass1_240.png")
    print(f"  - mst_pass2_480.png")
    print(f"  - mst_pass3_16.png")
    print(f"  - {MERGE_LOG_CSV}")

    print("\nFinal statistics:")
    print(f"  240-clusters: {len(clusters_240)}")
    print(f"  480-clusters: {len(clusters_480)}")
    print(f"  16-subclusters: {len(subclusters)}")
    print(f"  Total weight: {df_adj['Weighting'].sum():.2f}")

if __name__ == "__main__":
    main()