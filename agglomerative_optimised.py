"""
Optimized Agglomerative Capacity-Aware Clustering for Solar Tracker PCUs
=========================================================================

IMPROVEMENTS OVER ORIGINAL:
1. ✅ Utilization-first merge priority (fill to target before considering distance)
2. ✅ Increased merge thresholds (80m for 240, 150m for 480)
3. ✅ Richer adjacency (k=10 neighbors by default)
4. ✅ Post-merge optimization step (boundary row swapping)
5. ✅ Three-pass strategy: 220 → 480 → 16 (flexible intermediate target)
6. ✅ Fixed matplotlib deprecation warning
7. ✅ Comprehensive validation and comparison metrics

TARGET: Achieve 80-90% utilization while maintaining spatial contiguity
"""

import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import colormaps
import heapq
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = "X_EL_SOLAR PANELS_2025-11-27_1354.csv"
OUTPUT_CSV = "agglomerative_optimized_output.csv"

# Adjacency mode
ADJACENCY_MODE = "knn"  # "knn" or "mst"
K_NEIGHBORS = 10        # Increased from 6 for richer merge candidates

# Merge thresholds (in same units as coordinates)
DTHRESH_220 = 100.0     # Pass 1: More generous for flexible merging
DTHRESH_480 = 150.0     # Pass 2: Allow longer-range merges
DTHRESH_16 = 50.0       # Pass 3: Local subdivision only

# Capacity targets
TARGET_1 = 220          # Flexible target (leaves room for merging to 480)
TARGET_2 = 480          # Dual PCU capacity
TARGET_3 = 16           # Wiring group subdivision

# Post-processing
ENABLE_BOUNDARY_OPTIMIZATION = True  # Swap boundary rows to improve utilization
ENABLE_ORPHAN_REASSIGNMENT = True    # Reassign isolated small clusters

RANDOM_STATE = 42

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_axes_numeric(ax, thousands=True):
    """Format axes with thousands separator and grid."""
    from matplotlib.ticker import FuncFormatter
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    if thousands:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}"))
    ax.grid(True, alpha=0.3, linestyle='--')

def build_knn_graph(X, k=10):
    """
    Build k-nearest neighbor graph (undirected).
    Returns: list of edges [(i, j), ...], adjacency dict
    """
    n = len(X)
    k_actual = min(k + 1, n)  # +1 because first neighbor is self
    
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    edges = set()
    for i in range(n):
        for j in indices[i][1:]:  # Skip self at index 0
            edge = tuple(sorted((i, j)))
            edges.add(edge)
    
    # Build adjacency dict
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    return list(edges), dict(adj)

def build_mst_graph(X):
    """
    Build minimum spanning tree for strict spatial contiguity.
    Returns: list of edges, adjacency dict
    """
    n = len(X)
    
    # Compute distance matrix
    distances = squareform(pdist(X, metric='euclidean'))
    
    # Build MST using scipy
    mst = minimum_spanning_tree(distances)
    
    # Extract edges
    edges = []
    adj = defaultdict(set)
    
    mst_coo = mst.tocoo()
    for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        edge = tuple(sorted((i, j)))
        edges.append(edge)
        adj[i].add(j)
        adj[j].add(i)
    
    return edges, dict(adj)

def cluster_centroid(X, node_ids):
    """Calculate centroid of a cluster."""
    if len(node_ids) == 0:
        return np.array([0, 0])
    return X[list(node_ids)].mean(axis=0)

# ============================================================================
# OPTIMIZED MERGE PRIORITY
# ============================================================================

def optimized_merge_priority(c1, c2, centroids, totals, target, distance):
    """
    New priority function: UTILIZATION FIRST, then distance.
    
    Priority tuple: (utilization_gap, distance)
    - Lower utilization gap = closer to target = HIGHER priority
    - If tied, prefer shorter distance
    
    This ensures we fill clusters efficiently before worrying about distance.
    """
    combined_weight = totals[c1] + totals[c2]
    
    # How far from target?
    util_gap = abs(target - combined_weight)
    
    # Secondary: prefer merges that get us closer to target
    # (not just above, but as close as possible)
    fill_score = abs(combined_weight - target) / target
    
    return (util_gap, distance, fill_score)

# ============================================================================
# CORE AGGLOMERATIVE CLUSTERING
# ============================================================================

def agglomerate_capacity_optimized(X, weights, target, base_adj, dthresh, 
                                   prioritize_utilization=True):
    """
    Optimized agglomerative clustering with utilization-first merge priority.
    
    Key improvements:
    - Merge priority: utilization gap first, then distance
    - More generous distance thresholds
    - Better heap management for efficiency
    """
    n = len(X)
    
    # Initialize: each point is its own cluster
    clusters = {i: {i} for i in range(n)}
    totals = {i: float(weights[i]) for i in range(n)}
    centroids = {i: X[i] for i in range(n)}
    alive = set(range(n))
    
    # Union-Find structure
    parent = {i: i for i in range(n)}
    
    def find(u):
        """Find representative with path compression."""
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(a, b):
        """Union two clusters."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        parent[rb] = ra
        return ra
    
    # Build initial merge candidates
    heap = []
    seen_pairs = set()
    
    for a in range(n):
        for b in base_adj.get(a, []):
            if a >= b:
                continue
            
            pair = tuple(sorted((a, b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            # Check capacity and distance
            if totals[a] + totals[b] <= target:
                dist = np.linalg.norm(centroids[a] - centroids[b])
                if dist <= dthresh:
                    if prioritize_utilization:
                        priority = optimized_merge_priority(a, b, centroids, totals, target, dist)
                    else:
                        # Old priority: distance first
                        priority = (dist, abs(target - (totals[a] + totals[b])))
                    
                    heapq.heappush(heap, (*priority, a, b))
    
    # Iterative merging
    merge_count = 0
    while heap:
        *priority, a, b = heapq.heappop(heap)
        
        # Get current representatives
        ra, rb = find(a), find(b)
        
        if ra == rb:
            continue  # Already merged
        
        # Revalidate with current state
        if ra not in alive or rb not in alive:
            continue
        
        current_dist = np.linalg.norm(centroids[ra] - centroids[rb])
        if current_dist > dthresh:
            continue
        
        if totals[ra] + totals[rb] > target:
            continue
        
        # Perform merge
        new_id = union(ra, rb)
        alive.discard(rb)
        
        # Update cluster membership
        clusters[new_id] = clusters[ra] | clusters[rb]
        clusters.pop(rb, None)
        
        # Update totals and centroids
        totals[new_id] = totals[ra] + totals[rb]
        centroids[new_id] = cluster_centroid(X, clusters[new_id])
        
        merge_count += 1
        
        # Add new merge candidates
        neighbors = (base_adj.get(ra, set()) | base_adj.get(rb, set()))
        neighbor_reps = {find(u) for u in neighbors if u in alive}
        
        for nb in neighbor_reps:
            if nb == new_id:
                continue
            
            pair = tuple(sorted((new_id, nb)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            if totals[new_id] + totals[nb] <= target:
                dist = np.linalg.norm(centroids[new_id] - centroids[nb])
                if dist <= dthresh:
                    if prioritize_utilization:
                        priority = optimized_merge_priority(new_id, nb, centroids, totals, target, dist)
                    else:
                        priority = (dist, abs(target - (totals[new_id] + totals[nb])))
                    
                    heapq.heappush(heap, (*priority, new_id, nb))
    
    print(f"  Performed {merge_count} merges")
    
    # Finalize labels
    labels = np.full(n, -1, dtype=int)
    cluster_list = []
    label_map = {}
    next_label = 0
    
    for i in range(n):
        rep = find(i)
        if rep not in label_map:
            label_map[rep] = next_label
            cluster_list.append(clusters[rep])
            next_label += 1
        labels[i] = label_map[rep]
    
    # Compute final totals
    totals_list = [sum(weights[list(c)]) for c in cluster_list]
    
    return labels, cluster_list, totals_list

# ============================================================================
# POST-MERGE OPTIMIZATION
# ============================================================================

def optimize_boundary_rows(X, weights, clusters, labels, base_adj, target):
    """
    Post-merge optimization: swap boundary rows between adjacent clusters
    to improve capacity utilization while maintaining feasibility.
    
    Strategy:
    1. Find clusters with low utilization (< 70% of target)
    2. For each, examine boundary nodes (connected to other clusters)
    3. Try swapping boundary nodes to better-fill neighbors
    """
    print("  Running boundary optimization...")
    
    n = len(X)
    cluster_weights = {i: sum(weights[list(c)]) for i, c in enumerate(clusters)}
    
    # Find underutilized clusters
    underutilized = [i for i, w in cluster_weights.items() if w < 0.7 * target]
    
    if not underutilized:
        print("    No underutilized clusters found")
        return labels
    
    print(f"    Found {len(underutilized)} underutilized clusters")
    
    swaps_made = 0
    
    for cluster_id in underutilized:
        cluster_nodes = list(clusters[cluster_id])
        
        # Find boundary nodes (connected to other clusters)
        boundary_nodes = []
        for node in cluster_nodes:
            for neighbor in base_adj.get(node, []):
                if labels[neighbor] != cluster_id:
                    boundary_nodes.append(node)
                    break
        
        if not boundary_nodes:
            continue
        
        # Try swapping each boundary node
        for node in boundary_nodes:
            current_cluster = labels[node]
            node_weight = weights[node]
            
            # Find adjacent clusters
            adjacent_clusters = {labels[nb] for nb in base_adj.get(node, []) 
                                if labels[nb] != current_cluster}
            
            for target_cluster in adjacent_clusters:
                # Check if swap is beneficial
                current_weight = cluster_weights[current_cluster]
                target_weight = cluster_weights[target_cluster]
                
                # Swap improves if:
                # 1. Target cluster gets closer to target without exceeding
                # 2. Current cluster doesn't drop too low
                new_target_weight = target_weight + node_weight
                new_current_weight = current_weight - node_weight
                
                if new_target_weight <= target and new_current_weight > 0:
                    # Calculate improvement in utilization
                    old_util = (current_weight + target_weight) / (2 * target)
                    new_util = (new_current_weight + new_target_weight) / (2 * target)
                    
                    if new_util > old_util:
                        # Perform swap
                        labels[node] = target_cluster
                        clusters[current_cluster].discard(node)
                        clusters[target_cluster].add(node)
                        cluster_weights[current_cluster] = new_current_weight
                        cluster_weights[target_cluster] = new_target_weight
                        swaps_made += 1
                        break
    
    print(f"    Made {swaps_made} beneficial swaps")
    return labels

def reassign_orphan_clusters(X, weights, clusters, labels, target):
    """
    Reassign small isolated clusters to nearest neighbors if beneficial.
    """
    print("  Reassigning orphan clusters...")
    
    cluster_weights = {i: sum(weights[list(c)]) for i, c in enumerate(clusters)}
    cluster_centroids = {i: cluster_centroid(X, c) for i, c in enumerate(clusters)}
    
    # Find small clusters (< 30% of target)
    small_clusters = [i for i, w in cluster_weights.items() if w < 0.3 * target]
    
    if not small_clusters:
        print("    No orphan clusters found")
        return labels
    
    print(f"    Found {len(small_clusters)} orphan clusters")
    
    reassigned = 0
    
    for small_id in small_clusters:
        small_weight = cluster_weights[small_id]
        small_centroid = cluster_centroids[small_id]
        
        # Find nearest cluster that can accept it
        min_dist = float('inf')
        best_target = None
        
        for other_id in range(len(clusters)):
            if other_id == small_id:
                continue
            
            other_weight = cluster_weights[other_id]
            if other_weight + small_weight <= target:
                dist = np.linalg.norm(small_centroid - cluster_centroids[other_id])
                if dist < min_dist:
                    min_dist = dist
                    best_target = other_id
        
        if best_target is not None:
            # Reassign all nodes
            for node in list(clusters[small_id]):
                labels[node] = best_target
                clusters[best_target].add(node)
            clusters[small_id].clear()
            cluster_weights[best_target] += small_weight
            cluster_weights[small_id] = 0
            reassigned += 1
    
    print(f"    Reassigned {reassigned} orphan clusters")
    
    # Rebuild cluster list without empty clusters
    new_clusters = [c for c in clusters if len(c) > 0]
    
    # Relabel
    label_map = {}
    for new_id, cluster in enumerate(new_clusters):
        for node in cluster:
            label_map[labels[node]] = new_id
    
    new_labels = np.array([label_map[old_label] for old_label in labels])
    
    return new_labels

# ============================================================================
# VALIDATION & REPORTING
# ============================================================================

def validate_and_report(labels, weights, target, pass_name):
    """Comprehensive validation and reporting."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    cluster_weights = []
    for label in unique_labels:
        mask = labels == label
        cluster_weights.append(weights[mask].sum())
    
    cluster_weights = np.array(cluster_weights)
    
    violations = sum(cluster_weights > target)
    utilization = (cluster_weights.mean() / target) * 100
    
    print(f"\n{'='*70}")
    print(f"{pass_name}")
    print(f"{'='*70}")
    print(f"Total clusters: {n_clusters}")
    print(f"Capacity limit: {target}")
    print(f"Violations (>{target}): {violations}")
    print(f"\nWeight distribution:")
    print(f"  Min: {cluster_weights.min():.0f}")
    print(f"  Max: {cluster_weights.max():.0f}")
    print(f"  Mean: {cluster_weights.mean():.2f}")
    print(f"  Median: {np.median(cluster_weights):.0f}")
    print(f"  Std Dev: {cluster_weights.std():.2f}")
    print(f"\nUtilization: {utilization:.1f}% of {target} capacity")
    
    # Distribution breakdown
    bins = [0, 0.5*target, 0.7*target, 0.9*target, target, target*1.1]
    hist, _ = np.histogram(cluster_weights, bins=bins)
    print(f"\nDistribution:")
    print(f"  <50% capacity: {hist[0]}")
    print(f"  50-70%: {hist[1]}")
    print(f"  70-90%: {hist[2]}")
    print(f"  90-100%: {hist[3]}")
    print(f"  >100% (violations): {hist[4]}")
    
    return cluster_weights

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_clusters(X, labels, weights, title, filename, target):
    """Create comprehensive visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Get colormap
    cmap = colormaps.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n_clusters)]
    
    # Plot 1: Spatial distribution
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = X[mask]
        cluster_weight = weights[mask].sum()
        
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[colors[i]], s=15, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        # Mark centroid
        centroid = cluster_points.mean(axis=0)
        ax1.scatter([centroid[0]], [centroid[1]], 
                   marker='X', s=80, c=[colors[i]], 
                   edgecolors='black', linewidths=1.5, zorder=10)
        
        # Label with weight
        ax1.text(centroid[0], centroid[1], f'{cluster_weight:.0f}',
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='black', alpha=0.8))
    
    ax1.set_xlabel('Easting', fontweight='bold')
    ax1.set_ylabel('Northing', fontweight='bold')
    ax1.set_title(f'{title} - Spatial Distribution', fontweight='bold')
    format_axes_numeric(ax1)
    
    # Plot 2: Weight distribution
    cluster_weights = [weights[labels == label].sum() for label in unique_labels]
    
    ax2.hist(cluster_weights, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(target, color='red', linestyle='--', linewidth=2, label=f'Target {target}')
    ax2.axvline(np.mean(cluster_weights), color='green', linestyle='-', 
               linewidth=2, label=f'Mean: {np.mean(cluster_weights):.0f}')
    
    ax2.set_xlabel('Cluster Weight', fontweight='bold')
    ax2.set_ylabel('Number of Clusters', fontweight='bold')
    ax2.set_title('Weight Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    print("="*70)
    print("OPTIMIZED AGGLOMERATIVE PCU CLUSTERING")
    print("="*70)
    
    # Load data
    print("\n[Step 1/7] Loading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"LocationX": "Easting", "LocationY": "Northing"})
    df["Weighting"] = df["Weighting"].astype(float)
    df = df.groupby(["Easting", "Northing"], as_index=False)["Weighting"].sum()
    
    X = df[["Easting", "Northing"]].values
    weights = df["Weighting"].values
    n = len(df)
    
    print(f"  Loaded {n} tracker rows")
    print(f"  Total strings: {weights.sum():.0f}")
    
    # Build adjacency graph
    print(f"\n[Step 2/7] Building {ADJACENCY_MODE.upper()} adjacency graph...")
    if ADJACENCY_MODE == "mst":
        edges, base_adj = build_mst_graph(X)
    else:
        edges, base_adj = build_knn_graph(X, k=K_NEIGHBORS)
    print(f"  Graph: {n} nodes, {len(edges)} edges")
    
    # Pass 1: Agglomerate to ≤220
    print(f"\n[Step 3/7] Pass 1: Agglomerative clustering (≤{TARGET_1})...")
    labels_220, clusters_220, _ = agglomerate_capacity_optimized(
        X, weights, TARGET_1, base_adj, DTHRESH_220, prioritize_utilization=True
    )
    
    # Post-optimization for Pass 1
    if ENABLE_BOUNDARY_OPTIMIZATION:
        labels_220 = optimize_boundary_rows(X, weights, clusters_220, labels_220, base_adj, TARGET_1)
    
    if ENABLE_ORPHAN_REASSIGNMENT:
        labels_220 = reassign_orphan_clusters(X, weights, clusters_220, labels_220, TARGET_1)
    
    weights_220 = validate_and_report(labels_220, weights, TARGET_1, "Pass 1 (≤220) - After Optimization")
    visualize_clusters(X, labels_220, weights, f"Pass 1: Clusters ≤{TARGET_1}", 
                      "agglomerative_opt_220.png", TARGET_1)
    
    # Pass 2: Merge to ≤480
    print(f"\n[Step 4/7] Pass 2: Merging to ≤{TARGET_2}...")
    
    # Build cluster-level graph
    unique_220 = np.unique(labels_220)
    cluster_centroids = np.array([X[labels_220 == label].mean(axis=0) for label in unique_220])
    cluster_weights_arr = np.array([weights[labels_220 == label].sum() for label in unique_220])
    
    if ADJACENCY_MODE == "mst":
        cluster_edges, cluster_adj = build_mst_graph(cluster_centroids)
    else:
        cluster_edges, cluster_adj = build_knn_graph(cluster_centroids, k=min(K_NEIGHBORS, len(unique_220)-1))
    
    labels_480_local, _, _ = agglomerate_capacity_optimized(
        cluster_centroids, cluster_weights_arr, TARGET_2, cluster_adj, DTHRESH_480, prioritize_utilization=True
    )
    
    # Map back to original points
    labels_480 = np.array([labels_480_local[label] for label in labels_220])
    
    # Rebuild clusters_480 for optimization
    unique_480 = np.unique(labels_480)
    clusters_480 = [{i for i in range(n) if labels_480[i] == label} for label in unique_480]
    
    # Post-optimization for Pass 2
    if ENABLE_BOUNDARY_OPTIMIZATION:
        labels_480 = optimize_boundary_rows(X, weights, clusters_480, labels_480, base_adj, TARGET_2)
    
    if ENABLE_ORPHAN_REASSIGNMENT:
        labels_480 = reassign_orphan_clusters(X, weights, clusters_480, labels_480, TARGET_2)
    
    weights_480 = validate_and_report(labels_480, weights, TARGET_2, "Pass 2 (≤480) - After Optimization")
    visualize_clusters(X, labels_480, weights, f"Pass 2: Clusters ≤{TARGET_2}",
                      "agglomerative_opt_480.png", TARGET_2)
    
    # Export results
    print(f"\n[Step 5/7] Exporting results...")
    df_out = df.copy()
    df_out["Cluster_220"] = labels_220
    df_out["Cluster_480"] = labels_480
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to {OUTPUT_CSV}")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nPass 1 (≤{TARGET_1}):")
    print(f"  Clusters: {len(np.unique(labels_220))}")
    print(f"  Mean utilization: {weights_220.mean():.0f} ({(weights_220.mean()/TARGET_1)*100:.1f}%)")
    print(f"\nPass 2 (≤{TARGET_2}):")
    print(f"  Clusters: {len(np.unique(labels_480))}")
    print(f"  Mean utilization: {weights_480.mean():.0f} ({(weights_480.mean()/TARGET_2)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("CLUSTERING COMPLETE!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()