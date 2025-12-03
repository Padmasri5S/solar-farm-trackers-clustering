"""
MST-Based Capacitated Clustering for Solar Tracker Rows
========================================================

Approach:
1. Build Minimum Spanning Tree (MST) over all tracker locations
2. Cut MST edges to create capacity-constrained clusters (≤240)
3. Merge adjacent clusters up to 480 capacity
4. Subdivide clusters to ≤16 for final grouping

Advantages:
- Guarantees spatial contiguity
- Naturally separates across large gaps (roads, boundaries)
- Fast and deterministic
- Perfect for linear/chain-like solar farm layouts

Author: Claude + User
"""

import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Point, LineString
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = "X_EL_SOLAR PANELS_2025-11-27_1354.csv"
TARGET_MAX = 240        # First pass capacity
MERGE_MAX = 480         # Second pass capacity  
SUBDIVIDE_MAX = 16      # Final subdivision capacity
OUTPUT_CSV = "mst_clustered_output.csv"

# Road geometry - MODIFY THIS based on your site
ROAD_GEOMETRY = LineString([
    (681000, 6407000),  # Start point
    (681600, 6408600)   # End point
])

# Row geometry
ROW_LENGTH = 100
ROW_ORIENTATION = 'auto'

# ============================================================================
# ROAD GEOMETRY PREPROCESSING
# ============================================================================

def estimate_row_orientation(df, sample_size=100):
    """Estimate predominant row orientation."""
    from scipy.spatial.distance import cdist
    
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    X = sample[['Easting', 'Northing']].values
    
    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)
    nearest_idx = np.argmin(dists, axis=1)
    
    angles = []
    for i, j in enumerate(nearest_idx):
        dx = X[j, 0] - X[i, 0]
        dy = X[j, 1] - X[i, 1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angles.append(angle)
    
    angles = np.array(angles)
    mean_angle = np.mean(angles)
    
    if -30 <= mean_angle <= 30 or 150 <= abs(mean_angle) <= 180:
        return 'horizontal'
    elif 60 <= mean_angle <= 120 or -120 <= mean_angle <= -60:
        return 'vertical'
    else:
        return mean_angle

def adjust_to_road_end(df, road_geometry, row_length=ROW_LENGTH, orientation='auto'):
    """Adjust midpoints to row end nearest to access road."""
    df = df.copy()
    
    if orientation == 'auto':
        orientation = estimate_row_orientation(df)
        print(f"Auto-detected row orientation: {orientation}")
    
    if orientation == 'horizontal':
        direction = np.array([1, 0])
    elif orientation == 'vertical':
        direction = np.array([0, 1])
    else:
        angle_rad = float(orientation) * np.pi / 180
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    adjusted_points = []
    
    for idx, row in df.iterrows():
        half_length = row_length / 2
        end1 = Point(
            row['Easting'] + direction[0] * half_length,
            row['Northing'] + direction[1] * half_length
        )
        end2 = Point(
            row['Easting'] - direction[0] * half_length,
            row['Northing'] - direction[1] * half_length
        )
        
        dist1 = end1.distance(road_geometry)
        dist2 = end2.distance(road_geometry)
        
        if dist1 < dist2:
            adjusted_points.append((end1.x, end1.y))
        else:
            adjusted_points.append((end2.x, end2.y))
    
    df['Easting'] = [p[0] for p in adjusted_points]
    df['Northing'] = [p[1] for p in adjusted_points]
    
    return df

# ============================================================================
# MST-BASED CLUSTERING
# ============================================================================

def build_mst(coordinates):
    """
    Build Minimum Spanning Tree from coordinates.
    Returns: NetworkX graph with edge weights = distances
    """
    n = len(coordinates)
    
    # Compute pairwise distances
    print("  Computing distance matrix...")
    distances = squareform(pdist(coordinates, metric='euclidean'))
    
    # Build MST using scipy (Prim's algorithm)
    print("  Building MST...")
    mst_scipy = minimum_spanning_tree(distances)
    
    # Convert to NetworkX for easier manipulation
    G = nx.Graph()
    
    # Add all nodes
    for i in range(n):
        G.add_node(i, pos=coordinates[i])
    
    # Add edges from MST
    mst_coo = mst_scipy.tocoo()
    for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        G.add_edge(i, j, weight=weight)
    
    print(f"  MST created: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

def mst_capacitated_clustering(G, weights, capacity=TARGET_MAX):
    """
    Cluster nodes by traversing MST and cutting when capacity exceeded.
    
    Algorithm:
    1. Start from arbitrary root
    2. Do DFS traversal, accumulating weights
    3. When adding next node would exceed capacity, cut that edge
    4. Continue with new cluster from cut node
    
    Returns: list of clusters (each cluster is list of node indices)
    """
    print(f"  Clustering with capacity {capacity}...")
    
    clusters = []
    visited = set()
    node_to_cluster = {}
    
    # Process each connected component (though MST should be connected)
    for start_node in G.nodes():
        if start_node in visited:
            continue
        
        # DFS with capacity tracking
        stack = [(start_node, None)]  # (node, parent)
        current_cluster = []
        current_weight = 0
        cluster_id = len(clusters)
        
        while stack:
            node, parent = stack.pop()
            
            if node in visited:
                continue
            
            node_weight = weights[node]
            
            # Check if we can add this node to current cluster
            if current_weight + node_weight <= capacity:
                # Add to current cluster
                current_cluster.append(node)
                current_weight += node_weight
                visited.add(node)
                node_to_cluster[node] = cluster_id
                
                # Add unvisited neighbors to stack
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        stack.append((neighbor, node))
            else:
                # Cannot add - start new cluster
                if current_cluster:
                    clusters.append({
                        'nodes': current_cluster,
                        'weight': current_weight
                    })
                
                # Start new cluster with this node
                current_cluster = [node]
                current_weight = node_weight
                visited.add(node)
                cluster_id = len(clusters)
                node_to_cluster[node] = cluster_id
                
                # Add neighbors
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        stack.append((neighbor, node))
        
        # Add final cluster
        if current_cluster:
            clusters.append({
                'nodes': current_cluster,
                'weight': current_weight
            })
    
    print(f"  Created {len(clusters)} clusters")
    return clusters, node_to_cluster

def merge_adjacent_clusters(G, clusters, node_to_cluster, weights, max_capacity=MERGE_MAX):
    """
    Merge adjacent clusters (connected in original MST) up to max_capacity.
    
    Two clusters are adjacent if there exists an edge in MST connecting them.
    """
    print(f"  Merging adjacent clusters up to capacity {max_capacity}...")
    
    # Build cluster adjacency graph
    cluster_graph = nx.Graph()
    for i in range(len(clusters)):
        cluster_graph.add_node(i, weight=clusters[i]['weight'])
    
    # Find edges between clusters in original MST
    for u, v in G.edges():
        cluster_u = node_to_cluster[u]
        cluster_v = node_to_cluster[v]
        
        if cluster_u != cluster_v:
            # Edge connects different clusters
            if not cluster_graph.has_edge(cluster_u, cluster_v):
                cluster_graph.add_edge(cluster_u, cluster_v)
    
    # Greedy merging: sort cluster pairs by combined weight (prefer balanced merges)
    merge_candidates = []
    for u, v in cluster_graph.edges():
        weight_u = cluster_graph.nodes[u]['weight']
        weight_v = cluster_graph.nodes[v]['weight']
        combined = weight_u + weight_v
        
        if combined <= max_capacity:
            # Prefer merging smaller clusters first to reach target
            merge_candidates.append((combined, u, v))
    
    merge_candidates.sort()
    
    # Union-Find structure for tracking merges
    parent = {i: i for i in range(len(clusters))}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
            return True
        return False
    
    # Track cluster weights after merges
    cluster_weights = {i: clusters[i]['weight'] for i in range(len(clusters))}
    
    merged_count = 0
    for combined_weight, u, v in merge_candidates:
        pu, pv = find(u), find(v)
        
        if pu == pv:
            continue  # Already merged
        
        # Check if merge is still valid (weights may have changed)
        if cluster_weights[pu] + cluster_weights[pv] <= max_capacity:
            union(u, v)
            cluster_weights[pu] += cluster_weights[pv]
            merged_count += 1
    
    # Build final merged clusters
    merged_clusters = {}
    for i, cluster in enumerate(clusters):
        root = find(i)
        if root not in merged_clusters:
            merged_clusters[root] = {
                'nodes': [],
                'weight': 0
            }
        merged_clusters[root]['nodes'].extend(cluster['nodes'])
        merged_clusters[root]['weight'] += cluster['weight']
    
    final_clusters = list(merged_clusters.values())
    
    # Update node_to_cluster mapping
    new_node_to_cluster = {}
    for new_id, cluster in enumerate(final_clusters):
        for node in cluster['nodes']:
            new_node_to_cluster[node] = new_id
    
    print(f"  Merged {merged_count} pairs → {len(final_clusters)} clusters")
    return final_clusters, new_node_to_cluster

def subdivide_clusters(clusters, coordinates, weights, max_sub=SUBDIVIDE_MAX):
    """
    Subdivide each cluster into subclusters with weight ≤ max_sub.
    Uses greedy spatial packing within each cluster.
    """
    print(f"  Subdividing clusters to ≤{max_sub}...")
    
    all_subclusters = []
    node_to_subcluster = {}
    
    for cluster_id, cluster in enumerate(clusters):
        nodes = cluster['nodes']
        
        if cluster['weight'] <= max_sub:
            # No subdivision needed
            all_subclusters.append({
                'nodes': nodes,
                'weight': cluster['weight'],
                'parent': cluster_id
            })
            for node in nodes:
                node_to_subcluster[node] = len(all_subclusters) - 1
        else:
            # Need to subdivide
            node_coords = coordinates[nodes]
            node_weights = weights[nodes]
            
            # Calculate local centroid
            centroid = np.mean(node_coords, axis=0)
            
            # Sort by distance to centroid
            distances = np.linalg.norm(node_coords - centroid, axis=1)
            sorted_indices = np.argsort(distances)
            
            # Greedy packing
            current_sub = []
            current_weight = 0
            
            for idx in sorted_indices:
                node = nodes[idx]
                w = node_weights[idx]
                
                if current_weight + w > max_sub and current_sub:
                    # Save current subcluster and start new one
                    all_subclusters.append({
                        'nodes': current_sub,
                        'weight': current_weight,
                        'parent': cluster_id
                    })
                    for n in current_sub:
                        node_to_subcluster[n] = len(all_subclusters) - 1
                    
                    current_sub = []
                    current_weight = 0
                
                current_sub.append(node)
                current_weight += w
            
            # Add final subcluster
            if current_sub:
                all_subclusters.append({
                    'nodes': current_sub,
                    'weight': current_weight,
                    'parent': cluster_id
                })
                for n in current_sub:
                    node_to_subcluster[n] = len(all_subclusters) - 1
    
    print(f"  Created {len(all_subclusters)} subclusters")
    return all_subclusters, node_to_subcluster

# ============================================================================
# VALIDATION & REPORTING
# ============================================================================

def validate_clusters(clusters, max_capacity, label="Clusters"):
    """Validate and report cluster statistics."""
    weights = [c['weight'] for c in clusters]
    violations = [(i, w) for i, w in enumerate(weights) if w > max_capacity]
    
    print(f"\n{'='*70}")
    print(f"{label} - Validation Report")
    print(f"{'='*70}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Capacity limit: {max_capacity}")
    print(f"Violations: {len(violations)}")
    
    if violations:
        print("\nClusters exceeding capacity:")
        for i, w in violations[:10]:  # Show first 10
            print(f"  Cluster {i}: {w:.2f} (excess: {w - max_capacity:.2f})")
    
    print(f"\nWeight distribution:")
    print(f"  Min: {min(weights):.2f}")
    print(f"  Max: {max(weights):.2f}")
    print(f"  Mean: {np.mean(weights):.2f}")
    print(f"  Median: {np.median(weights):.2f}")
    print(f"  Total: {sum(weights):.2f}")
    
    return violations

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_mst(G, coordinates, title="Minimum Spanning Tree", filename=None):
    """Visualize the MST."""
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=20, c='blue', alpha=0.6, zorder=2)
    
    # Draw edges
    for u, v in G.edges():
        x = [coordinates[u, 0], coordinates[v, 0]]
        y = [coordinates[u, 1], coordinates[v, 1]]
        plt.plot(x, y, 'k-', alpha=0.3, linewidth=0.5, zorder=1)
    
    plt.xlabel('Easting', fontsize=12)
    plt.ylabel('Northing', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def plot_clusters(coordinates, node_to_cluster, clusters, title, filename=None):
    """Visualize clusters with colors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Spatial distribution
    n_clusters = len(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    
    for i, cluster in enumerate(clusters):
        nodes = cluster['nodes']
        color = colors[i % 20]
        ax1.scatter(coordinates[nodes, 0], coordinates[nodes, 1],
                   s=30, c=[color], alpha=0.7, label=f"C{i} (w={cluster['weight']:.0f})")
    
    ax1.set_xlabel('Easting', fontsize=12)
    ax1.set_ylabel('Northing', fontsize=12)
    ax1.set_title(f'{title} - Spatial Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight distribution
    weights = [c['weight'] for c in clusters]
    cluster_ids = range(len(clusters))
    
    bars = ax2.bar(cluster_ids, weights, color=colors[:len(clusters)])
    ax2.axhline(y=TARGET_MAX, color='orange', linestyle='--', 
                linewidth=2, label=f'Target {TARGET_MAX}')
    ax2.axhline(y=MERGE_MAX, color='red', linestyle='--', 
                linewidth=2, label=f'Max {MERGE_MAX}')
    
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Total Weight', fontsize=12)
    ax2.set_title(f'{title} - Weight Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison(X_original, X_adjusted):
    """Show before/after road adjustment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(X_original[:, 0], X_original[:, 1], s=10, alpha=0.6)
    ax1.set_title('Original Midpoints', fontsize=14)
    ax1.set_xlabel('Easting')
    ax1.set_ylabel('Northing')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(X_adjusted[:, 0], X_adjusted[:, 1], s=10, alpha=0.6, color='green')
    ax2.set_title('Adjusted to Road Endpoints', fontsize=14)
    ax2.set_xlabel('Easting')
    ax2.set_ylabel('Northing')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mst_road_adjustment.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    print("="*70)
    print("MST-Based Capacitated Clustering for Solar Trackers")
    print("="*70)
    
    # 1. Load data
    print("\n[Step 1/8] Loading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"LocationX": "Easting", "LocationY": "Northing"})
    df["Weighting"] = df["Weighting"].astype(float)
    df = df.groupby(["Easting", "Northing"], as_index=False)["Weighting"].sum()
    
    print(f"  Loaded {len(df)} unique tracker rows")
    print(f"  Total weight: {df['Weighting'].sum():.2f}")
    
    X_original = df[["Easting", "Northing"]].values.copy()
    
    # 2. Road geometry preprocessing
    print("\n[Step 2/8] Applying road geometry preprocessing...")
    df_adjusted = adjust_to_road_end(df, ROAD_GEOMETRY, ROW_LENGTH, ROW_ORIENTATION)
    coordinates = df_adjusted[["Easting", "Northing"]].values
    weights = df_adjusted["Weighting"].values
    
    plot_comparison(X_original, coordinates)
    
    # 3. Build MST
    print("\n[Step 3/8] Building Minimum Spanning Tree...")
    G = build_mst(coordinates)
    plot_mst(G, coordinates, "Minimum Spanning Tree", "mst_tree.png")
    
    # 4. Initial clustering (≤240)
    print("\n[Step 4/8] MST-based clustering (≤240)...")
    clusters_240, node_to_cluster_240 = mst_capacitated_clustering(G, weights, TARGET_MAX)
    validate_clusters(clusters_240, TARGET_MAX, "Pass 1 (≤240)")
    plot_clusters(coordinates, node_to_cluster_240, clusters_240, 
                 "Pass 1: MST Clusters ≤240", "mst_pass1_240.png")
    
    # 5. Merge to 480
    print("\n[Step 5/8] Merging adjacent clusters (≤480)...")
    clusters_480, node_to_cluster_480 = merge_adjacent_clusters(
        G, clusters_240, node_to_cluster_240, weights, MERGE_MAX)
    validate_clusters(clusters_480, MERGE_MAX, "Pass 2 (≤480)")
    plot_clusters(coordinates, node_to_cluster_480, clusters_480,
                 "Pass 2: Merged Clusters ≤480", "mst_pass2_480.png")
    
    # 6. Subdivide to 16
    print("\n[Step 6/8] Subdividing clusters (≤16)...")
    subclusters, node_to_subcluster = subdivide_clusters(
        clusters_480, coordinates, weights, SUBDIVIDE_MAX)
    validate_clusters(subclusters, SUBDIVIDE_MAX, "Pass 3 (≤16)")
    plot_clusters(coordinates, node_to_subcluster, subclusters,
                 "Pass 3: Subdivided ≤16", "mst_pass3_16.png")
    
    # 7. Prepare output
    print("\n[Step 7/8] Preparing output data...")
    
    # Create assignments array
    assignments_240 = np.array([node_to_cluster_240[i] for i in range(len(coordinates))])
    assignments_480 = np.array([node_to_cluster_480[i] for i in range(len(coordinates))])
    assignments_sub = np.array([node_to_subcluster[i] for i in range(len(coordinates))])
    
    # Find parent cluster for each subcluster
    parent_clusters = np.array([subclusters[node_to_subcluster[i]]['parent'] 
                                for i in range(len(coordinates))])
    
    df_adjusted['Cluster_240'] = assignments_240
    df_adjusted['Cluster_480'] = assignments_480
    df_adjusted['Cluster_Parent'] = parent_clusters
    df_adjusted['Cluster_Sub'] = assignments_sub
    
    # 8. Export
    print("\n[Step 8/8] Exporting results...")
    df_adjusted.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*70}")
    print("MST Clustering Complete!")
    print(f"{'='*70}")
    print(f"Output files:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - mst_tree.png")
    print(f"  - mst_road_adjustment.png")
    print(f"  - mst_pass1_240.png")
    print(f"  - mst_pass2_480.png")
    print(f"  - mst_pass3_16.png")
    
    # Summary statistics
    print(f"\nFinal Statistics:")
    print(f"  240-clusters: {len(clusters_240)}")
    print(f"  480-clusters: {len(clusters_480)}")
    print(f"  16-subclusters: {len(subclusters)}")
    print(f"  Total weight: {df_adjusted['Weighting'].sum():.2f}")

if __name__ == "__main__":
    main()