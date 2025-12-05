"""
Three PCU Clustering Algorithms for Solar Tracker Grouping
===========================================================

Algorithms:
1. HDBSCAN + Capacity Post-Processing
2. Balanced K-Means with Capacity Constraints
3. Louvain Community Detection + Capacity Repair

All include NetworkX-based hierarchical visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import Delaunay
import hdbscan
from sklearn.cluster import KMeans
import community.community_louvain as community_louvain
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = "X_EL_SOLAR PANELS_2025-11-27_1354.csv"
PCU_SINGLE_CAPACITY = 240   # Single inverter PCU
PCU_DUAL_CAPACITY = 480     # Dual inverter PCU
WIRING_GROUP_MAX = 16       # Sub-grouping for wiring
OUTPUT_PREFIX = "pcu_clustering"
RANDOM_STATE = 42

# ============================================================================
# UTILITY FUNCTIONS
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
        for i, w in violations[:5]:
            print(f"  Cluster {i}: {w:.0f} (excess: {w - max_capacity:.0f})")
    
    print(f"\nWeight distribution:")
    print(f"  Min: {min(weights):.0f}")
    print(f"  Max: {max(weights):.0f}")
    print(f"  Mean: {np.mean(weights):.2f}")
    print(f"  Median: {np.median(weights):.2f}")
    print(f"  Utilization (% of {max_capacity}): {(np.mean(weights)/max_capacity)*100:.1f}%")
    
    return violations

def build_spatial_graph(coordinates, k_neighbors=6):
    """Build k-nearest neighbor spatial graph."""
    n = len(coordinates)
    G = nx.Graph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i, pos=coordinates[i])
    
    # Add edges to k nearest neighbors
    distances = squareform(pdist(coordinates))
    
    for i in range(n):
        # Find k nearest neighbors (excluding self)
        nearest = np.argsort(distances[i])[1:k_neighbors+1]
        for j in nearest:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=distances[i, j])
    
    return G

# ============================================================================
# ALGORITHM 1: HDBSCAN + CAPACITY POST-PROCESSING
# ============================================================================

def hdbscan_pcu_clustering(coordinates, weights, min_cluster_size=10):
    """
    HDBSCAN clustering with capacity-aware post-processing.
    
    Steps:
    1. HDBSCAN finds density-based clusters
    2. Split clusters exceeding 240
    3. Merge small adjacent clusters to 480
    4. Subdivide to 16
    """
    print("\n" + "="*70)
    print("ALGORITHM 1: HDBSCAN + Capacity Post-Processing")
    print("="*70)
    
    # Step 1: Run HDBSCAN
    print("\n[1/4] Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        cluster_selection_epsilon=50.0,  # Adjust based on coordinate scale
        metric='euclidean'
    )
    labels = clusterer.fit_predict(coordinates)
    
    # Handle noise points (label -1)
    n_noise = sum(labels == -1)
    if n_noise > 0:
        print(f"  Found {n_noise} noise points - assigning to nearest cluster...")
        valid_labels = labels[labels != -1]
        if len(valid_labels) > 0:
            # Assign noise to nearest valid cluster
            noise_idx = np.where(labels == -1)[0]
            valid_idx = np.where(labels != -1)[0]
            
            for ni in noise_idx:
                dists = cdist([coordinates[ni]], coordinates[valid_idx])[0]
                nearest_valid = valid_idx[np.argmin(dists)]
                labels[ni] = labels[nearest_valid]
    
    initial_clusters = len(np.unique(labels))
    print(f"  Initial clusters: {initial_clusters}")
    
    # Step 2: Convert to cluster dictionaries
    clusters = []
    for label in np.unique(labels):
        mask = labels == label
        clusters.append({
            'nodes': np.where(mask)[0].tolist(),
            'weight': weights[mask].sum()
        })
    
    print(f"\n[2/4] Splitting oversized clusters (>{PCU_SINGLE_CAPACITY})...")
    clusters = split_oversized_clusters(clusters, coordinates, weights, PCU_SINGLE_CAPACITY)
    validate_clusters(clusters, PCU_SINGLE_CAPACITY, "After Split (≤240)")
    
    # Step 3: Merge to 480
    print(f"\n[3/4] Merging adjacent clusters to {PCU_DUAL_CAPACITY}...")
    node_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for node in cluster['nodes']:
            node_to_cluster[node] = i
    
    G = build_spatial_graph(coordinates, k_neighbors=6)
    clusters, node_to_cluster = merge_adjacent_clusters_generic(
        G, clusters, node_to_cluster, weights, PCU_DUAL_CAPACITY)
    validate_clusters(clusters, PCU_DUAL_CAPACITY, "After Merge (≤480)")
    
    # Step 4: Subdivide to 16
    print(f"\n[4/4] Subdividing to ≤{WIRING_GROUP_MAX}...")
    subclusters = subdivide_all_clusters(clusters, coordinates, weights, WIRING_GROUP_MAX)
    validate_clusters(subclusters, WIRING_GROUP_MAX, "Final Subdivision (≤16)")
    
    return clusters, node_to_cluster, subclusters

def split_oversized_clusters(clusters, coordinates, weights, max_capacity):
    """Split clusters that exceed capacity."""
    new_clusters = []
    
    for cluster in clusters:
        if cluster['weight'] <= max_capacity:
            new_clusters.append(cluster)
        else:
            # Split using K-Means
            nodes = cluster['nodes']
            n_splits = int(np.ceil(cluster['weight'] / max_capacity))
            
            if len(nodes) < n_splits:
                # Can't split further, just keep it
                new_clusters.append(cluster)
                continue
            
            # Cluster these nodes
            sub_coords = coordinates[nodes]
            sub_weights = weights[nodes]
            
            km = KMeans(n_clusters=n_splits, n_init=10, random_state=RANDOM_STATE)
            sub_labels = km.fit_predict(sub_coords)
            
            # Create new clusters
            for label in range(n_splits):
                mask = sub_labels == label
                sub_nodes = [nodes[i] for i in range(len(nodes)) if mask[i]]
                new_clusters.append({
                    'nodes': sub_nodes,
                    'weight': sub_weights[mask].sum()
                })
    
    return new_clusters

# ============================================================================
# ALGORITHM 2: BALANCED K-MEANS
# ============================================================================

def balanced_kmeans_pcu_clustering(coordinates, weights):
    """
    Balanced K-Means with capacity constraints.
    
    Iteratively:
    1. Assign to nearest centroid respecting capacity
    2. Recalculate centroids from actual assignments
    3. Repeat until convergence
    """
    print("\n" + "="*70)
    print("ALGORITHM 2: Balanced K-Means with Capacity Constraints")
    print("="*70)
    
    # Initial k estimate
    total_weight = weights.sum()
    k = int(np.ceil(total_weight / PCU_DUAL_CAPACITY))
    k = max(k, 1)
    
    print(f"\n[1/3] Initializing with k={k} clusters...")
    
    # Initialize centroids using K-Means++
    km = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=RANDOM_STATE)
    km.fit(coordinates)
    centroids = km.cluster_centers_
    
    # Iterative refinement
    print("\n[2/3] Iterative capacity-aware refinement...")
    max_iterations = 50
    
    for iteration in range(max_iterations):
        # Capacity-aware assignment
        assignments = np.full(len(coordinates), -1, dtype=int)
        cluster_totals = np.zeros(k)
        
        # Compute distances to all centroids
        dists = cdist(coordinates, centroids)
        
        # Sort points by minimum distance to any centroid (process closest first)
        min_dists = np.min(dists, axis=1)
        processing_order = np.argsort(min_dists)
        
        for idx in processing_order:
            w = weights[idx]
            # Try centroids in order of distance
            centroid_order = np.argsort(dists[idx])
            
            placed = False
            for c in centroid_order:
                if cluster_totals[c] + w <= PCU_DUAL_CAPACITY:
                    assignments[idx] = c
                    cluster_totals[c] += w
                    placed = True
                    break
            
            if not placed:
                # Assign to nearest (overflow)
                c = centroid_order[0]
                assignments[idx] = c
                cluster_totals[c] += w
        
        # Recalculate centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = coordinates[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]  # Keep old if empty
        
        # Check convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        if centroid_shift < 1.0:
            print(f"  Converged at iteration {iteration+1}")
            break
        
        centroids = new_centroids
    
    # Convert to cluster format
    clusters = []
    node_to_cluster = {}
    for c in range(k):
        mask = assignments == c
        if mask.any():
            nodes = np.where(mask)[0].tolist()
            clusters.append({
                'nodes': nodes,
                'weight': weights[mask].sum()
            })
            for node in nodes:
                node_to_cluster[node] = len(clusters) - 1
    
    print("\n[3/3] Final validation...")
    validate_clusters(clusters, PCU_DUAL_CAPACITY, "Balanced K-Means (≤480)")
    
    # Subdivide
    subclusters = subdivide_all_clusters(clusters, coordinates, weights, WIRING_GROUP_MAX)
    validate_clusters(subclusters, WIRING_GROUP_MAX, "Subdivision (≤16)")
    
    return clusters, node_to_cluster, subclusters

# ============================================================================
# ALGORITHM 3: LOUVAIN COMMUNITY DETECTION
# ============================================================================

def louvain_pcu_clustering(coordinates, weights, k_neighbors=6):
    """
    Louvain community detection + capacity repair.
    
    Steps:
    1. Build spatial k-NN graph
    2. Detect communities (natural spatial groups)
    3. Split communities > 240
    4. Merge communities to 480
    5. Subdivide to 16
    """
    print("\n" + "="*70)
    print("ALGORITHM 3: Louvain Community Detection + Capacity Repair")
    print("="*70)
    
    # Step 1: Build graph
    print(f"\n[1/4] Building spatial graph (k={k_neighbors})...")
    G = build_spatial_graph(coordinates, k_neighbors)
    print(f"  Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Step 2: Community detection
    print("\n[2/4] Running Louvain community detection...")
    partition = community_louvain.best_partition(G, random_state=RANDOM_STATE)
    
    # Convert to clusters
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    clusters = []
    for comm_id, nodes in communities.items():
        clusters.append({
            'nodes': nodes,
            'weight': weights[nodes].sum()
        })
    
    print(f"  Initial communities: {len(clusters)}")
    
    # Step 3: Split oversized
    print(f"\n[3/4] Splitting oversized communities (>{PCU_SINGLE_CAPACITY})...")
    clusters = split_oversized_clusters(clusters, coordinates, weights, PCU_SINGLE_CAPACITY)
    validate_clusters(clusters, PCU_SINGLE_CAPACITY, "After Split (≤240)")
    
    # Step 4: Merge to 480
    print(f"\n[4/4] Merging to {PCU_DUAL_CAPACITY}...")
    node_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for node in cluster['nodes']:
            node_to_cluster[node] = i
    
    clusters, node_to_cluster = merge_adjacent_clusters_generic(
        G, clusters, node_to_cluster, weights, PCU_DUAL_CAPACITY)
    validate_clusters(clusters, PCU_DUAL_CAPACITY, "After Merge (≤480)")
    
    # Subdivide
    subclusters = subdivide_all_clusters(clusters, coordinates, weights, WIRING_GROUP_MAX)
    validate_clusters(subclusters, WIRING_GROUP_MAX, "Subdivision (≤16)")
    
    return clusters, node_to_cluster, subclusters

# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================

def merge_adjacent_clusters_generic(G, clusters, node_to_cluster, weights, max_capacity):
    """Generic merge function using spatial graph."""
    # Build cluster adjacency
    cluster_graph = nx.Graph()
    for i, cluster in enumerate(clusters):
        cluster_graph.add_node(i, weight=cluster['weight'])
    
    # Find adjacent clusters in spatial graph
    for u, v in G.edges():
        cu = node_to_cluster[u]
        cv = node_to_cluster[v]
        if cu != cv:
            cluster_graph.add_edge(cu, cv)
    
    # Greedy merging
    merge_candidates = []
    for u, v in cluster_graph.edges():
        wu = cluster_graph.nodes[u]['weight']
        wv = cluster_graph.nodes[v]['weight']
        if wu + wv <= max_capacity:
            merge_candidates.append((wu + wv, u, v))
    
    merge_candidates.sort(reverse=True)  # Prefer larger merges
    
    # Union-Find
    parent = {i: i for i in range(len(clusters))}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    cluster_weights = {i: clusters[i]['weight'] for i in range(len(clusters))}
    
    merged_count = 0
    for _, u, v in merge_candidates:
        pu, pv = find(u), find(v)
        if pu != pv:
            if cluster_weights[pu] + cluster_weights[pv] <= max_capacity:
                parent[pv] = pu
                cluster_weights[pu] += cluster_weights[pv]
                merged_count += 1
    
    # Build merged clusters
    merged = {}
    for i, cluster in enumerate(clusters):
        root = find(i)
        if root not in merged:
            merged[root] = {'nodes': [], 'weight': 0}
        merged[root]['nodes'].extend(cluster['nodes'])
        merged[root]['weight'] += cluster['weight']
    
    final_clusters = list(merged.values())
    
    # Update mapping
    new_node_to_cluster = {}
    for new_id, cluster in enumerate(final_clusters):
        for node in cluster['nodes']:
            new_node_to_cluster[node] = new_id
    
    print(f"  Merged {merged_count} pairs → {len(final_clusters)} clusters")
    return final_clusters, new_node_to_cluster

def subdivide_all_clusters(clusters, coordinates, weights, max_sub):
    """Subdivide clusters to max_sub capacity."""
    subclusters = []
    
    for cluster in clusters:
        nodes = cluster['nodes']
        if cluster['weight'] <= max_sub:
            subclusters.append(cluster)
        else:
            # Spatial greedy packing
            node_coords = coordinates[nodes]
            node_weights = weights[nodes]
            centroid = node_coords.mean(axis=0)
            
            dists = np.linalg.norm(node_coords - centroid, axis=1)
            order = np.argsort(dists)
            
            current = []
            current_weight = 0
            
            for idx in order:
                node = nodes[idx]
                w = node_weights[idx]
                
                if current_weight + w > max_sub and current:
                    subclusters.append({
                        'nodes': current,
                        'weight': current_weight
                    })
                    current = []
                    current_weight = 0
                
                current.append(node)
                current_weight += w
            
            if current:
                subclusters.append({
                    'nodes': current,
                    'weight': current_weight
                })
    
    return subclusters

# ============================================================================
# NETWORKX VISUALIZATION
# ============================================================================

def visualize_pcu_clustering(coordinates, clusters, node_to_cluster, weights, 
                             title, filename):
    """NetworkX-based hierarchical visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    n_clusters = len(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    
    # Plot 1: Spatial with labels
    ax1 = axes[0, 0]
    for i, cluster in enumerate(clusters):
        nodes = cluster['nodes']
        color = colors[i % 20]
        
        ax1.scatter(coordinates[nodes, 0], coordinates[nodes, 1],
                   c=[color], s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Label
        cx = coordinates[nodes, 0].mean()
        cy = coordinates[nodes, 1].mean()
        ax1.annotate(f'{i}\n{cluster["weight"]:.0f}',
                    xy=(cx, cy), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='black', alpha=0.8))
    
    ax1.set_xlabel('Easting', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Northing', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Spatial Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight distribution
    ax2 = axes[0, 1]
    cluster_weights = [c['weight'] for c in clusters]
    ax2.hist(cluster_weights, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(PCU_SINGLE_CAPACITY, color='orange', linestyle='--', 
                linewidth=2, label=f'Single PCU ({PCU_SINGLE_CAPACITY})')
    ax2.axvline(PCU_DUAL_CAPACITY, color='red', linestyle='--', 
                linewidth=2, label=f'Dual PCU ({PCU_DUAL_CAPACITY})')
    ax2.axvline(np.mean(cluster_weights), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(cluster_weights):.0f}')
    ax2.set_xlabel('Strings per PCU', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of PCUs', fontsize=12, fontweight='bold')
    ax2.set_title('PCU Load Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: NetworkX cluster adjacency graph
    ax3 = axes[1, 0]
    
    # Build cluster graph
    G = nx.Graph()
    for i, cluster in enumerate(clusters):
        G.add_node(i, weight=cluster['weight'],
                  pos=(coordinates[cluster['nodes'], 0].mean(),
                       coordinates[cluster['nodes'], 1].mean()))
    
    # Add edges between spatially adjacent clusters
    positions = nx.get_node_attributes(G, 'pos')
    pos_array = np.array(list(positions.values()))
    dists = squareform(pdist(pos_array))
    
    threshold = np.percentile(dists[dists > 0], 20)  # Connect nearest 20%
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if dists[i, j] < threshold:
                G.add_edge(i, j)
    
    # Draw network
    node_sizes = [clusters[i]['weight'] * 2 for i in G.nodes()]
    node_colors = [clusters[i]['weight'] for i in G.nodes()]
    
    nx.draw_networkx_edges(G, positions, alpha=0.2, width=2, ax=ax3)
    nodes = nx.draw_networkx_nodes(G, positions, node_size=node_sizes,
                                   node_color=node_colors, cmap='YlOrRd',
                                   vmin=PCU_SINGLE_CAPACITY, vmax=PCU_DUAL_CAPACITY,
                                   edgecolors='black', linewidths=2, ax=ax3)
    
    labels = {i: f'{i}\n{clusters[i]["weight"]:.0f}' for i in G.nodes()}
    nx.draw_networkx_labels(G, positions, labels, font_size=7,
                           font_weight='bold', ax=ax3)
    
    ax3.set_xlabel('Easting', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Northing', fontsize=12, fontweight='bold')
    ax3.set_title('PCU Adjacency Network', fontsize=14, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                               norm=plt.Normalize(vmin=PCU_SINGLE_CAPACITY, 
                                                 vmax=PCU_DUAL_CAPACITY))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Strings per PCU', fontsize=10)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    single_pcu = sum(w <= PCU_SINGLE_CAPACITY for w in cluster_weights)
    dual_pcu = sum(w > PCU_SINGLE_CAPACITY for w in cluster_weights)
    
    stats_text = f"""
    PCU CLUSTERING SUMMARY
    {'='*40}
    
    Total PCUs:              {n_clusters}
    ├─ Single inverter:      {single_pcu}
    └─ Dual inverter:        {dual_pcu}
    
    Total tracker rows:      {len(coordinates)}
    Total strings:           {weights.sum():.0f}
    
    Strings per PCU:
    ├─ Min:                  {min(cluster_weights):.0f}
    ├─ Max:                  {max(cluster_weights):.0f}
    ├─ Mean:                 {np.mean(cluster_weights):.2f}
    └─ Median:               {np.median(cluster_weights):.0f}
    
    Capacity Utilization:
    ├─ Avg % of dual PCU:    {(np.mean(cluster_weights)/PCU_DUAL_CAPACITY)*100:.1f}%
    └─ PCUs at >90% cap:     {sum(w > 0.9*PCU_DUAL_CAPACITY for w in cluster_weights)}
    
    Configuration:
    └─ Prefer dual over single for efficiency
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("PCU CLUSTERING ALGORITHMS COMPARISON")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"LocationX": "Easting", "LocationY": "Northing"})
    df["Weighting"] = df["Weighting"].astype(float)
    df = df.groupby(["Easting", "Northing"], as_index=False)["Weighting"].sum()
    
    coordinates = df[["Easting", "Northing"]].values
    weights = df["Weighting"].values
    
    print(f"Loaded {len(df)} tracker rows, {weights.sum():.0f} total strings")
    
    # Run all three algorithms
    results = {}
    
    # Algorithm 1: HDBSCAN
    try:
        clusters1, mapping1, subs1 = hdbscan_pcu_clustering(coordinates, weights)
        results['HDBSCAN'] = (clusters1, mapping1, subs1)
        visualize_pcu_clustering(coordinates, clusters1, mapping1, weights,
                                "HDBSCAN + Capacity Post-Processing",
                                f"{OUTPUT_PREFIX}_hdbscan.png")
    except Exception as e:
        print(f"\nHDBSCAN failed: {e}")
        results['HDBSCAN'] = None
    
    # Algorithm 2: Balanced K-Means
    try:
        clusters2, mapping2, subs2 = balanced_kmeans_pcu_clustering(coordinates, weights)
        results['Balanced_KMeans'] = (clusters2, mapping2, subs2)
        visualize_pcu_clustering(coordinates, clusters2, mapping2, weights,
                                "Balanced K-Means with Capacity",
                                f"{OUTPUT_PREFIX}_kmeans.png")
    except Exception as e:
        print(f"\nBalanced K-Means failed: {e}")
        results['Balanced_KMeans'] = None
    
    # Algorithm 3: Louvain
    try:
        clusters3, mapping3, subs3 = louvain_pcu_clustering(coordinates, weights)
        results['Louvain'] = (clusters3, mapping3, subs3)
        visualize_pcu_clustering(coordinates, clusters3, mapping3, weights,
                                "Louvain Community Detection",
                                f"{OUTPUT_PREFIX}_louvain.png")
    except Exception as e:
        print(f"\nLouvain failed: {e}")
        results['Louvain'] = None
    
    # Comparison summary
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*70)
    
    comparison_data = []
    for name, result in results.items():
        if result is not None:
            clusters, mapping, subs = result
            cluster_weights = [c['weight'] for c in clusters]
            comparison_data.append({
                'Algorithm': name,
                'Num PCUs': len(clusters),
                'Mean Strings/PCU': np.mean(cluster_weights),
                'Utilization %': (np.mean(cluster_weights)/PCU_DUAL_CAPACITY)*100,
                'Single PCUs': sum(w <= PCU_SINGLE_CAPACITY for w in cluster_weights),
                'Dual PCUs': sum(w > PCU_SINGLE_CAPACITY for w in cluster_weights)
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        print("\n" + comp_df.to_string(index=False))
        comp_df.to_csv(f"{OUTPUT_PREFIX}_comparison.csv", index=False)
        print(f"\nComparison saved to {OUTPUT_PREFIX}_comparison.csv")
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    for name in results.keys():
        if results[name] is not None:
            print(f"  - {OUTPUT_PREFIX}_{name.lower()}.png")

if __name__ == "__main__":
    main()