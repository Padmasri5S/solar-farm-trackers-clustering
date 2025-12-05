"""
Improved Visualization for MST-Based Clustering
================================================

Better ways to visualize the clustering results:
1. Hierarchical view (480 → 16 subdivision)
2. Interactive cluster selection
3. Statistical summaries
4. Network graph of cluster adjacency

Run this AFTER the main MST clustering script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
import networkx as nx
from scipy.spatial import Delaunay
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_OUTPUT = "mst_clustered_output.csv"
FOCUS_LEVEL = "480"  # Which level to visualize: "240", "480", or "16"

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading clustered data...")
df = pd.read_csv(CSV_OUTPUT)
coordinates = df[['Easting', 'Northing']].values
weights = df['Weighting'].values

# ============================================================================
# VISUALIZATION 1: CLEAN PASS 2 (480) VIEW
# ============================================================================

def plot_clean_480_clusters(df):
    """
    Clean visualization of 480-max clusters with better color scheme.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Get cluster info
    cluster_ids = df['Cluster_480'].unique()
    n_clusters = len(cluster_ids)
    
    # Use a better color palette
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Plot 1: Spatial with labels
    ax1 = axes[0, 0]
    for i, cluster_id in enumerate(cluster_ids):
        mask = df['Cluster_480'] == cluster_id
        cluster_data = df[mask]
        
        ax1.scatter(cluster_data['Easting'], cluster_data['Northing'],
                   c=[colors[i]], s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add cluster label at centroid
        centroid_x = cluster_data['Easting'].mean()
        centroid_y = cluster_data['Northing'].mean()
        total_weight = cluster_data['Weighting'].sum()
        
        ax1.annotate(f'{cluster_id}\n({total_weight:.0f})',
                    xy=(centroid_x, centroid_y),
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='black', alpha=0.8))
    
    ax1.set_xlabel('Easting', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Northing', fontsize=12, fontweight='bold')
    ax1.set_title('Pass 2: Clusters ≤480 (with IDs and weights)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight distribution histogram
    ax2 = axes[0, 1]
    cluster_weights = df.groupby('Cluster_480')['Weighting'].sum().values
    
    ax2.hist(cluster_weights, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(240, color='orange', linestyle='--', linewidth=2, label='Target 240')
    ax2.axvline(480, color='red', linestyle='--', linewidth=2, label='Max 480')
    ax2.axvline(np.mean(cluster_weights), color='green', linestyle='-', 
                linewidth=2, label=f'Mean: {np.mean(cluster_weights):.0f}')
    
    ax2.set_xlabel('Total Weight', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax2.set_title('Weight Distribution (480-clusters)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cluster size (number of rows) vs weight
    ax3 = axes[1, 0]
    cluster_stats = df.groupby('Cluster_480').agg({
        'Weighting': 'sum',
        'Easting': 'count'
    }).rename(columns={'Easting': 'num_rows'})
    
    scatter = ax3.scatter(cluster_stats['num_rows'], cluster_stats['Weighting'],
                         s=100, c=cluster_stats.index, cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=1)
    
    ax3.axhline(240, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(480, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Number of Rows in Cluster', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Weight', fontsize=12, fontweight='bold')
    ax3.set_title('Cluster Size vs Weight', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Cluster ID', fontsize=10)
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    CLUSTERING SUMMARY (≤480)
    {'='*40}
    
    Total Clusters:          {n_clusters}
    Total Tracker Rows:      {len(df)}
    Total Weight:            {df['Weighting'].sum():.0f}
    
    Weight Statistics:
    ├─ Min:                  {cluster_weights.min():.0f}
    ├─ Max:                  {cluster_weights.max():.0f}
    ├─ Mean:                 {cluster_weights.mean():.2f}
    ├─ Median:               {np.median(cluster_weights):.0f}
    └─ Std Dev:              {cluster_weights.std():.2f}
    
    Utilization:
    ├─ % of 480 capacity:    {(cluster_weights.mean()/480)*100:.1f}%
    └─ Clusters at >450:     {sum(cluster_weights > 450)}
    
    Cluster Size (rows):
    ├─ Min rows/cluster:     {cluster_stats['num_rows'].min():.0f}
    ├─ Max rows/cluster:     {cluster_stats['num_rows'].max():.0f}
    └─ Mean rows/cluster:    {cluster_stats['num_rows'].mean():.1f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('enhanced_480_clusters.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return cluster_weights

# ============================================================================
# VISUALIZATION 2: HIERARCHICAL VIEW (480 → 16 subdivision)
# ============================================================================

def plot_hierarchical_subdivision(df, parent_cluster_id):
    """
    Show how one 480-cluster is subdivided into 16-max subclusters.
    """
    # Filter to specific parent cluster
    parent_data = df[df['Cluster_480'] == parent_cluster_id].copy()
    
    if len(parent_data) == 0:
        print(f"Cluster {parent_cluster_id} not found!")
        return
    
    # Get subclusters
    subclusters = parent_data['Cluster_Sub'].unique()
    n_sub = len(subclusters)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Spatial subdivision
    colors = plt.cm.Set3(np.linspace(0, 1, n_sub))
    
    for i, sub_id in enumerate(subclusters):
        sub_data = parent_data[parent_data['Cluster_Sub'] == sub_id]
        weight = sub_data['Weighting'].sum()
        
        ax1.scatter(sub_data['Easting'], sub_data['Northing'],
                   c=[colors[i]], s=60, alpha=0.8, 
                   edgecolors='black', linewidth=1,
                   label=f'Sub-{i} (w={weight:.0f})')
        
        # Label centroid
        cx = sub_data['Easting'].mean()
        cy = sub_data['Northing'].mean()
        ax1.annotate(f'{i}', xy=(cx, cy), fontsize=12, 
                    fontweight='bold', ha='center', va='center')
    
    ax1.set_xlabel('Easting', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Northing', fontsize=12, fontweight='bold')
    ax1.set_title(f'Cluster {parent_cluster_id} Subdivision (≤16 each)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight bars
    sub_weights = []
    for sub_id in subclusters:
        sub_data = parent_data[parent_data['Cluster_Sub'] == sub_id]
        sub_weights.append(sub_data['Weighting'].sum())
    
    bars = ax2.bar(range(n_sub), sub_weights, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax2.axhline(16, color='red', linestyle='--', linewidth=2, label='Max 16')
    ax2.set_xlabel('Subcluster ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Weight', fontsize=12, fontweight='bold')
    ax2.set_title(f'Subcluster Weights (Total: {sum(sub_weights):.0f})', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'hierarchy_cluster_{parent_cluster_id}.png', dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================================
# VISUALIZATION 3: CLUSTER ADJACENCY NETWORK
# ============================================================================

def plot_cluster_network(df):
    """
    Show 480-clusters as a network graph where edges connect adjacent clusters.
    """
    from scipy.spatial import Delaunay
    
    # Calculate cluster centroids
    cluster_centroids = df.groupby('Cluster_480')[['Easting', 'Northing']].mean()
    cluster_weights = df.groupby('Cluster_480')['Weighting'].sum()
    
    # Build network
    G = nx.Graph()
    
    # Add nodes
    for cluster_id, (easting, northing) in cluster_centroids.iterrows():
        G.add_node(cluster_id, 
                  pos=(easting, northing),
                  weight=cluster_weights[cluster_id])
    
    # Add edges between nearby clusters (using Delaunay triangulation)
    points = cluster_centroids.values
    tri = Delaunay(points)
    
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                c1 = cluster_centroids.index[simplex[i]]
                c2 = cluster_centroids.index[simplex[j]]
                
                # Only connect if distance is reasonable
                dist = np.linalg.norm(points[simplex[i]] - points[simplex[j]])
                if dist < 500:  # Adjust threshold based on your scale
                    G.add_edge(c1, c2, weight=dist)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_node_attributes(G, 'weight')
    
    # Node sizes proportional to weight
    node_sizes = [weights[n] * 2 for n in G.nodes()]
    
    # Node colors by weight
    node_colors = [weights[n] for n in G.nodes()]
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=ax)
    
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap='YlOrRd',
                                   vmin=240, vmax=480,
                                   edgecolors='black',
                                   linewidths=2,
                                   ax=ax)
    
    # Labels
    labels = {n: f'{n}\n{weights[n]:.0f}' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                           font_weight='bold', ax=ax)
    
    ax.set_xlabel('Easting', fontsize=12, fontweight='bold')
    ax.set_ylabel('Northing', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Adjacency Network (≤480)\nNode size = weight, Color = capacity usage', 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                               norm=plt.Normalize(vmin=240, vmax=480))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cluster Weight', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cluster_network_480.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return G

# ============================================================================
# VISUALIZATION 4: OPTIMIZATION ANALYSIS
# ============================================================================

def analyze_optimization_potential(df):
    """
    Analyze if we could better optimize to hit 480 targets.
    """
    cluster_weights = df.groupby('Cluster_480')['Weighting'].sum()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Gap from target
    ax1 = axes[0, 0]
    gaps_from_480 = 480 - cluster_weights
    colors = ['green' if g < 50 else 'orange' if g < 100 else 'red' for g in gaps_from_480]
    
    bars = ax1.bar(range(len(gaps_from_480)), gaps_from_480, color=colors, 
                   edgecolor='black', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gap from 480', fontsize=12, fontweight='bold')
    ax1.set_title('Capacity Headroom per Cluster', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Legend
    green_patch = mpatches.Patch(color='green', label='<50 gap (optimal)')
    orange_patch = mpatches.Patch(color='orange', label='50-100 gap')
    red_patch = mpatches.Patch(color='red', label='>100 gap (underutilized)')
    ax1.legend(handles=[green_patch, orange_patch, red_patch], fontsize=10)
    
    # Plot 2: Cumulative weight
    ax2 = axes[0, 1]
    sorted_weights = np.sort(cluster_weights)[::-1]
    cumulative = np.cumsum(sorted_weights)
    
    ax2.plot(range(len(cumulative)), cumulative, marker='o', 
            linewidth=2, markersize=4, color='steelblue')
    ax2.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Weight Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Annotate total
    ax2.annotate(f'Total: {cumulative[-1]:.0f}',
                xy=(len(cumulative)-1, cumulative[-1]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 3: Target achievement
    ax3 = axes[1, 0]
    
    at_240 = sum((cluster_weights >= 230) & (cluster_weights <= 240))
    near_240 = sum((cluster_weights >= 200) & (cluster_weights < 230))
    at_480 = sum((cluster_weights >= 470) & (cluster_weights <= 480))
    near_480 = sum((cluster_weights >= 430) & (cluster_weights < 470))
    below_400 = sum(cluster_weights < 400)
    
    categories = ['At 240\n(230-240)', 'Near 240\n(200-230)', 
                 'At 480\n(470-480)', 'Near 480\n(430-470)', 'Below 400']
    counts = [at_240, near_240, at_480, near_480, below_400]
    colors_cat = ['lightgreen', 'yellow', 'darkgreen', 'orange', 'red']
    
    bars = ax3.bar(categories, counts, color=colors_cat, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax3.set_title('Target Achievement Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Recommendations text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    optimal_480 = sum(cluster_weights >= 450)
    underutilized = sum(cluster_weights < 350)
    potential_merges = sum(gaps_from_480 > 150)
    
    recommendations = f"""
    OPTIMIZATION ANALYSIS
    {'='*45}
    
    Current State:
    ├─ Total clusters: {len(cluster_weights)}
    ├─ Mean weight: {cluster_weights.mean():.0f}
    └─ Mean gap from 480: {gaps_from_480.mean():.0f}
    
    Target Achievement:
    ├─ Clusters near 480 (>450): {optimal_480}
    ├─ Clusters near 240 (230-240): {at_240}
    └─ Underutilized (<350): {underutilized}
    
    Potential Improvements:
    ├─ Clusters with >150 headroom: {potential_merges}
    │  → Could merge adjacent pairs
    └─ Average utilization: {(cluster_weights.mean()/480)*100:.1f}%
    
    {'✓ GOOD' if cluster_weights.mean() > 400 else '⚠ COULD IMPROVE'}
    
    Recommendation:
    {"Current clustering is OPTIMAL - high utilization" if cluster_weights.mean() > 400 else "Consider merging more aggressively to reach 480"}
    """
    
    ax4.text(0.05, 0.95, recommendations, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig('optimization_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ENHANCED VISUALIZATION SUITE")
    print("="*70 + "\n")
    
    # Load data
    df = pd.read_csv(CSV_OUTPUT)
    
    # 1. Clean 480 view
    print("[1/4] Generating clean 480-cluster visualization...")
    cluster_weights = plot_clean_480_clusters(df)
    
    # 2. Show one hierarchical example
    print("\n[2/4] Generating hierarchical subdivision example...")
    largest_cluster = df.groupby('Cluster_480')['Weighting'].sum().idxmax()
    plot_hierarchical_subdivision(df, largest_cluster)
    
    # 3. Network view
    print("\n[3/4] Generating cluster adjacency network...")
    G = plot_cluster_network(df)
    
    # 4. Optimization analysis
    print("\n[4/4] Analyzing optimization potential...")
    analyze_optimization_potential(df)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - enhanced_480_clusters.png")
    print(f"  - hierarchy_cluster_{largest_cluster}.png")
    print("  - cluster_network_480.png")
    print("  - optimization_analysis.png")

if __name__ == "__main__":
    main()