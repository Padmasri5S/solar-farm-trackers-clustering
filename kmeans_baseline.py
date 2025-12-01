'''Implements a Capacity aware KMeans clustering algorithm for solar tracker data.

- Input CSV has columns: LocationX (Easting), LocationY (Northing), Weightings
- We target 240-max clusters in pass 1, with optional merges to 480 in pass 2
'''

import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

CSV_PATH = "X_EL_SOLAR PANELS_2025-11-27_1354.csv"
TARGET_MAX_CLUSTER_CAPACITY = 240
TARGET_MERGED_CLUSTER_CAPACITY = 480
RANDOM_STATE = 42

def format_axes_numeric(ax, thousands=False):
    from matplotlib.ticker import ScalarFormatter, FuncFormatter
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    if thousands:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}"))
    ax.grid(True, alpha=0.3, linestyle='--')

# Load data
data = pd.read_csv(CSV_PATH)
data = data.rename(columns={"LocationX": "Easting", "LocationY": "Northing", "Weighting": "Weighting"}).astype(float)

# Remove duplicate weightings
data = data.groupby(['Easting','Northing'], as_index=False)['Weighting'].sum()

# Road end processing hook (to be updated as needed later)
def adjust_to_road_ends(points_data, road_end_data=None):
    # Placeholder for road end adjustment logic
    return points_data
data = adjust_to_road_ends(data, road_end_data=None)

# Choose initial k
W_total = data['Weighting'].sum()
k = math.ceil(W_total / TARGET_MAX_CLUSTER_CAPACITY) # considering 240 capacity clusters

# Bind k between 1 and len(data)
k = max(1, min(k, len(data)))

# Run the spatial KMeans clustering
X = data[['Easting', 'Northing']].values
kmeans = KMeans(n_clusters=k,      # number of clusters (choose via W_total / target capacity)
    init="k-means++",  # good default; spreads initial centroids across the data
    n_init=10,         # number of different random initializations to try
    max_iter=300,      # per-run iterations; increase if convergence is slow
    tol=1e-4,          # convergence tolerance; lower for tighter convergence
    random_state=RANDOM_STATE)    # reproducibility
labels_initial = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_ # shape (k, 2)

# Capacity aware merging of clusters (reassignment)
"""
Assign each point to the nearest centroid, respecting cluster capacity.
    If the nearest centroid is full, try the next nearest, etc.

    X: (n, 2) array of coordinates
    weights: (n,) array of per-point weights
    centroids: (k, 2) array
    capacity: max cluster total weight

    Returns: assignments (n,), cluster_totals (k,)
    """
def capacity_assign(X, weights, centroids, target_capacity):
    n, k = X.shape[0], centroids.shape[0]
    distances = cdist(X, centroids, 'euclidean') # shape (n, k)
    order = np.argsort(distances, axis=1) # nearest -> farthest centroid indices for each point

    assignments = -np.ones(n, dtype=int) # unassigned n cluster assignments
    totals = np.zeros(k, dtype=float) # current total weights per cluster

    # Greedy assignment of points by increasing distance to nearest centroids
    for i in range(n):
        w = weights[i]
        placed = False
        for c in order[i]:
            if totals[c] + w <= target_capacity:
                assignments[i] = c
                totals[c] += w
                placed = True
                break
        if not placed:
            # If no centroid can take this point, create an overflow bucket
            # Here we assign to nearest with minimal overflow
            c = order[i][0]
            assignments[i] = c
            totals[c] += w # this indicates overflow/exceeding current capacity; consider increasing k
    return assignments, totals

weights = data["Weighting"].values
assignments_240, totals_240 = capacity_assign(X, weights, centroids, TARGET_MAX_CLUSTER_CAPACITY)

# Optional: Further merge to 480-capacity clusters
def build_cluster_centroids(assignments, X):
    k = len(np.unique(assignments))
    centroids = []
    for c in range(k):
        points_in_cluster = X[assignments == c]
        if len(points_in_cluster) > 0:
            centroids.append(np.mean(points_in_cluster, axis=0))
        else:
            centroids.append(np.array([np.nan, np.nan])) # empty cluster
    return np.array(centroids)

centroids_240 = build_cluster_centroids(assignments_240, X)

def merge_to_480(assignments, weights, max_capacity = TARGET_MERGED_CLUSTER_CAPACITY, X=X):
    # Greedily merge pairs of clusters whose combined weight <= max_merge_capacity
    # and whose centroids are closest
    clusters = np.unique(assignments)
    totals = {c: weights[assignments == c].sum() for c in clusters}
    cents = build_cluster_centroids(assignments, X)

    # build distance matrix between centroids
    C = cents[clusters]
    Cdist = cdist(C, C, 'euclidean')
    np.fill_diagonal(Cdist, np.inf) # ignore self-distances

    # Greedy merge
    merged = {c: c for c in clusters} # initially each cluster maps to itself (representative mapping)
    pairs = []
    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            if j<= i: continue
            pairs.append((Cdist[i,j], ci, cj))
    pairs.sort(key=lambda x:x[0]) # sort by distance

    # Union-Find structure for merges (simple mapping)
    def rep(c):
        while merged[c] != c:
            merged[c] = merged[merged[c]]
            c = merged[c]
        return c
    for _, a, b in pairs:
        ra, rb = rep(a), rep(b)
        if ra == rb:
            continue
        if totals[ra] + totals[rb] <= max_capacity:
            # merge rb into ra
            merged[rb] = ra
            totals[ra] += totals[rb]
    
    # Recompute final assignments
    # Map each original cluster to its representative
    final_assignments = assignments.copy()
    for idx, c in enumerate(final_assignments):
        final_assignments[idx] = rep(c)
    return final_assignments

assignments_480 = merge_to_480(assignments_240, weights, TARGET_MERGED_CLUSTER_CAPACITY, X)

# Reporting and Visualization
def summarize(assignments, weights, title):
    totals = {}
    for c in np.unique(assignments):
        totals[c] = weights[assignments == c].sum()
    print(f"--- {title} ---")
    for c, t in sorted(totals.items()):
        print(f"Cluster {c}: Total Weight = {t:.2f}")
    return totals

totals_pass1 = summarize(assignments_240, weights, "Pass 1 (240-capacity)")
totals_pass2 = summarize(assignments_480, weights, "Pass 2 (merged-480-capacity)")

# Visualize weighted density heatmap of original data
# X is Nx2 (Easting, Northing), weights is shape (N,)
x = data["Easting"].values
y = data["Northing"].values
w = data["Weighting"].values

# Define grid resolution in meters (adjust to taste, e.g., 10 m)
dx, dy = 10, 10
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

xbins = int(np.ceil((xmax - xmin) / dx))
ybins = int(np.ceil((ymax - ymin) / dy))

# 2D weighted histogram (weights = tracker row weighting)
H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], range=[[xmin, xmax], [ymin, ymax]], weights=w)

# Use pcolormesh so bin edges are respected; transpose H because histogram2d returns [x, y]
plt.figure(figsize=(9,7))
plt.pcolormesh(xedges, yedges, H.T, cmap="viridis", shading='auto')
cbar = plt.colorbar(label="Sum of row weighting per cell")
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.title("Solar Farm Weighted Density Heatmap")
format_axes_numeric(plt.gca(), thousands=True)
plt.tight_layout()
plt.show()

# Visualize final clusters after merging to 480-capacity with a legend
import matplotlib.patches as mpatches
import matplotlib.cm as cm

final = assignments_480  # your merged labels
unique_clusters = np.unique(final)

# Assign stable colors from a colormap
cmap = cm.get_cmap("tab20", len(unique_clusters))
cluster_to_color = {c: cmap(i) for i, c in enumerate(unique_clusters)}

# Scatter with those colors
plt.figure(figsize=(9,7))
for c in unique_clusters:
    idx = np.where(final == c)[0]
    plt.scatter(X[idx,0], X[idx,1], s=8, color=cluster_to_color[c], label=f"Cluster {c}")

plt.xlabel("Easting")
plt.ylabel("Northing")
plt.title("KMeans Clustering with Capacity Awareness (≤ 480-capacity)")
format_axes_numeric(plt.gca(), thousands=True)

# Build a compact legend with weight totals; place outside to avoid clutter
handles = []
for i, c in enumerate(unique_clusters):
    total_w = weights[final == c].sum()
    # if you have site/farm names per point, include them here:
    # farm_name = ...  # e.g., data.loc[idx, 'FarmName'].mode()[0] # append farm_name if available
    handles.append(mpatches.Patch(color=cluster_to_color[c], label=f"Group {i}: ID={c}, total={total_w:.0f}"))
plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=8, title="Merged ≤480 Groups")
plt.tight_layout()
plt.show()