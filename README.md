## Solar-farm-trackers-clustering

Baseline capacity-aware K-Means approach

Pass 1 (240-capacity):
   Most clusters hit 240 exactly, which means the greedy capacity assignment worked as intended.
   A few clusters are underfilled (e.g., 232, 205, 138) because adding another row would exceed 240.
   This is expected and acceptable for a baseline.

Pass 2 (merged to 480):
   Many clusters are 480, some slightly less (479, 478), and a few smaller (376, 240).
   Smaller totals happen when no nearby cluster could merge without exceeding 480 or violating adjacency.
   Non-contiguous IDs are due to union-find mapping; you can reindex for presentation while preserving lineage.

In this first iteration:
- Spatial compactness is preserved.
- Capacity constraints are respected.
- Merge logic works.
- Heatmap shows density and weighting distribution clearly.

This is a solid baseline for:
- Validating feasibility.
- Comparing against other methods (e.g., hierarchical clustering with constraints or graph-based partitioning).

Note: The heatmap uses 10 m bins, so rows appear as horizontal streaks. That’s normal for CAD-based layouts.
If you want smoother gradients:
   Reduce bin size (e.g., 5 m) for finer resolution.
   Use `shading='auto'` in `pcolormesh` (you already did).
   Optionally apply Gaussian smoothing for aesthetics (not for engineering accuracy).

Better visualization ideas:
1.  Cluster map with legend  
    Shows merged clusters (≤480) with distinct colors and a legend listing cluster ID and total weight.

2.  Add cluster centroids  
    Overlay centroids as `plt.scatter(..., marker='x', s=50)` to show cluster centers.

3.  Annotate totals  
    Use `plt.text()` near centroids to display total weight per cluster for quick interpretation.

4.  Aspect ratio  
    Add `plt.gca().set_aspect('equal')` so Easting/Northing scale is true to meters.

Updated formatting fix for Easting and Northing; which are now printed as full integers (no `1e6` scientific notation).

 🔒 Data integrity

Keep:
   `kmeans_label`, `cap240_label`, `merged480_label` in your output table.
   Export mapping CSVs for lineage.
   Use `merged480_label_ui` only for visuals.

This ensures traceability from raw K-Means to final merged clusters.

Next steps:
- If you want stricter adjacency merging, add a centroid distance threshold (e.g., only merge if distance < 60 m).
- If you want no overflow at all, implement the iterative `capacity_assign_no_overflow` logic I shared earlier.
- As for visuals, generate:
       Heatmap (density),
       Cluster map with legend and centroids,
       Summary table of cluster totals.

Bottom line:  
This is a strong baseline for this dataset. It respects engineering constraints, produces interpretable clusters, and gives you a foundation for refinement.