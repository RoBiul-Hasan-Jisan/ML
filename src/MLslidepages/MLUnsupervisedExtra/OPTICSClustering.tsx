
//import React from "react";

export default function OPTICSClustering() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-800">
      <h1 className="text-2xl font-bold text-blue-700">OPTICS Clustering</h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is OPTICS?</h2>
        <p>
          OPTICS is a density-based clustering algorithm similar to DBSCAN but designed to handle datasets with varying densities.
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Does not explicitly produce clusters but creates an ordering of points showing clustering structure.</li>
          <li>Clusters can be extracted at different density levels via reachability thresholds.</li>
          <li>Stands for Ordering Points To Identify the Clustering Structure.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How OPTICS Works</h2>
        <p><strong>Parameters:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li><code>eps</code>: Maximum radius to consider neighbors (usually large).</li>
          <li><code>minPts</code>: Minimum points to form a dense region.</li>
        </ul>
        <p>
          Unlike DBSCAN, OPTICS:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Processes points ordered by their reachability distance.</li>
          <li>Calculates core distance and reachability distance for each point.</li>
          <li>Produces a reachability plot showing cluster structure at multiple scales.</li>
          <li>Clusters correspond to valleys in the reachability plot.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> When to Use OPTICS?</h2>
        <p><strong> When you want to:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Find clusters of varying densities.</li>
          <li>Use a flexible clustering without fixing epsilon tightly.</li>
          <li>Explore clustering structure visually via reachability plots.</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Very large datasets without indexing structures (may be slow).</li>
          <li>Data where density-based clusters don’t apply.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Handles clusters with varying densities better than DBSCAN.</li>
          <li>No need to choose a global epsilon.</li>
          <li>Identifies hierarchical clustering structure.</li>
          <li>Robust to noise and outliers.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>More complex to interpret (reachability plot needed).</li>
          <li>Computationally more expensive than DBSCAN.</li>
          <li>Cluster extraction from ordering can be tricky.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data with varied densities
X1, _ = make_blobs(n_samples=150, centers=[[2, 2]], cluster_std=0.3)
X2, _ = make_blobs(n_samples=150, centers=[[7, 7]], cluster_std=1.0)
X = np.vstack((X1, X2))

# Fit OPTICS
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(X)
labels = optics.labels_

# Plot clustering results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral')
plt.title("OPTICS Clustering")
plt.show()`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Key Parameters</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><code>min_samples</code>: Minimum points to form cluster (like DBSCAN's minPts).</li>
          <li><code>max_eps</code>: Maximum neighborhood radius (default large).</li>
          <li><code>xi</code>: Steepness threshold to identify cluster boundaries from reachability plot.</li>
          <li><code>min_cluster_size</code>: Minimum size to extract clusters.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Reachability Plot (Optional Visualization)</h2>
        <p>
          Reachability plot visualizes clusters:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>X-axis: Points ordered by OPTICS.</li>
          <li>Y-axis: Reachability distance.</li>
          <li>Valleys correspond to clusters.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Spatial data clustering with varying density (geospatial data)</li>
          <li>Anomaly detection with density variation</li>
          <li>Market segmentation with clusters of varying compactness</li>
          <li>Exploratory data analysis of density structures</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Comparison with DBSCAN</h2>
        <table className="w-full text-sm table-auto border border-gray-300">
          <thead>
            <tr className="bg-blue-100">
              <th className="p-2 text-left">Feature</th>
              <th className="p-2 text-left">DBSCAN</th>
              <th className="p-2 text-left">OPTICS</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2">Density Thresholds</td>
              <td className="p-2">Single global epsilon</td>
              <td className="p-2">Multiple densities</td>
            </tr>
            <tr>
              <td className="p-2">Cluster Shape</td>
              <td className="p-2">Arbitrary</td>
              <td className="p-2">Arbitrary</td>
            </tr>
            <tr>
              <td className="p-2">Handles Varying Density</td>
              <td className="p-2">Poor</td>
              <td className="p-2">Good</td>
            </tr>
            <tr>
              <td className="p-2">Parameter Sensitivity</td>
              <td className="p-2">High</td>
              <td className="p-2">Lower</td>
            </tr>
            <tr>
              <td className="p-2">Output</td>
              <td className="p-2">Flat clusters</td>
              <td className="p-2">Ordering + hierarchical</td>
            </tr>
            <tr>
              <td className="p-2">Computational Cost</td>
              <td className="p-2">Lower</td>
              <td className="p-2">Higher</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
