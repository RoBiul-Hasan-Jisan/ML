//import React from "react";

export default function HierarchicalClustering() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-800">
      <h1 className="text-2xl font-bold text-blue-700">Hierarchical Clustering</h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is Hierarchical Clustering?</h2>
        <p>
          Hierarchical clustering is an <strong>unsupervised learning</strong> method that builds a hierarchy of clusters either:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Agglomerative (bottom-up):</strong> Start with each point as its own cluster, then iteratively merge the closest clusters.</li>
          <li><strong>Divisive (top-down):</strong> Start with all points in one cluster and iteratively split clusters.</li>
        </ul>
        <p>
          It produces a <strong>dendrogram</strong>, a tree-like structure representing the nested cluster merges or splits.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How Hierarchical Clustering Works</h2>
        <p><strong>Agglomerative (most common) steps:</strong></p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Assign each data point to its own cluster.</li>
          <li>Calculate distance between clusters (using linkage criteria).</li>
          <li>Merge the two closest clusters.</li>
          <li>Repeat steps 2-3 until all points are merged into one cluster.</li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold">Linkage Criteria (How to measure distance between clusters)</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Single linkage:</strong> Distance between the closest points of two clusters.</li>
          <li><strong>Complete linkage:</strong> Distance between the farthest points of two clusters.</li>
          <li><strong>Average linkage:</strong> Average distance between all points of two clusters.</li>
          <li><strong>Ward’s method:</strong> Minimizes the variance within clusters.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold">. When to Use Hierarchical Clustering?</h2>
        <p><strong> When you want to:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Understand the data hierarchy or nested clusters</li>
          <li>Visualize cluster relationships via dendrograms</li>
          <li>Avoid specifying number of clusters upfront (you can cut dendrogram at any level)</li>
          <li>Handle small to medium datasets</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Very large datasets (computationally expensive, O(n²))</li>
          <li>Data with noisy or high-dimensional features (distance metrics lose meaning)</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>No need to pre-specify number of clusters</li>
          <li>Produces a dendrogram for easy interpretation</li>
          <li>Works with various distance and linkage metrics</li>
          <li>Can handle non-globular cluster shapes</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Computationally expensive on large datasets</li>
          <li>Sensitive to noise and outliers</li>
          <li>Once merged/split, decisions cannot be undone (no reassignments)</li>
          <li>Choice of linkage and distance metrics affects results significantly</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (Agglomerative Clustering with SciPy)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1, 2], [2, 3], [3, 2], [8, 7], [7, 8], [8, 8]])

# Generate linkage matrix using Ward’s method
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Cutting the Dendrogram</h2>
        <p>Cut the dendrogram at a chosen height to get the desired number of clusters.</p>
        <p>You can use <code>fcluster</code> from <code>scipy.cluster.hierarchy</code> for cluster assignments:</p>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from scipy.cluster.hierarchy import fcluster

max_d = 5  # maximum distance threshold
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Gene expression analysis</li>
          <li>Document clustering and text mining</li>
          <li>Customer segmentation with hierarchical relationships</li>
          <li>Social network analysis</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Visualization Tips</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Use dendrograms to visually choose number of clusters.</li>
          <li>Combine with dimensionality reduction (PCA, t-SNE) for clearer cluster separation.</li>
        </ul>
      </section>
    </div>
  );
}
