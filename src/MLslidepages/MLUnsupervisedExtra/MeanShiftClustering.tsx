//import React from "react";

export default function MeanShiftClustering() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-800">
      <h1 className="text-2xl font-bold text-blue-700">Mean Shift Clustering</h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is Mean Shift?</h2>
        <p>
          Mean Shift is a <strong>non-parametric, iterative clustering</strong> algorithm that seeks modes (high-density regions) in a feature space by shifting points towards the average of data points in their neighborhood.
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Doesn’t require specifying the number of clusters upfront.</li>
          <li>Can find clusters of arbitrary shape.</li>
          <li>Works well for mode-seeking and peak detection.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How Mean Shift Works (Step-by-Step)</h2>
        <ol className="list-decimal list-inside space-y-1">
          <li>For each data point, define a window (kernel) around it — usually a radius (bandwidth).</li>
          <li>Compute the mean of points inside this window.</li>
          <li>Shift the window center to the mean.</li>
          <li>Repeat until convergence (window centers stabilize).</li>
          <li>Points whose shifted windows converge to the same mode form a cluster.</li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Key Parameter</h2>
        <p><strong>Bandwidth (radius):</strong> Controls the size of the window/kernel. Critical for results:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>Small bandwidth → many small clusters</li>
          <li>Large bandwidth → fewer large clusters</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> When to Use Mean Shift?</h2>
        <p><strong> When you want to:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Discover clusters without specifying their number</li>
          <li>Identify arbitrary-shaped clusters</li>
          <li>Find modes in data density (e.g., peak detection)</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Very large datasets (computationally expensive)</li>
          <li>High-dimensional data (curse of dimensionality affects kernel density estimation)</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>No need to specify number of clusters beforehand</li>
          <li>Can detect clusters of any shape</li>
          <li>Robust to outliers/noise</li>
          <li>Provides a natural way to find cluster centers</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Computationally expensive (especially for large datasets)</li>
          <li>Sensitive to bandwidth selection</li>
          <li>Not efficient in very high dimensions</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (Using scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

# Fit Mean Shift
ms = MeanShift(bandwidth=1)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=150)
plt.title("Mean Shift Clustering")
plt.show()`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How to Choose Bandwidth?</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Use rule of thumb like <code>bandwidth = 2 * std_deviation</code></li>
          <li>Use grid search or cross-validation on a subset</li>
          <li>Use sklearn’s <code>estimate_bandwidth()</code> function to find a good value</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold">Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Image segmentation (color clustering)</li>
          <li>Object tracking (mean shift tracking in videos)</li>
          <li>Peak detection in signal processing</li>
          <li>Mode detection in density estimation</li>
        </ul>
      </section>
    </div>
  );
}
