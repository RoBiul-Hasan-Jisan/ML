//import React from "react";

export default function BIRCHClustering() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-900">
      <h1 className="text-2xl font-bold text-blue-700">BIRCH Clustering</h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is BIRCH?</h2>
        <p>
          BIRCH is a scalable, hierarchical clustering algorithm designed to efficiently cluster very large datasets — especially suited for streaming or incremental data.
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Builds a Clustering Feature (CF) tree to summarize data incrementally.</li>
          <li>Performs hierarchical clustering on summarized data.</li>
          <li>Works in a single or few scans of the dataset.</li>
          <li>Handles noise and outliers effectively.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How BIRCH Works</h2>
        <p><strong>Main ideas:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>
            <strong>Clustering Feature (CF):</strong> Compact summary of a cluster containing:
            <ul className="list-disc list-inside ml-5 space-y-1">
              <li>Number of points (N)</li>
              <li>Linear sum of points (LS)</li>
              <li>Squared sum of points (SS)</li>
            </ul>
          </li>
          <li>
            <strong>CF Tree:</strong> Height-balanced tree with:
            <ul className="list-disc list-inside ml-5 space-y-1">
              <li>Leaf nodes: CF entries summarizing subclusters.</li>
              <li>Non-leaf nodes: Summaries of child nodes.</li>
            </ul>
          </li>
        </ul>
        <p><strong>Algorithm steps:</strong></p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Insert points incrementally by finding closest leaf entry in CF tree.</li>
          <li>Split nodes if overflow occurs.</li>
          <li>Optionally condense tree by removing outliers or merging clusters.</li>
          <li>Apply global clustering (e.g., agglomerative, K-Means) on leaf entries.</li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold">. When to Use BIRCH?</h2>
        <p><strong> When you want to:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Cluster very large datasets efficiently</li>
          <li>Perform incremental or streaming clustering</li>
          <li>Work with datasets too large for memory</li>
          <li>Get hierarchical clustering quickly</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Data not well summarized by CF vectors</li>
          <li>Very high-dimensional sparse data</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Scalable and fast, handles millions of points in one pass</li>
          <li>Works well with limited memory</li>
          <li>Incremental and dynamic clustering updates</li>
          <li>Detects outliers automatically during CF tree building</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Performance depends heavily on threshold parameter</li>
          <li>Clustering quality depends on initial tree structure</li>
          <li>Less effective if clusters aren't compact or spherical</li>
          <li>Tuning parameters can be complex</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.6, random_state=42)

# Apply BIRCH clustering
birch = Birch(n_clusters=4, threshold=0.5)
labels = birch.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=10)
plt.title("BIRCH Clustering")
plt.show()
`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Key Parameters</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><code>threshold</code>: Maximum diameter of subclusters stored in leaves</li>
          <li><code>branching_factor</code>: Max children per node in CF tree</li>
          <li><code>n_clusters</code>: Number of final clusters (optional)</li>
          <li><code>compute_labels</code>: Whether to predict labels for training data</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Clustering streaming data like sensor data or logs</li>
          <li>Large-scale customer segmentation</li>
          <li>Preprocessing for other clustering algorithms</li>
          <li>Anomaly detection in massive datasets</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Comparison with Other Algorithms</h2>
        <table className="w-full text-sm table-auto border border-gray-300">
          <thead>
            <tr className="bg-blue-100">
              <th className="p-2 text-left">Feature</th>
              <th className="p-2 text-left">BIRCH</th>
              <th className="p-2 text-left">K-Means</th>
              <th className="p-2 text-left">DBSCAN</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2">Scalability</td>
              <td className="p-2">High (one scan)</td>
              <td className="p-2">Moderate</td>
              <td className="p-2">Moderate to low</td>
            </tr>
            <tr>
              <td className="p-2">Incremental</td>
              <td className="p-2">Yes</td>
              <td className="p-2">No</td>
              <td className="p-2">No</td>
            </tr>
            <tr>
              <td className="p-2">Handles Noise</td>
              <td className="p-2">Somewhat</td>
              <td className="p-2">No</td>
              <td className="p-2">Yes</td>
            </tr>
            <tr>
              <td className="p-2">Requires k (clusters)</td>
              <td className="p-2">Optional</td>
              <td className="p-2">Yes</td>
              <td className="p-2">No</td>
            </tr>
            <tr>
              <td className="p-2">Suitable for Large Data</td>
              <td className="p-2">Yes</td>
              <td className="p-2">Sometimes</td>
              <td className="p-2">Sometimes</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
