//import React from "react";

export default function SpectralClustering() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-800">
      <h1 className="text-2xl font-bold text-blue-700">Spectral Clustering</h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is Spectral Clustering?</h2>
        <p>
          Spectral Clustering is an unsupervised algorithm that uses the spectrum (eigenvalues) of a similarity matrix from data to reduce dimensionality before clustering (usually with K-Means).
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Effective for non-convex, connected, or manifold-shaped clusters.</li>
          <li>Interprets clustering as a graph partitioning problem.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How Spectral Clustering Works</h2>
        <ol className="list-decimal list-inside space-y-1">
          <li>
            <strong>Build similarity graph:</strong> nodes = data points; edges = similarity (e.g., Gaussian kernel).
          </li>
          <li>
            <strong>Compute graph Laplacian:</strong> L = D - W where W = adjacency matrix, D = degree matrix.
          </li>
          <li>
            <strong>Compute eigenvectors:</strong> find first k eigenvectors of Laplacian (smallest eigenvalues).
          </li>
          <li>
            <strong>Cluster in spectral space:</strong> apply K-Means on rows of eigenvector matrix.
          </li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold">When to Use Spectral Clustering?</h2>
        <p><strong> When you want to:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Detect complex-shaped clusters (spirals, rings)</li>
          <li>Cluster data where distances alone aren’t enough</li>
          <li>Handle non-linearly separable data</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Very large datasets (costly eigen-decomposition)</li>
          <li>Hard to construct similarity graphs properly</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Finds clusters of arbitrary shape</li>
          <li>Uses global similarity graph info</li>
          <li>Does not assume convex clusters like K-Means</li>
          <li>Flexible similarity measures</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Must choose similarity function and parameters (e.g., kernel bandwidth)</li>
          <li>Computationally expensive (O(n³) eigen decomposition)</li>
          <li>Number of clusters (k) must be specified beforehand</li>
          <li>Sensitive to noise in similarity graph</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate sample data (two interleaving half circles)
X, _ = make_moons(n_samples=300, noise=0.05)

# Apply Spectral Clustering
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0)
labels = sc.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm')
plt.title("Spectral Clustering")
plt.show()`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Key Parameters</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><code>n_clusters</code>: Number of clusters (k)</li>
          <li><code>affinity</code>: Similarity matrix construction method</li>
          <li><code>nearest_neighbors</code>: Graph from nearest neighbors</li>
          <li><code>rbf</code>: Gaussian kernel similarity (needs <code>gamma</code>)</li>
          <li><code>gamma</code>: Kernel coefficient for RBF affinity</li>
          <li><code>n_neighbors</code>: Number of neighbors if using nearest_neighbors affinity</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Image segmentation and computer vision</li>
          <li>Social network community detection</li>
          <li>Bioinformatics (gene clustering)</li>
          <li>Manifold learning and dimensionality reduction</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Comparison with Other Clustering Algorithms</h2>
        <table className="w-full text-sm table-auto border border-gray-300">
          <thead>
            <tr className="bg-blue-100">
              <th className="p-2 text-left">Feature</th>
              <th className="p-2 text-left">K-Means</th>
              <th className="p-2 text-left">DBSCAN</th>
              <th className="p-2 text-left">Spectral Clustering</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2">Cluster Shape</td>
              <td className="p-2">Convex, spherical</td>
              <td className="p-2">Arbitrary shape</td>
              <td className="p-2">Arbitrary, non-convex</td>
            </tr>
            <tr>
              <td className="p-2">Requires k?</td>
              <td className="p-2">Yes</td>
              <td className="p-2">No</td>
              <td className="p-2">Yes</td>
            </tr>
            <tr>
              <td className="p-2">Handles noise</td>
              <td className="p-2">No</td>
              <td className="p-2">Yes</td>
              <td className="p-2">Somewhat</td>
            </tr>
            <tr>
              <td className="p-2">Suitable for manifold data</td>
              <td className="p-2">No</td>
              <td className="p-2">No</td>
              <td className="p-2">Yes</td>
            </tr>
            <tr>
              <td className="p-2">Computational cost</td>
              <td className="p-2">Low</td>
              <td className="p-2">Moderate</td>
              <td className="p-2">High (eigen decomposition)</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
