//import React from "react";

export default function AffinityPropagationGuide() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-900">
      <h1 className="text-2xl font-bold text-blue-700">Affinity Propagation Clustering</h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is Affinity Propagation?</h2>
        <p>
          Affinity Propagation is a clustering algorithm that identifies exemplar points (representative cluster centers) by passing messages between data points.
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Does not require specifying the number of clusters beforehand.</li>
          <li>Clusters are formed around exemplars, chosen automatically.</li>
          <li>Uses a message-passing approach called “belief propagation.”</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How Affinity Propagation Works</h2>
        <p><strong>Core concepts:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Similarity matrix (s):</strong> Measures how well-suited point j is to be exemplar for point i.</li>
          <li><strong>Responsibility (r):</strong> Suitability of j as exemplar for i considering other exemplars.</li>
          <li><strong>Availability (a):</strong> Appropriateness of i choosing j as exemplar considering others' preferences.</li>
        </ul>
        <p><strong>Algorithm steps:</strong></p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Initialize responsibility and availability matrices to zero.</li>
          <li>Iteratively update responsibilities and availabilities:
            <ul className="list-disc ml-5 space-y-1">
              <li>Update responsibility r(i,j) based on similarity minus max of other availabilities.</li>
              <li>Update availability a(i,j) based on self-responsibility and sum of positive responsibilities.</li>
            </ul>
          </li>
          <li>After convergence, identify exemplars where responsibility + availability is highest.</li>
          <li>Assign each point to the exemplar with the highest combined score.</li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> When to Use Affinity Propagation?</h2>
        <p><strong> When you want:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Automatic determination of number of clusters.</li>
          <li>Identify exemplars that are actual data points.</li>
          <li>A method suitable for non-spherical clusters.</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Very large datasets (high memory and computation cost).</li>
          <li>Data where similarity matrix is noisy or hard to define.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Automatically determines the number of clusters.</li>
          <li>Finds exemplars that are real data points.</li>
          <li>Works with arbitrary similarity measures.</li>
          <li>No need to specify initial cluster centers.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Requires full similarity matrix — memory expensive for large data.</li>
          <li>Sensitive to the preference parameter affecting cluster count.</li>
          <li>Convergence might be slow or oscillate without damping.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

# Apply Affinity Propagation
ap = AffinityPropagation(preference=-50, random_state=42)
ap.fit(X)
labels = ap.labels_
cluster_centers = ap.cluster_centers_

# Plot clusters and exemplars
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=150)
plt.title("Affinity Propagation Clustering")
plt.show()
`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Key Parameters</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><code>preference</code>: Controls number of exemplars; higher → more clusters.</li>
          <li><code>damping</code>: Controls update stability (0.5–1.0); higher reduces oscillations.</li>
          <li><code>max_iter</code>: Maximum number of iterations.</li>
          <li><code>convergence_iter</code>: Iterations with no change to declare convergence.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Document clustering and summarization</li>
          <li>Image segmentation</li>
          <li>Recommendation systems (finding representative items)</li>
          <li>Bioinformatics (e.g., protein clustering)</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Comparison with Other Clustering Methods</h2>
        <table className="w-full text-sm table-auto border border-gray-300">
          <thead>
            <tr className="bg-blue-100">
              <th className="p-2 text-left">Feature</th>
              <th className="p-2 text-left">K-Means</th>
              <th className="p-2 text-left">GMM</th>
              <th className="p-2 text-left">Affinity Propagation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2">Number of clusters</td>
              <td className="p-2">Must specify</td>
              <td className="p-2">Must specify</td>
              <td className="p-2">Determined automatically</td>
            </tr>
            <tr>
              <td className="p-2">Cluster centers</td>
              <td className="p-2">Centroids (not data points)</td>
              <td className="p-2">Means of Gaussians</td>
              <td className="p-2">Exemplars (actual points)</td>
            </tr>
            <tr>
              <td className="p-2">Model type</td>
              <td className="p-2">Centroid-based</td>
              <td className="p-2">Probabilistic</td>
              <td className="p-2">Message-passing based</td>
            </tr>
            <tr>
              <td className="p-2">Suitable for</td>
              <td className="p-2">Spherical clusters</td>
              <td className="p-2">Elliptical clusters</td>
              <td className="p-2">Arbitrary similarity</td>
            </tr>
            <tr>
              <td className="p-2">Complexity</td>
              <td className="p-2">O(nkt)</td>
              <td className="p-2">O(nkt)</td>
              <td className="p-2">O(n²) (high for large n)</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
