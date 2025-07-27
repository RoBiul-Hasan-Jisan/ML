//import React from "react";

export default function KMeansClustering() {
  return (
    <div className="max-w-full px-4 sm:px-6 py-6 sm:py-10 mx-auto space-y-6 text-gray-800 text-sm sm:text-base leading-relaxed">
      <h1 className="text-2xl sm:text-3xl font-bold text-blue-700">K-Means Clustering</h1>

      {/* Section 1 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> What is K-Means?</h2>
        <p>K-Means is an <strong>unsupervised machine learning</strong> algorithm used to group data into K distinct clusters based on feature similarity.</p>
        <ul className="list-disc ml-5">
          <li>Unsupervised = No labeled output</li>
          <li>Goal = Minimize intra-cluster distance, maximize inter-cluster distance</li>
        </ul>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> How K-Means Works (Step-by-Step)</h2>
        <ol className="list-decimal ml-5 space-y-1">
          <li>Choose the number of clusters K</li>
          <li>Initialize K centroids randomly</li>
          <li>Assign each data point to the nearest centroid</li>
          <li>Compute new centroids</li>
          <li>Repeat until convergence</li>
        </ol>
        <p><strong> Objective:</strong> Minimize the within-cluster sum of squares (WCSS)</p>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-3 rounded whitespace-pre-wrap text-sm sm:text-base">
WCSS = ∑ᵢ=1ᵏ ∑ₓ∈Cᵢ ∥x − μᵢ∥²
          </pre>
        </div>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Intuition Behind K-Means</h2>
        <ul className="list-disc ml-5">
          <li>Think of dropping K pins on your data</li>
          <li>Each point joins the nearest pin</li>
          <li>Move pins to center of groups</li>
          <li>Repeat until pins stop moving</li>
        </ul>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Real-Life Example</h2>
        <p><strong>Dataset:</strong> Customer segmentation</p>
        <p><strong>Features:</strong> Age, Income, Spending Score</p>
        <p><strong>Goal:</strong> Group customers for targeted marketing</p>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> K-Means in Python (sklearn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-3 rounded text-sm whitespace-pre-wrap">{`
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.title("K-Means Clustering")
plt.show()
          `}</pre>
        </div>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> How to Choose K?</h2>
        <p><strong> Elbow Method:</strong> Plot WCSS vs. K and find the elbow</p>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-3 rounded text-sm whitespace-pre-wrap">{`
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()
          `}</pre>
        </div>
        <p><strong> Silhouette Score:</strong> Measures how well a point fits within its cluster</p>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Advanced Concepts</h2>
        <ul className="list-disc ml-5">
          <li><strong>k-means++:</strong> Better centroid initialization</li>
          <li><strong>Convergence:</strong> Based on movement or iteration limit</li>
          <li><strong>Limitations:</strong> Assumes spherical clusters, sensitive to outliers, local minima</li>
        </ul>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> K-Means Variants</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm border border-gray-300">
            <thead className="bg-gray-200">
              <tr>
                <th className="p-2 text-left">Variant</th>
                <th className="p-2 text-left">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="p-2">MiniBatch K-Means</td><td className="p-2">Faster for large data</td></tr>
              <tr><td className="p-2">K-Medoids</td><td className="p-2">Uses real data points as centers</td></tr>
              <tr><td className="p-2">Fuzzy C-Means</td><td className="p-2">Soft clustering (probabilities)</td></tr>
              <tr><td className="p-2">Spectral Clustering</td><td className="p-2">Works well on complex clusters</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Tips for Pros</h2>
        <ul className="list-disc ml-5">
          <li>Normalize features (e.g., StandardScaler)</li>
          <li>Try PCA before clustering</li>
          <li>Run multiple times with different seeds</li>
          <li>Use k-means++</li>
          <li>Detect and remove outliers</li>
        </ul>
      </section>

      {/* Section 10 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold">Applications</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm border border-gray-300">
            <thead className="bg-gray-200">
              <tr>
                <th className="p-2 text-left">Domain</th>
                <th className="p-2 text-left">Use Case</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="p-2">Marketing</td><td className="p-2">Customer segmentation</td></tr>
              <tr><td className="p-2">Image Processing</td><td className="p-2">Color quantization</td></tr>
              <tr><td className="p-2">Biology</td><td className="p-2">Gene grouping</td></tr>
              <tr><td className="p-2">Retail</td><td className="p-2">Product categorization</td></tr>
              <tr><td className="p-2">Finance</td><td className="p-2">Fraud detection</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 11 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Interview Questions</h2>
        <ul className="list-disc ml-5">
          <li>What is the objective function in K-Means?</li>
          <li>What are the assumptions of K-Means?</li>
          <li>How do you choose the number of clusters?</li>
          <li>What are the drawbacks of K-Means?</li>
          <li>Compare K-Means vs DBSCAN vs Hierarchical Clustering</li>
        </ul>
      </section>

      {/* Section 12 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Summary Table</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm border border-gray-300">
            <thead className="bg-blue-100">
              <tr>
                <th className="p-2 text-left">Feature</th>
                <th className="p-2 text-left">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="p-2">Type</td><td className="p-2">Unsupervised</td></tr>
              <tr><td className="p-2">Task</td><td className="p-2">Clustering</td></tr>
              <tr><td className="p-2">Input</td><td className="p-2">Feature matrix</td></tr>
              <tr><td className="p-2">Output</td><td className="p-2">Cluster labels</td></tr>
              <tr><td className="p-2">Complexity</td><td className="p-2">O(n × k × i × d)</td></tr>
              <tr><td className="p-2">Use Case</td><td className="p-2">Segmentation, Pattern discovery</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
