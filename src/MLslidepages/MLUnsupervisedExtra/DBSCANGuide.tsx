//import React from "react";

export default function DBSCANGuide() {
  return (
    <div className="max-w-full px-4 sm:px-6 py-6 sm:py-10 mx-auto space-y-6 text-gray-800 text-sm sm:text-base leading-relaxed">
      <h1 className="text-2xl sm:text-3xl font-bold text-blue-700">DBSCAN Clustering</h1>

      {/* 1. What is DBSCAN */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> What is DBSCAN?</h2>
        <p>DBSCAN is an <strong>unsupervised clustering algorithm</strong> that groups together points that are close to each other based on a distance metric and a minimum number of points in a neighborhood.</p>
        <ul className="list-disc ml-5">
          <li><strong>Density-based:</strong> Finds core samples of high density and expands clusters from them</li>
          <li>Can find arbitrarily shaped clusters</li>
          <li>Automatically detects outliers (noise)</li>
        </ul>
      </section>

      {/* 2. How DBSCAN Works */}
      
<section className="space-y-4">
  <h2 className="text-lg sm:text-xl font-semibold"> How DBSCAN Works (Step-by-Step)</h2>

  <p className="font-semibold">Key Parameters:</p>
  <ul className="list-disc list-inside space-y-1">
    <li><code>eps</code>: Radius of neighborhood around a point</li>
    <li><code>minPts</code>: Minimum number of points to form a dense region</li>
  </ul>

  <p className="font-semibold">Definitions:</p>
  <ul className="list-disc list-inside space-y-1">
    <li><strong>Core Point:</strong> ≥ minPts within distance eps</li>
    <li><strong>Border Point:</strong> &lt; minPts but near a core point</li>
    <li><strong>Noise Point:</strong> Neither core nor border</li>
  </ul>

  <p className="font-semibold">Algorithm Steps:</p>
  <ol className="list-decimal list-inside space-y-1">
    <li>Start from an unvisited point</li>
    <li>If it's a core point, form a new cluster</li>
    <li>Expand by visiting all density-reachable points</li>
    <li>Mark border/noise accordingly</li>
    <li>Repeat until all points are visited</li>
  </ol>
</section>

      {/* 3. When to Use */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> When to Use DBSCAN?</h2>
        <p><strong> Use when you:</strong></p>
        <ul className="list-disc ml-5">
          <li>Don't know the number of clusters</li>
          <li>Expect irregular-shaped clusters</li>
          <li>Want to detect noise or anomalies</li>
          <li>Handle spatial or moderately dense data</li>
        </ul>
        <p><strong> Avoid when:</strong></p>
        <ul className="list-disc ml-5">
          <li>Working with high-dimensional data</li>
          <li>Clusters have very different densities</li>
        </ul>
      </section>

      {/* 4. Advantages */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold">Advantages of DBSCAN</h2>
        <ul className="list-disc ml-5">
          <li>No need to specify number of clusters</li>
          <li>Can detect non-convex, irregular clusters</li>
          <li>Automatically identifies outliers</li>
          <li>Efficient with large geographic/spatial data</li>
        </ul>
      </section>

      {/* 5. Disadvantages */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages of DBSCAN</h2>
        <ul className="list-disc ml-5">
          <li>Choosing optimal <code>eps</code> and <code>minPts</code> is tricky</li>
          <li>Fails with varying densities</li>
          <li>Not scalable to massive datasets (though variants exist)</li>
        </ul>
      </section>

      {/* 6. Code Example */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Python Code Example (Scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-3 rounded text-sm whitespace-pre-wrap">
{`from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.1)

dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()`}
          </pre>
        </div>
      </section>

      {/* 7. Choosing Parameters */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> How to Choose Parameters</h2>
        <ul className="list-disc ml-5">
          <li><strong>eps:</strong> Use a k-distance plot (sorted distance to k-th nearest neighbor)</li>
          <li><strong>minPts:</strong> Rule of thumb: <code>minPts = 2 * number_of_dimensions</code></li>
        </ul>
      </section>

      {/* 8. Variants */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> DBSCAN Variants</h2>
        <ul className="list-disc ml-5">
          <li><strong>HDBSCAN:</strong> Hierarchical DBSCAN, handles varying density</li>
          <li><strong>OPTICS:</strong> Orders points to reflect cluster structure without fixed eps</li>
          <li><strong>GDBSCAN:</strong> Generalized version with flexible distance metrics</li>
        </ul>
      </section>

      {/* 9. Use Cases */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Real-World Use Cases</h2>
        <ul className="list-disc ml-5">
          <li>Anomaly detection in cybersecurity</li>
          <li>Geospatial clustering (e.g., GPS data)</li>
          <li>Image segmentation</li>
          <li>Customer segmentation with flexible patterns</li>
        </ul>
      </section>

      {/* 10. Visualization Tip */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold"> Visualization Tip</h2>
        <p>Use <strong>t-SNE</strong> or <strong>PCA</strong> to visualize high-dimensional data before applying DBSCAN. Helps in tuning parameters better.</p>
      </section>
    </div>
  );
}
