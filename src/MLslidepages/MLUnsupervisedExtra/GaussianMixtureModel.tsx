//import React from "react";

export default function GaussianMixtureModel() {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6 pt-16 text-gray-800">
      <h1 className="text-2xl font-bold text-blue-700">Gaussian Mixture Model </h1>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> What is Gaussian Mixture Model (GMM)?</h2>
        <p>
          GMM is a probabilistic clustering algorithm that models data as a mixture of multiple Gaussian (normal) distributions. Each cluster corresponds to one Gaussian component.
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Soft clustering: points belong to clusters with probabilities.</li>
          <li>Captures clusters with different shapes, sizes, and orientations.</li>
          <li>Uses Expectation-Maximization (EM) to fit model parameters.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How GMM Works</h2>
        <p>Assumes data is generated from a mixture of several Gaussian distributions with unknown parameters:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>Mean vector (center) for each Gaussian</li>
          <li>Covariance matrix (shape & orientation) for each Gaussian</li>
          <li>Mixing coefficients (weights) representing cluster proportions</li>
        </ul>
        <p><strong>Steps:</strong></p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Initialization: guess parameters for each Gaussian.</li>
          <li>Expectation step (E-step): calculate probability each point belongs to each Gaussian.</li>
          <li>Maximization step (M-step): update Gaussian parameters to maximize likelihood.</li>
          <li>Repeat E-step and M-step until convergence.</li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> When to Use GMM?</h2>
        <p><strong> When you want:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Probabilistic soft assignments (uncertainty in cluster membership)</li>
          <li>To model elliptical clusters (not just spherical)</li>
          <li>More flexibility than K-Means for cluster shape</li>
        </ul>
        <p><strong> Not suitable for:</strong></p>
        <ul className="list-disc list-inside space-y-1">
          <li>Very high-dimensional data without dimensionality reduction</li>
          <li>Very large datasets (EM can be slow)</li>
          <li>When cluster number is very uncertain (need model selection)</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Soft clustering with probabilistic memberships</li>
          <li>Models ellipsoidal clusters with different covariance</li>
          <li>More flexible cluster shapes than K-Means</li>
          <li>Statistical interpretation within probabilistic framework</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Disadvantages</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Must specify number of clusters beforehand</li>
          <li>Sensitive to initialization (may get stuck in local optima)</li>
          <li>Can be slow on large datasets</li>
          <li>Assumes data fits Gaussian mixture (not always true)</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> Python Example (Using scikit-learn)</h2>
        <div className="overflow-x-auto">
          <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
{`from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=400, centers=3, cluster_std=1.2, random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

# Plot clusters with ellipses
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)

def plot_ellipse(position, covariance, ax=None, **kwargs):
    import matplotlib.patches as patches
    from scipy.linalg import eigh
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = patches.Ellipse(position, width, height, angle, **kwargs)
    ax.add_patch(ellipse)

ax = plt.gca()
for pos, covar in zip(gmm.means_, gmm.covariances_):
    plot_ellipse(pos, covar, ax=ax, alpha=0.3, color='red')

plt.title("Gaussian Mixture Model Clustering")
plt.show()`}
          </pre>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg sm:text-xl font-semibold"> How to Choose Number of Components?</h2>
        <p>
          Use Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC) to compare models with different numbers of components.
        </p>
        <p>Lower BIC/AIC means better balance of fit and model complexity.</p>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold">. Real-World Use Cases</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Speech and image recognition</li>
          <li>Anomaly detection (modeling normal data distribution)</li>
          <li>Financial modeling and risk assessment</li>
          <li>Customer segmentation with overlapping groups</li>
        </ul>
      </section>

      <section className="space-y-3 pb-20">
        <h2 className="text-lg sm:text-xl font-semibold"> Comparison with K-Means</h2>
        <table className="w-full text-sm table-auto border border-gray-300">
          <thead>
            <tr className="bg-blue-100">
              <th className="p-2 text-left">Aspect</th>
              <th className="p-2 text-left">K-Means</th>
              <th className="p-2 text-left">GMM</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2">Cluster Shape</td>
              <td className="p-2">Spherical, equal size</td>
              <td className="p-2">Elliptical, varied size</td>
            </tr>
            <tr>
              <td className="p-2">Assignments</td>
              <td className="p-2">Hard (one cluster only)</td>
              <td className="p-2">Soft (probabilistic)</td>
            </tr>
            <tr>
              <td className="p-2">Model Type</td>
              <td className="p-2">Centroid-based</td>
              <td className="p-2">Probabilistic mixture model</td>
            </tr>
            <tr>
              <td className="p-2">Parameters</td>
              <td className="p-2">Cluster centers</td>
              <td className="p-2">Mean, covariance, weights</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
