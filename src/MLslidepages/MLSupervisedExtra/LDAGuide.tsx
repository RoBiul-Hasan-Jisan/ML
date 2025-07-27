// import React from "react";

export default function LDAGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-base sm:text-lg leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">
        Linear Discriminant Analysis 
      </h1>

      {/* Section 1 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> What is LDA?</h2>
        <p>Linear Discriminant Analysis (LDA) is a supervised learning algorithm used for:</p>
        <ul className="list-disc ml-6">
          <li>Classification</li>
          <li>Dimensionality reduction</li>
        </ul>
        <p>
          Unlike PCA (which is unsupervised), LDA considers class labels and finds a linear combination of features that best separates two or more classes.
        </p>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Intuition</h2>
        <p>LDA projects high-dimensional data onto a lower-dimensional space while maximizing class separability.</p>
        <p>Goals:</p>
        <ul className="list-disc ml-6">
          <li>Maximize the distance between class means.</li>
          <li>Minimize the variance within each class.</li>
        </ul>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">LDA vs PCA</h2>
        <div className="overflow-auto text-sm sm:text-base">
          <table className="min-w-full border border-gray-300 text-left text-sm">
            <thead>
              <tr className="bg-gray-200">
                <th className="border px-2 py-1">Aspect</th>
                <th className="border px-2 py-1">LDA</th>
                <th className="border px-2 py-1">PCA</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Type</td>
                <td className="border px-2 py-1">Supervised</td>
                <td className="border px-2 py-1">Unsupervised</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Goal</td>
                <td className="border px-2 py-1">Maximize class separability</td>
                <td className="border px-2 py-1">Maximize variance</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Considers labels</td>
                <td className="border px-2 py-1">Yes</td>
                <td className="border px-2 py-1"> No</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Use case</td>
                <td className="border px-2 py-1">Classification</td>
                <td className="border px-2 py-1">Visualization, noise reduction</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">. Mathematics Behind LDA</h2>
        <p>Notations:</p>
        <ul className="list-disc ml-6">
          <li><code>μᵢ</code>: Mean vector of class i</li>
          <li><code>μ</code>: Overall mean</li>
          <li><code>S<sub>W</sub></code>: Within-class scatter matrix</li>
          <li><code>S<sub>B</sub></code>: Between-class scatter matrix</li>
        </ul>
        <p><strong> Compute Scatter Matrices</strong></p>
        <p>Within-class scatter:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`S_W = Σ (x - μᵢ)(x - μᵢ)ᵀ over all classes`}
        </pre>
        <p>Between-class scatter:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`S_B = Σ Nᵢ (μᵢ - μ)(μᵢ - μ)ᵀ`}
        </pre>

        <p><strong>Find Projection Matrix W</strong></p>
        <p>
          Solve the generalized eigenvalue problem:
        </p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`S_W⁻¹ S_B W = λW`}
        </pre>
        <p>The top k eigenvectors form the transformation matrix to project data to k dimensions.</p>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Python Implementation (Scikit-learn)</h2>
        <div className="bg-gray-100 p-1 rounded text-xs sm:text-sm overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X, y = data.data, data.target

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plotting
plt.figure(figsize=(8,6))
for label in range(3):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=f"Class {label}")
plt.legend()
plt.title("LDA Projection of Iris Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.show()`}
          </pre>
        </div>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">. LDA for Classification</h2>
        <p>
          LDA is also used as a classifier based on Bayes' theorem:
        </p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`P(y=k | x) = P(x | y=k) * P(y=k) / P(x)`}
        </pre>
        <p>Assumes:</p>
        <ul className="list-disc ml-6">
          <li>Features are normally distributed</li>
          <li>Equal class covariances</li>
        </ul>
        <p>Use in sklearn:</p>
        <div className="bg-gray-100 p-1 rounded text-xs sm:text-sm overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          </pre>
        </div>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">Applications of LDA</h2>
        <ul className="list-disc ml-6">
          <li>Face recognition</li>
          <li>Pattern recognition</li>
          <li>Bioinformatics</li>
          <li>Credit scoring</li>
          <li>Preprocessing before classification</li>
        </ul>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Tips for Using LDA</h2>
        <div className="overflow-auto">
          <table className="min-w-full border border-gray-300 text-left text-sm">
            <thead>
              <tr className="bg-gray-200">
                <th className="border px-2 py-1">Tip</th>
                <th className="border px-2 py-1">Why</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Normalize data</td>
                <td className="border px-2 py-1">LDA assumes Gaussian distribution</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Use only for labeled datasets</td>
                <td className="border px-2 py-1">It’s supervised</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Check covariance assumptions</td>
                <td className="border px-2 py-1">Assumes same covariance</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Don’t use for non-linear problems</td>
                <td className="border px-2 py-1">LDA is linear</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">Advantages & Disadvantages</h2>
        <p className="font-semibold"> Advantages:</p>
        <ul className="list-disc ml-6">
          <li>Simple and fast</li>
          <li>Effective when assumptions hold</li>
          <li>Good for high-dimensional data (e.g., text)</li>
        </ul>
        <p className="font-semibold mt-2">Disadvantages:</p>
        <ul className="list-disc ml-6">
          <li>Assumes normally distributed features</li>
          <li>Linear decision boundaries only</li>
          <li>Poor performance if classes share similar means</li>
        </ul>
      </section>

      {/* Section 10 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Interview Questions on LDA</h2>
        <ul className="list-disc ml-6">
          <li>What’s the difference between LDA and PCA?</li>
          <li>What assumptions does LDA make?</li>
          <li>How does LDA perform dimensionality reduction?</li>
          <li>Can LDA be used for multiclass classification?</li>
          <li>What are the within-class and between-class scatter matrices?</li>
        </ul>
      </section>
    </div>
  );
}
