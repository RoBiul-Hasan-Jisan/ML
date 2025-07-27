// import React from "react";

export default function KNNGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-base sm:text-lg leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">
        K-Nearest Neighbors 
      </h1>

      {/* Section 1 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> What is KNN?</h2>
        <p>KNN is a supervised learning algorithm used for:</p>
        <ul className="list-disc ml-6">
          <li>Classification (main use case)</li>
          <li>Regression (less common)</li>
        </ul>
        <p className="mt-2">🔹 <strong>Lazy Learner</strong>: No model is trained; it memorizes and predicts later.</p>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> How Does KNN Work?</h2>
        <ol className="list-decimal ml-6 space-y-1">
          <li>Choose K</li>
          <li>Calculate distances (usually Euclidean)</li>
          <li>Sort the distances</li>
          <li>Select K nearest neighbors</li>
          <li>
            Vote:
            <ul className="list-disc ml-6">
              <li>Classification: majority class</li>
              <li>Regression: average value</li>
            </ul>
          </li>
        </ol>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Distance Metrics</h2>
        <ul className="list-disc ml-6">
          <li><strong>Euclidean</strong>: √((x₁−y₁)² + ... + (xₙ−yₙ)²)</li>
          <li>Manhattan distance</li>
          <li>Minkowski distance</li>
          <li>Hamming distance (for categorical)</li>
        </ul>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Choosing the Right K</h2>
        <ul className="list-disc ml-6">
          <li>Small K → Overfitting</li>
          <li>Large K → Underfitting</li>
          <li>
            <p>
              Use <strong>odd K</strong> for binary classification.
              <br />
              Use cross-validation for tuning.
            </p>
          </li>
        </ul>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Pros and Cons</h2>
        <p className="font-semibold"> Pros:</p>
        <ul className="list-disc ml-6">
          <li>Simple and intuitive</li>
          <li>No training phase</li>
          <li>Good for small data</li>
        </ul>
        <p className="font-semibold mt-2"> Cons:</p>
        <ul className="list-disc ml-6">
          <li>Slow prediction</li>
          <li>Needs scaling</li>
          <li>Memory-intensive</li>
        </ul>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Feature Scaling</h2>
        <p>Because KNN relies on distances, feature scaling is critical:</p>
        <ul className="list-disc ml-6">
          <li>Min-Max: (x - min) / (max - min)</li>
          <li>Standardization: (x - μ) / σ</li>
        </ul>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> KNN Classification Example (Python)</h2>
        <div className="bg-gray-100 p-2 rounded text-sm overflow-x-auto">
          <code className="whitespace-pre-wrap break-words break-all font-mono block text-[13px] sm:text-sm">
{`from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)`}
          </code>
        </div>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> KNN for Regression</h2>
        <p>Use <code>KNeighborsRegressor</code> instead of <code>KNeighborsClassifier</code>.</p>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Real-Life Applications</h2>
        <ul className="list-disc ml-6">
          <li>Recommender systems</li>
          <li>Medical diagnosis</li>
          <li>Handwriting recognition</li>
          <li>Image classification</li>
        </ul>
      </section>

      {/* Section 10 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Tips for Using KNN</h2>
        <ul className="list-disc ml-6">
          <li>Scale your features</li>
          <li>Use KD-Tree or Ball Tree for faster searches</li>
          <li>Tune K with cross-validation</li>
          <li>Remove irrelevant features</li>
          <li>Use <code>weights="distance"</code> to give more weight to closer neighbors</li>
        </ul>
        <div className="bg-gray-100 mt-2 p-2 rounded text-sm font-mono overflow-x-auto">
          <code className="block whitespace-pre-wrap break-words break-all">
{`KNeighborsClassifier(n_neighbors=5, weights='distance')`}
          </code>
        </div>
      </section>

      {/* Section 12 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Visualization</h2>
        <div className="bg-gray-100 p-4 rounded text-sm font-mono overflow-x-auto">
          <code className="block whitespace-pre-wrap break-words break-all">
{`      🟥
   🟥
🟦     ❓ ← Predict this point
   🟦
      🟦

If K=3 → 2 🟦, 1 🟥 → Predict = 🟦`}
          </code>
        </div>
      </section>

      {/* Section 13 */}
      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Common Interview Questions</h2>
        <ul className="list-disc ml-6">
          <li>Why is KNN called lazy?</li>
          <li>What happens when K = 1?</li>
          <li>How does it perform with high-dimensional data?</li>
          <li>How do you choose the best K?</li>
          <li>How to scale KNN for big data?</li>
        </ul>
      </section>
    </div>
  );
}
