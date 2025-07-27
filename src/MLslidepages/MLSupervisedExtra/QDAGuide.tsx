// import React from "react";

export default function QDAGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-base sm:text-lg leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">
        Quadratic Discriminant Analysis 
      </h1>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> What is QDA?</h2>
        <p>Quadratic Discriminant Analysis (QDA) is a supervised classification algorithm that models the probability distribution of each class separately using a multivariate Gaussian distribution.</p>
        <p>🔸 Unlike LDA, QDA allows each class to have its own covariance matrix, resulting in quadratic decision boundaries.</p>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> When to Use QDA?</h2>
        <ul className="list-disc ml-6">
          <li>Classes have different covariance structures</li>
          <li>You want non-linear decision boundaries</li>
          <li>You can afford a more complex model (higher variance)</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">QDA vs LDA</h2>
        <div className="overflow-auto text-sm sm:text-base">
          <table className="min-w-full border border-gray-300 text-left text-sm">
            <thead>
              <tr className="bg-gray-200">
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">LDA</th>
                <th className="border px-2 py-1">QDA</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Covariance Matrix</td>
                <td className="border px-2 py-1">Shared among classes</td>
                <td className="border px-2 py-1">One per class</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Decision Boundary</td>
                <td className="border px-2 py-1">Linear</td>
                <td className="border px-2 py-1">Quadratic</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Bias-Variance</td>
                <td className="border px-2 py-1">Lower variance, higher bias</td>
                <td className="border px-2 py-1">Higher variance, lower bias</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Speed</td>
                <td className="border px-2 py-1">Faster</td>
                <td className="border px-2 py-1">Slower (more parameters)</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Accuracy</td>
                <td className="border px-2 py-1">Better if assumptions are met</td>
                <td className="border px-2 py-1">Better with flexible class distributions</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Mathematical Formulation</h2>
        <p>QDA uses Bayes’ Theorem:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`P(y=k | x) = P(x | y=k) * P(y=k) / P(x)`}
        </pre>
        <p>Assumes:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`x | y=k ~ N(μ_k, Σ_k)`}
        </pre>
        <p>Discriminant function:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto whitespace-pre-wrap font-mono">
{`δ_k(x) = -0.5 * log|Σ_k| - 0.5 * (x - μ_k)^T Σ_k⁻¹ (x - μ_k) + log π_k`}
        </pre>
        <p>Prediction: choose the class <code>k</code> with the maximum <code>δ_k(x)</code></p>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Python Example with QDA</h2>
        <div className="bg-gray-100 p-1 rounded text-xs sm:text-sm overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

y_pred = qda.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`}
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Assumptions of QDA</h2>
        <ul className="list-disc ml-6">
          <li>Each class is normally distributed</li>
          <li>Covariance matrix is different for each class</li>
        </ul>
        <p>🚫 Violated when:</p>
        <ul className="list-disc ml-6">
          <li>Data is not normally distributed</li>
          <li>Too few data points → unstable covariance estimation</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Use Cases of QDA</h2>
        <ul className="list-disc ml-6">
          <li>Medical diagnosis (e.g., tumor classification)</li>
          <li>Finance (fraud detection)</li>
          <li>Biometric verification</li>
          <li>Non-linear class separation scenarios</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Visual Example</h2>
        <p>QDA fits classes shaped like ellipses better than LDA.</p>
        <p><strong>In 2D:</strong></p>
        <ul className="list-disc ml-6">
          <li>LDA → Straight line separates circle vs square</li>
          <li>QDA → Curved boundary separates moon/concentric shapes</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2">Pros and Cons</h2>
        <p className="font-semibold">Advantages:</p>
        <ul className="list-disc ml-6">
          <li>Flexible: Can model complex boundaries</li>
          <li>Accurate when classes have different covariances</li>
        </ul>
        <p className="font-semibold mt-2">Disadvantages:</p>
        <ul className="list-disc ml-6">
          <li>More parameters → risk of overfitting</li>
          <li>Sensitive to outliers</li>
          <li>Requires more data</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> QDA vs Other Classifiers</h2>
        <div className="overflow-auto">
          <table className="min-w-full border border-gray-300 text-left text-sm">
            <thead>
              <tr className="bg-gray-200">
                <th className="border px-2 py-1">Model</th>
                <th className="border px-2 py-1">Linear/Nonlinear</th>
                <th className="border px-2 py-1">Parametric?</th>
                <th className="border px-2 py-1">Covariance Assumption</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Logistic Regression</td>
                <td className="border px-2 py-1">Linear</td>
                <td className="border px-2 py-1">Yes</td>
                <td className="border px-2 py-1">No covariance</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">LDA</td>
                <td className="border px-2 py-1">Linear</td>
                <td className="border px-2 py-1">Yes</td>
                <td className="border px-2 py-1">Shared</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">QDA</td>
                <td className="border px-2 py-1">Non-linear</td>
                <td className="border px-2 py-1">Yes</td>
                <td className="border px-2 py-1">One per class</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">KNN</td>
                <td className="border px-2 py-1">Non-linear</td>
                <td className="border px-2 py-1">No</td>
                <td className="border px-2 py-1">No assumption</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-lg sm:text-xl font-semibold mb-2"> Common Interview Questions on QDA</h2>
        <ul className="list-disc ml-6">
          <li>What are the assumptions behind QDA?</li>
          <li>When would QDA outperform LDA?</li>
          <li>Why does QDA have quadratic decision boundaries?</li>
          <li>How does QDA handle different class distributions?</li>
          <li>What happens when QDA is used on small datasets?</li>
        </ul>
      </section>
    </div>
  );
}
