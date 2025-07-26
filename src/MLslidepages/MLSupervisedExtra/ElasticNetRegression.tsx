//import React from "react";

const ElasticNetRegression = () => {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-5xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold mb-4  text-blue-600">Elastic Net Regression </h1>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> What is Elastic Net Regression?</h2>
        <p>
          Elastic Net Regression is a linear regression model combining:
        </p>
        <ul className="list-disc list-inside ml-6 mt-2">
          <li>L1 regularization (Lasso) for feature selection</li>
          <li>L2 regularization (Ridge) for coefficient shrinkage</li>
        </ul>
        <p className="mt-2">
          Useful when you have many features, features are highly correlated, and you want both automatic feature selection and stable predictions.
        </p>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> Objective Function (Mathematical Formulation)</h2>
        <p>
          Elastic Net minimizes:
        </p>
        <pre className="bg-gray-100 p-3 rounded overflow-hidden whitespace-pre-wrap break-words text-sm">
          <code>
            {`Loss = ||y - Xβ||₂² + λ₁ ∑|β_j| + λ₂ ∑β_j²`}
          </code>
        </pre>
        <p className="mt-2">
          Or equivalently in scikit-learn form:
        </p>
        <pre className="bg-gray-100 p-3 rounded overflow-hidden whitespace-pre-wrap break-words text-sm">
          <code>
            {`Loss = ||y - Xβ||₂² + α * (ρ * ||β||₁ + (1 - ρ) * ||β||₂²)`}
          </code>
        </pre>
        <p className="mt-2">
          Where:
          <ul className="list-disc list-inside ml-6 mt-1">
            <li><code>α</code> = overall regularization strength</li>
            <li><code>ρ</code> = mix ratio between L1 and L2 penalties</li>
            <li><code>ρ = 1</code> → pure Lasso</li>
            <li><code>ρ = 0</code> → pure Ridge</li>
          </ul>
        </p>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> Intuition Behind Elastic Net</h2>
        <div className="overflow-x-hidden mt-2">
          <table className="w-full table-auto border border-collapse border-gray-300 text-sm sm:text-base break-words">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-2 text-left">Model</th>
                <th className="border px-2 py-2 text-left">Effect on Coefficients</th>
                <th className="border px-2 py-2 text-left">Feature Selection</th>
                <th className="border px-2 py-2 text-left">Correlated Features Handling</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-2">Ridge</td>
                <td className="border px-2 py-2">Shrinks all (none to zero)</td>
                <td className="border px-2 py-2">❌ No</td>
                <td className="border px-2 py-2">✅ Yes</td>
              </tr>
              <tr>
                <td className="border px-2 py-2">Lasso</td>
                <td className="border px-2 py-2">Shrinks and sets some to zero</td>
                <td className="border px-2 py-2">✅ Yes</td>
                <td className="border px-2 py-2">❌ Can be unstable</td>
              </tr>
              <tr>
                <td className="border px-2 py-2">Elastic Net</td>
                <td className="border px-2 py-2">Shrinks some, sets others to zero</td>
                <td className="border px-2 py-2">✅ Yes</td>
                <td className="border px-2 py-2">✅ Groups correlated features</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> Why Use Elastic Net?</h2>
        <p><strong>Pros:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Combines best of Lasso and Ridge</li>
          <li>Handles high-dimensional data (features &gt; samples)</li>
          <li>Automatic feature selection</li>
          <li>Stabilizes Lasso with correlated features</li>
        </ul>
        <p className="mt-2"><strong>Cons:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Needs tuning of two hyperparameters (<code>α</code>, <code>ρ</code>)</li>
          <li>More complex than Ridge or Lasso alone</li>
        </ul>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> Use Cases</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Genomics (thousands of genes, few relevant)</li>
          <li>Text classification (sparse data)</li>
          <li>Financial modeling (correlated indicators)</li>
          <li>Marketing and customer segmentation</li>
        </ul>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> Scikit-learn Implementation</h2>

        <p><strong>Basic Example:</strong></p>
        <pre className="bg-gray-900 text-white p-4 rounded overflow-hidden whitespace-pre-wrap break-words text-sm">
          <code>
{`from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          </code>
        </pre>

        <p className="mt-4"><strong>Hyperparameter tuning with cross-validation:</strong></p>
        <pre className="bg-gray-900 text-white p-4 rounded overflow-hidden whitespace-pre-wrap break-words text-sm">
          <code>
{`from sklearn.linear_model import ElasticNetCV

model_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 1],
    alphas=[0.01, 0.1, 1, 10],
    cv=5
)
model_cv.fit(X_train, y_train)

print("Best alpha:", model_cv.alpha_)
print("Best l1_ratio:", model_cv.l1_ratio_)`}
          </code>
        </pre>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6"> Elastic Net vs Ridge vs Lasso</h2>
        <div className="overflow-x-hidden mt-2">
          <table className="w-full border border-collapse border-gray-300 text-sm sm:text-base break-words">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-2 text-left">Feature</th>
                <th className="border px-2 py-2 text-left">Ridge</th>
                <th className="border px-2 py-2 text-left">Lasso</th>
                <th className="border px-2 py-2 text-left">Elastic Net</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-2">Regularization</td>
                <td className="border px-2 py-2">L2</td>
                <td className="border px-2 py-2">L1</td>
                <td className="border px-2 py-2">L1 + L2</td>
              </tr>
              <tr>
                <td className="border px-2 py-2">Feature Selection</td>
                <td className="border px-2 py-2">❌</td>
                <td className="border px-2 py-2">✅</td>
                <td className="border px-2 py-2">✅</td>
              </tr>
              <tr>
                <td className="border px-2 py-2">Handles Collinearity</td>
                <td className="border px-2 py-2">✅</td>
                <td className="border px-2 py-2">❌</td>
                <td className="border px-2 py-2">✅</td>
              </tr>
              <tr>
                <td className="border px-2 py-2">Shrinks Coefficients</td>
                <td className="border px-2 py-2">✅</td>
                <td className="border px-2 py-2">✅</td>
                <td className="border px-2 py-2">✅</td>
              </tr>
              <tr>
                <td className="border px-2 py-2">Coefficients Zero?</td>
                <td className="border px-2 py-2">❌</td>
                <td className="border px-2 py-2">✅</td>
                <td className="border px-2 py-2">✅</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6">  Visual Comparison (Intuition)</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Lasso: Sparse models (many zeros)</li>
          <li>Ridge: Dense models (all coefficients shrunk, none zero)</li>
          <li>Elastic Net: Sparse but more stable than Lasso — best of both</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-6">  Theoretical Insight</h2>
        <p>
          Solved via convex optimization. Algorithms include coordinate descent and cyclic coordinate descent. The objective is convex, guaranteeing a global minimum.
        </p>
      </section>

      <section>
        <h2 className="text-1xl font-semibold mt-6">  Summary</h2>
        <div className="overflow-x-hidden mt-2">
          <table className="w-full border border-collapse border-gray-300 text-sm sm:text-base break-words">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-3 py-2 text-left">Aspect</th>
                <th className="border px-3 py-2 text-left">Elastic Net</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-3 py-2 text-left">Model Type</td>
                <td className="border px-3 py-2 text-left">Linear Regression + L1 + L2</td>
              </tr>
              <tr>
                <td className="border px-3 py-2 text-left">Key Hyperparameters</td>
                <td className="border px-3 py-2 text-left">α, l1_ratio</td>
              </tr>
              <tr>
                <td className="border px-3 py-2 text-left">Feature Selection</td>
                <td className="border px-3 py-2 text-left"> Yes</td>
              </tr>
              <tr>
                <td className="border px-3 py-2 text-left">Handles Multicollinearity</td>
                <td className="border px-3 py-2 text-left"> Yes</td>
              </tr>
              <tr>
                <td className="border px-3 py-2 text-left">Suitable for High-Dimensional Data</td>
                <td className="border px-3 py-2 text-left"> Yes</td>
              </tr>
              <tr>
                <td className="border px-3 py-2 text-left">Scikit-learn Classes</td>
                <td className="border px-3 py-2 text-left">ElasticNet, ElasticNetCV</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};

export default ElasticNetRegression;
