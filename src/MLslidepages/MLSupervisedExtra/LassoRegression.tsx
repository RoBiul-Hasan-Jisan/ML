//import React from "react";

const LassoRegression = () => {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-5xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold mb-4  text-blue-600">Lasso Regression</h1>

      <p>
        <strong>What is Lasso Regression?</strong>
        <br />
        Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique using <strong>L1 regularization</strong> to:
      </p>
      <ul className="list-disc list-inside my-2 ml-4">
        <li>Prevent overfitting</li>
        <li>Perform automatic feature selection (set some coefficients to exactly zero)</li>
      </ul>

      <h2 className="text-1xl font-semibold mt-6"> Key Concepts</h2>
      <div className="overflow-x-hidden mt-2">
        <table className="w-full border border-collapse border-gray-300 text-sm sm:text-base break-words">
          <thead className="bg-gray-200">
            <tr>
              <th className="border px-3 py-2 text-left">Term</th>
              <th className="border px-3 py-2 text-left">Explanation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-3 py-2">Regularization</td>
              <td className="border px-3 py-2">Penalizes large coefficients to reduce model complexity</td>
            </tr>
            <tr>
              <td className="border px-3 py-2">L1 Penalty</td>
              <td className="border px-3 py-2">Sum of absolute values of coefficients</td>
            </tr>
            <tr>
              <td className="border px-3 py-2">Feature Selection</td>
              <td className="border px-3 py-2">Can force some coefficients to become exactly zero</td>
            </tr>
            <tr>
              <td className="border px-3 py-2">Sparsity</td>
              <td className="border px-3 py-2">Model retains only most important features</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-1xl font-semibold mt-6"> Mathematical Formulation</h2>
      <p>Objective Function:</p>
      <pre className="bg-gray-100 p-2 rounded overflow-hidden whitespace-pre-wrap break-words text-sm">
        <code>
          min(‖y − Xβ‖² + λ ∑ |β<sub>j</sub>|)
        </code>
      </pre>

      <h2 className="text-1xl font-semibold mt-6"> Intuition</h2>
      <p>
        Lasso minimizes both the prediction error and the total weight of coefficients.
        As <code>λ</code> increases, more coefficients shrink to zero, simplifying the model.
      </p>

      <h2 className="text-1xl font-semibold mt-6"> Comparison with Other Methods</h2>
      <div className="overflow-x-hidden mt-2">
        <table className="w-full border border-collapse border-gray-300 text-sm sm:text-base break-words">
          <thead className="bg-gray-200">
            <tr>
              <th className="border px-2 py-2 text-left">Method</th>
              <th className="border px-2 py-2 text-left">Penalty</th>
              <th className="border px-2 py-2 text-left">Coefficients = 0?</th>
              <th className="border px-2 py-2 text-left">Feature Selection</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-2 py-2">OLS</td>
              <td className="border px-2 py-2">None</td>
              <td className="border px-2 py-2">❌</td>
              <td className="border px-2 py-2">❌</td>
            </tr>
            <tr>
              <td className="border px-2 py-2">Ridge</td>
              <td className="border px-2 py-2">L2</td>
              <td className="border px-2 py-2">❌</td>
              <td className="border px-2 py-2">❌</td>
            </tr>
            <tr>
              <td className="border px-2 py-2">Lasso</td>
              <td className="border px-2 py-2">L1</td>
              <td className="border px-2 py-2">✅</td>
              <td className="border px-2 py-2">✅</td>
            </tr>
            <tr>
              <td className="border px-2 py-2">ElasticNet</td>
              <td className="border px-2 py-2">L1 + L2</td>
              <td className="border px-2 py-2">✅</td>
              <td className="border px-2 py-2">✅</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-1xl font-semibold mt-6"> Python Example with scikit-learn</h2>
      <pre className="bg-gray-900 text-white text-sm p-4 rounded overflow-hidden whitespace-pre-wrap break-words">
        <code>
{`from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Data
np.random.seed(0)
X = np.random.randn(100, 10)
true_coef = np.array([1.5, 0, 0, -3.0, 0, 0, 0, 0, 2.0, 0])
y = X @ true_coef + np.random.randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficients:", lasso.coef_)`}
        </code>
      </pre>

      <h2 className="text-1xl font-semibold mt-6"> Advantages</h2>
      <ul className="list-disc list-inside ml-4 my-2">
        <li>Performs automatic feature selection</li>
        <li>Reduces overfitting</li>
        <li>Simpler and more interpretable models</li>
      </ul>

      <h2 className="text-1xl font-semibold mt-6"> Disadvantages</h2>
      <ul className="list-disc list-inside ml-4 my-2">
        <li>Can perform poorly when features are highly correlated</li>
        <li>Can be unstable — small data changes affect feature selection</li>
      </ul>

      <h2 className="text-1xl font-semibold mt-6"> Hyperparameter Tuning</h2>
      <pre className="bg-gray-900 text-white text-sm p-4 rounded overflow-hidden whitespace-pre-wrap break-words">
        <code>
{`from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5)
lasso_cv.fit(X_train, y_train)

print("Best alpha:", lasso_cv.alpha_)`}
        </code>
      </pre>

      <h2 className="text-1xl font-semibold mt-6"> Real-World Applications</h2>
      <ul className="list-disc list-inside ml-4 my-2">
        <li>Genomics: Select a few genes from thousands</li>
        <li>Finance: Key predictors for stock returns</li>
        <li>Marketing: Identify influential customer behaviors</li>
      </ul>

      <h2 className="text-1xl font-semibold mt-6"> Summary</h2>
      <div className="overflow-x-hidden mt-2">
        <table className="w-full border border-collapse border-gray-300 text-sm sm:text-base break-words">
          <thead className="bg-gray-200">
            <tr>
              <th className="border px-3 py-2 text-left">Aspect</th>
              <th className="border px-3 py-2 text-left">Lasso Regression</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-3 py-2 text-left">Type</td>
              <td className="border px-3 py-2 text-left">Linear with L1 Regularization</td>
            </tr>
            <tr>
              <td className="border px-3 py-2 text-left">Goal</td>
              <td className="border px-3 py-2 text-left">Shrink coefficients, remove irrelevant features</td>
            </tr>
            <tr>
              <td className="border px-3 py-2 text-left">Feature Selection</td>
              <td className="border px-3 py-2 text-left"> Yes</td>
            </tr>
            <tr>
              <td className="border px-3 py-2 text-left">Scaling Needed</td>
              <td className="border px-3 py-2 text-left"> Yes</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default LassoRegression;
