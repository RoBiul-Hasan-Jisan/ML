const RidgeRegression = () => {
  return (
    <div className="max-w-full sm:max-w-3xl md:max-w-5xl mx-auto p-4 sm:p-6 pt-16 space-y-6">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">Ridge Regression</h1>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">What is Ridge Regression?</h2>
      <p className="text-sm sm:text-base">
        Ridge Regression is a type of regularized linear regression that addresses multicollinearity
        — when predictor variables are highly correlated, leading to unstable and large coefficients.
        It introduces L2 regularization to shrink the coefficients and improve model generalization.
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Key Concepts</h2>
      <ul className="list-disc ml-5 text-sm sm:text-base">
        <li><strong>Regularization:</strong> Penalizes large coefficients to reduce overfitting.</li>
        <li><strong>L2 Penalty:</strong> Adds squared coefficient magnitudes to the loss.</li>
        <li><strong>Shrinkage:</strong> Coefficients are pushed toward zero (but never exactly zero).</li>
      </ul>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Mathematical Formulation</h2>
      <pre className="bg-gray-100 p-4 rounded text-xs sm:text-sm overflow-x-auto w-full whitespace-pre-wrap">
{`Objective:
   minimize β ( ||y - Xβ||² + λ||β||² )
     Where:
        - ||y - Xβ||² = least squares loss
        - λ||β||² = L2 regularization term
        - λ (alpha in sklearn) controls regularization strength`}
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Intuition</h2>
      <p className="text-sm sm:text-base">
        Unlike ordinary least squares which only minimizes prediction error,
        Ridge minimizes both prediction error and a penalty on large coefficients.
        This introduces bias but reduces variance and overfitting.
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Comparison with Other Methods</h2>
      <div className="overflow-x-auto w-full">
        <table className="min-w-full table-auto border border-collapse border-gray-300 mt-4 text-xs sm:text-sm">
          <thead>
            <tr className="bg-gray-200">
              <th className="border px-2 py-1 text-left">Method</th>
              <th className="border px-2 py-1 text-left">Penalty</th>
              <th className="border px-2 py-1 text-left">Can Coefficients Be Zero?</th>
              <th className="border px-2 py-1 text-left">Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-2 py-1">OLS</td>
              <td className="border px-2 py-1">None</td>
              <td className="border px-2 py-1">❌</td>
              <td className="border px-2 py-1">Simple linear models</td>
            </tr>
            <tr>
              <td className="border px-2 py-1">Ridge</td>
              <td className="border px-2 py-1">L2</td>
              <td className="border px-2 py-1">❌</td>
              <td className="border px-2 py-1">Multicollinearity</td>
            </tr>
            <tr>
              <td className="border px-2 py-1">Lasso</td>
              <td className="border px-2 py-1">L1</td>
              <td className="border px-2 py-1">✅</td>
              <td className="border px-2 py-1">Feature selection</td>
            </tr>
            <tr>
              <td className="border px-2 py-1">ElasticNet</td>
              <td className="border px-2 py-1">L1 + L2</td>
              <td className="border px-2 py-1">✅</td>
              <td className="border px-2 py-1">Mixed Lasso + Ridge</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Python Example (scikit-learn)</h2>
      <pre className="bg-gray-100 p-4 rounded text-xs sm:text-sm overflow-x-auto w-full whitespace-pre-wrap">
{`from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1) * 10
y = 3 * X.flatten() + np.random.randn(100) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.title("Ridge Regression")
plt.show()`}
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Use Cases</h2>
      <ul className="list-disc ml-5 text-sm sm:text-base">
        <li>When features are highly correlated</li>
        <li>High-dimensional datasets (p &gt; n)</li>
        <li>Preventing overfitting in linear models</li>
      </ul>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Advantages</h2>
      <ul className="list-disc ml-5 text-sm sm:text-base">
        <li>Reduces model complexity</li>
        <li>Retains all features (unlike Lasso)</li>
        <li>Only one hyperparameter (alpha) to tune</li>
      </ul>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Disadvantages</h2>
      <ul className="list-disc ml-5 text-sm sm:text-base">
        <li>Does not perform feature selection</li>
        <li>Requires feature scaling</li>
      </ul>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">🔧 Hyperparameter Tuning</h2>
      <pre className="bg-gray-100 p-4 rounded text-xs sm:text-sm overflow-x-auto w-full whitespace-pre-wrap">
{`from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_train, y_train)

print("Best alpha:", ridge_cv.alpha_)`}
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">Summary</h2>
      <table className="table-auto border border-collapse border-gray-300 mt-4 w-full text-xs sm:text-sm">
        <thead>
          <tr className="bg-gray-200">
            <th className="border px-3 py-2">Aspect</th>
            <th className="border px-3 py-2">Ridge Regression</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-3 py-2">Type</td>
            <td className="border px-3 py-2">Linear Model with L2 Regularization</td>
          </tr>
          <tr>
            <td className="border px-3 py-2">Goal</td>
            <td className="border px-3 py-2">Reduce overfitting &amp; handle multicollinearity</td>
          </tr>
          <tr>
            <td className="border px-3 py-2">Key Parameter</td>
            <td className="border px-3 py-2">alpha</td>
          </tr>
          <tr>
            <td className="border px-3 py-2">Feature Selection?</td>
            <td className="border px-3 py-2">❌</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default RidgeRegression;
