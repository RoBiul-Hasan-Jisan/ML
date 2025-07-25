//import React from "react";

export default function PolynomialRegression() {
  return (
    <div className="max-w-4xl mx-auto p-6 pt-16 space-y-6">
    <section className="p-6 max-w-4xl mx-auto space-y-6 text-gray-800">
      <h1 className="text-4xl font-bold text-blue-600">Polynomial Regression </h1>

      {/* 1. What is Polynomial Regression? */}
      <div>
        <h2 className="text-2xl font-semibold">  What is Polynomial Regression?</h2>
        <p>
          Polynomial Regression is a Supervised Machine Learning algorithm used when the relationship between the independent variable (X) and the dependent variable (Y) is non-linear (curved), but can be represented as a polynomial equation.
        </p>
        <p>It fits a curve rather than a straight line.</p>
      </div>

      {/* 2. Equation */}
      <div>
        <h2 className="text-2xl font-semibold">Equation of Polynomial Regression</h2>
        <p className="font-mono">
          Y = b₀ + b₁X + b₂X² + b₃X³ + ⋯ + bₙXⁿ
        </p>
        <ul className="list-disc list-inside">
          <li><strong>Y</strong>: predicted output</li>
          <li><strong>X</strong>: input feature</li>
          <li><strong>b₀, b₁, ..., bₙ</strong>: model coefficients</li>
          <li><strong>n</strong>: degree of the polynomial</li>
        </ul>
      </div>

      {/* 3. Why Use */}
      <div>
        <h2 className="text-2xl font-semibold">Why Use Polynomial Regression?</h2>
        <p>Linear regression fails on curved data.</p>
        <p>Polynomial Regression models non-linear patterns using a transformed linear model.</p>
      </div>

      {/* 4. Example */}
      <div>
        <h2 className="text-2xl font-semibold"> Example Use Case</h2>
        <p> Predicting Car Price Based on Age:</p>
        <table className="table-auto border border-collapse border-gray-400 mt-2">
          <thead>
            <tr className="bg-gray-200">
              <th className="px-4 py-2 border">Car Age (years)</th>
              <th className="px-4 py-2 border">Price ($)</th>
            </tr>
          </thead>
          <tbody>
            {[["1", "30,000"], ["2", "25,000"], ["3", "20,000"], ["4", "17,000"], ["5", "16,500"]].map(([age, price]) => (
              <tr key={age}>
                <td className="px-4 py-2 border">{age}</td>
                <td className="px-4 py-2 border">{price}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 5. Visualization */}
      <div>
        <h2 className="text-2xl font-semibold"> Visualization</h2>
        <p>
          Polynomial Regression can create U-shaped, inverted-U, or complex curves depending on the degree.
        </p>
        <p>Higher degree = more flexibility, but also risk of overfitting.</p>
      </div>

      {/* 6. Degree Meaning */}
      <div>
        <h2 className="text-2xl font-semibold">Degree of the Polynomial</h2>
        <ul className="list-disc list-inside">
          <li>Degree 1 → Simple Linear Regression (straight line)</li>
          <li>Degree 2 → Quadratic (parabola)</li>
          <li>Degree 3+ → Cubic, Quartic, etc.</li>
          <li>Too high degree → Overfitting (fits noise)</li>
        </ul>
      </div>

      {/* 7. Pros and 8. Cons */}
      <div>
        <h2 className="text-2xl font-semibold"> Advantages</h2>
        <ul className="list-disc list-inside">
          <li>Captures non-linear relationships</li>
          <li>Easy to implement</li>
          <li>Extends linear regression framework</li>
        </ul>

        <h2 className="text-2xl font-semibold mt-4"> Disadvantages</h2>
        <ul className="list-disc list-inside">
          <li>Overfitting with high-degree polynomials</li>
          <li>Very sensitive to outliers</li>
          <li>May generalize poorly to new data</li>
        </ul>
      </div>

      {/* 9. Interview Questions */}
      <div>
        <h2 className="text-2xl font-semibold">Interview Questions</h2>
        <p className="font-semibold">Basic:</p>
        <ul className="list-disc list-inside">
          <li>What is Polynomial Regression?</li>
          <li>When to use it over Linear Regression?</li>
          <li>What does the degree of the polynomial mean?</li>
        </ul>
        <p className="font-semibold mt-2">Advanced:</p>
        <ul className="list-disc list-inside">
          <li>What is the risk of using high-degree polynomials?</li>
          <li>Can Polynomial Regression be used with multiple features?</li>
          <li>Is it linear or non-linear? (Trick: Linear in coefficients, non-linear in inputs)</li>
        </ul>
      </div>

      {/* 10. Python Example */}
      <div>
        <h2 className="text-2xl font-semibold"> Python Code Example</h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([30000, 25000, 20000, 17000, 16500])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_test = np.linspace(1, 5, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Polynomial Fit')
plt.xlabel("Car Age (years)")
plt.ylabel("Price ($)")
plt.title("Polynomial Regression (Degree 2)")
plt.legend()
plt.show()`}
        </pre>
      </div>

      {/* Summary */}
      <div className="border-t pt-4">
        <h2 className="text-2xl font-semibold"> Summary</h2>
        <ul className="list-disc list-inside">
          <li><strong>Type:</strong> Regression (supervised learning)</li>
          <li><strong>Model Equation:</strong> Y = b₀ + b₁X + b₂X² + ... + bₙXⁿ</li>
          <li><strong>Use Case:</strong> Modeling non-linear relationships</li>
          <li><strong>Degree:</strong> Controls curve flexibility</li>
          <li><strong>Risk:</strong> Overfitting with high-degree polynomials</li>
        </ul>
      </div>

    </section>




 <section className="p-6 max-w-4xl mx-auto text-gray-800 space-y-6">
      <h1 className="text-3xl font-bold text-blue-600"> Polynomial Regression – Step-by-Step Guide</h1>

      {/* Step 1 */}
      <div>
        <h2 className="text-2xl font-semibold"> Import Libraries</h2>
        <p>These are the tools you'll need:</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures`}
        </pre>
      </div>

      {/* Step 2 */}
      <div>
        <h2 className="text-2xl font-semibold"> Prepare the Dataset</h2>
        <p>Your input (X) must be 2D and the target (y) must be 1D.</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`# Example dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([30000, 25000, 20000, 17000, 16500])`}
        </pre>
      </div>

      {/* Step 3 */}
      <div>
        <h2 className="text-2xl font-semibold">Transform Input Features to Polynomial</h2>
        <p>Choose the degree (e.g., 2 for quadratic, 3 for cubic):</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`poly = PolynomialFeatures(degree=2)  # Change degree as needed
X_poly = poly.fit_transform(X)`}
        </pre>
        <p> This expands the features to include X², X³, etc.</p>
      </div>

      {/* Step 4 */}
      <div>
        <h2 className="text-2xl font-semibold">Train the Model</h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`model = LinearRegression()
model.fit(X_poly, y)`}
        </pre>
      </div>

      {/* Step 5 */}
      <div>
        <h2 className="text-2xl font-semibold">  Make Predictions</h2>
        <p>On training data:</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`y_pred = model.predict(X_poly)`}
        </pre>
        <p>Or on new data:</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`X_new = np.array([6]).reshape(-1, 1)
X_new_poly = poly.transform(X_new)
print("Predicted value:", model.predict(X_new_poly))`}
        </pre>
      </div>

      {/* Step 6 */}
      <div>
        <h2 className="text-2xl font-semibold">  Plot the Curve </h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`X_curve = np.linspace(1, 6, 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)
y_curve = model.predict(X_curve_poly)

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_curve, y_curve, color='red', label='Polynomial Regression')
plt.xlabel("X (e.g., Car Age)")
plt.ylabel("Y (e.g., Price)")
plt.title("Polynomial Regression Curve")
plt.legend()
plt.show()`}
        </pre>
      </div>

      {/* Step 7 */}
      <div>
        <h2 className="text-2xl font-semibold"> Evaluate the Model </h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))`}
        </pre>
      </div>

      {/* Summary Table */}
      <div className="border-t pt-4">
        <h2 className="text-2xl font-semibold">Summary Table of Steps</h2>
        <table className="w-full text-left table-auto border border-collapse mt-2">
          <thead>
            <tr className="bg-blue-100">
              <th className="border px-3 py-2">Step</th>
              <th className="border px-3 py-2">Description</th>
              <th className="border px-3 py-2">Code Example</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-3 py-2">1</td>
              <td className="border px-3 py-2">Import libraries</td>
              <td className="border px-3 py-2"><code>import numpy as np ...</code></td>
            </tr>
            <tr>
              <td className="border px-3 py-2">2</td>
              <td className="border px-3 py-2">Define X and y</td>
              <td className="border px-3 py-2"><code>X = np.array(...)</code></td>
            </tr>
            <tr>
              <td className="border px-3 py-2">3</td>
              <td className="border px-3 py-2">Transform with PolynomialFeatures</td>
              <td className="border px-3 py-2"><code>poly.fit_transform(X)</code></td>
            </tr>
            <tr>
              <td className="border px-3 py-2">4</td>
              <td className="border px-3 py-2">Fit model</td>
              <td className="border px-3 py-2"><code>model.fit(X_poly, y)</code></td>
            </tr>
            <tr>
              <td className="border px-3 py-2">5</td>
              <td className="border px-3 py-2">Predict</td>
              <td className="border px-3 py-2"><code>model.predict(...)</code></td>
            </tr>
            <tr>
              <td className="border px-3 py-2">6</td>
              <td className="border px-3 py-2">Plot the curve</td>
              <td className="border px-3 py-2"><code>matplotlib.pyplot</code></td>
            </tr>
            <tr>
              <td className="border px-3 py-2">7</td>
              <td className="border px-3 py-2">Evaluate (MSE, R²)</td>
              <td className="border px-3 py-2"><code>r2_score, mean_squared_error</code></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    </div>
  );
}
