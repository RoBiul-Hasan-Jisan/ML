//import React from "react";

export default function MultipleLinearRegression() {
  return (
    <div className="max-w-full sm:max-w-3xl mx-auto px-4 py-6 sm:px-6 lg:px-8 space-y-6">
      <h1 className="text-2xl sm:text-4xl font-bold text-blue-600">Multiple Linear Regression</h1>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">What is Multiple Linear Regression?</h2>
        <p className="mb-2">
          Multiple Linear Regression is a <strong>Supervised Learning</strong> algorithm used to model the relationship between:
        </p>
        <ul className="list-disc list-inside ml-4 mb-2 space-y-1 text-sm sm:text-base">
          <li>Two or more independent variables (<code className="bg-gray-100 px-1 rounded">X₁, X₂, ..., Xₙ</code>)</li>
          <li>One continuous dependent variable (<code className="bg-gray-100 px-1 rounded">Y</code>)</li>
        </ul>
        <p>
          It finds the best-fit <em>hyperplane</em> (instead of a line) in an n-dimensional space.
        </p>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Equation of MLR</h2>
        <p className="mb-2 text-lg bg-blue-50 p-3 rounded">
          <strong>Y = w₁X₁ + w₂X₂ + ... + wₙXₙ + b</strong>
        </p>
        <p className="mb-1">Where:</p>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li><code className="bg-gray-100 px-1 rounded">Y</code>: Target/output</li>
          <li><code className="bg-gray-100 px-1 rounded">Xᵢ</code>: Independent feature variables</li>
          <li><code className="bg-gray-100 px-1 rounded">wᵢ</code>: Coefficients (slopes)</li>
          <li><code className="bg-gray-100 px-1 rounded">b</code>: Intercept (bias)</li>
        </ul>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Example: Predicting House Price</h2>
        <p className="font-semibold mb-1 text-sm sm:text-base">Features (Inputs):</p>
        <ul className="list-disc list-inside ml-4 mb-2 space-y-1 text-sm sm:text-base">
          <li><code className="bg-gray-100 px-1 rounded">X₁</code>: Area (sq ft)</li>
          <li><code className="bg-gray-100 px-1 rounded">X₂</code>: Number of bedrooms</li>
          <li><code className="bg-gray-100 px-1 rounded">X₃</code>: Distance from city</li>
        </ul>
        <p className="font-semibold mb-1 text-sm sm:text-base">Target:</p>
        <ul className="list-disc list-inside ml-4 mb-3 space-y-1 text-sm sm:text-base">
          <li><code className="bg-gray-100 px-1 rounded">Y</code>: Price</li>
        </ul>
        <p className="italic text-base sm:text-lg bg-gray-50 p-3 rounded">
          Model might look like:<br />
          Price = 150 × Area + 50000 × Bedrooms − 2000 × Distance + 25000
        </p>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Objective</h2>
        <p className="mb-2 text-sm sm:text-base">
          Minimize the error between actual and predicted values by minimizing the <strong>Mean Squared Error (MSE)</strong>:
        </p>
        <p className="italic ml-4 bg-gray-50 p-2 rounded inline-block text-sm sm:text-base">
          MSE = (1/n) ∑<sub>i=1</sub><sup>n</sup> (Yᵢ − Ŷᵢ)²
        </p>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Graphical Understanding</h2>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>For 1 input: Straight line</li>
          <li>For 2 inputs: Plane in 3D</li>
          <li>For 3+ inputs: Hyperplane in higher dimensions (cannot visualize)</li>
        </ul>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Assumptions of Multiple Linear Regression</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-300 text-sm sm:text-base">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-300 px-3 py-2 text-left">Assumption</th>
                <th className="border border-gray-300 px-3 py-2 text-left">Meaning</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 px-3 py-2">Linearity</td>
                <td className="border border-gray-300 px-3 py-2">Linear relation between Xs and Y</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-3 py-2">No multicollinearity</td>
                <td className="border border-gray-300 px-3 py-2">Inputs should not be strongly correlated with each other</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">Homoscedasticity</td>
                <td className="border border-gray-300 px-3 py-2">Constant variance of residuals</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-3 py-2">Independence</td>
                <td className="border border-gray-300 px-3 py-2">Observations and errors are independent</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">Normality</td>
                <td className="border border-gray-300 px-3 py-2">Residuals are normally distributed</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Advantages of MLR</h2>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>Handles multiple features</li>
          <li>Easy to interpret (coefficients show feature importance)</li>
          <li>Efficient for linearly separable data</li>
        </ul>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Limitations of MLR</h2>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>Assumes a linear relationship</li>
          <li>Affected by multicollinearity</li>
          <li>Poor performance on non-linear problems</li>
          <li>Sensitive to outliers</li>
        </ul>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Interview Questions</h2>

        <h3 className="font-semibold text-base sm:text-lg mt-4 mb-2">Basic</h3>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>What is Multiple Linear Regression?</li>
          <li>How is it different from Simple Linear Regression?</li>
          <li>What does each coefficient represent?</li>
        </ul>

        <h3 className="font-semibold text-base sm:text-lg mt-4 mb-2">Intermediate</h3>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>
            What is multicollinearity and how do you detect it?<br />
            <em className="text-gray-600">Answer: When input variables are highly correlated; check using VIF (Variance Inflation Factor).</em>
          </li>
          <li>
            What evaluation metrics do you use for MLR?<br />
            <em className="text-gray-600">Answer: MSE, R², RMSE, MAE</em>
          </li>
        </ul>

        <h3 className="font-semibold text-base sm:text-lg mt-4 mb-2">Advanced</h3>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>How do you interpret a negative coefficient?</li>
          <li>
            What if the model has a high R² but performs poorly on new data?<br />
            <em className="text-gray-600">Answer: Overfitting; use regularization methods such as Ridge or Lasso regression.</em>
          </li>
        </ul>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Python Example</h2>
        <pre className="bg-gray-100 rounded p-4 overflow-x-auto text-xs sm:text-sm">
{`from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data (Area, Bedrooms, Distance from city)
X = [
    [1000, 2, 10],
    [1500, 3, 5],
    [2000, 4, 8],
    [2500, 3, 3]
]
y = [200000, 300000, 400000, 500000]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
predicted = model.predict([[1800, 3, 7]])
print("Predicted Price:", predicted[0])

# Model Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Model Evaluation (R² Score)
print("R² Score:", r2_score(y, model.predict(X)))
`}
        </pre>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Evaluation Metrics</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-300 text-sm sm:text-base">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-300 px-3 py-2 text-left">Metric</th>
                <th className="border border-gray-300 px-3 py-2 text-left">Meaning</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 px-3 py-2">R² Score</td>
                <td className="border border-gray-300 px-3 py-2">Variance explained by model</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-3 py-2">Adjusted R²</td>
                <td className="border border-gray-300 px-3 py-2">Adjusted for number of features</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">MSE</td>
                <td className="border border-gray-300 px-3 py-2">Mean Squared Error</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-3 py-2">RMSE</td>
                <td className="border border-gray-300 px-3 py-2">Root Mean Squared Error</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">MAE</td>
                <td className="border border-gray-300 px-3 py-2">Mean Absolute Error</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">When to Use Multiple Linear Regression</h2>
        <p className="font-semibold mb-1 text-sm sm:text-base">Use MLR if:</p>
        <ul className="list-disc list-inside ml-4 mb-3 space-y-1 text-sm sm:text-base">
          <li>You have multiple numerical input features</li>
          <li>Relationship between inputs and output is approximately linear</li>
          <li>You need interpretability (feature weights)</li>
        </ul>
        <p className="font-semibold mb-1 text-sm sm:text-base">Avoid MLR if:</p>
        <ul className="list-disc list-inside ml-4 space-y-1 text-sm sm:text-base">
          <li>Relationship is non-linear</li>
          <li>Features are highly correlated (consider PCA or remove features)</li>
          <li>You expect complex interactions (use tree models instead)</li>
        </ul>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-lg shadow-sm">
        <h2 className="text-lg sm:text-2xl font-semibold mb-3">Summary</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-300 text-sm sm:text-base">
            <tbody>
              <tr className="bg-gray-100">
                <td className="border border-gray-300 px-3 py-2 font-semibold">Feature</td>
                <td className="border border-gray-300 px-3 py-2 font-semibold">Description</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">Input</td>
                <td className="border border-gray-300 px-3 py-2">Multiple features (X₁, X₂, ..., Xₙ)</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-3 py-2">Output</td>
                <td className="border border-gray-300 px-3 py-2">One continuous target (Y)</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">Model Equation</td>
                <td className="border border-gray-300 px-3 py-2">Y = w₁X₁ + w₂X₂ + ... + b</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-3 py-2">Best For</td>
                <td className="border border-gray-300 px-3 py-2">Predicting continuous outputs using many inputs</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-3 py-2">Model Type</td>
                <td className="border border-gray-300 px-3 py-2">Linear model (Hyperplane)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
