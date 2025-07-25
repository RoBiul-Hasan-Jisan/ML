//import React from "react";

const SVROverview = () => {
  return (
        <div className="max-w-5xl mx-auto p-6 pt-16 space-y-6">
      <h1 className="text-3xl font-bold mb-4">Support Vector Regression </h1>

      <section>
        <h2 className="text-2xl font-semibold mt-4">Introduction</h2>
        <p>
          Support Vector Regression (SVR) is an adaptation of Support Vector Machines (SVM) for regression tasks.
          While SVM is widely used for classification, SVR estimates a continuous function <code>f(x)</code> that
          predicts a real-valued target variable.
        </p>
        <p>
          SVR aims to find a function that approximates training data within a margin of tolerance <code>ε</code>,
          while being as flat as possible.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Key Concepts</h2>
        <ul className="list-disc pl-6">
          <li><b>ε-insensitive tube:</b> A zone around the regression line where no penalty is given for prediction error.</li>
          <li><b>Support Vectors:</b> Data points that lie outside the ε-tube; these determine the model.</li>
          <li><b>Flatness:</b> Keeping the weight vector <code>w</code> small minimizes model complexity.</li>
          <li><b>Slack variables:</b> <code>ξᵢ, ξᵢ*</code> allow flexibility for points outside the ε-tube (soft margin).</li>
          <li><b>Kernel Trick:</b> A technique to transform data to higher dimensions to capture nonlinearity.</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> SVR Objective: Mathematical Formulation</h2>
     <p>
  {"Given a dataset D = {(x₁ , y₁), ..., (xₙ, yₙ)}, we want to find:"}
  <br />
  <code>{"f(x) = wᵀx + b"} with {"|yᵢ - f(xᵢ)| ≤ ε"} for most points.</code>
</p>


        <p>
          Objective:
          <br />
          <code>
            minimize (1/2)‖w‖² + C Σ (ξᵢ + ξᵢ*) <br />
            subject to:
            <ul className="list-disc pl-6">
              <li>yᵢ - wᵀxᵢ - b ≤ ε + ξᵢ</li>
              <li>wᵀxᵢ + b - yᵢ ≤ ε + ξᵢ*</li>
              <li>ξᵢ, ξᵢ* ≥ 0</li>
            </ul>
          </code>
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Kernel Functions in SVR</h2>
        <table className="w-full table-auto border mt-2">
          <thead>
            <tr className="bg-gray-200">
              <th className="border px-4 py-2">Kernel</th>
              <th className="border px-4 py-2">Formula</th>
              <th className="border px-4 py-2">Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-4 py-2">Linear</td>
              <td className="border px-4 py-2">K(xᵢ, xⱼ) = xᵢᵀxⱼ</td>
              <td className="border px-4 py-2">Linearly separable data</td>
            </tr>
            <tr>
              <td className="border px-4 py-2">Polynomial</td>
              <td className="border px-4 py-2">K(xᵢ, xⱼ) = (γxᵢᵀxⱼ + r)ᵈ</td>
              <td className="border px-4 py-2">Polynomial relationships</td>
            </tr>
            <tr>
              <td className="border px-4 py-2">RBF (Gaussian)</td>
              <td className="border px-4 py-2">K(xᵢ, xⱼ) = exp(−γ‖xᵢ − xⱼ‖²)</td>
              <td className="border px-4 py-2">Nonlinear, most common</td>
            </tr>
            <tr>
              <td className="border px-4 py-2">Sigmoid</td>
              <td className="border px-4 py-2">K(xᵢ, xⱼ) = tanh(γxᵢᵀxⱼ + r)</td>
              <td className="border px-4 py-2">Neural network-like</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4">Python Example with Scikit-learn</h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.5, 1.7, 3.2, 3.8, 5.1])

model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='SVR Prediction')
plt.title("SVR Prediction")
plt.legend()
plt.show()`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Advantages</h2>
        <ul className="list-disc pl-6">
          <li> Works for both linear and nonlinear regression.</li>
          <li> Robust to outliers due to ε-insensitive loss.</li>
          <li> Flexible with different kernel choices.</li>
          <li> Handles high-dimensional data well.</li>
        </ul>

        <h2 className="text-2xl font-semibold mt-4">Disadvantages</h2>
        <ul className="list-disc pl-6">
          <li> Computationally expensive.</li>
          <li> Needs careful tuning of C, ε, and γ.</li>
          <li> Less interpretable than simpler models.</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Performance Metrics</h2>
        <ul className="list-disc pl-6">
          <li><b>MAE:</b> Mean Absolute Error</li>
          <li><b>MSE:</b> Mean Squared Error</li>
          <li><b>RMSE:</b> Root Mean Squared Error</li>
          <li><b>R² Score:</b> Coefficient of determination</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Hyperparameter Tuning</h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVR(), param_grid, cv=5)
grid.fit(X, y)

print("Best parameters:", grid.best_params_)`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Advanced Topics</h2>
        <ul className="list-disc pl-6">
          <li>Dual Problem: Solved using Lagrange multipliers</li>
          <li>KKT Conditions: Optimality conditions</li>
          <li>Custom Kernels: Domain-specific kernels</li>
          <li>SVR in Time Series: Forecasting & signal smoothing</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> When to Use SVR?</h2>
        <ul className="list-disc pl-6">
          <li>You need robust regression that can ignore small errors.</li>
          <li>Data has nonlinear trends and other models overfit.</li>
          <li>Interpretability is not a top concern.</li>
          <li>Dataset is small to medium (SVR is slow on large data).</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Real-World Applications</h2>
        <ul className="list-disc pl-6">
          <li>Stock price prediction</li>
          <li>House price estimation</li>
          <li>Weather and demand forecasting</li>
          <li>Signal smoothing and denoising</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mt-4"> Summary Table</h2>
        <table className="w-full table-auto border mt-2">
          <tbody>
            <tr>
              <td className="border px-4 py-2 font-medium">Type</td>
              <td className="border px-4 py-2">Supervised Regression</td>
            </tr>
            <tr>
              <td className="border px-4 py-2 font-medium">Based On</td>
              <td className="border px-4 py-2">Support Vector Machine</td>
            </tr>
            <tr>
              <td className="border px-4 py-2 font-medium">Strengths</td>
              <td className="border px-4 py-2">Robust, effective for non-linear problems</td>
            </tr>
            <tr>
              <td className="border px-4 py-2 font-medium">Weaknesses</td>
              <td className="border px-4 py-2">Slow, sensitive to hyperparameters</td>
            </tr>
            <tr>
              <td className="border px-4 py-2 font-medium">Key Parameters</td>
              <td className="border px-4 py-2">C, epsilon, kernel, gamma</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
};

export default SVROverview;
