//import React from "react";

export default function MultipleLinearRegression() {
  return (
   <div className="max-w-4xl mx-auto p-6 pt-16 space-y-6">
      <h1 className="text-4xl font-bold text-blue-600">Multiple Linear Regression (MLR)</h1>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> What is Multiple Linear Regression?</h2>
        <p>
          Multiple Linear Regression is a <strong>Supervised Learning</strong> algorithm used to model the relationship between:
        </p>
        <ul className="list-disc list-inside ml-6 mb-2">
          <li>Two or more independent variables (<code>X₁, X₂, ..., Xₙ</code>)</li>
          <li>One continuous dependent variable (<code>Y</code>)</li>
        </ul>
        <p>
          It finds the best-fit <em>hyperplane</em> (instead of a line) in an n-dimensional space.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Equation of MLR</h2>
        <p className="mb-2">
          <strong>Y = w₁X₁ + w₂X₂ + ... + wₙXₙ + b</strong>
        </p>
        <p>Where:</p>
        <ul className="list-disc list-inside ml-6">
          <li><code>Y</code>: Target/output</li>
          <li><code>Xᵢ</code>: Independent feature variables</li>
          <li><code>wᵢ</code>: Coefficients (slopes)</li>
          <li><code>b</code>: Intercept (bias)</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Example: Predicting House Price</h2>
        <p><strong>Features (Inputs):</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li><code>X₁</code>: Area (sq ft)</li>
          <li><code>X₂</code>: Number of bedrooms</li>
          <li><code>X₃</code>: Distance from city</li>
        </ul>
        <p><strong>Target:</strong></p>
        <ul className="list-disc list-inside ml-6 mb-2">
          <li><code>Y</code>: Price</li>
        </ul>
        <p className="italic text-lg">
          Model might look like:<br />
          Price = 150 × Area + 50000 × Bedrooms − 2000 × Distance + 25000
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Objective</h2>
        <p>Minimize the error between actual and predicted values by minimizing the <strong>Mean Squared Error (MSE)</strong>:</p>
        <p className="italic ml-6">
          MSE = (1/n) ∑<sub>i=1</sub><sup>n</sup> (Yᵢ − Ŷᵢ)²
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Graphical Understanding</h2>
        <ul className="list-disc list-inside ml-6">
          <li>For 1 input: Straight line</li>
          <li>For 2 inputs: Plane in 3D</li>
          <li>For 3+ inputs: Hyperplane in higher dimensions (cannot visualize)</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Assumptions of Multiple Linear Regression</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border border-gray-300 px-3 py-1">Assumption</th>
              <th className="border border-gray-300 px-3 py-1">Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Linearity</td>
              <td className="border border-gray-300 px-3 py-1">Linear relation between Xs and Y</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">No multicollinearity</td>
              <td className="border border-gray-300 px-3 py-1">Inputs should not be strongly correlated with each other</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Homoscedasticity</td>
              <td className="border border-gray-300 px-3 py-1">Constant variance of residuals</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Independence</td>
              <td className="border border-gray-300 px-3 py-1">Observations and errors are independent</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Normality</td>
              <td className="border border-gray-300 px-3 py-1">Residuals are normally distributed</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Advantages of MLR</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Handles multiple features</li>
          <li>Easy to interpret (coefficients show feature importance)</li>
          <li>Efficient for linearly separable data</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Limitations of MLR</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Assumes a linear relationship</li>
          <li>Affected by multicollinearity</li>
          <li>Poor performance on non-linear problems</li>
          <li>Sensitive to outliers</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Interview Questions</h2>
        <h3 className="font-semibold">Basic</h3>
        <ul className="list-disc list-inside ml-6 mb-3">
          <li>What is Multiple Linear Regression?</li>
          <li>How is it different from Simple Linear Regression?</li>
          <li>What does each coefficient represent?</li>
        </ul>
        <h3 className="font-semibold">Intermediate</h3>
        <ul className="list-disc list-inside ml-6 mb-3">
          <li>What is multicollinearity and how do you detect it?<br />
            <em>Answer: When input variables are highly correlated; check using VIF (Variance Inflation Factor).</em>
          </li>
          <li>What evaluation metrics do you use for MLR?<br />
            <em>Answer: MSE, R², RMSE, MAE</em>
          </li>
        </ul>
        <h3 className="font-semibold">Advanced</h3>
        <ul className="list-disc list-inside ml-6">
          <li>How do you interpret a negative coefficient?</li>
          <li>What if the model has a high R² but performs poorly on new data?<br />
            <em>Answer: Overfitting; use regularization methods such as Ridge or Lasso regression.</em>
          </li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Python Example</h2>
        <pre className="bg-gray-100 rounded p-4 overflow-x-auto text-sm">
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

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Evaluation Metrics</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border border-gray-300 px-3 py-1">Metric</th>
              <th className="border border-gray-300 px-3 py-1">Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-3 py-1">R² Score</td>
              <td className="border border-gray-300 px-3 py-1">Variance explained by model</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Adjusted R²</td>
              <td className="border border-gray-300 px-3 py-1">Adjusted for number of features</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">MSE</td>
              <td className="border border-gray-300 px-3 py-1">Mean Squared Error</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">RMSE</td>
              <td className="border border-gray-300 px-3 py-1">Root Mean Squared Error</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">MAE</td>
              <td className="border border-gray-300 px-3 py-1">Mean Absolute Error</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">When to Use Multiple Linear Regression</h2>
        <p><strong>Use MLR if:</strong></p>
        <ul className="list-disc list-inside ml-6 mb-3">
          <li>You have multiple numerical input features</li>
          <li>Relationship between inputs and output is approximately linear</li>
          <li>You need interpretability (feature weights)</li>
        </ul>
        <p><strong>Avoid MLR if:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Relationship is non-linear</li>
          <li>Features are highly correlated (consider PCA or remove features)</li>
          <li>You expect complex interactions (use tree models instead)</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Summary</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <tbody>
            <tr>
              <td className="border border-gray-300 px-3 py-1 font-semibold">Feature</td>
              <td className="border border-gray-300 px-3 py-1 font-semibold">Description</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Input</td>
              <td className="border border-gray-300 px-3 py-1">Multiple features (X₁, X₂, ..., Xₙ)</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Output</td>
              <td className="border border-gray-300 px-3 py-1">One continuous target (Y)</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Model Equation</td>
              <td className="border border-gray-300 px-3 py-1">Y = w₁X₁ + w₂X₂ + ... + b</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Best For</td>
              <td className="border border-gray-300 px-3 py-1">Predicting continuous outputs using many inputs</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Model Type</td>
              <td className="border border-gray-300 px-3 py-1">Linear model (Hyperplane)</td>
            </tr>
          </tbody>
        </table>
      </section>

       
    
      <h1 className="text-4xl font-bold mb-6">Multiple Linear Regression: Step-by-Step Python Tutorial</h1>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Import Libraries</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error`}
        </pre>
        <p className="mt-2">Import libraries before use.</p>
        <p>Use <code>pandas</code> for data handling, <code>scikit-learn</code> for modeling.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Load and Explore Data</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`df = pd.read_csv("data.csv")
print(df.head())`}
        </pre>
        <p> Check data shape and sample rows.</p>
        <p>Ensure there are no missing values or handle them:</p>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`df.dropna()
# or
df.fillna(value)`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3">Feature Selection & Target Variable</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`X = df[["Area", "Bedrooms", "Distance"]]
y = df["Price"]`}
        </pre>
        <p>Separate independent variables (<code>X</code>) and dependent variable (<code>y</code>).</p>
        <p> Only include numeric or encoded features in <code>X</code>.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Train-Test Split</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)`}
        </pre>
        <p>Split data into training and testing sets.</p>
        <p> Fix <code>random_state</code> for reproducibility.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Model Initialization and Training</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`model = LinearRegression()
model.fit(X_train, y_train)`}
        </pre>
        <p> Use <code>LinearRegression()</code> from <code>sklearn.linear_model</code>.</p>
        <p> Call <code>fit()</code> on training data.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Model Prediction</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`y_pred = model.predict(X_test)`}
        </pre>
        <p> Use <code>predict()</code> to generate predictions on test data.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Evaluation</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))`}
        </pre>
        <p> Use R² score to measure model accuracy (closer to 1 is better).</p>
        <p> Print MSE or RMSE for error measurement.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Model Interpretation</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)`}
        </pre>
        <p> Check which features impact the output most.</p>
        <p> Positive coefficient means feature increases <code>Y</code>.</p>
        <p> Negative coefficient means feature decreases <code>Y</code>.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Visualize (Optional but Helpful)</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()`}
        </pre>
        <p> Visualization helps spot large errors and overall trends.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3"> Final Code Template</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
{`import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("housing.csv")

# Define features and target
X = df[["Area", "Bedrooms", "Distance"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Model interpretation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-3">Summary: What to Check at Each Step</h2>
        <table className="table-auto border border-gray-300 w-full text-left">
          <thead>
            <tr className="bg-gray-200">
              <th className="border border-gray-300 px-3 py-1">Step</th>
              <th className="border border-gray-300 px-3 py-1">What to Check / Do</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Load Data</td>
              <td className="border border-gray-300 px-3 py-1">Clean, no missing, correct columns</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Select Features</td>
              <td className="border border-gray-300 px-3 py-1">Only numeric or encoded features</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Train-Test Split</td>
              <td className="border border-gray-300 px-3 py-1">Proper splitting, reproducibility</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Train Model</td>
              <td className="border border-gray-300 px-3 py-1">Use .fit() properly</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Predict</td>
              <td className="border border-gray-300 px-3 py-1">Use .predict() on test data</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Evaluate</td>
              <td className="border border-gray-300 px-3 py-1">Use r2_score(), mean_squared_error()</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Interpret Model</td>
              <td className="border border-gray-300 px-3 py-1">Coefficients + Intercept = feature impact</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Visualize</td>
              <td className="border border-gray-300 px-3 py-1">Optional, helps insights</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
