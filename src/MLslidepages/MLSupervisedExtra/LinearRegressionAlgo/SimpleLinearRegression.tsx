import React from 'react';

export default function SimpleLinearRegression() {
  return (
   <div className="max-w-full sm:max-w-3xl mx-auto px-4 py-6 sm:px-6 lg:px-8 space-y-6">
      <h1 className="text-3xl sm:text-4xl font-bold text-blue-600">Simple Linear Regression</h1>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">What is Simple Linear Regression?</h2>
        <p>
          Simple Linear Regression is a <strong>Supervised Learning</strong> algorithm used to model the relationship between:
        </p>
        <ul className="list-disc list-inside">
          <li>One independent variable (X)</li>
          <li>One dependent variable (Y)</li>
        </ul>
        <p>
          It assumes a <em>linear relationship</em>, meaning as X increases or decreases, Y also increases or decreases proportionally.
        </p>
        <p>
          <strong>Equation of SLR:</strong> <code>Y = mX + c</code>
        </p>
        <ul>
          <li><code>Y</code> = Predicted output (target)</li>
          <li><code>X</code> = Input feature (predictor)</li>
          <li><code>m</code> = Slope of the line (change in Y per unit X)</li>
          <li><code>c</code> = Intercept (Y when X = 0)</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Example Problem: Predicting House Price Based on Size</h2>

        <div className="overflow-hidden">
          <table className="table-auto border border-gray-300 border-collapse w-full">
            <thead>
              <tr>
                <th className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Area (sq ft)</th>
                <th className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Price ($)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">1000</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">200,000</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">1500</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">300,000</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">2000</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">400,000</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p>The model learns that for every +500 sq ft, price increases by +100,000, so:</p>
        <p><strong>Price = 200 × Area</strong></p>
        <p>If Area = 1800 → Predicted Price = 200 × 1800 = 360,000</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Objective of SLR</h2>
        <p>
          Minimize the difference between actual and predicted values using a <strong>Loss Function</strong> called <strong>Mean Squared Error (MSE)</strong>:
        </p>
        <p className="font-mono bg-gray-100 p-2 rounded break-words whitespace-normal">
          MSE = (1/n) × ∑(Yᵢ - Ŷᵢ)²
        </p>
        <ul>
          <li><code>Yᵢ</code>: Actual value</li>
          <li><code>Ŷᵢ</code>: Predicted value</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Graphical Representation</h2>
        <p>
          The regression line tries to best fit the data points by minimizing the error between actual and predicted values.
        </p>
        <p>
          - X-axis: Input feature (e.g., Area)<br />
          - Y-axis: Target value (e.g., Price)
        </p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Assumptions of Simple Linear Regression</h2>

        <div className="overflow-hidden">
          <table className="table-auto border border-gray-300 border-collapse w-full">
            <thead>
              <tr>
                <th className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Assumption</th>
                <th className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Linearity</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Relationship between X and Y is linear</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Homoscedasticity</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Constant variance of residuals</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Independence of errors</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">No autocorrelation in residuals</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Normality of errors</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Errors are normally distributed</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Advantages of SLR</h2>
        <ul className="list-disc list-inside">
          <li>Easy to implement</li>
          <li>Computationally fast</li>
          <li>Great for linearly correlated data</li>
          <li>Interpretable and explainable</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Limitations of SLR</h2>
        <ul className="list-disc list-inside text-red-600">
          <li>Works only with one input variable</li>
          <li>Assumes perfect linearity</li>
          <li>Sensitive to outliers</li>
          <li>Cannot capture non-linear relationships</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Interview Questions</h2>
        <h3 className="font-semibold">Basic</h3>
        <ul className="list-disc list-inside mb-2">
          <li>What is Simple Linear Regression?</li>
          <li>What does the slope (m) represent?</li>
          <li>What is the difference between actual and predicted values?</li>
          <li>How is the regression line fitted?</li>
        </ul>
        <h3 className="font-semibold">Technical</h3>
        <ul className="list-disc list-inside">
          <li>What is the cost function in Simple Linear Regression?</li>
          <li>What are residuals?</li>
          <li>How do you evaluate model performance?</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Python Example (Using scikit-learn)</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
X = [[1000], [1500], [2000], [2500]]
y = [200000, 300000, 400000, 500000]

# Model training
model = LinearRegression()
model.fit(X, y)

# Predict
area = [[1800]]
predicted_price = model.predict(area)
print("Predicted price:", predicted_price[0])

# Visualization
plt.scatter([x[0] for x in X], y, color='blue')         # Actual points
plt.plot([x[0] for x in X], model.predict(X), color='red')  # Regression line
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("Simple Linear Regression")
plt.show()`}
        </pre>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Evaluation Metrics</h2>
        <ul className="list-disc list-inside">
          <li><strong>MSE:</strong> Mean Squared Error</li>
          <li><strong>RMSE:</strong> Root Mean Squared Error</li>
          <li><strong>MAE:</strong> Mean Absolute Error</li>
          <li><strong>R² Score:</strong> Coefficient of determination</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Summary</h2>
        <div className="overflow-hidden">
          <table className="table-auto border border-gray-300 border-collapse w-full">
            <tbody>
              <tr>
                <td className="border border-gray-300 px-1 py-1 font-semibold text-xs sm:text-sm break-words">Type</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Regression</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 font-semibold text-xs sm:text-sm break-words">Input</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">One feature (X)</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 font-semibold text-xs sm:text-sm break-words">Output</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">One continuous variable (Y)</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 font-semibold text-xs sm:text-sm break-words">Model Equation</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words"><code>Y = mX + c</code></td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 font-semibold text-xs sm:text-sm break-words">Goal</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Minimize prediction error (MSE)</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 font-semibold text-xs sm:text-sm break-words">Limitation</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Can't handle multiple inputs or non-linear patterns</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

          <section>
        <h1 className="text-2xl sm:text-3xl font-bold mb-4">Step-by-Step Guide to Write Simple Linear Regression Code</h1>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Import Required Libraries</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: You need libraries for data manipulation, visualization, modeling, and evaluation.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Load the Dataset</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`data = pd.read_csv('your_dataset.csv')  # Or create manually`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: You need data with one independent variable (X) and one dependent variable (Y).</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Explore the Data</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`print(data.head())
print(data.describe())
print(data.info())`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: To understand data structure, check for missing values or incorrect types.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Visualize the Data (Optional but Recommended)</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`plt.scatter(data['X'], data['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs Y')
plt.show()`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: Helps you visually confirm whether a linear relationship exists.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Prepare the Data</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`X = data[['X']]   # Feature (must be 2D)
y = data['Y']     # Target (1D)`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: Separate features and labels before training.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Split the Data (Training & Testing)</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: To evaluate the model on unseen data.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Train the Model</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`model = LinearRegression()
model.fit(X_train, y_train)`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: Fit the model to learn slope and intercept (line of best fit).</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Get Predictions</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`y_pred = model.predict(X_test)`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: Predict Y values for the test set.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Evaluate the Model</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: Check how well the model performs using metrics.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Visualize the Regression Line</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: To confirm visually how well the regression line fits the data.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">Inspect Model Parameters</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-hidden">
{`print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)`}
        </pre>
        <p className="italic text-sm sm:text-base">Why: Useful for explaining the model and understanding the equation.</p>
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-2">🧠 Summary: What to Look For at Each Step</h2>
        <div className="overflow-hidden">
          <table className="table-auto border border-gray-300 border-collapse w-full">
            <thead>
              <tr>
                <th className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Step</th>
                <th className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">What to Check / Look For</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Load Data</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">One independent and one dependent variable</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Explore Data</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Missing values, data types, linear trends</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Visualize</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Confirm linear pattern (scatter plot)</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Prepare</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Ensure correct input shape for X and y</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Split</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Avoid overfitting by separating test data</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Train</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Use <code>LinearRegression().fit()</code></td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Predict</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Use <code>model.predict()</code></td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Evaluate</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Check MSE, R² score</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Visualize</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Draw regression line on actual data</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words font-semibold">Inspect</td>
                <td className="border border-gray-300 px-1 py-1 text-xs sm:text-sm break-words">Print slope and intercept for insights</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>


    </div>
  );
}
