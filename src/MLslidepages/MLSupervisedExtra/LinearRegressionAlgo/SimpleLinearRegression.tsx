//import React from 'react';

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


<section className="px-4 sm:px-6 py-6 max-w-4xl mx-auto text-gray-800">
  <section className="mb-6">
    <h2 className="text-xl sm:text-2xl font-semibold mb-3">Interview Questions</h2>

    <h3 className="font-semibold mb-1">Basic</h3>
    <ul className="list-disc list-inside mb-4 space-y-1 text-sm sm:text-base">
      <li>What is Simple Linear Regression?</li>
      <li>What does the slope (m) represent?</li>
      <li>What is the difference between actual and predicted values?</li>
      <li>How is the regression line fitted?</li>
    </ul>

    <h3 className="font-semibold mb-1">Technical</h3>
    <ul className="list-disc list-inside space-y-1 text-sm sm:text-base">
      <li>What is the cost function in Simple Linear Regression?</li>
      <li>What are residuals?</li>
      <li>How do you evaluate model performance?</li>
    </ul>
  </section>

  <section className="mb-6">
    <h2 className="text-xl sm:text-2xl font-semibold mb-3">Python Example (Using scikit-learn)</h2>
    <pre className="bg-gray-100 p-3 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-x-auto">
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

  <section className="mb-6">
    <h2 className="text-xl sm:text-2xl font-semibold mb-3">Evaluation Metrics</h2>
    <ul className="list-disc list-inside space-y-1 text-sm sm:text-base">
      <li><strong>MSE:</strong> Mean Squared Error</li>
      <li><strong>RMSE:</strong> Root Mean Squared Error</li>
      <li><strong>MAE:</strong> Mean Absolute Error</li>
      <li><strong>R² Score:</strong> Coefficient of determination</li>
    </ul>
  </section>

  <section>
    <h2 className="text-xl sm:text-2xl font-semibold mb-3">Summary</h2>
    <div className="overflow-x-auto rounded border border-gray-300">
      <table className="min-w-full border-collapse border border-gray-300 text-xs sm:text-sm">
        <tbody>
          {[
            ["Type", "Regression"],
            ["Input", "One feature (X)"],
            ["Output", "One continuous variable (Y)"],
            ["Model Equation", <code key="eq">Y = mX + c</code>],
            ["Goal", "Minimize prediction error (MSE)"],
            ["Limitation", "Can't handle multiple inputs or non-linear patterns"],
          ].map(([key, value], i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
              <td className="border border-gray-300 px-2 py-1 font-semibold break-words">{key}</td>
              <td className="border border-gray-300 px-2 py-1 break-words">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </section>

  <section className="mt-8">
    <h1 className="text-2xl sm:text-3xl font-bold mb-4">
      Step-by-Step Guide to Write Simple Linear Regression Code
    </h1>
  </section>
</section>

<section className="px-4 sm:px-6 py-6 max-w-4xl mx-auto text-gray-800">
  {[
    {
      title: "Import Required Libraries",
      code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score`,
      reason: "You need libraries for data manipulation, visualization, modeling, and evaluation.",
    },
    {
      title: "Load the Dataset",
      code: `data = pd.read_csv('your_dataset.csv')  # Or create manually`,
      reason: "You need data with one independent variable (X) and one dependent variable (Y).",
    },
    {
      title: "Explore the Data",
      code: `print(data.head())
print(data.describe())
print(data.info())`,
      reason: "To understand data structure, check for missing values or incorrect types.",
    },
    {
      title: "Visualize the Data (Optional but Recommended)",
      code: `plt.scatter(data['X'], data['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs Y')
plt.show()`,
      reason: "Helps you visually confirm whether a linear relationship exists.",
    },
    {
      title: "Prepare the Data",
      code: `X = data[['X']]   # Feature (must be 2D)
y = data['Y']     # Target (1D)`,
      reason: "Separate features and labels before training.",
    },
    {
      title: "Split the Data (Training & Testing)",
      code: `X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)`,
      reason: "To evaluate the model on unseen data.",
    },
    {
      title: "Train the Model",
      code: `model = LinearRegression()
model.fit(X_train, y_train)`,
      reason: "Fit the model to learn slope and intercept (line of best fit).",
    },
    {
      title: "Get Predictions",
      code: `y_pred = model.predict(X_test)`,
      reason: "Predict Y values for the test set.",
    },
    {
      title: "Evaluate the Model",
      code: `print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))`,
      reason: "Check how well the model performs using metrics.",
    },
    {
      title: "Visualize the Regression Line",
      code: `plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()`,
      reason: "To confirm visually how well the regression line fits the data.",
    },
    {
      title: "Inspect Model Parameters",
      code: `print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)`,
      reason: "Useful for explaining the model and understanding the equation.",
    },
  ].map(({ title, code, reason }, index) => (
    <section key={index} className="mb-6">
      <h2 className="text-lg sm:text-xl font-semibold mb-2">{title}</h2>
      <pre className="bg-gray-100 p-3 rounded text-xs sm:text-sm whitespace-pre-wrap overflow-x-auto">
        {code}
      </pre>
      <p className="italic text-sm sm:text-base mt-1">{`Why: ${reason}`}</p>
    </section>
  ))}

  {/* Summary Table */}
  <section className="mt-8">
    <h2 className="text-lg sm:text-xl font-semibold mb-2"> Summary: What to Look For at Each Step</h2>
    <div className="overflow-x-auto rounded border border-gray-300">
      <table className="min-w-full text-left text-xs sm:text-sm table-auto border-collapse">
        <thead className="bg-gray-100">
          <tr>
            <th className="border border-gray-300 px-2 py-1 font-semibold">Step</th>
            <th className="border border-gray-300 px-2 py-1 font-semibold">What to Check / Look For</th>
          </tr>
        </thead>
        <tbody>
          {[
            ["Load Data", "One independent and one dependent variable"],
            ["Explore Data", "Missing values, data types, linear trends"],
            ["Visualize", "Confirm linear pattern (scatter plot)"],
            ["Prepare", "Ensure correct input shape for X and y"],
            ["Split", "Avoid overfitting by separating test data"],
            ["Train", "Use LinearRegression().fit()"],
            ["Predict", "Use model.predict()"],
            ["Evaluate", "Check MSE, R² score"],
            ["Visualize", "Draw regression line on actual data"],
            ["Inspect", "Print slope and intercept for insights"],
          ].map(([step, desc], idx) => (
            <tr key={idx}>
              <td className="border border-gray-300 px-2 py-1 font-medium">{step}</td>
              <td className="border border-gray-300 px-2 py-1">{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </section>
</section>


    </div>
  );
}
