//import React from "react";

export default function DecisionTreeRegression() {
  return (
    <div className="max-w-4xl mx-auto p-6 pt-16 space-y-6">
      <h1 className="text-4xl font-bold text-green-600"> Decision Tree Regression — From Beginner to Pro</h1>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> What is Decision Tree Regression?</h2>
        <p>
          Decision Tree Regression is a <strong>supervised learning</strong> algorithm used for predicting continuous numerical values. It models data using a tree structure, where:
        </p>
        <ul className="list-disc list-inside ml-6">
          <li>Each internal node represents a decision rule on a feature</li>
          <li>Each leaf node holds a prediction</li>
        </ul>
        <p>
          <strong>Objective:</strong> Learn a mapping from input features to target values by recursively partitioning the feature space and minimizing error.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> How It Works</h2>
        <ul className="list-disc list-inside ml-6">
          <li>The tree splits the dataset based on feature values that minimize a cost (like MSE).</li>
          <li>Each internal node uses a decision rule (e.g., <code>X[i] ≤ threshold</code>).</li>
          <li>Each leaf node contains the average of target values in that region.</li>
          <li>Splits continue until a stopping criterion is met (e.g., <code>max_depth</code>).</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> When to Use</h2>
        <p><strong> Suitable for:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Non-linear relationships</li>
          <li>Numerical and categorical features</li>
          <li>High interpretability (e.g., business decisions)</li>
        </ul>
        <p><strong> Not Ideal When:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Dataset is small and noisy</li>
          <li>Need strong generalization and smooth predictions</li>
          <li>Extrapolating beyond training data</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Key Concepts</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border border-gray-300 px-3 py-1">Term</th>
              <th className="border border-gray-300 px-3 py-1">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Root Node</td>
              <td className="border border-gray-300 px-3 py-1">Initial split of the dataset</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Leaf Node</td>
              <td className="border border-gray-300 px-3 py-1">Final node with predicted value</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Split</td>
              <td className="border border-gray-300 px-3 py-1">Decision rule (e.g., X ≤ 3.5)</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">MSE</td>
              <td className="border border-gray-300 px-3 py-1">Mean Squared Error (loss function)</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Overfitting</td>
              <td className="border border-gray-300 px-3 py-1">Tree becomes too complex and fits noise</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Pruning</td>
              <td className="border border-gray-300 px-3 py-1">Reducing tree size to improve generalization</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Python Implementation Steps</h2>

        <h3 className="font-semibold">Step 1: Import Libraries</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt`}
        </pre>

        <h3 className="font-semibold">Step 2: Load/Create Data</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.9, 3.2, 4.0, 5.1])`}
        </pre>

        <h3 className="font-semibold">Step 3: Train the Model</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`model = DecisionTreeRegressor()
model.fit(X, y)`}
        </pre>

        <h3 className="font-semibold">Step 4: Predict and Visualize</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`X_test = np.arange(0, 6, 0.1).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Decision Tree Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.show()`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Important Hyperparameters</h2>
        <ul className="list-disc list-inside ml-6">
          <li><code>max_depth</code>: Maximum depth of the tree</li>
          <li><code>min_samples_split</code>: Minimum samples to split an internal node</li>
          <li><code>min_samples_leaf</code>: Minimum samples required at a leaf</li>
          <li><code>criterion</code>: Loss function ("squared_error", "friedman_mse")</li>
          <li><code>max_features</code>: Max features to consider per split</li>
        </ul>

        <h3 className="font-semibold mt-3">Example:</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`model = DecisionTreeRegressor(max_depth=3, min_samples_split=4)`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Model Evaluation</h2>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`from sklearn.metrics import mean_squared_error, r2_score

y_train_pred = model.predict(X)
mse = mean_squared_error(y, y_train_pred)
r2 = r2_score(y, y_train_pred)

print(f"MSE: {mse:.3f}")
print(f"R² Score: {r2:.3f}")`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Pros and  Cons</h2>
        <p><strong>Advantages:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Simple and interpretable</li>
          <li>Captures non-linear relationships</li>
          <li>No need for feature scaling or normalization</li>
        </ul>

        <p className="mt-2"><strong>Disadvantages:</strong></p>
        <ul className="list-disc list-inside ml-6">
          <li>Overfits easily</li>
          <li>Unstable with small data changes</li>
          <li>Poor extrapolation outside training range</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Pro-Level Techniques</h2>

        <h3 className="font-semibold"> GridSearchCV for Tuning</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`from sklearn.model_selection import GridSearchCV

param_grid = {
  'max_depth': [3, 5, 10, None],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
grid.fit(X, y)
print("Best Parameters:", grid.best_params_)`}
        </pre>

        <h3 className="font-semibold mt-3">Ensemble Techniques</h3>

        <p><strong>1. Random Forest:</strong></p>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)`}
        </pre>

        <p><strong>2. Gradient Boosting:</strong></p>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gb.fit(X, y)`}
        </pre>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Real-World Use Cases</h2>
        <ul className="list-disc list-inside ml-6">
          <li>House Price Prediction</li>
          <li>Energy Usage Forecasting</li>
          <li>Short-term Stock or Time-Series Prediction</li>
          <li>Medical Cost Estimation</li>
          <li>Product Sales Forecasting</li>
        </ul>
      </section>


    <div className="max-w-4xl mx-auto p-6 pt-16 space-y-6">
      <h1 className="text-4xl font-bold text-green-700">How Does a Decision Tree Work?</h1>

      <section>
        <p>
          A Decision Tree is a flowchart-like structure used for decision-making and predictive modeling. It works by recursively splitting the dataset based on feature values, aiming to create pure subsets where outcomes are as homogeneous as possible.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> General Structure</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border border-gray-300 px-3 py-1">Component</th>
              <th className="border border-gray-300 px-3 py-1">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Root Node</td>
              <td className="border border-gray-300 px-3 py-1">Initial split using most informative feature</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Internal Nodes</td>
              <td className="border border-gray-300 px-3 py-1">Intermediate decision points (e.g., feature ≤ threshold)</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-3 py-1">Leaf Nodes</td>
              <td className="border border-gray-300 px-3 py-1">Terminal nodes giving final prediction</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Step-by-Step Working</h2>
        <h3 className="font-semibold mt-2">➤ Step 1: Select the Best Feature to Split</h3>
        <p>
          For each feature and threshold, the algorithm evaluates how well the split separates the target:
        </p>
        <ul className="list-disc list-inside ml-6">
          <li><strong>Regression:</strong> Minimize Mean Squared Error (MSE)</li>
          <li><strong>Classification:</strong> Maximize Information Gain (e.g., Gini, Entropy)</li>
        </ul>

        <h3 className="font-semibold mt-4">➤ Step 2: Split the Data</h3>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`if X[feature] <= threshold:
    go to left branch
else:
    go to right branch`}
        </pre>

        <h3 className="font-semibold mt-4">➤ Step 3: Repeat Recursively</h3>
        <p>Repeat on child nodes until:</p>
        <ul className="list-disc list-inside ml-6">
          <li>Max depth is reached</li>
          <li>Minimum samples per node</li>
          <li>No reduction in error</li>
        </ul>

        <h3 className="font-semibold mt-4">➤ Step 4: Predict at Leaf Nodes</h3>
        <ul className="list-disc list-inside ml-6">
          <li><strong>Regression:</strong> Predict mean value</li>
          <li><strong>Classification:</strong> Predict majority class</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Splitting Criterion for Regression</h2>
        <p><strong>Mean Squared Error (MSE):</strong></p>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`MSE = (1/n) ∑(yᵢ − ŷᵢ)²`}
        </pre>
        <p>The algorithm selects splits that minimize total MSE across resulting branches.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Simple Example: Predicting Test Scores</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left mb-3">
          <thead>
            <tr>
              <th className="border border-gray-300 px-3 py-1">Hours Studied</th>
              <th className="border border-gray-300 px-3 py-1">Score</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="border px-3 py-1">1</td><td className="border px-3 py-1">50</td></tr>
            <tr><td className="border px-3 py-1">2</td><td className="border px-3 py-1">55</td></tr>
            <tr><td className="border px-3 py-1">3</td><td className="border px-3 py-1">65</td></tr>
            <tr><td className="border px-3 py-1">5</td><td className="border px-3 py-1">85</td></tr>
            <tr><td className="border px-3 py-1">6</td><td className="border px-3 py-1">95</td></tr>
          </tbody>
        </table>

        <p className="italic">
          Try all splits (e.g., Hours ≤ 1.5, 2.5, ...) and pick one with lowest MSE. Then repeat for child nodes.
        </p>

        <p><strong>Example:</strong> Student studied 4 hours → Model follows tree conditions → Predicts average of matching leaf group.</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Visual Analogy</h2>
        <p>
          Think of it like a game of <strong>"20 Questions"</strong>, but with numeric comparisons. Each internal node asks a question. You go left/right based on the answer until reaching the leaf (final prediction).
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Classification Example: Will a Customer Buy?</h2>

        <h3 className="font-semibold"> Features:</h3>
        <ul className="list-disc list-inside ml-6 mb-2">
          <li>Income</li>
          <li>Age</li>
          <li>Previous Purchases</li>
        </ul>

        <h3 className="font-semibold"> Tree Logic:</h3>
        <ul className="list-disc list-inside ml-6">
          <li>Income &gt; $50k? → No → ❌ No Purchase</li>
<li>Yes → Age &gt; 30? → No → ❌ No Purchase</li>

          <li>Yes → Has purchased before? → Yes →  Purchase</li>
        </ul>

        <h3 className="font-semibold mt-2"> Case:</h3>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border px-3 py-1">Feature</th>
              <th className="border px-3 py-1">Value</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="border px-3 py-1">Income</td><td className="border px-3 py-1">$65,000</td></tr>
            <tr><td className="border px-3 py-1">Age</td><td className="border px-3 py-1">35</td></tr>
            <tr><td className="border px-3 py-1">Previous Purchases</td><td className="border px-3 py-1">Yes</td></tr>
          </tbody>
        </table>

        <p className="mt-2 font-semibold"> Final Prediction: Purchase</p>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Ensemble Example: Two Trees</h2>

        <h3 className="font-semibold">Tree 1: Demographics</h3>
        <ul className="list-disc list-inside ml-6">
           <li>Income &gt; $50k? → No → ❌</li>
          <li>Yes → Age &gt; 30? → No → ❌</li>

          <li>Yes →  Purchase</li>
        </ul>

        <h3 className="font-semibold mt-2">Tree 2: Purchase History</h3>
        <ul className="list-disc list-inside ml-6">
          <li>Previous Purchases &gt; 0? → Yes → ✅</li>

          <li>No → ❌</li>
        </ul>

        <h3 className="font-semibold mt-4"> Case 1:</h3>
        <ul className="list-disc list-inside ml-6">
          <li>Income: $60k, Age: 28, Previous: No</li>
          <li>Tree 1: ❌, Tree 2: ❌ → <strong>Final: No Purchase</strong></li>
        </ul>

        <h3 className="font-semibold mt-2"> Case 2:</h3>
        <ul className="list-disc list-inside ml-6">
          <li>Income: $80k, Age: 32, Previous: No</li>
          <li>Tree 1: ✅, Tree 2: ❌ → <strong>Final: Depends on ensemble weighting</strong></li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2"> Summary</h2>
        <table className="table-auto border-collapse border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border px-3 py-1">Step</th>
              <th className="border px-3 py-1">Purpose</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-3 py-1">Select best feature</td>
              <td className="border px-3 py-1">Reduce impurity or error</td>
            </tr>
            <tr>
              <td className="border px-3 py-1">Split data recursively</td>
              <td className="border px-3 py-1">Create informative branches</td>
            </tr>
            <tr>
              <td className="border px-3 py-1">Stop splitting (prune)</td>
              <td className="border px-3 py-1">Avoid overfitting</td>
            </tr>
            <tr>
              <td className="border px-3 py-1">Predict at leaf</td>
              <td className="border px-3 py-1">Use average (regression) or majority (classification)</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
       




    </div>
  );
}
