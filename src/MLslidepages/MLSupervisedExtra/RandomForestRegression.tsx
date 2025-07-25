//import React from "react";

export default function RandomForestRegression() {
  return (
    <div className="max-w-5xl mx-auto p-6 pt-16 space-y-6">
      <h1 className="text-4xl font-bold text-green-700">Random Forest Regression </h1>

      {/* 1. Introduction */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> What is Random Forest Regression?</h2>
        <p>
          Random Forest Regression is an <strong>ensemble learning algorithm</strong> that builds multiple decision trees during training. Each tree predicts a value, and the final output is the <strong>average of all predictions</strong>. It combines:
        </p>
        <ul className="list-disc list-inside ml-6">
          <li><strong>Bagging:</strong> Random bootstrapped datasets</li>
          <li><strong>Random feature selection:</strong> Reduces correlation between trees</li>
        </ul>
      </section>

      {/* 2. Why Use It */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Why Use Random Forest Regression?</h2>
        <table className="table-auto border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border px-3 py-1">Advantage</th>
              <th className="border px-3 py-1">Explanation</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="border px-3 py-1">Reduces overfitting</td><td className="border px-3 py-1">Averaging reduces variance</td></tr>
            <tr><td className="border px-3 py-1">Handles non-linear data</td><td className="border px-3 py-1">Captures complex patterns</td></tr>
            <tr><td className="border px-3 py-1">Robust to outliers</td><td className="border px-3 py-1">Errors cancel out</td></tr>
            <tr><td className="border px-3 py-1">High-dimensional support</td><td className="border px-3 py-1">Selects random features</td></tr>
            <tr><td className="border px-3 py-1">Minimal preprocessing</td><td className="border px-3 py-1">No scaling needed</td></tr>
          </tbody>
        </table>
      </section>

      {/* 3. How It Works */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> How Does It Work?</h2>
        <ul className="list-decimal list-inside ml-6">
          <li><strong>Bootstrapping:</strong> Random samples from original data</li>
          <li><strong>Build trees:</strong> Train each tree with random feature selection and MSE split</li>
          <li><strong>Aggregate:</strong> Average predictions from all trees</li>
        </ul>
      </section>

      {/* 4. Hyperparameters */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Key Hyperparameters</h2>
        <table className="table-auto border border-gray-300 w-full text-left text-sm">
          <thead>
            <tr>
              <th className="border px-3 py-1">Hyperparameter</th>
              <th className="border px-3 py-1">Description</th>
              <th className="border px-3 py-1">Effect</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="border px-3 py-1">n_estimators</td><td className="border px-3 py-1"># of trees</td><td className="border px-3 py-1">More trees → better performance, slower</td></tr>
            <tr><td className="border px-3 py-1">max_depth</td><td className="border px-3 py-1">Tree depth</td><td className="border px-3 py-1">Deeper trees → overfit risk</td></tr>
            <tr><td className="border px-3 py-1">min_samples_split</td><td className="border px-3 py-1">Min samples to split</td><td className="border px-3 py-1">Higher = simpler tree</td></tr>
            <tr><td className="border px-3 py-1">min_samples_leaf</td><td className="border px-3 py-1">Min samples per leaf</td><td className="border px-3 py-1">Helps smoothing</td></tr>
            <tr><td className="border px-3 py-1">max_features</td><td className="border px-3 py-1">Features at split</td><td className="border px-3 py-1">Lower = more randomness</td></tr>
            <tr><td className="border px-3 py-1">bootstrap</td><td className="border px-3 py-1">Use bootstrapping?</td><td className="border px-3 py-1">Usually True</td></tr>
          </tbody>
        </table>
      </section>

      {/* 5. Evaluation */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Model Evaluation Metrics</h2>
        <ul className="list-disc list-inside ml-6 mb-3">
          <li>Mean Squared Error (MSE)</li>
          <li>Root Mean Squared Error (RMSE)</li>
          <li>Mean Absolute Error (MAE)</li>
          <li>R² Score (Coefficient of Determination)</li>
        </ul>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}, R²: {r2:.4f}")`}
        </pre>
      </section>

      {/* 6. Feature Importance */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Feature Importance</h2>
        <p>Random Forest estimates feature importance based on impurity reduction:</p>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`import matplotlib.pyplot as plt

importances = model.feature_importances_
plt.barh(range(len(importances)), importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Index")
plt.show()`}
        </pre>
      </section>

      {/* 7. Pros vs Cons */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Advantages & Disadvantages</h2>
        <table className="table-auto border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border px-3 py-1">Pros</th>
              <th className="border px-3 py-1">Cons</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="border px-3 py-1">Works well on complex data</td><td className="border px-3 py-1">Slower than single trees</td></tr>
            <tr><td className="border px-3 py-1">Handles noise & outliers</td><td className="border px-3 py-1">Harder to interpret</td></tr>
            <tr><td className="border px-3 py-1">Less preprocessing needed</td><td className="border px-3 py-1">Larger memory footprint</td></tr>
          </tbody>
        </table>
      </section>

      {/* 8. Comparison */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Random Forest vs Other Models</h2>
        <table className="table-auto border border-gray-300 w-full text-left text-sm">
          <thead>
            <tr>
              <th className="border px-3 py-1">Model</th>
              <th className="border px-3 py-1">When to Use</th>
              <th className="border px-3 py-1">Strength</th>
              <th className="border px-3 py-1">Weakness</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-3 py-1">Decision Tree</td>
              <td className="border px-3 py-1">Quick, interpretable model</td>
              <td className="border px-3 py-1">Easy to visualize</td>
              <td className="border px-3 py-1">Overfits easily</td>
            </tr>
            <tr>
              <td className="border px-3 py-1">Random Forest</td>
              <td className="border px-3 py-1">Need better accuracy</td>
              <td className="border px-3 py-1">Reduces variance</td>
              <td className="border px-3 py-1">Slower & less interpretable</td>
            </tr>
            <tr>
              <td className="border px-3 py-1">Gradient Boosting</td>
              <td className="border px-3 py-1">Maximize accuracy</td>
              <td className="border px-3 py-1">High performance</td>
              <td className="border px-3 py-1">Needs tuning</td>
            </tr>
            <tr>
              <td className="border px-3 py-1">Linear Regression</td>
              <td className="border px-3 py-1">Linear patterns</td>
              <td className="border px-3 py-1">Simple, fast</td>
              <td className="border px-3 py-1">Can't handle non-linearity</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* 9. Interview Q&A */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Common Interview Questions</h2>
        <ul className="list-disc list-inside ml-6">
          <li><strong>Q:</strong> What is Random Forest Regression?<br /><strong>A:</strong> Ensemble of decision trees using bagging and random features. Predicts by averaging tree outputs.</li>
          <li><strong>Q:</strong> How does it reduce overfitting?<br /><strong>A:</strong> Uses different subsets of data & features → reduces variance.</li>
          <li><strong>Q:</strong> Important hyperparameters?<br /><strong>A:</strong> n_estimators, max_depth, min_samples_split, max_features, etc.</li>
          <li><strong>Q:</strong> What is feature importance?<br /><strong>A:</strong> Shows how much each feature helps reduce prediction error.</li>
          <li><strong>Q:</strong> Limitations?<br /><strong>A:</strong> Slower, large memory usage, harder to interpret.</li>
        </ul>
      </section>

      {/* 10. Tips */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Tips for Effective Usage</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Tune hyperparameters via GridSearchCV or RandomizedSearchCV</li>
          <li>Use <code>oob_score=True</code> for built-in validation</li>
          <li>No need to scale features</li>
          <li>Limit tree depth to reduce overfitting</li>
          <li>Use <code>n_jobs=-1</code> to train in parallel</li>
        </ul>
      </section>

      {/* 11. Sample Code */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Sample Code with Grid Search</h2>
        <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor(oob_score=True, random_state=42)

param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [None, 10, 20],
  'min_samples_split': [2, 5],
  'max_features': ['auto', 'sqrt']
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("OOB score:", grid.best_estimator_.oob_score_)`}
        </pre>
      </section>

      {/* 12. Summary Table */}
      <section>
        <h2 className="text-2xl font-semibold mb-2"> Summary</h2>
        <table className="table-auto border border-gray-300 w-full text-left">
          <thead>
            <tr>
              <th className="border px-3 py-1">Aspect</th>
              <th className="border px-3 py-1">Details</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="border px-3 py-1">Task</td><td className="border px-3 py-1">Regression (predict continuous)</td></tr>
            <tr><td className="border px-3 py-1">Algorithm</td><td className="border px-3 py-1">Ensemble of Decision Trees</td></tr>
            <tr><td className="border px-3 py-1">Strengths</td><td className="border px-3 py-1">Robust, reduces overfitting</td></tr>
            <tr><td className="border px-3 py-1">Weaknesses</td><td className="border px-3 py-1">Slower, less interpretable</td></tr>
            <tr><td className="border px-3 py-1">Library</td><td className="border px-3 py-1">scikit-learn</td></tr>
            <tr><td className="border px-3 py-1">Use Cases</td><td className="border px-3 py-1">Prices, forecasts, etc.</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
