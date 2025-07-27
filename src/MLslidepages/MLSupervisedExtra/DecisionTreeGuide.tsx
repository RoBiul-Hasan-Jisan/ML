// DecisionTreeGuide.tsx

export default function DecisionTreeGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">Decision Tree Classification </h1>

      {/* Section 1 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is a Decision Tree?</h2>
        <p>
          A Decision Tree is a supervised machine learning algorithm used for classification and regression.
          It mimics human decision-making using tree-like structures, where each internal node is a feature test,
          branches are outcomes, and leaf nodes represent predicted classes.
        </p>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> How It Works</h2>
        <p>
          Core idea: Recursively split the dataset into smaller subsets based on the feature that gives the best separation between classes.
          Each split aims to create purer subsets, i.e., groups with mostly the same class label.
        </p>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Key Concepts</h2>
        <p><strong> Impurity Measures:</strong></p>
        <p>Used to decide how to split:</p>
        <ul className="list-disc ml-6">
          <li>
            <strong>Gini Impurity</strong> (default in sklearn):
            <br />
            <code>Gini = 1 − ∑(pᵢ²)</code>
          </li>
          <li>
            <strong>Entropy / Information Gain:</strong>
            <br />
            <code>Entropy = − ∑ pᵢ log₂(pᵢ)</code>
            <br />
            Information Gain = Entropy(parent) − Weighted Avg. Entropy(children)
          </li>
        </ul>
        <p>Lower Gini or Entropy → better purity.</p>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Decision Tree Building Steps</h2>
        <ol className="list-decimal ml-6 space-y-1">
          <li>Start with the full dataset.</li>
          <li>Choose the best feature to split on (based on impurity).</li>
          <li>Split the dataset into subsets.</li>
          <li>Repeat recursively for each subset.</li>
          <li>Stop when:
            <ul className="list-disc ml-6">
              <li>All samples are pure</li>
              <li>Maximum depth is reached</li>
              <li>Minimum samples per node is reached</li>
            </ul>
          </li>
        </ol>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Python Implementation (Using sklearn)</h2>
        <div className="bg-gray-100 p-2 rounded text-xs overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Decision Tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(X_train, y_train)

# Predict & accuracy
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Plot Tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()`}
          </pre>
        </div>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Parameters in DecisionTreeClassifier</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Parameter</th>
              <th className="p-2 border">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">criterion</td>
              <td className="p-2 border">"gini" or "entropy"</td>
            </tr>
            <tr>
              <td className="p-2 border">max_depth</td>
              <td className="p-2 border">Max tree depth</td>
            </tr>
            <tr>
              <td className="p-2 border">min_samples_split</td>
              <td className="p-2 border">Minimum samples to split an internal node</td>
            </tr>
            <tr>
              <td className="p-2 border">min_samples_leaf</td>
              <td className="p-2 border">Minimum samples at a leaf node</td>
            </tr>
            <tr>
              <td className="p-2 border">max_features</td>
              <td className="p-2 border">Number of features to consider for best split</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Advantages</h2>
        <ul className="list-disc ml-6">
          <li>Easy to understand and visualize</li>
          <li>Requires little data preparation</li>
          <li>Works for both numerical and categorical data</li>
          <li>Nonlinear relationships handled well</li>
        </ul>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">. Disadvantages</h2>
        <ul className="list-disc ml-6">
          <li>High risk of overfitting</li>
          <li>Can be unstable (small changes in data can lead to different trees)</li>
          <li>Biased towards features with more levels</li>
          <li>Not great at generalization (fixable with pruning or ensembles like Random Forests)</li>
        </ul>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Pruning the Tree</h2>
        <p>Pruning reduces overfitting:</p>
        <ul className="list-disc ml-6">
          <li>Pre-pruning: limit depth, min samples, etc.</li>
          <li>Post-pruning: grow full tree, then trim branches based on validation score.</li>
        </ul>
      </section>

      {/* Section 10 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Decision Tree Use Cases</h2>
        <ul className="list-disc ml-6">
          <li>Credit scoring</li>
          <li>Medical diagnosis</li>
          <li>Customer churn prediction</li>
          <li>Fraud detection</li>
          <li>Spam detection</li>
        </ul>
      </section>

      {/* Section 11 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Decision Tree vs Other Models</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Model</th>
              <th className="p-2 border">Type</th>
              <th className="p-2 border">Interpretable?</th>
              <th className="p-2 border">Handles Nonlinearity</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Decision Tree</td>
              <td className="p-2 border">White-box</td>
              <td className="p-2 border"> Yes</td>
              <td className="p-2 border">Yes</td>
            </tr>
            <tr>
              <td className="p-2 border">Logistic Regression</td>
              <td className="p-2 border">White-box</td>
              <td className="p-2 border"> Yes</td>
              <td className="p-2 border"> No</td>
            </tr>
            <tr>
              <td className="p-2 border">SVM</td>
              <td className="p-2 border">Black-box</td>
              <td className="p-2 border"> No</td>
              <td className="p-2 border"> Yes</td>
            </tr>
            <tr>
              <td className="p-2 border">Neural Network</td>
              <td className="p-2 border">Black-box</td>
              <td className="p-2 border"> No</td>
              <td className="p-2 border"> Yes</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Section 12 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Common Interview Questions</h2>
        <ul className="list-disc ml-6">
          <li>How does a decision tree decide where to split?</li>
          <li>What’s the difference between Gini impurity and entropy?</li>
          <li>How do you prevent overfitting in a decision tree?</li>
          <li>What is pruning and why is it used?</li>
          <li>Compare decision trees with random forests.</li>
        </ul>
      </section>

      {/* Section 13 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Difference Between Decision Tree Classification vs Regression</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Feature</th>
              <th className="p-2 border">Decision Tree Classification</th>
              <th className="p-2 border">Decision Tree Regression</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border"> Purpose</td>
              <td className="p-2 border">Predict a class label (e.g., "cat", "dog")</td>
              <td className="p-2 border">Predict a continuous value (e.g., 3.75)</td>
            </tr>
            <tr>
              <td className="p-2 border">Target Variable</td>
              <td className="p-2 border">Categorical / discrete</td>
              <td className="p-2 border">Continuous / real-valued</td>
            </tr>
            <tr>
              <td className="p-2 border"> Splitting Criterion</td>
              <td className="p-2 border">Gini impurity or Entropy (Information Gain)</td>
              <td className="p-2 border">Mean Squared Error (MSE) or Mean Absolute Error (MAE)</td>
            </tr>
            <tr>
              <td className="p-2 border"> Leaf Node Output</td>
              <td className="p-2 border">A class label (e.g., 0, 1, "yes", "no")</td>
              <td className="p-2 border">A number (e.g., average of target values)</td>
            </tr>
            <tr>
              <td className="p-2 border">Loss Function</td>
              <td className="p-2 border">Classification Error / Gini / Entropy</td>
              <td className="p-2 border">Mean Squared Error (MSE) / MAE</td>
            </tr>
            <tr>
              <td className="p-2 border"> Evaluation Metrics</td>
              <td className="p-2 border">Accuracy, Precision, Recall, F1-score</td>
              <td className="p-2 border">RMSE, MAE, R² score</td>
            </tr>
            <tr>
              <td className="p-2 border"> Use Cases</td>
              <td className="p-2 border">Spam detection, disease diagnosis, image classification</td>
              <td className="p-2 border">House price prediction, stock forecasting</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Section 14 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Example Comparison</h2>
        <p> Decision Tree Classification:</p>
        <p>Input: Pet features</p>
        <p>Target: "dog" or "cat"</p>
        <p>Leaf Node: class = "cat"</p>
        <br />
        <p> Decision Tree Regression:</p>
        <p>Input: House features</p>
        <p>Target: Price = $350,000</p>
        <p>Leaf Node: value = 350000</p>
      </section>

      {/* Section 15 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Code Snippet (Difference)</h2>
        <p><strong> Classification:</strong></p>
        <div className="bg-gray-100 p-2 rounded text-xs overflow-auto mb-4">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # y_train is categorical`}
          </pre>
        </div>

        <p><strong> Regression:</strong></p>
        <div className="bg-gray-100 p-2 rounded text-xs overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)  # y_train is continuous`}
          </pre>
        </div>
      </section>

      {/* Section 16 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Summary</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Aspect</th>
              <th className="p-2 border">Classification</th>
              <th className="p-2 border">Regression</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Output Type</td>
              <td className="p-2 border">Class</td>
              <td className="p-2 border">Real value</td>
            </tr>
            <tr>
              <td className="p-2 border">Use Case</td>
              <td className="p-2 border">Predict categories</td>
              <td className="p-2 border">Predict quantities</td>
            </tr>
            <tr>
              <td className="p-2 border">Leaf Node</td>
              <td className="p-2 border">Majority class</td>
              <td className="p-2 border">Average value</td>
            </tr>
            <tr>
              <td className="p-2 border">Criterion</td>
              <td className="p-2 border">Gini / Entropy</td>
              <td className="p-2 border">MSE / MAE</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
