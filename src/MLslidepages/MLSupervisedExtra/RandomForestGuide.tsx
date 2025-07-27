// RandomForestGuide.tsx

export default function RandomForestGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-green-700">
        Random Forest Classifier 
      </h1>

      {/* Section 1 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is Random Forest?</h2>
        <p>
          Random Forest is an ensemble learning algorithm that builds multiple decision trees
          and combines their outputs to improve accuracy and avoid overfitting.
        </p>
        <p> It is used for classification and regression, but here we focus on classification.</p>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Intuition</h2>
        <p>
          Imagine asking multiple experts (trees) instead of relying on one.
          Each tree gives a classification, and the forest picks the majority vote.
          This reduces errors caused by overfitting or noisy data.
        </p>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> How it Works (Step-by-Step)</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>
            <strong>Bootstrapping:</strong><br />
            Random samples (with replacement) are taken from the training dataset.<br />
            Each sample trains a separate decision tree (called a base learner).
          </li>
          <li>
            <strong>Random Feature Selection:</strong><br />
            At each node split, only a random subset of features is considered.
          </li>
          <li>
            <strong>Tree Building:</strong><br />
            Each tree is grown deep (often to full depth without pruning).
          </li>
          <li>
            <strong>Voting:</strong><br />
            Classification: Majority class among all trees.<br />
            Regression: Average of outputs.
          </li>
        </ul>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Example in scikit-learn</h2>
        <div className="bg-gray-100 p-2 rounded text-xs overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train model
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)`}
          </pre>
        </div>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Key Parameters</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Parameter</th>
              <th className="p-2 border">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">n_estimators</td>
              <td className="p-2 border">Number of trees in the forest</td>
            </tr>
            <tr>
              <td className="p-2 border">max_depth</td>
              <td className="p-2 border">Maximum depth of each tree</td>
            </tr>
            <tr>
              <td className="p-2 border">max_features</td>
              <td className="p-2 border">Subset of features to consider for splits</td>
            </tr>
            <tr>
              <td className="p-2 border">bootstrap</td>
              <td className="p-2 border">Whether to use bootstrapping (sampling with replacement)</td>
            </tr>
            <tr>
              <td className="p-2 border">criterion</td>
              <td className="p-2 border">Split quality function (e.g., gini, entropy)</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Advantages</h2>
        <ul className="list-disc ml-6">
          <li> Reduces overfitting (better generalization than single trees)</li>
          <li> Handles high-dimensional data well</li>
          <li> Robust to noise and outliers</li>
          <li> Works well even with missing data</li>
        </ul>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Disadvantages</h2>
        <ul className="list-disc ml-6">
          <li> Less interpretable than a single decision tree</li>
          <li> More computationally intensive</li>
          <li> Larger memory usage</li>
        </ul>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Use Cases</h2>
        <ul className="list-disc ml-6">
          <li>Email spam detection</li>
          <li>Medical diagnosis</li>
          <li>Credit card fraud detection</li>
          <li>Sentiment analysis</li>
        </ul>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Comparison: Decision Tree vs Random Forest</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Feature</th>
              <th className="p-2 border">Decision Tree</th>
              <th className="p-2 border">Random Forest</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Overfitting</td>
              <td className="p-2 border">High</td>
              <td className="p-2 border">Reduced</td>
            </tr>
            <tr>
              <td className="p-2 border">Accuracy</td>
              <td className="p-2 border">Lower</td>
              <td className="p-2 border">Higher</td>
            </tr>
            <tr>
              <td className="p-2 border">Interpretability</td>
              <td className="p-2 border">Easy to understand</td>
              <td className="p-2 border">Harder to interpret</td>
            </tr>
            <tr>
              <td className="p-2 border">Training Time</td>
              <td className="p-2 border">Faster</td>
              <td className="p-2 border">Slower</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
