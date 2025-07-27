export default function BaggingGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">Bagging </h1>

      {/* 1. What is Bagging? */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> What is Bagging?</h2>
        <p>
          Bagging (Bootstrap Aggregating) is an ensemble learning method that trains multiple models on different subsets of the training data and combines their predictions.
        </p>
      </section>

      {/* 2. Why Use Bagging */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Why Use Bagging?</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> Reduces overfitting</li>
          <li> Improves accuracy</li>
          <li> Works well with high-variance models (e.g., decision trees)</li>
          <li> Stabilizes predictions</li>
        </ul>
      </section>

      {/* 3. How Bagging Works */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> How Bagging Works (Step-by-Step)</h2>
        <ul className="list-decimal ml-6 space-y-1">
          <li>Generate multiple bootstrap samples (random sampling with replacement).</li>
          <li>Train a separate model (usually a decision tree) on each sample.</li>
          <li>
            Combine predictions:
            <ul className="list-disc ml-6">
              <li>Classification: majority vote</li>
              <li>Regression: average</li>
            </ul>
          </li>
        </ul>
      </section>

      {/* 4. Common Bagging Algorithms */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Common Bagging Algorithms</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Algorithm</th>
                <th className="border px-2 py-1">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Bagged Trees</td>
                <td className="border px-2 py-1">Multiple trees trained on bootstrap samples</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Random Forest</td>
                <td className="border px-2 py-1">Bagging + feature randomness</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 5. Python Code Example */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Python Code Example</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

base_clf = DecisionTreeClassifier()
bag_clf = BaggingClassifier(base_estimator=base_clf, n_estimators=10, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))`}
        </pre>
      </section>

      {/* 6. Parameters */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Key Parameters (Pro Level)</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Parameter</th>
                <th className="border px-2 py-1">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="border px-2 py-1">n_estimators</td><td className="border px-2 py-1">Number of base models</td></tr>
              <tr><td className="border px-2 py-1">max_samples</td><td className="border px-2 py-1">Samples per base model</td></tr>
              <tr><td className="border px-2 py-1">bootstrap</td><td className="border px-2 py-1">Use sampling with replacement?</td></tr>
              <tr><td className="border px-2 py-1">bootstrap_features</td><td className="border px-2 py-1">Sample features too?</td></tr>
              <tr><td className="border px-2 py-1">oob_score</td><td className="border px-2 py-1">Evaluate with OOB samples</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 7. Out-of-Bag */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Out-of-Bag (OOB) Evaluation</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>OOB samples = data not in the bootstrap sample.</li>
          <li>Acts as a built-in validation set.</li>
          <li>Set <code>oob_score=True</code> in the classifier.</li>
        </ul>
      </section>

      {/* 8. When to Use */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> When to Use Bagging?</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>When using high-variance, low-bias models like trees</li>
          <li>When data size is small to medium</li>
          <li>When you want to improve accuracy without increasing bias</li>
        </ul>
      </section>

      {/* 9. Bagging vs Boosting */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Bagging vs Boosting</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">Bagging</th>
                <th className="border px-2 py-1">Boosting</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="border px-2 py-1">Focus</td><td className="border px-2 py-1">Reduce variance</td><td className="border px-2 py-1">Reduce bias</td></tr>
              <tr><td className="border px-2 py-1">Learners</td><td className="border px-2 py-1">Parallel</td><td className="border px-2 py-1">Sequential</td></tr>
              <tr><td className="border px-2 py-1">Example</td><td className="border px-2 py-1">Random Forest</td><td className="border px-2 py-1">AdaBoost, XGBoost</td></tr>
              <tr><td className="border px-2 py-1">Overfitting</td><td className="border px-2 py-1">Less prone</td><td className="border px-2 py-1">More prone if unregularized</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 10. Advanced Tips */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Advanced Tips</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Random Forest = Bagging + Random Feature Selection</li>
          <li>Try other base models (SVM, KNN) in BaggingClassifier</li>
          <li>Use GridSearchCV to tune <code>max_samples</code>, <code>n_estimators</code>, etc.</li>
        </ul>
      </section>
    </div>
  );
}
