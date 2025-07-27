export default function ExtraTreesGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">Extra Trees</h1>

      {/* 1. What is Extra Trees? */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> What is Extra Trees?</h2>
        <p>
          Extra Trees is an ensemble learning method that builds many uncorrelated decision trees and averages their predictions. It adds more randomness than Random Forest to reduce variance.
        </p>
        <ul className="list-disc ml-6 mt-2 space-y-1">
          <li>Type: Ensemble, Bagging-based</li>
          <li>Used For: Classification and Regression</li>
        </ul>
      </section>

      {/* 2. Extra Trees vs Random Forest */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Extra Trees vs Random Forest</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">Random Forest</th>
                <th className="border px-2 py-1">Extra Trees</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Bootstrapping</td>
                <td className="border px-2 py-1">Yes (sampling with replacement)</td>
                <td className="border px-2 py-1">No (uses whole dataset)</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Feature Splits</td>
                <td className="border px-2 py-1">Best split from random subset</td>
                <td className="border px-2 py-1">Random split from random subset</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">More Randomness</td>
                <td className="border px-2 py-1">Less</td>
                <td className="border px-2 py-1">More</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Speed</td>
                <td className="border px-2 py-1">Slower</td>
                <td className="border px-2 py-1">Faster (due to random thresholds)</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Bias-Variance Tradeoff</td>
                <td className="border px-2 py-1">Lower bias, moderate variance</td>
                <td className="border px-2 py-1">Slightly higher bias, lower variance</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 3. How Extra Trees Work */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> How Extra Trees Work (Step-by-Step)</h2>
        <ol className="list-decimal ml-6 space-y-1">
          <li>No Bootstrapping: each tree trains on full dataset.</li>
          <li>Random Feature Selection: select random feature subsets at each split.</li>
          <li>Random Split Value: choose random split thresholds, not best splits.</li>
          <li>Ensemble Prediction:
            <ul className="list-disc ml-6 mt-1">
              <li>Classification: majority vote</li>
              <li>Regression: average prediction</li>
            </ul>
          </li>
        </ol>
      </section>

      {/* 4. Advantages */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Advantages</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> Fast training (random thresholds save computation)</li>
          <li> Less overfitting due to extra randomness</li>
          <li> Works well with high-dimensional data</li>
          <li> Generally requires less tuning</li>
        </ul>
      </section>

      {/* 5. Disadvantages */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Disadvantages</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> Slightly less accurate than Random Forest in some cases</li>
          <li> Lower interpretability</li>
          <li> High randomness can hurt performance on small datasets</li>
        </ul>
      </section>

      {/* 6. Python Example */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2">Python Example (sklearn)</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")`}
        </pre>
      </section>

      {/* 7. Hyperparameters to Tune */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Hyperparameters to Tune</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Parameter</th>
                <th className="border px-2 py-1">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="border px-2 py-1">n_estimators</td><td className="border px-2 py-1">Number of trees</td></tr>
              <tr><td className="border px-2 py-1">max_depth</td><td className="border px-2 py-1">Maximum tree depth</td></tr>
              <tr><td className="border px-2 py-1">max_features</td><td className="border px-2 py-1">Features considered per split</td></tr>
              <tr><td className="border px-2 py-1">min_samples_split</td><td className="border px-2 py-1">Min samples to split node</td></tr>
              <tr><td className="border px-2 py-1">min_samples_leaf</td><td className="border px-2 py-1">Min samples at leaf node</td></tr>
              <tr><td className="border px-2 py-1">bootstrap</td><td className="border px-2 py-1">False (usually for Extra Trees)</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 8. When to Use Extra Trees */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> When to Use Extra Trees?</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> For fast training and robust predictions</li>
          <li> To reduce overfitting more than Random Forest</li>
          <li> On high-dimensional data</li>
          <li> As a strong base model in stacking ensembles</li>
        </ul>
      </section>

      {/* 9. Pro-Level Tips */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Tips for Pro-Level Usage</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Use in stacking ensembles with XGBoost, SVM, etc.</li>
          <li>Analyze feature importance via <code>model.feature_importances_</code></li>
          <li>Handle imbalanced data with class weights or sampling</li>
          <li>Tune hyperparameters with RandomizedSearchCV for better performance</li>
        </ul>
      </section>
    </div>
  );
}
