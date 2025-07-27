export default function StackingGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">Stacking </h1>

      {/* 1. What is Stacking */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> What is Stacking?</h2>
        <p>
          Stacking is an ensemble learning technique that combines multiple base models and uses a
          meta-model (blender) to learn from their predictions, improving overall performance.
        </p>
        <p className="mt-2"> Goal: Better performance than any single model by combining their strengths.</p>
      </section>

      {/* 2. How Stacking Works */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> How Stacking Works</h2>
        <ul className="list-decimal ml-6 space-y-1">
          <li>Split training data into K folds.</li>
          <li>Train base models (e.g., SVM, Tree, XGBoost) on K-1 folds.</li>
          <li>Use held-out fold to generate out-of-fold predictions.</li>
          <li>Stack predictions → become features for meta-model.</li>
          <li>Train meta-model (e.g., Logistic Regression) on stacked features.</li>
          <li>For test data: Base models predict → meta-model combines results.</li>
        </ul>
      </section>

      {/* 3. Stacking vs Others */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Stacking vs Other Ensemble Methods</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">Bagging</th>
                <th className="border px-2 py-1">Boosting</th>
                <th className="border px-2 py-1">Stacking</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Combines Models", "Same", "Sequential weak learners", " Different models"],
                ["Model Diversity", "Low", "Low", " High"],
                ["Learning Strategy", "Parallel", "Sequential", " Meta-model"],
                ["Overfitting Risk", "Low", "Medium", " Can be high"],
              ].map(([feature, bagging, boosting, stacking], i) => (
                <tr key={i}>
                  <td className="border px-2 py-1">{feature}</td>
                  <td className="border px-2 py-1">{bagging}</td>
                  <td className="border px-2 py-1">{boosting}</td>
                  <td className="border px-2 py-1">{stacking}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 4. Python Example */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Python Example (sklearn)</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

base_models = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

meta_model = LogisticRegression()

model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`}
        </pre>
      </section>

      {/* 5. Use Cases */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Common Base and Meta Models</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Use Case</th>
                <th className="border px-2 py-1">Base Models</th>
                <th className="border px-2 py-1">Meta Model</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1">Classification</td>
                <td className="border px-2 py-1">SVM, Tree, KNN, XGBoost</td>
                <td className="border px-2 py-1">Logistic Regression</td>
              </tr>
              <tr>
                <td className="border px-2 py-1">Regression</td>
                <td className="border px-2 py-1">Lasso, Ridge, LightGBM</td>
                <td className="border px-2 py-1">Linear Regression</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 6. Tips */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2">Tips for Effective Stacking</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Use diverse base models (tree + linear + distance-based).</li>
          <li>Meta-model should be simple and regularized.</li>
          <li>Use out-of-fold predictions to prevent overfitting.</li>
          <li>Don't leak test data into training!</li>
          <li>Stack ensembles too (e.g., RF + XGBoost).</li>
        </ul>
      </section>

      {/* 7. Manual Stacking */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Manual Stacking with CV</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5)
base_preds_train = np.zeros((X_train.shape[0], len(base_models)))
base_preds_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    test_fold_preds = []
    for train_idx, val_idx in kf.split(X_train):
        X_t, X_val = X_train[train_idx], X_train[val_idx]
        y_t, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_t, y_t)
        base_preds_train[val_idx, i] = model.predict(X_val)
        test_fold_preds.append(model.predict(X_test))
    base_preds_test[:, i] = np.mean(test_fold_preds, axis=0)

meta_model.fit(base_preds_train, y_train)
final_preds = meta_model.predict(base_preds_test)`}
        </pre>
      </section>

      {/* 8. Libraries */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Libraries for Stacking</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> <code>sklearn.ensemble.StackingClassifier</code></li>
          <li> <code>mlxtend</code> – Flexible, customizable stacking</li>
          <li> <code>vecstack</code> – Fast stacking for competitions</li>
          <li> <code>PyCaret</code> – AutoML with stacking</li>
        </ul>
      </section>

      {/* 9. Real Use Cases */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Real-World Use Cases</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Kaggle competitions</li>
          <li>Combining ML and DL models</li>
          <li>Credit risk scoring</li>
          <li>Medical diagnosis systems</li>
          <li>Model blending in production APIs</li>
        </ul>
      </section>

      {/* Summary */}
      <section>
        <h2 className="font-semibold text-base sm:text-lg mb-2"> Summary</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Pros</th>
                <th className="border px-2 py-1">Cons</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border px-2 py-1"> Combines strengths of models</td>
                <td className="border px-2 py-1"> Can be slow to train</td>
              </tr>
              <tr>
                <td className="border px-2 py-1"> Often improves performance</td>
                <td className="border px-2 py-1"> Risk of overfitting</td>
              </tr>
              <tr>
                <td className="border px-2 py-1"> Easy with sklearn</td>
                <td className="border px-2 py-1"> More complex to debug</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
