export default function CatBoostGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">CatBoost </h1>

      {/* 1. What is CatBoost? */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is CatBoost?</h2>
        <p>
          CatBoost (Categorical Boosting) is a gradient boosting library developed by Yandex.
          It offers high performance and native support for categorical features.
        </p>
        <ul className="list-disc ml-6">
          <li> Handles categorical features natively</li>
          <li> Great out-of-the-box performance</li>
          <li> Supports classification, regression, and ranking</li>
          <li> Less hyperparameter tuning than XGBoost/LightGBM</li>
        </ul>
      </section>

      {/* 2. Why CatBoost is Special */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Why CatBoost is Special</h2>
        <div className="overflow-x-auto">
          <table className="w-full border text-xs sm:text-sm border-gray-300">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">Benefit</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Native Categorical Support", "No one-hot or label encoding needed"],
                ["Ordered Boosting", "Prevents target leakage"],
                ["Efficient with Defaults", "Performs well without tuning"],
                ["Robust to Overfitting", "Especially on smaller datasets"],
                ["GPU Support", "Speeds up training"],
              ].map(([feature, benefit], i) => (
                <tr key={i}>
                  <td className="border px-2 py-1">{feature}</td>
                  <td className="border px-2 py-1">{benefit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 3. Installation */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Installation</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
pip install catboost
        </pre>
      </section>

      {/* 4. Python Example */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Basic Python Example</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('titanic.csv')
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

cat_features = ['Pclass', 'Sex']
X['Age'].fillna(X['Age'].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
model.fit(X_train, y_train, cat_features=cat_features)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`}
        </pre>
      </section>

      {/* 5. Key Parameters */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Key Parameters</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Parameter</th>
                <th className="border px-2 py-1">Meaning</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["iterations", "Number of boosting rounds"],
                ["learning_rate", "Step size shrinkage"],
                ["depth", "Tree depth"],
                ["cat_features", "List of categorical feature names or indices"],
                ["eval_metric", "Metric to evaluate (e.g., Accuracy, Logloss)"],
                ["loss_function", "Loss to minimize (e.g., Logloss, RMSE)"],
                ["verbose", "Training output verbosity"],
              ].map(([param, desc], i) => (
                <tr key={i}>
                  <td className="border px-2 py-1 font-mono">{param}</td>
                  <td className="border px-2 py-1">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 6. Advanced Features */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Advanced Features</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>Categorical Handling:</strong> No preprocessing needed.</li>
          <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
model.fit(X, y, cat_features=['gender', 'country'])
          </pre>
          <li><strong>Missing Values:</strong> Handled internally.</li>
          <li><strong>Feature Importance:</strong></li>
          <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
model.get_feature_importance(prettified=True)
          </pre>
          <li><strong>Cross-Validation:</strong></li>
          <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`from catboost import Pool, cv

dataset = Pool(X, label=y, cat_features=cat_features)
params = {"iterations": 100, "learning_rate": 0.1, "loss_function": "Logloss"}

cv_result = cv(pool=dataset, params=params, fold_count=5, verbose=False)`}
          </pre>
        </ul>
      </section>

      {/* 7. Comparison Table */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">CatBoost vs Others</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">CatBoost</th>
                <th className="border px-2 py-1">XGBoost</th>
                <th className="border px-2 py-1">LightGBM</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Native Categorical", " Yes", " No", " Partial"],
                ["Training Speed", "Medium", "Medium-Fast", " Fastest"],
                ["Accuracy", " High", " High", " High"],
                ["GPU Support", " Yes", " Yes", " Yes"],
                ["Default Performance", " Excellent", " Needs tuning", " Needs tuning"],
              ].map(([feat, cat, xgb, lgb], i) => (
                <tr key={i}>
                  <td className="border px-2 py-1">{feat}</td>
                  <td className="border px-2 py-1">{cat}</td>
                  <td className="border px-2 py-1">{xgb}</td>
                  <td className="border px-2 py-1">{lgb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 8. Real Use Cases */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">8. Real Use Cases</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Bank loan approval</li>
          <li>Medical diagnosis</li>
          <li>CTR prediction</li>
          <li>Recommender systems</li>
          <li>Fraud detection</li>
        </ul>
      </section>

      {/* Summary */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Summary: Why Use CatBoost?</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> Native categorical handling</li>
          <li> Strong default performance</li>
          <li> Less prone to overfitting</li>
          <li> Easy to use, integrates with pandas + scikit-learn</li>
          <li> GPU support built-in</li>
        </ul>
      </section>
    </div>
  );
}
