export default function LightGBMGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-green-700">LightGBM </h1>

      {/* 1. What is LightGBM? */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is LightGBM?</h2>
        <p>
          LightGBM is a high-performance gradient boosting framework developed by Microsoft, optimized for large-scale data and speed.
        </p>
        <ul className="list-disc ml-6">
          <li> Faster than XGBoost</li>
          <li> Lower memory usage</li>
          <li> Native categorical feature support</li>
          <li> Parallel and GPU learning</li>
        </ul>
      </section>

      {/* 2. Why Use LightGBM */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Why Use LightGBM?</h2>
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
                ["Histogram-based split", "Faster training, lower memory"],
                ["Leaf-wise growth", "More accurate than level-wise"],
                ["Categorical handling", "No one-hot encoding needed"],
                ["Parallel & GPU support", "Scalable and lightning-fast"],
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
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Installation</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
pip install lightgbm
        </pre>
        
      </section>

      {/* 4. How LightGBM Works */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> How LightGBM Works</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>Leaf-wise growth:</strong> Grows the leaf with the highest gain (deeper trees, better accuracy)</li>
          <li><strong>Histogram binning:</strong> Groups features into bins to speed up training and reduce memory</li>
        </ul>
        <p> Can overfit — tune <code>num_leaves</code> and <code>min_data_in_leaf</code></p>
      </section>

      {/* 5. Parameters */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Key Parameters</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Parameter</th>
                <th className="border px-2 py-1">Description</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["num_leaves", "Tree complexity (larger = more expressive)"],
                ["max_depth", "Limit depth to prevent overfitting"],
                ["learning_rate", "Step size shrinkage"],
                ["n_estimators", "Boosting rounds"],
                ["min_data_in_leaf", "Min samples per leaf"],
                ["feature_fraction", "Fraction of features used per round"],
                ["bagging_fraction", "Fraction of rows used"],
                ["lambda_l1 / lambda_l2", "L1 / L2 regularization"],
                ["objective", "'binary', 'regression', etc."],
                ["boosting_type", "'gbdt', 'dart', 'goss'"],
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

      {/* 6. Python Example */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">LightGBM in Python</h2>

        <p><strong> Basic Classification:</strong></p>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`}
        </pre>

        <p><strong> Native Dataset + Validation:</strong></p>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
  'objective': 'binary',
  'metric': 'binary_logloss',
  'num_leaves': 31,
  'learning_rate': 0.05
}

model = lgb.train(params, train_data, valid_sets=[val_data],
                  num_boost_round=100, early_stopping_rounds=10)`}
        </pre>
      </section>

      {/* 7. Feature Importance */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Feature Importance</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`lgb.plot_importance(model, max_num_features=10)`}
        </pre>
        <p>Or use <code>SHAP</code> for deeper interpretability:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">pip install shap</pre>
      </section>

      {/* 8. LightGBM vs XGBoost */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">LightGBM vs XGBoost</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Feature</th>
                <th className="border px-2 py-1">LightGBM</th>
                <th className="border px-2 py-1">XGBoost</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Tree Growth", "Leaf-wise", "Level-wise"],
                ["Speed", " Faster", "Slower"],
                ["Accuracy", "Slightly better", "Good"],
                ["Memory Usage", "Lower", "Higher"],
                ["Categorical Features", " Native", " Needs encoding"],
                ["GPU Support", " Yes", " Yes"],
                ["Parallelization", "Efficient", " Efficient"],
              ].map(([feat, lgb, xgb], i) => (
                <tr key={i}>
                  <td className="border px-2 py-1">{feat}</td>
                  <td className="border px-2 py-1">{lgb}</td>
                  <td className="border px-2 py-1">{xgb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 9. Use Cases */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Real Use Cases</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Credit Scoring</li>
          <li>CTR Prediction</li>
          <li>Product Recommendation</li>
          <li>Real-time ranking systems</li>
          <li>Large-scale ML pipelines</li>
        </ul>
      </section>

      

      {/* 11. Variants */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Advanced Variants</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>DART Boosting:</strong> Drops trees randomly (prevents overfitting)</li>
          <li><strong>GOSS:</strong> Focuses on samples with large gradients (fast + accurate)</li>
        </ul>
      </section>

      {/* Summary */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Summary: Why Use LightGBM?</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> Very fast and scalable</li>
          <li> High accuracy (tune <code>num_leaves</code> to avoid overfitting)</li>
          <li> Native support for categorical features</li>
          <li> Easy to use with scikit-learn</li>
        </ul>
      </section>
    </div>
  );
}
