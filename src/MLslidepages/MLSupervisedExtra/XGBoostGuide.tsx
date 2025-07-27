export default function XGBoostGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">XGBoost </h1>

      {/* 1. What is XGBoost? */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is XGBoost?</h2>
        <p>
          XGBoost stands for <strong>Extreme Gradient Boosting</strong>. It’s an optimized and regularized version of Gradient Boosting, designed for speed and performance.
        </p>
        <ul className="list-disc ml-6">
          <li>Fast</li>
          <li> Accurate</li>
          <li> Regularized to reduce overfitting</li>
          <li> Works for classification and regression</li>
        </ul>
      </section>

      {/* 2. Why Is It So Powerful? */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Why Is XGBoost So Powerful?</h2>
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
                ["Parallelized Trees", "Faster training"],
                ["Regularization", "Avoids overfitting (L1 and L2)"],
                ["Missing Value Handling", "Handles NaNs natively"],
                ["Tree Pruning", "Uses max depth + loss gain (not greedy)"],
                ["Column Subsampling", "Prevents overfitting, speeds training"],
                ["Cache Awareness", "Efficient memory usage"],
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

      {/* 3. How XGBoost Works */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> How XGBoost Works (Step-by-Step)</h2>
        <p><strong>Step 1:</strong> Initialize with mean:</p>
        <p><code>ŷ(0) = mean(y)</code></p>

        <p><strong>Step 2:</strong> For t = 1 to T:</p>
        <ul className="list-disc ml-6">
          <li>Calculate gradients (g<sub>i</sub>) and hessians (h<sub>i</sub>)</li>
          <li>Train tree to minimize loss using g and h</li>
          <li>Update prediction: <code>ŷ(t) = ŷ(t−1) + η × f<sub>t</sub>(x)</code></li>
        </ul>
        <p>Where:</p>
        <ul className="ml-6 list-disc">
          <li><code>f<sub>t</sub>(x)</code>: prediction from tree t</li>
          <li><code>η</code>: learning rate</li>
        </ul>
      </section>

      {/* 4. Loss + Regularization */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> XGBoost Loss + Regularization</h2>
        <p>
          XGBoost minimizes: <br />
          <code>L = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)</code>
        </p>
        <p>Where:</p>
        <ul className="list-disc ml-6">
          <li><code>Ω(f) = γT + ½λΣwⱼ²</code></li>
          <li><strong>γ</strong>: cost of each leaf (controls complexity)</li>
          <li><strong>λ</strong>: L2 regularization on leaf weights</li>
        </ul>
      </section>

      {/* 5. Python Example */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> XGBoost in Python</h2>
        <p><strong> Installation:</strong></p>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`pip install xgboost`}
        </pre>

        <p><strong> Basic Classification Example:</strong></p>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`}
        </pre>
      </section>

      {/* 6. Important Hyperparameters */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Important Hyperparameters</h2>
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
                ["n_estimators", "Number of trees"],
                ["max_depth", "Max depth of tree"],
                ["learning_rate", "Step size shrinkage"],
                ["subsample", "Row sampling ratio"],
                ["colsample_bytree", "Column sampling per tree"],
                ["gamma", "Min loss reduction for split"],
                ["lambda", "L2 regularization term"],
                ["alpha", "L1 regularization term"],
                ["objective", "Loss function (e.g., binary:logistic)"],
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

      {/* 7. Early Stopping */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Evaluation & Early Stopping</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`model = xgb.XGBClassifier(n_estimators=100)
model.fit(X_train, y_train,
          early_stopping_rounds=10,
          eval_set=[(X_test, y_test)],
          verbose=True)`}
        </pre>
      </section>

      {/* 8. Feature Importance */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Feature Importance</h2>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`xgb.plot_importance(model)`}
        </pre>
        <p>Or for better explainability:</p>
        <pre className="bg-gray-100 p-2 rounded text-xs font-mono whitespace-pre-wrap">
{`pip install shap`}
        </pre>
      </section>

      {/* 9. Real Use Cases */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Real Use Cases</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> Fraud Detection</li>
          <li> Credit Scoring</li>
          <li> Search Ranking</li>
          <li> CTR Prediction</li>
          <li> Time Series Forecasting</li>
        </ul>
      </section>

      {/* 10. Learn More */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Learn More</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li> XGBoost Docs</li>
          <li> StatQuest – XGBoost Explained</li>
          <li> Datasets: Titanic, Ames Housing, Santander</li>
          <li> Practice: Kaggle XGBoost Challenges</li>
        </ul>
      </section>

      {/* 11. XGBoost vs Others */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> XGBoost vs Other Algorithms</h2>
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-300 text-xs sm:text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="border px-2 py-1">Model</th>
                <th className="border px-2 py-1">Speed</th>
                <th className="border px-2 py-1">Accuracy</th>
                <th className="border px-2 py-1">Categorical?</th>
                <th className="border px-2 py-1">Interpretability</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Random Forest", "Medium", "Medium", "✅ (encode)", "✅✅✅"],
                ["AdaBoost", "Slow", "Medium", "No", "✅✅"],
                ["Gradient Boost", "Medium", "High", "No", "✅"],
                ["XGBoost", "✅ Fastest", "✅✅✅", "No", "✅✅ w/ SHAP"],
                ["LightGBM", "✅✅ Faster", "✅✅✅", "✅ Native", "✅✅ w/ SHAP"],
                ["CatBoost", "Fast", "✅✅ High", "✅✅✅ Native", "✅✅ w/ SHAP"],
              ].map(([model, speed, acc, cat, interp], i) => (
                <tr key={i}>
                  <td className="border px-2 py-1">{model}</td>
                  <td className="border px-2 py-1">{speed}</td>
                  <td className="border px-2 py-1">{acc}</td>
                  <td className="border px-2 py-1">{cat}</td>
                  <td className="border px-2 py-1">{interp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Summary */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Summary: Why Use XGBoost?</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Fast and scalable</li>
          <li>Regularized to reduce overfitting</li>
          <li>Parallel training</li>
          <li>Best-in-class accuracy</li>
          <li>Easy integration with scikit-learn</li>
        </ul>
      </section>
    </div>
  );
}
