export default function GradientBoostingGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">
        Gradient Boosting 
      </h1>

      {/* 1. What is Gradient Boosting? */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is Gradient Boosting?</h2>
        <p>
          Gradient Boosting is a machine learning technique used for regression and classification problems.
          It builds a strong model by combining multiple weak learners, typically decision trees.
        </p>
      </section>

      {/* 2. Basic Concepts */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Basic Concepts</h2>
        <p><strong> Boosting vs Bagging</strong></p>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>Bagging:</strong> Parallel training, reduces variance (e.g., Random Forest)</li>
          <li><strong>Boosting:</strong> Sequential training, reduces bias (e.g., Gradient Boosting, AdaBoost)</li>
        </ul>

        <p><strong> Idea Behind Gradient Boosting:</strong></p>
        <ol className="list-decimal ml-6 space-y-1">
          <li>Fit an initial model.</li>
          <li>Calculate residuals (errors).</li>
          <li>Train next model on residuals.</li>
          <li>Repeat until stopping condition (e.g., max iterations).</li>
        </ol>
      </section>

      {/* 3. Step-by-Step Working */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Step-by-Step Working</h2>
        <p><strong>Step 1:</strong> Initialize the model (mean of targets).</p>

        <p><strong>Step 2:</strong> For m = 1 to M (number of trees):</p>
        <ul className="list-disc ml-6 space-y-1">
          <li>Compute residuals from previous predictions.</li>
          <li>Fit a regression tree to residuals.</li>
          <li>Calculate update value (gamma).</li>
          <li>Update model: Fm = Fm-1 + learning_rate * gamma * h(x)</li>
        </ul>
      </section>

      {/* 4. Key Hyperparameters */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Key Hyperparameters</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>n_estimators:</strong> Number of boosting stages (trees)</li>
          <li><strong>learning_rate:</strong> Shrinks contribution of each tree</li>
          <li><strong>max_depth:</strong> Max depth of individual trees</li>
          <li><strong>min_samples_split:</strong> Minimum samples to split a node</li>
          <li><strong>subsample:</strong> Fraction of samples for each tree</li>
          <li><strong>loss:</strong> Loss function to be minimized</li>
        </ul>
      </section>

      {/* 5. Pros and Cons */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Pros and Cons</h2>
        <p><strong> Pros</strong></p>
        <ul className="list-disc ml-6">
          <li>High accuracy</li>
          <li>Works with mixed data</li>
          <li>Robust to outliers with tuning</li>
        </ul>
        <p className="mt-2"><strong> Cons</strong></p>
        <ul className="list-disc ml-6">
          <li>Slow to train</li>
          <li>Sensitive to parameters</li>
          <li>Less interpretable</li>
        </ul>
      </section>

      {/* 6. Popular Implementations */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Popular Implementations</h2>

        <p><strong>Scikit-learn</strong></p>
        <div className="bg-gray-100 p-2 rounded text-xs whitespace-pre-wrap font-mono">
{`from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)`}
        </div>

        <p><strong>XGBoost</strong></p>
        <div className="bg-gray-100 p-2 rounded text-xs whitespace-pre-wrap font-mono">
{`import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)`}
        </div>

        <p><strong>Others:</strong></p>
        <ul className="list-disc ml-6">
          <li><strong>LightGBM:</strong> Fast and GPU-friendly</li>
          <li><strong>CatBoost:</strong> Best for categorical features</li>
        </ul>
      </section>

      {/* 7. Evaluation Metrics */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Evaluation Metrics</h2>
        <p><strong>Classification:</strong> Accuracy, Precision, Recall, F1, AUC</p>
        <p><strong>Regression:</strong> RMSE, MAE, R²</p>
      </section>

      {/* 8. Advanced Topics */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Advanced Topics</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>Regularization:</strong> Control overfitting with learning_rate, subsample, etc.</li>
          <li><strong>Early stopping:</strong> Stop training if validation doesn't improve</li>
          <li><strong>Feature importance:</strong> Use .feature_importances_ or SHAP for interpretability</li>
        </ul>

        <div className="bg-gray-100 p-2 rounded text-xs whitespace-pre-wrap font-mono">
{`from xgboost import XGBClassifier

model = XGBClassifier(early_stopping_rounds=10, eval_set=[(X_val, y_val)])`}
        </div>
      </section>

      {/* 9. Real Use Cases */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Real Use Cases</h2>
        <ul className="list-disc ml-6">
          <li>Credit scoring</li>
          <li>Fraud detection</li>
          <li>Marketing predictions</li>
          <li>Search ranking systems</li>
        </ul>
      </section>
    </div>
  );
}
