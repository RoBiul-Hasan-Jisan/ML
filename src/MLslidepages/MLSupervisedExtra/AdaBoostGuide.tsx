export default function AdaBoostGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">AdaBoost </h1>

      {/* 1. What is AdaBoost? */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is AdaBoost?</h2>
        <p>
          AdaBoost stands for <strong>Adaptive Boosting</strong>It’s one of the earliest and most popular boosting algorithms. It combines multiple weak learners (usually decision stumps) to form a strong classifier.
        </p>
        <p><strong> Main Idea:</strong> Focus more on samples that previous classifiers got wrong.</p>
      </section>

      {/* 2. How It Works */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> How AdaBoost Works (Step-by-Step)</h2>
        <p>We assume binary classification where labels are -1 and +1.</p>

        <p><strong>Step 1: Initialize Weights</strong></p>
        <p>Assign equal weight to all training examples: <code>w_i = 1/n</code></p>

        <p><strong>Step 2: For m = 1 to M (number of weak learners)</strong></p>
        <ul className="list-disc ml-6 space-y-1">
          <li>Train a weak learner on the data using weights <code>w_i</code></li>
          <li>Compute weighted error:
            <br /><code>error = ∑ w_i × I(y_i ≠ h_m(x_i))</code>
          </li>
          <li>Compute model weight:
            <br /><code>α_m = 0.5 × ln((1 - error) / error)</code>
          </li>
          <li>Update sample weights:
            <br /><code>w_i = w_i × e^(−α_m × y_i × h_m(x_i))</code>
          </li>
          <li>Normalize the weights so they sum to 1.</li>
        </ul>

        <p><strong>Step 3: Final Prediction</strong></p>
        <p><code>H(x) = sign(Σ α_m × h_m(x))</code></p>
      </section>

      {/* 3. Visual Intuition */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Visual Intuition</h2>
        <p>
          The first model misclassifies some data points. The next model focuses on these hard examples by assigning them more weight. Models are combined using weighted voting.
        </p>
      </section>

      {/* 4. AdaBoost Parameters */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> AdaBoost Parameters</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>n_estimators:</strong> Number of weak learners</li>
          <li><strong>learning_rate:</strong> Shrinks contribution of each learner</li>
          <li><strong>base_estimator:</strong> Weak learner (default: decision stump)</li>
        </ul>
      </section>

      {/* 5. When to Use */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> When to Use AdaBoost</h2>
        <p><strong> Use When:</strong></p>
        <ul className="list-disc ml-6 space-y-1">
          <li>Data is structured/tabular</li>
          <li>You want higher accuracy than a single tree</li>
          <li>Data is not noisy or full of outliers</li>
        </ul>
        <p><strong> Avoid When:</strong></p>
        <ul className="list-disc ml-6 space-y-1">
          <li>Data has lots of noise</li>
          <li>You need very fast training (AdaBoost is sequential)</li>
        </ul>
      </section>

      {/* 6. Pros and Cons */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Pros and Cons</h2>
        <p><strong> Pros:</strong></p>
        <ul className="list-disc ml-6">
          <li>Simple to implement</li>
          <li>Significant accuracy improvement</li>
          <li>Works well with simple weak learners</li>
        </ul>

        <p className="mt-2"><strong> Cons:</strong></p>
        <ul className="list-disc ml-6">
          <li>Sensitive to outliers</li>
          <li>Not parallelizable due to sequential nature</li>
        </ul>
      </section>

      {/* 7. AdaBoost in Scikit-Learn */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> AdaBoost in Scikit-Learn</h2>
        <div className="bg-gray-100 p-2 rounded text-xs whitespace-pre-wrap font-mono">
{`from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
        </div>
      </section>

      {/* 8. Evaluation Metrics */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Evaluation Metrics</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Accuracy</li>
          <li>Precision, Recall</li>
          <li>F1 Score</li>
          <li>ROC AUC</li>
        </ul>
      </section>

      {/* 9. Related Algorithms */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Related Algorithms</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li><strong>Gradient Boosting:</strong> Uses gradients to minimize loss</li>
          <li><strong>XGBoost:</strong> Regularized, efficient GBM</li>
          <li><strong>LightGBM:</strong> Fast for large datasets</li>
          <li><strong>CatBoost:</strong> Great with categorical data</li>
          <li><strong>LogitBoost:</strong> Boosting for logistic regression</li>
        </ul>
      </section>
    </div>
  );
}
