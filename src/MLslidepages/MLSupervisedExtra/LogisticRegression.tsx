//import React from "react";

export default function LogisticRegression() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-5xl mx-auto space-y-6">
      <h1 className="text-3xl sm:text-4xl font-bold text-blue-700">
        Logistic Regression 
      </h1>

      {/* Section 1 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">What is Logistic Regression?</h2>
        <p>
          Logistic Regression is a supervised machine learning algorithm used for <strong>classification</strong> problems. Despite the name “regression,” it is used for classification, not regression.
        </p>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">When to Use Logistic Regression?</h2>
        <ul className="list-disc list-inside">
          <li>Binary: Yes/No, 0/1, Spam/Not Spam</li>
          <li>Multiclass: Class A, B, or C (with extensions)</li>
        </ul>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">How Does It Work?</h2>
        <p>It fits a sigmoid curve to the data, predicting probabilities between 0 and 1.</p>
        <ul className="list-disc list-inside">
          <li>Linear: <code>z = w1·x1 + w2·x2 + ... + wn·xn + b</code></li>
          <li>Sigmoid: <code>ŷ = 1 / (1 + e<sup>−z</sup>)</code></li>
        </ul>
        <p><strong>Prediction Rule:</strong> If ŷ ≥ 0.5 → Class = 1, else → Class = 0</p>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Types of Logistic Regression</h2>
        <ul className="list-disc list-inside">
          <li>Binary</li>
          <li>Multinomial</li>
          <li>Ordinal</li>
        </ul>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Code Example in Python</h2>
        <pre className="bg-gray-100 p-4 rounded text-sm sm:text-base overflow-x-auto whitespace-pre-wrap break-words">
{`from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))`}
        </pre>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Evaluation Metrics</h2>
        <ul className="list-disc list-inside">
          <li>Accuracy</li>
          <li>Precision</li>
          <li>Recall</li>
          <li>F1 Score</li>
          <li>Confusion Matrix</li>
          <li>ROC-AUC</li>
        </ul>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Real-Life Use Cases</h2>
        <ul className="list-disc list-inside">
          <li>Email spam detection</li>
          <li>Disease diagnosis</li>
          <li>Loan approval</li>
          <li>Image classification</li>
          <li>Customer churn prediction</li>
        </ul>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Assumptions</h2>
        <ul className="list-disc list-inside">
          <li>Linearity of log odds</li>
          <li>Independent observations</li>
          <li>No multicollinearity</li>
          <li>Large sample size</li>
        </ul>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Advantages</h2>
        <ul className="list-disc list-inside">
          <li>Simple and fast</li>
          <li>Probabilistic outputs</li>
          <li>Easy to interpret</li>
        </ul>
      </section>

      {/* Section 10 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Disadvantages</h2>
        <ul className="list-disc list-inside">
          <li>Can’t solve non-linear problems</li>
          <li>Prone to underfitting</li>
          <li>Sensitive to outliers</li>
        </ul>
      </section>

      {/* Section 11 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Interview Questions</h2>
        <h3 className="font-semibold mt-2">Basic</h3>
        <ul className="list-disc list-inside">
          <li>What is Logistic Regression?</li>
          <li>Difference from Linear Regression?</li>
          <li>Why is it called “regression”?</li>
        </ul>
        <h3 className="font-semibold mt-2">Intermediate</h3>
        <ul className="list-disc list-inside">
          <li>What is sigmoid function?</li>
          <li>What is the cost function?</li>
          <li>Interpret model coefficients?</li>
        </ul>
        <h3 className="font-semibold mt-2">Advanced</h3>
        <ul className="list-disc list-inside">
          <li>What is log-odds?</li>
          <li>Multiclass Logistic Regression?</li>
          <li>What are L1/L2 regularization?</li>
          <li>How to handle imbalanced data?</li>
        </ul>
      </section>

      {/* Section 12 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Cost Function</h2>
        <p>Log Loss (Binary Cross Entropy):</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap break-words">
{`J(θ) = -1/n * Σ [ yᵢ·log(ŷᵢ) + (1 - yᵢ)·log(1 - ŷᵢ) ]`}
        </pre>
        <p>
          Where:<br />
          • 𝑦ᵢ = actual class (0 or 1)<br />
          • ŷᵢ = predicted probability
        </p>
      </section>

      {/* Section 13 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Regularization</h2>
        <p>To prevent overfitting:</p>
        <ul className="list-disc list-inside">
          <li>L1 (Lasso): adds <code>Σ|wᵢ|</code></li>
          <li>L2 (Ridge): adds <code>Σwᵢ²</code></li>
        </ul>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap break-words">
{`LogisticRegression(penalty='l1', solver='liblinear')  # L1
LogisticRegression(penalty='l2')  # L2`}
        </pre>
      </section>

      {/* Section 14 */}
      <section>
        <h2 className="text-xl sm:text-2xl font-semibold">Summary Table</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left border border-gray-300 text-sm sm:text-base">
            <thead className="bg-blue-100">
              <tr>
                <th className="p-2 border">Feature</th>
                <th className="p-2 border">Logistic Regression</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2 border">Type</td>
                <td className="p-2 border">Classification (binary/multiclass)</td>
              </tr>
              <tr>
                <td className="p-2 border">Equation</td>
                <td className="p-2 border">Sigmoid of linear combination</td>
              </tr>
              <tr>
                <td className="p-2 border">Output</td>
                <td className="p-2 border">Probability (0–1)</td>
              </tr>
              <tr>
                <td className="p-2 border">Decision Boundary</td>
                <td className="p-2 border">Usually 0.5</td>
              </tr>
              <tr>
                <td className="p-2 border">Cost Function</td>
                <td className="p-2 border">Log Loss</td>
              </tr>
              <tr>
                <td className="p-2 border">Evaluation</td>
                <td className="p-2 border">Accuracy, Precision, Recall, F1</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <div className="px-4 sm:px-6 lg:px-8 max-w-screen-lg mx-auto">
        <h1 className="text-3xl sm:text-4xl font-bold text-blue-700 mb-6 text-center">
          Logistic Regression Coding Steps
        </h1>

        {/* Each Step Section */}
        {[
          {
            title: "Import Required Libraries",
            code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns`,
          },
          {
            title: "Load or Create the Dataset",
            code: `# Load a dataset
data = pd.read_csv("your_dataset.csv")

# OR create a simple dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification`,
          },
          {
            title: "Explore and Preprocess Data",
            code: `print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Optional: Encoding or Scaling
# from sklearn.preprocessing import StandardScaler`,
          },
          {
            title: "Split the Dataset",
            code: `X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)`,
          },
          {
            title: "Train the Logistic Regression Model",
            code: `model = LogisticRegression()
model.fit(X_train, y_train)`,
          },
          {
            title: "Make Predictions",
            code: `y_pred = model.predict(X_test)`,
          },
          {
            title: "Evaluate the Model",
            code: `# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))`,
          },
          {
            title: "Use the Model for New Predictions",
            code: `new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_data)
print("Predicted class:", prediction)`,
          },
        ].map((step, idx) => (
          <section key={idx} className="mb-8">
            <h2 className="text-xl sm:text-2xl font-semibold mb-2">{step.title}</h2>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap">
              {step.code}
            </pre>
          </section>
        ))}

        {/* Pro Tips Section */}
        <section className="mb-12">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4">Optional Pro Tips</h2>

          <p className="mb-1 font-medium">Feature Scaling:</p>
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap mb-4">
{`from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)`}
          </pre>

          <p className="mb-1 font-medium">Hyperparameter Tuning (GridSearch):</p>
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap">
{`from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)`}
          </pre>
        </section>
      </div>
    </div>
  );
}
