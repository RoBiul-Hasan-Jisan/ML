export default function SVMGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">Support Vector Machines Classification </h1>

      {/* 1 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is SVM?</h2>
        <p>
          Support Vector Machines are supervised learning models used for classification and regression, but primarily famous for classification tasks.
          SVM finds the best hyperplane that maximally separates classes in the feature space.
        </p>
      </section>

      {/* 2 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Core Idea — Maximum Margin Classifier</h2>
        <p>
          The hyperplane separates classes such that the margin (distance between the hyperplane and closest points from each class) is maximized.
          These closest points are called support vectors.
          Maximizing margin helps improve generalization.
        </p>
      </section>

      {/* 3 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Mathematical Formulation</h2>
        <p>
          Given data points <code>(xᵢ, yᵢ)</code>, where <code>yᵢ ∈ {'{-1, +1}'}</code>, find a hyperplane <code>w ⋅ x + b = 0</code> that satisfies:
        </p>
        <p className="ml-6 italic">
          <code>yᵢ (w ⋅ xᵢ + b) ≥ 1 ∀ i</code>
        </p>
        <p>
          Objective:
        </p>
        <p className="ml-6 italic">
          <code>min<sub>w,b</sub> ½ ∥w∥²</code><br />
          subject to constraints above.
        </p>
      </section>

      {/* 4 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Soft Margin for Non-Separable Data</h2>
        <p>
          Real-world data may not be perfectly separable, so SVM uses slack variables <code>ξᵢ ≥ 0</code> to allow some misclassifications:
        </p>
        <p className="ml-6 italic">
          <code>yᵢ (w ⋅ xᵢ + b) ≥ 1 − ξᵢ</code>
        </p>
        <p>Objective becomes:</p>
        <p className="ml-6 italic">
          <code>min<sub>w,b,ξ</sub> ½ ∥w∥² + C ∑ ξᵢ</code>
        </p>
        <p>
          <code>C</code> controls the trade-off between maximizing margin and minimizing classification error.
        </p>
      </section>

      {/* 5 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Kernels — Handling Non-Linear Data</h2>
        <p>
          If data isn’t linearly separable, the kernel trick maps data to higher-dimensional space.
          Common kernels include:
        </p>
        <ul className="list-disc ml-6">
          <li>Linear</li>
          <li>Polynomial</li>
          <li>Radial Basis Function (RBF) (Gaussian)</li>
          <li>Sigmoid</li>
        </ul>
      </section>

      {/* 6 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Visualization of SVM (2D example)</h2>
        <p>
          Support vectors are circled points closest to the hyperplane.<br />
          Margin is the band between two dashed lines.<br />
          Hyperplane is the solid line.
        </p>
      </section>

      {/* 7 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Python Example (Using Scikit-learn)</h2>
        <div className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# For binary classification, select two classes
X = X[y != 2]
y = y[y != 2]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create SVM classifier
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`}
          </pre>
        </div>
      </section>

      {/* 8 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Important Parameters in SVC</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Parameter</th>
              <th className="p-2 border">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="p-2 border">kernel</td><td className="p-2 border">Kernel type: 'linear', 'rbf', 'poly', 'sigmoid'</td></tr>
            <tr><td className="p-2 border">C</td><td className="p-2 border">Regularization parameter (trade-off margin vs error)</td></tr>
            <tr><td className="p-2 border">gamma</td><td className="p-2 border">Kernel coefficient for 'rbf', 'poly', 'sigmoid'</td></tr>
            <tr><td className="p-2 border">degree</td><td className="p-2 border">Degree for polynomial kernel</td></tr>
          </tbody>
        </table>
      </section>

      {/* 9 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Advantages of SVM</h2>
        <ul className="list-disc ml-6">
          <li>Effective in high-dimensional spaces</li>
          <li>Works well with clear margin of separation</li>
          <li>Robust against overfitting (especially with proper C)</li>
          <li>Kernel trick enables non-linear classification</li>
        </ul>
      </section>

      {/* 10 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Limitations</h2>
        <ul className="list-disc ml-6">
          <li>Not suitable for very large datasets (slow training)</li>
          <li>Sensitive to choice of kernel and parameters</li>
          <li>Less effective when classes heavily overlap</li>
        </ul>
      </section>

      {/* 11 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Intuition Recap</h2>
        <p>
          Try to find a line (or hyperplane) that separates classes with the widest gap.
          Points closest to this line (support vectors) are most important.
          If data is not separable, allow some errors with a soft margin.
          Use kernels to handle complex boundaries.
        </p>
      </section>

      {/* 12 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Common Interview Questions</h2>
        <ul className="list-disc ml-6">
          <li>What is a support vector?</li>
          <li>Explain the margin in SVM.</li>
          <li>How does the kernel trick work?</li>
          <li>Difference between hard margin and soft margin?</li>
          <li>How do you tune hyperparameters like C and gamma?</li>
        </ul>
      </section>

      {/* 13 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Difference Between SVM Classification and SVM Regression</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Aspect</th>
              <th className="p-2 border">SVM Classification</th>
              <th className="p-2 border">SVM Regression (SVR)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Purpose</td>
              <td className="p-2 border">Classify data into discrete categories</td>
              <td className="p-2 border">Predict continuous real-valued output</td>
            </tr>
            <tr>
              <td className="p-2 border">Output</td>
              <td className="p-2 border">Class label (e.g., -1 or +1)</td>
              <td className="p-2 border">Real number (e.g., price, temperature)</td>
            </tr>
            <tr>
              <td className="p-2 border">Decision Function</td>
              <td className="p-2 border">Finds hyperplane maximizing margin between classes</td>
              <td className="p-2 border">Finds function f(x) that fits data within ε-insensitive tube</td>
            </tr>
            <tr>
              <td className="p-2 border">Loss Function</td>
              <td className="p-2 border">Hinge loss</td>
              <td className="p-2 border">ε-insensitive loss (errors within ε ignored)</td>
            </tr>
            <tr>
              <td className="p-2 border">Margin</td>
              <td className="p-2 border">Margin between classes (maximize separation)</td>
              <td className="p-2 border">Margin of tolerance ε around the regression function</td>
            </tr>
            <tr>
              <td className="p-2 border">Optimization Goal</td>
              <td className="p-2 border">Minimize classification error and maximize margin</td>
              <td className="p-2 border">Minimize error within ε margin and model complexity</td>
            </tr>
            <tr>
              <td className="p-2 border">Support Vectors</td>
              <td className="p-2 border">Points closest to decision boundary</td>
              <td className="p-2 border">Points outside ε-tube or on boundary of tube</td>
            </tr>
            <tr>
              <td className="p-2 border">Kernel Trick</td>
              <td className="p-2 border">Used to handle non-linear classification</td>
              <td className="p-2 border">Used to handle non-linear regression</td>
            </tr>
            <tr>
              <td className="p-2 border">Typical Use Cases</td>
              <td className="p-2 border">Spam detection, image classification</td>
              <td className="p-2 border">Stock price prediction, temperature forecasting</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* 14 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Intuition Summary</h2>
        <p>
          <strong>SVM Classification:</strong> Finds the best boundary to separate classes with the largest margin.<br />
          <strong>SVM Regression (SVR):</strong> Finds a function (line or curve) that fits data points while keeping deviations within a small margin ε.
        </p>
      </section>

      {/* 15 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Basic SVR Formulation</h2>
        <p>
          Given data <code>(xᵢ, yᵢ)</code>, SVR tries to find function <code>f(x) = w ⋅ x + b</code> such that:
        </p>
        <p className="ml-6 italic">
          <code>|yᵢ − f(xᵢ)| ≤ ε</code>
        </p>
        <p>
          And penalizes deviations larger than <code>ε</code>.
        </p>
      </section>

      {/* 16 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Usage in scikit-learn</h2>
        <p><strong>Classification:</strong></p>
        <div className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train, y_train)`}
          </pre>
        </div>
        <p className="mt-4"><strong>Regression:</strong></p>
        <div className="bg-gray-100 p-2 rounded text-xs sm:text-sm overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.svm import SVR
model = SVR(kernel='rbf', epsilon=0.1)
model.fit(X_train, y_train)`}
          </pre>
        </div>
      </section>

      {/* 17 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Summary Table</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Feature</th>
              <th className="p-2 border">SVM Classification</th>
              <th className="p-2 border">SVM Regression (SVR)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td className="p-2 border">Problem Type</td><td className="p-2 border">Classification</td><td className="p-2 border">Regression</td></tr>
            <tr><td className="p-2 border">Objective</td><td className="p-2 border">Maximize margin between classes</td><td className="p-2 border">Fit data within ε margin</td></tr>
            <tr><td className="p-2 border">Loss Function</td><td className="p-2 border">Hinge loss</td><td className="p-2 border">ε-insensitive loss</td></tr>
            <tr><td className="p-2 border">Output</td><td className="p-2 border">Class label</td><td className="p-2 border">Continuous value</td></tr>
            <tr><td className="p-2 border">Support Vectors</td><td className="p-2 border">Points closest to boundary</td><td className="p-2 border">Points outside ε tube</td></tr>
            <tr><td className="p-2 border">Kernel Trick</td><td className="p-2 border">Used for non-linear classification</td><td className="p-2 border">Used for non-linear regression</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
