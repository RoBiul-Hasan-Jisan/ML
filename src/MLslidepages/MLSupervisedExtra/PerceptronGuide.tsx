// PerceptronGuide.tsx

export default function PerceptronGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
     <h1 className="text-xl sm:text-3xl font-bold text-blue-600 text-center mb-4 animate-fade-in">
  Perceptron
</h1>


      {/* Section 1 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is a Perceptron?</h2>
        <p>
          A perceptron is the simplest neural network model, introduced by Frank Rosenblatt in 1958.
          It is a binary classifier that maps input features to one of two possible outputs (classes),
          using a linear decision boundary.
        </p>
      </section>

      {/* Section 2 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> How Does It Work?</h2>
        <p>
          Given inputs <code>x₁, x₂, ..., xₙ</code>, weights <code>w₁, w₂, ..., wₙ</code>, and bias <code>b</code>, the perceptron computes:
        </p>
        <p><code>z = w · x + b = ∑(wᵢxᵢ) + b</code></p>
        <p>Then applies an activation function:</p>
        <p><code>ŷ = 1 if z ≥ 0, else 0</code></p>
      </section>

      {/* Section 3 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Perceptron Learning Algorithm</h2>
        <p>The perceptron updates weights during training using misclassified examples.</p>
        <p><strong>Steps:</strong></p>
        <ul className="list-decimal ml-6 space-y-1">
          <li>Initialize weights and bias to 0 or small random values.</li>
          <li>For each training example <code>(x, y)</code>:
            <ul className="list-disc ml-6">
              <li>Predict output <code>ŷ</code></li>
              <li>Update weights and bias:</li>
              <li><code>w = w + α(y − ŷ)x</code></li>
              <li><code>b = b + α(y − ŷ)</code></li>
            </ul>
          </li>
          <li>Repeat until convergence or for fixed epochs.</li>
        </ul>
        <p>Where <code>α</code> is the learning rate.</p>
      </section>

      {/* Section 4 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Python Implementation (From Scratch)</h2>
        <div className="bg-gray-100 p-2 rounded text-xs overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

# Sample usage:
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, 0, 1)

model = Perceptron()
model.fit(X, y)
predictions = model.predict(X)`}
          </pre>
        </div>
      </section>

      {/* Section 5 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Visualization</h2>
        <p>
          Imagine this in 2D:
          <br />
          Blue dots = class 0<br />
          Red dots = class 1<br />
          Perceptron learns a straight line that separates them.
        </p>
      </section>

      {/* Section 6 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Key Properties</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Feature</th>
              <th className="p-2 border">Perceptron</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Type</td>
              <td className="p-2 border">Supervised Learning</td>
            </tr>
            <tr>
              <td className="p-2 border">Use case</td>
              <td className="p-2 border">Binary classification</td>
            </tr>
            <tr>
              <td className="p-2 border">Decision Boundary</td>
              <td className="p-2 border">Linear</td>
            </tr>
            <tr>
              <td className="p-2 border">Activation</td>
              <td className="p-2 border">Step function</td>
            </tr>
            <tr>
              <td className="p-2 border">Training</td>
              <td className="p-2 border">Online learning</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Section 7 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Limitations</h2>
        <ul className="list-disc ml-6">
          <li> Can only solve linearly separable problems</li>
          <li> Fails for XOR, spiral, or complex decision boundaries</li>
          <li> Step activation = non-differentiable (can't use gradient-based optimization)</li>
        </ul>
      </section>

      {/* Section 8 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Perceptron vs Logistic Regression vs Modern Neural Nets</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Model</th>
              <th className="p-2 border">Decision Boundary</th>
              <th className="p-2 border">Activation</th>
              <th className="p-2 border">Optimization</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Perceptron</td>
              <td className="p-2 border">Linear</td>
              <td className="p-2 border">Step</td>
              <td className="p-2 border">Update rule</td>
            </tr>
            <tr>
              <td className="p-2 border">Logistic Regression</td>
              <td className="p-2 border">Linear</td>
              <td className="p-2 border">Sigmoid</td>
              <td className="p-2 border">Gradient descent</td>
            </tr>
            <tr>
              <td className="p-2 border">Neural Net (MLP)</td>
              <td className="p-2 border">Non-linear</td>
              <td className="p-2 border">ReLU, Sigmoid</td>
              <td className="p-2 border">Backpropagation</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Section 9 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Applications (Classic)</h2>
        <ul className="list-disc ml-6">
          <li>Simple spam detection</li>
          <li>Binary image classification</li>
          <li>Basic signal processing</li>
        </ul>
        <p className="mt-2"> In modern ML, perceptrons are building blocks for deeper neural networks (MLPs).</p>
      </section>

      {/* Section 10 */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Common Interview Questions</h2>
        <ul className="list-disc ml-6">
          <li>What is the perceptron learning rule?</li>
          <li>Why does the perceptron fail on non-linear data?</li>
          <li>How does the perceptron differ from logistic regression?</li>
          <li>How do you handle multi-class classification using perceptrons?</li>
        </ul>
      </section>
    </div>
  );
}
