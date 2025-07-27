// NaiveBayesGuide.tsx

export default function NaiveBayesGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-600">Naive Bayes </h1>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> What is Naive Bayes?</h2>
        <p>
          Naive Bayes is a probabilistic classifier based on Bayes' Theorem, assuming independence between features (naive assumption).
        </p>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Bayes' Theorem Formula:</h2>
        <p><code>P(C∣X) = [P(X∣C) ⋅ P(C)] / P(X)</code></p>
        <ul className="list-disc ml-6">
          <li><code>P(C∣X)</code>: Posterior – probability of class C given features X</li>
          <li><code>P(X∣C)</code>: Likelihood – probability of X given class C</li>
          <li><code>P(C)</code>: Prior – probability of class C</li>
          <li><code>P(X)</code>: Marginal likelihood – probability of the features</li>
        </ul>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Why “Naive”?</h2>
        <p>
          Because it assumes all features are independent:
          <br />
          <code>P(X₁,X₂,...,Xₙ∣C) = P(X₁∣C)⋅P(X₂∣C)⋅...⋅P(Xₙ∣C)</code>
        </p>
        <p>This is rarely true in real-world data, but it still works surprisingly well!</p>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Types of Naive Bayes</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Type</th>
              <th className="p-2 border">Use Case</th>
              <th className="p-2 border">Feature Distribution</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">GaussianNB</td>
              <td className="p-2 border">Continuous data</td>
              <td className="p-2 border">Normal distribution</td>
            </tr>
            <tr>
              <td className="p-2 border">MultinomialNB</td>
              <td className="p-2 border">Text classification</td>
              <td className="p-2 border">Counts (e.g., word frequencies)</td>
            </tr>
            <tr>
              <td className="p-2 border">BernoulliNB</td>
              <td className="p-2 border">Binary features</td>
              <td className="p-2 border">0/1 values</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Steps of Naive Bayes Algorithm</h2>
        <ul className="list-decimal ml-6 space-y-1">
          <li>Calculate Prior: <code>P(Cᵢ) = (# samples in class Cᵢ) / (Total samples)</code></li>
          <li>Calculate Likelihood for each feature:
            <br />For Gaussian:
            <br /><code>P(x∣C) = (1 / √(2πσ²)) ⋅ e^(-(x−μ)² / 2σ²)</code>
          </li>
          <li>Multiply probabilities using independence:
            <br /><code>P(C∣X) ∝ P(C)⋅∏P(xᵢ∣C)</code>
          </li>
          <li>Choose class with max posterior</li>
        </ul>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Example (Simple)</h2>
        <p><strong>Data:</strong></p>
        <ul className="list-disc ml-6">
          <li>Sunny, Hot → No</li>
          <li>Sunny, Cool → Yes</li>
          <li>Rainy, Cool → Yes</li>
        </ul>
        <p>To predict "Sunny, Cool":</p>
        <p><code>P(Yes∣Sunny,Cool) ∝ P(Yes) ⋅ P(Sunny∣Yes) ⋅ P(Cool∣Yes)</code></p>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Pros & Cons</h2>
        <p><strong> Pros:</strong></p>
        <ul className="list-disc ml-6">
          <li>Simple and fast</li>
          <li>Works well with high-dimensional data</li>
          <li>Performs well with small data</li>
        </ul>
        <p><strong> Cons:</strong></p>
        <ul className="list-disc ml-6">
          <li>Assumes feature independence (often unrealistic)</li>
          <li>Not great with correlated/numeric features (unless GaussianNB)</li>
        </ul>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Applications</h2>
        <ul className="list-disc ml-6">
          <li>Spam filtering</li>
          <li>Sentiment analysis</li>
          <li>Document classification</li>
          <li>Medical diagnosis</li>
        </ul>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2">Scikit-Learn Example</h2>
        <div className="bg-gray-100 p-2 rounded text-xs overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)`}
          </pre>
        </div>
        <div className="bg-gray-100 p-2 mt-2 rounded text-xs overflow-auto">
          <pre className="whitespace-pre-wrap break-words font-mono leading-tight">
{`from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)`}
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2"> Summary Table</h2>
        <table className="w-full text-left text-xs sm:text-sm border border-gray-300">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Feature</th>
              <th className="p-2 border">Naive Bayes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Type</td>
              <td className="p-2 border">Probabilistic</td>
            </tr>
            <tr>
              <td className="p-2 border">Assumption</td>
              <td className="p-2 border">Feature independence</td>
            </tr>
            <tr>
              <td className="p-2 border">Output</td>
              <td className="p-2 border">Class label</td>
            </tr>
            <tr>
              <td className="p-2 border">Performance</td>
              <td className="p-2 border">Good for large text data</td>
            </tr>
            <tr>
              <td className="p-2 border">Speed</td>
              <td className="p-2 border">Very fast</td>
            </tr>
            <tr>
              <td className="p-2 border">Interpretability</td>
              <td className="p-2 border">Easy to understand</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
