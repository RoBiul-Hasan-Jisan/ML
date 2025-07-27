//import React from "react";

type Algorithm = {
  title: string;
  what: string;
  when: string;
  why: string;
  use: string;
};

export default function ClassificationAlgorithmsGuide() {



  const algorithms: Algorithm[] = [
    {
      title: " 1. Logistic Regression",
      what: "Linear model for binary/multi-class classification.",
      when: "Classes are linearly separable.",
      why: "Simple, interpretable, good baseline.",
      use: "Email spam detection, loan approval.",
    },
    {
      title: " 2. K-Nearest Neighbors (KNN)",
      what: "Majority vote of k closest neighbors.",
      when: "Low-dimensional, small dataset.",
      why: "Simple, effective for local patterns.",
      use: "Recommender systems, image recognition.",
    },
    {
      title: " 3. Support Vector Machine (SVM)",
      what: "Finds optimal hyperplane to separate classes.",
      when: "High-dimensional data, complex boundaries.",
      why: "Accurate, especially with kernels.",
      use: "Text classification, bioinformatics.",
    },
    {
      title: " 4. Decision Trees",
      what: "Splits data based on feature thresholds.",
      when: "Interpretability is key.",
      why: "Visual, handles both types of data.",
      use: "Customer churn prediction.",
    },
    {
      title: " 5. Random Forest Classifier",
      what: "Ensemble of decision trees (majority vote).",
      when: "High accuracy needed with low variance.",
      why: "General-purpose, robust to overfitting.",
      use: "Fraud detection, diagnostics.",
    },
    {
      title: " 6. Gradient Boosting Classifier",
      what: "Sequential trees to fix previous errors.",
      when: "Best accuracy is needed.",
      why: "Excellent on non-linear data.",
      use: "Customer segmentation, click prediction.",
    },
    {
      title: " 7. XGBoost Classifier",
      what: "Boosted trees with regularization.",
      when: "Competitions, large structured data.",
      why: "Fast, accurate, scalable.",
      use: "Credit scoring, ranking problems.",
    },
    {
      title: " 8. LightGBM Classifier",
      what: "Gradient boosting with leaf-wise growth.",
      when: "Large, high-dimensional datasets.",
      why: "Super-fast training, high accuracy.",
      use: "Online ads, classification at scale.",
    },
    {
      title: " 9. CatBoost Classifier",
      what: "Boosting model for categorical features.",
      when: "Many non-numeric columns.",
      why: "Automatic handling, less preprocessing.",
      use: "Marketing, product recommendation.",
    },
    {
      title: " 10. Naive Bayes",
      what: "Probabilistic model using Bayes’ rule.",
      when: "Text or high-dimensional sparse data.",
      why: "Fast, works well with independence assumption.",
      use: "Spam filters, sentiment analysis.",
    },
    {
      title: " 11. Stochastic Gradient Descent (SGD) Classifier",
      what: "Uses SGD to optimize a loss function.",
      when: "Massive-scale learning problems.",
      why: "Fast and memory-efficient.",
      use: "Text classification, recommender systems.",
    },
    {
      title: " 12. Perceptron",
      what: "Basic binary classifier (linear).",
      when: "Linearly separable data.",
      why: "Simple, foundational to neural nets.",
      use: "Educational use, basic classification.",
    },
    {
      title: " 13. Quadratic Discriminant Analysis (QDA)",
      what: "Assumes Gaussian distributions with different covariances.",
      when: "Complex class boundaries.",
      why: "Captures quadratic decision surfaces.",
      use: "Medical classification, sensor data.",
    },
    {
      title: " 14. Linear Discriminant Analysis (LDA)",
      what: "Projects features to maximize class separation.",
      when: "Normal distributions with equal covariance.",
      why: "Works well on small data, dimensionality reduction.",
      use: "Face recognition, pattern recognition.",
    },
  ];

  const summary: [string, string, string][] = [
    ["Logistic Regression", "Linearly separable problems", "Interpretable, fast"],
    ["KNN", "Local patterns, small datasets", "Simple, no training"],
    ["SVM", "High-dimensional, complex data", "Robust, kernel-powered"],
    ["Decision Tree", "Visual explanation, mixed features", "Easy to understand"],
    ["Random Forest", "General-purpose, tabular data", "Accurate and robust"],
    ["Gradient Boosting", "Accuracy-demanding tasks", "Top performance"],
    ["XGBoost", "Structured large data", "Fast, regularized"],
    ["LightGBM", "Huge datasets, fast training", "Speed and memory efficiency"],
    ["CatBoost", "Categorical feature-rich datasets", "Handles non-numeric features easily"],
    ["Naive Bayes", "Text/NLP", "Fast, handles high dimensions"],
    ["SGD Classifier", "Online or huge data", "Fast and scalable"],
    ["Perceptron", "Simple binary problems", "Foundational model"],
    ["QDA", "Gaussian non-linear separation", "Captures curved boundaries"],
    ["LDA", "Small datasets, dimensionality reduction", "Works well with linear separation"],
  ];

  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">
        Classification Algorithms – Predict Discrete Labels
      </h1>

      {/* Detailed Explanation */}
      {algorithms.map((algo, idx) => (
        <section key={idx}>
          <h2 className="text-sm sm:text-base font-semibold text-blue-600 mb-1">
            {algo.title}
          </h2>
          <p><strong>What:</strong> {algo.what}</p>
          <p><strong>When:</strong> {algo.when}</p>
          <p><strong>Why:</strong> {algo.why}</p>
          <p><strong>Use Case:</strong> {algo.use}</p>
        </section>
      ))}

      {/* Summary Table */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2 mt-6 text-blue-700">
          Classification Summary Table
        </h2>
        <div className="overflow-x-auto text-xs sm:text-sm">
          <table className="w-full border border-gray-300">
            <thead className="bg-gray-200">
              <tr>
                <th className="p-2 border">Algorithm</th>
                <th className="p-2 border">Best For</th>
                <th className="p-2 border">Key Advantage</th>
              </tr>
            </thead>
            <tbody>
              {summary.map(([name, bestFor, advantage], i) => (
                <tr key={i}>
                  <td className="p-2 border">{name}</td>
                  <td className="p-2 border">{bestFor}</td>
                  <td className="p-2 border">{advantage}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
