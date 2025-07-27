
//import React from "react";

type Algorithm = {
  title: string;
  what: string;
  when: string;
  why: string;
  use: string;
};

export default function RegressionAlgorithmsGuide() {
  const algorithms: Algorithm[] = [
    {
      title: "1. Linear Regression",
      what: "Fits a straight line to the data.",
      when: "Simple, linear relationships between variables.",
      why: "Fast, interpretable, ideal as a baseline model.",
      use: "Predicting house prices, salary based on experience.",
    },
    {
      title: "2. Ridge Regression (L2 Regularization)",
      what: "Adds penalty for large coefficients (squares them).",
      when: "Multicollinearity exists; lots of correlated features.",
      why: "Reduces model complexity, prevents overfitting.",
      use: "Economic or financial models with many predictors.",
    },
    {
      title: "3. Lasso Regression (L1 Regularization)",
      what: "Adds penalty to absolute coefficient values.",
      when: "Need automatic feature selection.",
      why: "Shrinks some weights to zero → model becomes simpler.",
      use: "Medical or biological data with many irrelevant features.",
    },
    {
      title: "4. Polynomial Regression",
      what: "Adds polynomial terms (e.g., x², x³) to linear model.",
      when: "Data is non-linear but still smooth.",
      why: "Models curves with relatively low effort.",
      use: "Modeling learning curves, population growth.",
    },
    {
      title: "5. ElasticNet Regression",
      what: "Hybrid of Lasso and Ridge.",
      when: "When you need both regularization and feature selection.",
      why: "Balances L1 and L2 benefits; better for complex datasets.",
      use: "Genomics data, economic forecasting.",
    },
    {
      title: "6. Bayesian Regression",
      what: "Uses probability distributions for coefficients.",
      when: "When estimating confidence in predictions is crucial.",
      why: "Provides uncertainty estimates.",
      use: "Stock prediction, scientific research with small datasets.",
    },
    {
      title: "7. Support Vector Regression (SVR)",
      what: "SVR tries to fit the best line within a margin of tolerance.",
      when: "Non-linear and high-dimensional data.",
      why: "Powerful and flexible using kernels.",
      use: "Weather forecasting, demand prediction.",
    },
    {
      title: "8. Decision Tree Regression",
      what: "Splits data into regions based on feature thresholds.",
      when: "Non-linear relationships, interactions between variables.",
      why: "Non-parametric, interpretable.",
      use: "Energy consumption prediction, insurance costs.",
    },
    {
      title: "9. Random Forest Regression",
      what: "Ensemble of decision trees (averages predictions).",
      when: "You want a more accurate and stable model.",
      why: "Reduces overfitting; handles non-linearities well.",
      use: "Real estate value, sales forecasting.",
    },
    {
      title: "10. Gradient Boosting Regression",
      what: "Builds trees sequentially to correct previous errors.",
      when: "You want top accuracy and can afford training time.",
      why: "Powerful with strong predictive performance.",
      use: "Customer lifetime value, risk modeling.",
    },
    {
      title: "11. XGBoost Regression",
      what: "Optimized gradient boosting (regularization, parallel).",
      when: "Large, sparse data; competitions.",
      why: "Fast, robust to overfitting.",
      use: "Kaggle competitions, sales predictions.",
    },
    {
      title: "12. LightGBM Regression",
      what: "Fast tree-based boosting using leaf-wise splits.",
      when: "Very large datasets, many features.",
      why: "Faster than XGBoost, good with high-dimensional data.",
      use: "Recommender systems, ad click rate prediction.",
    },
    {
      title: "13. CatBoost Regression",
      what: "Boosting model that handles categorical features natively.",
      when: "Data has many categorical columns.",
      why: "Reduces preprocessing; accurate and fast.",
      use: "Ecommerce pricing, credit risk modeling.",
    },
    {
      title: "14. KNN Regression",
      what: "Predicts by averaging k-nearest neighbors.",
      when: "Local trends in small datasets.",
      why: "Simple, non-parametric.",
      use: "House price estimation in neighborhoods.",
    },
  ];

  const summary: [string, string, string][] = [
    ["Linear Regression", "Simple trends, baseline models", "Fast, interpretable"],
    ["Ridge/Lasso", "High-dimensional, correlated data", "Regularization, feature selection"],
    ["Polynomial", "Smooth non-linear trends", "Captures curves"],
    ["ElasticNet", "Complex, high-dimensional data", "Balanced, flexible"],
    ["Bayesian", "Small data, uncertainty estimation", "Probabilistic predictions"],
    ["SVR", "High-dimension, non-linear", "Uses kernel trick"],
    ["Decision Tree", "Interpretability, non-linearity", "Simple to explain"],
    ["Random Forest", "General-purpose, high accuracy", "Reduces overfitting"],
    ["Gradient Boosting", "Accuracy-focused applications", "Powerful, but slower"],
    ["XGBoost", "Large datasets, competitions", "Fast, accurate"],
    ["LightGBM", "High-speed large data", "Super fast, memory-efficient"],
    ["CatBoost", "Categorical features", "No encoding needed"],
    ["KNN", "Local trends, small datasets", "Easy to understand, no training"],
  ];

  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words overflow-x-hidden">
      <h1 className="text-xl sm:text-3xl font-bold text-blue-700">
        Regression Algorithms – Predict Continuous Values
      </h1>

      {/* Detailed Explanation */}
      {algorithms.map((algo, idx) => (
        <section key={idx}>
          <h2 className="text-sm sm:text-base font-semibold text-blue-600 mb-1">{algo.title}</h2>
          <p><strong>What:</strong> {algo.what}</p>
          <p><strong>When:</strong> {algo.when}</p>
          <p><strong>Why:</strong> {algo.why}</p>
          <p><strong>Use Case:</strong> {algo.use}</p>
        </section>
      ))}

      {/* Summary Table */}
      <section>
        <h2 className="text-base sm:text-lg font-semibold mb-2 mt-6 text-blue-700">
          Regression Summary Table
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
