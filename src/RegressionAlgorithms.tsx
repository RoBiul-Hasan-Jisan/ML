//import React from "react";
//dfdf
type Algorithm = {
  id: number;
  name: string;
  useWhen: string[];
  notes?: string[];
  warnings?: string[];
};

const algorithms: Algorithm[] = [
  {
    id: 1,
    name: "Linear Regression",
    useWhen: [
      "Data has a linear relationship between input and output.",
      "Fast and interpretable model is needed.",
      "Dataset is not too large or noisy.",
    ],
    notes: ["Example: Predicting house prices based on size, number of rooms, etc., when relationship is linear."],
  },
  {
    id: 2,
    name: "Ridge Regression (L2 Regularization)",
    useWhen: [
      "Linear regression is overfitting.",
      "You want to penalize large coefficients to improve generalization.",
    ],
    notes: ["Use if features are correlated — Ridge helps reduce multicollinearity."],
  },
  {
    id: 3,
    name: "Lasso Regression (L1 Regularization)",
    useWhen: [
      "You want feature selection (Lasso sets some coefficients to 0).",
      "The dataset has many irrelevant features.",
    ],
    notes: ["Sparse models, great when only a few features matter."],
  },
  {
    id: 4,
    name: "ElasticNet Regression (L1 + L2 Regularization)",
    useWhen: [
      "You want a balance between Ridge and Lasso.",
      "Dataset has many features and you’re unsure which are useful.",
    ],
  },
  {
    id: 5,
    name: "Decision Tree Regression",
    useWhen: [
      "Relationships are non-linear.",
      "You want a fast, interpretable, rule-based model.",
      "Outliers or missing values are present.",
    ],
    warnings: ["Can overfit easily without pruning."],
  },
  {
    id: 6,
    name: "Random Forest Regression",
    useWhen: [
      "Non-linear relationships.",
      "Higher accuracy and robustness than a single decision tree.",
      "Dataset is noisy or has many features.",
    ],
    notes: ["Ensemble of trees → reduces overfitting."],
  },
  {
    id: 7,
    name: "Gradient Boosting Regression",
    useWhen: [
      "You need high prediction accuracy.",
      "The relationship is complex and non-linear.",
      "You're okay with longer training time.",
    ],
    notes: ["Often wins in competitions. Can be sensitive to overfitting."],
  },
  {
    id: 8,
    name: "Support Vector Regression (SVR)",
    useWhen: [
      "You want to control margin of error.",
      "The dataset is small to medium-sized.",
      "Data is not too noisy.",
    ],
    warnings: ["Slower on large datasets. Needs scaling."],
  },
  {
    id: 9,
    name: "K-Nearest Neighbors Regression (KNN)",
    useWhen: [
      "You want a simple, non-parametric model.",
      "You suspect local patterns in data are important.",
    ],
    warnings: ["Sensitive to scaling and noisy data. Not great for high dimensions."],
  },
  {
    id: 10,
    name: "XGBoost Regression",
    useWhen: [
      "You want the best performance.",
      "You're working on a real-world production project or competition.",
      "You're okay with tuning parameters.",
    ],
    notes: ["Powerful, fast, regularized gradient boosting."],
  },
];

const summaryTableData = [
  {
    algorithm: "Linear",
    handlesNonLinearity: false,
    fast: true,
    avoidsOverfitting: false,
    featureSelection: false,
    worksOnLargeData: true,
  },
  {
    algorithm: "Ridge",
    handlesNonLinearity: false,
    fast: true,
    avoidsOverfitting: true,
    featureSelection: false,
    worksOnLargeData: true,
  },
  {
    algorithm: "Lasso",
    handlesNonLinearity: false,
    fast: true,
    avoidsOverfitting: true,
    featureSelection: true,
    worksOnLargeData: true,
  },
  {
    algorithm: "ElasticNet",
    handlesNonLinearity: false,
    fast: true,
    avoidsOverfitting: true,
    featureSelection: true,
    worksOnLargeData: true,
  },
  {
    algorithm: "Decision Tree",
    handlesNonLinearity: true,
    fast: true,
    avoidsOverfitting: false,
    featureSelection: false,
    worksOnLargeData: true,
  },
  {
    algorithm: "Random Forest",
    handlesNonLinearity: true,
    fast: "medium",
    avoidsOverfitting: true,
    featureSelection: false,
    worksOnLargeData: true,
  },
  {
    algorithm: "Gradient Boosting",
    handlesNonLinearity: true,
    fast: "slower",
    avoidsOverfitting: true,
    featureSelection: false,
    worksOnLargeData: true,
  },
  {
    algorithm: "SVR",
    handlesNonLinearity: true,
    fast: false,
    avoidsOverfitting: true,
    featureSelection: false,
    worksOnLargeData: false,
  },
  {
    algorithm: "KNN",
    handlesNonLinearity: true,
    fast: false,
    avoidsOverfitting: false,
    featureSelection: false,
    worksOnLargeData: false,
  },
  {
    algorithm: "XGBoost",
    handlesNonLinearity: true,
    fast: true,
    avoidsOverfitting: "very good",
    featureSelection: false,
    worksOnLargeData: "very good",
  },
];

const boolToEmoji = (value: boolean | string) => {
  if (value === true) return "✅";
  if (value === false) return "❌";
  if (typeof value === "string") {
    // Interpret text hints for speed and quality
    if (value.toLowerCase().includes("very good")) return "✅✅";
    if (value.toLowerCase().includes("slower")) return "⏱⏱";
    if (value.toLowerCase().includes("medium")) return "⏱";
  }
  return "";
};

export default function RegressionAlgorithms() {
  return (
    <div className="max-w-4xl mx-auto p-6 font-sans">
      <h1 className="text-3xl font-bold mb-6 text-center">Regression Algorithms Overview</h1>

      <section className="space-y-8">
        {algorithms.map(({ id, name, useWhen, notes, warnings }) => (
          <article key={id} className="border rounded-md p-4 shadow-sm">
            <h2 className="text-xl font-semibold mb-2">
              {id}. {name}
            </h2>
            <h3 className="font-semibold mb-1"> Use When:</h3>
            <ul className="list-disc list-inside mb-2">
              {useWhen.map((point, i) => (
                <li key={i}>{point}</li>
              ))}
            </ul>
            {notes && (
              <>
                <h3 className="font-semibold mb-1"> Notes:</h3>
                <ul className="list-disc list-inside mb-2">
                  {notes.map((note, i) => (
                    <li key={i}>{note}</li>
                  ))}
                </ul>
              </>
            )}
            {warnings && (
              <>
                <h3 className="font-semibold mb-1 text-red-600"> Warnings:</h3>
                <ul className="list-disc list-inside mb-2 text-red-600">
                  {warnings.map((warn, i) => (
                    <li key={i}>{warn}</li>
                  ))}
                </ul>
              </>
            )}
          </article>
        ))}
      </section>

      <section className="mt-12">
        <h2 className="text-2xl font-bold mb-4 text-center">Summary Table</h2>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300 text-center">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-300 px-3 py-2">Algorithm</th>
                <th className="border border-gray-300 px-3 py-2">Handles Non-Linearity</th>
                <th className="border border-gray-300 px-3 py-2">Fast</th>
                <th className="border border-gray-300 px-3 py-2">Avoids Overfitting</th>
                <th className="border border-gray-300 px-3 py-2">Feature Selection</th>
                <th className="border border-gray-300 px-3 py-2">Works on Large Data</th>
              </tr>
            </thead>
            <tbody>
              {summaryTableData.map((row) => (
                <tr key={row.algorithm} className="hover:bg-gray-50">
                  <td className="border border-gray-300 px-3 py-2 font-semibold">{row.algorithm}</td>
                  <td className="border border-gray-300 px-3 py-2">{boolToEmoji(row.handlesNonLinearity)}</td>
                  <td className="border border-gray-300 px-3 py-2">{boolToEmoji(row.fast)}</td>
                  <td className="border border-gray-300 px-3 py-2">{boolToEmoji(row.avoidsOverfitting)}</td>
                  <td className="border border-gray-300 px-3 py-2">{boolToEmoji(row.featureSelection)}</td>
                  <td className="border border-gray-300 px-3 py-2">{boolToEmoji(row.worksOnLargeData)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
