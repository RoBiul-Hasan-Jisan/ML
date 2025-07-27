//import React from "react";
import { useNavigate } from "react-router-dom";

export default function LinearRegression() {
  const navigate = useNavigate();

  const handleRowClick = (path: string) => {
    navigate(path);
  };

  return (
    <div className="max-w-full px-4 py-6 sm:max-w-5xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold mb-6 text-blue-700">Linear Regression</h1>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Definition</h2>
        <p>
          Linear Regression is a Supervised Machine Learning algorithm used for predicting continuous numerical values. It finds the best-fit straight line (called the regression line) that explains the relationship between:
        </p>
        <ul className="list-disc list-inside ml-6">
          <li>Independent variable(s) (input features, denoted as <code>X</code>)</li>
          <li>Dependent variable (target/output, denoted as <code>Y</code>)</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">How It Works</h2>
        <p>It assumes a linear relationship between input and output.</p>
        <p className="font-mono bg-gray-100 p-3 rounded whitespace-pre-wrap">Y = mX + c</p>
        <p>
          Where:<br />
          <code>Y</code> = predicted output (dependent variable)<br />
          <code>X</code> = input feature (independent variable)<br />
          <code>m</code> = slope (how much Y changes for every unit of X)<br />
          <code>c</code> = intercept (Y value when X = 0)
        </p>

        <p>For multiple linear regression with multiple inputs:</p>
        <p className="font-mono bg-gray-100 p-3 rounded whitespace-pre-wrap">
          Y = w₁X₁ + w₂X₂ + ⋯ + wₙXₙ + b
        </p>
        <p>
          Where:<br />
          <code>wᵢ</code> = weight coefficient for each input feature<br />
          <code>b</code> = bias (intercept)
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Goal of the Model</h2>
        <p>
          Minimize the error between predicted values (<code>Ŷ</code>) and actual values (<code>Y</code>) using a loss function, commonly:
        </p>
        <p className="font-mono bg-yellow-100 p-3 rounded whitespace-pre-wrap">
          Mean Squared Error (MSE) = (1/n) ∑(Yᵢ - Ŷᵢ)²
        </p>
      </section>

     <section className="space-y-3">
  <h2 className="text-2xl font-semibold">Types of Linear Regression</h2>
  <table className="w-full border-collapse border border-gray-300">
    <thead>
      <tr className="bg-gray-200">
        <th className="border border-gray-300 p-2 text-left">Type</th>
        <th className="border border-gray-300 p-2 text-left">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr
        className="cursor-pointer hover:bg-gray-100"
        onClick={() => handleRowClick('/ml/supervised/simple-linear-regression')}
      >
        <td className="border border-gray-300 p-2 text-blue-600">Simple Linear Regression</td>
        <td className="border border-gray-300 p-2">1 input variable (X), 1 output (Y)</td>
      </tr>
      <tr
        className="bg-gray-50 cursor-pointer hover:bg-gray-100"
        onClick={() => handleRowClick('/ml/supervised/multiple-linear-regression')}
      >
        <td className="border border-gray-300 p-2 text-blue-600">Multiple Linear Regression</td>
        <td className="border border-gray-300 p-2">Multiple inputs (X₁, X₂, ..., Xₙ), 1 output (Y)</td>
      </tr>
      <tr
        className="cursor-pointer hover:bg-gray-100"
        onClick={() => handleRowClick('/ml/supervised/polynomial-regression')}
      >
        <td className="border border-gray-300 p-2 text-blue-600">Polynomial Regression</td>
        <td className="border border-gray-300 p-2">X is raised to power (non-linear but linear in coefficients)</td>
      </tr>
    </tbody>
  </table>
</section>


      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Real-World Example: House Price Prediction</h2>
        <p>Training Data:</p>
        <table className="w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-200">
              <th className="border border-gray-300 p-2 text-left">Area (sq ft)</th>
              <th className="border border-gray-300 p-2 text-left">Price ($)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 p-2">1000</td>
              <td className="border border-gray-300 p-2">200,000</td>
            </tr>
            <tr className="bg-gray-50">
              <td className="border border-gray-300 p-2">1500</td>
              <td className="border border-gray-300 p-2">300,000</td>
            </tr>
            <tr>
              <td className="border border-gray-300 p-2">2000</td>
              <td className="border border-gray-300 p-2">400,000</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-3">Model learns the linear pattern: <code>Price = 200 × Area</code></p>
        <p>For a new house of 1800 sq ft, prediction is:</p>
        <p className="font-mono bg-gray-100 p-3 rounded">Price = 200 × 1800 = 360,000</p>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Key Points</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Type: Regression (continuous output)</li>
          <li>Linearity Assumption: Works best with linear data patterns</li>
          <li>Fast and interpretable</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Advantages</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Simple and efficient</li>
          <li>Easy to interpret</li>
          <li>Works well when features and output are linearly correlated</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Limitations</h2>
        <ul className="list-disc list-inside ml-6">
          <li>Not suitable for non-linear relationships</li>
          <li>Sensitive to outliers</li>
          <li>Assumes no multicollinearity (independent features should not be highly correlated)</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-2xl font-semibold">Interview Questions on Linear Regression</h2>

        <h3 className="text-xl font-semibold mt-4">Basic Questions</h3>
        <ul className="list-disc list-inside ml-6">
          <li>What is linear regression?</li>
          <li>What is the difference between simple and multiple linear regression?</li>
          <li>What are the assumptions of linear regression?</li>
          <li>What is the cost function used in linear regression?</li>
          <li>How do you evaluate a linear regression model?</li>
        </ul>

        <h3 className="text-xl font-semibold mt-4">Intermediate Questions</h3>
        <ul className="list-disc list-inside ml-6">
          <li>
            What is multicollinearity and how do you detect it?<br />
            <em>Answer:</em> High correlation between independent variables; detected using VIF (Variance Inflation Factor).
          </li>
          <li>
            What is the difference between R² and Adjusted R²?<br />
            <em>Answer:</em> R² explains variance; Adjusted R² adjusts for the number of predictors.
          </li>
          <li>
            How does linear regression handle missing values?<br />
            <em>Answer:</em> It doesn’t handle them natively — preprocessing (e.g., imputation) is required.
          </li>
        </ul>

        <h3 className="text-xl font-semibold mt-4">Advanced Questions</h3>
        <ul className="list-disc list-inside ml-6">
          <li>
            What is Regularization in Linear Regression?<br />
            <em>Answer:</em> Techniques like Ridge and Lasso to avoid overfitting.
          </li>
          <li>
            When would you use linear regression over a decision tree?<br />
            <em>Answer:</em> When the relationship is linear and interpretability is needed.
          </li>
        </ul>
      </section>

      <section className="max-w-3xl mx-auto p-6 bg-white rounded-md shadow-sm border border-gray-200 space-y-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          When <span className="text-red-600 underline">NOT</span> to Use Linear Regression
        </h2>
        <p>Avoid Linear Regression if:</p>
        <ul className="list-disc list-inside ml-6 text-gray-700 space-y-1">
          <li>The relationship is non-linear</li>
          <li>There is high multicollinearity</li>
          <li>You have many outliers</li>
          <li>The error terms are not normally distributed</li>
        </ul>
      </section>
    </div>
  );
}
