//import React from "react";

export default function HowMLWorks() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
      <h1 className="text-2xl sm:text-3xl font-bold mb-6 text-center sm:text-left">
        How Do Machine Learning Algorithms Work?
      </h1>

      <p className="text-base sm:text-lg">
        Machine Learning algorithms work by learning patterns from data and using those patterns to make decisions or predictions. This process involves several key steps:
      </p>

      <ol className="list-decimal list-inside space-y-6 text-base sm:text-lg">
        {/* Step 1 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">📥</span>
            <span>
              <strong>Input Data:</strong> Raw data is collected from various sources (databases, sensors, web, etc.) and cleaned, formatted, and split into:
            </span>
          </p>
          <ul className="list-disc list-inside ml-6 mt-2">
            <li>Training set (to learn)</li>
            <li>Testing set (to evaluate)</li>
          </ul>
          <p className="mt-2 italic text-gray-600">
            Example: Customer data, images, text, sensor logs.
          </p>
        </li>

        {/* Step 2 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">🧠</span>
            <span>
              <strong>Feature Extraction:</strong> Important characteristics (features) are selected or created to help the model learn better.
            </span>
          </p>
          <p className="mt-2 italic text-gray-600">
            Example: Email spam detection might use features like the number of links or specific keywords.
          </p>
        </li>

        {/* Step 3 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">🔍</span>
            <span>
              <strong>Model Selection:</strong> Choose an algorithm based on the problem type:
            </span>
          </p>
          <ul className="list-disc list-inside ml-6 mt-2">
            <li>Regression → Linear Regression</li>
            <li>Classification → Decision Tree, SVM</li>
            <li>Complex tasks → Neural Networks, Random Forest</li>
          </ul>
        </li>

        {/* Step 4 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">🏋️</span>
            <span>
              <strong>Training the Model:</strong> Feed the training data into the model. It learns patterns by minimizing errors and adjusting internal parameters.
            </span>
          </p>
          <p className="mt-2 italic text-gray-600">
            Example: In image classification, the model learns features distinguishing cats from dogs.
          </p>
        </li>

        {/* Step 5 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">📊</span>
            <span>
              <strong>Evaluation:</strong> Test the model using the testing dataset and evaluate its performance using appropriate metrics:
            </span>
          </p>
          <ul className="list-disc list-inside ml-6 mt-2">
            <li>Accuracy</li>
            <li>Precision</li>
            <li>Recall</li>
            <li>F1-Score</li>
            <li>Mean Squared Error (MSE) — for regression</li>
          </ul>
        </li>

        {/* Step 6 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">🤖</span>
            <span>
              <strong>Prediction:</strong> The trained model can now make predictions on new, unseen data.
            </span>
          </p>
          <p className="mt-2 italic text-gray-600">
            Example: Predicting house prices or recommending products.
          </p>
        </li>

        {/* Step 7 */}
        <li>
          <p className="flex items-start">
            <span className="mr-2">🔁</span>
            <span>
              <strong>Improvement and Tuning:</strong> Improve the model’s performance by:
            </span>
          </p>
          <ul className="list-disc list-inside ml-6 mt-2">
            <li>Adding more training data</li>
            <li>Tuning hyperparameters (e.g., learning rate, tree depth)</li>
            <li>Trying different algorithms</li>
            <li>Using cross-validation to avoid overfitting</li>
          </ul>
        </li>
      </ol>
    </div>
  );
}
