// import React from "react";

export default function MLLifecycle() {
  return (
    <div className="max-w-full sm:max-w-3xl mx-auto px-4 sm:px-6 pt-4 pb-10 space-y-6">
      <h1 className="text-xl sm:text-3xl md:text-4xl font-bold text-center text-blue-700">
        Machine Learning Lifecycle
      </h1>

      <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
        The Machine Learning Lifecycle is a step-by-step process that guides how
        an ML project is planned, developed, deployed, and maintained. It ensures
        that the model is efficient, reliable, and sustainable in a real-world
        environment.
      </p>

      <ol className="list-decimal list-inside space-y-8 text-sm sm:text-base text-gray-800">
        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Problem Definition:</strong> Clearly define the goal of the ML project.
              <br />
              Determine:
              <ul className="list-disc list-inside ml-6 mt-2">
                <li>What problem are we solving?</li>
                <li>Is it classification, regression, clustering, etc.?</li>
                <li>What will be the input and expected output?</li>
              </ul>
              <div className="mt-1 text-gray-600">
                Example: Predict customer churn, detect fraud, classify emails.
              </div>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Data Collection:</strong> Gather relevant and sufficient data from various sources (databases, APIs, sensors, user logs, etc.).
              <div className="mt-1 text-gray-600">
                Quality and quantity of data directly affect the model's success.
              </div>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Data Cleaning & Preparation:</strong> Handle missing values, duplicates, outliers, and inconsistent formats.
              <br />
              Normalize, scale, or encode data as needed. Split data into training, validation, and testing sets.
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Feature Engineering:</strong> Select or create relevant features that help the model understand the data better.
              <ul className="list-disc list-inside ml-6 mt-2">
                <li>Feature selection (removing irrelevant features)</li>
                <li>Feature transformation (e.g., log scaling)</li>
                <li>Encoding categorical data</li>
              </ul>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Model Selection:</strong> Choose the right algorithm based on:
              <ul className="list-disc list-inside ml-6 mt-2">
                <li>Type of problem</li>
                <li>Data size and quality</li>
                <li>Accuracy vs. complexity trade-off</li>
              </ul>
              <div className="mt-2">
                Examples:
                <ul className="list-disc list-inside ml-6 mt-2">
                  <li>Decision Trees</li>
                  <li>Random Forest</li>
                  <li>Logistic Regression</li>
                  <li>Support Vector Machine</li>
                  <li>Neural Networks</li>
                </ul>
              </div>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Model Training:</strong> Feed the training data into the chosen model. The model learns by adjusting internal parameters to reduce error/loss.
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Model Evaluation:</strong> Evaluate the trained model using the test data.
              <ul className="list-disc list-inside ml-6 mt-2">
                <li>Accuracy, Precision, Recall, F1-score (for classification)</li>
                <li>RMSE, MAE, R² (for regression)</li>
              </ul>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Hyperparameter Tuning:</strong> Improve model performance by adjusting hyperparameters (not learned by the model).
              <ul className="list-disc list-inside ml-6 mt-2">
                <li>Grid Search</li>
                <li>Random Search</li>
                <li>Bayesian Optimization</li>
              </ul>
              <div className="mt-1 text-gray-600">
                Example: Learning rate, number of trees, depth of tree, etc.
              </div>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Deployment:</strong> Deploy the model into a production environment (app, website, server).
              <div className="mt-1 text-gray-600">
                Users or systems can now use the model to make live predictions.
              </div>
            </span>
          </p>
        </li>

        <li>
          <p className="flex items-start">
            <span className="mr-2"></span>
            <span>
              <strong>Monitoring & Maintenance:</strong> Continuously monitor the model’s performance.
              <ul className="list-disc list-inside ml-6 mt-2">
                <li>Model drift (performance degradation)</li>
                <li>Data changes</li>
                <li>Update or retrain the model as needed</li>
              </ul>
            </span>
          </p>
        </li>
      </ol>
    </div>
  );
}
