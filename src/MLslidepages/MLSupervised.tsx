import { useNavigate } from "react-router-dom";

export default function MLSupervised() {
  const navigate = useNavigate();

  const goTo = (path: string) => navigate(path);

  return (
     <div className="max-w-4xl mx-auto p-6 pt-16 space-y-6">

      <h1 className="text-2xl font-bold mb-4 text-blue-700">Supervised Machine Learning</h1>

      <p>
        <strong>Supervised Machine Learning</strong> is a type of machine learning where the algorithm is trained on a <strong>labeled dataset</strong>. The goal is for the model to learn the mapping from input features (<code>X</code>) to the output/target (<code>Y</code>) and make predictions on unseen data.
      </p>

      <div className="bg-gray-100 p-4 rounded-lg shadow-sm">
        <h2 className="text-xl font-semibold mb-2">🔍 Example: House Price Prediction</h2>
        <p><strong>Inputs (Features):</strong> Size, Location, Number of Rooms</p>
        <p><strong>Output (Label):</strong> Price</p>
      </div>

      <div className="space-y-2">
        <h2 className="text-xl font-semibold">⚙️ How It Works</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Data Collection:</strong> Gather labeled input-output pairs.</li>
          <li><strong>Training:</strong> Feed the data into a model to learn patterns.</li>
          <li><strong>Model Learning:</strong> Adjust parameters to reduce prediction error.</li>
          <li><strong>Evaluation:</strong> Test model performance on unseen data.</li>
          <li><strong>Prediction:</strong> Use trained model for real-world predictions.</li>
        </ul>
      </div>

      <div className="bg-yellow-100 p-4 rounded-md">
        <p className="font-mono text-sm"><strong>📌 Conceptual Formula:</strong> Y = f(X) + error</p>
        <p className="text-sm">The model approximates a function <code>f</code> that maps inputs <code>X</code> to outputs <code>Y</code>.</p>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold">📚 Types of Supervised Learning</h2>

        {/* 1. Regression */}
        <div>
          <h3 className="text-lg font-semibold">1. Regression</h3>
          <p>Predicts continuous values (e.g., house prices, temperatures).</p>
          <p><strong>Algorithms:</strong></p>
          <ul className="list-disc list-inside ml-4">
            <li onClick={() => goTo("/ml/supervised/linear-regression")} className="text-blue-600 cursor-pointer hover:underline">🔷 Linear Regression</li>
            <li onClick={() => goTo("/ml/supervised/ridge")} className="text-blue-600 cursor-pointer hover:underline">🔷 Ridge Regression</li>
            <li onClick={() => goTo("/ml/supervised/lasso")} className="text-blue-600 cursor-pointer hover:underline">🔷 Lasso Regression</li>
            <li onClick={() => goTo("/ml/supervised/elastic-net")} className="text-blue-600 cursor-pointer hover:underline">🔷 Elastic Net</li>
            {/* <li onClick={() => goTo("/ml/supervised/polynomial")} className="text-blue-600 cursor-pointer hover:underline">🔷 Polynomial Regression</li> */}
            <li onClick={() => goTo("/ml/supervised/decision-tree")} className="text-blue-600 cursor-pointer hover:underline">🔷 Decision Tree Regression</li>
            <li onClick={() => goTo("/ml/supervised/random-forest")} className="text-blue-600 cursor-pointer hover:underline">🔷 Random Forest Regression</li>
            <li onClick={() => goTo("/ml/supervised/svr")} className="text-blue-600 cursor-pointer hover:underline">🔷 Support Vector Regression (SVR)</li>
          </ul>
        </div>

        {/* 2. Classification */}
        <div>
          <h3 className="text-lg font-semibold">2. Classification</h3>
          <p>Predicts categorical values (e.g., spam vs. not spam, disease positive/negative).</p>
          <p><strong>Algorithms:</strong></p>
          <ul className="list-disc list-inside ml-4">
            <li onClick={() => goTo("/ml/supervised/logistic-regression")} className="text-blue-600 cursor-pointer hover:underline">🔷 Logistic Regression</li>
            <li onClick={() => goTo("/ml/supervised/lda")} className="text-blue-600 cursor-pointer hover:underline">🔷 Linear Discriminant Analysis (LDA)</li>
            <li onClick={() => goTo("/ml/supervised/qda")} className="text-blue-600 cursor-pointer hover:underline">🔷 Quadratic Discriminant Analysis (QDA)</li>
            <li onClick={() => goTo("/ml/supervised/perceptron")} className="text-blue-600 cursor-pointer hover:underline">🔷 Perceptron</li>
            <li onClick={() => goTo("/ml/supervised/knn")} className="text-blue-600 cursor-pointer hover:underline">🔷 k-Nearest Neighbors (KNN)</li>
            <li onClick={() => goTo("/ml/supervised/treeclassification")} className="text-blue-600 cursor-pointer hover:underline">🔷 Decision Trees</li>
            <li onClick={() => goTo("/ml/supervised/svm")} className="text-blue-600 cursor-pointer hover:underline">🔷 Support Vector Machines (SVM)</li>
            <li onClick={() => goTo("/ml/supervised/naive-bayes")} className="text-blue-600 cursor-pointer hover:underline">🔷 Naive Bayes</li>
            <li onClick={() => goTo("/ml/supervised/random-forest")} className="text-blue-600 cursor-pointer hover:underline">🔷 Random Forest Classifier</li>
          </ul>
        </div>

        {/* 3. Advanced Ensemble Methods */}
        <div>
          <h3 className="text-lg font-semibold">3. Advanced Ensemble Methods</h3>
          <p>Combine multiple models to improve accuracy and robustness.</p>
          <ul className="list-disc list-inside ml-4">
            <li onClick={() => goTo("/ml/supervised/gbm")} className="text-blue-600 cursor-pointer hover:underline">🔷 Gradient Boosting Machines (GBM)</li>
            <li onClick={() => goTo("/ml/supervised/adaboost")} className="text-blue-600 cursor-pointer hover:underline">🔷 AdaBoost</li>
            <li onClick={() => goTo("/ml/supervised/xgboost")} className="text-blue-600 cursor-pointer hover:underline">🔷 XGBoost</li>
            <li onClick={() => goTo("/ml/supervised/lightgbm")} className="text-blue-600 cursor-pointer hover:underline">🔷 LightGBM</li>
            <li onClick={() => goTo("/ml/supervised/catboost")} className="text-blue-600 cursor-pointer hover:underline">🔷 CatBoost</li>
            <li onClick={() => goTo("/ml/supervised/stacking")} className="text-blue-600 cursor-pointer hover:underline">🔷 Stacking</li>
            <li onClick={() => goTo("/ml/supervised/bagging")} className="text-blue-600 cursor-pointer hover:underline">🔷 Bagging</li>
            <li onClick={() => goTo("/ml/supervised/extra-trees")} className="text-blue-600 cursor-pointer hover:underline">🔷 Extra Trees</li>
          <li onClick={() => goTo("/ml/supervised/fe")} className="text-blue-600 cursor-pointer hover:underline">🔷Feature Engineering </li>
          
          </ul>
        </div>
      </div>
      

      <div className="bg-green-100 p-4 rounded-md shadow-sm">
        <h2 className="text-xl font-semibold mb-2">🧠 Summary</h2>
        <table className="w-full text-sm table-auto">
          <thead>
            <tr className="bg-green-200 text-left">
              <th className="p-2">Feature</th>
              <th className="p-2">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 font-medium">Data</td>
              <td className="p-2">Labeled (input-output pairs)</td>
            </tr>
            <tr>
              <td className="p-2 font-medium">Goal</td>
              <td className="p-2">Predict output from input</td>
            </tr>
            <tr>
              <td className="p-2 font-medium">Types</td>
              <td className="p-2">Regression and Classification</td>
            </tr>
            <tr>
              <td className="p-2 font-medium">Use Cases</td>
              <td className="p-2">Price prediction, email spam detection, etc.</td>
            </tr>
            <tr>
              <td className="p-2 font-medium">Common Algorithms</td>
              <td className="p-2">Linear/Logistic Regression, SVM, Trees, KNN, etc.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
