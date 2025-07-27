// import React from "react";

type TypesOfMLProps = {
  goTo: (path: string) => void;
};

export default function TypesOfML({ goTo }: TypesOfMLProps) {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 space-y-6 text-gray-800">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center sm:text-left">
        Types of Machine Learning
      </h1>
      <p className="text-base sm:text-lg">
        Machine Learning is broadly classified into four main types, based on how models learn from data.
      </p>

      {/* Supervised Learning */}
      <div
        onClick={() => goTo("/ml/supervised")}
        className="cursor-pointer border-l-4 border-green-500 bg-green-50 hover:bg-green-100 rounded p-4 shadow hover:shadow-lg transition"
      >
        <h2 className="text-lg sm:text-xl font-semibold mb-1"> Supervised Learning</h2>
        <p>
          <strong>Definition:</strong> The model learns from labeled data.<br />
          <strong>Goal:</strong> Learn a mapping from input to output.<br />
          <strong>Data Type:</strong> Input → Output pairs<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5 mt-1">
            <li>Regression: Predicting house prices</li>
            <li>Classification: Spam detection</li>
          </ul>
        </p>
      </div>

      {/* Unsupervised Learning */}
      <div
        onClick={() => goTo("/ml/unsupervised")}
        className="cursor-pointer border-l-4 border-blue-500 bg-blue-50 hover:bg-blue-100 rounded p-4 shadow hover:shadow-lg transition"
      >
        <h2 className="text-lg sm:text-xl font-semibold mb-1"> Unsupervised Learning</h2>
        <p>
          <strong>Definition:</strong> The model is given unlabeled data.<br />
          <strong>Goal:</strong> Discover hidden patterns or structure.<br />
          <strong>Data Type:</strong> Input only<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5 mt-1">
            <li>Clustering: Customer segmentation</li>
            <li>Dimensionality Reduction: PCA</li>
          </ul>
        </p>
      </div>

      {/* Semi-Supervised Learning */}
      <div
        className="cursor-pointer border-l-4 border-yellow-500 bg-yellow-50 hover:bg-yellow-100 rounded p-4 shadow hover:shadow-lg transition"
      >
        <h2 className="text-lg sm:text-xl font-semibold mb-1"> Semi-Supervised Learning</h2>
        <p>
          <strong>Definition:</strong> Combines small labeled with large unlabeled data.<br />
          <strong>Use Case:</strong> When labeling is expensive.<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5 mt-1">
            <li>Image recognition with few labels</li>
            <li>NLP with limited annotations</li>
          </ul>
        </p>
      </div>

      {/* Reinforcement Learning */}
      <div
        onClick={() => goTo("/ml/reinforcement")}
        className="cursor-pointer border-l-4 border-purple-500 bg-purple-50 hover:bg-purple-100 rounded p-4 shadow hover:shadow-lg transition"
      >
        <h2 className="text-lg sm:text-xl font-semibold mb-1"> Reinforcement Learning</h2>
        <p>
          <strong>Definition:</strong> An agent learns via feedback from actions.<br />
          <strong>Goal:</strong> Maximize cumulative rewards.<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5 mt-1">
            <li>Game-playing agents (e.g., AlphaGo)</li>
            <li>Autonomous robots</li>
          </ul>
        </p>
      </div>

      {/* Summary Table */}
      <div className="mt-8 overflow-x-auto rounded shadow">
        <table className="min-w-full text-sm sm:text-base text-left border-collapse">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-3 border">Type</th>
              <th className="p-3 border">Data Used</th>
              <th className="p-3 border">Goal</th>
              <th className="p-3 border">Examples</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-white">
              <td className="p-3 border">Supervised</td>
              <td className="p-3 border">Labeled data</td>
              <td className="p-3 border">Predict output</td>
              <td className="p-3 border">Spam detection, price prediction</td>
            </tr>
            <tr className="bg-gray-50">
              <td className="p-3 border">Unsupervised</td>
              <td className="p-3 border">Unlabeled data</td>
              <td className="p-3 border">Discover patterns</td>
              <td className="p-3 border">Clustering, PCA</td>
            </tr>
            <tr className="bg-white">
              <td className="p-3 border">Semi-Supervised</td>
              <td className="p-3 border">Labeled + Unlabeled</td>
              <td className="p-3 border">Improve learning</td>
              <td className="p-3 border">Image classification</td>
            </tr>
            <tr className="bg-gray-50">
              <td className="p-3 border">Reinforcement</td>
              <td className="p-3 border">Feedback (Reward)</td>
              <td className="p-3 border">Maximize reward</td>
              <td className="p-3 border">Games, robotics</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
