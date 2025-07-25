//import React from "react";

type TypesOfMLProps = {
  goTo: (path: string) => void;
};

export default function TypesOfML({ goTo }: TypesOfMLProps) {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold mb-6">Types of Machine Learning</h1>
      <p>
        Machine Learning is broadly classified into four main types, based on how models learn from data.
      </p>

      {/* 1. Supervised Learning */}
      <div
        onClick={() => goTo("/ml/supervised")}
        className="cursor-pointer border-l-4 border-green-500 bg-green-50 hover:bg-green-100 rounded p-4 shadow transition"
      >
        <h2 className="text-xl font-semibold mb-1">✅ Supervised Learning</h2>
        <p>
          <strong>Definition:</strong> The model learns from labeled data, where each input has a known output.<br />
          <strong>Goal:</strong> Learn a mapping function from input to output.<br />
          <strong>Data Type:</strong> Input → Output pairs<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5">
            <li>Regression: Predicting house prices, estimating temperature</li>
            <li>Classification: Spam detection, disease diagnosis</li>
          </ul>
        </p>
      </div>

      {/* 2. Unsupervised Learning */}
      <div
        onClick={() => goTo("/ml/unsupervised")}
        className="cursor-pointer border-l-4 border-blue-500 bg-blue-50 hover:bg-blue-100 rounded p-4 shadow transition"
      >
        <h2 className="text-xl font-semibold mb-1">🧩 Unsupervised Learning</h2>
        <p>
          <strong>Definition:</strong> The model is given unlabeled data and must identify patterns or structure.<br />
          <strong>Goal:</strong> Discover hidden patterns or groupings.<br />
          <strong>Data Type:</strong> Input only<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5">
            <li>Clustering: Customer segmentation, market analysis</li>
            <li>Dimensionality Reduction: PCA, t-SNE</li>
          </ul>
        </p>
      </div>

      {/* 3. Semi-Supervised Learning */}
      <div className="border-l-4 border-yellow-500 bg-yellow-50 rounded p-4 shadow">
        <h2 className="text-xl font-semibold mb-1">⚖️ Semi-Supervised Learning</h2>
        <p>
          <strong>Definition:</strong> Uses a small amount of labeled data and a large amount of unlabeled data.<br />
          <strong>Use Case:</strong> When labeling is expensive or time-consuming.<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5">
            <li>Image recognition with few labels</li>
            <li>NLP with limited annotation</li>
          </ul>
        </p>
      </div>

      {/* 4. Reinforcement Learning */}
      <div
        onClick={() => goTo("/ml/reinforcement")}
        className="cursor-pointer border-l-4 border-purple-500 bg-purple-50 hover:bg-purple-100 rounded p-4 shadow transition"
      >
        <h2 className="text-xl font-semibold mb-1">🎮 Reinforcement Learning</h2>
        <p>
          <strong>Definition:</strong> An agent learns by interacting with an environment and receiving feedback in the form of rewards or penalties.<br />
          <strong>Goal:</strong> Learn the optimal policy to maximize cumulative rewards.<br />
          <strong>Examples:</strong>
          <ul className="list-disc list-inside ml-5">
            <li>Game-playing agents (e.g., AlphaGo)</li>
            <li>Robotics and autonomous driving</li>
          </ul>
        </p>
      </div>

      {/* Summary Table */}
      <div className="mt-8 border rounded overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-2 border">Type</th>
              <th className="p-2 border">Data Used</th>
              <th className="p-2 border">Goal</th>
              <th className="p-2 border">Examples</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border">Supervised</td>
              <td className="p-2 border">Labeled data</td>
              <td className="p-2 border">Predict output</td>
              <td className="p-2 border">Spam detection, price prediction</td>
            </tr>
            <tr>
              <td className="p-2 border">Unsupervised</td>
              <td className="p-2 border">Unlabeled data</td>
              <td className="p-2 border">Discover patterns</td>
              <td className="p-2 border">Clustering, PCA</td>
            </tr>
            <tr>
              <td className="p-2 border">Semi-Supervised</td>
              <td className="p-2 border">Labeled + Unlabeled</td>
              <td className="p-2 border">Improve learning</td>
              <td className="p-2 border">Image classification</td>
            </tr>
            <tr>
              <td className="p-2 border">Reinforcement</td>
              <td className="p-2 border">Feedback (Reward)</td>
              <td className="p-2 border">Maximize reward</td>
              <td className="p-2 border">Games, robotics</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
