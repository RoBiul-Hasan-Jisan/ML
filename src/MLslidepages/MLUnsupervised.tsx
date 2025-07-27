import { useNavigate } from "react-router-dom";

export default function MLUnsupervised() {
  const navigate = useNavigate();
  const goTo = (path: string) => navigate(path);

  return (
    <div className="max-w-4xl mx-auto p-6 pt-16 space-y-6">
      <h1 className="text-2xl font-bold mb-4 text-purple-700">Unsupervised Machine Learning</h1>

      <p>
        <strong>Unsupervised Learning</strong> is a type of machine learning where the algorithm is given data <strong>without labeled outputs</strong>. The goal is to discover hidden patterns or structures from the data.
      </p>

      <div className="bg-gray-100 p-4 rounded-lg shadow-sm">
        <h2 className="text-xl font-semibold mb-2"> Example: Customer Segmentation</h2>
        <p><strong>Inputs:</strong> Age, Income, Spending Score</p>
        <p><strong>Output:</strong> Automatically discovered customer groups (clusters)</p>
      </div>

      <div className="space-y-2">
        <h2 className="text-xl font-semibold">How It Works</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Data Collection:</strong> Gather unlabeled data.</li>
          <li><strong>Feature Extraction:</strong> Identify relevant features.</li>
          <li><strong>Model Learning:</strong> Group, compress, or associate based on structure.</li>
          <li><strong>Pattern Discovery:</strong> Find clusters, associations, or reduce dimensions.</li>
        </ul>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold"> Types of Unsupervised Learning</h2>

        {/* 1. Clustering */}
        <div>
          <h3 className="text-lg font-semibold">1. Clustering</h3>
          <p>Group data based on similarity.</p>
          <ul className="list-disc list-inside ml-4">
            <li onClick={() => goTo("/ml/unsupervised/K-Means")} className="text-blue-600 cursor-pointer hover:underline">🔷 K-Means Clustering</li>
            <li onClick={() => goTo("/ml/unsupervised/hierarchical")} className="text-blue-600 cursor-pointer hover:underline">🔷 Hierarchical Clustering</li>
            <li onClick={() => goTo("/ml/unsupervised/dbscan")} className="text-blue-600 cursor-pointer hover:underline">🔷 DBSCAN</li>
            <li onClick={() => goTo("/ml/unsupervised/mean-shift")} className="text-blue-600 cursor-pointer hover:underline">🔷 Mean-Shift Clustering</li>
            <li onClick={() => goTo("/ml/unsupervised/spectral")} className="text-blue-600 cursor-pointer hover:underline">🔷 Spectral Clustering</li>
          <li onClick={() => goTo("/ml/unsupervised/gmm")} className="text-blue-600 cursor-pointer hover:underline">🔷 Gaussian Mixture Model </li>
          <li onClick={() => goTo("/ml/unsupervised/OP")} className="text-blue-600 cursor-pointer hover:underline">🔷 OPTICS  </li>
           <li onClick={() => goTo("/ml/unsupervised/BIRCH")} className="text-blue-600 cursor-pointer hover:underline">🔷 BIRCH  </li>
            <li onClick={() => goTo("/ml/unsupervised/affinity")} className="text-blue-600 cursor-pointer hover:underline">🔷 Affinity Propagation  </li>
          
          </ul>
        </div>

        {/* 2. Association Rule Learning */}
        <div>
          <h3 className="text-lg font-semibold">2. Association Rule Learning</h3>
          <p>Discover relationships between variables/items.</p>
          <ul className="list-disc list-inside ml-4">
            <li onClick={() => goTo("/ml/unsupervised/apriori")} className="text-blue-600 cursor-pointer hover:underline">🔷 Apriori Algorithm</li>
            <li onClick={() => goTo("/ml/unsupervised/eclat")} className="text-blue-600 cursor-pointer hover:underline">🔷 Eclat Algorithm</li>
          </ul>
        </div>

        {/* 3. Dimensionality Reduction */}
        <div>
          <h3 className="text-lg font-semibold">3. Dimensionality Reduction</h3>
          <p>Reduce the number of features while preserving structure.</p>
          <ul className="list-disc list-inside ml-4">
            <li onClick={() => goTo("/ml/unsupervised/pca")} className="text-blue-600 cursor-pointer hover:underline">🔷 PCA (Principal Component Analysis)</li>
            <li onClick={() => goTo("/ml/unsupervised/tsne")} className="text-blue-600 cursor-pointer hover:underline">🔷 t-SNE</li>
            <li onClick={() => goTo("/ml/unsupervised/umap")} className="text-blue-600 cursor-pointer hover:underline">🔷 UMAP</li>
            <li onClick={() => goTo("/ml/unsupervised/autoencoder")} className="text-blue-600 cursor-pointer hover:underline">🔷 Autoencoders</li>
          </ul>
        </div>
      </div>

      <div className="bg-red-100 p-4 rounded-md shadow-sm">
        <h2 className="text-xl font-semibold mb-2"> Challenges</h2>
        <ul className="list-disc list-inside">
          <li>No ground truth for evaluation</li>
          <li>Hard to interpret the results</li>
          <li>Parameter tuning can be difficult</li>
          <li>Not scalable for all algorithms</li>
        </ul>
      </div>

      <div className="bg-green-100 p-4 rounded-md shadow-sm">
        <h2 className="text-xl font-semibold mb-2">Applications</h2>
        <ul className="list-disc list-inside">
          <li>Customer segmentation in marketing</li>
          <li>Product recommendation (market basket analysis)</li>
          <li>Anomaly detection in cybersecurity</li>
          <li>Document/topic clustering in NLP</li>
          <li>Image compression and segmentation</li>
        </ul>
      </div>
    </div>
  );
}
