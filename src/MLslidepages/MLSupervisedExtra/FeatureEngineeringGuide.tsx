export default function FeatureEngineeringGuide() {
  return (
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
      <h1 className="text-2xl sm:text-3xl font-bold text-green-700">Feature Engineering – Full Guide</h1>

      {/* 1. What is Feature Engineering? */}
      <section>
        <h2 className="font-semibold text-lg mb-2">1. What is Feature Engineering?</h2>
        <p>
          Feature Engineering is the process of creating, transforming, selecting, or extracting features (input variables) from raw data to improve machine learning model performance.
        </p>
        <p>Good features make it easier for models to learn meaningful patterns.</p>
        <ul className="list-disc ml-6 mt-2 space-y-1">
          <li><strong>Feature:</strong> measurable property or characteristic (e.g., age, income).</li>
          <li><strong>Raw data:</strong> original unprocessed data.</li>
          <li><strong>Feature:</strong> processed or derived data used as ML input.</li>
        </ul>
      </section>

      {/* 2. Steps in Feature Engineering */}
      <section>
        <h2 className="font-semibold text-lg mb-2">2. Steps in Feature Engineering</h2>
        <table className="w-full border border-gray-300 text-xs sm:text-sm">
          <thead className="bg-gray-200">
            <tr>
              <th className="border px-2 py-1 text-left">Step</th>
              <th className="border px-2 py-1 text-left">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border px-2 py-1">1. Data Cleaning</td>
              <td className="border px-2 py-1">Handle missing values, remove duplicates, fix errors</td>
            </tr>
            <tr>
              <td className="border px-2 py-1">2. Feature Extraction</td>
              <td className="border px-2 py-1">Create new features from existing raw data</td>
            </tr>
            <tr>
              <td className="border px-2 py-1">3. Feature Transformation</td>
              <td className="border px-2 py-1">Normalize, scale, encode categorical variables</td>
            </tr>
            <tr>
              <td className="border px-2 py-1">4. Feature Selection</td>
              <td className="border px-2 py-1">Select the most relevant features</td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* 3. Common Feature Engineering Techniques */}
      <section>
        <h2 className="font-semibold text-lg mb-2">3. Common Feature Engineering Techniques</h2>

        {/* Handling Missing Data */}
        <h3 className="font-semibold mb-1">Handling Missing Data</h3>
        <ul className="list-disc ml-6 mb-4 space-y-1">
          <li>Fill missing values with mean, median, or mode</li>
          <li>Model-based imputation (KNN, MICE)</li>
          <li>Flag missingness as a separate binary feature</li>
        </ul>

        {/* Encoding Categorical Variables */}
        <h3 className="font-semibold mb-1">Encoding Categorical Variables</h3>
        <ul className="list-disc ml-6 mb-4 space-y-1">
          <li>One-Hot Encoding (dummy variables)</li>
          <li>Label Encoding (ordinal)</li>
          <li>Target / Mean Encoding</li>
          <li>Frequency Encoding</li>
        </ul>

        {/* Feature Scaling */}
        <h3 className="font-semibold mb-1">Feature Scaling</h3>
        <ul className="list-disc ml-6 mb-4 space-y-1">
          <li>Min-Max Scaling (0-1 normalization)</li>
          <li>Standardization (z-score)</li>
          <li>Robust Scaling (median and IQR)</li>
        </ul>

        {/* Feature Creation / Extraction */}
        <h3 className="font-semibold mb-1">Feature Creation / Extraction</h3>
        <ul className="list-disc ml-6 mb-4 space-y-1">
          <li>Date/time features (day of week, holiday flag)</li>
          <li>Text features (TF-IDF, word embeddings)</li>
          <li>Aggregations (mean, sum, count per group)</li>
          <li>Polynomial features (interaction terms, powers)</li>
        </ul>

        {/* Dimensionality Reduction */}
        <h3 className="font-semibold mb-1">Dimensionality Reduction</h3>
        <ul className="list-disc ml-6 space-y-1">
          <li>PCA (Principal Component Analysis)</li>
          <li>t-SNE / UMAP (for visualization, not typical features)</li>
        </ul>
      </section>

      {/* 4. Advanced Feature Engineering */}
      <section>
        <h2 className="font-semibold text-lg mb-2">4. Advanced Feature Engineering</h2>
        <ul className="list-disc ml-6 space-y-1">
          <li>Feature Crossing: Combine features (e.g., Age * Income)</li>
          <li>Binning: Convert numerical to categorical bins (e.g., age groups)</li>
          <li>Use domain knowledge for custom features</li>
          <li>Use external data (weather, demographics)</li>
          <li>Automated tools (FeatureTools, AutoML)</li>
        </ul>
      </section>

      {/* 5. Python Workflow Example */}
      <section>
        <h2 className="font-semibold text-lg mb-2">5. Feature Engineering Workflow Example (Python)</h2>
        <pre className="bg-gray-100 p-3 rounded text-xs sm:text-sm font-mono whitespace-pre-wrap">
{`import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv('data.csv')

# 1. Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)

# 2. Encode categorical variables
ohe = OneHotEncoder(sparse=False)
cat_features = ohe.fit_transform(df[['gender', 'country']])

# 3. Scale numerical features
scaler = StandardScaler()
num_features = scaler.fit_transform(df[['age', 'income']])

# 4. Combine features
import numpy as np
X = np.hstack([num_features, cat_features])`}
        </pre>
      </section>

 
    <div className="max-w-full px-4 py-6 sm:max-w-3xl mx-auto space-y-6 text-sm sm:text-base leading-relaxed break-words">
    

      {/* Handling Missing Data */}
      <section>
        <h2 className="font-semibold text-lg mb-2">Handling Missing Data</h2>

        <h3 className="font-semibold">Fill missing values with mean, median, or mode</h3>
        <p><strong>Why:</strong> Keeps dataset complete for algorithms that can’t handle missing values. Mean and median are simple and fast; median better if outliers exist. Mode is for categorical features.</p>
        <p><strong>When:</strong> Missingness is random or minimal; quick fix or baseline.</p>

        <h3 className="font-semibold mt-4">Model-based imputation (KNN, MICE)</h3>
        <p><strong>Why:</strong> Uses feature relationships to predict missing values; often more accurate than simple filling.</p>
        <p><strong>When:</strong> Missing data is significant or structured; enough data available to train models.</p>

        <h3 className="font-semibold mt-4">Flag missingness as a separate binary feature</h3>
        <p><strong>Why:</strong> Missing data might carry information itself; helps model learn missingness patterns.</p>
        <p><strong>When:</strong> Missingness is informative or non-random; use alongside imputation.</p>
      </section>

      {/* Encoding Categorical Variables */}
      <section>
        <h2 className="font-semibold text-lg mb-2">Encoding Categorical Variables</h2>

        <h3 className="font-semibold">One-Hot Encoding (dummy variables)</h3>
        <p><strong>Why:</strong> Converts categories to binary vectors so algorithms can process them.</p>
        <p><strong>When:</strong> Nominal categories, small category count, used with linear models, SVM, neural nets.</p>

        <h3 className="font-semibold mt-4">Label Encoding (ordinal encoding)</h3>
        <p><strong>Why:</strong> Assigns integer labels implying order; models leverage ordinal info.</p>
        <p><strong>When:</strong> Categories have natural order; tree-based models handle label encoding even if nominal.</p>

        <h3 className="font-semibold mt-4">Target / Mean Encoding</h3>
        <p><strong>Why:</strong> Captures category-target relation; can improve power by summarizing target stats.</p>
        <p><strong>When:</strong> Enough data to avoid overfitting; apply carefully to avoid leakage (cross-validation).</p>

        <h3 className="font-semibold mt-4">Frequency Encoding</h3>
        <p><strong>Why:</strong> Encodes categories by their frequency, capturing prevalence.</p>
        <p><strong>When:</strong> When category frequency correlates with target; good for high-cardinality features.</p>
      </section>

      {/* Feature Scaling */}
      <section>
        <h2 className="font-semibold text-lg mb-2">Feature Scaling</h2>

        <h3 className="font-semibold">Min-Max Scaling (0-1 normalization)</h3>
        <p><strong>Why:</strong> Rescales features to [0,1], keeps relative relationships intact.</p>
        <p><strong>When:</strong> Needed for bounded inputs (e.g., neural nets); no outliers present.</p>

        <h3 className="font-semibold mt-4">Standardization (z-score)</h3>
        <p><strong>Why:</strong> Centers data around 0 mean, std dev 1; makes features comparable.</p>
        <p><strong>When:</strong> Features normally distributed or close; algorithms sensitive to scale (SVM, logistic regression).</p>

        <h3 className="font-semibold mt-4">Robust Scaling (median and IQR)</h3>
        <p><strong>Why:</strong> Uses median and interquartile range; robust to outliers.</p>
        <p><strong>When:</strong> Data has outliers or skewed distribution.</p>
      </section>

      {/* Feature Creation / Extraction */}
      <section>
        <h2 className="font-semibold text-lg mb-2">Feature Creation / Extraction</h2>

        <h3 className="font-semibold">Date/time features (day of week, holiday flag)</h3>
        <p><strong>Why:</strong> Time affects behavior; extracting components captures temporal patterns.</p>
        <p><strong>When:</strong> Timestamp data with suspected temporal trends.</p>

        <h3 className="font-semibold mt-4">Text features (TF-IDF, word embeddings)</h3>
        <p><strong>Why:</strong> Transform raw text to numerical form capturing importance or semantics.</p>
        <p><strong>When:</strong> Working with reviews, tweets, documents, etc.</p>

        <h3 className="font-semibold mt-4">Aggregations (mean, sum, count per group)</h3>
        <p><strong>Why:</strong> Summarize behavior or relationships per group (e.g., avg spend per customer).</p>
        <p><strong>When:</strong> Hierarchical or transactional data with multiple records per entity.</p>

        <h3 className="font-semibold mt-4">Polynomial features (interaction terms, powers)</h3>
        <p><strong>Why:</strong> Model nonlinear relationships and interactions.</p>
        <p><strong>When:</strong> Using linear models, expect interactions to improve performance.</p>
      </section>

      {/* Dimensionality Reduction */}
      <section>
        <h2 className="font-semibold text-lg mb-2">Dimensionality Reduction</h2>

        <h3 className="font-semibold">PCA (Principal Component Analysis)</h3>
        <p><strong>Why:</strong> Reduce dimensionality by combining correlated features; explains variance; helps noisy/high-dim data.</p>
        <p><strong>When:</strong> Reduce complexity, speed up models, visualize data; preprocessing step for many models.</p>

        <h3 className="font-semibold mt-4">t-SNE / UMAP</h3>
        <p><strong>Why:</strong> Nonlinear visualization of high-dim data in 2D/3D.</p>
        <p><strong>When:</strong> Explore clusters and data structure; not used as predictive features.</p>
      </section>

      {/* Advanced Feature Engineering */}
      <section>
        <h2 className="font-semibold text-lg mb-2">Advanced Feature Engineering (Why &amp; When)</h2>

        <ul className="list-disc ml-6 space-y-1">
          <li><strong>Feature Crossing:</strong> Combine features (e.g., Age * Income) to capture interaction effects.</li>
          <li><strong>Binning:</strong> Convert numeric to categorical bins to simplify relationships and reduce noise.</li>
          <li><strong>Use domain knowledge:</strong> Add meaningful features informed by problem context.</li>
          <li><strong>Use external data:</strong> Enrich dataset with related info (weather, demographics) influencing target.</li>
          <li><strong>Automated tools:</strong> FeatureTools, AutoML for large datasets or rapid prototyping.</li>
        </ul>
      </section>
    </div>



    </div>
  );
}
