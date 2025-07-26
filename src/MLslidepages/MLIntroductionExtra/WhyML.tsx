// import React from "react";

export default function WhyML() {
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-6 text-gray-800">
      <h1 className="text-3xl sm:text-4xl font-bold mb-4 text-center sm:text-left">
        Why is Machine Learning Needed?
      </h1>

      <p className="text-base sm:text-lg">
        Machine Learning is needed because it provides intelligent solutions to problems that are too complex, dynamic, or vast for traditional programming methods. Here are the main reasons:
      </p>

      <ol className="list-decimal list-inside space-y-6 text-base sm:text-lg">
        <li>
          <div className="flex items-start mb-1">
            <span className="mr-2 text-2xl leading-none">🤖</span>
            <strong>Automation of Tasks</strong>
          </div>
          <p>
            ML enables automation of repetitive, rule-based, or time-consuming tasks.
          </p>
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-8 mt-1 space-y-1">
            <li>Spam detection in emails</li>
            <li>Automatic photo tagging</li>
            <li>Customer support chatbots</li>
          </ul>
        </li>

        <li>
          <div className="flex items-start mb-1">
            <span className="mr-2 text-2xl leading-none">📊</span>
            <strong>Handling Large-Scale and Complex Data</strong>
          </div>
          <p>
            Traditional programming struggles with massive datasets or unstructured data. ML algorithms can learn from complex, high-dimensional, or noisy data.
          </p>
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-8 mt-1 space-y-1">
            <li>Big Data analytics</li>
            <li>Social media monitoring</li>
          </ul>
        </li>

        <li>
          <div className="flex items-start mb-1">
            <span className="mr-2 text-2xl leading-none">🎯</span>
            <strong>Improved Accuracy Over Time</strong>
          </div>
          <p>
            ML systems improve as they receive more data. They adapt and learn without needing manual updates.
          </p>
          <em>Example:</em> Voice assistants like Alexa or Siri becoming more accurate with use.
        </li>

        <li>
          <div className="flex items-start mb-1">
            <span className="mr-2 text-2xl leading-none">⚡</span>
            <strong>Real-Time Decision Making</strong>
          </div>
          <p>
            ML models can provide instant predictions or decisions as data arrives.
          </p>
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-8 mt-1 space-y-1">
            <li>Stock price prediction</li>
            <li>Live fraud detection</li>
            <li>Real-time content recommendations</li>
          </ul>
        </li>

        <li>
          <div className="flex items-start mb-1">
            <span className="mr-2 text-2xl leading-none">🔁</span>
            <strong>Adaptability to Change</strong>
          </div>
          <p>
            ML systems can automatically adjust to changing patterns or environments.
          </p>
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-8 mt-1 space-y-1">
            <li>Dynamic ad targeting</li>
            <li>Recommendation systems adapting to user interests</li>
          </ul>
        </li>

        <li>
          <div className="flex items-start mb-1">
            <span className="mr-2 text-2xl leading-none">🔍</span>
            <strong>Pattern Recognition and Insight Discovery</strong>
          </div>
          <p>
            ML can uncover hidden patterns and insights in large datasets that humans might overlook.
          </p>
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-8 mt-1 space-y-1">
            <li>Medical diagnosis using MRI or X-ray analysis</li>
            <li>Consumer behavior prediction in marketing</li>
          </ul>
        </li>
      </ol>

      <p className="mt-8 font-semibold text-indigo-700 text-lg">
        In Summary:<br />
        Machine Learning is essential for building systems that are intelligent, scalable, efficient, and self-improving — especially in data-rich and rapidly evolving environments.
      </p>
    </div>
  );
}
