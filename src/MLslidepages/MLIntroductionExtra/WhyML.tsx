//import React from "react";

export default function WhyML() {
  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold mb-4">Why is Machine Learning Needed?</h1>
      <p>
        Machine Learning is needed because it provides intelligent solutions to problems that are too complex, dynamic, or vast for traditional programming methods. Here are the main reasons:
      </p>

      <ol className="list-decimal list-inside space-y-4 text-lg">
        <li>
          <span className="mr-2">🤖</span>
          <strong>Automation of Tasks</strong><br />
          ML enables automation of repetitive, rule-based, or time-consuming tasks.
          <br />
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-6 mt-1">
            <li>Spam detection in emails</li>
            <li>Automatic photo tagging</li>
            <li>Customer support chatbots</li>
          </ul>
        </li>

        <li>
          <span className="mr-2">📊</span>
          <strong>Handling Large-Scale and Complex Data</strong><br />
          Traditional programming struggles with massive datasets or unstructured data. ML algorithms can learn from complex, high-dimensional, or noisy data.
          <br />
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-6 mt-1">
            <li>Big Data analytics</li>
            <li>Social media monitoring</li>
          </ul>
        </li>

        <li>
          <span className="mr-2">🎯</span>
          <strong>Improved Accuracy Over Time</strong><br />
          ML systems improve as they receive more data. They adapt and learn without needing manual updates.
          <br />
          <em>Example:</em> Voice assistants like Alexa or Siri becoming more accurate with use.
        </li>

        <li>
          <span className="mr-2">⚡</span>
          <strong>Real-Time Decision Making</strong><br />
          ML models can provide instant predictions or decisions as data arrives.
          <br />
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-6 mt-1">
            <li>Stock price prediction</li>
            <li>Live fraud detection</li>
            <li>Real-time content recommendations</li>
          </ul>
        </li>

        <li>
          <span className="mr-2">🔁</span>
          <strong>Adaptability to Change</strong><br />
          ML systems can automatically adjust to changing patterns or environments.
          <br />
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-6 mt-1">
            <li>Dynamic ad targeting</li>
            <li>Recommendation systems adapting to user interests</li>
          </ul>
        </li>

        <li>
          <span className="mr-2">🔍</span>
          <strong>Pattern Recognition and Insight Discovery</strong><br />
          ML can uncover hidden patterns and insights in large datasets that humans might overlook.
          <br />
          <em>Examples:</em>
          <ul className="list-disc list-inside ml-6 mt-1">
            <li>Medical diagnosis using MRI or X-ray analysis</li>
            <li>Consumer behavior prediction in marketing</li>
          </ul>
        </li>
      </ol>

      <p className="mt-6 font-semibold text-indigo-700">
        In Summary:<br />
        Machine Learning is essential for building systems that are intelligent, scalable, efficient, and self-improving — especially in data-rich and rapidly evolving environments.
      </p>
    </div>
  );
}
