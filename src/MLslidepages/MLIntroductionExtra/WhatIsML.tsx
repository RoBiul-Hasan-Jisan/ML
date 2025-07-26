// import React from "react";

export default function WhatIsML() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 text-gray-800">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-6 text-center sm:text-left text-blue-700">
        ML Introduction
      </h1>

      <p className="mb-6 text-base sm:text-lg md:text-xl leading-relaxed">
        Machine Learning (ML) is a field of computer science that gives machines the ability to learn from data and experiences—much like humans do—and make predictions or decisions without being explicitly programmed to perform those tasks.
      </p>

      <p className="mb-8 text-base sm:text-lg md:text-xl leading-relaxed">
        Instead of following strict instructions coded by a developer, ML algorithms are trained using data, and from this data, they learn patterns, trends, and relationships to solve problems or automate processes.
      </p>

      <h2 className="text-xl sm:text-2xl font-semibold mb-4 text-blue-600">
        What is Machine Learning?
      </h2>
      <p className="mb-8 text-base sm:text-lg md:text-xl leading-relaxed">
        Machine Learning (ML) is a field of computer science that gives machines the ability to learn from data and experiences—much like humans do—and make predictions or decisions without being explicitly programmed to perform those tasks.
      </p>

      <h2 className="text-xl sm:text-2xl font-semibold mb-4 text-blue-600">
        How it Differs from Traditional Programming
      </h2>

      <div className="overflow-x-auto rounded shadow-md">
        <table className="w-full border border-gray-300 text-sm sm:text-base">
          <thead>
            <tr className="bg-gray-100 text-left">
              <th className="border border-gray-300 px-4 py-3">Traditional Programming</th>
              <th className="border border-gray-300 px-4 py-3">Machine Learning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-4 py-3">Programmer writes rules</td>
              <td className="border border-gray-300 px-4 py-3">Algorithm learns rules from data</td>
            </tr>
            <tr className="bg-gray-50">
              <td className="border border-gray-300 px-4 py-3">Input + Rules → Output</td>
              <td className="border border-gray-300 px-4 py-3">Input + Output → Algorithm learns rules</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-3">Example: if statements, loops</td>
              <td className="border border-gray-300 px-4 py-3">Example: decision trees, neural networks</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
