// import React from "react";

export default function MLvsTraditional() {
  return (
    <div className="px-4 sm:px-6  max-w-4xl mx-auto text-gray-800">
      <h2 className="text-lg sm:text-2xl font-semibold mb-6 mt-4 text-center sm:text-left text-blue-700">
        How it Differs from Traditional Programming
      </h2>

      <div className="overflow-x-auto rounded-md shadow-md">
        <table className="w-full border border-gray-300 text-sm sm:text-base">
          <thead>
            <tr className="bg-gray-100 text-left">
              <th className="border border-gray-300 px-4 py-3 whitespace-nowrap">
                Traditional Programming
              </th>
              <th className="border border-gray-300 px-4 py-3 whitespace-nowrap">
                Machine Learning
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-4 py-3">
                Programmer writes rules
              </td>
              <td className="border border-gray-300 px-4 py-3">
                Algorithm learns rules from data
              </td>
            </tr>
            <tr className="bg-gray-50">
              <td className="border border-gray-300 px-4 py-3">
                Input + Rules → Output
              </td>
              <td className="border border-gray-300 px-4 py-3">
                Input + Output → Algorithm learns rules
              </td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-3">
                Example: if statements, loops
              </td>
              <td className="border border-gray-300 px-4 py-3">
                Example: decision trees, neural networks
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
