//import React from "react";

export default function MLvsTraditional() {
  return (
    <div className="px-4 sm:px-6 py-6 max-w-4xl mx-auto text-gray-800">
      <h2 className="text-xl sm:text-2xl font-semibold mb-4 mt-8 text-blue-700">
        How it Differs from Traditional Programming
      </h2>

      <div className="overflow-x-auto">
        <table className="w-full border border-gray-300 text-sm sm:text-base">
          <thead>
            <tr className="bg-gray-100 text-left">
              <th className="border border-gray-300 px-4 py-2">
                Traditional Programming
              </th>
              <th className="border border-gray-300 px-4 py-2">
                Machine Learning
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-4 py-2">
                Programmer writes rules
              </td>
              <td className="border border-gray-300 px-4 py-2">
                Algorithm learns rules from data
              </td>
            </tr>
            <tr className="bg-gray-50">
              <td className="border border-gray-300 px-4 py-2">
                Input + Rules → Output
              </td>
              <td className="border border-gray-300 px-4 py-2">
                Input + Output → Algorithm learns rules
              </td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">
                Example: if statements, loops
              </td>
              <td className="border border-gray-300 px-4 py-2">
                Example: decision trees, neural networks
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
