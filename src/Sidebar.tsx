// src/Sidebar.tsx
import React, { useState } from "react";
import { NavLink } from "react-router-dom";

interface LinkItem {
  name: string;
  to: string;
}

interface SectionProps {
  title: string;
  links: LinkItem[];
  initiallyOpen?: boolean;
}

const SidebarSection: React.FC<SectionProps> = ({ title, links, initiallyOpen = false }) => {
  const [isOpen, setIsOpen] = useState(initiallyOpen);

  const toggleOpen = () => setIsOpen(!isOpen);

  // Create unique id for aria attributes
  const listId = title.toLowerCase().replace(/\s+/g, "-") + "-list";

  return (
    <div className="mt-6">
      <button
        onClick={toggleOpen}
        aria-expanded={isOpen}
        aria-controls={listId}
        id={listId + "-button"}
        className="flex justify-between items-center w-full text-gray-700 font-semibold hover:text-blue-700 focus:outline-none"
      >
        <span>{title}</span>
        <svg
          className={`w-4 h-4 transition-transform duration-200 ${isOpen ? "rotate-90" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {isOpen && (
        <ul
          id={listId}
          role="region"
          aria-labelledby={listId + "-button"}
          className="mt-2 ml-4 space-y-1"
        >
          {links.map(({ name, to }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) =>
                  "block px-2 py-1 rounded transition-colors " +
                  (isActive
                    ? "bg-blue-600 text-white font-semibold"
                    : "text-blue-600 hover:bg-blue-200")
                }
              >
                {name}
              </NavLink>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

const mainLinks: LinkItem[] = [
  { name: "Introduction", to: "/ml/introduction" },
  { name: "Supervised Learning", to: "/ml/supervised" },
  { name: "Unsupervised Learning", to: "/ml/unsupervised" },
  { name: "Reinforcement Learning", to: "/ml/reinforcement" },
];

const supervisedRegression: LinkItem[] = [
  { name: "Linear Regression", to: "/ml/supervised/linear-regression" },
  { name: "Decision Tree", to: "/ml/supervised/decision-tree" },
  { name: "Random Forest", to: "/ml/supervised/random-forest" },
  { name: "SVR", to: "/ml/supervised/svr" },
  { name: "Ridge Regression", to: "/ml/supervised/ridge" },
  { name: "Lasso Regression", to: "/ml/supervised/lasso" },
  { name: "Elastic Net", to: "/ml/supervised/elastic-net" },
];

const supervisedClassification: LinkItem[] = [
  { name: "Logistic Regression", to: "/ml/supervised/logistic-regression" },
  { name: "KNN", to: "/ml/supervised/knn" },
  { name: "LDA", to: "/ml/supervised/lda" },
  { name: "QDA", to: "/ml/supervised/qda" },
  { name: "Perceptron", to: "/ml/supervised/perceptron" },
  { name: "Decision Tree", to: "/ml/supervised/decision-treec" },
  { name: "SVM", to: "/ml/supervised/svm" },
  { name: "Naive Bayes", to: "/ml/supervised/naive-bayes" },
  { name: "Random Forest", to: "/ml/supervised/random-forests" },
];

const linearAlgorithms: LinkItem[] = [
  { name: "Simple Linear Regression", to: "/ml/supervised/simple-linear-regression" },
  { name: "Multiple Linear Regression", to: "/ml/supervised/multiple-linear-regression" },
  { name: "Polynomial Regression", to: "/ml/supervised/polynomial-regression" },
];

const methods: LinkItem[] = [
  { name: "Gradient Boosting Machines", to: "/ml/supervised/gbm" },
  { name: "Adaboost", to: "/ml/supervised/adaboost" },
  { name: "XGBoost", to: "/ml/supervised/xgboost" },
  { name: "LightGBM", to: "/ml/supervised/lightgbm" },
  { name: "CatBoost", to: "/ml/supervised/catboost" },
  { name: "Stacking", to: "/ml/supervised/stacking" },
  { name: "Bagging", to: "/ml/supervised/bagging" },
  { name: "Extra Trees", to: "/ml/supervised/extra-trees" },
  { name: "Feature Engineering", to: "/ml/supervised/fe" },
];

const miscLinks: LinkItem[] = [
  { name: "Regression Algorithms Overview", to: "/regression-algorithms-s" },
  { name: "Classification Algorithms Overview", to: "/classification-algorithms-s" },
];

export default function Sidebar() {
  return (
    <nav className="w-64 min-h-screen bg-gray-100 p-6 sticky top-0 overflow-y-auto shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-blue-800">Machine Learning</h2>

      {/* Main links */}
      <ul className="space-y-2">
        {mainLinks.map(({ name, to }) => (
          <li key={to}>
            <NavLink
              to={to}
              className={({ isActive }) =>
                "block px-3 py-2 rounded hover:bg-blue-300 transition " +
                (isActive ? "bg-blue-600 text-white font-semibold" : "text-blue-700")
              }
            >
              {name}
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Collapsible sections */}
      <SidebarSection title="Supervised Learning - Regression" links={supervisedRegression} initiallyOpen />
      <SidebarSection title="Supervised Learning - Classification" links={supervisedClassification} />
      <SidebarSection title="Linear Algorithms" links={linearAlgorithms} />
      <SidebarSection title="Methods" links={methods} />
      <SidebarSection title="More" links={miscLinks} />

      <hr className="my-6 border-gray-300" />

      <a
        href="https://craft-byte-hq.vercel.app/"
        className="block px-3 py-2 rounded hover:bg-gray-300 text-sm font-semibold text-gray-700"
      >
        ← Back to All Topics
      </a>
    </nav>
  );
}
