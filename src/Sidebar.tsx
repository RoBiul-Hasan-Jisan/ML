// src/Sidebar.tsx
import { NavLink, useLocation } from "react-router-dom";

export default function Sidebar() {
  const location = useLocation();

  const links = [
    { name: "Introduction", to: "/ml/introduction" },
    { name: "Supervised Learning", to: "/ml/supervised" },
    { name: "Unsupervised Learning", to: "/ml/unsupervised" },
    { name: "Reinforcement Learning", to: "/ml/reinforcement" },
  ];

  const supervisedDetails = [
    { name: "Linear Regression", to: "/ml/supervised/linear-regression" },
    { name: "Logistic Regression", to: "/ml/supervised/logistic-regression" },
    { name: "Decision Tree", to: "/ml/supervised/decision-tree" },
    { name: "Random Forest", to: "/ml/supervised/random-forest" },
    { name: "SVR", to: "/ml/supervised/svr" },
    { name: "Ridge Regression", to: "/ml/supervised/ridge" },
    { name: "Lasso Regression", to: "/ml/supervised/lasso" },
    { name: "Elastic Net", to: "/ml/supervised/elastic-net" },
  ];

  const linearAlgorithms = [
    { name: "Simple Linear Regression", to: "/ml/supervised/simple-linear-regression" },
    { name: "Multiple Linear Regression", to: "/ml/supervised/multiple-linear-regression" },
    { name: "Polynomial Regression", to: "/ml/supervised/polynomial-regression" },
  ];

  const miscLinks = [
    { name: "Regression Algorithms Overview", to: "/regression-algorithms" },
  ];

  const linkClass = (path: string) =>
    "block px-2 py-1 rounded hover:bg-blue-200 " +
    (location.pathname === path ? "bg-blue-600 text-white font-semibold" : "text-blue-600");

  return (
    <nav className="w-64 min-h-screen bg-gray-100 p-6 sticky top-0 overflow-y-auto">
      <h2 className="text-xl font-bold mb-6">Machine Learning</h2>

      <ul className="space-y-2">
        {links.map(({ name, to }) => (
          <li key={to}>
            <NavLink to={to} className={linkClass(to)}>
              {name}
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="mt-6 font-semibold text-gray-700">Supervised Details</div>
      <ul className="space-y-1 ml-4 mt-2">
        {supervisedDetails.map(({ name, to }) => (
          <li key={to}>
            <NavLink to={to} className={linkClass(to)}>
              {name}
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="mt-6 font-semibold text-gray-700">Linear Algorithms</div>
      <ul className="space-y-1 ml-4 mt-2">
        {linearAlgorithms.map(({ name, to }) => (
          <li key={to}>
            <NavLink to={to} className={linkClass(to)}>
              {name}
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="mt-6 font-semibold text-gray-700">More</div>
      <ul className="space-y-1 ml-4 mt-2">
        {miscLinks.map(({ name, to }) => (
          <li key={to}>
            <NavLink to={to} className={linkClass(to)}>
              {name}
            </NavLink>
          </li>
        ))}
      </ul>

      <hr className="my-4" />

      <a
        href="https://craft-byte-hq.vercel.app/"
        className="block px-3 py-2 rounded hover:bg-gray-300 text-sm font-semibold"
      >
        ← Back to All Topics
      </a>
    </nav>
  );
}
