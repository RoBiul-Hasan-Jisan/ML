import { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";


import Header from "./Header";
import Sidebar from "./Sidebar"; // import Sidebar here
import RegressionAlgorithms from "./RegressionAlgorithms";

// Import other pages as before
import MLIntroduction from "./MLslidepages/MLIntroduction";
import MLSupervised from "./MLslidepages/MLSupervised";
import MLUnsupervised from "./MLslidepages/MLUnsupervised";
import MLReinforcement from "./MLslidepages/MLReinforcement";

import LinearRegression from "./MLslidepages/MLSupervisedExtra/LinearRegression";
import LogisticRegression from "./MLslidepages/MLSupervisedExtra/LogisticRegression";
import DecisionTreeRegression from "./MLslidepages/MLSupervisedExtra/DecisionTreeRegression";
import RandomForestRegression from "./MLslidepages/MLSupervisedExtra/RandomForestRegression";
import SVRO from "./MLslidepages/MLSupervisedExtra/SVRO";
import RidgeRegression from "./MLslidepages/MLSupervisedExtra/RidgeRegression";
import LassoRegression from "./MLslidepages/MLSupervisedExtra/LassoRegression";
import ElasticNetRegression from "./MLslidepages/MLSupervisedExtra/ElasticNetRegression";

import SimpleLinearRegression from "./MLslidepages/MLSupervisedExtra/LinearRegressionAlgo/SimpleLinearRegression";
import MultipleLinearRegression from "./MLslidepages/MLSupervisedExtra/LinearRegressionAlgo/MultipleLinearRegression";
import PolynomialRegression from "./MLslidepages/MLSupervisedExtra/LinearRegressionAlgo/PolynomialRegression";

function AppContent() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) setIsSidebarOpen(false);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <>
      <Header />
      <div className="pt-20 flex min-h-screen relative">
        {/* Hamburger button */}
        {isMobile && (
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="fixed top-16 left-4 z-50 bg-transparent rounded-md shadow-md flex items-center justify-center"
            style={{ width: "32px", height: "32px" }}
            aria-label="Toggle Menu"
          >
            <svg
              className="w-5 h-5 text-blue"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        )}

        {/* Sidebar */}
        <aside
          className={`
            ${isMobile ? "fixed top-0 left-0 h-full z-40 transition-transform duration-300 bg-white shadow-lg" : "sticky top-0"}
            w-64 overflow-y-auto bg-gray-100 p-6
            ${isMobile ? (isSidebarOpen ? "translate-x-0" : "-translate-x-full") : ""}
          `}
        >
          <Sidebar />
        </aside>

        {/* Main content */}
        <main className="flex-1 p-4 md:p-8">
          <Routes>
            <Route path="/" element={<Navigate to="/ml/introduction" replace />} />
            <Route path="/ml/introduction" element={<MLIntroduction />} />
            <Route path="/ml/supervised" element={<MLSupervised />} />
            <Route path="/ml/unsupervised" element={<MLUnsupervised />} />
            <Route path="/ml/reinforcement" element={<MLReinforcement />} />
            <Route path="/ml/supervised/linear-regression" element={<LinearRegression />} />
            <Route path="/ml/supervised/logistic-regression" element={<LogisticRegression />} />
            <Route path="/ml/supervised/decision-tree" element={<DecisionTreeRegression />} />
            <Route path="/ml/supervised/random-forest" element={<RandomForestRegression />} />
            <Route path="/ml/supervised/svr" element={<SVRO />} />
            <Route path="/ml/supervised/ridge" element={<RidgeRegression />} />
            <Route path="/ml/supervised/lasso" element={<LassoRegression />} />
            <Route path="/ml/supervised/elastic-net" element={<ElasticNetRegression />} />
            <Route path="/ml/supervised/simple-linear-regression" element={<SimpleLinearRegression />} />
            <Route path="/ml/supervised/multiple-linear-regression" element={<MultipleLinearRegression />} />
            <Route path="/ml/supervised/polynomial-regression" element={<PolynomialRegression />} />
            <Route path="/regression-algorithms" element={<RegressionAlgorithms />} />
            <Route path="*" element={<Navigate to="/ml/introduction" replace />} />
          </Routes>
        </main>
      </div>
    </>
  );
}

// Main App wrapper
export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
