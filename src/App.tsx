import { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";


import Header from "./Header";
import Sidebar from "./Sidebar"; // import Sidebar here


// Import other pages as before
import MLIntroduction from "./MLslidepages/MLIntroduction";
import MLSupervised from "./MLslidepages/MLSupervised";
import MLUnsupervised from "./MLslidepages/MLUnsupervised";
import MLReinforcement from "./MLslidepages/MLReinforcement";
//MlSupervised Regression and classification 
import LinearRegression from "./MLslidepages/MLSupervisedExtra/LinearRegression";
import LogisticRegression from "./MLslidepages/MLSupervisedExtra/LogisticRegression";
import DecisionTreeRegression from "./MLslidepages/MLSupervisedExtra/DecisionTreeRegression";
import RandomForestRegression from "./MLslidepages/MLSupervisedExtra/RandomForestRegression";
import SVRO from "./MLslidepages/MLSupervisedExtra/SVRO";
import RidgeRegression from "./MLslidepages/MLSupervisedExtra/RidgeRegression";
import LassoRegression from "./MLslidepages/MLSupervisedExtra/LassoRegression";
import ElasticNetRegression from "./MLslidepages/MLSupervisedExtra/ElasticNetRegression";
import KNNGuide from "./MLslidepages/MLSupervisedExtra/KNNGuide";
import LDAGuide from "./MLslidepages/MLSupervisedExtra/LDAGuide";
import QDAGuide from "./MLslidepages/MLSupervisedExtra/QDAGuide";
import PerceptronGuide from "./MLslidepages/MLSupervisedExtra/PerceptronGuide";
import DecisionTreeGuide from "./MLslidepages/MLSupervisedExtra/DecisionTreeGuide";
import SVMGuide from "./MLslidepages/MLSupervisedExtra/SVMGuide";
import NaiveBayesGuide from "./MLslidepages/MLSupervisedExtra/NaiveBayesGuide";
import RandomForestGuide from "./MLslidepages/MLSupervisedExtra/RandomForestGuide";
import ClassificationAlgorithmsGuide from "./MLslidepages/ClassificationAlgorithmsGuide";
import GradientBoostingGuide from "./MLslidepages/MLSupervisedExtra/GradientBoostingGuide";
import AdaBoostGuide from "./MLslidepages/MLSupervisedExtra/AdaBoostGuide";
import XGBoostGuide from "./MLslidepages/MLSupervisedExtra/XGBoostGuide";
import LightGBMGuide from "./MLslidepages/MLSupervisedExtra/LightGBMGuide";
import CatBoostGuide from "./MLslidepages/MLSupervisedExtra/CatBoostGuide";
import StackingGuide from "./MLslidepages/MLSupervisedExtra/StackingGuide";
import BaggingGuide from "./MLslidepages/MLSupervisedExtra/BaggingGuide";
import ExtraTreesGuide from "./MLslidepages/MLSupervisedExtra/ExtraTreesGuide";
import FeatureEngineeringGuide from "./MLslidepages/MLSupervisedExtra/FeatureEngineeringGuide";


import RegressionAlgorithmsGuide from "./MLslidepages/RegressionAlgorithmsGuide";
import SimpleLinearRegression from "./MLslidepages/MLSupervisedExtra/LinearRegressionAlgo/SimpleLinearRegression";
import MultipleLinearRegression from "./MLslidepages/MLSupervisedExtra/LinearRegressionAlgo/MultipleLinearRegression";
import PolynomialRegression from "./MLslidepages/MLSupervisedExtra/LinearRegressionAlgo/PolynomialRegression";


import KMeansClustering from "./MLslidepages/MLUnsupervisedExtra/KMeansClustering";
import DBSCANGuide from "./MLslidepages/MLUnsupervisedExtra/DBSCANGuide";
import HierarchicalClustering from "./MLslidepages/MLUnsupervisedExtra/HierarchicalClustering";
import MeanShiftClustering from "./MLslidepages/MLUnsupervisedExtra/MeanShiftClustering";
import GaussianMixtureModel from "./MLslidepages/MLUnsupervisedExtra/GaussianMixtureModel";
import SpectralClustering from "./MLslidepages/MLUnsupervisedExtra/SpectralClustering";
import OPTICSClustering from "./MLslidepages/MLUnsupervisedExtra/OPTICSClustering";
import BIRCHClustering from "./MLslidepages/MLUnsupervisedExtra/BIRCHClustering";
import AffinityPropagationGuide from "./MLslidepages/MLUnsupervisedExtra/AffinityPropagationGuide";








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
    ${isMobile ? "fixed top-16 left-0 h-full z-40 transition-transform duration-300 bg-white shadow-lg" : "sticky top-16"}
    w-64 overflow-y-auto bg-gray-100 p-6
    ${isMobile ? (isSidebarOpen ? "translate-x-0" : "-translate-x-full") : ""}
  `}
>
  <Sidebar />
</aside>


        {/* Main content */}
      <main className="flex-1 p-0">


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
            <Route path="/ml/supervised/knn" element={<KNNGuide />} />
            <Route path="/ml/supervised/lda" element={<LDAGuide />} />
            <Route path="/ml/supervised/qda" element={<QDAGuide />} />
            <Route path="/ml/supervised/perceptron" element={<PerceptronGuide/>} />
            <Route path="/ml/supervised/decision-treec" element={<DecisionTreeGuide/>} />
            <Route path="/ml/supervised/svm" element={<SVMGuide/>} />
            <Route path="/ml/supervised/naive-bayes" element={<NaiveBayesGuide/>} />
            <Route path="/ml/supervised/random-forests" element={<RandomForestGuide/>} />
            <Route path="/ml/supervised/gbm" element={<GradientBoostingGuide/>} />
            <Route path="/ml/supervised/adaboost" element={<AdaBoostGuide/>} />
            <Route path="/ml/supervised/xgboost" element={<XGBoostGuide/>} />
            <Route path="/ml/supervised/lightgbm" element={<LightGBMGuide/>} />
            <Route path="/ml/supervised/catboost" element={<CatBoostGuide/>} />
            <Route path="/ml/supervised/stacking" element={<StackingGuide/>} />
            <Route path="/ml/supervised/bagging" element={<BaggingGuide/>} />
            <Route path="/ml/supervised/extra-trees" element={<ExtraTreesGuide/>} />
            <Route path="/ml/supervised/fe" element={<FeatureEngineeringGuide/>} />

    <Route path="/ml/unsupervised/K-Means" element={<KMeansClustering/>} />
   <Route path="/ml/unsupervised/dbscan" element={<DBSCANGuide/>} />
   <Route path="/ml/unsupervised/hierarchical" element={<HierarchicalClustering/>} />
   <Route path="/ml/unsupervised/mean-shift" element={<MeanShiftClustering/>} />
<Route path="/ml/unsupervised/gmm" element={<GaussianMixtureModel/>} />
<Route path="/ml/unsupervised/spectral" element={<SpectralClustering/>} />
<Route path="/ml/unsupervised/OP" element={<OPTICSClustering/>} />
<Route path="/ml/unsupervised/BIRCH" element={<BIRCHClustering/>} />
<Route path="/ml/unsupervised/affinity" element={<AffinityPropagationGuide/>} />
       
            
            <Route path="/regression-algorithms-s" element={<RegressionAlgorithmsGuide />} />
           <Route path="/classification-algorithms-s" element={<ClassificationAlgorithmsGuide/>} />
          
           
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
