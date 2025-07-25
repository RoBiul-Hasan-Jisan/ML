import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";

export default function SidebarML() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) setIsOpen(false);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const isActive = (path: string) =>
    location.pathname === path ? "bg-blue-400 text-white" : "";

  return (
    <>
      {/* Sidebar: fixed on mobile, relative on desktop */}
      <aside
        className={`fixed top-0 left-0 h-full w-64 bg-gray-200 p-4 pt-16 z-50 transform transition-transform duration-300 ease-in-out
          ${isMobile ? (isOpen ? "translate-x-0" : "-translate-x-full") : "translate-x-0 relative"}`}
      >
        <nav className="flex flex-col space-y-1">
          <h2 className="font-bold mb-4 mt-8 md:mt-0 text-xl">Learn ML</h2>

          <Link
            to="/ml/introduction"
            className={`block py-2 px-3 rounded hover:bg-gray-300 ${isActive(
              "/ml/introduction"
            )}`}
            onClick={() => isMobile && setIsOpen(false)}
          >
            Introduction
          </Link>

          <Link
            to="/ml/supervised"
            className={`block py-2 px-3 rounded hover:bg-gray-300 ${isActive(
              "/ml/supervised"
            )}`}
            onClick={() => isMobile && setIsOpen(false)}
          >
            Supervised Learning
          </Link>

          <Link
            to="/ml/unsupervised"
            className={`block py-2 px-3 rounded hover:bg-gray-300 ${isActive(
              "/ml/unsupervised"
            )}`}
            onClick={() => isMobile && setIsOpen(false)}
          >
            Unsupervised Learning
          </Link>

          <Link
            to="/ml/reinforcement"
            className={`block py-2 px-3 rounded hover:bg-gray-300 ${isActive(
              "/ml/reinforcement"
            )}`}
            onClick={() => isMobile && setIsOpen(false)}
          >
            Reinforcement Learning
          </Link>

          <hr className="my-4" />

          <Link
            to="/"
            className="block py-2 px-3 rounded hover:bg-gray-300 text-sm font-semibold"
            onClick={() => isMobile && setIsOpen(false)}
          >
            ← Back to All Topics
          </Link>
        </nav>
      </aside>

      {/* Overlay behind sidebar on mobile */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-40 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
