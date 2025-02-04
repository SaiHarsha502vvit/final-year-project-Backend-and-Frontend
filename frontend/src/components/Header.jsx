import React from "react";

const Header = () => {
  return (
    <header className="flex flex-col items-center justify-center">
      <svg
        className="w-20 h-20 text-blue-600"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          strokeWidth="2" 
          d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 110 20 10 10 0 010-20z" 
        />
      </svg>
      <h1 className="text-4xl font-bold mt-4">Fraud Detection</h1>
      <p className="mt-2 text-gray-700 text-center">
        Enter transaction features to predict potential fraud
      </p>
    </header>
  );
};

export default Header;