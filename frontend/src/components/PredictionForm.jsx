import React from "react";
import { useState } from "react";

// Updated labels with 30 elements: "Time", V1...V28, "Amount"
const labels = [
  "Time",
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  "Amount",
];

const PredictionForm = () => {
  // Set the initial array length based on labels
  const initialArray = new Array(labels.length).fill(0);
  const initialText = initialArray.join(", ");
  const [features, setFeatures] = useState(initialText);
  const [featuresData, setFeaturesData] = useState(initialArray);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  const handleFeaturesChange = (e) => {
    const text = e.target.value;
    setFeatures(text);
    const parts = text.split(",").map((s) => parseFloat(s.trim()));
    if (parts.length === labels.length && !parts.some(isNaN)) {
      setFeaturesData(parts);
    }
  };

  const handleSliderChange = (index, value) => {
    // Ensure Time (index 0) and Amount (index 29) remain non-negative.
    if ((index === 0 || index === 29) && value < 0) {
      value = 0;
    }
    const updated = [...featuresData];
    updated[index] = value;
    setFeaturesData(updated);
    setFeatures(updated.join(", "));
  };

  const handleLoadExample = () => {
    // Create a random example:
    // Time: random between 0 and 100 (2 decimal places)
    const randomTime = parseFloat((Math.random() * 100).toFixed(2));
    // 28 high-precision features between -10 and 10
    const randomMiddle = Array.from({ length: 28 }, () =>
      parseFloat((Math.random() * 20 - 10).toFixed(6))
    );
    // Amount: random between 0 and 1,000,000 (2 decimal places)
    const randomAmount = parseFloat((Math.random() * 1000000).toFixed(2));
    const exampleData = [
      randomTime, 
      ...randomMiddle, 
      randomAmount
    ];
    setFeaturesData(exampleData);
    setFeatures(exampleData.join(", "));
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (
      featuresData.length !== labels.length ||
      featuresData.some(isNaN)
    ) {
      alert(`Please enter exactly ${labels.length} numerical values, separated by commas.`);
      return;
    }
    // Enforce non-negative for Time and Amount.
    if (featuresData[0] < 0 || featuresData[29] < 0) {
      alert("Time and Amount cannot be negative.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: featuresData }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed.");
      }
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFeatures(initialText);
    setFeaturesData(initialArray);
    setResult(null);
  };

  const toggleDarkMode = () => {
    setDarkMode((prevMode) => !prevMode);
  };

  return (
    <div
      className={`min-h-screen ${
        darkMode ? "bg-gray-900" : "bg-gradient-to-br from-blue-100 to-purple-200"
      } flex flex-col items-center justify-center p-6 transition-colors`}
    >
      <div className="mb-4 self-end">
        <button
          onClick={toggleDarkMode}
          className="py-2 px-4 rounded-full bg-purple-600 hover:bg-purple-700 text-white shadow-md transition"
        >
          {darkMode ? "Light Mode" : "Dark Mode"}
        </button>
      </div>
      <div className="bg-white dark:bg-gray-800 bg-opacity-95 shadow-2xl rounded-3xl max-w-7xl w-full overflow-hidden backdrop-blur-md transition-colors">
        <div className="flex flex-col md:flex-row">
          {/* Left side: Form */}
          <div className="md:w-1/2 p-10 border-b md:border-b-0 md:border-r border-gray-200 dark:border-gray-700">
            <h2 className="text-4xl font-extrabold text-gray-800 dark:text-gray-100 mb-8 text-center">
              Transaction Prediction
            </h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label
                  htmlFor="features"
                  className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2"
                >
                  Enter {labels.length} Features (comma-separated)
                </label>
                <textarea
                  id="features"
                  rows="6"
                  value={features}
                  onChange={handleFeaturesChange}
                  className="w-full p-4 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-400 transition bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100"
                  placeholder="e.g. 0.1, 0.2, 0.3, …, 3.0"
                />
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                  Provide exactly {labels.length} numerical values.
                </p>
              </div>
              <div className="flex justify-center gap-4 flex-wrap">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-8 rounded-xl shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <svg
                        className="animate-spin h-6 w-6 text-white"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8v8H4z"
                        ></path>
                      </svg>
                      Loading...
                    </>
                  ) : (
                    <>
                      <svg
                        className="w-6 h-6"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M14 5l7 7m0 0l-7 7m7-7H3"
                        />
                      </svg>
                      Predict
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="bg-gray-500 hover:bg-gray-600 text-white font-medium py-3 px-8 rounded-xl shadow-lg transition"
                >
                  Clear
                </button>
                <button
                  type="button"
                  onClick={handleLoadExample}
                  className="bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-8 rounded-xl shadow-lg transition"
                >
                  Load Example
                </button>
              </div>
            </form>
            {result && (
              <div
                className={`mt-8 p-5 rounded-xl text-center border-2 ${
                  result.prediction === 1
                    ? "bg-red-50 border-red-500 text-red-800"
                    : "bg-green-50 border-green-500 text-green-800"
                }`}
              >
                <p className="text-2xl font-bold mb-2">
                  Prediction: {result.prediction}
                </p>
                <p className="mb-2">
                  Probability: {result.probability.toFixed(4)}
                </p>
                {result.prediction === 1 ? (
                  <p className="font-semibold">
                    Fraud Transaction Detected!
                  </p>
                ) : (
                  <p className="font-semibold">Transaction is Not Fraud</p>
                )}
              </div>
            )}
          </div>
          {/* Right side: Live sliders */}
          <div className="md:w-1/2 p-10 max-h-[600px] overflow-y-auto">
            <h3 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-6">
              Live Feature Adjustment
            </h3>
            {labels.map((label, index) => {
              // For V1-V28 show high precision; Time and Amount slider settings are special.
              const isHighPrecision = index >= 1 && index <= 28;
              const sliderMax =
                index === 29 ? "1000000" : "100"; // index 29 is Amount
              const sliderStep =
                index === 29 ? "1" : isHighPrecision ? "0.000001" : "0.1";

              // For Time (index 0), Amount (index 29) remain non-negative.
              const sliderMin = index === 0 || index === 29 ? "0" : "-100";

              return (
                <div key={index} className="mb-6">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-700 dark:text-gray-300 font-medium">
                      {label}
                    </span>
                    <span className="text-gray-700 dark:text-gray-300">
                      {isHighPrecision && index !== 29
                        ? featuresData[index].toFixed(7)
                        : featuresData[index]}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={sliderMin}
                    max={sliderMax}
                    step={sliderStep}
                    value={featuresData[index]}
                    onChange={(e) =>
                      handleSliderChange(index, parseFloat(e.target.value))
                    }
                    className="w-full h-2 bg-gray-300 rounded-full appearance-none cursor-pointer accent-purple-600 transition"
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;
