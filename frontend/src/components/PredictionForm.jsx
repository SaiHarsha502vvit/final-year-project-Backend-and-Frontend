import React, { useState } from "react";

// Create labels with 30 elements: "Time", then V1...V28, then "Amount".
const labels = [
  "Time",
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  "Amount",
];

const PredictionForm = () => {
  // Initialize with 30 zeros.
  const initialArray = new Array(30).fill(0);
  const initialText = initialArray.join(", ");
  const [features, setFeatures] = useState(initialText);
  const [featuresData, setFeaturesData] = useState(initialArray);
  const [result, setResult] = useState(null);

  const handleFeaturesChange = (e) => {
    const text = e.target.value;
    setFeatures(text);
    const parts = text.split(",").map((s) => parseFloat(s.trim()));
    if (parts.length === 30 && !parts.some(isNaN)) {
      setFeaturesData(parts);
    }
  };

  const handleSliderChange = (index, value) => {
    // For "Time" and "Amount", force non-negative values.
    if ((index === 0 || index === 29) && value < 0) {
      value = 0;
    }
    const updated = [...featuresData];
    updated[index] = value;
    setFeaturesData(updated);
    setFeatures(updated.join(", "));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (featuresData.length !== 30 || featuresData.some(isNaN)) {
      alert("Please enter exactly 30 numerical values, separated by commas.");
      return;
    }

    if (featuresData[0] < 0 || featuresData[29] < 0) {
      alert("Time and Amount cannot be negative.");
      return;
    }

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
    }
  };

  const handleClear = () => {
    setFeatures(initialText);
    setFeaturesData(initialArray);
    setResult(null);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <form onSubmit={handleSubmit}>
        <div className="flex flex-col md:flex-row gap-4">
          {/* Left side: Textarea and buttons */}
          <div className="flex-1">
            <div className="mb-4">
              <label htmlFor="features" className="block text-gray-800 font-medium mb-2">
                Enter 30 Features (comma-separated)
              </label>
              <textarea
                id="features"
                rows="10"
                value={features}
                onChange={handleFeaturesChange}
                className="w-full border border-gray-300 rounded-lg p-3 resize-none transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
                placeholder="e.g. 0.1, 0.2, 0.3, ... , 3.0"
              ></textarea>
              <p className="text-xs text-gray-500 mt-1">
                Provide exactly 30 comma-separated numerical values.
              </p>
            </div>
            <div className="flex gap-4">
              <button
                type="submit"
                className="bg-blue-500 hover:bg-green-600 text-white py-2 px-4 rounded transition duration-200 ease-in-out flex items-center justify-center"
              >
                <svg
                  className="inline-block w-5 h-5 mr-2"
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
              </button>
              <button
                type="button"
                onClick={handleClear}
                className="bg-gray-500 hover:bg-gray-600 text-white py-2 px-4 rounded transition duration-200 ease-in-out"
              >
                Clear
              </button>
            </div>
          </div>
          {/* Right side: Live sliders */}
          <div className="flex-1 overflow-y-auto max-h-[500px] p-2 border border-gray-200 rounded">
            <h3 className="text-gray-800 font-medium mb-2">Live Feature Adjustment</h3>
            {labels.map((label, index) => {
              const isHighPrecision = index >= 1 && index <= 28;
              // For Amount (index 29), allow values up to 1,000,000.
              const sliderMax = index === 29 ? "1000000" : "100";
              const sliderStep =
                index === 29 ? "1" : isHighPrecision ? "0.000001" : "0.1";

              return (
                <div key={index} className="mb-4">
                  <div className="flex justify-between mb-1">
                    <span className="text-gray-700">{label}</span>
                    <span className="text-gray-700">
                      {isHighPrecision && index !== 29
                        ? featuresData[index].toFixed(7)
                        : featuresData[index]}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={index === 0 || index === 29 ? "0" : "-100"}
                    max={sliderMax}
                    step={sliderStep}
                    value={featuresData[index]}
                    onChange={(e) =>
                      handleSliderChange(index, parseFloat(e.target.value))
                    }
                    className="w-full"
                  />
                </div>
              );
            })}
          </div>
        </div>
      </form>
      {result && (
        <div
          className={
            result.prediction === 1
              ? "mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded"
              : "mt-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded"
          }
          role="alert"
        >
          {result.prediction === 1 ? (
            <>
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              <p>
                <strong>Probability:</strong> {result.probability.toFixed(4)}
              </p>
              <p>
                <strong>Fraud Transaction Detected!</strong>
              </p>
              <p>This transaction is flagged as fraud.</p>
            </>
          ) : (
            <>
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              <p>
                <strong>Probability:</strong> {result.probability.toFixed(4)}
              </p>
              <p>
                <strong>Transaction is Not Fraud</strong>
              </p>
              <p>Can Allow this Transaction</p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
