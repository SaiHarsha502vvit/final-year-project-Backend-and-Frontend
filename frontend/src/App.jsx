// frontend/src/App.jsx
import React from 'react';
import Header from './components/Header.jsx';
import PredictionForm from './components/PredictionForm.jsx';

const App = () => {
  return (
    <div className="container mx-auto p-6">
      <Header />
      <div className="flex justify-center">
        <div className="w-full md:w-1/2">
          <PredictionForm />
        </div>
      </div>
    </div>
  );
};

export default App;