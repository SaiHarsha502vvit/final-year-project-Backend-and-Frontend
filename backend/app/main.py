from fastapi import FastAPI, HTTPException
from app.models import model
from app.schemas import PredictionRequest, PredictionResponse
import numpy as np

app = FastAPI(
    title="XGBoost Prediction API",
    description="API for making predictions using a pre-trained XGBoost model.",
    version="1.0.0"
)

@app.get("/health", tags=["Health Check"])
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "OK"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Prediction endpoint that accepts feature data and returns the prediction and probability.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Convert the features list to a NumPy array and reshape for prediction
        input_features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)[0]
        proba = model.predict_proba(input_features)[0][1]  # Probability for class '1'

        return PredictionResponse(prediction=int(prediction), probability=float(proba))

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Input data error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")