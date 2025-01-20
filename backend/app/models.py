import joblib
import os

# Define the path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgb_classifier.joblib')

# Load the XGBoost model
try:
    model = joblib.load(MODEL_PATH)
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None