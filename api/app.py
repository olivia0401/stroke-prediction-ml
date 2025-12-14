"""FastAPI Prediction Service"""
from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
import pandas as pd  # Add this line

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predictor import StrokePredictor

app = FastAPI(title="Stroke Prediction API")

# Load model
# This now loads a single artifact containing the entire pipeline and threshold
predictor = StrokePredictor(artifact_path="models/model.pkl")


class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.get("/")
def root():
    return {"message": "Stroke Prediction API", "status": "running"}


@app.post("/predict")
def predict(data: PatientData):
    """Predict stroke risk"""
    # Use compatible way to get dict from Pydantic model
    input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # The predictor's `predict` method now applies the optimized threshold
    prediction = predictor.predict(input_df)[0]
    
    # The predictor's `predict_proba` returns probabilities for all classes
    proba = predictor.predict_proba(input_df)[0]

    return {
        "prediction": int(prediction),
        "risk_probability": float(proba[1]),
        "risk_level": "High" if prediction == 1 else "Low"
    }
