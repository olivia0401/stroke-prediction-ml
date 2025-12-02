"""FastAPI Prediction Service"""
from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predictor import StrokePredictor

app = FastAPI(title="Stroke Prediction API")

# Load model
predictor = StrokePredictor(
    model_path="models/model.pkl",
    scaler_path="models/scaler.pkl",
    encoders_path="models/encoders.pkl"
)


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
    input_dict = data.dict()
    prediction = predictor.predict(input_dict)[0]
    proba = predictor.predict_proba(input_dict)[0]

    return {
        "prediction": int(prediction),
        "risk_probability": float(proba[1]),
        "risk_level": "High" if proba[1] > 0.5 else "Low"
    }
