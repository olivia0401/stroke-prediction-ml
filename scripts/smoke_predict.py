"""Manual smoke test: load models/model.pkl and score a few example patients.

Run after training:  python scripts/smoke_predict.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.predictor import StrokePredictor  # noqa: E402

CASES = [
    ("Male 67, heart disease, high glucose, high BMI",
     dict(gender="Male", age=67, hypertension=0, heart_disease=1, ever_married="Yes",
          work_type="Private", residence_type="Urban", avg_glucose_level=228.69,
          bmi=36.6, smoking_status="formerly smoked")),
    ("Female 30, normal indicators",
     dict(gender="Female", age=30, hypertension=0, heart_disease=0, ever_married="No",
          work_type="Private", residence_type="Urban", avg_glucose_level=85.0,
          bmi=22.5, smoking_status="never smoked")),
    ("Female 45, hypertension",
     dict(gender="Female", age=45, hypertension=1, heart_disease=0, ever_married="Yes",
          work_type="Self-employed", residence_type="Rural", avg_glucose_level=180.5,
          bmi=28.3, smoking_status="never smoked")),
    ("Male 80, hypertension + heart disease, smoker",
     dict(gender="Male", age=80, hypertension=1, heart_disease=1, ever_married="Yes",
          work_type="Private", residence_type="Urban", avg_glucose_level=200.0,
          bmi=30.0, smoking_status="smokes")),
]


def main():
    predictor = StrokePredictor(artifact_path=str(ROOT / "models" / "model.pkl"))
    print(f"Model: {predictor.model_type} | threshold: {predictor.threshold:.3f}")
    for name, data in CASES:
        df = pd.DataFrame([data])
        label = "High risk" if predictor.predict(df)[0] == 1 else "Low risk"
        prob = float(predictor.predict_proba(df)[0][1]) * 100
        print(f"- {name}: {label} (stroke probability {prob:.2f}%)")


if __name__ == "__main__":
    main()
