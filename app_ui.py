import gradio as gr
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import StrokePredictor

predictor = StrokePredictor(artifact_path="models/model.pkl")

def predict_stroke(gender, age, hypertension, heart_disease, ever_married,
                   work_type, residence_type, avg_glucose_level, bmi, smoking_status):

    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    input_df = pd.DataFrame([input_data])
    prediction = predictor.predict(input_df)[0]
    proba = predictor.predict_proba(input_df)[0]

    risk_prob = float(proba[1]) * 100
    result = "High Risk" if prediction == 1 else "Low Risk"

    return f"Prediction: {result}\nStroke Probability: {risk_prob:.2f}%"

with gr.Blocks(title="Stroke Prediction System") as demo:
    gr.Markdown("# Stroke Risk Prediction System")
    gr.Markdown("Enter patient information to predict stroke risk")

    with gr.Row():
        with gr.Column():
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender", value="Male")
            age = gr.Number(label="Age", value=50)
            hypertension = gr.Radio([0, 1], label="Hypertension (0=No, 1=Yes)", value=0)
            heart_disease = gr.Radio([0, 1], label="Heart Disease (0=No, 1=Yes)", value=0)
            ever_married = gr.Radio(["Yes", "No"], label="Ever Married", value="Yes")

        with gr.Column():
            work_type = gr.Dropdown(
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                label="Work Type",
                value="Private"
            )
            residence_type = gr.Radio(["Urban", "Rural"], label="Residence Type", value="Urban")
            avg_glucose_level = gr.Number(label="Average Glucose Level", value=100)
            bmi = gr.Number(label="BMI", value=25)
            smoking_status = gr.Dropdown(
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                label="Smoking Status",
                value="never smoked"
            )

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Prediction Result", lines=3)

    predict_btn.click(
        fn=predict_stroke,
        inputs=[gender, age, hypertension, heart_disease, ever_married,
                work_type, residence_type, avg_glucose_level, bmi, smoking_status],
        outputs=output
    )

    gr.Markdown("---")
    gr.Markdown("### Example Data")
    gr.Examples(
        examples=[
            ["Male", 67, 0, 1, "Yes", "Private", "Urban", 228.69, 36.6, "formerly smoked"],
            ["Female", 45, 1, 0, "Yes", "Self-employed", "Rural", 180.5, 28.3, "never smoked"],
            ["Male", 80, 1, 1, "Yes", "Private", "Urban", 200.0, 30.0, "smokes"]
        ],
        inputs=[gender, age, hypertension, heart_disease, ever_married,
                work_type, residence_type, avg_glucose_level, bmi, smoking_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
