# Stroke Prediction ML Service

This project is an end-to-end machine learning application designed to predict the likelihood of a patient having a stroke. It covers the complete ML lifecycle, from data preprocessing and model training to serving predictions via a REST API.

## Features

-   **Data Preprocessing:** Implements a robust data cleaning and feature engineering pipeline, including medical-informed binning for features like Age, BMI, and Average Glucose Level.
-   **Imbalanced Data Handling:** Utilizes `SMOTEENN` (a combination of over- and under-sampling) to effectively handle the severe class imbalance in the stroke dataset.
-   **Model Training:** Trains a high-performance classifier (Random Forest or XGBoost) and optimizes the classification threshold to maximize the F1-score, which is critical for imbalanced datasets.
-   **Unified Model Artifact:** The entire preprocessing pipeline, sampler, and trained model are saved as a single, portable `model.pkl` file, simplifying deployment.
-   **Web UI:** A user-friendly Gradio interface for interactive stroke risk prediction.
-   **REST API:** A FastAPI service exposes the prediction logic, allowing for easy integration with other applications. The API is compatible with both Pydantic v1 and v2.
-   **CI/CD:** A GitHub Actions workflow is included to automate testing and ensure code quality.

## Project Structure

```
.
├── api/
│   └── app.py            # FastAPI service
├── data/
│   └── stroke-data.csv   # Dataset
├── models/
│   └── model.pkl         # Saved model artifact (generated after training)
├── scripts/
│   └── train.py          # CLI script to train the model
├── src/
│   ├── data_loader.py    # Data loading utility
│   ├── preprocessor.py   # Preprocessing and feature engineering pipeline
│   ├── trainer.py        # Model training and threshold optimization logic
│   └── predictor.py      # Prediction logic using the saved model
├── tests/
│   └── test_preprocessor.py # Unit tests for the preprocessing pipeline
├── app_ui.py             # Gradio web interface
├── test_ui.py            # UI testing script
└── requirements.txt      # Project dependencies
```

## How to Use

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Unit Tests (Optional):**
    Verify that the preprocessing pipeline works as expected.
    ```bash
    pytest -v
    ```

3.  **Train the Model:**
    This script will load the data, preprocess it, train the model, and save the final artifact to `models/model.pkl`.
    ```bash
    # Train a Random Forest model (default)
    python3 scripts/train.py --model rf

    # Or train an XGBoost model
    python3 scripts/train.py --model xgb
    ```

4.  **Run the Web UI (Recommended):**
    Launch the interactive Gradio interface for easy predictions.
    ```bash
    python3 app_ui.py
    ```
    Then open your browser and navigate to `http://localhost:7860`

5.  **Run the API Service (Alternative):**
    This command starts the FastAPI server.
    ```bash
    uvicorn api.app:app --host 0.0.0.0 --port 8000
    ```

6.  **Make a Prediction via API:**
    Once the server is running, you can send a `POST` request to the `/predict` endpoint. You can also access the interactive API documentation at `http://localhost:8000/docs`.

    **Example `curl` request:**
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
            "gender": "Male",
            "age": 67,
            "hypertension": 0,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "smoking_status": "formerly smoked"
          }'
    ```

## Web UI Features

The Gradio web interface (`app_ui.py`) provides:

-   **Interactive Form**: Easy-to-use input fields for all patient parameters
-   **Real-time Prediction**: Instant stroke risk assessment with probability scores
-   **Example Data**: Pre-loaded test cases for quick demonstration
-   **Responsive Design**: Clean, professional interface accessible from any browser

**Input Parameters:**
- Gender (Male/Female/Other)
- Age
- Hypertension (0=No, 1=Yes)
- Heart Disease (0=No, 1=Yes)
- Ever Married (Yes/No)
- Work Type (Private/Self-employed/Govt_job/children/Never_worked)
- Residence Type (Urban/Rural)
- Average Glucose Level
- BMI
- Smoking Status (never smoked/formerly smoked/smokes/Unknown)

**Output:**
- Prediction: High Risk / Low Risk
- Stroke Probability: Percentage score
