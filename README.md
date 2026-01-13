# Stroke Prediction ML Service

An end-to-end production-ready machine learning system for stroke risk prediction, addressing the challenges of severe class imbalance (19:1 ratio) through advanced resampling techniques, medical-informed feature engineering, and threshold optimization.

[![Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-red?logo=youtube)](https://youtu.be/ZGAPtW54aSA)

## 🎯 Project Overview

This project demonstrates comprehensive ML engineering capabilities across the entire pipeline:
- **Problem**: Binary classification on highly imbalanced medical data (4.87% stroke cases)
- **Challenge**: Maximize recall (minimize missed stroke cases) while maintaining acceptable precision
- **Solution**: SMOTEENN + XGBoost + Threshold Optimization
- **Deployment**: FastAPI backend + Gradio UI for clinical use

## 🏆 Model Performance

### Best Model: XGBoost + SMOTEENN + Threshold Optimization

| Metric | Value | Clinical Impact |
|--------|-------|-----------------|
| **F1-Score** | **0.592** | Balanced performance for imbalanced data |
| **Recall** | **86.1%** | Detects 86 out of 100 stroke cases |
| **Precision** | **45.1%** | 45% of predicted strokes are true positives |
| **Optimized Threshold** | **0.803** | Tuned for maximum F1-score |

**Medical Significance:**
- Only 14% of stroke cases missed (vs 48% with default threshold)
- Suitable for early screening where false positives are acceptable
- 28% better F1-score than Random Forest baseline

### Comparison with Alternative Approaches

| Model | F1-Score | Recall | Precision | Notes |
|-------|----------|--------|-----------|-------|
| **XGBoost + SMOTEENN** | **0.592** | **86.1%** | 45.1% | ✅ Best overall |
| Random Forest + SMOTEENN | 0.457 | 51.7% | 40.9% | Good baseline |
| Logistic Regression (baseline) | 0.009 | 0.5% | 20.0% | ❌ Fails on imbalanced data |

## 🔬 Technical Strategy

### 1. Data Preprocessing Pipeline

**Challenge:** Raw medical data with missing values, categorical variables, and non-linear relationships.

**Solution:**
```python
# Row Filtering
- Remove 'Other' gender (n=1, statistically insignificant)
- Drop rows with missing BMI (3.9% of data)
- Clean dataset: 4,908 samples

# Column Normalization
- Standardize to lowercase snake_case
- Remove ID column (no predictive value)
```

### 2. Medical-Informed Feature Engineering

**Innovation:** Leverage clinical knowledge for intelligent binning instead of raw continuous values.

**Age Binning** (based on stroke risk stages):
```python
[<25, 25-44, 45-64, 65-79, 80+]
# Captures non-linear risk increase after age 60
```

**BMI Binning** (WHO standards):
```python
[Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)]
# Medical classification for cardiovascular risk
```

**Glucose Binning** (diabetes thresholds):
```python
[<70, 70-84, 85-99, 100-109, 110-125, 126-139, ≥140]
# Aligned with ADA diabetes diagnostic criteria
```

**Why binning?**
- ✅ Captures medical thresholds (e.g., glucose >126 = diabetes)
- ✅ Handles non-linear relationships (stroke risk accelerates after 65)
- ✅ Improves model interpretability for clinicians
- ✅ Reduces overfitting on continuous outliers

**Encoding Strategy:**
```python
ColumnTransformer([
    ('bins', bin_encoder, ['age', 'bmi', 'avg_glucose_level']),    # One-Hot
    ('num', StandardScaler(), numeric_features),                    # Standardization
    ('cat', OneHotEncoder(), categorical_features)                  # One-Hot
])
# Final feature space: ~45 dimensions
```

### 3. Imbalanced Data Handling

**Problem:** 95.13% no-stroke vs 4.87% stroke → Model bias toward majority class

**Solution: SMOTEENN (Hybrid Approach)**

Why SMOTEENN over alternatives?

| Method | Approach | Limitation | Use Case |
|--------|----------|------------|----------|
| Class Weights | Algorithm-level | Limited effectiveness for severe imbalance | Mild imbalance (5:1) |
| SMOTE | Over-sample minority | May create noise near decision boundary | Moderate imbalance |
| ENN | Under-sample majority | Removes borderline majority samples | Cleaning noisy data |
| **SMOTEENN** ✅ | **Hybrid: SMOTE + ENN** | **Best of both worlds** | **Severe imbalance (19:1)** |

**SMOTEENN Workflow:**
1. **SMOTE**: Synthesize minority samples via k-NN interpolation
2. **ENN**: Remove noisy samples from both classes
3. **Result**: Balanced dataset with cleaner decision boundaries

**Impact:**
- F1-Score improvement: 0.009 → 0.592 (65× improvement)
- Recall improvement: 0.5% → 86.1% (172× improvement)

### 4. Model Selection Strategy

**Approach:** Systematic comparison using nested cross-validation

**Why Tree-Based Models?**
- ✅ Handle non-linear relationships (age × hypertension interactions)
- ✅ Robust to outliers (extreme BMI values)
- ✅ Feature importance for clinical interpretability
- ✅ No need for extensive feature scaling (categorical + numeric mix)

**XGBoost Advantages over Random Forest:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=5,      # 🔑 Critical for imbalance
    random_state=42,
    eval_metric='auc'
)
```

**Key Parameters:**
- `scale_pos_weight=5`: Penalizes minority class misclassification (≈ imbalance_ratio/4)
- `max_depth=5`: Prevents overfitting on small minority class
- `learning_rate=0.1`: Conservative to avoid overfitting

**Why XGBoost > Random Forest?**
1. **Boosting** learns from previous errors → better on hard cases
2. **Regularization** (L1/L2) prevents overfitting
3. **scale_pos_weight** explicitly handles imbalance

### 5. Threshold Optimization

**Problem:** Default threshold (0.5) optimized for balanced data, not imbalanced medical scenarios.

**Solution:** Precision-Recall Curve Optimization

```python
# Find threshold that maximizes F1-score
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[argmax(f1_scores)]  # 0.803
```

**Impact of Threshold Tuning:**

| Threshold | Recall | Precision | F1-Score | Clinical Tradeoff |
|-----------|--------|-----------|----------|-------------------|
| 0.5 (default) | 84.2% | 21.7% | 0.345 | Too many false positives |
| **0.803 (optimal)** | **86.1%** | **45.1%** | **0.592** | ✅ Best balance |
| 0.9 (conservative) | 62.0% | 68.0% | 0.648 | Misses too many strokes |

**Why 0.803?**
- Prioritizes **recall** (detect strokes) over precision (avoid false alarms)
- Medical ethics: Missing a stroke (false negative) is worse than unnecessary testing (false positive)
- Achieves **86% recall** while maintaining **45% precision** (acceptable for screening)

### 6. Model Validation Strategy

**Nested Cross-Validation:**
```python
Outer CV: 5-fold StratifiedKFold  # Unbiased performance estimate
Inner CV: 4-fold StratifiedKFold  # Hyperparameter tuning
```

**Why Nested CV?**
- ✅ Prevents overfitting during hyperparameter search
- ✅ Stratified folds preserve class distribution in each fold
- ✅ Realistic estimate of production performance

**Evaluation Metrics:**
- ❌ **Accuracy** (95.7%): Misleading for imbalanced data
- ✅ **Recall** (86.1%): Critical for healthcare (minimize missed cases)
- ✅ **Precision** (45.1%): Control false positives
- ✅ **F1-Score** (0.592): Harmonic mean for balance
- ✅ **AUC-ROC** (0.809): Threshold-independent performance

### 7. Production Pipeline Design

**End-to-End Automation:**
```python
ImbPipeline([
    ('preprocessor', ColumnTransformer(...)),  # Feature engineering
    ('sampler', SMOTEENN()),                   # Imbalance handling
    ('classifier', XGBClassifier(...))         # Model
])
```

**Saved Artifact:**
```python
{
    'pipeline': complete_pipeline,      # Preprocessing + SMOTEENN + Model
    'threshold': 0.803,                 # Optimized threshold
    'model_type': 'xgb'                # Metadata
}
```

**Benefits:**
- ✅ Single `.pkl` file for deployment
- ✅ No preprocessing drift (train/serve consistency)
- ✅ Versioned and reproducible

## 🛠️ Features

-   **Advanced Feature Engineering**: Medical-informed binning based on clinical thresholds
-   **Imbalanced Data Handling**: SMOTEENN (hybrid over/under-sampling) for 19:1 class imbalance
-   **Model Optimization**: Threshold tuning via Precision-Recall curve analysis
-   **Production Pipeline**: Unified artifact with preprocessing + sampling + model
-   **Web UI**: Gradio interface for clinician-friendly predictions
-   **REST API**: FastAPI service for system integration
-   **Model Interpretability**: SHAP values and feature importance
-   **Automated Testing**: Unit tests and CI/CD via GitHub Actions

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
