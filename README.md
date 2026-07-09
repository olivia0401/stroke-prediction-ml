# Stroke Prediction ML Service

An end-to-end production-ready machine learning system for stroke risk prediction, addressing the challenges of severe class imbalance (19:1 ratio) through advanced resampling techniques, medical-informed feature engineering, and threshold optimization.

[![Demo Video](https://img.youtube.com/vi/ZGAPtW54aSA/hqdefault.jpg)](https://www.youtube.com/watch?v=ZGAPtW54aSA)


## 🎯 Project Overview

This project demonstrates comprehensive ML engineering capabilities across the entire pipeline:
- **Problem**: Binary classification on highly imbalanced medical data (4.87% stroke cases)
- **Challenge**: Maximize recall (minimize missed stroke cases) while maintaining acceptable precision
- **Solution**: SMOTEENN + XGBoost + Threshold Optimization
- **Deployment**: FastAPI backend + Gradio UI for clinical use

## 🏆 Model Performance

> **How these numbers are measured.** All metrics below are **held-out estimates
> from 5-fold stratified cross-validation** (mean ± standard deviation across
> folds). Within each fold the model is fit on the training split and the
> decision threshold is tuned on that same training split, then both are scored
> on the untouched validation split. This avoids the optimistic bias you get
> from scoring on the data used to fit the model and tune the threshold.
> Numbers are reproducible with `python3 scripts/train.py --model xgb`.

### Best Model: XGBoost + SMOTEENN + Threshold Optimization

| Metric | Value (5-fold CV) | Clinical Impact |
|--------|-------------------|-----------------|
| **F1-Score** | **0.22 ± 0.02** | Modest at the F1-optimal threshold under severe imbalance |
| **Recall** | **0.29 ± 0.05** | Detects ~29 out of 100 stroke cases at this operating point |
| **Precision** | **0.18 ± 0.02** | ~18% of predicted strokes are true positives |
| **AUC-ROC** | **0.79 ± 0.02** | Solid threshold-independent ranking ability |

The **AUC-ROC of ~0.79** shows the model has real discriminative power; the low
F1/precision reflect the genuine difficulty of a 19:1 imbalanced screening task,
where the F1-maximizing threshold still admits many false positives. The
operating threshold can be shifted along the precision–recall curve to trade
recall for precision depending on the clinical use case.

### Comparison of Models (5-fold CV)

| Model | F1-Score | Recall | Precision | AUC-ROC |
|-------|----------|--------|-----------|---------|
| **XGBoost + SMOTEENN** | **0.22 ± 0.02** | **0.29 ± 0.05** | 0.18 ± 0.02 | 0.79 ± 0.02 |
| Random Forest + SMOTEENN | 0.20 ± 0.06 | 0.22 ± 0.09 | 0.19 ± 0.05 | 0.81 ± 0.01 |

> **Note on earlier figures.** Prior versions of this README reported much
> higher numbers (e.g. F1 ≈ 0.59, recall ≈ 86%). Those were computed on the
> same data used to fit the model and tune the threshold, so they overstated
> real-world performance. The table above replaces them with honest
> cross-validated estimates.

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

> **Important:** SMOTEENN is applied **inside** the cross-validation loop —
> only to each fold's training split, never to the held-out validation split.
> Resampling before splitting is a classic source of leakage; keeping it inside
> the pipeline (via `imblearn.pipeline`) prevents synthetic minority samples
> from leaking into evaluation.

### 4. Model Selection Strategy

**Approach:** Comparison of tree-based models with fixed, pre-selected
hyperparameters (carried over from earlier notebook experimentation), evaluated
with stratified 5-fold cross-validation. Hyperparameters are **not** re-searched
in this pipeline — see the validation section below for exactly what the CV
measures.

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
optimal_threshold = thresholds[argmax(f1_scores)]
```

**Leakage-safe thresholding.** The threshold is tuned **only on training data**
— inside each CV fold during evaluation, and on the full training set for the
deployed model. It is never fit on data used to report performance. Because the
positive class is rare, the F1-optimal threshold typically lands well above 0.5.

**Why tune the threshold at all?**
- Prioritizes **recall** (detect strokes) over precision (avoid false alarms)
- Medical ethics: Missing a stroke (false negative) is worse than an
  unnecessary follow-up test (false positive)
- The operating point can be moved along the precision–recall curve to match
  the tolerance for false positives in a given screening context

### 6. Model Validation Strategy

**Stratified K-Fold Cross-Validation (leakage-safe):**
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Per fold:
#   1. Fit pipeline (preprocess -> SMOTEENN -> model) on the TRAIN split only
#   2. Tune the decision threshold on the TRAIN split only
#   3. Score F1 / precision / recall / AUC on the untouched VALIDATION split
# Report the mean ± std of each metric across folds.
```

**Why this design?**
- ✅ Preprocessing, resampling, and threshold tuning all happen **inside** each
  fold, so nothing from the validation split leaks into fitting
- ✅ Stratified folds preserve the ~5% stroke rate in each split
- ✅ Produces a realistic, honest estimate of production performance

The final deployed pipeline is then re-fit on the **entire** dataset (with its
threshold tuned on the full data). The reported metrics remain the
cross-validated estimates above, not in-sample scores.

**Evaluation Metrics (why these):**
- ❌ **Accuracy**: Misleading for imbalanced data (predicting "no stroke" scores ~95%)
- ✅ **Recall**: Critical for healthcare (minimize missed cases)
- ✅ **Precision**: Control false positives
- ✅ **F1-Score**: Harmonic mean for balance at the operating threshold
- ✅ **AUC-ROC**: Threshold-independent ranking quality

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
    'threshold': best_threshold,        # Optimized decision threshold
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
