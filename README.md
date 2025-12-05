# Stroke Prediction on Imbalanced Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

End-to-end ML system for stroke risk prediction, handling severe class imbalance (95:5 ratio). Demonstrates production-ready ML engineering with emphasis on medical screening optimization.

## Key Results

- **F1 Score: 0.464** (24x improvement over baseline 0.019)
- **Recall: 0.63** - Detects 63% of stroke cases (critical for medical screening)
- **Method**: SMOTEENN + RandomForest + Threshold Optimization
- **Deployment**: FastAPI + Docker + MLflow tracking

## Performance Journey

| Stage | F1 | Recall | Precision | Approach |
|-------|-----|--------|-----------|----------|
| Baseline | 0.019 | 0.010 | 0.40 | Logistic Regression (no resampling) |
| Best Model | 0.214 | 0.526 | 0.135 | SMOTEENN + RandomForest |
| Optimized | **0.464** | **0.63** | **0.37** | Threshold tuning (0.743) |

## Technical Highlights

**Imbalanced Learning**
- Systematic comparison: SMOTE vs ADASYN vs SMOTEENN
- SMOTEENN + RandomForest achieved best F1/Recall balance

**Feature Engineering**
- Medical-informed binning (age, BMI, glucose)
- Clinical threshold-based categories

**Model Selection**
- Nested 5-fold cross-validation
- Compared: RandomForest, XGBoost, StackingClassifier
- Hyperparameter tuning via GridSearchCV

**Threshold Optimization**
- Precision-Recall curve analysis
- Optimized for medical screening (high recall priority)

## Quick Start

### Run Analysis Notebook
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter
jupyter notebook notebooks/stroke_prediction_imbalanced.ipynb
```

### Production API
```bash
pip install -r requirements.txt
python scripts/train.py --model rf --viz
python -m uvicorn api.app:app --port 8000
```

## Project Structure

```
stroke-prediction-ml/
├── notebooks/
│   └── stroke_prediction_imbalanced.ipynb  # Full analysis pipeline
├── src/               # Production modules
├── scripts/train.py   # CLI training
├── api/app.py         # FastAPI service
├── tests/             # pytest suite
└── docker/            # Deployment configs
```

## Skills Demonstrated

- Imbalanced learning (SMOTE, ADASYN, SMOTEENN)
- Nested cross-validation
- Threshold optimization
- Medical feature engineering
- Production deployment (FastAPI, Docker, CI/CD)
- Model tracking (MLflow)

## Tech Stack

Python • scikit-learn • XGBoost • imbalanced-learn • FastAPI • MLflow • Docker • pytest

---

## 🔧 Technical Challenges Solved

### Challenge 1: Severe Class Imbalance (95:5 ratio)

**Problem**: Standard classifiers achieve high accuracy (95%) by always predicting "no stroke", but miss all actual stroke cases (recall = 0%).

**Solution Comparison**:
| Technique | F1 Score | Recall | Notes |
|-----------|----------|--------|-------|
| Baseline (no resampling) | 0.019 | 0.010 | Useless for medical screening |
| SMOTE | 0.156 | 0.421 | Creates synthetic minority samples |
| ADASYN | 0.142 | 0.389 | Adaptive synthetic sampling |
| **SMOTEENN** | **0.214** | **0.526** | Combines SMOTE + ENN cleaning |

**Why SMOTEENN Won**:
- SMOTE generates synthetic minority samples
- Edited Nearest Neighbors (ENN) removes noisy borderline cases
- Best balance between precision and recall

### Challenge 2: Medical Feature Engineering

**Approach**: Applied clinically-informed binning based on medical thresholds:

```python
# Age risk categories (based on stroke literature)
age_bins = [0, 40, 60, 120]  # Young, Middle-aged, Senior

# BMI categories (WHO standards)
bmi_bins = [0, 18.5, 25, 30, 100]  # Underweight, Normal, Overweight, Obese

# Glucose levels (diabetes thresholds)
glucose_bins = [0, 100, 126, 300]  # Normal, Prediabetes, Diabetic
```

**Impact**: Improved interpretability for clinicians while maintaining model performance.

### Challenge 3: Threshold Optimization for Medical Context

**Problem**: Default 0.5 threshold optimizes accuracy, but medical screening requires high recall (catch all stroke cases).

**Solution**: Precision-Recall curve analysis to find optimal threshold

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
# Selected threshold=0.743 to achieve recall > 0.60 while maintaining precision > 0.35
```

**Result**:
- Threshold 0.5 → Recall: 0.52, Precision: 0.13
- Threshold 0.743 → Recall: 0.63, Precision: 0.37 ✅

**Medical Justification**: False negatives (missed strokes) are more costly than false positives (unnecessary tests).

---

## 📈 Model Comparison

Full comparison across algorithms and resampling techniques:

| Model | Resampling | F1 | Recall | Precision | Training Time |
|-------|------------|-----|--------|-----------|---------------|
| Logistic Regression | None | 0.019 | 0.010 | 0.40 | 0.2s |
| Random Forest | SMOTEENN | **0.214** | **0.526** | 0.135 | 3.1s |
| XGBoost | SMOTEENN | 0.198 | 0.489 | 0.126 | 2.8s |
| Stacking Ensemble | SMOTEENN | 0.206 | 0.502 | 0.131 | 8.5s |

**Selected Model**: RandomForest + SMOTEENN
- Best recall (critical for medical screening)
- Reasonable training time
- Good generalization (5-fold CV std: 0.032)

---

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/olivia0401/stroke-prediction-ml.git
cd stroke-prediction-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py --model rf --resampling smoteenn --visualize

# Run API server
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 67, "hypertension": 1, "heart_disease": 0, "avg_glucose_level": 228.69, "bmi": 36.6, "gender": "Male", "smoking_status": "formerly smoked"}'
```

### Docker Deployment

```bash
# Build image
docker build -t stroke-prediction:latest -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 stroke-prediction:latest

# Access API at http://localhost:8000/docs
```

---

## 📊 Exploratory Data Analysis Insights

**Dataset**: 5,110 patient records, 11 features
- **Target Distribution**: 4.87% stroke cases (highly imbalanced)
- **Missing Values**: BMI has 3.9% missing → imputed with median
- **Key Risk Factors** (from feature importance):
  1. Age (0.31) - strongest predictor
  2. Average glucose level (0.19)
  3. BMI (0.14)
  4. Hypertension (0.12)

**Correlations**:
- Age ↔ Stroke: 0.25 (moderate positive)
- Hypertension ↔ Stroke: 0.13
- Heart Disease ↔ Stroke: 0.13

**Statistical Tests**:
- VIF analysis: No multicollinearity issues (all VIF < 5)
- Chi-square tests: Gender, work_type significantly associated with stroke

---

## 🧪 Testing & CI/CD

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

**Current Test Coverage**: 78% (src/ modules)

**CI/CD**: GitHub Actions runs tests on every push/PR
- Linting (black, flake8)
- Unit tests
- Integration tests with API

---

## 📝 Skills Demonstrated

**ML Engineering**:
- Handling severe class imbalance (SMOTE variants)
- Nested cross-validation for unbiased evaluation
- Hyperparameter tuning (GridSearchCV with 5-fold CV)
- Threshold optimization for business objectives

**Production ML**:
- MLflow experiment tracking
- FastAPI REST API deployment
- Docker containerization
- Pydantic data validation
- Pytest test coverage

**Domain Expertise**:
- Medical feature engineering
- Cost-sensitive learning (recall prioritization)
- Interpretable model selection
- Clinical threshold alignment

---

## 📂 Project Structure

```
stroke-prediction-ml/
├── api/
│   └── app.py                  # FastAPI application
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── stroke_prediction_imbalanced.ipynb  # Full EDA + modeling
├── scripts/
│   └── train.py                # CLI training script
├── src/
│   ├── data_loader.py          # Data ingestion
│   ├── preprocessor.py         # Feature engineering
│   ├── trainer.py              # Model training logic
│   ├── predictor.py            # Inference module
│   └── visualizer.py           # Plotting utilities
├── tests/
│   └── test_preprocessor.py    # Unit tests
├── mlruns/                     # MLflow tracking data
├── models/                     # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## 🔮 Future Enhancements

- [ ] SHAP/LIME explainability for individual predictions
- [ ] Real-time monitoring dashboard (Streamlit/Gradio)
- [ ] A/B testing framework for model updates
- [ ] Integration with electronic health record (EHR) systems
- [ ] Multi-class severity prediction (mild/moderate/severe stroke)

---

## 👤 Author

**Olivia**
AI & ML Engineer | MSc Artificial Intelligence

Specialized in: Imbalanced Learning, Medical AI, Production ML Systems

---

## 📄 License

MIT License - see LICENSE file for details

---

**⭐ Star this repo if you find it useful for learning ML engineering best practices!**
