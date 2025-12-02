# Stroke Prediction on Imbalanced Data

End-to-end ML system for stroke risk prediction, handling severe class imbalance (95:5 ratio).

## Key Results

- **F1 Score: 0.464** (24x improvement over baseline)
- **Recall: 0.63** - Detects 63% of stroke cases
- **Method**: SMOTEENN + RandomForest + Threshold Optimization

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
