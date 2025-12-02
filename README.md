# Stroke Prediction - Production ML System

A **production-ready machine learning system** demonstrating end-to-end ML engineering capabilities.

## 🎯 Engineering Capabilities

This project showcases:

✅ **Data Pipeline** - ETL, preprocessing, feature engineering  
✅ **Model Training** - RandomForest/XGBoost with hyperparameters  
✅ **Experiment Tracking** - MLflow integration  
✅ **Model Persistence** - joblib serialization  
✅ **CLI Interface** - argparse command-line tool  
✅ **REST API** - FastAPI production service  
✅ **Web UI** - Gradio interactive dashboard  
✅ **Unit Testing** - pytest test suite  
✅ **Containerization** - Docker deployment  
✅ **CI/CD** - GitHub Actions automation  
✅ **Logging** - Structured logging system  

## 📦 Quick Start

### 1. Setup
```bash
cd ~/stroke-prediction-ml
pip3 install -r requirements.txt
```

### 2. Train Model
```bash
./run.sh train
```

### 3. Run Tests
```bash
./run.sh test
```

### 4. Start API
```bash
./run.sh api
```

### 5. Test Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
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

## 📂 Project Structure

```
stroke-prediction-ml/
├── src/               # Core ML modules (8 files, 325 LOC)
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── trainer.py     # MLflow tracking
│   └── predictor.py
├── scripts/train.py   # CLI interface
├── api/app.py         # FastAPI service
├── ui/gradio_app.py   # Web dashboard
├── tests/             # pytest tests
├── docker/            # Dockerfiles
└── .github/workflows/ # CI/CD
```

## 📊 Model Performance

- **Algorithm**: RandomForest (n_estimators=200)
- **F1 Score**: 0.206
- **Precision**: 0.213
- **Recall**: 0.200

## 🛠️ Technology Stack

**ML/Data**: pandas, scikit-learn, xgboost  
**Tracking**: MLflow  
**API**: FastAPI, uvicorn  
**UI**: Gradio  
**Testing**: pytest  
**DevOps**: Docker, GitHub Actions
