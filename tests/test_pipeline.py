"""End-to-end smoke test: train on a real-data subset, save, reload, predict."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.predictor import StrokePredictor  # noqa: E402
from src.preprocessor import preprocess_data  # noqa: E402
from src.trainer import ModelTrainer  # noqa: E402
from src.exceptions import ModelNotFittedError  # noqa: E402

DATA = ROOT / 'data' / 'stroke-data.csv'


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    df = pd.read_csv(DATA)
    # stratified subset keeps the test fast while keeping ~5% positives
    df, _ = train_test_split(df, train_size=0.3, stratify=df['stroke'], random_state=0)
    X, y, pre = preprocess_data(df)
    trainer = ModelTrainer(model_type='xgb', use_mlflow=False)
    metrics = trainer.train(X, y, preprocessor=pre, cv_folds=3)
    path = tmp_path_factory.mktemp('m') / 'model.pkl'
    trainer.save_model(str(path))
    return trainer, metrics, path


def test_cv_metrics_are_valid(trained):
    _, metrics, _ = trained
    for key in ('f1_score', 'precision', 'recall', 'roc_auc'):
        assert 0.0 <= metrics[key] <= 1.0
        assert f'{key}_std' in metrics
    assert 0.0 < metrics['threshold'] < 1.0


def test_saved_artifact_serves_raw_api_payload(trained):
    _, _, path = trained
    predictor = StrokePredictor(artifact_path=str(path))
    # API payloads use the raw CSV casing ("Residence_type"); the predictor must cope
    payload = pd.DataFrame([dict(
        gender='Male', age=67, hypertension=0, heart_disease=1, ever_married='Yes',
        work_type='Private', Residence_type='Urban', avg_glucose_level=228.69,
        bmi=36.6, smoking_status='formerly smoked')])
    pred = predictor.predict(payload)
    proba = predictor.predict_proba(payload)
    assert pred[0] in (0, 1)
    assert proba.shape == (1, 2)
    assert int(proba[0, 1] >= predictor.threshold) == pred[0]


def test_untrained_trainer_raises():
    with pytest.raises(ModelNotFittedError):
        ModelTrainer(model_type='rf', use_mlflow=False).predict(pd.DataFrame())
