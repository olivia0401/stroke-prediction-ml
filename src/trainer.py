"""Model training with SMOTEENN and threshold optimization"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from .logger import setup_logger

logger = setup_logger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class ModelTrainer:
    """Train ML models with imbalanced learning and threshold optimization"""

    def __init__(self, model_type: str = 'rf', use_mlflow: bool = True, use_smoteenn: bool = True):
        """
        Initialize trainer

        Args:
            model_type: 'rf' for RandomForest or 'xgb' for XGBoost
            use_mlflow: Whether to use MLflow tracking
            use_smoteenn: Whether to use SMOTEENN resampling (recommended for imbalanced data)
        """
        self.model_type = model_type
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.use_smoteenn = use_smoteenn
        self.model = None
        self.best_threshold = 0.5  # Default threshold

        # Best hyperparameters from Exam Notebook (Step 3)
        if model_type == 'rf':
            self.base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=3,
                random_state=42
            )
        elif model_type == 'xgb':
            self.base_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='auc'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} model with SMOTEENN={'ON' if use_smoteenn else 'OFF'}")

    def train(self, X, y, preprocessor=None, use_full_data=True):
        """
        Train model with SMOTEENN resampling

        Args:
            X, y: Complete dataset (will use all data if use_full_data=True)
            preprocessor: sklearn ColumnTransformer
            use_full_data: If True, train on all data like Exam Notebook

        Returns:
            dict: Performance metrics
        """
        if self.use_mlflow:
            mlflow.start_run()

        # Build pipeline
        if self.use_smoteenn:
            if preprocessor:
                self.model = ImbPipeline([
                    ('pre', preprocessor),
                    ('sampler', SMOTEENN(random_state=42)),
                    ('clf', self.base_model)
                ])
            else:
                self.model = ImbPipeline([
                    ('sampler', SMOTEENN(random_state=42)),
                    ('clf', self.base_model)
                ])
        else:
            if preprocessor:
                self.model = ImbPipeline([
                    ('pre', preprocessor),
                    ('clf', self.base_model)
                ])
            else:
                self.model = self.base_model

        # Train on full data (like Exam Notebook)
        logger.info("Training model on full dataset with SMOTEENN...")
        self.model.fit(X, y)

        # Get predictions on full data for threshold optimization
        y_proba = self.model.predict_proba(X)[:, 1]

        # Optimize threshold (Step 4 from Exam Notebook)
        self.best_threshold = self._optimize_threshold(y, y_proba)
        logger.info(f"Optimized threshold: {self.best_threshold:.3f}")

        # Re-predict with optimized threshold
        y_pred_optimized = (y_proba >= self.best_threshold).astype(int)

        # Calculate metrics
        metrics = {
            'f1_score': f1_score(y, y_pred_optimized),
            'precision': precision_score(y, y_pred_optimized),
            'recall': recall_score(y, y_pred_optimized),
            'threshold': self.best_threshold
        }

        logger.info(f"F1: {metrics['f1_score']:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")

        if self.use_mlflow:
            mlflow.log_params({
                'model_type': self.model_type,
                'use_smoteenn': self.use_smoteenn
            })
            mlflow.log_metrics(metrics)
            mlflow.end_run()

        return metrics

    def _optimize_threshold(self, y_true, y_proba):
        """
        Optimize classification threshold using Precision-Recall curve
        (Based on Exam Notebook Step 4)

        Returns:
            float: Optimal threshold that maximizes F1 score
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    def predict(self, X):
        """Predict with optimized threshold"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= self.best_threshold).astype(int)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str = 'models/model.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'threshold': self.best_threshold,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """Load trained model"""
        data = joblib.load(filepath)
        trainer = ModelTrainer(model_type=data['model_type'], use_smoteenn=False)
        trainer.model = data['model']
        trainer.best_threshold = data.get('threshold', 0.5)
        return trainer
