"""Model training with SMOTEENN and threshold optimization"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve
)
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
        self.cv_metrics = None  # Cross-validated (held-out) performance estimate

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

    def _build_pipeline(self, preprocessor=None):
        """Assemble the (preprocessor ->) SMOTEENN -> classifier pipeline."""
        steps = []
        if preprocessor is not None:
            steps.append(('pre', preprocessor))
        if self.use_smoteenn:
            steps.append(('sampler', SMOTEENN(random_state=42)))
        steps.append(('clf', self.base_model))

        if len(steps) == 1:
            # Only the classifier, no resampling/preprocessing wrapper needed
            return clone(self.base_model)
        return ImbPipeline(steps)

    def train(self, X, y, preprocessor=None, cv_folds=5):
        """
        Fit the final deployment pipeline and produce an honest, cross-validated
        performance estimate.

        The reported metrics come from ``cv_folds``-fold stratified
        cross-validation: within each fold the model is fit on the training
        split and the decision threshold is tuned on that same training split,
        then both are evaluated on the held-out split. This avoids the
        optimistic bias of scoring on the data used for fitting/thresholding.

        Separately, a final pipeline is fit on the full dataset (with its
        threshold tuned on the full data) and saved for deployment.

        Args:
            X, y: Complete dataset.
            preprocessor: sklearn ColumnTransformer (fit inside the pipeline).
            cv_folds: Number of stratified CV folds for evaluation.

        Returns:
            dict: Cross-validated performance metrics (mean over folds), with
                  per-metric standard deviations and the deployment threshold.
        """
        if self.use_mlflow:
            mlflow.start_run()

        # 1. Honest evaluation via stratified cross-validation
        logger.info(f"Evaluating with {cv_folds}-fold stratified cross-validation...")
        metrics = self._cross_val_evaluate(X, y, preprocessor, cv_folds)

        # 2. Fit the final deployment pipeline on ALL data
        logger.info("Fitting final deployment pipeline on full dataset...")
        self.model = self._build_pipeline(preprocessor)
        self.model.fit(X, y)

        # Tune the operating threshold for the deployed model on the full data.
        # (Reported performance still comes from the CV estimate above.)
        full_proba = self.model.predict_proba(X)[:, 1]
        self.best_threshold = self._optimize_threshold(y, full_proba)
        metrics['threshold'] = self.best_threshold
        self.cv_metrics = metrics

        logger.info(
            "Cross-validated  F1: %.4f (+/- %.4f), Recall: %.4f (+/- %.4f), "
            "Precision: %.4f (+/- %.4f)",
            metrics['f1_score'], metrics['f1_score_std'],
            metrics['recall'], metrics['recall_std'],
            metrics['precision'], metrics['precision_std'],
        )
        logger.info(f"Deployment threshold (tuned on full data): {self.best_threshold:.3f}")

        if self.use_mlflow:
            mlflow.log_params({
                'model_type': self.model_type,
                'use_smoteenn': self.use_smoteenn,
                'cv_folds': cv_folds
            })
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.end_run()

        return metrics

    def _cross_val_evaluate(self, X, y, preprocessor, cv_folds):
        """
        Estimate held-out performance with stratified K-fold cross-validation.

        For each fold the pipeline is cloned and fit on the training split, the
        threshold is tuned on the training split, and F1/precision/recall/AUC
        are computed on the untouched held-out split. Returns the mean and
        standard deviation of each metric across folds.
        """
        X = X.reset_index(drop=True)
        y = np.asarray(y)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = {'f1_score': [], 'precision': [], 'recall': [], 'roc_auc': []}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = self._build_pipeline(preprocessor)
            pipeline.fit(X_train, y_train)

            # Tune threshold ONLY on the training split to avoid leakage
            train_proba = pipeline.predict_proba(X_train)[:, 1]
            threshold = self._optimize_threshold(y_train, train_proba)

            # Evaluate on the held-out split
            test_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (test_proba >= threshold).astype(int)

            fold_scores['f1_score'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_scores['roc_auc'].append(roc_auc_score(y_test, test_proba))
            logger.info(
                "  Fold %d/%d  F1=%.4f Recall=%.4f Precision=%.4f (thr=%.3f)",
                fold, cv_folds, fold_scores['f1_score'][-1],
                fold_scores['recall'][-1], fold_scores['precision'][-1], threshold,
            )

        metrics = {}
        for name, values in fold_scores.items():
            metrics[name] = float(np.mean(values))
            metrics[f'{name}_std'] = float(np.std(values))
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
        """Save the entire trained pipeline and threshold."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save the entire pipeline, not just the classifier
        artifact = {
            'pipeline': self.model,
            'threshold': self.best_threshold,
            'model_type': self.model_type
        }
        joblib.dump(artifact, filepath)
        logger.info(f"Entire model artifact (pipeline + threshold) saved to {filepath}")
