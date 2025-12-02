"""Model training with MLflow tracking and joblib serialization."""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from .logger import setup_logger

logger = setup_logger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class ModelTrainer:
    """Train and save ML models with experiment tracking."""

    def __init__(self, model_type: str = 'rf', use_mlflow: bool = True):
        """Initialize trainer.

        Args:
            model_type: 'rf' for RandomForest or 'xgb' for XGBoost
            use_mlflow: Whether to use MLflow tracking
        """
        self.model_type = model_type
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.model = None

        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'xgb':
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                scale_pos_weight=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} model")

    def train(self, X_train, y_train, X_test, y_test):
        """Train model with optional MLflow tracking.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        if self.use_mlflow:
            with mlflow.start_run():
                return self._train_with_mlflow(X_train, y_train, X_test, y_test)
        else:
            return self._train_basic(X_train, y_train, X_test, y_test)

    def _train_basic(self, X_train, y_train, X_test, y_test):
        """Train without MLflow."""
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

        logger.info(f"F1: {metrics['f1_score']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        return metrics

    def _train_with_mlflow(self, X_train, y_train, X_test, y_test):
        """Train with MLflow tracking."""
        # Log parameters
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_param("n_estimators", self.model.n_estimators if hasattr(self.model, 'n_estimators') else 'N/A')

        # Train
        logger.info(f"Training {self.model_type} with MLflow tracking...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

        # Log metrics
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])

        # Log model
        mlflow.sklearn.log_model(self.model, "model")

        logger.info(f"MLflow run logged: F1={metrics['f1_score']:.3f}")
        return metrics

    def save_model(self, filepath: str = 'models/model.pkl'):
        """Save model with joblib.

        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
