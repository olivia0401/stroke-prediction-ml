"""Prediction module for stroke risk"""
import joblib
import pandas as pd

from src.preprocessor import normalize_columns


class StrokePredictor:
    """Stroke risk predictor using a unified pipeline artifact"""

    def __init__(self, artifact_path: str):
        """
        Load the entire model artifact, which includes the preprocessing
        pipeline, the classifier, and the prediction threshold.
        """
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact.get('pipeline') or artifact.get('model')
        self.threshold = artifact['threshold']
        self.model_type = artifact.get('model_type', 'unknown')

    def _prepare(self, X) -> pd.DataFrame:
        """
        Coerce input to a DataFrame and normalize column names to the
        lowercase snake_case form the fitted pipeline expects.

        The training pipeline lowercases columns (e.g. ``Residence_type`` ->
        ``residence_type``); applying the same normalization here keeps
        train/serve column names consistent so the ColumnTransformer can
        locate every feature.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return normalize_columns(X)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict stroke risk using the optimized threshold.

        Args:
            X: DataFrame with input features.

        Returns:
            A Series of binary predictions (0 or 1).
        """
        X = self._prepare(X)

        # The pipeline handles all preprocessing
        y_proba = self.pipeline.predict_proba(X)[:, 1]

        # Apply the optimized threshold
        return (y_proba >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict stroke risk probabilities.

        Args:
            X: DataFrame with input features.

        Returns:
            A Series of prediction probabilities for the positive class.
        """
        X = self._prepare(X)

        # The pipeline handles all preprocessing
        return self.pipeline.predict_proba(X)
