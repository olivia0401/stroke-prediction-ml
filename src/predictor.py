"""Prediction module for stroke risk"""
import joblib
import pandas as pd


class StrokePredictor:
    """Stroke risk predictor using a unified pipeline artifact"""

    def __init__(self, artifact_path: str):
        """
        Load the entire model artifact, which includes the preprocessing
        pipeline, the classifier, and the prediction threshold.
        """
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact['pipeline']
        self.threshold = artifact['threshold']
        self.model_type = artifact.get('model_type', 'unknown')

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict stroke risk using the optimized threshold.

        Args:
            X: DataFrame with input features.

        Returns:
            A Series of binary predictions (0 or 1).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # The pipeline handles all preprocessing
        return self.pipeline.predict_proba(X)
