"""Prediction module for stroke risk"""
import joblib
import pandas as pd


class StrokePredictor:
    """Stroke risk predictor"""

    def __init__(self, model_path, scaler_path, encoders_path):
        """Load model and preprocessors"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)

    def predict(self, X):
        """Predict stroke risk
        
        Args:
            X: DataFrame or dict
            
        Returns:
            Predictions (0 or 1)
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X = X.copy()
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict probabilities"""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
