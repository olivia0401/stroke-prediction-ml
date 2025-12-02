"""Unit tests for preprocessor"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessor import preprocess_data


def test_bmi_imputation():
    """Test BMI missing value imputation"""
    df = pd.DataFrame({
        'bmi': [25.0, np.nan, 30.0, np.nan, 28.0],
        'age': [50, 60, 55, 45, 52],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'stroke': [0, 1, 0, 1, 0]
    })

    X, y, scaler, encoders = preprocess_data(df)

    # Check no missing values
    assert X.isna().sum().sum() == 0
    assert len(X) > 0


def test_preprocess_pipeline():
    """Test full preprocessing pipeline"""
    df = pd.DataFrame({
        'age': [50, 60, 55],
        'gender': ['Male', 'Female', 'Male'],
        'bmi': [25.0, 30.0, 28.0],
        'stroke': [0, 1, 0]
    })

    X, y, scaler, encoders = preprocess_data(df)

    # Check output shapes
    assert len(X) == 3
    assert len(y) == 3
    assert 'stroke' not in X.columns

    # Check scaler and encoders exist
    assert scaler is not None
    assert 'gender' in encoders
