"""Unit tests for preprocessor"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessor import preprocess_data, create_preprocessor

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'gender': ['Male', 'Female', 'Other', 'Male', 'Female'],
        'age': [67.0, 61.0, 80.0, 49.0, 79.0],
        'hypertension': [0, 0, 0, 1, 0],
        'heart_disease': [1, 0, 1, 0, 0],
        'ever_married': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'work_type': ['Private', 'Self-employed', 'Private', 'Private', 'Self-employed'],
        'Residence_type': ['Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
        'avg_glucose_level': [228.69, 202.21, 105.92, 171.23, 174.12],
        'bmi': [36.6, np.nan, 32.5, 34.4, 24.0],
        'smoking_status': ['formerly smoked', 'never smoked', 'never smoked', 'smokes', 'never smoked'],
        'stroke': [1, 1, 1, 1, 1]
    })


def test_preprocess_data_returns(sample_data):
    """Test that preprocess_data returns the correct types and drops unwanted rows."""
    X, y, preprocessor = preprocess_data(sample_data)

    # Check that 'Other' gender is dropped and NaNs are dropped
    assert len(X) == 3 
    assert len(y) == 3
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(preprocessor, ColumnTransformer)
    assert 'id' not in X.columns
    assert 'stroke' not in X.columns


def test_preprocessor_fit_transform(sample_data):
    """Test that the created preprocessor can fit and transform the data."""
    X, y, preprocessor = preprocess_data(sample_data)
    
    assert len(X) > 0, "Preprocessing should not result in an empty DataFrame"

    X_transformed = preprocessor.fit_transform(X, y)
    
    # Check that transformed data has no NaNs
    assert not np.isnan(X_transformed).any(), "Transformed data should not contain NaNs"
    
    # Check shape: rows should match, columns will be more due to OHE
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > X.shape[1]

    
def test_create_preprocessor():
    """Test the preprocessor creation directly."""
    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    # Check that it has the expected transformers by name
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert 'bins' in transformer_names
    assert 'num' in transformer_names
    assert 'cat' in transformer_names

