"""Bin edges must agree with their labels (bins are left-closed)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessor import make_bins, normalize_columns  # noqa: E402


def _bins(age=50.0, bmi=22.0, glucose=90.0):
    X = pd.DataFrame({'age': [age], 'bmi': [bmi], 'avg_glucose_level': [glucose]})
    return make_bins(X).iloc[0]


@pytest.mark.parametrize("age,label", [
    (0.08, '<25'), (24.9, '<25'), (25, '25-44'), (44.9, '25-44'), (45, '45-64'),
    (64.9, '45-64'), (65, '65-79'), (79.9, '65-79'), (80, '80+'), (120, '80+'),
])
def test_age_bins(age, label):
    assert _bins(age=age)['age_bin'] == label


@pytest.mark.parametrize("bmi,label", [
    (18.4, 'Underweight'), (18.5, 'Normal'), (24.9, 'Normal'), (25.0, 'Overweight'),
    (29.9, 'Overweight'), (30.0, 'Obese'), (97.6, 'Obese'), (150.0, 'Obese'),
])
def test_bmi_bins(bmi, label):
    assert _bins(bmi=bmi)['bmi_bin'] == label


@pytest.mark.parametrize("glucose,label", [
    (69.9, '<70'), (70, '70-84'), (84.9, '70-84'), (85, '85-99'), (99.9, '85-99'),
    (100, '100-109'), (110, '110-125'), (125.9, '110-125'), (126, '126-139'),
    (139.9, '126-139'), (140, '≥140'), (271.7, '≥140'),
])
def test_glucose_bins(glucose, label):
    assert _bins(glucose=glucose)['glu_bin'] == label


def test_normalize_columns_matches_training_names():
    df = normalize_columns(pd.DataFrame(columns=['Residence_type', 'avg glucose']))
    assert list(df.columns) == ['residence_type', 'avg_glucose']
