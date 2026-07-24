"""Data preprocessing pipeline with medical feature engineering"""
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline


def normalize_columns(df):
    """
    Normalize column names to lowercase snake_case.

    This is applied both at training time and at serving time so that the
    fitted ColumnTransformer (which references columns such as
    ``residence_type``) always receives the exact column names it expects,
    regardless of the raw casing used by the caller (e.g. ``Residence_type``).

    Returns:
        A copy of ``df`` with normalized column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().map(lambda s: re.sub(r'[^0-9a-z_]', '_', s))
    return df


def preprocess_data(df):
    """
    Complete preprocessing pipeline based on Exam Notebook approach

    Steps:
    1. Row filtering (remove 'Other' gender, drop NaN)
    2. Column cleaning (remove ID, normalize names)
    3. Medical feature binning (age, BMI, glucose)
    4. One-hot encoding + standardization

    Returns:
        X, y, preprocessor (sklearn ColumnTransformer)
    """
    df = df.copy()

    # 1. Row Filtering
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Other']
    df = df.dropna()

    # 2. Column Cleaning
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Normalize column names (lowercase + snake_case)
    df = normalize_columns(df)

    # 3. Separate X and y
    X = df.drop(columns='stroke')
    y = df['stroke']

    # 4. Create preprocessor with medical feature binning
    preprocessor = create_preprocessor()

    return X, y, preprocessor


def make_bins(X):
    """Medical-informed binning function"""
    # Ensure input is a DataFrame with correct columns
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['age', 'bmi', 'avg_glucose_level'])

    df = X[['age', 'bmi', 'avg_glucose_level']].copy()

    df['age_bin'] = pd.cut(
        df['age'],
        bins=[0, 24, 44, 64, 79, 150],
        labels=['<25', '25-44', '45-64', '65-79', '80+'],
        right=False
    )

    df['bmi_bin'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
        right=False
    )

    df['glu_bin'] = pd.cut(
        df['avg_glucose_level'],
        bins=[0, 70, 84, 99, 109, 125, 139, float('inf')],
        labels=['<70', '70-84', '85-99', '100-109', '110-125', '126-139', '≥140'],
        right=False
    )

    return df[['age_bin', 'bmi_bin', 'glu_bin']]


def create_preprocessor():
    """
    Build the ColumnTransformer used for both training and serving.

    age / bmi / avg_glucose_level are fed through in two parallel forms on the
    same rows: binned into medical bands then one-hot encoded (to capture the
    non-linear risk thresholds), and also kept as standardized continuous values.
    The remaining categorical and binary columns are one-hot encoded.
    """
    bin_encoder = ImbPipeline([
        ('bin', FunctionTransformer(make_bins, validate=False)),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_features = ['age', 'bmi', 'avg_glucose_level']
    preprocessor = ColumnTransformer([
        ('bins', bin_encoder, ['age', 'bmi', 'avg_glucose_level']),
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), [
            'gender', 'ever_married', 'work_type',
            'residence_type', 'smoking_status',
            'hypertension', 'heart_disease'
        ])
    ])
    return preprocessor
