"""Data preprocessing pipeline"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    df = df.copy()
    
    # 1. Remove id column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # 2. Fill BMI missing values
    if 'bmi' in df.columns:
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
    
    # 3. Drop remaining missing values
    df = df.dropna()
    
    # 4. Encode categorical features
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'stroke':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    
    # 5. Separate X and y
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # 6. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y, scaler, encoders
