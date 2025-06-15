import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os
from pathlib import Path
from medoptix_ai_treatment_optimizer.utils import prepare_features

def validate_data(data):
    """Validate the input data"""
    required_columns = [
        'session_count', 'treatment_duration', 'session_frequency',
        'pain_level_mean', 'pain_level_std', 'pain_change', 'pain_change_rate',
        'pain_volatility', 'home_adherence_mean', 'home_adherence_std',
        'adherence_change', 'adherence_trend', 'adherence_volatility',
        'satisfaction_mean', 'satisfaction_std', 'satisfaction_change',
        'satisfaction_trend', 'age', 'bmi', 'gender', 'chronic_condition',
        'injury_type', 'dropout', 'adherence'
    ]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if data.isnull().any().any():
        raise ValueError("Data contains missing values")
    
    # Check data types
    numeric_columns = [col for col in data.columns if col not in ['gender', 'chronic_condition', 'injury_type']]
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column {col} must be numeric")
    
    # Check categorical columns
    categorical_columns = ['gender', 'chronic_condition', 'injury_type']
    for col in categorical_columns:
        if not pd.api.types.is_object_dtype(data[col]):
            raise ValueError(f"Column {col} must be categorical")
    
    return True

def load_and_preprocess_data(file_path='data/patient_data.csv'):
    """Load and preprocess the data"""
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Validate data
        validate_data(data)
        
        # Prepare features
        features = prepare_features(data)
        
        # Split features and targets
        X = features
        y_dropout = data['dropout']
        y_adherence = data['adherence']
        
        return X, y_dropout, y_adherence
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def train_models():
    """Train the models"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load and preprocess data
        X, y_dropout, y_adherence = load_and_preprocess_data()
        
        # Train dropout prediction model
        dropout_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        dropout_model.fit(X, y_dropout)
        
        # Train adherence forecasting model
        adherence_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        adherence_model.fit(X, y_adherence)
        
        # Create and fit scalers
        dropout_scaler = StandardScaler()
        adherence_scaler = StandardScaler()
        dropout_scaler.fit(X)
        adherence_scaler.fit(X)
        
        # Save models and scalers
        joblib.dump(dropout_model, 'models/dropout_prediction_model.joblib')
        joblib.dump(adherence_model, 'models/adherence_forecasting_model.joblib')
        joblib.dump(dropout_scaler, 'models/dropout_prediction_scaler.joblib')
        joblib.dump(adherence_scaler, 'models/adherence_forecasting_scaler.joblib')
        
        return dropout_model, dropout_scaler, adherence_model, adherence_scaler
    
    except Exception as e:
        raise Exception(f"Error training models: {str(e)}")

if __name__ == "__main__":
    train_models() 