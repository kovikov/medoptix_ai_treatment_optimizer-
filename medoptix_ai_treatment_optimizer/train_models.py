import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os
from pathlib import Path

def validate_data(df: pd.DataFrame) -> bool:
    """Validate the input data has required columns and data types."""
    required_columns = [
        'patient_id', 'session_id', 'pain_level', 'home_adherence_pc',
        'satisfaction', 'age', 'bmi', 'gender', 'chronic_cond', 'injury_type'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate data types
    numeric_columns = ['pain_level', 'home_adherence_pc', 'satisfaction', 'age', 'bmi']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
    
    return True

def load_and_preprocess_data() -> pd.DataFrame:
    """Load and preprocess the data with proper error handling."""
    data_path = Path('data/processed/cleaned_merged_medoptix.csv')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        validate_data(df)
        
        # Convert date columns to datetime
        date_columns = ['created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
            else:
                print(f"Warning: Column '{col}' is missing in the dataset. Skipping datetime conversion.")
        
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

def train_models():
    """Train and save models"""
    print("Loading data...")
    data = load_and_preprocess_data()
    
    print("Preparing features...")
    X = prepare_features(data)
    y_dropout = data['dropout']
    y_adherence = data['adherence']
    
    # Split data
    X_train, X_test, y_dropout_train, y_dropout_test = train_test_split(
        X, y_dropout, test_size=0.2, random_state=42
    )
    _, _, y_adherence_train, y_adherence_test = train_test_split(
        X, y_adherence, test_size=0.2, random_state=42
    )
    
    print("Training models...")
    # Initialize scalers
    dropout_scaler = StandardScaler()
    adherence_scaler = StandardScaler()
    
    # Scale features
    X_train_scaled = dropout_scaler.fit_transform(X_train)
    X_test_scaled = dropout_scaler.transform(X_test)
    
    # Train dropout prediction model
    dropout_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dropout_model.fit(X_train_scaled, y_dropout_train)
    
    # Scale features for adherence model
    X_train_scaled = adherence_scaler.fit_transform(X_train)
    X_test_scaled = adherence_scaler.transform(X_test)
    
    # Train adherence forecasting model
    adherence_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    adherence_model.fit(X_train_scaled, y_adherence_train)
    
    # Evaluate models
    print("\nModel Performance:")
    print("Dropout Prediction Model:")
    print(f"Training Accuracy: {dropout_model.score(X_train_scaled, y_dropout_train):.3f}")
    print(f"Test Accuracy: {dropout_model.score(X_test_scaled, y_dropout_test):.3f}")
    
    print("\nAdherence Forecasting Model:")
    print(f"Training R² Score: {adherence_model.score(X_train_scaled, y_adherence_train):.3f}")
    print(f"Test R² Score: {adherence_model.score(X_test_scaled, y_adherence_test):.3f}")
    
    # Save models and scalers
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(dropout_model, 'models/dropout_prediction_model.joblib')
    joblib.dump(dropout_scaler, 'models/dropout_prediction_scaler.joblib')
    joblib.dump(adherence_model, 'models/adherence_forecasting_model.joblib')
    joblib.dump(adherence_scaler, 'models/adherence_forecasting_scaler.joblib')
    
    print("\nModels and scalers have been saved to the 'models' directory.")
    return dropout_model, dropout_scaler, adherence_model, adherence_scaler

def prepare_features(data):
    """Prepare features for model training or prediction"""
    # Create a copy of the data
    features = data.copy()
    
    # Handle categorical variables
    features = pd.get_dummies(features, columns=['gender', 'chronic_condition', 'injury_type'])
    
    # Ensure all required columns are present
    required_columns = [
        'session_count', 'treatment_duration', 'session_frequency',
        'pain_level_mean', 'pain_level_std', 'pain_change', 'pain_change_rate',
        'pain_volatility', 'home_adherence_mean', 'home_adherence_std',
        'adherence_change', 'adherence_trend', 'adherence_volatility',
        'satisfaction_mean', 'satisfaction_std', 'satisfaction_change',
        'satisfaction_trend', 'age', 'bmi',
        'gender_Male', 'gender_Female',
        'chronic_cond_None', 'chronic_cond_Asthma', 'chronic_cond_Cardio',
        'chronic_cond_Diabetes', 'chronic_cond_Hypertension',
        'injury_type_back', 'injury_type_knee', 'injury_type_shoulder',
        'injury_type_other'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in features.columns:
            features[col] = 0
    
    # Reorder columns to ensure consistent order
    features = features[required_columns]
    
    return features

if __name__ == "__main__":
    train_models() 