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

def load_and_preprocess_data(file_path='data/processed/cleaned_merged_medoptix.csv'):
    """Load and preprocess the data"""
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert datetime columns if they exist
        datetime_columns = ['created_at', 'updated_at']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
            else:
                print(f"Warning: Column '{col}' is missing in the dataset. Skipping datetime conversion.")
        
        # Calculate patient-level features
        patient_features = df.groupby('patient_id').agg({
            'session_id': 'count',  # session_count
            'pain_level': ['mean', 'std'],  # pain metrics
            'home_adherence_pc': ['mean', 'std'],  # adherence metrics
            'satisfaction': ['mean', 'std'],  # satisfaction metrics
            'age': 'first',
            'bmi': 'first',
            'gender': 'first',
            'chronic_cond': 'first',  # Note: using chronic_cond instead of chronic_condition
            'injury_type': 'first'
        }).reset_index()
        
        # Flatten column names
        patient_features.columns = ['patient_id', 'session_count', 
                                  'pain_level_mean', 'pain_level_std',
                                  'home_adherence_mean', 'home_adherence_std',
                                  'satisfaction_mean', 'satisfaction_std',
                                  'age', 'bmi', 'gender', 'chronic_cond', 'injury_type']
        
        # Calculate additional features
        if 'created_at' in df.columns:
            patient_features['treatment_duration'] = (df.groupby('patient_id')['created_at'].max() - 
                                                    df.groupby('patient_id')['created_at'].min()).dt.days
            patient_features['session_frequency'] = patient_features['session_count'] / patient_features['treatment_duration']
        else:
            print("Warning: 'created_at' column not found. Using default values for treatment duration and session frequency.")
            patient_features['treatment_duration'] = 30
            patient_features['session_frequency'] = patient_features['session_count'] / 30
        
        # Calculate changes and trends
        for metric, mean_col, std_col in [
            ('pain_level', 'pain_level_mean', 'pain_level_std'),
            ('home_adherence_pc', 'home_adherence_mean', 'home_adherence_std'),
            ('satisfaction', 'satisfaction_mean', 'satisfaction_std')
        ]:
            if metric in df.columns:
                # Calculate change
                patient_features[f'{metric}_change'] = df.groupby('patient_id')[metric].apply(
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
                ).values
                # Calculate change rate
                patient_features[f'{metric}_change_rate'] = patient_features[f'{metric}_change'] / patient_features['treatment_duration']
                # Calculate volatility
                patient_features[f'{metric}_volatility'] = patient_features[std_col] / (patient_features[mean_col] + 1e-6)
            else:
                print(f"Warning: '{metric}' column not found. Setting related features to 0.")
                patient_features[f'{metric}_change'] = 0
                patient_features[f'{metric}_change_rate'] = 0
                patient_features[f'{metric}_volatility'] = 0
        
        # Define dropout (patients with less than 8 sessions)
        patient_features['dropout'] = (patient_features['session_count'] < 8).astype(int)
        
        # Define adherence target
        patient_features['adherence'] = patient_features['home_adherence_mean']
        
        return patient_features
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

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
    
    # Rename columns to match expected names
    column_mapping = {
        'chronic_cond': 'chronic_condition'
    }
    features = features.rename(columns=column_mapping)
    
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
        'chronic_condition_None', 'chronic_condition_Asthma', 'chronic_condition_Cardio',
        'chronic_condition_Diabetes', 'chronic_condition_Hypertension',
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