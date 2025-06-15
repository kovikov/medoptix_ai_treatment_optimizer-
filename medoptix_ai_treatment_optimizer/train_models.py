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
    """Train and save the models."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Loading data...")
    df = load_and_preprocess_data()
    
    print("Preparing features...")
    # Calculate patient-level features
    patient_features = df.groupby('patient_id').agg({
        'session_id': 'count',  # session_count
        'pain_level': ['mean', 'std'],  # pain metrics
        'home_adherence_pc': ['mean', 'std'],  # adherence metrics
        'satisfaction': ['mean', 'std'],  # satisfaction metrics
        'age': 'first',
        'bmi': 'first',
        'gender': 'first',
        'chronic_cond': 'first',
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
    
    print("Preparing target variables...")
    # Define dropout (patients with less than 8 sessions)
    patient_features['is_dropout'] = (patient_features['session_count'] < 8).astype(int)
    
    # Prepare features for modeling
    categorical_columns = ['gender', 'chronic_cond', 'injury_type']
    X = pd.get_dummies(patient_features.drop(['patient_id', 'is_dropout'], axis=1), 
                      columns=categorical_columns)
    
    # Split data for dropout prediction
    X_dropout = X.copy()
    y_dropout = patient_features['is_dropout']
    
    # Split data for adherence forecasting
    X_adherence = X[patient_features['session_count'] >= 8].copy()
    y_adherence = patient_features.loc[patient_features['session_count'] >= 8, 'home_adherence_mean']
    
    print("Training models...")
    # Train dropout prediction model
    X_train_dropout, X_test_dropout, y_train_dropout, y_test_dropout = train_test_split(
        X_dropout, y_dropout, test_size=0.2, random_state=42
    )
    
    dropout_scaler = StandardScaler()
    X_train_dropout_scaled = dropout_scaler.fit_transform(X_train_dropout)
    X_test_dropout_scaled = dropout_scaler.transform(X_test_dropout)
    
    dropout_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dropout_model.fit(X_train_dropout_scaled, y_train_dropout)
    
    # Train adherence forecasting model
    X_train_adherence, X_test_adherence, y_train_adherence, y_test_adherence = train_test_split(
        X_adherence, y_adherence, test_size=0.2, random_state=42
    )
    
    adherence_scaler = StandardScaler()
    X_train_adherence_scaled = adherence_scaler.fit_transform(X_train_adherence)
    X_test_adherence_scaled = adherence_scaler.transform(X_test_adherence)
    
    adherence_model = RandomForestRegressor(n_estimators=100, random_state=42)
    adherence_model.fit(X_train_adherence_scaled, y_train_adherence)
    
    print("Saving models...")
    # Save models and scalers
    joblib.dump(dropout_model, 'models/dropout_prediction_model.joblib')
    joblib.dump(dropout_scaler, 'models/dropout_prediction_scaler.joblib')
    joblib.dump(adherence_model, 'models/adherence_forecasting_model.joblib')
    joblib.dump(adherence_scaler, 'models/adherence_forecasting_scaler.joblib')
    
    # Print model performance
    print("\nModel Performance:")
    print("Dropout Prediction Model:")
    print(f"Training Accuracy: {dropout_model.score(X_train_dropout_scaled, y_train_dropout):.3f}")
    print(f"Test Accuracy: {dropout_model.score(X_test_dropout_scaled, y_test_dropout):.3f}")
    
    print("\nAdherence Forecasting Model:")
    print(f"Training R² Score: {adherence_model.score(X_train_adherence_scaled, y_train_adherence):.3f}")
    print(f"Test R² Score: {adherence_model.score(X_test_adherence_scaled, y_test_adherence):.3f}")
    
    print("\nModels and scalers have been saved to the 'models' directory.")
    
    return dropout_model, dropout_scaler, adherence_model, adherence_scaler

if __name__ == "__main__":
    train_models() 