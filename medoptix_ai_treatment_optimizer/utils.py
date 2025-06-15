import pandas as pd

def prepare_features(data):
    """Prepare features for model training or prediction"""
    # Create a copy of the data
    features = data.copy()
    
    # Rename columns to match expected names
    column_mapping = {
        'chronic_condition': 'chronic_cond'
    }
    features = features.rename(columns=column_mapping)
    
    # Handle categorical variables
    features = pd.get_dummies(features, columns=['gender', 'chronic_cond', 'injury_type'])
    
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