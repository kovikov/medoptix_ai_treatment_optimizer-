import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from medoptix_ai_treatment_optimizer.train_models import train_models, validate_data, load_and_preprocess_data

def test_validate_data():
    """Test data validation function"""
    # Test with valid data
    valid_data = pd.DataFrame({
        'patient_id': [1, 2],
        'session_id': [1, 2],
        'pain_level': [5.0, 6.0],
        'home_adherence_pc': [80.0, 85.0],
        'satisfaction': [4.0, 4.5],
        'age': [45, 50],
        'bmi': [25.0, 26.0],
        'gender': ['M', 'F'],
        'chronic_cond': ['None', 'None'],
        'injury_type': ['back', 'knee']
    })
    assert validate_data(valid_data) is True
    
    # Test with missing column
    invalid_data = valid_data.drop('pain_level', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(invalid_data)
    
    # Test with invalid data type
    invalid_data = valid_data.copy()
    invalid_data['pain_level'] = ['high', 'low']
    with pytest.raises(ValueError, match="must be numeric"):
        validate_data(invalid_data)

def test_load_and_preprocess_data():
    """Test data loading and preprocessing"""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data("non_existent_file.csv")
    
    # Test with valid data
    data = load_and_preprocess_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'dropout' in data.columns
    assert 'adherence' in data.columns

def test_model_files_exist():
    """Test that model files exist"""
    assert os.path.exists('models/dropout_prediction_model.joblib')
    assert os.path.exists('models/dropout_prediction_scaler.joblib')
    assert os.path.exists('models/adherence_forecasting_model.joblib')
    assert os.path.exists('models/adherence_forecasting_scaler.joblib')

def test_model_types():
    """Test that models are of correct type"""
    dropout_model = joblib.load('models/dropout_prediction_model.joblib')
    adherence_model = joblib.load('models/adherence_forecasting_model.joblib')
    
    assert isinstance(dropout_model, RandomForestClassifier)
    assert isinstance(adherence_model, RandomForestRegressor)

def test_scaler_types():
    """Test that scalers are of correct type"""
    dropout_scaler = joblib.load('models/dropout_prediction_scaler.joblib')
    adherence_scaler = joblib.load('models/adherence_forecasting_scaler.joblib')
    
    assert isinstance(dropout_scaler, StandardScaler)
    assert isinstance(adherence_scaler, StandardScaler)

def test_model_predictions():
    """Test that models can make predictions"""
    # Load models and scalers
    dropout_model = joblib.load('models/dropout_prediction_model.joblib')
    dropout_scaler = joblib.load('models/dropout_prediction_scaler.joblib')
    adherence_model = joblib.load('models/adherence_forecasting_model.joblib')
    adherence_scaler = joblib.load('models/adherence_forecasting_scaler.joblib')
    
    # Create sample input data
    sample_data = pd.DataFrame({
        'session_count': [5],
        'treatment_duration': [30],
        'session_frequency': [0.17],
        'pain_level_mean': [6.5],
        'pain_level_std': [1.2],
        'pain_change': [-1.0],
        'pain_change_rate': [-0.03],
        'pain_volatility': [0.18],
        'home_adherence_mean': [75.0],
        'home_adherence_std': [10.0],
        'adherence_change': [5.0],
        'adherence_trend': [0.17],
        'adherence_volatility': [0.13],
        'satisfaction_mean': [4.2],
        'satisfaction_std': [0.8],
        'satisfaction_change': [0.5],
        'satisfaction_trend': [0.02],
        'age': [45],
        'bmi': [25.5],
        'gender': ['Male'],
        'chronic_condition': ['None'],
        'injury_type': ['back']
    })
    
    # Prepare features
    features = prepare_features(sample_data)
    
    # Test dropout prediction
    dropout_features = dropout_scaler.transform(features)
    dropout_prob = dropout_model.predict_proba(dropout_features)[0][1]
    assert 0 <= dropout_prob <= 1
    
    # Test adherence prediction
    adherence_features = adherence_scaler.transform(features)
    adherence_pred = adherence_model.predict(adherence_features)[0]
    assert 0 <= adherence_pred <= 100

def test_model_training():
    """Test the model training process"""
    # Train models
    dropout_model, dropout_scaler, adherence_model, adherence_scaler = train_models()
    
    # Verify model types
    assert isinstance(dropout_model, RandomForestClassifier)
    assert isinstance(adherence_model, RandomForestRegressor)
    assert isinstance(dropout_scaler, StandardScaler)
    assert isinstance(adherence_scaler, StandardScaler)
    
    # Verify model parameters
    assert dropout_model.n_estimators == 100
    assert dropout_model.max_depth == 10
    assert adherence_model.n_estimators == 100
    assert adherence_model.max_depth == 10

def test_model_feature_importance():
    """Test that models have learned meaningful feature importance"""
    dropout_model = joblib.load('models/dropout_prediction_model.joblib')
    adherence_model = joblib.load('models/adherence_forecasting_model.joblib')
    
    # Check dropout model feature importance
    dropout_importance = dropout_model.feature_importances_
    assert len(dropout_importance) > 0
    assert np.all(dropout_importance >= 0)
    assert np.any(dropout_importance > 0)  # At least one feature should be important
    
    # Check adherence model feature importance
    adherence_importance = adherence_model.feature_importances_
    assert len(adherence_importance) > 0
    assert np.all(adherence_importance >= 0)
    assert np.any(adherence_importance > 0)  # At least one feature should be important

def test_model_prediction_consistency():
    """Test that models make consistent predictions for similar inputs"""
    dropout_model = joblib.load('models/dropout_prediction_model.joblib')
    dropout_scaler = joblib.load('models/dropout_prediction_scaler.joblib')
    adherence_model = joblib.load('models/adherence_forecasting_model.joblib')
    adherence_scaler = joblib.load('models/adherence_forecasting_scaler.joblib')
    
    # Create two similar input samples
    sample1 = pd.DataFrame({
        'session_count': [5],
        'treatment_duration': [30],
        'session_frequency': [0.17],
        'pain_level_mean': [6.5],
        'pain_level_std': [1.2],
        'pain_change': [-1.0],
        'pain_change_rate': [-0.03],
        'pain_volatility': [0.18],
        'home_adherence_mean': [75.0],
        'home_adherence_std': [10.0],
        'adherence_change': [5.0],
        'adherence_trend': [0.17],
        'adherence_volatility': [0.13],
        'satisfaction_mean': [4.2],
        'satisfaction_std': [0.8],
        'satisfaction_change': [0.5],
        'satisfaction_trend': [0.02],
        'age': [45],
        'bmi': [25.5],
        'gender': ['Male'],
        'chronic_condition': ['None'],
        'injury_type': ['back']
    })
    
    sample2 = pd.DataFrame({
        'session_count': [5],
        'treatment_duration': [31],
        'session_frequency': [0.17],
        'pain_level_mean': [6.4],
        'pain_level_std': [1.2],
        'pain_change': [-1.1],
        'pain_change_rate': [-0.03],
        'pain_volatility': [0.18],
        'home_adherence_mean': [76.0],
        'home_adherence_std': [10.0],
        'adherence_change': [5.0],
        'adherence_trend': [0.17],
        'adherence_volatility': [0.13],
        'satisfaction_mean': [4.2],
        'satisfaction_std': [0.8],
        'satisfaction_change': [0.5],
        'satisfaction_trend': [0.02],
        'age': [45],
        'bmi': [25.5],
        'gender': ['Male'],
        'chronic_condition': ['None'],
        'injury_type': ['back']
    })
    
    # Prepare features
    features1 = prepare_features(sample1)
    features2 = prepare_features(sample2)
    
    # Get predictions
    dropout_pred1 = dropout_model.predict_proba(dropout_scaler.transform(features1))[0][1]
    dropout_pred2 = dropout_model.predict_proba(dropout_scaler.transform(features2))[0][1]
    adherence_pred1 = adherence_model.predict(adherence_scaler.transform(features1))[0]
    adherence_pred2 = adherence_model.predict(adherence_scaler.transform(features2))[0]
    
    # Check that predictions are similar for similar inputs
    assert abs(dropout_pred1 - dropout_pred2) < 0.1
    assert abs(adherence_pred1 - adherence_pred2) < 5.0 