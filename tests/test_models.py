import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from medoptix_ai_treatment_optimizer.train_models import train_models

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
        'gender_Male': [1],
        'chronic_cond_None': [1],
        'injury_type_back': [1]
    })
    
    # Test dropout prediction
    dropout_features = dropout_scaler.transform(sample_data)
    dropout_pred = dropout_model.predict_proba(dropout_features)
    assert dropout_pred.shape == (1, 2)  # Should return probabilities for 2 classes
    assert np.all(dropout_pred >= 0) and np.all(dropout_pred <= 1)  # Probabilities should be between 0 and 1
    
    # Test adherence forecasting
    adherence_features = adherence_scaler.transform(sample_data)
    adherence_pred = adherence_model.predict(adherence_features)
    assert adherence_pred.shape == (1,)  # Should return single prediction
    assert adherence_pred[0] >= 0 and adherence_pred[0] <= 100  # Adherence should be between 0 and 100

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
    assert dropout_model.n_estimators > 0
    assert adherence_model.n_estimators > 0
    assert dropout_model.max_depth is not None
    assert adherence_model.max_depth is not None

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
        'gender_Male': [1],
        'chronic_cond_None': [1],
        'injury_type_back': [1]
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
        'gender_Male': [1],
        'chronic_cond_None': [1],
        'injury_type_back': [1]
    })
    
    # Get predictions
    dropout_pred1 = dropout_model.predict_proba(dropout_scaler.transform(sample1))
    dropout_pred2 = dropout_model.predict_proba(dropout_scaler.transform(sample2))
    adherence_pred1 = adherence_model.predict(adherence_scaler.transform(sample1))
    adherence_pred2 = adherence_model.predict(adherence_scaler.transform(sample2))
    
    # Check that predictions are similar for similar inputs
    assert np.allclose(dropout_pred1, dropout_pred2, atol=0.1)
    assert np.allclose(adherence_pred1, adherence_pred2, atol=5.0) 