from fastapi.testclient import TestClient
import pytest
from medoptix_ai_treatment_optimizer.app import app
import pandas as pd
from medoptix_ai_treatment_optimizer.utils import prepare_features

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Welcome to MedOptix AI Treatment Optimizer API"

def test_predict_dropout():
    """Test dropout prediction endpoint"""
    # Test with valid data
    data = {
        "session_count": 5,
        "treatment_duration": 30,
        "session_frequency": 0.17,
        "pain_level_mean": 6.5,
        "pain_level_std": 1.2,
        "pain_change": -1.0,
        "pain_change_rate": -0.03,
        "pain_volatility": 0.18,
        "home_adherence_mean": 75.0,
        "home_adherence_std": 10.0,
        "adherence_change": 5.0,
        "adherence_trend": 0.17,
        "adherence_volatility": 0.13,
        "satisfaction_mean": 4.2,
        "satisfaction_std": 0.8,
        "satisfaction_change": 0.5,
        "satisfaction_trend": 0.02,
        "age": 45,
        "bmi": 25.5,
        "gender": "Male",
        "chronic_condition": "None",
        "injury_type": "back"
    }
    
    response = client.post("/predict_dropout", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "dropout_probability" in result
    assert "risk_category" in result
    assert 0 <= result["dropout_probability"] <= 1
    assert result["risk_category"] in ["Low", "Medium", "High"]
    
    # Test with invalid data
    invalid_data = data.copy()
    invalid_data["session_count"] = "invalid"
    response = client.post("/predict_dropout", json=invalid_data)
    assert response.status_code == 422

def test_forecast_adherence():
    """Test adherence forecasting endpoint"""
    # Test with valid data
    data = {
        "session_count": 5,
        "treatment_duration": 30,
        "session_frequency": 0.17,
        "pain_level_mean": 6.5,
        "pain_level_std": 1.2,
        "pain_change": -1.0,
        "pain_change_rate": -0.03,
        "pain_volatility": 0.18,
        "home_adherence_mean": 75.0,
        "home_adherence_std": 10.0,
        "adherence_change": 5.0,
        "adherence_trend": 0.17,
        "adherence_volatility": 0.13,
        "satisfaction_mean": 4.2,
        "satisfaction_std": 0.8,
        "satisfaction_change": 0.5,
        "satisfaction_trend": 0.02,
        "age": 45,
        "bmi": 25.5,
        "gender": "Male",
        "chronic_condition": "None",
        "injury_type": "back"
    }
    
    response = client.post("/forecast_adherence", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "forecasted_adherence" in result
    assert "adherence_category" in result
    assert 0 <= result["forecasted_adherence"] <= 100
    assert result["adherence_category"] in ["Low", "Medium", "High"]
    
    # Test with invalid data
    invalid_data = data.copy()
    invalid_data["session_count"] = "invalid"
    response = client.post("/forecast_adherence", json=invalid_data)
    assert response.status_code == 422

def test_model_loading():
    """Test that models are loaded correctly"""
    # Test dropout prediction endpoint
    data = {
        "session_count": 5,
        "treatment_duration": 30,
        "session_frequency": 0.17,
        "pain_level_mean": 6.5,
        "pain_level_std": 1.2,
        "pain_change": -1.0,
        "pain_change_rate": -0.03,
        "pain_volatility": 0.18,
        "home_adherence_mean": 75.0,
        "home_adherence_std": 10.0,
        "adherence_change": 5.0,
        "adherence_trend": 0.17,
        "adherence_volatility": 0.13,
        "satisfaction_mean": 4.2,
        "satisfaction_std": 0.8,
        "satisfaction_change": 0.5,
        "satisfaction_trend": 0.02,
        "age": 45,
        "bmi": 25.5,
        "gender": "Male",
        "chronic_condition": "None",
        "injury_type": "back"
    }
    
    # Test both endpoints
    dropout_response = client.post("/predict_dropout", json=data)
    adherence_response = client.post("/forecast_adherence", json=data)
    
    assert dropout_response.status_code == 200
    assert adherence_response.status_code == 200
    
    dropout_result = dropout_response.json()
    adherence_result = adherence_response.json()
    
    assert "dropout_probability" in dropout_result
    assert "risk_category" in dropout_result
    assert "forecasted_adherence" in adherence_result
    assert "adherence_category" in adherence_result
    assert 0 <= dropout_result["dropout_probability"] <= 1
    assert 0 <= adherence_result["forecasted_adherence"] <= 100
    assert dropout_result["risk_category"] in ["Low", "Medium", "High"]
    assert adherence_result["adherence_category"] in ["Low", "Medium", "High"]

def test_invalid_input():
    """Test API with invalid input"""
    response = client.post(
        "/predict_dropout",
        json={
            "session_count": "invalid",  # Should be integer
            "treatment_duration": 30,
            "session_frequency": 0.17,
            "pain_level_mean": 6.5,
            "pain_level_std": 1.2,
            "pain_change": -1.0,
            "pain_change_rate": -0.03,
            "pain_volatility": 0.18,
            "home_adherence_mean": 75.0,
            "home_adherence_std": 10.0,
            "adherence_change": 5.0,
            "adherence_trend": 0.17,
            "adherence_volatility": 0.13,
            "satisfaction_mean": 4.2,
            "satisfaction_std": 0.8,
            "satisfaction_change": 0.5,
            "satisfaction_trend": 0.02,
            "age": 45,
            "bmi": 25.5,
            "gender": "Male",
            "chronic_condition": "None",
            "injury_type": "back"
        }
    )
    assert response.status_code == 422  # Validation error

def test_missing_required_field():
    """Test API with missing required field"""
    response = client.post(
        "/predict_dropout",
        json={
            "treatment_duration": 30,
            "session_frequency": 0.17,
            "pain_level_mean": 6.5,
            "pain_level_std": 1.2,
            "pain_change": -1.0,
            "pain_change_rate": -0.03,
            "pain_volatility": 0.18,
            "home_adherence_mean": 75.0,
            "home_adherence_std": 10.0,
            "adherence_change": 5.0,
            "adherence_trend": 0.17,
            "adherence_volatility": 0.13,
            "satisfaction_mean": 4.2,
            "satisfaction_std": 0.8,
            "satisfaction_change": 0.5,
            "satisfaction_trend": 0.02,
            "age": 45,
            "bmi": 25.5,
            "gender": "Male",
            "chronic_condition": "None",
            "injury_type": "back"
        }
    )
    assert response.status_code == 422  # Validation error

def test_predict_dropout_edge_cases():
    """Test dropout prediction endpoint with edge cases"""
    # Test with minimum values
    response = client.post(
        "/predict_dropout",
        json={
            "session_count": 1,
            "treatment_duration": 1,
            "session_frequency": 0.01,
            "pain_level_mean": 0,
            "pain_level_std": 0,
            "pain_change": -10,
            "pain_change_rate": -1.0,
            "pain_volatility": 0,
            "home_adherence_mean": 0,
            "home_adherence_std": 0,
            "adherence_change": -100,
            "adherence_trend": -1.0,
            "adherence_volatility": 0,
            "satisfaction_mean": 1,
            "satisfaction_std": 0,
            "satisfaction_change": -5,
            "satisfaction_trend": -1.0,
            "age": 18,
            "bmi": 15,
            "gender": "Female",
            "chronic_condition": "Multiple",
            "injury_type": "other"
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "dropout_probability" in result
    assert "risk_category" in result
    assert 0 <= result["dropout_probability"] <= 1
    data = response.json()
    assert "dropout_probability" in data
    assert "risk_category" in data

    # Test with maximum values
    response = client.post(
        "/predict_dropout",
        json={
            "session_count": 100,
            "treatment_duration": 365,
            "session_frequency": 1.0,
            "pain_level_mean": 10,
            "pain_level_std": 5,
            "pain_change": 10,
            "pain_change_rate": 1.0,
            "pain_volatility": 1.0,
            "home_adherence_mean": 100,
            "home_adherence_std": 50,
            "adherence_change": 100,
            "adherence_trend": 1.0,
            "adherence_volatility": 1.0,
            "satisfaction_mean": 5,
            "satisfaction_std": 2,
            "satisfaction_change": 5,
            "satisfaction_trend": 1.0,
            "age": 100,
            "bmi": 50,
            "gender": "Male",
            "chronic_condition": "Multiple",
            "injury_type": "other"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "dropout_probability" in data
    assert "risk_category" in data 