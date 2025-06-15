from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from .train_models import train_models
from typing import List, Optional

app = FastAPI(
    title="MedOptix AI Treatment Optimizer",
    description="API for predicting patient dropout risk and forecasting treatment adherence",
    version="1.0.0"
)

# Initialize models
try:
    dropout_model = joblib.load('models/dropout_prediction_model.joblib')
    dropout_scaler = joblib.load('models/dropout_prediction_scaler.joblib')
    adherence_model = joblib.load('models/adherence_forecasting_model.joblib')
    adherence_scaler = joblib.load('models/adherence_forecasting_scaler.joblib')
    print("Models loaded successfully")
except FileNotFoundError:
    print("Model files not found. Training new models...")
    try:
        dropout_model, dropout_scaler, adherence_model, adherence_scaler = train_models()
        print("Models trained successfully")
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize models")

# Define input models
class DropoutFeatures(BaseModel):
    session_count: int
    treatment_duration: int
    session_frequency: float
    pain_level_mean: float
    pain_level_std: float
    pain_change: float
    pain_change_rate: float
    pain_volatility: float
    home_adherence_mean: float
    home_adherence_std: float
    adherence_change: float
    adherence_trend: float
    adherence_volatility: float
    satisfaction_mean: float
    satisfaction_std: float
    satisfaction_change: float
    satisfaction_trend: float
    age: int
    bmi: float
    gender: str
    chronic_condition: str
    injury_type: str

class AdherenceFeatures(BaseModel):
    session_count: int
    treatment_duration: int
    session_frequency: float
    pain_level_mean: float
    pain_level_std: float
    pain_change: float
    pain_change_rate: float
    pain_volatility: float
    satisfaction_mean: float
    satisfaction_std: float
    satisfaction_change: float
    satisfaction_trend: float
    age: int
    bmi: float
    gender: str
    chronic_condition: str
    injury_type: str

@app.get("/")
async def root():
    return {"message": "Welcome to MedOptix AI Treatment Optimizer API"}

@app.post("/predict_dropout")
async def predict_dropout(data: PatientData):
    """Predict dropout probability"""
    try:
        # Prepare features
        features = prepare_features(data)
        
        # Scale features
        scaled_features = dropout_scaler.transform(features)
        
        # Make prediction
        dropout_prob = dropout_model.predict_proba(scaled_features)[0][1]
        
        # Determine risk category
        if dropout_prob >= 0.7:
            risk_category = "High"
        elif dropout_prob >= 0.4:
            risk_category = "Medium"
        else:
            risk_category = "Low"
        
        return {
            "dropout_probability": float(dropout_prob),
            "risk_category": risk_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast_adherence")
async def forecast_adherence(data: PatientData):
    """Forecast adherence"""
    try:
        # Prepare features
        features = prepare_features(data)
        
        # Scale features
        scaled_features = adherence_scaler.transform(features)
        
        # Make prediction
        predicted_adherence = float(adherence_model.predict(scaled_features)[0])
        
        # Determine adherence category
        if predicted_adherence >= 80:
            adherence_category = "High"
        elif predicted_adherence >= 60:
            adherence_category = "Medium"
        else:
            adherence_category = "Low"
        
        return {
            "predicted_adherence": predicted_adherence,
            "adherence_category": adherence_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_adherence_category(adherence: float) -> str:
    if adherence >= 90:
        return "Excellent"
    elif adherence >= 75:
        return "Good"
    elif adherence >= 60:
        return "Fair"
    else:
        return "Poor"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 