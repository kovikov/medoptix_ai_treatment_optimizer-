from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional

app = FastAPI(
    title="MedOptix AI Treatment Optimizer",
    description="API for predicting patient dropout and forecasting adherence",
    version="1.0.0"
)

# Load models and scalers
try:
    dropout_model = joblib.load('models/dropout_prediction_model.joblib')
    dropout_scaler = joblib.load('models/dropout_prediction_scaler.joblib')
    adherence_model = joblib.load('models/adherence_forecasting_model.joblib')
    adherence_scaler = joblib.load('models/adherence_forecasting_scaler.joblib')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

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
async def predict_dropout(features: DropoutFeatures):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Handle categorical variables
        input_data = pd.get_dummies(input_data, columns=['gender', 'chronic_condition', 'injury_type'])
        
        # Ensure all required columns are present
        required_columns = dropout_scaler.feature_names_in_
        for col in required_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[required_columns]
        
        # Scale features
        scaled_features = dropout_scaler.transform(input_data)
        
        # Make prediction
        dropout_probability = dropout_model.predict_proba(scaled_features)[0][1]
        
        return {
            "dropout_probability": float(dropout_probability),
            "prediction": "High Risk" if dropout_probability > 0.5 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast_adherence")
async def forecast_adherence(features: AdherenceFeatures):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Handle categorical variables
        input_data = pd.get_dummies(input_data, columns=['gender', 'chronic_condition', 'injury_type'])
        
        # Ensure all required columns are present
        required_columns = adherence_scaler.feature_names_in_
        for col in required_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[required_columns]
        
        # Scale features
        scaled_features = adherence_scaler.transform(input_data)
        
        # Make prediction
        predicted_adherence = adherence_model.predict(scaled_features)[0]
        
        return {
            "predicted_adherence": float(predicted_adherence),
            "adherence_category": get_adherence_category(predicted_adherence)
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