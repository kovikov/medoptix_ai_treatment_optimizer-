import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

def clean_patients(df):
    """Clean patients dataframe"""
    # Convert date columns
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    
    # Clean categorical columns
    df['gender'] = df['gender'].str.lower()
    df['chronic_cond'] = df['chronic_cond'].fillna('None')
    df['injury_type'] = df['injury_type'].str.lower()
    df['referral_source'] = df['referral_source'].str.lower()
    df['insurance_type'] = df['insurance_type'].str.lower()
    
    # Clean numeric columns
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Handle outliers
    df.loc[df['bmi'] > 50, 'bmi'] = np.nan
    df.loc[df['age'] > 100, 'age'] = np.nan
    
    return df

def clean_clinics(df):
    """Clean clinics dataframe"""
    # Clean categorical columns
    df['city'] = df['city'].str.lower()
    df['country'] = df['country'].str.upper()
    df['type'] = df['type'].str.upper()
    df['speciality'] = df['speciality'].str.lower()
    
    # Clean numeric columns
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    df['staff_count'] = pd.to_numeric(df['staff_count'], errors='coerce')
    df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce')
    
    # Handle outliers
    df.loc[df['avg_rating'] > 5, 'avg_rating'] = np.nan
    
    return df

def clean_interventions(df):
    """Clean interventions dataframe"""
    # Convert date columns
    df['sent_at'] = pd.to_datetime(df['sent_at'])
    
    # Clean categorical columns
    df['channel'] = df['channel'].str.lower()
    df['message'] = df['message'].str.lower()
    
    # Convert boolean column
    df['responded'] = df['responded'].map({'True': True, 'False': False})
    
    return df

def clean_sessions(df):
    """Clean sessions dataframe"""
    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    # Clean numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def clean_feedback(df):
    """Clean feedback dataframe"""
    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    # Clean numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def get_latest_intervention(interventions_df):
    """Get the latest intervention per patient"""
    # Sort by patient_id and sent_at
    sorted_interventions = interventions_df.sort_values(['patient_id', 'sent_at'])
    
    # Get the latest intervention per patient
    latest_interventions = sorted_interventions.groupby('patient_id').last().reset_index()
    
    return latest_interventions

def main():
    # Create data directories if they don't exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Read raw data
    patients = pd.read_csv("data/raw/patients.csv")
    clinics = pd.read_csv("data/raw/clinics.csv")
    interventions = pd.read_csv("data/raw/interventions.csv")
    sessions = pd.read_csv("data/raw/sessions.csv")
    feedback = pd.read_csv("data/raw/feedback.csv")
    
    # Print column names for debugging
    print("\nFeedback columns:", feedback.columns.tolist())
    print("\nSessions columns:", sessions.columns.tolist())
    print("\nInterventions columns:", interventions.columns.tolist())
    
    # Clean data
    patients_clean = clean_patients(patients)
    clinics_clean = clean_clinics(clinics)
    interventions_clean = clean_interventions(interventions)
    sessions_clean = clean_sessions(sessions)
    feedback_clean = clean_feedback(feedback)
    
    # Save cleaned data
    patients_clean.to_csv("data/processed/patients_clean.csv", index=False)
    clinics_clean.to_csv("data/processed/clinics_clean.csv", index=False)
    interventions_clean.to_csv("data/processed/interventions_clean.csv", index=False)
    sessions_clean.to_csv("data/processed/sessions_clean.csv", index=False)
    feedback_clean.to_csv("data/processed/feedback_clean.csv", index=False)
    
    # Get latest intervention per patient
    latest_interventions = get_latest_intervention(interventions_clean)
    
    # Create merged dataset
    # Start with patients and merge with clinics
    merged = patients_clean.merge(clinics_clean, on='clinic_id', how='left')
    
    # Merge with latest interventions
    merged = merged.merge(latest_interventions, on='patient_id', how='left')
    
    # Merge sessions with feedback
    sessions_with_feedback = sessions_clean.merge(feedback_clean, on='session_id', how='left')
    
    # Merge the sessions+feedback with the main dataset
    merged = merged.merge(sessions_with_feedback, on='patient_id', how='left')
    
    # Save merged dataset with new filename
    merged.to_csv("data/processed/cleaned_merged_medoptix.csv", index=False)
    
    print("\nData cleaning and merging completed successfully!")
    print(f"Final merged dataset shape: {merged.shape}")
    print(f"Saved merged dataset to: data/processed/cleaned_merged_medoptix.csv")

if __name__ == "__main__":
    main() 