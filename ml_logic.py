import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model and encoder with error handling
try:
    model = joblib.load("crime_predictor.pkl")
    label_encoder = joblib.load("crime_label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

def predict_crime_type(hour, dayofweek, month, latitude, longitude):
    """Predict the most likely crime type for given parameters"""
    try:
        # Bin coordinates for better generalization
        lat_bin = pd.cut([latitude], bins=20, labels=False)[0]
        lon_bin = pd.cut([longitude], bins=20, labels=False)[0]
    except Exception:
        lat_bin, lon_bin = -1, -1

    input_data = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'lat_bin': lat_bin,
        'lon_bin': lon_bin
    }])

    try:
        prediction = model.predict(input_data)[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Unknown"

def predict_crime_proba(hour, dayofweek, month, latitude, longitude):
    """Get probability distribution for crime types"""
    try:
        lat_bin = pd.cut([latitude], bins=20, labels=False)[0]
        lon_bin = pd.cut([longitude], bins=20, labels=False)[0]
    except Exception:
        lat_bin, lon_bin = -1, -1

    input_data = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'lat_bin': lat_bin,
        'lon_bin': lon_bin
    }])

    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(input_data)[0]
            top_indices = np.argsort(proba)[::-1][:3]  # Top 3 crimes
            return [(label_encoder.inverse_transform([i])[0], round(proba[i] * 100, 2)) 
                   for i in top_indices]
        except Exception as e:
            print(f"Probability prediction error: {str(e)}")
    
    return [("Unknown", 0.0)]