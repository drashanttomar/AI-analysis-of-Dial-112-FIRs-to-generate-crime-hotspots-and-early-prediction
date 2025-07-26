import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

def retrain_model(data_path="FIR_Dataset_10K_India_Locations.csv", test_size=0.2):
    """Retrain the crime prediction model with improved features"""
    print("🔄 Retraining crime prediction model...")
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Feature engineering
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['month'] = df['Timestamp'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1
    
    # Enhanced geospatial binning
    df['lat_bin'] = pd.cut(df['Latitude'], bins=20, labels=False)
    df['lon_bin'] = pd.cut(df['Longitude'], bins=20, labels=False)
    
    # Drop missing values
    df = df.dropna(subset=['hour', 'dayofweek', 'month', 'lat_bin', 'lon_bin', 'Crime_Type'])
    
    # Prepare features and target
    X = df[['hour', 'dayofweek', 'month', 'season', 'lat_bin', 'lon_bin']]
    y = df['Crime_Type']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    
    # Train model with improved parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    joblib.dump(model, "crime_predictor.pkl")
    joblib.dump(le, "crime_label_encoder.pkl")
    
    print("✅ Model retrained successfully with enhanced features")
    return model, le

if __name__ == "__main__":
    retrain_model()