"""
Aquaponics Plant Growth Prediction Model
Author: Rivaldo Verckys SENONKIN
Description: ML model for predicting plant growth based on environmental sensors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse
import os

def load_and_preprocess_data(filepath):
    """Load sensor data and perform preprocessing"""
    print(f"Loading data from {filepath}...")
    
    # Chargement et nettoyage des données des capteurs
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna()
    
    print(f"Loaded {len(df)} records")
    
    # Feature engineering - calcul de moyennes glissantes
    # Rolling averages capture temporal trends in environmental parameters
    df['ph_rolling_mean'] = df['ph'].rolling(window=24).mean()
    df['temp_rolling_mean'] = df['temperature'].rolling(window=24).mean()
    df['humidity_rolling_mean'] = df['humidity'].rolling(window=24).mean()
    
    # Remove NaN values created by rolling
    df = df.dropna()
    
    print(f"After preprocessing: {len(df)} records")
    
    return df

def train_model(df, n_estimators=100, max_depth=10):
    """Train Random Forest model for growth prediction"""
    print("\nPreparing features and target...")
    
    # Préparation des features et target
    # Features include raw sensor data and engineered rolling averages
    X = df[['ph', 'temperature', 'humidity', 'tds', 
            'ph_rolling_mean', 'temp_rolling_mean', 'humidity_rolling_mean']]
    y = df['plant_growth_rate']
    
    # Split et entraînement
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Random Forest model
    # Chosen for its ability to handle non-linear relationships
    # and resistance to noise from IoT sensors
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\nTraining Random Forest with {n_estimators} estimators...")
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    return model, X_test, y_test, y_pred

def main():
    parser = argparse.ArgumentParser(description='Train aquaponics growth prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to sensor data CSV')
    parser.add_argument('--model', type=str, default='random_forest', help='Model type')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=10, help='Max tree depth')
    parser.add_argument('--output', type=str, default='models/', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data(args.data)
    
    # Train model
    model, X_test, y_test, y_pred = train_model(df, args.n_estimators, args.max_depth)
    
    # Save model
    model_path = os.path.join(args.output, 'aquaponics_model.pkl')
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
