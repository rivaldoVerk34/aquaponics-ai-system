"""
Prediction Script for Aquaponics System
Author: Rivaldo Verckys SENONKIN
Description: Use trained model to predict plant growth
"""

import pandas as pd
import joblib
import argparse

def predict_growth(model_path, sensor_data):
    """
    Predict plant growth rate from sensor data
    
    Args:
        model_path: Path to trained model (.pkl file)
        sensor_data: Dictionary with sensor readings
    
    Returns:
        Predicted growth rate
    """
    # Load trained model
    model = joblib.load(model_path)
    
    # Prepare input data
    input_df = pd.DataFrame([sensor_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return prediction[0]

def main():
    parser = argparse.ArgumentParser(description='Predict plant growth from sensor data')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ph', type=float, required=True, help='pH value')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature (°C)')
    parser.add_argument('--humidity', type=float, required=True, help='Humidity (%)')
    parser.add_argument('--tds', type=float, required=True, help='TDS value')
    
    args = parser.parse_args()
    
    # Prepare sensor data
    sensor_data = {
        'ph': args.ph,
        'temperature': args.temperature,
        'humidity': args.humidity,
        'tds': args.tds,
        'ph_rolling_mean': args.ph,  # Simplified for demo
        'temp_rolling_mean': args.temperature,
        'humidity_rolling_mean': args.humidity
    }
    
    # Predict
    growth_rate = predict_growth(args.model, sensor_data)
    
    print("\n" + "="*50)
    print("PLANT GROWTH PREDICTION")
    print("="*50)
    print(f"pH: {args.ph}")
    print(f"Temperature: {args.temperature}°C")
    print(f"Humidity: {args.humidity}%")
    print(f"TDS: {args.tds}")
    print("-"*50)
    print(f"Predicted Growth Rate: {growth_rate:.2f} cm/day")
    print("="*50)

if __name__ == "__main__":
    main()
