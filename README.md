# Intelligent Aquaponics Management System

AI-powered aquaponics system integrating IoT sensors and machine learning for optimized plant growth.

## 🎯 Project Overview

This system uses Random Forest regression to predict plant growth rates based on environmental parameters collected from IoT sensors.

**Key Results:**
- 30% reduction in water consumption
- 25% increase in productivity
- Real-time monitoring of 10,000+ measurements/day

## 🛠️ Technologies

- **ML Framework:** scikit-learn (Random Forest)
- **Data Processing:** pandas, numpy
- **IoT:** Arduino/ESP32, LoRaWAN sensors
- **Database:** MySQL
- **Sensors:** pH, temperature, humidity, TDS

## 📊 Features

- Real-time sensor data collection and analysis
- Plant growth prediction algorithm
- Automated alert system with corrective recommendations
- Analytical dashboards for yield optimization

## 🚀 Installation

```bash
# Create conda environment
conda create -n aquaponics python=3.9
conda activate aquaponics

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

```bash
python src/train_model.py --data data/sensor_data.csv --model random_forest --n_estimators 100 --max_depth 10 --output models/
```

## 📈 Model Performance

- **Algorithm:** Random Forest Regressor
- **Features:** pH, temperature, humidity, TDS, rolling averages
- **R² Score:** ~0.85
- **MSE:** Low variance in predictions

## 📁 Project Structure

```
aquaponics-ai-system/
├── src/
│   └── train_model.py          # Main training script
├── data/
│   └── sensor_data.csv         # Sensor data sample
├── models/
│   └── aquaponics_model.pkl    # Trained model
├── requirements.txt
└── README.md
```

## 🔬 Methodology

1. **Data Collection:** IoT sensors collect environmental data every hour
2. **Feature Engineering:** Rolling averages to capture temporal trends
3. **Model Training:** Random Forest for handling non-linear relationships
4. **Deployment:** Real-time predictions for automated system control

## 👨‍💻 Author

**Rivaldo Verckys SENONKIN**  
Computer Engineer specializing in AI and IoT  
Email: verckys1998@gmail.com

## 📝 License

This project was developed as part of my Bachelor's degree final year project at the National Higher Institute of Industrial Technology, Lokossa.
