# predictive-maintenance
End-to-end machine learning pipeline for predictive maintenance with real-time monitoring dashboard and automated model training
# Predictive Maintenance System

End-to-end machine learning pipeline for predictive maintenance with real-time monitoring dashboard and automated model training.

## Features

- **High-accuracy ML model**: Random Forest classifier achieving 99.67% accuracy
- **Real-time monitoring**: Interactive Streamlit dashboard for live predictions
- **Sensor-based detection**: Uses vibration, temperature, current, and pressure data
- **Automated pipeline**: Complete training, evaluation, and deployment workflow
Installation
bash

pip install -r requirements.txt

Usage
Train the model
bash

python src/train_model.py

Run the dashboard
bash

streamlit run src/dashboard.py

Model Performance
Accuracy: 99.67%
Algorithm: Random Forest Classifier
Features: Vibration, Temperature, Current, Pressure
Technologies
Python 3.x
scikit-learn
pandas
Streamlit
plotly
PyYAML
License
MIT
