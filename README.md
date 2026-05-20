# Predictive Maintenance System

End-to-end machine learning pipeline for predictive maintenance with real-time monitoring dashboard and automated model training.

## Features

- **High-accuracy ML model**: Random Forest classifier achieving 99.67% accuracy
- **Real-time monitoring**: Interactive Flask dashboard with live predictions and metrics
- **Sensor-based detection**: Uses vibration, temperature, current, and pressure data
- **Automated pipeline**: Complete training, evaluation, and deployment workflow
- **Visual analytics**: Chart.js-powered prediction visualization and performance tracking
- **Dual dashboard support**: Both Flask (web-based) and Streamlit (analytics) interfaces

## Installation
```bash
pip install -r requirements.txt

## Usage

### Train the model
bash
python src/train_model.py

### Convert predictions (if needed)
bash
python convert_predictions.py

### Run the Flask dashboard
bash
python src/dashboard_flask.py
Then open `http://127.0.0.1:5000` in your browser.

### Run the Streamlit dashboard (alternative)
bash
streamlit run src/dashboard.py

## Model Performance

- **Accuracy**: 99.67%
- **Algorithm**: Random Forest Classifier
- **Features**: Vibration, Temperature, Current, Pressure
- **States**: Normal (0), Degraded (1), Failure (2)

## Project Structure


predictive-maintenance/
├── data/
│   ├── raw/              # Raw sensor data
│   └── processed/        # Processed predictions and metrics
│       ├── predictions.json
│       ├── predictions.csv
│       └── metrics.json
├── models/               # Trained model artifacts
├── src/
│   ├── train_model.py    # Model training pipeline
│   ├── dashboard_flask.py # Flask web dashboard
│   └── dashboard.py      # Streamlit analytics dashboard
├── templates/
│   └── dashboard.html    # Flask dashboard template
├── convert_predictions.py # CSV to JSON converter
└── requirements.txt

## Technologies

- **Backend**: Python 3.x, Flask
- **ML/Data**: scikit-learn, pandas, numpy
- **Visualization**: Chart.js, Streamlit, plotly
- **Configuration**: PyYAML

## Applications

This system is applicable across multiple industries:

- **Manufacturing**: Equipment failure prediction, production line optimization
- **Energy**: Turbine and generator monitoring
- **Transportation**: Vehicle fleet maintenance
- **Oil & Gas**: Pipeline and drilling equipment monitoring
- **Data Centers**: Server and cooling system health tracking
- **Healthcare**: Medical equipment maintenance scheduling

## Value Proposition

- Proactive maintenance reduces downtime by up to 50%
- Cost savings through optimized maintenance schedules
- Real-time visual monitoring for quick decision-making
- Lightweight and scalable architecture
- Easy integration with existing sensor infrastructure

## License

MIT


This README now accurately reflects:
- Flask as the primary dashboard
- The conversion script for predictions
- Dual dashboard support (Flask + Streamlit)
- Complete project structure
- Real-world applications and value proposition
- Correct usage instructions with proper URLs
