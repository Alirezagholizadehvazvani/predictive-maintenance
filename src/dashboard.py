# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yaml

# Page config
st.set_page_config(
    page_title="Equipment Monitoring Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('data/models/rf_model_latest.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/models/scaler_latest.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load config
@st.cache_data
def load_config():
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

# Load historical data
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('data/raw/training_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Predict function
def predict_state(vibration, temperature, current, pressure, model, scaler):
    features = np.array([[vibration, temperature, current, pressure]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    return prediction, probabilities

# Main app
def main():
    st.title("⚙️ Equipment Monitoring Dashboard")
    st.markdown("---")
    
    # Load resources
    model, scaler = load_model()
    config = load_config()
    historical_data = load_historical_data()
    
    if model is None or scaler is None:
        st.error("Failed to load model. Please ensure model files exist.")
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Real-time Monitoring", "Historical Analysis", "Model Info"])
    
    if page == "Real-time Monitoring":
        show_realtime_monitoring(model, scaler, config)
    elif page == "Historical Analysis":
        show_historical_analysis(historical_data)
    elif page == "Model Info":
        show_model_info(model, config)

def show_realtime_monitoring(model, scaler, config):
    st.header("Real-time Equipment Monitoring")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sensor Inputs")
        
        # Get sensor ranges from config
        if config and 'sensors' in config:
            sensor_config = config['sensors']
            vib_range = sensor_config.get('vibration', {}).get('range', [0, 10])
            temp_range = sensor_config.get('temperature', {}).get('range', [20, 100])
            curr_range = sensor_config.get('current', {}).get('range', [0, 50])
            press_range = sensor_config.get('pressure', {}).get('range', [0, 200])
        else:
            vib_range, temp_range, curr_range, press_range = [0, 10], [20, 100], [0, 50], [0, 200]
        
        vibration = st.slider("Vibration (mm/s)", 
                             float(vib_range[0]), float(vib_range[1]), 
                             value=2.0, step=0.1)
        
        temperature = st.slider("Temperature (°C)", 
                               float(temp_range[0]), float(temp_range[1]), 
                               value=50.0, step=1.0)