import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml
import pickle
import sys
from pathlib import Path

# Add error tracking
error_log = []

def log_error(stage, error):
    error_log.append(f"[{stage}] {str(error)}")
    st.error(f"Error in {stage}: {str(error)}")

try:
    # Page config
    st.set_page_config(
        page_title="Synthetic Data Quality Monitor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔍 Synthetic Data Quality Monitoring Dashboard")
    
    # Load config
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        st.success("✅ Config loaded")
    except Exception as e:
        log_error("Config Loading", e)
        config = {}
    
    # Load models
    try:
        with open('data/models/rf_model_latest.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/models/scaler_latest.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('data/models/label_encoders_latest.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        st.success("✅ Models loaded")
    except Exception as e:
        log_error("Model Loading", e)
        model = scaler = label_encoders = None
    
    # Load training data
    try:
        df_train = pd.read_csv('data/raw/training_data.csv')
        st.success(f"✅ Training data loaded: {len(df_train)} rows")
    except Exception as e:
        log_error("Training Data Loading", e)
        df_train = None
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload synthetic data CSV",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df_synthetic = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df_synthetic)} rows")
            
            # Display basic info
            st.header("📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df_synthetic))
            with col2:
                st.metric("Features", len(df_synthetic.columns))
            with col3:
                if df_train is not None:
                    st.metric("Training Records", len(df_train))
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df_synthetic.head(10))
            
            # Show column info
            st.subheader("Column Information")
            st.write(df_synthetic.dtypes)
            
        except Exception as e:
            log_error("Synthetic Data Processing", e)
    
    else:
        st.info("👆 Upload a synthetic data CSV file to begin analysis")
    
    # Show any errors
    if error_log:
        st.warning("⚠️ Errors encountered:")
        for err in error_log:
            st.code(err)

except Exception as e:
    st.error(f"Critical error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
