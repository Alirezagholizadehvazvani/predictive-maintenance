# src/test_dashboard_content.py
import streamlit as st
import pandas as pd
import yaml

st.set_page_config(page_title="Test Dashboard", layout="wide")

st.title("🔧 Test Dashboard")
st.write("If you see this, Streamlit is working!")

# Test config loading
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    st.success("✅ Config loaded")
    st.json(config)
except Exception as e:
    st.error(f"❌ Config error: {e}")

# Test data loading
try:
    df = pd.read_csv('data/training_data.csv')
    st.success(f"✅ Data loaded: {len(df)} rows")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Data error: {e}")
