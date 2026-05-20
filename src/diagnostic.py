import streamlit as st
import sys
import traceback

st.title("Diagnostic Dashboard")

try:
    st.write("Step 1: Importing libraries...")
    import pandas as pd
    import pickle
    import yaml
    from pathlib import Path
    st.success("✓ Libraries imported")
    
    st.write("Step 2: Loading config...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    st.success(f"✓ Config loaded: {list(config.keys())}")
    
    st.write("Step 3: Loading models...")
    models = {}
    model_dir = Path('data/models')
    for model_file in model_dir.glob('*.pkl'):
        st.write(f"  Loading {model_file.name}...")
        with open(model_file, 'rb') as f:
            models[model_file.stem] = pickle.load(f)
        st.success(f"  ✓ {model_file.stem} loaded")
    st.success(f"✓ All models loaded: {list(models.keys())}")
    
    st.write("Step 4: Loading training data...")
    df = pd.read_csv('data/raw/training_data.csv')
    st.success(f"✓ Data loaded: {len(df)} rows")
    
    st.write("Step 5: Testing prediction...")
    test_input = df.iloc[0][['temperature', 'vibration', 'pressure', 'rpm']].values.reshape(1, -1)
    prediction = models['rf_model_latest'].predict(test_input)
    st.success(f"✓ Prediction works: {prediction[0]}")
    
    st.balloons()
    st.write("## All systems operational! ✓")
    
except Exception as e:
    st.error(f"Error at current step: {str(e)}")
    st.code(traceback.format_exc())
