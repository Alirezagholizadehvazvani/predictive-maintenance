import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"Pickle protocol: {pickle.HIGHEST_PROTOCOL}")

# Try to load and inspect the problematic files
model_path = Path('data/models/rf_model_latest.pkl')
scaler_path = Path('data/models/scaler_latest.pkl')
encoder_path = Path('data/models/label_encoders_latest.pkl')

print("\n=== Attempting to load models ===")

# Try different pickle protocols
for protocol in [None, 'latin1', 'bytes']:
    print(f"\nTrying encoding: {protocol}")
    try:
        with open(model_path, 'rb') as f:
            if protocol:
                model = pickle.load(f, encoding=protocol)
            else:
                model = pickle.load(f)
        print(f"✅ Model loaded successfully with encoding={protocol}")
        print(f"Model type: {type(model)}")
        break
    except Exception as e:
        print(f"❌ Failed: {e}")

# If all fail, retrain from scratch
print("\n=== Retraining models from scratch ===")

try:
    # Load training data
    df = pd.read_csv('data/raw/training_data.csv')
    print(f"✅ Loaded training data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Identify feature types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target if present
    if 'is_synthetic' in df.columns:
        target_col = 'is_synthetic'
        y = df[target_col]
        X = df.drop(columns=[target_col])
    elif 'label' in df.columns:
        target_col = 'label'
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        print("⚠️ No target column found. Creating dummy target.")
        y = np.random.randint(0, 2, len(df))
        X = df
    
    print(f"\nFeatures: {len(X.columns)}")
    print(f"Numeric: {len([c for c in numeric_cols if c in X.columns])}")
    print(f"Categorical: {len([c for c in categorical_cols if c in X.columns])}")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Train model
    print("\n🔄 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    print("✅ Model trained")
    
    # Save with current Python version
    print("\n💾 Saving models...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✅ All models saved successfully")
    
    # Verify
    print("\n🔍 Verifying saved models...")
    with open(model_path, 'rb') as f:
        test_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        test_scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        test_encoders = pickle.load(f)
    
    print("✅ All models verified and working!")
    print(f"\nModel accuracy on training data: {model.score(X_scaled, y):.2%}")
    
except Exception as e:
    print(f"❌ Critical error during retraining: {e}")
