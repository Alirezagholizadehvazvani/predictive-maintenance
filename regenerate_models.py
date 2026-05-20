import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from datetime import datetime
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Delete old model files
model_dir = 'data/models/'
print("Deleting old model files...")
for file in os.listdir(model_dir):
    if file.endswith('.pkl'):
        filepath = os.path.join(model_dir, file)
        os.remove(filepath)
        print(f"  Deleted: {filepath}")

# Load training data
print("\nLoading training data...")
df = pd.read_csv('data/raw/training_data.csv')
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Prepare features and target
feature_cols = ['vibration', 'temperature', 'current', 'pressure', 'state']
X = df[feature_cols].copy()
y = df['label'].copy()

# Encode 'state' column (categorical)
print("\nEncoding 'state' column...")
state_encoder = LabelEncoder()
X['state'] = state_encoder.fit_transform(X['state'])
print(f"State classes: {state_encoder.classes_}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, y)
print(f"Model trained. Score: {model.score(X_scaled, y):.4f}")

# Save with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save model
model_path = os.path.join(model_dir, f'rf_model_{timestamp}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"\nSaved model: {model_path}")

# Save scaler
scaler_path = os.path.join(model_dir, f'scaler_{timestamp}.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved scaler: {scaler_path}")

# Save state encoder
encoder_path = os.path.join(model_dir, f'state_encoder_{timestamp}.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(state_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved encoder: {encoder_path}")

print("\n✓ All models regenerated successfully!")
print(f"\nUpdate config/config.yaml with:")
print(f"  model_path: data/models/rf_model_{timestamp}.pkl")
print(f"  scaler_path: data/models/scaler_{timestamp}.pkl")
print(f"  encoder_path: data/models/state_encoder_{timestamp}.pkl")
