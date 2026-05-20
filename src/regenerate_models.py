import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("MODEL REGENERATION SCRIPT")
print("=" * 60)

# Load training data
print("\n1. Loading training data...")
df = pd.read_csv('data/raw/training_data.csv')
print(f"   ✓ Loaded {len(df)} rows")

# Prepare features and target
print("\n2. Preparing features...")
X = df[['temperature', 'vibration', 'pressure', 'rpm']]
y = df['failure']
print(f"   ✓ Features: {list(X.columns)}")
print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")

# Train scaler
print("\n3. Training scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✓ Scaler trained")

# Train model
print("\n4. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)
accuracy = model.score(X_scaled, y)
print(f"   ✓ Model trained (accuracy: {accuracy:.2%})")

# Train label encoders
print("\n5. Training label encoders...")
label_encoders = {}
for col in ['machine_id', 'component']:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le
        print(f"   ✓ Encoder for '{col}': {len(le.classes_)} classes")

# Delete all old pickle files
print("\n6. Cleaning old model files...")
model_dir = Path('data/models')
deleted = []
for pkl_file in model_dir.glob('*.pkl'):
    pkl_file.unlink()
    deleted.append(pkl_file.name)
    print(f"   ✗ Deleted: {pkl_file.name}")
print(f"   ✓ Removed {len(deleted)} old files")

# Save new models with highest protocol
print("\n7. Saving new models...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save model
model_path = model_dir / 'rf_model_latest.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"   ✓ Saved: {model_path.name}")

# Save scaler
scaler_path = model_dir / 'scaler_latest.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"   ✓ Saved: {scaler_path.name}")

# Save label encoders
encoders_path = model_dir / 'label_encoders_latest.pkl'
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"   ✓ Saved: {encoders_path.name}")

# Verify by loading
print("\n8. Verifying saved models...")
with open(model_path, 'rb') as f:
    test_model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    test_scaler = pickle.load(f)
with open(encoders_path, 'rb') as f:
    test_encoders = pickle.load(f)
print("   ✓ All models load successfully")

# Test prediction
print("\n9. Testing prediction...")
test_input = X.iloc[0].values.reshape(1, -1)
test_scaled = test_scaler.transform(test_input)
prediction = test_model.predict(test_scaled)
print(f"   ✓ Test prediction: {prediction[0]}")

print("\n" + "=" * 60)
print("✓ MODEL REGENERATION COMPLETE")
print("=" * 60)
print("\nNew files created:")
print(f"  - {model_path.name}")
print(f"  - {scaler_path.name}")
print(f"  - {encoders_path.name}")
print("\nYou can now run: streamlit run src/diagnostic.py")
