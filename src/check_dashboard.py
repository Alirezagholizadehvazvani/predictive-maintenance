# src/check_dashboard.py
import sys
import traceback
from pathlib import Path

print("=" * 60)
print("DASHBOARD DIAGNOSTIC CHECK")
print("=" * 60)

# 1. Check config file
print("\n1. Checking config file...")
try:
    import yaml
    config_path = Path("config/config.yaml")
    if config_path.exists():
        print(f"   ✓ Config file exists: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Config loaded successfully")
        print(f"   Config keys: {list(config.keys())}")
    else:
        print(f"   ✗ Config file NOT found: {config_path}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error loading config: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. Check model files
print("\n2. Checking model files...")
for key in ['model_path', 'scaler_path', 'encoder_path']:
    if key in config:
        path = Path(config[key])
        if path.exists():
            print(f"   ✓ {key}: {path} (exists)")
        else:
            print(f"   ✗ {key}: {path} (NOT FOUND)")
    else:
        print(f"   ✗ {key} not in config")

# 3. Try loading models
print("\n3. Attempting to load models...")
try:
    import joblib
    
    model_path = Path(config['model_path'])
    print(f"   Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"   ✓ Model loaded: {type(model).__name__}")
    
    scaler_path = Path(config['scaler_path'])
    print(f"   Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"   ✓ Scaler loaded: {type(scaler).__name__}")
    
    encoder_path = Path(config['encoder_path'])
    print(f"   Loading encoder from: {encoder_path}")
    encoder = joblib.load(encoder_path)
    print(f"   ✓ Encoder loaded: {type(encoder).__name__}")
    
except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    traceback.print_exc()
    sys.exit(1)

# 4. Check training data
print("\n4. Checking training data...")
try:
    import pandas as pd
    data_path = Path("data/raw/training_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"   ✓ Training data loaded: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
    else:
        print(f"   ✗ Training data NOT found: {data_path}")
except Exception as e:
    print(f"   ✗ Error loading training data: {e}")
    traceback.print_exc()

# 5. Try importing dashboard
print("\n5. Checking dashboard imports...")
try:
    sys.path.insert(0, str(Path.cwd() / "src"))
    print("   Attempting to import dashboard module...")
    import dashboard
    print("   ✓ Dashboard module imported successfully")
except Exception as e:
    print(f"   ✗ Error importing dashboard: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED!")
print("=" * 60)
print("\nIf dashboard still shows blank, check browser console (F12)")
print("for JavaScript errors.")
