import pandas as pd
import json
from pathlib import Path

# Read the CSV
csv_path = Path("data/processed/predictions.csv")
df = pd.read_csv(csv_path)

# Map states to numeric values
state_map = {'normal': 0, 'degraded': 1, 'failure': 2}

# Create the predictions list
predictions = []
for _, row in df.iterrows():
    predictions.append({
        'timestamp': row['timestamp'],
        'actual': state_map.get(row['actual_state'], 0),
        'predicted': state_map.get(row['predicted_state'], 0),
        'confidence': float(row['confidence']) if 'confidence' in row else 0.85
    })

# Save to JSON
json_path = Path("data/processed/predictions.json")
with open(json_path, 'w') as f:
    json.dump(predictions, f, indent=2)

print(f"✓ Created {json_path}")
print(f"✓ Converted {len(predictions)} predictions")
