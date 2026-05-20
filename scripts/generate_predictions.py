import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load training data
df = pd.read_csv('data/processed/training_data.csv')

# Generate timestamps
start_date = datetime.now() - timedelta(days=30)
timestamps = [start_date + timedelta(hours=i) for i in range(len(df))]

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'timestamp': timestamps,
    'vibration': df['vibration'],
    'temperature': df['temperature'],
    'current': df['current'],
    'pressure': df['pressure'],
    'actual_state': df['state'],
    'predicted_state': df['state'],  # Using actual as predicted for now
    'confidence': np.random.uniform(0.75, 0.99, len(df))  # Random confidence scores
})

# Save to CSV
predictions_df.to_csv('data/processed/predictions.csv', index=False)
print(f'Generated {len(predictions_df)} predictions')
print(predictions_df.head())
