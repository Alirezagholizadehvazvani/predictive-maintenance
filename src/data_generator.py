import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def generate_sensor_data(config):
    """Generate synthetic sensor data for pump monitoring"""
    np.random.seed(42)
    
    samples_per_state = config['data_generation']['samples_per_state']
    noise = config['data_generation']['noise_factor']
    
    # Define state parameters based on sensor ranges
    states = {
        'normal': {
            'vibration': np.mean(config['sensors']['vibration']['normal_range']),
            'temperature': np.mean(config['sensors']['temperature']['normal_range']),
            'current': np.mean(config['sensors']['current']['normal_range']),
            'pressure': np.mean(config['sensors']['pressure']['normal_range']),
            'label': 0
        },
        'degraded': {
            'vibration': config['sensors']['vibration']['warning_threshold'],
            'temperature': config['sensors']['temperature']['warning_threshold'],
            'current': config['sensors']['current']['warning_threshold'],
            'pressure': config['sensors']['pressure']['warning_threshold'] - 5,
            'label': 1
        },
        'failure': {
            'vibration': config['sensors']['vibration']['critical_threshold'],
            'temperature': config['sensors']['temperature']['critical_threshold'],
            'current': config['sensors']['current']['critical_threshold'],
            'pressure': config['sensors']['pressure']['critical_threshold'],
            'label': 2
        }
    }
    
    data = []
    
    for state, params in states.items():
        for _ in range(samples_per_state):
            record = {
                'vibration': params['vibration'] * (1 + np.random.normal(0, noise)),
                'temperature': params['temperature'] * (1 + np.random.normal(0, noise)),
                'current': params['current'] * (1 + np.random.normal(0, noise)),
                'pressure': params['pressure'] * (1 + np.random.normal(0, noise)),
                'state': state,
                'label': params['label']
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Generating synthetic training data...")
    df = generate_sensor_data(config)
    
    # Save to CSV
    output_path = Path('data/raw/training_data.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Saved to {output_path}")
    print("\nData distribution:")
    print(df['state'].value_counts())
    print("\nSample statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
