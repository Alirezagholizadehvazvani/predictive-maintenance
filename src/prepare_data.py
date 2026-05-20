import pandas as pd
import yaml
import os

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(config):
    # Load raw data
    raw_data_path = config['paths']['raw_data']
    df = pd.read_csv(raw_data_path)
    
    print(f"Loaded {len(df)} rows from {raw_data_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Encode categorical 'state' column if it exists
    if 'state' in df.columns:
        # Convert state to numeric codes
        df['state_encoded'] = df['state'].astype('category').cat.codes
        print(f"Encoded 'state' column: {df['state'].unique()} -> {df['state_encoded'].unique()}")
    
    # Save processed data
    processed_data_path = config['paths']['processed_data']
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    
    print(f"Processed data saved to {processed_data_path}")
    return df

if __name__ == '__main__':
    config = load_config()
    prepare_data(config)
