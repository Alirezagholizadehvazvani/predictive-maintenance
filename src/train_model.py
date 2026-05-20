import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

def train_model():
    print("Loading configuration...")
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Loading training data...")
    df = pd.read_csv('data/raw/training_data.csv')
    print(f"Loaded {len(df)} samples")
    print(f"\nColumns in dataset: {df.columns.tolist()}")
    print("\nState distribution:")
    print(df['state'].value_counts())
    
    # Use only sensor features (exclude timestamp, state, and label)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'state', 'label']]
    print(f"\nUsing features: {feature_cols}")
    
    X = df[feature_cols]
    y = df['state']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest model...")
    model_params = config.get('model', {})
    rf_params = {
        'n_estimators': model_params.get('n_estimators', 100),
        'max_depth': model_params.get('max_depth', 10),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(feature_cols, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    # Save model and scaler
    os.makedirs('data/models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = f'data/models/rf_model_{timestamp}.pkl'
    scaler_path = f'data/models/scaler_{timestamp}.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save latest versions
    joblib.dump(model, 'data/models/rf_model_latest.pkl')
    joblib.dump(scaler, 'data/models/scaler_latest.pkl')
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("Latest versions saved as rf_model_latest.pkl and scaler_latest.pkl")
    
    return model, scaler, test_score

if __name__ == "__main__":
    train_model()
