import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import os

def train_model():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    data = pd.read_csv(config['paths']['processed_data'])
    
    # Prepare features and target
    X = data[config['ml_model']['features']]
    y = data[config['ml_model']['target']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['ml_model']['test_size'],
        random_state=config['ml_model']['random_state']
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=config['ml_model']['model_params']['n_estimators'],
        max_depth=config['ml_model']['model_params']['max_depth'],
        random_state=config['ml_model']['random_state']
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Save model
    os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
    with open(config['paths']['model_output'], 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    os.makedirs(os.path.dirname(config['paths']['metrics_output']), exist_ok=True)
    with open(config['paths']['metrics_output'], 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print(f"Model saved to: {config['paths']['model_output']}")
    print(f"Metrics saved to: {config['paths']['metrics_output']}")

if __name__ == "__main__":
    train_model()
