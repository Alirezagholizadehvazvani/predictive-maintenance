from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime
from pathlib import Path

# File is in src/, so go up one level to Synthetic/
BASE_DIR = Path(__file__).parent.parent

app = Flask(__name__, template_folder=BASE_DIR / 'templates')

def load_metrics():
    """Load metrics from JSON file"""
    metrics_file = BASE_DIR / 'data' / 'processed' / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                # Handle both array and object formats
                if isinstance(data, list):
                    metrics_list = data
                elif isinstance(data, dict) and 'value' in data:
                    metrics_list = data['value']
                else:
                    metrics_list = []
                
                if metrics_list:
                    current = metrics_list[-1]  # Most recent metrics
                    history = metrics_list
                    return current, history
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    
    # Return default values if file doesn't exist or is empty
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }, []

def load_predictions():
    """Load predictions from JSON file"""
    predictions_file = BASE_DIR / 'data' / 'processed' / 'predictions.json'
    if predictions_file.exists():
        try:
            with open(predictions_file, 'r') as f:
                data = json.load(f)
                # Handle both array and object formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'value' in data:
                    return data['value']
        except (json.JSONDecodeError, KeyError):
            pass
    return []

@app.route('/')
def dashboard():
    """Main dashboard page"""
    current_metrics, metrics_history = load_metrics()
    predictions = load_predictions()
    
    # Limit data for display
    recent_predictions = predictions[-10:] if len(predictions) > 10 else predictions
    plot_data = predictions[-200:] if len(predictions) > 200 else predictions
    
    return render_template('dashboard.html',
                         metrics=current_metrics,
                         metrics_history=metrics_history,
                         predictions=recent_predictions,
                         plot_data=plot_data)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics data"""
    current_metrics, metrics_history = load_metrics()
    return jsonify({
        'current': current_metrics,
        'history': metrics_history
    })

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for predictions data"""
    predictions = load_predictions()
    # Limit to last 200 for performance
    limited_predictions = predictions[-200:] if len(predictions) > 200 else predictions
    return jsonify(limited_predictions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
