from flask import Flask, render_template
import yaml
import json
import os

# Set template folder to parent directory's templates
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

class MetricsDict(dict):
    def __getattr__(self, key):
        return self.get(key, 'N/A')

@app.route('/')
def dashboard():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load metrics
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'metrics.json')
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            metrics = MetricsDict(metrics_data)
    except FileNotFoundError:
        metrics = MetricsDict({
            'accuracy': 'N/A',
            'precision': 'N/A',
            'recall': 'N/A',
            'f1_score': 'N/A',
            'lstm_rmse': 'N/A',
            'lstm_mae': 'N/A',
            'lstm_r2': 'N/A',
            'gru_rmse': 'N/A',
            'gru_mae': 'N/A',
            'gru_r2': 'N/A'
        })
    
    return render_template('dashboard.html', metrics=metrics, config=config)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
