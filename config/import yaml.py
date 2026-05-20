import yaml

config = {
    'mqtt': {
        'broker': 'localhost',
        'port': 1883,
        'topics': {
            'sensor_data': 'pump/sensors',
            'alerts': 'pump/alerts',
            'predictions': 'pump/predictions'
        }
    },
    'sensors': {
        'vibration': {
            'normal_range': [1.5, 2.5],
            'warning_threshold': 3.0,
            'critical_threshold': 4.0,
            'unit': 'mm/s'
        },
        'temperature': {
            'normal_range': [60, 70],
            'warning_threshold': 80,
            'critical_threshold': 90,
            'unit': 'C'
        },
        'current': {
            'normal_range': [8.0, 9.0],
            'warning_threshold': 10.0,
            'critical_threshold': 11.0,
            'unit': 'A'
        },
        'pressure': {
            'normal_range': [145, 155],
            'warning_threshold': 140,
            'critical_threshold': 135,
            'unit': 'bar'
        }
    },
    'model': {
        'algorithm': 'RandomForest',
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'test_size': 0.2
    },
    'data_generation': {
        'samples_per_state': 1000,
        'normal_probability': 0.7,
        'degraded_probability': 0.2,
        'failure_probability': 0.1,
        'noise_factor': 0.05
    },
    'alerts': {
        'thresholds': {
            'info': 0.3,
            'warning': 0.5,
            'critical': 0.7
        }
    }
}

with open('config/config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("Config file created successfully!")
