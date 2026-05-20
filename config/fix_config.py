import os

os.chdir(r'C:\Users\Lion\OneDrive\Desktop\Synthetic')

# Delete the corrupted file first
if os.path.exists('config/config.yaml'):
    os.remove('config/config.yaml')

# Create fresh config with proper YAML structure
config_lines = [
    "mqtt:",
    "  broker: localhost",
    "  port: 1883",
    "  topic: pump/sensors",
    "",
    "sensors:",
    "  vibration:",
    "    unit: mm/s",
    "    normal_range: [0.5, 2.5]",
    "    warning_threshold: 4.5",
    "    critical_threshold: 7.0",
    "  temperature:",
    "    unit: C",
    "    normal_range: [20, 60]",
    "    warning_threshold: 75",
    "    critical_threshold: 85",
    "  current:",
    "    unit: A",
    "    normal_range: [8, 12]",
    "    warning_threshold: 15",
    "    critical_threshold: 18",
    "  pressure:",
    "    unit: bar",
    "    normal_range: [2.0, 3.5]",
    "    warning_threshold: 4.5",
    "    critical_threshold: 5.5",
    "",
    "model:",
    "  type: RandomForest",
    "  n_estimators: 100",
    "  max_depth: 10",
    "  random_state: 42",
    "  test_size: 0.2",
    "",
    "data_generation:",
    "  samples_per_state: 1000",
    "  normal_probability: 0.7",
    "  degraded_probability: 0.2",
    "  failure_probability: 0.1",
    "  noise_factor: 0.1",
    "",
    "alerts:",
    "  warning_threshold: 0.6",
    "  critical_threshold: 0.8",
]

with open('config/config.yaml', 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(config_lines))

print("Config file recreated successfully!")

# Verify it can be parsed
import yaml
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    print("Config validation passed!")
    print(f"Loaded {len(config)} top-level keys: {list(config.keys())}")
