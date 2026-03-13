"""
Run manifest generator for reproducible experiments.
"""
import json
import os
import subprocess
from datetime import datetime

def create_run_directory(config, base_dir='runs'):
    """Create a standardized run directory with config.json."""
    
    run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_L{config['L']}_chi{config['chi']}_seed{config['seed']}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Get git commit
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                         stderr=subprocess.DEVNULL).decode().strip()
    except:
        commit = 'unknown'
    
    # Add metadata
    config_full = {
        **config,
        'commit': commit,
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id
    }
    
    # Write config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_full, f, indent=2)
    
    return run_dir, run_id

def save_metrics(run_dir, metrics):
    """Save metrics.json to run directory."""
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

def save_log(run_dir, log_text):
    """Save logs.txt to run directory."""
    with open(os.path.join(run_dir, 'logs.txt'), 'w') as f:
        f.write(log_text)
