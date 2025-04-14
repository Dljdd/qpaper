# src/utils/experiment_tracker.py
import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiments_dir='./experiments'):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
    
    def log_experiment(self, config, results):
        """
        Log experiment configuration and results
        
        Args:
            config: Dictionary of experiment configuration
            results: Dictionary of experiment results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
        
        experiment_data = {
            "config": config,
            "results": results,
            "timestamp": timestamp
        }
        
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_name}.json")
        with open(experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Experiment logged: {experiment_path}")
    
    def load_experiment(self, experiment_name):
        """
        Load experiment data
        
        Args:
            experiment_name: Name of the experiment file (without .json extension)
        
        Returns:
            Dictionary of experiment data
        """
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_name}.json")
        with open(experiment_path, 'r') as f:
            experiment_data = json.load(f)
        return experiment_data
    
    def list_experiments(self):
        """
        List all experiments in the experiments directory
        
        Returns:
            List of experiment names (without .json extension)
        """
        experiments = [f[:-5] for f in os.listdir(self.experiments_dir) if f.endswith('.json')]
        return sorted(experiments)
