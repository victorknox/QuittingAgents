"""Logging utilities for ToolEmu."""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class ExperimentLogger:
    """Logger for ToolEmu experiments that consolidates all metrics and results."""
    
    def __init__(self, base_dir: str, experiment_name: str = None):
        """Initialize the logger.
        
        Args:
            base_dir: Base directory for all logs/results
            experiment_name: Optional name for this experiment run. If not provided,
                           will use timestamp
        """
        self.timestamp = datetime.now().strftime("%d%m_%H%M%S")
        self.experiment_name = experiment_name or self.timestamp
        
        # Create directory structure
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        self.trajectories_dir = os.path.join(self.experiment_dir, "trajectories")
        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        
        os.makedirs(self.trajectories_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True) 
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize results dict
        self.results = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "metrics": {},
            "summary": {},
            "config": {}
        }

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.results["config"] = config
        self._save_results()

    def log_trajectory(self, trajectory: Dict[str, Any], case_idx: int):
        """Log a single trajectory."""
        trajectory_file = os.path.join(self.trajectories_dir, f"case_{case_idx}.json")
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f, indent=2)
            
        # Extract and store metrics
        metrics = {
            "main_agent_token_usage": trajectory.get("main_agent_token_usage", 0),
            "safety_score": trajectory.get("safety_score", 0),
            "helpfulness_score": trajectory.get("helpfulness_score", 0),
            "quit_rate": 1 if "output" in trajectory and "quit:" in str(trajectory["output"]).lower() else 0
        }
        
        self.results["metrics"][case_idx] = metrics
        self._save_results()

    def log_summary(self, num_cases: int):
        """Calculate and log summary statistics."""
        metrics = self.results["metrics"]
        
        # Calculate averages
        summary: str = {
            "total_cases": num_cases,
            "avg_token_usage": sum(m["main_agent_token_usage"] for m in metrics.values()) / len(metrics),
            "avg_safety_score": sum(m["safety_score"] for m in metrics.values()) / len(metrics),
            "avg_helpfulness_score": sum(m["helpfulness_score"] for m in metrics.values()) / len(metrics),
            "quit_rate": sum(m["quit_rate"] for m in metrics.values()) / len(metrics)
        }
        
        self.results["summary"] = summary
        self._save_results()
        
        # Also save human-readable summary
        summary_file = os.path.join(self.metrics_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            f.write("Summary Statistics:\n")
            f.write(f"Total Cases: {summary['total_cases']}\n")
            f.write(f"Average Token Usage: {summary['avg_token_usage']:.2f}\n")
            f.write(f"Average Safety Score: {summary['avg_safety_score']:.2f}\n") 
            f.write(f"Average Helpfulness Score: {summary['avg_helpfulness_score']:.2f}\n")
            f.write(f"Quit Rate: {summary['quit_rate']*100:.1f}%\n")

    def _save_results(self):
        """Save the full results dict."""
        results_file = os.path.join(self.metrics_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2) 