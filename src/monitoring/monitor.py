# src/monitoring/monitor.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging
import json
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class ModelMonitor:

    def __init__(self, model_path="models/housing_model.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load("models/scaler.pkl")
        self.baseline_metrics = self.load_baseline_metrics()

    def load_baseline_metrics(self):
        """Load baseline metrics from training"""
        with open("metrics.json", "r") as f:
            return json.load(f)

    def detect_data_drift(self, new_data, reference_data):
        """Detect if new data distribution differs from training data"""
        drift_scores = {}

        for column in new_data.columns:
            if column in reference_data.columns:
                # Simple drift detection using mean and std comparison
                ref_mean = reference_data[column].mean()
                ref_std = reference_data[column].std()
                new_mean = new_data[column].mean()
                new_std = new_data[column].std()

                # Calculate drift score
                mean_drift = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0
                std_drift = abs(new_std - ref_std) / ref_std if ref_std > 0 else 0

                drift_scores[column] = max(mean_drift, std_drift)

        return drift_scores

    def monitor_model_performance(self, predictions, actuals):
        """Monitor model performance over time"""
        current_metrics = {
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "r2": r2_score(actuals, predictions),
            "timestamp": datetime.now().isoformat(),
        }

        # Compare with baseline
        performance_degradation = {
            "rmse_change": current_metrics["rmse"] - self.baseline_metrics["test_rmse"],
            "r2_change": current_metrics["r2"] - self.baseline_metrics["test_r2"],
        }

        # Alert if performance degrades significantly
        if performance_degradation["rmse_change"] > 0.1:
            logging.warning(
                f"Model RMSE increased by {performance_degradation['rmse_change']:.3f}"
            )

        if performance_degradation["r2_change"] < -0.05:
            logging.warning(
                f"Model R² decreased by {abs(performance_degradation['r2_change']):.3f}"
            )

        return current_metrics, performance_degradation

    def log_prediction(self, features, prediction, actual=None):
        """Log individual prediction for monitoring"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": float(prediction),
            "actual": float(actual) if actual is not None else None,
        }

        # In production, you'd send this to a logging service
        logging.info(f"Prediction logged: {log_entry}")

        return log_entry


# Usage example
def run_monitoring():
    """Example monitoring workflow"""
    monitor = ModelMonitor()

    # Simulate new data and predictions
    new_data = pd.read_csv("data/california_housing.csv").sample(100)
    reference_data = pd.read_csv("data/california_housing.csv")

    # Check for data drift
    drift_scores = monitor.detect_data_drift(new_data, reference_data)
    print("🔍 Data Drift Scores:")
    for feature, score in drift_scores.items():
        print(f"  {feature}: {score:.3f}")
        if score > 1.0:  # Threshold for significant drift
            print(f"  ⚠️  High drift detected in {feature}")

    # Monitor performance (if we have actuals)
    # This would typically be done with production data
    print("\n📊 Model monitoring setup complete")


if __name__ == "__main__":
    run_monitoring()
