import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestModel:

    def test_model_files_exist(self):
        """Test that model files exist"""
        assert os.path.exists("models/housing_model.pkl"), "Model file not found"
        assert os.path.exists("models/scaler.pkl"), "Scaler file not found"
        assert os.path.exists("metrics.json"), "Metrics file not found"

    def test_model_can_predict(self):
        """Test model can make predictions"""
        if not os.path.exists("models/housing_model.pkl"):
            pytest.skip("Model not trained yet")

        model = joblib.load("models/housing_model.pkl")
        scaler = joblib.load("models/scaler.pkl")

        # Create test sample
        test_data = np.array(
            [[5.0, 10.0, 6.0, 1.0, 3000.0, 3.0, 34.0, -118.0, 2.0, 0.17, 1000.0]]
        )
        test_scaled = scaler.transform(test_data)
        prediction = model.predict(test_scaled)[0]

        # California housing prices should be reasonable
        assert (
            0.5 <= prediction <= 5.0
        ), f"Prediction {prediction} out of reasonable range"

    def test_model_performance(self):
        """Test model meets minimum performance requirements"""
        if not os.path.exists("metrics.json"):
            pytest.skip("Metrics not available yet")

        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        # Minimum performance thresholds
        assert (
            metrics["test_r2"] >= 0.6
        ), f"R² score {metrics['test_r2']} below threshold"
        assert (
            metrics["test_rmse"] <= 1.0
        ), f"RMSE {metrics['test_rmse']} above threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
