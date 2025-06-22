import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import sys
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def train_model(X_train, y_train, X_test, y_test, model_type="random_forest"):
    """Train model with simplified MLflow tracking"""

    print(f"🚀 Training {model_type} model...")

    # Simplified MLflow setup - just track metrics and params
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow_enabled = True
    except:
        print("⚠️ MLflow not available, continuing without tracking...")
        mlflow_enabled = False

    if mlflow_enabled:
        mlflow.start_run(
            run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

    try:
        # Choose model
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "model_type": "RandomForest",
            }
        else:
            model = LinearRegression()
            params = {"model_type": "LinearRegression"}

        # Log hyperparameters (if MLflow works)
        if mlflow_enabled:
            try:
                mlflow.log_params(params)
            except:
                print("⚠️ Could not log parameters to MLflow")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
        }

        # Log metrics (if MLflow works)
        if mlflow_enabled:
            try:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                print("✅ Metrics logged to MLflow")
            except:
                print("⚠️ Could not log metrics to MLflow")

        # Save model locally (this always works)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/housing_model.pkl")

        # Save metrics for CI/CD (this always works)
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("✅ Model training completed!")
        print(f"  Test RMSE: {metrics['test_rmse']:.3f}")
        print(f"  Test R²: {metrics['test_r2']:.3f}")

        return model, metrics

    finally:
        if mlflow_enabled:
            try:
                mlflow.end_run()
            except:
                pass


def compare_models(X_train, y_train, X_test, y_test):
    """Compare different models"""
    print("🔍 Comparing models...")

    models = ["linear_regression", "random_forest"]
    results = {}

    for model_type in models:
        model, metrics = train_model(X_train, y_train, X_test, y_test, model_type)
        results[model_type] = metrics

    return results


def main():
    # Import here to avoid circular imports
    try:
        from src.features.build_features import prepare_data, engineer_features
    except ImportError:
        # If running from project root
        sys.path.insert(0, ".")
        from src.features.build_features import prepare_data, engineer_features

    # Load and prepare data
    if not os.path.exists("data/california_housing.csv"):
        print("❌ Data file not found. Please run 'python src/data/load_data.py' first")
        return

    df = pd.read_csv("data/california_housing.csv")
    df_engineered = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df_engineered)

    # Train and compare models
    results = compare_models(X_train, y_train, X_test, y_test)

    print("\n🏆 Model Comparison Results:")
    for model_name, metrics in results.items():
        print(
            f"{model_name}: R² = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
