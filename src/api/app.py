from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and scaler at startup
model = joblib.load("models/housing_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Expected feature names
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "RoomsPerHousehold",
    "BedroomsPerRoom",
    "PopulationPerHousehold",
]


def engineer_features_api(data):
    """Apply same feature engineering as training"""
    df = pd.DataFrame([data])

    # Create engineered features
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
    df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"]

    return df[FEATURE_NAMES].iloc[0].to_dict()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Make house price prediction"""
    try:
        # Get input data
        data = request.json

        # Validate required fields
        required_fields = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        # Engineer features
        engineered_data = engineer_features_api(data)

        # Prepare features array
        features = [engineered_data[name] for name in FEATURE_NAMES]
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Log prediction for monitoring
        logger.info(f"Prediction made: {prediction:.3f} for input: {data}")

        # Return prediction
        return jsonify(
            {
                "predicted_price": round(prediction * 100000, 2),  # Convert to dollars
                "prediction_raw": float(prediction),
                "timestamp": datetime.now().isoformat(),
                "input_features": engineered_data,
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Make predictions for multiple samples"""
    try:
        data = request.json
        samples = data.get("samples", [])

        if not samples:
            return jsonify({"error": "No samples provided"}), 400

        predictions = []

        for sample in samples:
            engineered_data = engineer_features_api(sample)
            features = [engineered_data[name] for name in FEATURE_NAMES]
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]

            predictions.append(
                {"predicted_price": round(prediction * 100000, 2), "input": sample}
            )

        return jsonify(
            {
                "predictions": predictions,
                "count": len(predictions),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
