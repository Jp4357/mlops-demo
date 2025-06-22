# src/features/build_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def engineer_features(df):
    """Create new features and prepare data"""
    print("🔧 Engineering features...")

    # Create new features
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
    df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"]

    # Handle outliers (cap at 95th percentile)
    for col in ["AveRooms", "AveBedrms", "Population", "AveOccup"]:
        cap_value = df[col].quantile(0.95)
        df[col] = np.where(df[col] > cap_value, cap_value, df[col])

    print(f"✅ Features engineered. New shape: {df.shape}")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Split and scale data"""
    print("📊 Preparing train/test splits...")

    # Separate features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for production use
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"✅ Data prepared:")
    print(f"  Training set: {X_train_scaled.shape[0]} samples")
    print(f"  Test set: {X_test_scaled.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)


if __name__ == "__main__":
    df = pd.read_csv("data/california_housing.csv")
    df_engineered = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df_engineered)
