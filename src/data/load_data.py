import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def load_california_housing():
    """Load and prepare California housing dataset"""
    print("🔄 Loading California Housing dataset...")

    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df, housing.feature_names, housing.target_names


def explore_data(df):
    """Basic data exploration"""
    print("\n📊 Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(
        f"Target range: ${df['MedHouseVal'].min():.1f}k - ${df['MedHouseVal'].max():.1f}k"
    )

    # Basic statistics
    print("\n📈 Target Statistics:")
    print(df["MedHouseVal"].describe())

    return df.describe()


def save_data(df, filepath="data/california_housing.csv"):
    """Save dataset for version control"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"💾 Data saved to {filepath}")


def main():
    df, features, target = load_california_housing()
    stats = explore_data(df)
    save_data(df)


if __name__ == "__main__":
    main()
