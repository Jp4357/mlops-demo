import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestData:

    def test_data_exists(self):
        """Test that data file exists"""
        assert os.path.exists("data/california_housing.csv"), "Data file not found"

    @pytest.fixture
    def data(self):
        return pd.read_csv("data/california_housing.csv")

    def test_data_shape(self, data):
        """Test data has expected shape"""
        assert data.shape[0] > 20000, "Dataset too small"
        assert data.shape[1] >= 9, "Missing features"

    def test_no_missing_values(self, data):
        """Test no missing values"""
        assert data.isnull().sum().sum() == 0, "Missing values found"

    def test_target_range(self, data):
        """Test target values are reasonable"""
        assert data["MedHouseVal"].min() > 0, "Negative house values found"
        assert data["MedHouseVal"].max() < 10, "Unreasonably high house values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
