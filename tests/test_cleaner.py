"""Tests for src.data_processing.cleaner.DataCleaner"""

import pandas as pd
import numpy as np
import pytest

from src.data_processing.cleaner import DataCleaner


class TestDataCleaner:
    """Test suite for the DataCleaner class."""

    def setup_method(self):
        self.cleaner = DataCleaner()

    # ── Initialization ────────────────────────────────

    def test_init_creates_empty_report(self):
        """Should initialize with an empty cleaning report."""
        assert self.cleaner.cleaning_report == {}

    # ── Data Quality Assessment ───────────────────────

    def test_assess_data_quality_returns_dict(self, sample_numeric_df):
        """Should return a dictionary with quality metrics."""
        report = self.cleaner.assess_data_quality(sample_numeric_df)
        assert isinstance(report, dict)

    def test_assess_data_quality_with_missing_values(self, sample_df_with_missing):
        """Should correctly detect missing values in assessment."""
        report = self.cleaner.assess_data_quality(sample_df_with_missing)
        assert isinstance(report, dict)

    # ── Column Standardization ────────────────────────

    def test_standardize_columns_lowercases(self):
        """Should lowercase all column names."""
        df = pd.DataFrame({"Age": [1], "SALARY": [2], "Name": [3]})
        result = self.cleaner._standardize_columns(df)
        for col in result.columns:
            assert col == col.lower()

    def test_standardize_columns_strips_whitespace(self):
        """Should strip leading/trailing whitespace from column names."""
        df = pd.DataFrame({" age ": [1], "salary  ": [2]})
        result = self.cleaner._standardize_columns(df)
        for col in result.columns:
            assert col == col.strip()

    # ── Numeric Conversion Detection ──────────────────

    def test_can_convert_to_numeric_true(self):
        """Should return True for a series with mostly numeric strings."""
        series = pd.Series(["1", "2", "3.5", "4", "5", "6", "7", "8", "9", "10"])
        assert self.cleaner._can_convert_to_numeric(series) is True

    def test_can_convert_to_numeric_false(self):
        """Should return False for a series with mostly text values."""
        series = pd.Series(["Alice", "Bob", "Charlie", "Diana", "Eve",
                            "Frank", "Grace", "Henry", "Iris", "Jack"])
        assert self.cleaner._can_convert_to_numeric(series) is False

    # ── Dtype Optimization ────────────────────────────

    def test_optimize_dtypes_reduces_memory(self, sample_numeric_df):
        """Should reduce memory usage after optimization."""
        original_mem = sample_numeric_df.memory_usage(deep=True).sum()
        optimized = self.cleaner._optimize_dtypes(sample_numeric_df.copy())
        optimized_mem = optimized.memory_usage(deep=True).sum()
        assert optimized_mem <= original_mem

    def test_optimize_dtypes_preserves_shape(self, sample_numeric_df):
        """Should not change the shape of the DataFrame."""
        result = self.cleaner._optimize_dtypes(sample_numeric_df.copy())
        assert result.shape == sample_numeric_df.shape

    # ── Clean Data Pipeline ───────────────────────────

    def test_clean_data_returns_dataframe(self, sample_df_with_missing):
        """Should return a DataFrame from the main cleaning pipeline."""
        result = self.cleaner.clean_data(sample_df_with_missing.copy())
        assert isinstance(result, pd.DataFrame)

    def test_clean_data_reduces_missing(self, sample_df_with_missing):
        """Should reduce the total number of missing values."""
        original_missing = sample_df_with_missing.isnull().sum().sum()
        result = self.cleaner.clean_data(sample_df_with_missing.copy())
        result_missing = result.isnull().sum().sum()
        assert result_missing <= original_missing
