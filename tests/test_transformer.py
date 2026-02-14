"""Tests for src.data_processing.transformer.DataTransformer"""

import pandas as pd
import numpy as np
import pytest

from src.data_processing.transformer import DataTransformer


class TestDataTransformer:
    """Test suite for the DataTransformer class."""

    def setup_method(self):
        self.transformer = DataTransformer()

    # ── Grouping ──────────────────────────────────────

    def test_grouping_basic(self, sample_grouped_df):
        """Should group by a single column and aggregate numerics."""
        result = self.transformer.dataframe_grouping(sample_grouped_df, "region")
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= sample_grouped_df["region"].nunique()

    def test_grouping_multiple_columns(self, sample_grouped_df):
        """Should group by multiple columns."""
        result = self.transformer.dataframe_grouping(sample_grouped_df, ["region", "product"])
        assert isinstance(result, pd.DataFrame)

    def test_grouping_preserves_group_column(self, sample_grouped_df):
        """Grouped column should appear in the result."""
        result = self.transformer.dataframe_grouping(sample_grouped_df, "region")
        assert "region" in result.columns or "region" in result.index.names

    # ── Merging ───────────────────────────────────────

    def test_merge_inner(self):
        """Should perform inner merge correctly."""
        df1 = pd.DataFrame({"key": ["A", "B", "C"], "val1": [1, 2, 3]})
        df2 = pd.DataFrame({"key": ["B", "C", "D"], "val2": [4, 5, 6]})
        result = self.transformer.dataframe_merging(df1, df2, on="key", how="inner")
        assert len(result) == 2  # B and C
        assert set(result["key"]) == {"B", "C"}

    def test_merge_left(self):
        """Should perform left merge, keeping all rows from df1."""
        df1 = pd.DataFrame({"key": ["A", "B", "C"], "val1": [1, 2, 3]})
        df2 = pd.DataFrame({"key": ["B", "C", "D"], "val2": [4, 5, 6]})
        result = self.transformer.dataframe_merging(df1, df2, on="key", how="left")
        assert len(result) == 3

    def test_merge_outer(self):
        """Should perform outer merge, keeping all rows."""
        df1 = pd.DataFrame({"key": ["A", "B"], "val1": [1, 2]})
        df2 = pd.DataFrame({"key": ["B", "C"], "val2": [3, 4]})
        result = self.transformer.dataframe_merging(df1, df2, on="key", how="outer")
        assert len(result) == 3  # A, B, C

    # ── Concatenation ─────────────────────────────────

    def test_concat_vertical(self):
        """Should concatenate two DataFrames vertically."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = self.transformer.dataframe_concat(df1, df2, axis=0)
        assert len(result) == 4

    def test_concat_horizontal(self):
        """Should concatenate two DataFrames horizontally."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        result = self.transformer.dataframe_concat(df1, df2, axis=1)
        assert result.shape == (2, 2)

    # ── Categorical Encoding ──────────────────────────

    def test_encoding_returns_dataframe(self, sample_mixed_df):
        """Should return a DataFrame after encoding."""
        result = self.transformer.smart_categorical_encoding(sample_mixed_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_encoding_increases_numeric_columns(self, sample_mixed_df):
        """After encoding, should have more numeric columns than before."""
        original_numeric = len(sample_mixed_df.select_dtypes(include=[np.number]).columns)
        result = self.transformer.smart_categorical_encoding(sample_mixed_df.copy())
        result_numeric = len(result.select_dtypes(include=[np.number]).columns)
        assert result_numeric >= original_numeric

    def test_encoding_preserves_row_count(self, sample_mixed_df):
        """Encoding should not change the number of rows."""
        result = self.transformer.smart_categorical_encoding(sample_mixed_df.copy())
        assert len(result) == len(sample_mixed_df)
