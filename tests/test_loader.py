"""Tests for src.data_processing.loader.DataLoader"""

import pandas as pd
import pytest
import tempfile
import os

from src.data_processing.loader import DataLoader


class TestDataLoader:
    """Test suite for the DataLoader class."""

    def setup_method(self):
        self.loader = DataLoader()

    # ── CSV Loading ────────────────────────────────────

    def test_load_csv_basic(self, sample_csv_file):
        """Should load a CSV file and return a DataFrame."""
        df = self.loader.loader(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_csv_preserves_columns(self, sample_csv_file):
        """Should preserve expected column names."""
        df = self.loader.loader(sample_csv_file)
        expected = {"name", "city", "age", "salary", "department", "is_active"}
        assert expected.issubset(set(df.columns))

    def test_load_csv_strips_unnamed_columns(self):
        """Should remove 'Unnamed' columns from the result."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write(",col1,col2\n0,a,1\n1,b,2\n")
            filepath = f.name
        try:
            df = self.loader.loader(filepath)
            unnamed = [c for c in df.columns if "unnamed" in c.lower()]
            assert len(unnamed) == 0
        finally:
            os.unlink(filepath)

    # ── Excel Loading ──────────────────────────────────

    def test_load_excel_basic(self, sample_excel_file):
        """Should load an Excel file and return a DataFrame."""
        df = self.loader.loader(sample_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    # ── Error Handling ─────────────────────────────────

    def test_unsupported_format_raises(self):
        """Should raise ValueError for unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"key": "value"}')
            filepath = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                self.loader.loader(filepath)
        finally:
            os.unlink(filepath)

    # ── Keyword Row Detection ──────────────────────────

    def test_find_keyword_rows_returns_matching_indices(self):
        """Should find rows containing date-like keywords."""
        df = pd.DataFrame({
            "A": ["hello", "date", "world"],
            "B": [1, 2, 3],
        })
        result = self.loader.find_keyword_rows(df, ["date"])
        assert 1 in result

    def test_find_keyword_rows_case_insensitive(self):
        """Should match keywords regardless of case."""
        df = pd.DataFrame({
            "A": ["TIMESTAMP", "data", "info"],
            "B": [1, 2, 3],
        })
        result = self.loader.find_keyword_rows(df, ["timestamp"])
        assert 0 in result

    def test_find_keyword_rows_no_match(self):
        """Should return empty list when no keywords match."""
        df = pd.DataFrame({
            "A": ["hello", "world"],
            "B": [1, 2],
        })
        result = self.loader.find_keyword_rows(df, ["nonexistent"])
        assert result == []

    # ── Numeric Conversion ────────────────────────────

    def test_convert_object_columns_to_numeric(self):
        """Should convert mostly-numeric object columns to numeric dtype."""
        df = pd.DataFrame({
            "values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        })
        result = self.loader._convert_object_columns_to_numeric(df.copy())
        assert pd.api.types.is_numeric_dtype(result["values"])

    def test_convert_object_columns_keeps_text(self):
        """Should NOT convert columns with mostly text values."""
        df = pd.DataFrame({
            "names": ["Alice", "Bob", "Charlie", "Diana", "Eve",
                      "Frank", "Grace", "Henry", "Iris", "Jack"],
        })
        result = self.loader._convert_object_columns_to_numeric(df.copy())
        assert result["names"].dtype == object
