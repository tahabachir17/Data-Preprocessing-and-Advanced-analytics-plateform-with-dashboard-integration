"""
Shared pytest fixtures for the DataFlow Pro test suite.
"""

import pandas as pd
import numpy as np
import pytest
import tempfile
import os


# ─── Sample DataFrames ───────────────────────────────

@pytest.fixture
def sample_numeric_df():
    """DataFrame with only numeric columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(18, 70, size=100),
        "salary": np.random.uniform(30000, 120000, size=100).round(2),
        "score": np.random.normal(75, 10, size=100).round(2),
        "experience": np.random.randint(0, 40, size=100),
    })


@pytest.fixture
def sample_mixed_df():
    """DataFrame with mixed numeric and categorical columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "name": [f"Person_{i}" for i in range(n)],
        "city": np.random.choice(["Paris", "London", "Berlin", "Madrid", "Rome"], size=n),
        "age": np.random.randint(18, 70, size=n),
        "salary": np.random.uniform(30000, 120000, size=n).round(2),
        "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR"], size=n),
        "is_active": np.random.choice([True, False], size=n),
    })


@pytest.fixture
def sample_df_with_missing():
    """DataFrame with missing values in various columns."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "col_a": np.random.randint(1, 100, size=n).astype(float),
        "col_b": np.random.choice(["X", "Y", "Z", None], size=n),
        "col_c": np.random.normal(0, 1, size=n),
        "col_d": np.random.uniform(0, 100, size=n),
    })
    # Inject missing values
    df.loc[0:4, "col_a"] = np.nan
    df.loc[10:14, "col_c"] = np.nan
    df.loc[20:24, "col_d"] = np.nan
    return df


@pytest.fixture
def sample_csv_file(sample_mixed_df):
    """Create a temporary CSV file from the mixed DataFrame."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        sample_mixed_df.to_csv(f, index=False)
        filepath = f.name
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def sample_excel_file(sample_mixed_df):
    """Create a temporary Excel file from the mixed DataFrame."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        filepath = f.name
    sample_mixed_df.to_excel(filepath, index=False)
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def sample_grouped_df():
    """DataFrame suitable for grouping operations."""
    np.random.seed(42)
    return pd.DataFrame({
        "region": np.random.choice(["North", "South", "East", "West"], size=80),
        "product": np.random.choice(["A", "B", "C"], size=80),
        "revenue": np.random.uniform(100, 10000, size=80).round(2),
        "quantity": np.random.randint(1, 100, size=80),
    })
