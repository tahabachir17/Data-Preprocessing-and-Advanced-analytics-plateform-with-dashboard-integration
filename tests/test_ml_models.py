"""Tests for src.analytics.ml_models.MLModels regression pipeline."""

import pandas as pd
import numpy as np
import pytest

from src.analytics.ml_models import MLModels


@pytest.fixture
def regression_df():
    """DataFrame with numeric, boolean, and categorical columns for regression."""
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    noise = np.random.normal(0, 0.5, n)
    return pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "bool_flag": np.random.choice([True, False], size=n),
        "category": np.random.choice(["A", "B", "C"], size=n),
        "target": 3 * x1 + 2 * x2 + noise,
    })


class TestMLModelsTraining:
    """Verify that the regression pipeline trains successfully."""

    def test_full_pipeline_completes(self, regression_df):
        """Pipeline should finish without raising on mixed-type data."""
        ml = MLModels()
        results = ml.full_regression_pipeline(
            df=regression_df,
            target_column="target",
            correlation_threshold=0.01,
            test_size=0.2,
            cv_folds=3,
            save_model=False,
        )
        assert results is not None
        assert "best_model" in results
        assert "best_model_name" in results
        assert results["best_score"] is not None

    def test_pipeline_returns_valid_r2(self, regression_df):
        """Best R² score should be positive for a well-correlated dataset."""
        ml = MLModels()
        results = ml.full_regression_pipeline(
            df=regression_df,
            target_column="target",
            cv_folds=3,
            save_model=False,
        )
        assert results["best_score"] > 0

    def test_prepare_data_all_numeric(self, regression_df):
        """After prepare_data_pipeline, all columns should be numeric."""
        ml = MLModels()
        X, y = ml.prepare_data_pipeline(regression_df, "target")
        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), (
                f"Column '{col}' has non-numeric dtype {X[col].dtype}"
            )
