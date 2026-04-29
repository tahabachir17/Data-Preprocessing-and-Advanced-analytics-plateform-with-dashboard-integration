"""Tests for src.analytics.ml_models.MLModels regression pipeline."""

import pandas as pd
import numpy as np
import pytest

from src.analytics.ml_models import (
    MLModels,
    RegressionFeaturePreprocessor,
    SmartCategoricalEncoder,
)


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
        X, y, _ = ml.prepare_data_pipeline(regression_df, "target")
        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), (
                f"Column '{col}' has non-numeric dtype {X[col].dtype}"
            )

    def test_compatibility_preprocessors_return_numeric_frames(self, regression_df):
        """Compatibility preprocessors should remain importable and numeric."""
        raw_features = regression_df.drop(columns=["target"])
        target = regression_df["target"]

        categorical_encoder = SmartCategoricalEncoder()
        encoded = categorical_encoder.fit_transform(raw_features, target)

        regression_preprocessor = RegressionFeaturePreprocessor(correlation_threshold=0.01)
        prepared = regression_preprocessor.fit_transform(encoded, target)

        assert not prepared.empty
        assert all(np.issubdtype(prepared[col].dtype, np.number) for col in prepared.columns)

    def test_pipeline_accepts_optional_raw_inputs(self, regression_df):
        """The Streamlit training path passes raw_input_df and categorical_preprocessor."""
        ml = MLModels()
        results = ml.full_regression_pipeline(
            df=regression_df,
            target_column="target",
            cv_folds=3,
            save_model=False,
            raw_input_df=regression_df.copy(),
            categorical_preprocessor=SmartCategoricalEncoder(),
        )

        assert results["best_model_name"]

    def test_optional_all_zero_row_filter_removes_only_fully_zero_rows(self):
        """Only rows filled entirely with zeros should be removed when enabled."""
        df = pd.DataFrame({
            "feature_a": [0, 0, 1, 0.01],
            "feature_b": [0, 0, 2, 0],
            "target": [0, 0, 3, 4],
        })

        ml = MLModels()
        filtered_df, removed_count = ml.remove_all_zero_rows(df)

        assert removed_count == 2
        assert len(filtered_df) == 2
        assert filtered_df.iloc[0]["feature_a"] == 1
        assert filtered_df.iloc[1]["feature_a"] == 0.01

    def test_optional_all_zero_row_filter_ignores_datetime_columns(self):
        """Datetime columns should not count against the all-zero row rule."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "feature_a": [0, 1, 0],
            "feature_b": [0, 0, 0],
            "target": [0, 2, 0],
        })

        ml = MLModels()
        filtered_df, removed_count = ml.remove_all_zero_rows(df)

        assert removed_count == 2
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["feature_a"] == 1

    def test_optional_all_zero_row_filter_ignores_datetime_like_text_columns(self):
        """Date/time columns stored as text should also be ignored."""
        df = pd.DataFrame({
            "date_time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
            "feature_a": [0, 0, 5],
            "feature_b": [0, 0, 0],
            "target": [0, 0, 9],
        })

        ml = MLModels()
        filtered_df, removed_count = ml.remove_all_zero_rows(df)

        assert removed_count == 2
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["feature_a"] == 5

    def test_optional_all_zero_row_filter_ignores_categorical_columns(self):
        """Categorical columns should not affect zero-only row detection."""
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "feature_a": [0, 0, 4],
            "feature_b": [0, np.nan, 0],
            "target": [0, 0, 7],
        })

        ml = MLModels()
        filtered_df, removed_count = ml.remove_all_zero_rows(df)

        assert removed_count == 2
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["category"] == "C"
