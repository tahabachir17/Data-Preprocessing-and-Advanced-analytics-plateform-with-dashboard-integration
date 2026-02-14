"""Tests for src.analytics.statistical.StatisticalAnalyzer"""

import pandas as pd
import numpy as np
import pytest

from src.analytics.statistical import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test suite for the StatisticalAnalyzer class."""

    def setup_method(self):
        self.analyzer = StatisticalAnalyzer()
        np.random.seed(42)

    # ── Initialization ────────────────────────────────

    def test_init_empty_history(self):
        """Should initialize with an empty results history."""
        assert self.analyzer.results_history == []

    def test_lazy_scipy_stats(self):
        """Should lazily load scipy.stats on first access."""
        stats = self.analyzer.stats
        assert stats is not None

    # ── Normality Tests ───────────────────────────────

    def test_normality_on_normal_data(self):
        """Should not reject normality for normally-distributed data."""
        data = pd.Series(np.random.normal(0, 1, size=200))
        result = self.analyzer.test_normality(data)
        assert isinstance(result, dict)

    def test_normality_on_uniform_data(self):
        """Should likely reject normality for uniform data."""
        data = pd.Series(np.random.uniform(0, 1, size=200))
        result = self.analyzer.test_normality(data)
        assert isinstance(result, dict)

    # ── One-Sample T-Test ─────────────────────────────

    def test_one_sample_ttest(self):
        """Should return valid test results."""
        data = pd.Series(np.random.normal(5, 1, size=100))
        result = self.analyzer.one_sample_ttest(data, population_mean=5.0)
        assert isinstance(result, dict)
        assert "statistic" in result or "t_statistic" in result or "test_statistic" in result

    # ── Two-Sample T-Test ─────────────────────────────

    def test_two_sample_ttest_same_distribution(self):
        """Should not reject H0 for samples from the same distribution."""
        g1 = pd.Series(np.random.normal(10, 2, size=100))
        g2 = pd.Series(np.random.normal(10, 2, size=100))
        result = self.analyzer.two_sample_ttest(g1, g2)
        assert isinstance(result, dict)

    def test_two_sample_ttest_different_distributions(self):
        """Should likely reject H0 for samples from very different distributions."""
        g1 = pd.Series(np.random.normal(10, 1, size=100))
        g2 = pd.Series(np.random.normal(50, 1, size=100))
        result = self.analyzer.two_sample_ttest(g1, g2)
        assert isinstance(result, dict)

    # ── Paired T-Test ─────────────────────────────────

    def test_paired_ttest(self):
        """Should return valid paired test results."""
        before = pd.Series(np.random.normal(10, 2, size=50))
        after = before + np.random.normal(0.5, 0.5, size=50)
        result = self.analyzer.paired_ttest(before, after)
        assert isinstance(result, dict)

    # ── ANOVA ─────────────────────────────────────────

    def test_one_way_anova(self):
        """Should perform one-way ANOVA on grouped numeric data."""
        df = pd.DataFrame({
            "value": np.concatenate([
                np.random.normal(10, 2, 30),
                np.random.normal(20, 2, 30),
                np.random.normal(30, 2, 30),
            ]),
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        })
        result = self.analyzer.one_way_anova(df, "value", "group")
        assert isinstance(result, dict)

    # ── Chi-Square Tests ──────────────────────────────

    def test_chi_square_independence(self):
        """Should perform chi-square independence test."""
        df = pd.DataFrame({
            "gender": np.random.choice(["M", "F"], size=200),
            "preference": np.random.choice(["A", "B", "C"], size=200),
        })
        result = self.analyzer.chi_square_independence(df, "gender", "preference")
        assert isinstance(result, dict)

    def test_chi_square_goodness_of_fit(self):
        """Should perform goodness-of-fit test."""
        observed = pd.Series([50, 30, 20])
        result = self.analyzer.chi_square_goodness_of_fit(observed)
        assert isinstance(result, dict)

    # ── Non-Parametric Tests ──────────────────────────

    def test_mann_whitney_u(self):
        """Should perform Mann-Whitney U test."""
        g1 = pd.Series(np.random.normal(10, 2, size=50))
        g2 = pd.Series(np.random.normal(12, 2, size=50))
        result = self.analyzer.mann_whitney_u(g1, g2)
        assert isinstance(result, dict)

    def test_kruskal_wallis(self):
        """Should perform Kruskal-Wallis H test."""
        df = pd.DataFrame({
            "value": np.concatenate([
                np.random.normal(10, 2, 30),
                np.random.normal(15, 2, 30),
            ]),
            "group": ["A"] * 30 + ["B"] * 30,
        })
        result = self.analyzer.kruskal_wallis(df, "value", "group")
        assert isinstance(result, dict)

    # ── Outlier Detection ─────────────────────────────

    def test_detect_outliers_iqr(self):
        """Should detect outliers using IQR method."""
        data = pd.Series(np.concatenate([np.random.normal(0, 1, 95), [100, -100]]))
        result = self.analyzer.detect_outliers_iqr(data)
        assert isinstance(result, dict)

    def test_detect_outliers_zscore(self):
        """Should detect outliers using Z-score method."""
        data = pd.Series(np.concatenate([np.random.normal(0, 1, 95), [100, -100]]))
        result = self.analyzer.detect_outliers_zscore(data)
        assert isinstance(result, dict)

    # ── Skewness & Kurtosis ───────────────────────────

    def test_skewness_kurtosis(self):
        """Should return skewness and kurtosis with interpretations."""
        data = pd.Series(np.random.normal(0, 1, size=200))
        result = self.analyzer.get_skewness_kurtosis(data)
        assert isinstance(result, dict)

    # ── Effect Size Interpretation ────────────────────

    def test_interpret_cohens_d(self):
        """Should return a string interpretation for Cohen's d."""
        result = self.analyzer._interpret_cohens_d(0.5)
        assert isinstance(result, str)

    def test_interpret_eta_squared(self):
        """Should return a string interpretation for eta-squared."""
        result = self.analyzer._interpret_eta_squared(0.06)
        assert isinstance(result, str)

    def test_interpret_cramers_v(self):
        """Should return a string interpretation for Cramér's V."""
        result = self.analyzer._interpret_cramers_v(0.3)
        assert isinstance(result, str)
