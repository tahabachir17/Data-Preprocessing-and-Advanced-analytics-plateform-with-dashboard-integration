"""
Statistical Analysis Module
===========================
Provides comprehensive statistical analysis and hypothesis testing capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    A comprehensive statistical analyzer for hypothesis testing and advanced analytics.
    """
    
    def __init__(self):
        self.results_history = []
        self._stats = None  # Lazy-loaded scipy.stats

    @property
    def stats(self):
        """Lazy-load scipy.stats on first access."""
        if self._stats is None:
            from scipy import stats
            self._stats = stats
        return self._stats
    
    # ==========================================
    # NORMALITY TESTS
    # ==========================================
    
    def test_normality(self, data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform multiple normality tests on the data.
        
        Args:
            data: Series of numeric values
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary with test results and interpretation
        """
        clean_data = data.dropna()
        
        if len(clean_data) < 8:
            return {
                'error': 'Insufficient data for normality testing (need at least 8 observations)',
                'is_normal': None
            }
        
        results = {
            'sample_size': len(clean_data),
            'alpha': alpha,
            'tests': {}
        }
        
        # Shapiro-Wilk Test (best for small samples < 5000)
        if len(clean_data) <= 5000:
            try:
                stat, p_value = self.stats.shapiro(clean_data)
                results['tests']['shapiro_wilk'] = {
                    'statistic': round(stat, 6),
                    'p_value': round(p_value, 6),
                    'is_normal': p_value > alpha,
                    'interpretation': 'Data appears normally distributed' if p_value > alpha 
                                     else 'Data does not appear normally distributed'
                }
            except Exception as e:
                results['tests']['shapiro_wilk'] = {'error': str(e)}
        
        # D'Agostino-Pearson Test (good for larger samples)
        if len(clean_data) >= 20:
            try:
                stat, p_value = self.stats.normaltest(clean_data)
                results['tests']['dagostino_pearson'] = {
                    'statistic': round(stat, 6),
                    'p_value': round(p_value, 6),
                    'is_normal': p_value > alpha,
                    'interpretation': 'Data appears normally distributed' if p_value > alpha 
                                     else 'Data does not appear normally distributed'
                }
            except Exception as e:
                results['tests']['dagostino_pearson'] = {'error': str(e)}
        
        # Anderson-Darling Test
        try:
            result = self.stats.anderson(clean_data, dist='norm')
            # Use 5% significance level (index 2)
            critical_value = result.critical_values[2]
            is_normal = result.statistic < critical_value
            results['tests']['anderson_darling'] = {
                'statistic': round(result.statistic, 6),
                'critical_value_5pct': round(critical_value, 6),
                'is_normal': is_normal,
                'interpretation': 'Data appears normally distributed' if is_normal 
                                 else 'Data does not appear normally distributed'
            }
        except Exception as e:
            results['tests']['anderson_darling'] = {'error': str(e)}
        
        # Overall conclusion
        normal_votes = sum(1 for test in results['tests'].values() 
                         if isinstance(test, dict) and test.get('is_normal', False))
        total_tests = sum(1 for test in results['tests'].values() 
                         if isinstance(test, dict) and 'is_normal' in test)
        
        results['overall_conclusion'] = {
            'is_normal': normal_votes > total_tests / 2 if total_tests > 0 else None,
            'normal_tests': normal_votes,
            'total_tests': total_tests
        }
        
        return results
    
    # ==========================================
    # T-TESTS
    # ==========================================
    
    def one_sample_ttest(self, data: pd.Series, population_mean: float, 
                         alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform one-sample t-test.
        
        Args:
            data: Sample data
            population_mean: Hypothesized population mean
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'
        """
        clean_data = data.dropna()
        
        if len(clean_data) < 2:
            return {'error': 'Insufficient data for t-test'}
        
        stat, p_value = self.stats.ttest_1samp(clean_data, population_mean, alternative=alternative)
        
        sample_mean = clean_data.mean()
        sample_std = clean_data.std()
        se = sample_std / np.sqrt(len(clean_data))
        
        # Effect size (Cohen's d)
        cohens_d = (sample_mean - population_mean) / sample_std
        
        return {
            'test_type': 'One-Sample T-Test',
            'sample_size': len(clean_data),
            'sample_mean': round(sample_mean, 4),
            'population_mean': population_mean,
            'sample_std': round(sample_std, 4),
            'standard_error': round(se, 4),
            't_statistic': round(stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'alternative': alternative,
            'reject_null': p_value < alpha,
            'cohens_d': round(cohens_d, 4),
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The sample mean ({sample_mean:.4f}) is {'significantly' if p_value < alpha else 'not significantly'} "
                            f"different from {population_mean}."
        }
    
    def two_sample_ttest(self, group1: pd.Series, group2: pd.Series, 
                         alpha: float = 0.05, equal_var: bool = True,
                         alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform independent two-sample t-test.
        
        Args:
            group1: First group data
            group2: Second group data
            alpha: Significance level
            equal_var: Assume equal variances (True for Student's t-test, False for Welch's)
            alternative: 'two-sided', 'less', or 'greater'
        """
        clean_g1 = group1.dropna()
        clean_g2 = group2.dropna()
        
        if len(clean_g1) < 2 or len(clean_g2) < 2:
            return {'error': 'Insufficient data in one or both groups'}
        
        stat, p_value = self.stats.ttest_ind(clean_g1, clean_g2, equal_var=equal_var, alternative=alternative)
        
        mean1, mean2 = clean_g1.mean(), clean_g2.mean()
        std1, std2 = clean_g1.std(), clean_g2.std()
        
        # Pooled standard deviation for Cohen's d
        n1, n2 = len(clean_g1), len(clean_g2)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Levene's test for equal variances
        levene_stat, levene_p = self.stats.levene(clean_g1, clean_g2)
        
        return {
            'test_type': "Student's T-Test" if equal_var else "Welch's T-Test",
            'group1_size': n1,
            'group2_size': n2,
            'group1_mean': round(mean1, 4),
            'group2_mean': round(mean2, 4),
            'group1_std': round(std1, 4),
            'group2_std': round(std2, 4),
            'mean_difference': round(mean1 - mean2, 4),
            't_statistic': round(stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'alternative': alternative,
            'reject_null': p_value < alpha,
            'cohens_d': round(cohens_d, 4),
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'levene_test': {
                'statistic': round(levene_stat, 4),
                'p_value': round(levene_p, 6),
                'equal_variances': levene_p > 0.05
            },
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The means are {'significantly' if p_value < alpha else 'not significantly'} different."
        }
    
    def paired_ttest(self, before: pd.Series, after: pd.Series, 
                     alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform paired t-test for related samples.
        """
        # Align data
        combined = pd.DataFrame({'before': before, 'after': after}).dropna()
        
        if len(combined) < 2:
            return {'error': 'Insufficient paired observations'}
        
        clean_before = combined['before']
        clean_after = combined['after']
        
        stat, p_value = self.stats.ttest_rel(clean_before, clean_after, alternative=alternative)
        
        differences = clean_after - clean_before
        mean_diff = differences.mean()
        std_diff = differences.std()
        
        # Cohen's d for paired samples
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        return {
            'test_type': 'Paired T-Test',
            'n_pairs': len(combined),
            'before_mean': round(clean_before.mean(), 4),
            'after_mean': round(clean_after.mean(), 4),
            'mean_difference': round(mean_diff, 4),
            'std_difference': round(std_diff, 4),
            't_statistic': round(stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'alternative': alternative,
            'reject_null': p_value < alpha,
            'cohens_d': round(cohens_d, 4),
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"There is {'a significant' if p_value < alpha else 'no significant'} difference between paired observations."
        }
    
    # ==========================================
    # ANOVA
    # ==========================================
    
    def one_way_anova(self, df: pd.DataFrame, value_column: str, 
                      group_column: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test.
        
        Args:
            df: DataFrame containing the data
            value_column: Column with numeric values
            group_column: Column with group labels
            alpha: Significance level
        """
        # Prepare groups
        groups = []
        group_names = df[group_column].unique()
        
        for name in group_names:
            group_data = df[df[group_column] == name][value_column].dropna()
            if len(group_data) >= 2:
                groups.append(group_data)
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups with sufficient data'}
        
        # Perform ANOVA
        f_stat, p_value = self.stats.f_oneway(*groups)
        
        # Calculate group statistics
        group_stats = []
        for i, name in enumerate(group_names):
            if i < len(groups):
                group_stats.append({
                    'group': str(name),
                    'n': len(groups[i]),
                    'mean': round(groups[i].mean(), 4),
                    'std': round(groups[i].std(), 4)
                })
        
        # Effect size (eta-squared)
        all_data = pd.concat(groups)
        ss_between = sum(len(g) * (g.mean() - all_data.mean())**2 for g in groups)
        ss_total = sum((all_data - all_data.mean())**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Levene's test for homogeneity of variances
        levene_stat, levene_p = self.stats.levene(*groups)
        
        result = {
            'test_type': 'One-Way ANOVA',
            'n_groups': len(groups),
            'group_statistics': group_stats,
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'eta_squared': round(eta_squared, 4),
            'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
            'levene_test': {
                'statistic': round(levene_stat, 4),
                'p_value': round(levene_p, 6),
                'equal_variances': levene_p > 0.05
            },
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"There is {'a significant' if p_value < alpha else 'no significant'} difference between group means."
        }
        
        # Post-hoc test (Tukey HSD) if significant
        if p_value < alpha and len(groups) > 2:
            try:
                from scipy.stats import tukey_hsd
                tukey_result = tukey_hsd(*groups)
                result['post_hoc'] = {
                    'test': 'Tukey HSD',
                    'note': 'Pairwise comparisons available'
                }
            except:
                result['post_hoc'] = {'note': 'Post-hoc test not available'}
        
        return result
    
    # ==========================================
    # CHI-SQUARE TESTS
    # ==========================================
    
    def chi_square_independence(self, df: pd.DataFrame, column1: str, 
                                column2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform chi-square test for independence between two categorical variables.
        """
        # Create contingency table
        contingency_table = pd.crosstab(df[column1], df[column2])
        
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return {'error': 'Need at least 2 categories in each variable'}
        
        chi2, p_value, dof, expected = self.stats.chi2_contingency(contingency_table)
        
        # Effect size (Cramér's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'test_type': 'Chi-Square Test of Independence',
            'variable1': column1,
            'variable2': column2,
            'contingency_table_shape': contingency_table.shape,
            'chi_square_statistic': round(chi2, 4),
            'degrees_of_freedom': dof,
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'cramers_v': round(cramers_v, 4),
            'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
            'contingency_table': contingency_table.to_dict(),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The variables are {'significantly' if p_value < alpha else 'not significantly'} associated."
        }
    
    def chi_square_goodness_of_fit(self, observed: pd.Series, 
                                   expected: Optional[List[float]] = None,
                                   alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform chi-square goodness-of-fit test.
        """
        obs_counts = observed.value_counts()
        
        if expected is None:
            # Test against uniform distribution
            expected = [len(observed) / len(obs_counts)] * len(obs_counts)
        
        if len(obs_counts) != len(expected):
            return {'error': 'Observed and expected frequencies must have same length'}
        
        chi2, p_value = self.stats.chisquare(obs_counts.values, f_exp=expected)
        
        return {
            'test_type': 'Chi-Square Goodness-of-Fit Test',
            'n_categories': len(obs_counts),
            'observed_frequencies': obs_counts.to_dict(),
            'expected_frequencies': dict(zip(obs_counts.index, expected)),
            'chi_square_statistic': round(chi2, 4),
            'degrees_of_freedom': len(obs_counts) - 1,
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The distribution {'differs' if p_value < alpha else 'does not differ'} significantly from expected."
        }
    
    # ==========================================
    # NON-PARAMETRIC TESTS
    # ==========================================
    
    def mann_whitney_u(self, group1: pd.Series, group2: pd.Series, 
                       alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to two-sample t-test).
        """
        clean_g1 = group1.dropna()
        clean_g2 = group2.dropna()
        
        if len(clean_g1) < 2 or len(clean_g2) < 2:
            return {'error': 'Insufficient data in one or both groups'}
        
        stat, p_value = self.stats.mannwhitneyu(clean_g1, clean_g2, alternative=alternative)
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(clean_g1), len(clean_g2)
        r = 1 - (2 * stat) / (n1 * n2)
        
        return {
            'test_type': 'Mann-Whitney U Test',
            'group1_size': n1,
            'group2_size': n2,
            'group1_median': round(clean_g1.median(), 4),
            'group2_median': round(clean_g2.median(), 4),
            'u_statistic': round(stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'alternative': alternative,
            'reject_null': p_value < alpha,
            'rank_biserial_r': round(r, 4),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The distributions are {'significantly' if p_value < alpha else 'not significantly'} different."
        }
    
    def wilcoxon_signed_rank(self, x: pd.Series, y: pd.Series = None, 
                             alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        """
        if y is not None:
            # Paired comparison
            combined = pd.DataFrame({'x': x, 'y': y}).dropna()
            differences = combined['y'] - combined['x']
        else:
            differences = x.dropna()
        
        # Remove zeros
        differences = differences[differences != 0]
        
        if len(differences) < 2:
            return {'error': 'Insufficient non-zero differences'}
        
        stat, p_value = self.stats.wilcoxon(differences, alternative=alternative)
        
        return {
            'test_type': 'Wilcoxon Signed-Rank Test',
            'n_observations': len(differences),
            'median_difference': round(differences.median(), 4),
            'w_statistic': round(stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'alternative': alternative,
            'reject_null': p_value < alpha,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The median {'is' if p_value < alpha else 'is not'} significantly different from zero."
        }
    
    def kruskal_wallis(self, df: pd.DataFrame, value_column: str, 
                       group_column: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA).
        """
        groups = []
        group_names = df[group_column].unique()
        
        for name in group_names:
            group_data = df[df[group_column] == name][value_column].dropna()
            if len(group_data) >= 2:
                groups.append(group_data)
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups with sufficient data'}
        
        h_stat, p_value = self.stats.kruskal(*groups)
        
        # Group statistics
        group_stats = []
        for i, name in enumerate(group_names):
            if i < len(groups):
                group_stats.append({
                    'group': str(name),
                    'n': len(groups[i]),
                    'median': round(groups[i].median(), 4)
                })
        
        return {
            'test_type': 'Kruskal-Wallis H Test',
            'n_groups': len(groups),
            'group_statistics': group_stats,
            'h_statistic': round(h_stat, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis. "
                            f"The distributions are {'significantly' if p_value < alpha else 'not significantly'} different."
        }
    
    # ==========================================
    # OUTLIER DETECTION
    # ==========================================
    
    def detect_outliers_iqr(self, data: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers using the IQR method.
        """
        clean_data = data.dropna()
        
        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers_mask = (clean_data < lower_bound) | (clean_data > upper_bound)
        outliers = clean_data[outliers_mask]
        
        return {
            'method': 'IQR',
            'multiplier': multiplier,
            'q1': round(q1, 4),
            'q3': round(q3, 4),
            'iqr': round(iqr, 4),
            'lower_bound': round(lower_bound, 4),
            'upper_bound': round(upper_bound, 4),
            'n_outliers': len(outliers),
            'outlier_percentage': round(len(outliers) / len(clean_data) * 100, 2),
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist()
        }
    
    def detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        """
        clean_data = data.dropna()
        
        mean = clean_data.mean()
        std = clean_data.std()
        
        if std == 0:
            return {'error': 'Standard deviation is zero, cannot compute z-scores'}
        
        z_scores = (clean_data - mean) / std
        outliers_mask = np.abs(z_scores) > threshold
        outliers = clean_data[outliers_mask]
        
        return {
            'method': 'Z-Score',
            'threshold': threshold,
            'mean': round(mean, 4),
            'std': round(std, 4),
            'n_outliers': len(outliers),
            'outlier_percentage': round(len(outliers) / len(clean_data) * 100, 2),
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist(),
            'outlier_z_scores': z_scores[outliers_mask].tolist()
        }
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "Negligible effect"
        elif d < 0.5:
            return "Small effect"
        elif d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            return "Negligible effect"
        elif eta_sq < 0.06:
            return "Small effect"
        elif eta_sq < 0.14:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "Negligible association"
        elif v < 0.3:
            return "Weak association"
        elif v < 0.5:
            return "Moderate association"
        else:
            return "Strong association"
    
    def get_skewness_kurtosis(self, data: pd.Series) -> Dict[str, Any]:
        """
        Calculate skewness and kurtosis with interpretations.
        """
        clean_data = data.dropna()
        
        skewness = self.stats.skew(clean_data)
        kurtosis = self.stats.kurtosis(clean_data)
        
        # Skewness interpretation
        if abs(skewness) < 0.5:
            skew_interp = "Approximately symmetric"
        elif skewness > 0:
            skew_interp = "Right-skewed (positive skew)" if skewness < 1 else "Highly right-skewed"
        else:
            skew_interp = "Left-skewed (negative skew)" if skewness > -1 else "Highly left-skewed"
        
        # Kurtosis interpretation (excess kurtosis)
        if abs(kurtosis) < 0.5:
            kurt_interp = "Mesokurtic (normal-like tails)"
        elif kurtosis > 0:
            kurt_interp = "Leptokurtic (heavy tails, more outliers)"
        else:
            kurt_interp = "Platykurtic (light tails, fewer outliers)"
        
        return {
            'n': len(clean_data),
            'skewness': round(skewness, 4),
            'skewness_interpretation': skew_interp,
            'kurtosis': round(kurtosis, 4),
            'kurtosis_interpretation': kurt_interp
        }
