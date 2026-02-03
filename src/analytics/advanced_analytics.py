"""
Advanced Analytics Module
=========================
Provides advanced data analysis, profiling, and quality reporting capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedAnalytics:
    """
    Advanced analytics for data quality, profiling, and comprehensive analysis.
    """
    
    def __init__(self):
        pass
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics and issues
        """
        n_rows, n_cols = df.shape
        
        # Column-level analysis
        column_analysis = []
        for col in df.columns:
            col_info = {
                'column': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': round(df[col].isnull().sum() / n_rows * 100, 2),
                'unique_count': df[col].nunique(),
                'unique_percentage': round(df[col].nunique() / n_rows * 100, 2)
            }
            
            # Check for potential issues
            issues = []
            
            # High missing values
            if col_info['null_percentage'] > 50:
                issues.append('High missing values (>50%)')
            elif col_info['null_percentage'] > 20:
                issues.append('Moderate missing values (>20%)')
            
            # Low cardinality check
            if col_info['unique_count'] == 1:
                issues.append('Constant column (single value)')
            elif col_info['unique_percentage'] < 1 and n_rows > 100:
                issues.append('Very low cardinality')
            
            # High cardinality for non-numeric
            if df[col].dtype == 'object' and col_info['unique_percentage'] > 95:
                issues.append('Potentially unique identifier')
            
            col_info['issues'] = issues
            col_info['has_issues'] = len(issues) > 0
            
            column_analysis.append(col_info)
        
        # Overall quality score (0-100)
        completeness_score = (1 - df.isnull().sum().sum() / (n_rows * n_cols)) * 100
        uniqueness_issues = sum(1 for c in column_analysis if c['unique_count'] == 1)
        uniqueness_score = (1 - uniqueness_issues / n_cols) * 100
        
        overall_score = (completeness_score * 0.6 + uniqueness_score * 0.4)
        
        return {
            'overview': {
                'total_rows': n_rows,
                'total_columns': n_cols,
                'total_cells': n_rows * n_cols,
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': round(df.isnull().sum().sum() / (n_rows * n_cols) * 100, 2),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            },
            'quality_score': {
                'overall': round(overall_score, 1),
                'completeness': round(completeness_score, 1),
                'uniqueness': round(uniqueness_score, 1),
                'grade': self._get_quality_grade(overall_score)
            },
            'column_types': {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                'boolean': len(df.select_dtypes(include=['bool']).columns)
            },
            'column_analysis': column_analysis,
            'issues_summary': {
                'columns_with_issues': sum(1 for c in column_analysis if c['has_issues']),
                'high_missing_columns': sum(1 for c in column_analysis if c['null_percentage'] > 50),
                'constant_columns': sum(1 for c in column_analysis if c['unique_count'] == 1)
            }
        }
    
    def generate_feature_profile(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Generate detailed profiles for each feature/column.
        """
        numeric_profiles = []
        categorical_profiles = []
        
        # Numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            profile = {
                'column': col,
                'dtype': str(df[col].dtype),
                'count': len(data),
                'mean': round(data.mean(), 4),
                'std': round(data.std(), 4),
                'min': round(data.min(), 4),
                'q1': round(data.quantile(0.25), 4),
                'median': round(data.median(), 4),
                'q3': round(data.quantile(0.75), 4),
                'max': round(data.max(), 4),
                'range': round(data.max() - data.min(), 4),
                'iqr': round(data.quantile(0.75) - data.quantile(0.25), 4),
                'skewness': round(data.skew(), 4),
                'kurtosis': round(data.kurtosis(), 4),
                'zeros_count': int((data == 0).sum()),
                'zeros_percentage': round((data == 0).sum() / len(data) * 100, 2),
                'negative_count': int((data < 0).sum()),
                'negative_percentage': round((data < 0).sum() / len(data) * 100, 2)
            }
            
            # Distribution shape
            if abs(profile['skewness']) < 0.5:
                profile['distribution_shape'] = 'Symmetric'
            elif profile['skewness'] > 0:
                profile['distribution_shape'] = 'Right-skewed'
            else:
                profile['distribution_shape'] = 'Left-skewed'
            
            numeric_profiles.append(profile)
        
        # Categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            value_counts = data.value_counts()
            
            profile = {
                'column': col,
                'dtype': str(df[col].dtype),
                'count': len(data),
                'unique_count': data.nunique(),
                'unique_percentage': round(data.nunique() / len(data) * 100, 2),
                'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'mode_percentage': round(value_counts.iloc[0] / len(data) * 100, 2) if len(value_counts) > 0 else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }
            
            # Cardinality assessment
            if profile['unique_count'] <= 5:
                profile['cardinality'] = 'Very Low'
            elif profile['unique_count'] <= 20:
                profile['cardinality'] = 'Low'
            elif profile['unique_percentage'] < 50:
                profile['cardinality'] = 'Medium'
            else:
                profile['cardinality'] = 'High'
            
            categorical_profiles.append(profile)
        
        return {
            'numeric_features': numeric_profiles,
            'categorical_features': categorical_profiles
        }
    
    def detect_multicollinearity(self, df: pd.DataFrame, threshold: float = 0.85) -> Dict[str, Any]:
        """
        Detect multicollinearity using correlation analysis.
        
        Args:
            df: DataFrame with numeric columns
            threshold: Correlation threshold to flag (default 0.85)
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns'}
        
        corr_matrix = numeric_df.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= threshold:
                    high_correlations.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': round(corr_matrix.iloc[i, j], 4),
                        'abs_correlation': round(corr_val, 4)
                    })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Identify potentially redundant features
        redundant_features = set()
        for pair in high_correlations:
            redundant_features.add(pair['feature_2'])  # Keep first, mark second as redundant
        
        return {
            'threshold': threshold,
            'n_features_analyzed': len(corr_matrix.columns),
            'n_high_correlations': len(high_correlations),
            'high_correlation_pairs': high_correlations,
            'potentially_redundant_features': list(redundant_features),
            'recommendation': f"Consider removing or combining {len(redundant_features)} features with high correlation." 
                            if redundant_features else "No severe multicollinearity detected."
        }
    
    def analyze_variance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze variance across numeric features.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns found'}
        
        variance_analysis = []
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            if len(data) == 0:
                continue
            
            variance = data.var()
            std = data.std()
            cv = (std / abs(data.mean()) * 100) if data.mean() != 0 else None
            
            variance_analysis.append({
                'column': col,
                'variance': round(variance, 4),
                'std_dev': round(std, 4),
                'coefficient_of_variation': round(cv, 2) if cv else None,
                'is_low_variance': variance < 0.01,
                'is_constant': variance == 0
            })
        
        # Sort by variance
        variance_analysis.sort(key=lambda x: x['variance'])
        
        low_variance_count = sum(1 for v in variance_analysis if v['is_low_variance'])
        constant_count = sum(1 for v in variance_analysis if v['is_constant'])
        
        return {
            'n_features': len(variance_analysis),
            'variance_by_feature': variance_analysis,
            'low_variance_features': [v['column'] for v in variance_analysis if v['is_low_variance']],
            'constant_features': [v['column'] for v in variance_analysis if v['is_constant']],
            'summary': {
                'low_variance_count': low_variance_count,
                'constant_count': constant_count
            }
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_summary_statistics_extended(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get extended summary statistics beyond basic describe().
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns found'}
        
        # Basic stats
        basic_stats = numeric_df.describe()
        
        # Additional stats
        additional_stats = pd.DataFrame({
            'median': numeric_df.median(),
            'mode': numeric_df.mode().iloc[0] if not numeric_df.mode().empty else None,
            'variance': numeric_df.var(),
            'skewness': numeric_df.skew(),
            'kurtosis': numeric_df.kurtosis(),
            'range': numeric_df.max() - numeric_df.min(),
            'iqr': numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
            'null_count': numeric_df.isnull().sum(),
            'zero_count': (numeric_df == 0).sum()
        }).T
        
        # Combine
        extended_stats = pd.concat([basic_stats, additional_stats])
        
        return {
            'extended_statistics': extended_stats,
            'basic_statistics': basic_stats
        }
