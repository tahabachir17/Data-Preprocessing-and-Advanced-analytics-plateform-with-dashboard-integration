"""
Analytics Module
================
Provides statistical analysis, hypothesis testing, and advanced analytics capabilities.
"""

__all__ = [
    'StatisticalAnalyzer',
    'AdvancedAnalytics',
    'MLModels'
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at startup."""
    if name == 'StatisticalAnalyzer':
        from .statistical import StatisticalAnalyzer
        return StatisticalAnalyzer
    elif name == 'AdvancedAnalytics':
        from .advanced_analytics import AdvancedAnalytics
        return AdvancedAnalytics
    elif name == 'MLModels':
        from .ml_models import MLModels
        return MLModels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
