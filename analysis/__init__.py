"""
Analysis module for SAP incident reporting system.
Provides comprehensive analysis capabilities following clean code principles.
"""

# Import only production components (no test imports)
from .quality_analyzer import IncidentQualityAnalyzer, QualityAnalyzer
from .trend_analyzer import IncidentTrendAnalyzer, TrendAnalyzer
from .metrics_calculator import IncidentMetricsCalculator, MetricsCalculator
from .keyword_manager import IncidentKeywordManager, KeywordManager

# Package metadata
__version__ = "1.0.0"
__author__ = "Incident Reporting System"

# Define what gets imported with "from analysis import *"
__all__ = [
    'IncidentQualityAnalyzer', 'QualityAnalyzer',
    'IncidentTrendAnalyzer', 'TrendAnalyzer', 
    'IncidentMetricsCalculator', 'MetricsCalculator',
    'IncidentKeywordManager', 'KeywordManager'
]

# Package-level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Convenience functions for quick access
def get_quality_analyzer():
    """Get a configured quality analyzer instance."""
    return IncidentQualityAnalyzer()

def get_trend_analyzer():
    """Get a configured trend analyzer instance."""
    return IncidentTrendAnalyzer()

def get_metrics_calculator():
    """Get a configured metrics calculator instance."""
    return IncidentMetricsCalculator()

def get_keyword_manager():
    """Get a configured keyword manager instance."""
    return IncidentKeywordManager()