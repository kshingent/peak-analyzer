"""
Peak Analyzer API Package

Public API for peak detection and analysis.
"""

from .peak_detector import PeakAnalyzer
from .result_dataframe import PeakCollection
from .parameter_validation import ParameterValidator

__all__ = [
    'PeakAnalyzer',
    'PeakCollection', 
    'ParameterValidator',
]