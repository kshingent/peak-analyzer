"""
peak-analyzer: Topography-Aware Multidimensional Peak Detection

A sophisticated library for detecting peaks in N-dimensional data with
proper plateau handling and topographic feature extraction.
"""

from .core.peak_detector import PeakAnalyzer
from .core.peak_region import PeakRegion
from .core.lazy_dataframe import LazyDataFrame

__version__ = "0.1.0"
__author__ = "peak-analyzer team"

__all__ = [
    "PeakAnalyzer",
    "PeakRegion", 
    "LazyDataFrame",
]