"""
Peak Analyzer Models Package

Centralized data structures for the peak analyzer:
- peaks.py: Detection results (Peak, VirtualPeak, SaddlePoint)
- data_analysis.py: Analysis metadata (DataCharacteristics, BenchmarkResults)
"""

# Peak detection results
from .peaks import (
    Peak,
    VirtualPeak, 
    SaddlePoint,
    IndexTuple,
    CoordTuple
)

# Data analysis and metadata
from .data_analysis import (
    DataCharacteristics,
    BenchmarkResults
)

__all__ = [
    # Peak structures
    'Peak',
    'VirtualPeak',
    'SaddlePoint',
    'IndexTuple',
    'CoordTuple',
    
    # Analysis structures
    'DataCharacteristics',
    'BenchmarkResults',
]