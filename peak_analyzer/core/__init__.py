"""
Core Algorithm Components Package

Core algorithms and utilities for peak detection.
"""

from .plateau_detector import PlateauDetector
from .prominence_calculator import ProminenceCalculator
from .strategy_manager import StrategyManager
from .union_find import GridUnionFind
from .virtual_peak_handler import VirtualPeakHandler

__all__ = [
    'PlateauDetector',
    'ProminenceCalculator',
    'StrategyManager',
    'GridUnionFind',
    'VirtualPeakHandler',
]