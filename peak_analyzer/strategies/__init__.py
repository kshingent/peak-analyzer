"""
Peak Detection Strategies Package

Different algorithmic strategies for peak detection.
"""

from .base_strategy import BaseStrategy
from .union_find_strategy import UnionFindStrategy
from .plateau_first_strategy import PlateauFirstStrategy
from .strategy_factory import StrategyFactory

__all__ = [
    'BaseStrategy',
    'UnionFindStrategy',
    'PlateauFirstStrategy', 
    'StrategyFactory',
]