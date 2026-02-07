"""
Peak Analyzer Features Layer

Provides feature calculation framework for computing geometric, topographic,
morphological, and distance-based properties of peaks.

Classes:
--------
BaseCalculator : Abstract base class for feature calculators
GeometricCalculator : Geometric properties (area, volume, centroid, etc.)
TopographicCalculator : Topographic properties (prominence, isolation, gradients, etc.)
MorphologicalCalculator : Morphological properties (shape analysis, connectivity, etc.)
DistanceCalculator : Distance-based properties (clustering, spatial distribution, etc.)
LazyFeatureManager : Lazy evaluation and caching for efficient feature computation

Usage:
------
    from peak_analyzer.features import GeometricCalculator, TopographicCalculator
    
    # Initialize calculators
    geom_calc = GeometricCalculator(scale=[1.0, 1.0])
    topo_calc = TopographicCalculator(scale=[1.0, 1.0])
    
    # Calculate features
    geometric_features = geom_calc.calculate_features(peaks, data)
    topographic_features = topo_calc.calculate_features(peaks, data)
    
    # Use lazy feature manager for large datasets
    from peak_analyzer.features import LazyFeatureManager
    
    with LazyFeatureManager(peaks, data, scale=[1.0, 1.0]) as manager:
        # Features computed on demand and cached
        features = manager.get_features(peak, ['geometric', 'topographic'])
"""

from .base_calculator import BaseCalculator
from .geometric_calculator import GeometricCalculator
from .topographic_calculator import TopographicCalculator
from .morphological_calculator import MorphologicalCalculator
from .distance_calculator import DistanceCalculator
from .lazy_feature_manager import LazyFeatureManager, FeatureRequest, AsyncFeatureComputer

__all__ = [
    # Base classes
    'BaseCalculator',
    
    # Feature calculators
    'GeometricCalculator',
    'TopographicCalculator', 
    'MorphologicalCalculator',
    'DistanceCalculator',
    
    # Lazy evaluation and management
    'LazyFeatureManager',
    'FeatureRequest',
    'AsyncFeatureComputer'
]

# Version info
__version__ = '1.0.0'