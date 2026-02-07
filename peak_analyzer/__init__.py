"""
peak-analyzer: Topography-Aware Multidimensional Peak Detection

A sophisticated library for detecting peaks in N-dimensional data with
proper plateau handling and topographic feature extraction.

This library provides a 7-layer modular architecture for comprehensive
peak analysis including coordinate systems, spatial connectivity,
boundary handling, and advanced feature extraction.
"""

# Core components
from .core.peak_detector import PeakAnalyzer
from .core.peak_region import PeakRegion
from .core.lazy_dataframe import LazyDataFrame

# Layer imports with submodule access
from . import features
from . import connectivity
from . import coordinate_system
from . import boundary
from . import data
from . import utils

# Main feature extractors
from .features import (
    GeometricCalculator,
    TopographicCalculator,
    MorphologicalCalculator,
    DistanceCalculator
)

# Connectivity components
from .connectivity import (
    ConnectivityPattern,
    NeighborGenerator,
    PathFinder,
    DistanceMetric
)

# Coordinate system components
from .coordinate_system import (
    CoordinateMapping,
    GridManager,
    SpatialIndexFactory
)

# Boundary handling
from .boundary import (
    BoundaryManager,
    EdgeDetector,
    create_boundary_condition
)

# Data management
from .data import (
    validate_peak_data,
    load_data,
    save_data,
    DataFormat,
    ValidationLevel
)

# Utilities
from .utils import (
    normalize_array,
    find_local_maxima,
    create_test_data,
    calculate_statistics
)

__version__ = "0.1.0"
__author__ = "peak-analyzer team"

__all__ = [
    # Core components
    "PeakAnalyzer",
    "PeakRegion",
    "LazyDataFrame",
    
    # Layer modules
    "features",
    "connectivity", 
    "coordinate_system",
    "boundary",
    "data",
    "utils",
    
    # Feature extractors
    "GeometricCalculator",
    "TopographicCalculator",
    "MorphologicalCalculator",
    "DistanceCalculator",
    
    # Connectivity
    "ConnectivityPattern",
    "NeighborGenerator", 
    "PathFinder",
    "DistanceMetric",
    
    # Coordinate system
    "CoordinateMapping",
    "GridManager",
    "SpatialIndexFactory",
    
    # Boundary handling
    "BoundaryManager",
    "EdgeDetector",
    "create_boundary_condition",
    
    # Data management
    "validate_peak_data",
    "load_data",
    "save_data",
    "DataFormat",
    "ValidationLevel",
    
    # Utilities
    "normalize_array",
    "find_local_maxima",
    "create_test_data",
    "calculate_statistics",
]

# Convenience functions for common workflows
def create_peak_analyzer(data, coordinate_mapping=None, **kwargs):
    """
    Create a PeakAnalyzer with optional coordinate mapping.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data for peak analysis
    coordinate_mapping : CoordinateMapping, optional
        Coordinate mapping for the data
    **kwargs
        Additional arguments for PeakAnalyzer
        
    Returns:
    --------
    PeakAnalyzer
        Configured peak analyzer instance
    """
    # Validate data first
    validation_result = validate_peak_data(data)
    if not validation_result.is_valid:
        import warnings
        warnings.warn(f"Data validation issues: {validation_result.warnings}")
    
    return PeakAnalyzer(data, **kwargs)


def analyze_peaks_comprehensive(data, coordinate_mapping=None, 
                              include_features=True, include_connectivity=True,
                              validation_level="standard", **kwargs):
    """
    Perform comprehensive peak analysis with all available features.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data for peak analysis
    coordinate_mapping : CoordinateMapping, optional
        Coordinate mapping for the data
    include_features : bool
        Whether to extract geometric and topographic features
    include_connectivity : bool
        Whether to analyze spatial connectivity
    validation_level : str
        Level of data validation to perform
    **kwargs
        Additional analysis parameters
        
    Returns:
    --------
    dict
        Comprehensive analysis results
    """
    # Validate data
    validation_result = validate_peak_data(data, validation_level=validation_level)
    
    # Create analyzer
    analyzer = create_peak_analyzer(data, coordinate_mapping, **kwargs)
    
    # Perform basic peak detection
    peaks = analyzer.find_peaks()
    
    results = {
        'peaks': peaks,
        'validation': validation_result,
        'data_stats': calculate_statistics(data)
    }
    
    # Add feature analysis if requested
    if include_features and peaks:
        try:
            # Extract geometric features
            geom_calc = GeometricCalculator()
            results['geometric_features'] = {}
            for i, peak in enumerate(peaks):
                results['geometric_features'][i] = geom_calc.calculate_all_features(
                    data, peak.indices, coordinate_mapping
                )
            
            # Extract topographic features
            topo_calc = TopographicCalculator()
            results['topographic_features'] = {}
            for i, peak in enumerate(peaks):
                results['topographic_features'][i] = topo_calc.calculate_all_features(
                    data, peak.indices, coordinate_mapping
                )
                
        except Exception as e:
            import warnings
            warnings.warn(f"Feature extraction failed: {e}")
    
    # Add connectivity analysis if requested
    if include_connectivity and peaks:
        try:
            # Analyze peak connectivity
            pattern = ConnectivityPattern.from_data(data.shape)
            neighbor_gen = NeighborGenerator(pattern)
            
            results['connectivity'] = {}
            for i, peak in enumerate(peaks):
                neighbors = neighbor_gen.get_neighbors(peak.indices)
                results['connectivity'][i] = {
                    'neighbors': neighbors,
                    'neighbor_count': len(neighbors)
                }
                
        except Exception as e:
            import warnings
            warnings.warn(f"Connectivity analysis failed: {e}")
    
    return results