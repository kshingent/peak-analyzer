"""
Main Peak Detection Analysis Engine

Provides the primary interface for topographic peak detection and analysis.
"""

import numpy as np

from peak_analyzer.core.strategy_manager import StrategyManager
from peak_analyzer.coordinate_system.grid_manager import GridManager
from peak_analyzer.coordinate_system.coordinate_mapping import CoordinateMapping
from peak_analyzer.boundary.boundary_conditions import BoundaryManager
from peak_analyzer.data.validation import validate_peak_data
from .result_dataframe import PeakCollection
from .parameter_validation import ParameterValidator


class PeakAnalyzer:
    """
    Main analysis engine for topographic peak detection.
    
    This class provides a unified interface for detecting peaks in multidimensional
    data using various strategies and calculating topographic features.
    """
    
    def __init__(
        self, 
        strategy: str = 'auto',
        connectivity: int = 1, 
        boundary: str = 'infinite_height',
        scale: float | list[float | None] = None,
        minkowski_p: float = 2.0,
        **kwargs
    ):
        """
        Initialize PeakAnalyzer with analysis parameters.
        
        Parameters:
        -----------
        strategy : str
            Detection strategy ('auto', 'union_find', 'plateau_first')
        connectivity : str or int
            Integer k-connectivity (1 ≤ k ≤ ndim)
        boundary : str
            Boundary condition ('infinite_height', 'infinite_depth', 'periodic', 'custom')
        scale : float or list of float
            Physical scale for each dimension (real-world units per pixel)
        minkowski_p : float
            Minkowski distance parameter (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)
        **kwargs
            Additional strategy-specific parameters
        """
        self.strategy_name = strategy
        self.connectivity = connectivity
        self.boundary_type = boundary
        self.scale = scale
        self.minkowski_p = minkowski_p
        self.kwargs = kwargs
        
        # Initialize components
        self.validator = ParameterValidator()
        self.strategy_manager = StrategyManager()
        self.boundary_handler = None
        self.grid_manager = None
        
    def find_peaks(self, data: np.ndarray, **filters) -> PeakCollection:
        """
        Detect peaks in multidimensional data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input multidimensional data for peak detection
        **filters
            Filter criteria for peak selection (e.g., prominence > 0.5, area > 10)
            
        Returns:
        --------
        PeakCollection
            Collection of detected peaks with lazy feature calculation
        """
        # Validate input data
        validation_result = validate_peak_data(data)
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            raise ValueError(f"Input data validation failed: {error_msg}")
        
        # Show validation warnings if any
        if validation_result.warnings:
            import warnings
            warning_msg = "; ".join(validation_result.warnings)
            warnings.warn(f"Data validation warnings: {warning_msg}")
        
        # Set up coordinate mapping and grid manager
        if self.scale is None:
            self.scale = [1.0] * data.ndim
        elif isinstance(self.scale, (int, float)):
            self.scale = [float(self.scale)] * data.ndim
            
        coordinate_mapping = CoordinateMapping(
            indices_shape=data.shape,
            coordinate_origin=tuple([0.0] * data.ndim),
            coordinate_spacing=tuple(self.scale),
            axis_names=tuple([f"axis_{i}" for i in range(data.ndim)])
        )
        
        self.grid_manager = GridManager(coordinate_mapping, self.connectivity)
        
        # Set up boundary handling - needs data shape, so initialize here
        from peak_analyzer.boundary.boundary_conditions import NoBoundary, ConstantBoundary
        if self.boundary_type == 'infinite_height':
            boundary_condition = ConstantBoundary(value=float('inf'))
        elif self.boundary_type == 'infinite_depth':  
            boundary_condition = ConstantBoundary(value=float('-inf'))
        else:
            boundary_condition = NoBoundary()  # Default fallback
            
        self.boundary_handler = BoundaryManager(
            shape=data.shape, 
            boundary_conditions=boundary_condition
        )
        
        # Note: extend_data_with_boundary method doesn't exist in BoundaryManager
        # Using original data for now - boundary handling needs to be implemented
        padded_data = data  # TODO: Implement proper boundary extension
        
        # Select optimal strategy
        strategy = self.strategy_manager.select_optimal_strategy(data)
        
        # Execute peak detection
        peaks = strategy.detect_peaks(padded_data, **self.kwargs)
        
        # Note: remove_boundary_artifacts method doesn't exist in BoundaryManager  
        # Using original peaks for now - artifact removal needs to be implemented
        filtered_peaks = peaks  # TODO: Implement proper boundary artifact removal
        
        # Create peak collection
        peak_collection = PeakCollection(
            peaks=filtered_peaks,
            data=data,
            grid_manager=self.grid_manager,
            **filters
        )
        
        return peak_collection
        
    def analyze_prominence(self, peaks: PeakCollection, wlen: float | None = None):
        """
        Calculate prominence for detected peaks.
        
        Parameters:
        -----------
        peaks : PeakCollection
            Collection of peaks to analyze
        wlen : float, optional
            Window length for prominence calculation constraint
            
        Returns:
        --------
        ProminenceResults
            Results containing prominence values and metadata
        """
        # Implementation will be added
        pass
        
    def calculate_features(self, peaks: PeakCollection, features: str = 'all'):
        """
        Calculate topographic features for peaks.
        
        Parameters:
        -----------
        peaks : PeakCollection
            Collection of peaks to analyze
        features : str or list
            Features to calculate ('all', 'geometric', 'topographic', or specific list)
            
        Returns:
        --------
        FeatureDataFrame
            DataFrame with calculated features
        """
        # Implementation will be added
        pass
        
    def filter_peaks(self, peaks: PeakCollection, **criteria) -> PeakCollection:
        """
        Filter peaks based on specified criteria.
        
        Parameters:
        -----------
        peaks : PeakCollection
            Collection of peaks to filter
        **criteria
            Filtering criteria (e.g., prominence > 0.5, area > 10)
            
        Returns:
        --------
        PeakCollection
            Filtered peak collection
        """
        return peaks.filter(**criteria)
        
    def get_virtual_peaks(self, peaks: PeakCollection):
        """
        Create virtual peaks for same-height connected peaks.
        
        Parameters:
        -----------
        peaks : PeakCollection
            Collection of peaks to analyze for virtual peak creation
            
        Returns:
        --------
        VirtualPeakCollection
            Collection of virtual peaks representing connected same-height peaks
        """
        # Implementation will be added
        pass