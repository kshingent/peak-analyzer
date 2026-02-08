"""
Main Peak Detection Analysis Engine

Provides the primary interface for topographic peak detection and analysis.
"""

import numpy as np

from peak_analyzer.core.strategy_manager import StrategyManager
from peak_analyzer.coordinate_system.grid_manager import GridManager
from peak_analyzer.coordinate_system.coordinate_mapping import CoordinateMapping
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
            Boundary condition ('infinite_height', 'infinite_depth')
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
        
        # Simple boundary padding
        if self.boundary_type == 'infinite_height':
            pad_value = float('inf')
        elif self.boundary_type == 'infinite_depth':
            pad_value = float('-inf')
        else:
            raise ValueError(f"Invalid boundary type: {self.boundary_type}. "
                           "Only 'infinite_height' and 'infinite_depth' are supported.")
        
        self.padded_data = np.pad(data, 1, mode='constant', constant_values=pad_value)
        self.original_slice = tuple(slice(1, 1 + s) for s in data.shape)
        
        # Apply boundary extension for peak detection
        padding = self.kwargs.get('boundary_padding', 1)  # Default padding of 1
        padded_data, original_slice = self.boundary_handler.extend_data(data, padding)
        
        # Select optimal strategy
        strategy = self.strategy_manager.select_optimal_strategy(data)
        
        # Execute peak detection
        peaks = strategy.detect_peaks(padded_data, **self.kwargs)
        
        # Remove boundary artifacts - filter peaks too close to original data boundaries
        min_distance = self.kwargs.get('min_boundary_distance', padding)
        filtered_peaks = self._remove_boundary_artifacts(peaks, data.shape, padding, min_distance)
        
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

    def _remove_boundary_artifacts(self, peaks, original_shape: tuple[int, ...], 
                                 padding: int, min_distance: int) -> list:
        """
        Remove peaks that are too close to data boundaries (likely artifacts).
        
        Parameters:
        -----------
        peaks : list
            List of detected peaks
        original_shape : tuple of int
            Shape of original data (without padding)
        padding : int
            Padding applied to data
        min_distance : int
            Minimum distance from boundary for valid peaks
            
        Returns:
        --------
        list
            Filtered peaks with boundary artifacts removed
        """
        if not peaks:
            return peaks
            
        filtered_peaks = []
        
        for peak in peaks:
            # Adjust peak coordinates to original data space
            original_coords = tuple(coord - padding for coord in peak.coordinates)
            
            # Check if peak is too close to any boundary
            is_valid = True
            for dim, (coord, size) in enumerate(zip(original_coords, original_shape)):
                if coord < min_distance or coord >= size - min_distance:
                    is_valid = False
                    break
            
            if is_valid:
                # Create new peak with adjusted coordinates
                adjusted_peak = peak.__class__(
                    coordinates=original_coords,
                    height=peak.height,
                    # Copy other attributes if they exist
                    **{attr: getattr(peak, attr) for attr in ['prominence', 'area', 'isolation'] 
                       if hasattr(peak, attr)}
                )
                filtered_peaks.append(adjusted_peak)
                
        return filtered_peaks