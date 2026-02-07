"""
Plateau-First Strategy

Implements peak detection by first identifying plateaus using morphological operations,
then calculating prominence for validated plateaus.
"""

from typing import Any
import numpy as np
from scipy.ndimage import maximum_filter, binary_dilation, label

from .base_strategy import BaseStrategy, StrategyConfig
from ..api.result_dataframe import Peak
from ..core.plateau_detector import PlateauDetector, PlateauRegion
from ..core.prominence_calculator import ProminenceCalculator
from ..connectivity.connectivity_types import get_k_connectivity
from ..core.strategy_manager import PerformanceMetrics


class PlateauFirstStrategy(BaseStrategy):
    """
    Peak detection strategy that identifies plateaus first, then validates them.
    
    This strategy uses morphological operations to detect potential plateau regions,
    validates them using dilation tests, and then calculates prominence for valid plateaus.
    """
    
    def __init__(self, config: StrategyConfig | None = None, **kwargs):
        """
        Initialize Plateau-First strategy.
        
        Parameters:
        -----------
        config : StrategyConfig, optional
            Strategy configuration
        **kwargs
            Additional parameters including:
            - dilation_structure: Custom structuring element for dilation
            - validation_strict: Whether to apply strict plateau validation
        """
        super().__init__(config, **kwargs)
        self.plateau_detector = PlateauDetector(self.config.connectivity)
        self.prominence_calculator = ProminenceCalculator(self.config.connectivity)
        
        # Strategy-specific parameters
        self.dilation_structure = kwargs.get('dilation_structure', None)
        self.validation_strict = kwargs.get('validation_strict', True)
        
    def detect_peaks(self, data: np.ndarray, **params) -> list[Peak]:
        """
        Detect peaks using plateau-first approach.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        **params
            Additional parameters
            
        Returns:
        --------
        list[Peak]
            List of detected peaks
        """
        # Validate input
        self.validate_input(data)
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Detect and validate plateaus
        plateau_results = self.detect_plateaus_then_prominence(processed_data)
        
        # Convert plateaus to Peak objects
        peaks = []
        for plateau, prominence in plateau_results.items():
            if plateau.is_valid:
                peak = self._create_peak_from_plateau(plateau, processed_data)
                peaks.append(peak)
        
        # Postprocess peaks
        filtered_peaks = self.postprocess_peaks(peaks, data)
        
        return filtered_peaks
    
    def detect_plateaus_then_prominence(self, data: np.ndarray) -> dict[PlateauRegion, float]:
        """
        Detect plateaus and calculate prominence in batch.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        dict[PlateauRegion, float]
            Dictionary mapping plateaus to their prominence values
        """
        # Step 1: Detect potential plateaus
        plateau_candidates = self._detect_plateaus(data)
        
        # Step 2: Validate plateaus
        validated_plateaus = []
        for plateau in plateau_candidates:
            if self.plateau_detector.validate_plateau(plateau, data, self.config.connectivity):
                validated_plateaus.append(plateau)
        
        # Step 3: Calculate prominence for validated plateaus
        plateau_prominence = self._calculate_prominence_batch(validated_plateaus, data)
        
        return plateau_prominence
    
    def _detect_plateaus(self, data: np.ndarray) -> list[PlateauRegion]:
        """
        Detect potential plateau regions using morphological operations.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        list[PlateauRegion]
            List of potential plateau regions
        """
        # Step 1: Apply local maximum filter
        local_maxima_mask = self._apply_local_maximum_filter(data)
        
        # Step 2: Find connected components of same height
        plateau_candidates = self._find_connected_components(data, local_maxima_mask)
        
        # Step 3: Filter by minimum size if configured
        if self.config.min_plateau_size > 1:
            plateau_candidates = self.plateau_detector.filter_noise_plateaus(
                plateau_candidates, self.config.min_plateau_size
            )
        
        return plateau_candidates
    
    def _apply_local_maximum_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply local maximum filter to identify potential peak cells.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray
            Boolean mask of potential peak cells
        """
        # Get connectivity structure for filter
        connectivity_structure = get_k_connectivity(data.ndim, self.config.connectivity)
        
        # Apply maximum filter
        max_filtered = maximum_filter(data, footprint=connectivity_structure)
        
        # Local maxima are cells equal to their filtered value
        local_maxima = (data == max_filtered)
        
        return local_maxima
    
    def _find_connected_components(self, data: np.ndarray, mask: np.ndarray) -> list[PlateauRegion]:
        """
        Find connected components of same height within mask.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        mask : np.ndarray
            Boolean mask of candidate cells
            
        Returns:
        --------
        list[PlateauRegion]
            List of plateau regions
        """
        plateaus = []
        
        # Get connectivity structure
        connectivity_structure = get_k_connectivity(data.ndim, self.config.connectivity)
        
        # Get unique heights in masked region
        masked_data = data * mask
        unique_heights = np.unique(masked_data[mask])
        
        for height in unique_heights:
            if height == 0 and not mask.all():
                continue  # Skip background
                
            # Create mask for current height
            height_mask = (data == height) & mask
            
            if not height_mask.any():
                continue
                
            # Find connected components at this height
            labeled_array, num_features = label(height_mask, structure=connectivity_structure)
            
            for component_id in range(1, num_features + 1):
                component_mask = (labeled_array == component_id)
                indices = list(zip(*np.where(component_mask)))
                
                # Calculate centroid
                if indices:
                    centroid = tuple(np.mean([list(idx) for idx in indices], axis=0))
                    
                    plateau = PlateauRegion(
                        indices=indices,
                        height=float(height),
                        centroid=centroid,
                        is_valid=False  # Will be determined during validation
                    )
                    
                    plateaus.append(plateau)
        
        return plateaus
    
    def _validate_plateaus_by_dilation(self, candidates: list[PlateauRegion], data: np.ndarray) -> list[PlateauRegion]:
        """
        Validate plateau candidates using dilation test.
        
        Parameters:
        -----------
        candidates : list[PlateauRegion]
            Plateau candidates to validate
        data : np.ndarray
            Input data
            
        Returns:
        --------
        list[PlateauRegion]
            Validated plateau regions
        """
        validated_plateaus = []
        
        # Get dilation structure
        if self.dilation_structure is not None:
            structure = self.dilation_structure
        else:
            structure = get_k_connectivity(data.ndim, self.config.connectivity)
        
        for plateau in candidates:
            if self._validate_single_plateau(plateau, data, structure):
                plateau.is_valid = True
                validated_plateaus.append(plateau)
        
        return validated_plateaus
    
    def _validate_single_plateau(self, plateau: PlateauRegion, data: np.ndarray, structure: np.ndarray) -> bool:
        """
        Validate a single plateau using dilation test.
        
        A true peak plateau should have all boundary cells strictly lower
        than the plateau height after dilation.
        
        Parameters:
        -----------
        plateau : PlateauRegion
            Plateau to validate
        data : np.ndarray
            Input data
        structure : np.ndarray
            Dilation structuring element
            
        Returns:
        --------
        bool
            True if plateau is a valid peak
        """
        # Create binary mask for plateau region
        plateau_mask = np.zeros(data.shape, dtype=bool)
        for idx in plateau.indices:
            plateau_mask[idx] = True
            
        # Apply morphological dilation
        dilated_mask = binary_dilation(plateau_mask, structure=structure)
        
        # Find boundary (dilated - original)
        boundary_mask = dilated_mask & (~plateau_mask)
        
        # Get boundary indices
        boundary_indices = list(zip(*np.where(boundary_mask)))
        plateau.boundary_indices = boundary_indices
        
        # Check if all boundary cells are strictly lower
        plateau_height = plateau.height
        
        if self.validation_strict:
            # Strict validation: ALL boundary cells must be lower
            for boundary_idx in boundary_indices:
                if data[boundary_idx] >= plateau_height:
                    return False
        else:
            # Relaxed validation: MAJORITY of boundary cells must be lower
            lower_count = sum(1 for idx in boundary_indices if data[idx] < plateau_height)
            return lower_count > len(boundary_indices) * 0.5
                
        return True
    
    def _calculate_prominence_batch(self, plateaus: list[PlateauRegion], data: np.ndarray) -> dict[PlateauRegion, float]:
        """
        Calculate prominence for multiple plateaus efficiently.
        
        Parameters:
        -----------
        plateaus : list[PlateauRegion]
            Validated plateau regions
        data : np.ndarray
            Input data
            
        Returns:
        --------
        dict[PlateauRegion, float]
            Mapping of plateaus to prominence values
        """
        prominence_results = {}
        
        for plateau in plateaus:
            # Convert plateau to Peak for prominence calculation
            peak = self._create_peak_from_plateau(plateau, data)
            
            # Calculate prominence
            prominence = self.prominence_calculator.calculate_prominence(peak, data)
            
            prominence_results[plateau] = prominence
        
        return prominence_results
    
    def _create_peak_from_plateau(self, plateau: PlateauRegion, data: np.ndarray) -> Peak:
        """
        Convert PlateauRegion to Peak object.
        
        Parameters:
        -----------
        plateau : PlateauRegion
            Plateau region
        data : np.ndarray
            Input data
            
        Returns:
        --------
        Peak
            Peak object
        """
        # Use plateau centroid as peak center
        center_indices = plateau.centroid
        center_coordinates = center_indices  # Default: use indices as coordinates
        
        return Peak(
            center_indices=center_indices,
            center_coordinates=center_coordinates,
            plateau_indices=plateau.indices,
            height=plateau.height
        )
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray) -> dict[Peak, dict[str, Any]]:
        """
        Calculate topographic features for peaks.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to calculate features for
        data : np.ndarray
            Original data
            
        Returns:
        --------
        dict[Peak, dict[str, Any]]
            Feature dictionary for each peak
        """
        features = {}
        
        for peak in peaks:
            # Basic geometric features
            peak_features = {
                'height': peak.height,
                'area': len(peak.plateau_indices),
                'coordinates': peak.center_coordinates,
                'centroid': peak.center_indices
            }
            
            # Calculate additional features if requested
            if 'prominence' in self.kwargs.get('features', []):
                prominence = self.prominence_calculator.calculate_prominence(peak, data)
                peak_features['prominence'] = prominence
            
            features[peak] = peak_features
        
        return features
    
    @classmethod
    def estimate_performance(cls, data_shape: tuple[int, ...]) -> PerformanceMetrics:
        """
        Estimate performance for Plateau-First strategy.
        
        Parameters:
        -----------
        data_shape : tuple of int
            Shape of input data
            
        Returns:
        --------
        PerformanceMetrics
            Performance metrics
        """
        data_size = np.prod(data_shape)
        
        # Morphological operations are generally O(n)
        estimated_time = data_size * 2e-6  # seconds, accounting for multiple passes
        
        # Memory for temporary masks and labeled arrays
        estimated_memory = data_size * 12 / (1024 * 1024)  # MB
        
        # Good accuracy for well-separated peaks
        accuracy_score = 0.85
        
        # Good scalability
        scalability_factor = 0.9
        
        return PerformanceMetrics(
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            accuracy_score=accuracy_score,
            scalability_factor=scalability_factor
        )
    
    def _validate_strategy_specific(self, data: np.ndarray) -> bool:
        """
        Perform Plateau-First specific validation.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        bool
            True if data is suitable for this strategy
        """
        # Check if data has sufficient dynamic range
        data_range = np.ptp(data)  # peak-to-peak range
        if data_range < 1e-10:
            raise ValueError("Data has insufficient dynamic range for plateau detection")
        
        # Check for reasonable data size
        if data.size > 1e8:  # 100M elements
            import warnings
            warnings.warn("Large data size may cause memory issues with morphological operations")
        
        return True