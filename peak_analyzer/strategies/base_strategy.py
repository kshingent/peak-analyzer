"""
Base Strategy Class

Defines the common interface for all peak detection strategies.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from dataclasses import dataclass

from ..models import Peak


@dataclass
class StrategyConfig:
    """Configuration parameters for strategies."""
    connectivity: int = 1
    boundary_type: str = 'infinite_height'
    min_plateau_size: int = 1
    noise_threshold: float = 0.0
    parallel_processing: bool = False
    chunk_size: int | None = None


class BaseStrategy(ABC):
    """
    Abstract base class for peak detection strategies.
    
    All concrete strategies must implement the required abstract methods
    and can optionally override the preprocessing and postprocessing methods.
    """
    
    def __init__(self, config: StrategyConfig | None = None, **kwargs):
        """
        Initialize strategy with configuration.
        
        Parameters:
        -----------
        config : StrategyConfig, optional
            Strategy configuration parameters
        **kwargs
            Additional strategy-specific parameters
        """
        self.config = config or StrategyConfig()
        self.kwargs = kwargs
        self._performance_cache = {}
        
    @abstractmethod
    def detect_peaks(self, data: np.ndarray, **params) -> list[Peak]:
        """
        Detect peaks in the input data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input multidimensional data
        **params
            Strategy-specific parameters
            
        Returns:
        --------
        list[Peak]
            List of detected peaks
        """
        pass
    
    @abstractmethod
    def calculate_features(self, peaks: list[Peak], data: np.ndarray) -> dict[Peak, dict[str, Any]]:
        """
        Calculate topographic features for detected peaks.
        
        Parameters:
        -----------
        peaks : list[Peak]
            List of peaks to calculate features for
        data : np.ndarray
            Original input data
            
        Returns:
        --------
        dict[Peak, dict[str, Any]]
            Dictionary mapping peaks to their feature dictionaries
        """
        pass
    
    @classmethod
    @abstractmethod
    def estimate_performance(cls, data_shape: tuple[int, ...]) -> dict[str, float]:
        """
        Estimate performance metrics for given data shape.
        
        Parameters:
        -----------
        data_shape : tuple of int
            Shape of input data
            
        Returns:
        --------
        dict[str, float]
            Dictionary with performance metrics containing keys:
            'estimated_time', 'estimated_memory', 'accuracy_score', 'scalability_factor'
        """
        pass
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data before peak detection.
        
        Default implementation applies noise filtering if configured.
        
        Parameters:
        -----------
        data : np.ndarray
            Raw input data
            
        Returns:
        --------
        np.ndarray
            Preprocessed data
        """
        processed_data = data.copy()
        
        # Apply noise filtering if threshold is set
        if self.config.noise_threshold > 0:
            processed_data = self._apply_noise_filtering(processed_data)
        
        return processed_data
    
    def postprocess_peaks(self, peaks: list[Peak], data: np.ndarray) -> list[Peak]:
        """
        Postprocess detected peaks.
        
        Default implementation filters by minimum plateau size.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Raw detected peaks
        data : np.ndarray
            Original input data
            
        Returns:
        --------
        list[Peak]
            Filtered peaks
        """
        filtered_peaks = []
        
        for peak in peaks:
            # Apply minimum plateau size filter
            if len(peak.plateau_indices) >= self.config.min_plateau_size:
                filtered_peaks.append(peak)
        
        return filtered_peaks
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for strategy compatibility.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data to validate
            
        Returns:
        --------
        bool
            True if data is valid for this strategy
            
        Raises:
        -------
        ValueError
            If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if data.size == 0:
            raise ValueError("Input data is empty")
        
        if not np.isfinite(data).all():
            raise ValueError("Input data contains non-finite values")
        
        # Strategy-specific validation can be overridden
        return self._validate_strategy_specific(data)
    
    def _validate_strategy_specific(self, data: np.ndarray) -> bool:
        """
        Strategy-specific validation.
        
        Override in subclasses for additional validation.
        """
        return True
    
    def _apply_noise_filtering(self, data: np.ndarray) -> np.ndarray:
        """Apply noise filtering to data."""
        # Simple implementation using Gaussian filter
        from scipy.ndimage import gaussian_filter
        
        # Use small sigma to preserve detail while reducing noise
        sigma = 0.5
        return gaussian_filter(data, sigma=sigma)
    
    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
        --------
        dict[str, Any]
            Strategy information dictionary
        """
        return {
            'name': self.__class__.__name__,
            'config': self.config,
            'parameters': self.kwargs,
            'description': self.__doc__.split('\n')[1].strip() if self.__doc__ else ""
        }
    
    def supports_parallel_processing(self) -> bool:
        """
        Check if strategy supports parallel processing.
        
        Returns:
        --------
        bool
            True if parallel processing is supported
        """
        return self.config.parallel_processing
    
    def get_memory_requirements(self, data_shape: tuple[int, ...]) -> dict[str, float]:
        """
        Estimate memory requirements for given data shape.
        
        Parameters:
        -----------
        data_shape : tuple of int
            Shape of input data
            
        Returns:
        --------
        dict[str, float]
            Memory requirements in MB for different components
        """
        data_size = np.prod(data_shape)
        element_size = 8  # Assuming float64
        
        base_memory = data_size * element_size / (1024 * 1024)  # MB
        
        # Strategy-specific memory estimation (override in subclasses)
        strategy_memory = self._estimate_strategy_memory(data_shape)
        
        return {
            'input_data': base_memory,
            'strategy_overhead': strategy_memory,
            'total_estimated': base_memory + strategy_memory
        }
    
    def _estimate_strategy_memory(self, data_shape: tuple[int, ...]) -> float:
        """
        Estimate strategy-specific memory overhead.
        
        Override in subclasses for more accurate estimation.
        
        Parameters:
        -----------
        data_shape : tuple of int
            Shape of input data
            
        Returns:
        --------
        float
            Estimated memory overhead in MB
        """
        # Default conservative estimate
        data_size = np.prod(data_shape)
        return data_size * 4 / (1024 * 1024)  # 4 bytes per element overhead
    
    def create_peak_from_indices(
        self, 
        indices: list[tuple[int, ...]], 
        data: np.ndarray,
        grid_manager = None
    ) -> Peak:
        """
        Create Peak object from plateau indices.
        
        Parameters:
        -----------
        indices : list[tuple of int]
            Indices belonging to the peak plateau
        data : np.ndarray
            Input data array
        grid_manager : GridManager, optional
            Grid manager for coordinate conversion
            
        Returns:
        --------
        Peak
            Peak object with calculated properties
        """
        if not indices:
            raise ValueError("Cannot create peak from empty indices")
        
        # Calculate centroid in index space
        centroid_indices = tuple(np.mean([list(idx) for idx in indices], axis=0))
        
        # Convert to coordinates if grid manager available
        if grid_manager is not None:
            centroid_coordinates = grid_manager.indices_to_coordinates(centroid_indices)
        else:
            # Default: use index values as coordinates
            centroid_coordinates = centroid_indices
        
        # Get height (should be same for all plateau indices)
        height = data[indices[0]]
        
        # Verify all indices have same height (plateau property)
        for idx in indices[1:]:
            if not np.isclose(data[idx], height):
                raise ValueError("Plateau indices do not have uniform height")
        
        return Peak(
            center_indices=centroid_indices,
            center_coordinates=centroid_coordinates,
            plateau_indices=indices,
            height=height
        )