"""
Base Calculator Class

Provides the common interface and utilities for all feature calculators.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from ..api.result_dataframe import Peak


class BaseCalculator(ABC):
    """
    Abstract base class for feature calculators.
    
    All feature calculators must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, scale: float | list[float | None] = None):
        """
        Initialize base calculator.
        
        Parameters:
        -----------
        scale : float or list of float, optional
            Physical scale for each dimension (real-world units per pixel)
        """
        self.scale = scale
        self._cache = {}
        
    @abstractmethod
    def calculate_features(self, peaks: list[Peak], data: np.ndarray, **kwargs) -> dict[Peak, dict[str, Any]]:
        """
        Calculate features for a list of peaks.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to calculate features for
        data : np.ndarray
            Original data array
        **kwargs
            Additional parameters for feature calculation
            
        Returns:
        --------
        dict[Peak, dict[str, Any]]
            Dictionary mapping each peak to its feature dictionary
        """
        pass
    
    @abstractmethod
    def get_available_features(self) -> list[str]:
        """
        Get list of features this calculator can compute.
        
        Returns:
        --------
        list[str]
            List of feature names
        """
        pass
    
    def calculate_single_feature(self, peak: Peak, data: np.ndarray, feature_name: str, **kwargs) -> Any:
        """
        Calculate a single feature for one peak.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate feature for
        data : np.ndarray
            Original data array
        feature_name : str
            Name of feature to calculate
        **kwargs
            Additional parameters
            
        Returns:
        --------
        Any
            Calculated feature value
        """
        # Check if feature is available
        if feature_name not in self.get_available_features():
            raise ValueError(f"Feature '{feature_name}' not available in {self.__class__.__name__}")
        
        # Try to get from cache first
        cache_key = (id(peak), feature_name, str(kwargs))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate feature
        all_features = self.calculate_features([peak], data, **kwargs)
        feature_value = all_features[peak].get(feature_name)
        
        # Cache result
        self._cache[cache_key] = feature_value
        
        return feature_value
    
    def clear_cache(self):
        """Clear feature cache."""
        self._cache.clear()
    
    def validate_inputs(self, peaks: list[Peak], data: np.ndarray) -> bool:
        """
        Validate input parameters.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to validate
        data : np.ndarray
            Data to validate
            
        Returns:
        --------
        bool
            True if inputs are valid
            
        Raises:
        -------
        ValueError
            If inputs are invalid
        """
        if not peaks:
            raise ValueError("Peak list cannot be empty")
        
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        
        if data.size == 0:
            raise ValueError("Data array cannot be empty")
        
        # Validate that peak indices are within data bounds
        for peak in peaks:
            for idx in peak.plateau_indices:
                if not all(0 <= idx[i] < data.shape[i] for i in range(len(idx))):
                    raise ValueError(f"Peak index {idx} is out of bounds for data shape {data.shape}")
        
        return True
    
    def get_effective_scale(self, data_ndim: int) -> list[float]:
        """
        Get effective scale for calculations.
        
        Parameters:
        -----------
        data_ndim : int
            Number of dimensions in data
            
        Returns:
        --------
        list[float]
            Scale values for each dimension
        """
        if self.scale is None:
            return [1.0] * data_ndim
        elif isinstance(self.scale, (int, float)):
            return [float(self.scale)] * data_ndim
        elif isinstance(self.scale, (list, tuple)):
            if len(self.scale) != data_ndim:
                raise ValueError(f"Scale length ({len(self.scale)}) must match data dimensions ({data_ndim})")
            return list(self.scale)
        else:
            raise TypeError("Scale must be number, list, or None")
    
    def calculate_physical_area(self, area_pixels: int, data_ndim: int) -> float:
        """
        Convert pixel area to physical area.
        
        Parameters:
        -----------
        area_pixels : int
            Area in pixels (number of grid cells)
        data_ndim : int
            Number of dimensions
            
        Returns:
        --------
        float
            Physical area in scale units
        """
        scale = self.get_effective_scale(data_ndim)
        
        # For 2D: area = pixels * scale_x * scale_y
        # For 3D: volume = pixels * scale_x * scale_y * scale_z
        # etc.
        scale_product = np.prod(scale)
        return area_pixels * scale_product
    
    def calculate_physical_distance(self, point1: tuple[float, ...], point2: tuple[float, ...], data_ndim: int) -> float:
        """
        Calculate physical distance between two points.
        
        Parameters:
        -----------
        point1, point2 : tuple of float
            Coordinates in index or physical space
        data_ndim : int
            Number of dimensions
            
        Returns:
        --------
        float
            Physical distance in scale units
        """
        scale = self.get_effective_scale(data_ndim)
        
        # Calculate scaled Euclidean distance
        scaled_diff_sq = sum(
            ((p1 - p2) * s) ** 2 
            for p1, p2, s in zip(point1, point2, scale)
        )
        
        return np.sqrt(scaled_diff_sq)
    
    def get_peak_boundary_indices(self, peak: Peak, data_shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Get boundary indices of a peak plateau.
        
        Parameters:
        -----------
        peak : Peak
            Peak to get boundary for
        data_shape : tuple of int
            Shape of data array
            
        Returns:
        --------
        list[tuple of int]
            List of boundary indices
        """
        plateau_set = set(peak.plateau_indices)
        boundary_indices = []
        
        for idx in peak.plateau_indices:
            neighbors = self._get_neighbors(idx, data_shape)
            
            # If any neighbor is not in plateau, this is a boundary point
            for neighbor in neighbors:
                if neighbor not in plateau_set:
                    boundary_indices.append(neighbor)
                    break
        
        return list(set(boundary_indices))  # Remove duplicates
    
    def _get_neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid face-connected neighbors."""
        neighbors = []
        ndim = len(point)
        
        for dim in range(ndim):
            for delta in [-1, 1]:
                neighbor = list(point)
                neighbor[dim] += delta
                
                # Check bounds
                if 0 <= neighbor[dim] < shape[dim]:
                    neighbors.append(tuple(neighbor))
        
        return neighbors
    
    def calculate_centroid(self, indices: list[tuple[int, ...]], weights: list[float | None] = None) -> tuple[float, ...]:
        """
        Calculate centroid of a set of indices.
        
        Parameters:
        -----------
        indices : list[tuple of int]
            List of index tuples
        weights : list[float], optional
            Weights for each index
            
        Returns:
        --------
        tuple of float
            Centroid coordinates
        """
        if not indices:
            raise ValueError("Cannot calculate centroid of empty index list")
        
        if weights is None:
            weights = [1.0] * len(indices)
        elif len(weights) != len(indices):
            raise ValueError("Weights length must match indices length")
        
        # Calculate weighted centroid
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        
        ndim = len(indices[0])
        centroid = []
        
        for dim in range(ndim):
            weighted_sum = sum(idx[dim] * weight for idx, weight in zip(indices, weights))
            centroid.append(weighted_sum / total_weight)
        
        return tuple(centroid)
    
    def calculate_bounding_box(self, indices: list[tuple[int, ...]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Calculate bounding box of a set of indices.
        
        Parameters:
        -----------
        indices : list[tuple of int]
            List of index tuples
            
        Returns:
        --------
        tuple of (min_coords, max_coords)
            Bounding box coordinates
        """
        if not indices:
            raise ValueError("Cannot calculate bounding box of empty index list")
        
        ndim = len(indices[0])
        
        min_coords = []
        max_coords = []
        
        for dim in range(ndim):
            dim_values = [idx[dim] for idx in indices]
            min_coords.append(min(dim_values))
            max_coords.append(max(dim_values))
        
        return tuple(min_coords), tuple(max_coords)
    
    def get_calculator_info(self) -> dict[str, Any]:
        """
        Get information about this calculator.
        
        Returns:
        --------
        dict[str, Any]
            Calculator information
        """
        return {
            'name': self.__class__.__name__,
            'features': self.get_available_features(),
            'scale': self.scale,
            'cache_size': len(self._cache),
            'description': self.__doc__.split('\n')[1].strip() if self.__doc__ else ""
        }