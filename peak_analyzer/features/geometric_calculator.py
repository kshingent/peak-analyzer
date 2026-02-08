"""
Geometric Feature Calculator

Calculates geometric properties of peaks including area, volume, centroid,
aspect ratio, and bounding box.
"""

from typing import Any
import numpy as np

from .base_calculator import BaseCalculator
from ..models import Peak


class GeometricCalculator(BaseCalculator):
    """
    Calculator for geometric features of peaks.
    
    Computes spatial and volumetric properties including area, volume,
    centroid position, aspect ratios, and bounding box dimensions.
    """
    
    def __init__(self, scale: list[float | None] = None):
        """
        Initialize geometric calculator.
        
        Parameters:
        -----------
        scale : list of float, optional
            Physical scale for each dimension
        """
        super().__init__(scale)
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray, **kwargs) -> dict[Peak, dict[str, Any]]:
        """
        Calculate geometric features for peaks.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to calculate features for
        data : np.ndarray
            Original data array
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict[Peak, dict[str, Any]]
            Geometric features for each peak
        """
        self.validate_inputs(peaks, data)
        
        features = {}
        
        for peak in peaks:
            peak_features = {}
            
            # Calculate all geometric features
            peak_features['area'] = self.calculate_area(peak, data)
            peak_features['volume'] = self.calculate_volume(peak, data)
            peak_features['centroid'] = self.calculate_centroid_feature(peak, data)
            peak_features['aspect_ratio'] = self.calculate_aspect_ratio(peak)
            peak_features['bounding_box'] = self.calculate_bounding_box_feature(peak)
            peak_features['perimeter'] = self.calculate_perimeter(peak, data.shape)
            peak_features['compactness'] = self.calculate_compactness(peak, data.shape)
            peak_features['elongation'] = self.calculate_elongation(peak)
            peak_features['extent'] = self.calculate_extent(peak)
            
            features[peak] = peak_features
        
        return features
    
    def get_available_features(self) -> list[str]:
        """Get list of available geometric features."""
        return [
            'area',
            'volume', 
            'centroid',
            'aspect_ratio',
            'bounding_box',
            'perimeter',
            'compactness',
            'elongation',
            'extent'
        ]
    
    def calculate_area(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate area of peak plateau region.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate area for
        data : np.ndarray
            Data array for context
            
        Returns:
        --------
        float
            Area in physical units (or pixels if no scale)
        """
        pixel_count = len(peak.plateau_indices)
        return self.calculate_physical_area(pixel_count, data.ndim)
    
    def calculate_volume(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate volume of peak region above a reference level.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate volume for
        data : np.ndarray
            Data array
            
        Returns:
        --------
        float
            Volume in physical units
        """
        # Use minimum height in plateau as reference
        heights = [data[idx] for idx in peak.plateau_indices]
        min_height = min(heights)
        
        # Calculate volume as sum of heights above reference
        volume_pixels = sum(h - min_height for h in heights)
        
        # Convert to physical units
        scale = self.get_effective_scale(data.ndim)
        if data.ndim == 2:
            # 2D: volume is area * average_height
            unit_area = np.prod(scale)
            return volume_pixels * unit_area
        else:
            # 3D+: volume is sum of voxel volumes
            unit_volume = np.prod(scale)
            return volume_pixels * unit_volume
    
    def calculate_centroid_feature(self, peak: Peak, data: np.ndarray) -> tuple[float, ...]:
        """
        Calculate weighted centroid of peak region.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate centroid for
        data : np.ndarray
            Data array for weighting
            
        Returns:
        --------
        tuple of float
            Centroid coordinates in index space
        """
        # Use heights as weights
        weights = [data[idx] for idx in peak.plateau_indices]
        return self.calculate_centroid(peak.plateau_indices, weights)
    
    def calculate_aspect_ratio(self, peak: Peak) -> dict[str, float]:
        """
        Calculate aspect ratios along different dimensions.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
            
        Returns:
        --------
        dict[str, float]
            Aspect ratios for different dimension pairs
        """
        min_coords, max_coords = self.calculate_bounding_box(peak.plateau_indices)
        
        # Calculate extents in each dimension
        extents = [max_coords[i] - min_coords[i] + 1 for i in range(len(min_coords))]
        
        aspect_ratios = {}
        
        # Calculate pairwise aspect ratios
        for i in range(len(extents)):
            for j in range(i + 1, len(extents)):
                ratio_name = f"aspect_ratio_{i}_{j}"
                if extents[j] > 0:
                    aspect_ratios[ratio_name] = extents[i] / extents[j]
                else:
                    aspect_ratios[ratio_name] = float('inf')
        
        # Overall aspect ratio (max/min extent)
        if extents:
            max_extent = max(extents)
            min_extent = min(extents)
            if min_extent > 0:
                aspect_ratios['overall_aspect_ratio'] = max_extent / min_extent
            else:
                aspect_ratios['overall_aspect_ratio'] = float('inf')
        
        return aspect_ratios
    
    def calculate_bounding_box_feature(self, peak: Peak) -> dict[str, Any]:
        """
        Calculate bounding box properties.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
            
        Returns:
        --------
        dict[str, Any]
            Bounding box information
        """
        min_coords, max_coords = self.calculate_bounding_box(peak.plateau_indices)
        
        # Calculate dimensions
        dimensions = [max_coords[i] - min_coords[i] + 1 for i in range(len(min_coords))]
        
        # Calculate bounding box area/volume
        bounding_volume = np.prod(dimensions)
        
        return {
            'min_coords': min_coords,
            'max_coords': max_coords,
            'dimensions': dimensions,
            'bounding_volume': bounding_volume
        }
    
    def calculate_perimeter(self, peak: Peak, data_shape: tuple[int, ...]) -> float:
        """
        Calculate perimeter of peak region.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data_shape : tuple
            Shape of data array
            
        Returns:
        --------
        float
            Perimeter in physical units
        """
        # Get boundary points
        boundary_indices = self.get_peak_boundary_indices(peak, data_shape)
        
        # For 2D, perimeter is count of boundary edges
        # For higher dimensions, this is a generalized "surface area"
        scale = self.get_effective_scale(len(data_shape))
        
        if len(data_shape) == 2:
            # 2D perimeter: count boundary edges and scale
            edge_length = min(scale)  # Use smaller scale for edge length
            return len(boundary_indices) * edge_length
        else:
            # Higher dimensions: surface area calculation
            surface_area = len(boundary_indices) * np.prod(scale[:-1])  # Approximate
            return surface_area
    
    def calculate_compactness(self, peak: Peak, data_shape: tuple[int, ...]) -> float:
        """
        Calculate compactness ratio of peak region.
        
        Compactness = (perimeter^2) / area for 2D
        Higher values indicate less compact shapes.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data_shape : tuple
            Shape of data array
            
        Returns:
        --------
        float
            Compactness ratio
        """
        area = len(peak.plateau_indices)
        perimeter = len(self.get_peak_boundary_indices(peak, data_shape))
        
        if area > 0:
            return (perimeter ** 2) / area
        else:
            return float('inf')
    
    def calculate_elongation(self, peak: Peak) -> float:
        """
        Calculate elongation of peak region.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
            
        Returns:
        --------
        float
            Elongation factor
        """
        # Calculate moments of inertia to determine elongation
        indices_array = np.array(peak.plateau_indices)
        
        if len(indices_array) == 0:
            return 0.0
        
        # Calculate covariance matrix
        centroid = np.mean(indices_array, axis=0)
        centered = indices_array - centroid
        
        if len(centered) > 1:
            cov_matrix = np.cov(centered.T)
            
            # Get eigenvalues
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.real(eigenvals)  # Take real part
            eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
            
            # Elongation is ratio of largest to smallest eigenvalue
            if eigenvals[-1] > 1e-10:  # Avoid division by zero
                return eigenvals[0] / eigenvals[-1]
            else:
                return float('inf')
        else:
            return 1.0
    
    def calculate_extent(self, peak: Peak) -> float:
        """
        Calculate extent (fill ratio) of peak region.
        
        Extent = area / bounding_box_area
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
            
        Returns:
        --------
        float
            Extent ratio (0 to 1)
        """
        actual_area = len(peak.plateau_indices)
        
        # Calculate bounding box area
        min_coords, max_coords = self.calculate_bounding_box(peak.plateau_indices)
        bounding_dims = [max_coords[i] - min_coords[i] + 1 for i in range(len(min_coords))]
        bounding_area = np.prod(bounding_dims)
        
        if bounding_area > 0:
            return actual_area / bounding_area
        else:
            return 0.0
    
    def calculate_peak_moments(self, peak: Peak, data: np.ndarray) -> dict[str, float]:
        """
        Calculate statistical moments of peak region.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Data array
            
        Returns:
        --------
        dict[str, float]
            Statistical moments
        """
        indices_array = np.array(peak.plateau_indices)
        
        if len(indices_array) == 0:
            return {}
        
        # Calculate spatial moments
        moments = {}
        
        # First moments (centroid)
        centroid = np.mean(indices_array, axis=0)
        for i, coord in enumerate(centroid):
            moments[f'mean_{i}'] = coord
        
        # Second moments (variance)
        variance = np.var(indices_array, axis=0)
        for i, var in enumerate(variance):
            moments[f'variance_{i}'] = var
            moments[f'std_{i}'] = np.sqrt(var)
        
        # Skewness and kurtosis
        from scipy.stats import skew, kurtosis
        
        for i in range(indices_array.shape[1]):
            values = indices_array[:, i]
            moments[f'skewness_{i}'] = skew(values)
            moments[f'kurtosis_{i}'] = kurtosis(values)
        
        return moments