"""
Topographic Feature Calculator

Calculates topographic properties of peaks including prominence, isolation,
relative height, gradients, and morphological characteristics.
"""

from typing import Any
import numpy as np

from .base_calculator import BaseCalculator
from peak_analyzer.models import Peak
from peak_analyzer.connectivity.distance_metrics import MinkowskiDistance


class TopographicCalculator(BaseCalculator):
    """
    Calculator for topographic features of peaks.
    
    Computes elevation-based properties including prominence, isolation,
    relative height calculations, gradient analysis, and morphological features.
    """
    
    def __init__(self, scale: list[float | None] = None, minkowski_p: float = 2.0):
        """
        Initialize topographic calculator.
        
        Parameters:
        -----------
        scale : list of float, optional
            Physical scale for each dimension
        minkowski_p : float
            Minkowski distance parameter for isolation calculations
        """
        super().__init__(scale)
        self.minkowski_p = minkowski_p
        self._distance_calculator = MinkowskiDistance(p=minkowski_p, scale=scale)
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray, **kwargs) -> dict[Peak, dict[str, Any]]:
        """
        Calculate topographic features for peaks.
        
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
            Topographic features for each peak
        """
        self.validate_inputs(peaks, data)
        
        features = {}
        
        # Calculate isolation requires comparing all peaks
        isolation_data = self._calculate_all_isolations(peaks, data)
        
        for peak in peaks:
            peak_features = {}
            
            # Height-based features
            peak_features['prominence'] = self.calculate_prominence(peak, data)
            peak_features['isolation'] = isolation_data.get(peak, 0.0)
            peak_features['relative_height'] = self.calculate_relative_height(peak, data)
            
            # Gradient features
            peak_features['gradient_magnitude'] = self.calculate_gradient_magnitude(peak, data)
            peak_features['slope_angle'] = self.calculate_slope_angle(peak, data)
            peak_features['aspect_angle'] = self.calculate_aspect_angle(peak, data)
            
            # Morphological features
            peak_features['roughness'] = self.calculate_roughness(peak, data)
            peak_features['curvature'] = self.calculate_curvature(peak, data)
            peak_features['drainage_properties'] = self.calculate_drainage_properties(peak, data)
            
            # Topographic position
            peak_features['topographic_position'] = self.calculate_topographic_position(peak, data)
            peak_features['terrain_ruggedness'] = self.calculate_terrain_ruggedness(peak, data)
            
            features[peak] = peak_features
        
        return features
    
    def get_available_features(self) -> list[str]:
        """Get list of available topographic features."""
        return [
            'prominence',
            'isolation',
            'relative_height',
            'gradient_magnitude',
            'slope_angle',
            'aspect_angle',
            'roughness',
            'curvature',
            'drainage_properties',
            'topographic_position',
            'terrain_ruggedness'
        ]
    
    def calculate_prominence(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate prominence of peak.
        
        Prominence is the height difference between the peak and the lowest
        contour line enclosing the peak but no higher peaks.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate prominence for
        data : np.ndarray
            Height data
            
        Returns:
        --------
        float
            Prominence value
        """
        peak_height = peak.height
        
        # Simple prominence calculation: find minimum escape height
        # For more sophisticated calculation, would need watershed analysis
        
        # Get region around peak for analysis
        search_radius = self._get_prominence_search_radius(data.shape)
        peak_center = tuple(int(np.mean([idx[i] for idx in peak.plateau_indices])) for i in range(data.ndim))
        
        # Define search region
        region_slices = []
        for i, center in enumerate(peak_center):
            start = max(0, center - search_radius)
            end = min(data.shape[i], center + search_radius + 1)
            region_slices.append(slice(start, end))
        
        region_data = data[tuple(region_slices)]
        
        # Find minimum height that would need to be crossed to reach a higher peak
        # Simplified: find minimum height on region boundary
        boundary_heights = self._get_region_boundary_heights(region_data)
        
        if boundary_heights:
            escape_height = min(boundary_heights)
            prominence = peak_height - escape_height
        else:
            # If no boundary found, prominence equals peak height
            prominence = peak_height
        
        return max(0, prominence)
    
    def calculate_isolation(self, peak: Peak, other_peaks: list[Peak], data: np.ndarray) -> float:
        """
        Calculate isolation distance of peak.
        
        Isolation is the distance to the nearest peak of equal or greater height.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate isolation for
        other_peaks : list[Peak]
            Other peaks for comparison
        data : np.ndarray
            Height data
            
        Returns:
        --------
        float
            Isolation distance
        """
        peak_height = peak.height
        peak_center = self.calculate_centroid(peak.plateau_indices)
        
        min_distance = float('inf')
        
        for other_peak in other_peaks:
            if other_peak == peak:
                continue
            
            if other_peak.height >= peak_height:
                other_center = self.calculate_centroid(other_peak.plateau_indices)
                distance = self._distance_calculator.calculate_distance(
                    peak_center, other_center, self.get_effective_scale(data.ndim)
                )
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def calculate_relative_height(self, peak: Peak, data: np.ndarray) -> dict[str, float]:
        """
        Calculate relative height measures.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        dict[str, float]
            Relative height measures
        """
        peak_height = peak.height
        
        # Calculate heights at different radii around peak
        peak_center = self.calculate_centroid(peak.plateau_indices)
        relative_heights = {}
        
        # Standard radii for relative height calculation
        radii = [10, 25, 50, 100]  # in pixels
        
        for radius in radii:
            if radius < min(data.shape) // 4:  # Only if radius fits in data
                mean_height = self._get_mean_height_at_radius(peak_center, radius, data)
                rel_height = peak_height - mean_height
                relative_heights[f'rel_height_{radius}'] = rel_height
        
        # Local relative height (immediate neighborhood)
        local_mean = self._get_local_mean_height(peak_center, data)
        relative_heights['local_rel_height'] = peak_height - local_mean
        
        return relative_heights
    
    def calculate_gradient_magnitude(self, peak: Peak, data: np.ndarray) -> dict[str, float]:
        """
        Calculate gradient magnitude around peak.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        dict[str, float]
            Gradient statistics
        """
        # Calculate gradients at plateau indices
        gradients = []
        
        for idx in peak.plateau_indices[:10]:  # Sample for efficiency
            grad = self._calculate_local_gradient(idx, data)
            gradient_magnitude = np.linalg.norm(grad)
            gradients.append(gradient_magnitude)
        
        if gradients:
            return {
                'mean_gradient': np.mean(gradients),
                'max_gradient': np.max(gradients),
                'min_gradient': np.min(gradients),
                'std_gradient': np.std(gradients)
            }
        else:
            return {'mean_gradient': 0.0, 'max_gradient': 0.0, 'min_gradient': 0.0, 'std_gradient': 0.0}
    
    def calculate_slope_angle(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate average slope angle around peak.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        float
            Slope angle in degrees
        """
        gradient_stats = self.calculate_gradient_magnitude(peak, data)
        mean_gradient = gradient_stats['mean_gradient']
        
        # Convert gradient to angle
        slope_angle = np.degrees(np.arctan(mean_gradient))
        return slope_angle
    
    def calculate_aspect_angle(self, peak: Peak, data: np.ndarray) -> dict[str, float]:
        """
        Calculate aspect (direction of steepest descent).
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        dict[str, float]
            Aspect angles for available dimensions
        """
        # Only meaningful for 2D+ data
        if data.ndim < 2:
            return {}
        
        peak_center = self.calculate_centroid(peak.plateau_indices)
        center_idx = tuple(int(coord) for coord in peak_center)
        
        # Calculate gradient at peak center
        grad = self._calculate_local_gradient(center_idx, data)
        
        aspects = {}
        
        if data.ndim >= 2:
            # 2D aspect (compass direction)
            if len(grad) >= 2 and (grad[0] != 0 or grad[1] != 0):
                aspect_2d = np.degrees(np.arctan2(grad[1], grad[0]))
                # Convert to compass bearing (0 = North, 90 = East)
                compass_aspect = (90 - aspect_2d) % 360
                aspects['compass_aspect'] = compass_aspect
        
        if data.ndim >= 3:
            # 3D aspects
            if len(grad) >= 3:
                # XY plane aspect
                if grad[0] != 0 or grad[1] != 0:
                    xy_aspect = np.degrees(np.arctan2(grad[1], grad[0]))
                    aspects['xy_aspect'] = xy_aspect
                
                # XZ plane aspect  
                if grad[0] != 0 or grad[2] != 0:
                    xz_aspect = np.degrees(np.arctan2(grad[2], grad[0]))
                    aspects['xz_aspect'] = xz_aspect
                
                # YZ plane aspect
                if grad[1] != 0 or grad[2] != 0:
                    yz_aspect = np.degrees(np.arctan2(grad[2], grad[1]))
                    aspects['yz_aspect'] = yz_aspect
        
        return aspects
    
    def calculate_roughness(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate surface roughness of peak area.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        float
            Roughness measure
        """
        # Calculate standard deviation of heights in peak region
        heights = [data[idx] for idx in peak.plateau_indices]
        
        if len(heights) > 1:
            return np.std(heights)
        else:
            return 0.0
    
    def calculate_curvature(self, peak: Peak, data: np.ndarray) -> dict[str, float]:
        """
        Calculate curvature measures around peak.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        dict[str, float]
            Curvature measures
        """
        peak_center = self.calculate_centroid(peak.plateau_indices)
        center_idx = tuple(int(coord) for coord in peak_center)
        
        # Calculate second derivatives for curvature
        curvatures = {}
        
        try:
            # Mean curvature approximation
            laplacian = self._calculate_laplacian(center_idx, data)
            curvatures['mean_curvature'] = laplacian / 2.0
            
            # Gaussian curvature (for 2D surfaces)
            if data.ndim == 2:
                gaussian_curv = self._calculate_gaussian_curvature(center_idx, data)
                curvatures['gaussian_curvature'] = gaussian_curv
            
            # Principal curvatures
            principal_curvs = self._calculate_principal_curvatures(center_idx, data)
            curvatures.update(principal_curvs)
            
        except (IndexError, ValueError):
            # If calculation fails, return zeros
            curvatures = {'mean_curvature': 0.0}
        
        return curvatures
    
    def calculate_drainage_properties(self, peak: Peak, data: np.ndarray) -> dict[str, Any]:
        """
        Calculate drainage-related properties.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        dict[str, Any]
            Drainage properties
        """
        properties = {}
        
        # Drainage density (simplified)
        local_gradients = []
        
        # Sample gradients in neighborhood
        for idx in peak.plateau_indices[:20]:  # Sample for efficiency
            grad = self._calculate_local_gradient(idx, data)
            local_gradients.append(np.linalg.norm(grad))
        
        if local_gradients:
            properties['drainage_density'] = np.mean(local_gradients)
        else:
            properties['drainage_density'] = 0.0
        
        # Flow direction consistency
        flow_directions = []
        for idx in peak.plateau_indices[:10]:
            grad = self._calculate_local_gradient(idx, data)
            if np.linalg.norm(grad) > 1e-10:
                direction = grad / np.linalg.norm(grad)
                flow_directions.append(direction)
        
        if len(flow_directions) > 1:
            # Calculate consistency as inverse of standard deviation of directions
            flow_array = np.array(flow_directions)
            direction_std = np.mean(np.std(flow_array, axis=0))
            properties['flow_consistency'] = 1.0 / (1.0 + direction_std)
        else:
            properties['flow_consistency'] = 1.0
        
        return properties
    
    def calculate_topographic_position(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate Topographic Position Index (TPI).
        
        TPI = peak_height - mean_height_in_neighborhood
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        float
            Topographic Position Index
        """
        peak_height = peak.height
        peak_center = self.calculate_centroid(peak.plateau_indices)
        
        # Calculate mean height in neighborhood
        neighborhood_mean = self._get_local_mean_height(peak_center, data, radius=20)
        
        tpi = peak_height - neighborhood_mean
        return tpi
    
    def calculate_terrain_ruggedness(self, peak: Peak, data: np.ndarray) -> float:
        """
        Calculate Terrain Ruggedness Index (TRI).
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
            
        Returns:
        --------
        float
            Terrain Ruggedness Index
        """
        # Calculate TRI as mean of squared height differences
        height_diffs_squared = []
        
        for idx in peak.plateau_indices[:20]:  # Sample for efficiency
            # Get neighbors
            neighbors = self._get_valid_neighbors(idx, data.shape)
            
            center_height = data[idx]
            for neighbor_idx in neighbors:
                neighbor_height = data[neighbor_idx]
                diff_squared = (center_height - neighbor_height) ** 2
                height_diffs_squared.append(diff_squared)
        
        if height_diffs_squared:
            tri = np.sqrt(np.mean(height_diffs_squared))
        else:
            tri = 0.0
        
        return tri
    
    # Helper methods
    
    def _calculate_all_isolations(self, peaks: list[Peak], data: np.ndarray) -> dict[Peak, float]:
        """Calculate isolation for all peaks efficiently."""
        isolation_data = {}
        
        for i, peak in enumerate(peaks):
            other_peaks = peaks[:i] + peaks[i+1:]
            isolation = self.calculate_isolation(peak, other_peaks, data)
            isolation_data[peak] = isolation
        
        return isolation_data
    
    def _get_prominence_search_radius(self, data_shape: tuple[int, ...]) -> int:
        """Get search radius for prominence calculation."""
        return min(data_shape) // 4
    
    def _get_region_boundary_heights(self, region_data: np.ndarray) -> list[float]:
        """Get heights on boundary of region."""
        heights = []
        
        # Get boundary points (simplified - just edges)
        if region_data.ndim == 2:
            # 2D boundary
            heights.extend(region_data[0, :])  # Top edge
            heights.extend(region_data[-1, :])  # Bottom edge
            heights.extend(region_data[:, 0])  # Left edge
            heights.extend(region_data[:, -1])  # Right edge
        else:
            # Higher dimensions - use flattened boundary approximation
            for dim in range(region_data.ndim):
                # First and last slice in each dimension
                front_slice = tuple(0 if i == dim else slice(None) for i in range(region_data.ndim))
                back_slice = tuple(-1 if i == dim else slice(None) for i in range(region_data.ndim))
                
                heights.extend(region_data[front_slice].flatten())
                heights.extend(region_data[back_slice].flatten())
        
        return list(set(heights))  # Remove duplicates
    
    def _get_mean_height_at_radius(self, center: tuple[float, ...], radius: float, data: np.ndarray) -> float:
        """Get mean height at given radius from center."""
        # Create circular/spherical mask
        indices = np.indices(data.shape)
        distances = np.sqrt(sum((indices[i] - center[i])**2 for i in range(len(center))))
        
        # Points within radius band
        mask = (distances >= radius - 0.5) & (distances <= radius + 0.5)
        
        if np.any(mask):
            return np.mean(data[mask])
        else:
            return data[tuple(int(c) for c in center)]
    
    def _get_local_mean_height(self, center: tuple[float, ...], data: np.ndarray, radius: int = 5) -> float:
        """Get mean height in local neighborhood."""
        center_idx = tuple(int(coord) for coord in center)
        
        # Define neighborhood bounds
        slices = []
        for i, coord in enumerate(center_idx):
            start = max(0, coord - radius)
            end = min(data.shape[i], coord + radius + 1)
            slices.append(slice(start, end))
        
        neighborhood = data[tuple(slices)]
        return np.mean(neighborhood)
    
    def _calculate_local_gradient(self, idx: tuple[int, ...], data: np.ndarray) -> np.ndarray:
        """Calculate local gradient at given index."""
        gradient = np.zeros(len(idx))
        
        for dim in range(len(idx)):
            # Forward and backward differences
            forward_idx = list(idx)
            backward_idx = list(idx)
            
            if forward_idx[dim] < data.shape[dim] - 1:
                forward_idx[dim] += 1
                forward_diff = data[tuple(forward_idx)] - data[idx]
            else:
                forward_diff = 0
            
            if backward_idx[dim] > 0:
                backward_idx[dim] -= 1
                backward_diff = data[idx] - data[tuple(backward_idx)]
            else:
                backward_diff = 0
            
            # Central difference where possible
            if forward_idx[dim] < data.shape[dim] - 1 and backward_idx[dim] > 0:
                gradient[dim] = (forward_diff + backward_diff) / 2.0
            elif forward_diff != 0:
                gradient[dim] = forward_diff
            elif backward_diff != 0:
                gradient[dim] = backward_diff
            
        return gradient
    
    def _calculate_laplacian(self, idx: tuple[int, ...], data: np.ndarray) -> float:
        """Calculate Laplacian at given index."""
        laplacian = 0.0
        center_value = data[idx]
        
        for dim in range(len(idx)):
            # Second derivative in each dimension
            forward_idx = list(idx)
            backward_idx = list(idx)
            
            if (idx[dim] > 0 and idx[dim] < data.shape[dim] - 1):
                forward_idx[dim] += 1
                backward_idx[dim] -= 1
                
                second_deriv = (data[tuple(forward_idx)] - 2 * center_value + data[tuple(backward_idx)])
                laplacian += second_deriv
        
        return laplacian
    
    def _calculate_gaussian_curvature(self, idx: tuple[int, ...], data: np.ndarray) -> float:
        """Calculate Gaussian curvature for 2D surface."""
        if data.ndim != 2:
            return 0.0
        
        i, j = idx
        if not (1 <= i < data.shape[0] - 1 and 1 <= j < data.shape[1] - 1):
            return 0.0
        
        # Calculate second derivatives
        fxx = data[i-1, j] - 2*data[i, j] + data[i+1, j]
        fyy = data[i, j-1] - 2*data[i, j] + data[i, j+1]
        fxy = (data[i+1, j+1] - data[i+1, j-1] - data[i-1, j+1] + data[i-1, j-1]) / 4
        
        # Gaussian curvature
        gaussian_curvature = fxx * fyy - fxy**2
        return gaussian_curvature
    
    def _calculate_principal_curvatures(self, idx: tuple[int, ...], data: np.ndarray) -> dict[str, float]:
        """Calculate principal curvatures."""
        if data.ndim < 2:
            return {}
        
        # Simplified calculation for 2D case
        if data.ndim == 2:
            i, j = idx
            if not (1 <= i < data.shape[0] - 1 and 1 <= j < data.shape[1] - 1):
                return {'max_curvature': 0.0, 'min_curvature': 0.0}
            
            # Calculate Hessian matrix elements
            fxx = data[i-1, j] - 2*data[i, j] + data[i+1, j]
            fyy = data[i, j-1] - 2*data[i, j] + data[i, j+1]
            fxy = (data[i+1, j+1] - data[i+1, j-1] - data[i-1, j+1] + data[i-1, j-1]) / 4
            
            # Eigenvalues of Hessian are principal curvatures
            trace = fxx + fyy
            det = fxx * fyy - fxy**2
            discriminant = trace**2 - 4*det
            
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                k1 = (trace + sqrt_disc) / 2
                k2 = (trace - sqrt_disc) / 2
                return {'max_curvature': max(k1, k2), 'min_curvature': min(k1, k2)}
        
        return {'max_curvature': 0.0, 'min_curvature': 0.0}
    
    def _get_valid_neighbors(self, idx: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbor indices."""
        neighbors = []
        
        for dim in range(len(idx)):
            for delta in [-1, 1]:
                neighbor = list(idx)
                neighbor[dim] += delta
                
                if 0 <= neighbor[dim] < shape[dim]:
                    neighbors.append(tuple(neighbor))
        
        return neighbors