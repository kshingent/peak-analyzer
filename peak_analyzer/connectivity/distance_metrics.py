"""
Distance Metrics

Implements various distance calculation utilities for N-dimensional spaces
including Minkowski distances, weighted distances, and topographic distances.
"""

from typing import Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from .connectivity_types import ConnectivityPattern


class DistanceType(Enum):
    """
    Enumeration of distance metric types.
    """
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan" 
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"
    WEIGHTED_EUCLIDEAN = "weighted_euclidean"
    MAHALANOBIS = "mahalanobis"
    GEODESIC = "geodesic"
    TOPOGRAPHIC = "topographic"


class DistanceMetric(ABC):
    """
    Abstract base class for distance metrics.
    """
    
    def __init__(self, scale: list[float | None] = None):
        """
        Initialize distance metric.
        
        Parameters:
        -----------
        scale : list of float, optional
            Scaling factors for each dimension
        """
        self.scale = scale
    
    @abstractmethod
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """
        Calculate distance between two positions.
        
        Parameters:
        -----------
        pos1, pos2 : tuple of float
            Positions to calculate distance between
            
        Returns:
        --------
        float
            Distance between positions
        """
        pass
    
    def calculate_distances_to_points(self, center: tuple[float, ...], 
                                    points: list[tuple[float, ...]]) -> list[float]:
        """
        Calculate distances from center to multiple points.
        
        Parameters:
        -----------
        center : tuple of float
            Center position
        points : list[tuple of float]
            Points to calculate distances to
            
        Returns:
        --------
        list[float]
            Distances to each point
        """
        return [self.calculate_distance(center, point) for point in points]
    
    def calculate_pairwise_distances(self, points: list[tuple[float, ...]]) -> np.ndarray:
        """
        Calculate pairwise distances between all points.
        
        Parameters:
        -----------
        points : list[tuple of float]
            List of points
            
        Returns:
        --------
        np.ndarray
            Square distance matrix
        """
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.calculate_distance(points[i], points[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _apply_scale(self, pos: tuple[float, ...]) -> np.ndarray:
        """Apply scaling to position coordinates."""
        pos_array = np.array(pos)
        
        if self.scale is not None:
            if len(self.scale) != len(pos):
                raise ValueError(f"Scale length {len(self.scale)} does not match position dimensions {len(pos)}")
            pos_array *= np.array(self.scale)
        
        return pos_array


class EuclideanDistance(DistanceMetric):
    """
    Euclidean (L2) distance metric.
    """
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate Euclidean distance."""
        scaled_pos1 = self._apply_scale(pos1)
        scaled_pos2 = self._apply_scale(pos2)
        
        diff = scaled_pos1 - scaled_pos2
        return float(np.linalg.norm(diff))


class ManhattanDistance(DistanceMetric):
    """
    Manhattan (L1) distance metric.
    """
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate Manhattan distance."""
        scaled_pos1 = self._apply_scale(pos1)
        scaled_pos2 = self._apply_scale(pos2)
        
        diff = np.abs(scaled_pos1 - scaled_pos2)
        return float(np.sum(diff))


class ChebyshevDistance(DistanceMetric):
    """
    Chebyshev (L∞) distance metric.
    """
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate Chebyshev distance."""
        scaled_pos1 = self._apply_scale(pos1)
        scaled_pos2 = self._apply_scale(pos2)
        
        diff = np.abs(scaled_pos1 - scaled_pos2)
        return float(np.max(diff))


class MinkowskiDistance(DistanceMetric):
    """
    Minkowski (Lp) distance metric.
    """
    
    def __init__(self, p: float = 2.0, scale: list[float | None] = None):
        """
        Initialize Minkowski distance.
        
        Parameters:
        -----------
        p : float
            Order of the Minkowski distance (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)
        scale : list of float, optional
            Scaling factors
        """
        super().__init__(scale)
        self.p = p
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate Minkowski distance."""
        scaled_pos1 = self._apply_scale(pos1)
        scaled_pos2 = self._apply_scale(pos2)
        
        diff = np.abs(scaled_pos1 - scaled_pos2)
        
        if np.isinf(self.p):
            # Chebyshev case
            return float(np.max(diff))
        else:
            return float(np.power(np.sum(np.power(diff, self.p)), 1.0 / self.p))


class WeightedEuclideanDistance(DistanceMetric):
    """
    Weighted Euclidean distance metric.
    """
    
    def __init__(self, weights: list[float], scale: list[float | None] = None):
        """
        Initialize weighted Euclidean distance.
        
        Parameters:
        -----------
        weights : list[float]
            Weight for each dimension
        scale : list of float, optional
            Scaling factors
        """
        super().__init__(scale)
        self.weights = np.array(weights)
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate weighted Euclidean distance."""
        scaled_pos1 = self._apply_scale(pos1)
        scaled_pos2 = self._apply_scale(pos2)
        
        diff = scaled_pos1 - scaled_pos2
        weighted_diff_squared = self.weights * (diff ** 2)
        
        return float(np.sqrt(np.sum(weighted_diff_squared)))


class MahalanobisDistance(DistanceMetric):
    """
    Mahalanobis distance metric using covariance matrix.
    """
    
    def __init__(self, covariance_matrix: np.ndarray, scale: list[float | None] = None):
        """
        Initialize Mahalanobis distance.
        
        Parameters:
        -----------
        covariance_matrix : np.ndarray
            Covariance matrix for the metric
        scale : list of float, optional
            Scaling factors
        """
        super().__init__(scale)
        self.covariance_matrix = covariance_matrix
        self.inv_cov_matrix = np.linalg.inv(covariance_matrix)
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate Mahalanobis distance."""
        scaled_pos1 = self._apply_scale(pos1)
        scaled_pos2 = self._apply_scale(pos2)
        
        diff = scaled_pos1 - scaled_pos2
        mahal_dist_squared = diff.T @ self.inv_cov_matrix @ diff
        
        return float(np.sqrt(mahal_dist_squared))


class TopographicDistance(DistanceMetric):
    """
    Distance metric that incorporates topographic information.
    """
    
    def __init__(self, data: np.ndarray, 
                 height_weight: float = 1.0,
                 base_metric: DistanceMetric = None,
                 scale: list[float | None] = None):
        """
        Initialize topographic distance.
        
        Parameters:
        -----------
        data : np.ndarray
            Height data array
        height_weight : float
            Weight for elevation differences
        base_metric : DistanceMetric
            Base spatial distance metric
        scale : list of float, optional
            Scaling factors
        """
        super().__init__(scale)
        self.data = data
        self.height_weight = height_weight
        self.base_metric = base_metric or EuclideanDistance(scale)
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate topographic distance."""
        # Spatial distance
        spatial_distance = self.base_metric.calculate_distance(pos1, pos2)
        
        # Height difference
        try:
            # Convert to integer indices for data access
            idx1 = tuple(int(round(coord)) for coord in pos1)
            idx2 = tuple(int(round(coord)) for coord in pos2) 
            
            # Check bounds
            if (all(0 <= idx1[i] < self.data.shape[i] for i in range(len(idx1))) and
                all(0 <= idx2[i] < self.data.shape[i] for i in range(len(idx2)))):
                
                height1 = self.data[idx1]
                height2 = self.data[idx2]
                height_diff = abs(height1 - height2)
            else:
                height_diff = 0.0  # Out of bounds, no height component
                
        except (IndexError, ValueError):
            height_diff = 0.0
        
        # Combine spatial and height components
        total_distance = spatial_distance + self.height_weight * height_diff
        
        return float(total_distance)


class GeodesicDistance(DistanceMetric):
    """
    Geodesic distance along surface paths.
    """
    
    def __init__(self, connectivity: ConnectivityPattern, 
                 pathfinder_algorithm: str = "dijkstra",
                 scale: list[float | None] = None):
        """
        Initialize geodesic distance.
        
        Parameters:
        -----------
        connectivity : ConnectivityPattern
            Connectivity for pathfinding
        pathfinder_algorithm : str
            Pathfinding algorithm to use
        scale : list of float, optional
            Scaling factors
        """
        super().__init__(scale)
        self.connectivity = connectivity
        self.pathfinder_algorithm = pathfinder_algorithm
        self._path_cache = {}
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate geodesic distance via pathfinding."""
        # Convert to integer indices
        idx1 = tuple(int(round(coord)) for coord in pos1)
        idx2 = tuple(int(round(coord)) for coord in pos2)
        
        # Check cache
        cache_key = (idx1, idx2)
        reverse_key = (idx2, idx1)  # Geodesic distance is symmetric
        
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        elif reverse_key in self._path_cache:
            return self._path_cache[reverse_key]
        
        # Calculate geodesic distance via pathfinding
        # This is a simplified implementation - full geodesic would require actual pathfinding
        euclidean_metric = EuclideanDistance(self.scale)
        euclidean_dist = euclidean_metric.calculate_distance(pos1, pos2)
        
        # Cache result (using simpler approximation for now)
        geodesic_dist = euclidean_dist * 1.1  # Approximate increase for path vs straight line
        
        self._path_cache[cache_key] = geodesic_dist
        
        return geodesic_dist


class AdaptiveDistance(DistanceMetric):
    """
    Distance metric that adapts based on local conditions.
    """
    
    def __init__(self, 
                 base_metric: DistanceMetric,
                 adaptation_function: Callable[[tuple[float, ...], tuple[float, ...]], float],
                 scale: list[float | None] = None):
        """
        Initialize adaptive distance.
        
        Parameters:
        -----------
        base_metric : DistanceMetric
            Base distance metric
        adaptation_function : callable
            Function that returns adaptation factor based on positions
        scale : list of float, optional 
            Scaling factors
        """
        super().__init__(scale)
        self.base_metric = base_metric
        self.adaptation_function = adaptation_function
    
    def calculate_distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...]) -> float:
        """Calculate adaptive distance."""
        base_distance = self.base_metric.calculate_distance(pos1, pos2)
        adaptation_factor = self.adaptation_function(pos1, pos2)
        
        return base_distance * adaptation_factor


class DistanceCalculator:
    """
    Unified calculator for various distance computations.
    """
    
    def __init__(self, metric: DistanceMetric):
        """
        Initialize distance calculator.
        
        Parameters:
        -----------
        metric : DistanceMetric
            Distance metric to use
        """
        self.metric = metric
        self._distance_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def distance(self, pos1: tuple[float, ...], pos2: tuple[float, ...], use_cache: bool = True) -> float:
        """
        Calculate distance with optional caching.
        
        Parameters:
        -----------
        pos1, pos2 : tuple of float
            Positions
        use_cache : bool
            Whether to use caching
            
        Returns:
        --------
        float
            Distance between positions
        """
        if not use_cache:
            return self.metric.calculate_distance(pos1, pos2)
        
        # Create cache key (normalize order for symmetric distances)
        cache_key = (pos1, pos2) if pos1 <= pos2 else (pos2, pos1)
        
        if cache_key in self._distance_cache:
            self.cache_hits += 1
            return self._distance_cache[cache_key]
        
        # Calculate and cache
        dist = self.metric.calculate_distance(pos1, pos2)
        self._distance_cache[cache_key] = dist
        self.cache_misses += 1
        
        return dist
    
    def batch_distances(self, center: tuple[float, ...], 
                       points: list[tuple[float, ...]]) -> list[float]:
        """Calculate distances from center to multiple points."""
        return [self.distance(center, point) for point in points]
    
    def pairwise_distances(self, points: list[tuple[float, ...]]) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distance(points[i], points[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def nearest_neighbors(self, center: tuple[float, ...], 
                         candidates: list[tuple[float, ...]],
                         k: int = 1) -> list[tuple[tuple[float, ...], float]]:
        """
        Find k nearest neighbors.
        
        Parameters:
        -----------
        center : tuple of float
            Center position
        candidates : list[tuple of float]
            Candidate positions
        k : int
            Number of neighbors to return
            
        Returns:
        --------
        list[tuple[tuple of float, float]]
            List of (position, distance) pairs for k nearest neighbors
        """
        distances = [(candidate, self.distance(center, candidate)) for candidate in candidates]
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
    
    def neighbors_within_radius(self, center: tuple[float, ...], 
                               candidates: list[tuple[float, ...]],
                               radius: float) -> list[tuple[tuple[float, ...], float]]:
        """
        Find neighbors within radius.
        
        Parameters:
        -----------
        center : tuple of float
            Center position
        candidates : list[tuple of float]
            Candidate positions
        radius : float
            Search radius
            
        Returns:
        --------
        list[tuple[tuple of float, float]]
            List of (position, distance) pairs within radius
        """
        neighbors = []
        
        for candidate in candidates:
            dist = self.distance(center, candidate)
            if dist <= radius:
                neighbors.append((candidate, dist))
        
        return neighbors
    
    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._distance_cache)
        }
    
    def clear_cache(self):
        """Clear distance cache."""
        self._distance_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class DistanceMetricFactory:
    """
    Factory for creating distance metric instances.
    """
    
    @staticmethod
    def create_metric(distance_type: DistanceType, **kwargs) -> DistanceMetric:
        """
        Create distance metric instance.
        
        Parameters:
        -----------
        distance_type : DistanceType
            Type of distance metric
        **kwargs
            Metric-specific parameters
            
        Returns:
        --------
        DistanceMetric
            Distance metric instance
        """
        scale = kwargs.get('scale', None)
        
        if distance_type == DistanceType.EUCLIDEAN:
            return EuclideanDistance(scale)
        elif distance_type == DistanceType.MANHATTAN:
            return ManhattanDistance(scale)
        elif distance_type == DistanceType.CHEBYSHEV:
            return ChebyshevDistance(scale)
        elif distance_type == DistanceType.MINKOWSKI:
            p = kwargs.get('p', 2.0)
            return MinkowskiDistance(p, scale)
        elif distance_type == DistanceType.WEIGHTED_EUCLIDEAN:
            weights = kwargs.get('weights', [1.0])
            return WeightedEuclideanDistance(weights, scale)
        elif distance_type == DistanceType.MAHALANOBIS:
            cov_matrix = kwargs.get('covariance_matrix')
            if cov_matrix is None:
                raise ValueError("Mahalanobis distance requires covariance_matrix parameter")
            return MahalanobisDistance(cov_matrix, scale)
        elif distance_type == DistanceType.TOPOGRAPHIC:
            data = kwargs.get('data')
            if data is None:
                raise ValueError("Topographic distance requires data parameter")
            height_weight = kwargs.get('height_weight', 1.0)
            base_metric = kwargs.get('base_metric', None)
            return TopographicDistance(data, height_weight, base_metric, scale)
        elif distance_type == DistanceType.GEODESIC:
            connectivity = kwargs.get('connectivity')
            if connectivity is None:
                raise ValueError("Geodesic distance requires connectivity parameter")
            algorithm = kwargs.get('pathfinder_algorithm', 'dijkstra')
            return GeodesicDistance(connectivity, algorithm, scale)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
    
    @staticmethod
    def create_from_string(name: str, **kwargs) -> DistanceMetric:
        """
        Create distance metric from string name.
        
        Parameters:
        -----------
        name : str
            Distance metric name
        **kwargs
            Metric-specific parameters
            
        Returns:
        --------
        DistanceMetric
            Distance metric instance
        """
        name_lower = name.lower()
        
        if name_lower in ['euclidean', 'l2']:
            return DistanceMetricFactory.create_metric(DistanceType.EUCLIDEAN, **kwargs)
        elif name_lower in ['manhattan', 'l1', 'cityblock']:
            return DistanceMetricFactory.create_metric(DistanceType.MANHATTAN, **kwargs)
        elif name_lower in ['chebyshev', 'linf', 'maximum']:
            return DistanceMetricFactory.create_metric(DistanceType.CHEBYSHEV, **kwargs)
        elif name_lower in ['minkowski', 'lp']:
            return DistanceMetricFactory.create_metric(DistanceType.MINKOWSKI, **kwargs)
        else:
            raise ValueError(f"Unknown distance name: {name}")


def calculate_distance_statistics(distances: list[float]) -> dict[str, float]:
    """
    Calculate statistics for a list of distances.
    
    Parameters:
    -----------
    distances : list[float]
        List of distance values
        
    Returns:
    --------
    dict[str, float]
        Statistical measures of distances
    """
    if not distances:
        return {}
    
    distances_array = np.array(distances)
    
    return {
        'mean': float(np.mean(distances_array)),
        'median': float(np.median(distances_array)),
        'std': float(np.std(distances_array)),
        'min': float(np.min(distances_array)),
        'max': float(np.max(distances_array)),
        'q25': float(np.percentile(distances_array, 25)),
        'q75': float(np.percentile(distances_array, 75)),
        'count': len(distances)
    }