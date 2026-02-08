"""
Distance Feature Calculator

Calculates distance-based properties of peaks including nearest neighbor distances,
density measures, spatial distribution patterns, and clustering characteristics.
"""

from typing import Any
import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from .base_calculator import BaseCalculator
from ..models import Peak
from ..connectivity.distance_metrics import MinkowskiDistance


class DistanceCalculator(BaseCalculator):
    """
    Calculator for distance-based features of peaks.
    
    Analyzes spatial relationships, clustering patterns, and distribution
    characteristics of peaks in the dataset.
    """
    
    def __init__(self, scale: list[float | None] = None, distance_metric: str = 'euclidean'):
        """
        Initialize distance calculator.
        
        Parameters:
        -----------
        scale : list of float, optional
            Physical scale for each dimension
        distance_metric : str
            Distance metric to use ('euclidean', 'manhattan', 'minkowski')
        """
        super().__init__(scale)
        self.distance_metric = distance_metric
        self._distance_calculator = MinkowskiDistance()
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray, **kwargs) -> dict[Peak, dict[str, Any]]:
        """
        Calculate distance features for peaks.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to calculate features for
        data : np.ndarray
            Original data array
        **kwargs
            Additional parameters including:
            - k_neighbors: Number of neighbors to analyze
            - cluster_threshold: Threshold for clustering analysis
            
        Returns:
        --------
        dict[Peak, dict[str, Any]]
            Distance features for each peak
        """
        self.validate_inputs(peaks, data)
        
        if len(peaks) < 2:
            # Not enough peaks for distance analysis
            return {peak: self._empty_distance_features() for peak in peaks}
        
        k_neighbors = kwargs.get('k_neighbors', min(5, len(peaks) - 1))
        cluster_threshold = kwargs.get('cluster_threshold', 10.0)
        
        # Calculate all pairwise distances once
        peak_positions = [self.calculate_centroid(peak.plateau_indices) for peak in peaks]
        distance_matrix = self._calculate_distance_matrix(peak_positions, data.ndim)
        
        # Perform clustering analysis
        cluster_info = self._analyze_clustering(distance_matrix, cluster_threshold)
        
        # Calculate spatial statistics
        spatial_stats = self._calculate_spatial_statistics(peak_positions, data.shape)
        
        features = {}
        
        for i, peak in enumerate(peaks):
            peak_features = {}
            
            # Nearest neighbors analysis
            peak_features['nearest_neighbors'] = self.calculate_nearest_neighbors(
                i, distance_matrix, k_neighbors
            )
            
            # Local density analysis
            peak_features['local_density'] = self.calculate_local_density(
                i, peak_positions, distance_matrix
            )
            
            # Clustering features
            peak_features['clustering'] = self.extract_clustering_features(
                i, cluster_info, distance_matrix
            )
            
            # Spatial distribution features
            peak_features['spatial_distribution'] = self.calculate_spatial_distribution(
                i, peak_positions, spatial_stats
            )
            
            # Isolation and connectivity
            peak_features['isolation_measures'] = self.calculate_isolation_measures(
                i, peak, peaks, distance_matrix
            )
            
            # Distance to boundary
            peak_features['boundary_distance'] = self.calculate_boundary_distance(
                peak, data.shape
            )
            
            features[peak] = peak_features
        
        return features
    
    def get_available_features(self) -> list[str]:
        """Get list of available distance features."""
        return [
            'nearest_neighbors',
            'local_density',
            'clustering',
            'spatial_distribution',
            'isolation_measures',
            'boundary_distance'
        ]
    
    def calculate_nearest_neighbors(self, peak_index: int, distance_matrix: np.ndarray, k: int) -> dict[str, Any]:
        """
        Calculate nearest neighbor statistics for a peak.
        
        Parameters:
        -----------
        peak_index : int
            Index of peak in the list
        distance_matrix : np.ndarray
            Precomputed distance matrix
        k : int
            Number of nearest neighbors to analyze
            
        Returns:
        --------
        dict[str, Any]
            Nearest neighbor statistics
        """
        # Get distances to all other peaks
        distances = distance_matrix[peak_index]
        distances = distances[distances > 0]  # Exclude self-distance
        
        if len(distances) == 0:
            return self._empty_neighbor_features()
        
        # Sort distances
        sorted_distances = np.sort(distances)
        k_actual = min(k, len(sorted_distances))
        
        # Statistics for k nearest neighbors
        knn_distances = sorted_distances[:k_actual]
        
        return {
            'nearest_distance': float(sorted_distances[0]),
            'mean_knn_distance': float(np.mean(knn_distances)),
            'std_knn_distance': float(np.std(knn_distances)),
            'max_knn_distance': float(np.max(knn_distances)),
            'knn_distances': knn_distances.tolist(),
            'k_used': k_actual
        }
    
    def calculate_local_density(self, peak_index: int, positions: list[tuple], distance_matrix: np.ndarray) -> dict[str, float]:
        """
        Calculate local density measures around a peak.
        
        Parameters:
        -----------
        peak_index : int
            Index of peak in the list
        positions : list[Tuple]
            Peak positions
        distance_matrix : np.ndarray
            Precomputed distance matrix
            
        Returns:
        --------
        dict[str, float]
            Local density measures
        """
        # Define density calculation radii
        radii = [5.0, 10.0, 20.0, 50.0]
        densities = {}
        
        for radius in radii:
            # Count peaks within radius
            distances = distance_matrix[peak_index]
            peaks_in_radius = np.sum(distances <= radius) - 1  # Exclude self
            
            # Calculate density (peaks per unit area/volume)
            if len(positions[0]) == 2:
                area = np.pi * radius**2
                density = peaks_in_radius / area
            elif len(positions[0]) == 3:
                volume = (4/3) * np.pi * radius**3
                density = peaks_in_radius / volume
            else:
                # Higher dimensions - generalized hypersphere
                # Simplified calculation using radius^ndim
                ndim = len(positions[0])
                hypervolume = radius**ndim
                density = peaks_in_radius / hypervolume
            
            densities[f'density_r{radius}'] = density
        
        # Calculate adaptive density using nearest neighbor distance
        distances = distance_matrix[peak_index]
        distances = distances[distances > 0]
        
        if len(distances) > 0:
            adaptive_radius = np.mean(distances)
            peaks_adaptive = np.sum(distances <= adaptive_radius)
            
            if len(positions[0]) == 2:
                adaptive_area = np.pi * adaptive_radius**2
            elif len(positions[0]) == 3:
                adaptive_area = (4/3) * np.pi * adaptive_radius**3
            else:
                adaptive_area = adaptive_radius**len(positions[0])
            
            densities['adaptive_density'] = peaks_adaptive / adaptive_area if adaptive_area > 0 else 0.0
        else:
            densities['adaptive_density'] = 0.0
        
        return densities
    
    def extract_clustering_features(self, peak_index: int, cluster_info: dict, distance_matrix: np.ndarray) -> dict[str, Any]:
        """
        Extract clustering-related features for a peak.
        
        Parameters:
        -----------
        peak_index : int
            Index of peak in the list
        cluster_info : Dict
            Clustering analysis results
        distance_matrix : np.ndarray
            Distance matrix
            
        Returns:
        --------
        dict[str, Any]
            Clustering features
        """
        features = {}
        
        # Cluster membership
        if 'cluster_labels' in cluster_info:
            peak_cluster = cluster_info['cluster_labels'][peak_index]
            features['cluster_id'] = int(peak_cluster)
            
            # Cluster size
            cluster_size = np.sum(cluster_info['cluster_labels'] == peak_cluster)
            features['cluster_size'] = int(cluster_size)
            
            # Intra-cluster distances
            same_cluster_indices = np.where(cluster_info['cluster_labels'] == peak_cluster)[0]
            if len(same_cluster_indices) > 1:
                intra_distances = distance_matrix[peak_index, same_cluster_indices]
                intra_distances = intra_distances[intra_distances > 0]  # Exclude self
                
                if len(intra_distances) > 0:
                    features['mean_intra_cluster_distance'] = float(np.mean(intra_distances))
                    features['max_intra_cluster_distance'] = float(np.max(intra_distances))
                else:
                    features['mean_intra_cluster_distance'] = 0.0
                    features['max_intra_cluster_distance'] = 0.0
            else:
                features['mean_intra_cluster_distance'] = 0.0
                features['max_intra_cluster_distance'] = 0.0
            
            # Inter-cluster distances
            other_cluster_indices = np.where(cluster_info['cluster_labels'] != peak_cluster)[0]
            if len(other_cluster_indices) > 0:
                inter_distances = distance_matrix[peak_index, other_cluster_indices]
                features['mean_inter_cluster_distance'] = float(np.mean(inter_distances))
                features['min_inter_cluster_distance'] = float(np.min(inter_distances))
            else:
                features['mean_inter_cluster_distance'] = 0.0
                features['min_inter_cluster_distance'] = 0.0
        
        else:
            features = {
                'cluster_id': 0,
                'cluster_size': 1,
                'mean_intra_cluster_distance': 0.0,
                'max_intra_cluster_distance': 0.0,
                'mean_inter_cluster_distance': 0.0,
                'min_inter_cluster_distance': 0.0
            }
        
        return features
    
    def calculate_spatial_distribution(self, peak_index: int, positions: list[tuple], spatial_stats: dict) -> dict[str, float]:
        """
        Calculate spatial distribution characteristics.
        
        Parameters:
        -----------
        peak_index : int
            Index of peak
        positions : list[Tuple]
            Peak positions
        spatial_stats : Dict
            Global spatial statistics
            
        Returns:
        --------
        dict[str, float]
            Spatial distribution features
        """
        peak_pos = np.array(positions[peak_index])
        
        # Distance from center of data
        data_center = spatial_stats['center']
        distance_from_center = np.linalg.norm(peak_pos - data_center)
        
        # Normalized distance from center
        max_distance_from_center = spatial_stats['max_distance_from_center']
        normalized_distance_from_center = distance_from_center / max_distance_from_center if max_distance_from_center > 0 else 0.0
        
        # Position relative to data bounds
        data_ranges = spatial_stats['ranges']
        relative_position = []
        for i, (pos_val, data_range) in enumerate(zip(peak_pos, data_ranges)):
            if data_range > 0:
                relative_pos = pos_val / data_range
            else:
                relative_pos = 0.0
            relative_position.append(relative_pos)
        
        # Deviation from uniform distribution
        expected_uniform_pos = np.array([0.5] * len(peak_pos))  # Center of normalized space
        uniformity_deviation = np.linalg.norm(np.array(relative_position) - expected_uniform_pos)
        
        return {
            'distance_from_center': distance_from_center,
            'normalized_distance_from_center': normalized_distance_from_center,
            'relative_position': relative_position,
            'uniformity_deviation': uniformity_deviation
        }
    
    def calculate_isolation_measures(self, peak_index: int, peak: Peak, all_peaks: list[Peak], distance_matrix: np.ndarray) -> dict[str, float]:
        """
        Calculate various isolation measures.
        
        Parameters:
        -----------
        peak_index : int
            Index of peak
        peak : Peak
            The peak object
        all_peaks : list[Peak]
            All peaks in dataset
        distance_matrix : np.ndarray
            Distance matrix
            
        Returns:
        --------
        dict[str, float]
            Isolation measures
        """
        # Topographic isolation (distance to higher peak)
        peak_height = peak.height
        distances_to_higher = []
        
        for i, other_peak in enumerate(all_peaks):
            if i != peak_index and other_peak.height >= peak_height:
                distances_to_higher.append(distance_matrix[peak_index, i])
        
        if distances_to_higher:
            topographic_isolation = min(distances_to_higher)
        else:
            # No higher peaks - use maximum distance
            distances = distance_matrix[peak_index]
            distances = distances[distances > 0]
            topographic_isolation = np.max(distances) if len(distances) > 0 else 0.0
        
        # General isolation (distance to nearest peak regardless of height)
        distances = distance_matrix[peak_index]
        distances = distances[distances > 0]
        general_isolation = np.min(distances) if len(distances) > 0 else 0.0
        
        # Prominence-weighted isolation
        prominence_weights = []
        for other_peak in all_peaks:
            # Simple prominence approximation
            prominence = other_peak.height
            prominence_weights.append(prominence)
        
        if len(prominence_weights) > 0:
            max_prominence = max(prominence_weights)
            if max_prominence > 0:
                weighted_distances = []
                for i, other_peak in enumerate(all_peaks):
                    if i != peak_index:
                        weight = prominence_weights[i] / max_prominence
                        weighted_distance = distance_matrix[peak_index, i] / (weight + 0.1)  # Avoid division by zero
                        weighted_distances.append(weighted_distance)
                
                prominence_isolation = min(weighted_distances) if weighted_distances else 0.0
            else:
                prominence_isolation = general_isolation
        else:
            prominence_isolation = general_isolation
        
        return {
            'topographic_isolation': topographic_isolation,
            'general_isolation': general_isolation,
            'prominence_isolation': prominence_isolation
        }
    
    def calculate_boundary_distance(self, peak: Peak, data_shape: tuple[int, ...]) -> dict[str, float]:
        """
        Calculate distances to data boundaries.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data_shape : tuple[int, ...]
            Shape of data array
            
        Returns:
        --------
        dict[str, float]
            Boundary distance measures
        """
        peak_center = self.calculate_centroid(peak.plateau_indices)
        
        # Distance to each boundary face
        boundary_distances = []
        
        for dim in range(len(data_shape)):
            # Distance to lower boundary
            lower_distance = peak_center[dim]
            # Distance to upper boundary
            upper_distance = data_shape[dim] - 1 - peak_center[dim]
            
            boundary_distances.extend([lower_distance, upper_distance])
        
        # Calculate statistics
        min_boundary_distance = min(boundary_distances)
        max_boundary_distance = max(boundary_distances)
        mean_boundary_distance = np.mean(boundary_distances)
        
        # Normalized distances (relative to data size)
        normalized_distances = []
        for dim in range(len(data_shape)):
            size = data_shape[dim]
            lower_norm = peak_center[dim] / size if size > 0 else 0.0
            upper_norm = (size - 1 - peak_center[dim]) / size if size > 0 else 0.0
            normalized_distances.extend([lower_norm, upper_norm])
        
        min_normalized_distance = min(normalized_distances)
        
        return {
            'min_boundary_distance': min_boundary_distance,
            'max_boundary_distance': max_boundary_distance,
            'mean_boundary_distance': mean_boundary_distance,
            'min_normalized_boundary_distance': min_normalized_distance
        }
    
    # Helper methods
    
    def _calculate_distance_matrix(self, positions: list[tuple], ndim: int) -> np.ndarray:
        """Calculate distance matrix between all peak positions."""
        positions_array = np.array(positions)
        scale = self.get_effective_scale(ndim)
        
        # Scale positions
        scaled_positions = positions_array * scale
        
        # Calculate pairwise distances
        if self.distance_metric == 'euclidean':
            distance_matrix = cdist(scaled_positions, scaled_positions, 'euclidean')
        elif self.distance_metric == 'manhattan':
            distance_matrix = cdist(scaled_positions, scaled_positions, 'cityblock')
        else:
            # Default to euclidean
            distance_matrix = cdist(scaled_positions, scaled_positions, 'euclidean')
        
        return distance_matrix
    
    def _analyze_clustering(self, distance_matrix: np.ndarray, threshold: float) -> dict[str, Any]:
        """Perform hierarchical clustering analysis."""
        try:
            # Convert distance matrix to condensed form for linkage
            condensed_distances = squareform(distance_matrix)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Get cluster labels
            cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
            
            # Calculate clustering statistics
            num_clusters = len(np.unique(cluster_labels))
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(1, num_clusters + 1)]
            
            return {
                'cluster_labels': cluster_labels,
                'linkage_matrix': linkage_matrix,
                'num_clusters': num_clusters,
                'cluster_sizes': cluster_sizes
            }
            
        except Exception:
            # Fallback: each peak is its own cluster
            n_peaks = distance_matrix.shape[0]
            return {
                'cluster_labels': np.arange(1, n_peaks + 1),
                'num_clusters': n_peaks,
                'cluster_sizes': [1] * n_peaks
            }
    
    def _calculate_spatial_statistics(self, positions: list[tuple], data_shape: tuple[int, ...]) -> dict[str, Any]:
        """Calculate global spatial statistics."""
        positions_array = np.array(positions)
        
        # Data bounds
        data_center = np.array(data_shape) / 2
        data_ranges = np.array(data_shape)
        
        # Peak distribution statistics
        peak_center = np.mean(positions_array, axis=0)
        peak_std = np.std(positions_array, axis=0)
        
        # Maximum distance from data center
        distances_from_center = [np.linalg.norm(pos - data_center) for pos in positions_array]
        max_distance_from_center = max(distances_from_center) if distances_from_center else 0.0
        
        return {
            'center': data_center,
            'ranges': data_ranges,
            'peak_center': peak_center,
            'peak_std': peak_std,
            'max_distance_from_center': max_distance_from_center
        }
    
    def _empty_distance_features(self) -> dict[str, Any]:
        """Return empty distance features structure."""
        return {
            'nearest_neighbors': self._empty_neighbor_features(),
            'local_density': {},
            'clustering': {},
            'spatial_distribution': {},
            'isolation_measures': {},
            'boundary_distance': {}
        }
    
    def _empty_neighbor_features(self) -> dict[str, Any]:
        """Return empty nearest neighbor features."""
        return {
            'nearest_distance': 0.0,
            'mean_knn_distance': 0.0,
            'std_knn_distance': 0.0,
            'max_knn_distance': 0.0,
            'knn_distances': [],
            'k_used': 0
        }