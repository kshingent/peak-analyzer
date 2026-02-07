"""
Spatial Indexing

Provides efficient spatial indexing and query capabilities for coordinate spaces
including KD-trees, spatial hashing, and range queries.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

from .grid_manager import GridManager


class SpatialIndex(ABC):
    """
    Abstract base class for spatial indexing structures.
    """
    
    def __init__(self, coordinates: np.ndarray, data_indices: np.ndarray | None = None):
        """
        Initialize spatial index.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinate positions (shape: [n_points, ndim])
        data_indices : np.ndarray, optional
            Associated data indices for each coordinate
        """
        self.coordinates = coordinates
        self.ndim = coordinates.shape[1] if coordinates.ndim > 1 else 1
        self.n_points = len(coordinates)
        
        if data_indices is None:
            self.data_indices = np.arange(self.n_points)
        else:
            if len(data_indices) != self.n_points:
                raise ValueError("Data indices length must match coordinates length")
            self.data_indices = data_indices
    
    @abstractmethod
    def query_radius(self, center: np.ndarray, radius: float) -> tuple[list[int], list[float]]:
        """
        Query points within radius of center.
        
        Parameters:
        -----------
        center : np.ndarray
            Center coordinate
        radius : float
            Search radius
            
        Returns:
        --------
        tuple of (indices, distances)
            Indices and distances of points within radius
        """
        pass
    
    @abstractmethod
    def query_knn(self, center: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """
        Query k nearest neighbors.
        
        Parameters:
        -----------
        center : np.ndarray
            Center coordinate
        k : int
            Number of neighbors
            
        Returns:
        --------
        tuple of (indices, distances)
            Indices and distances of k nearest neighbors
        """
        pass
    
    @abstractmethod
    def query_range(self, min_coords: np.ndarray, max_coords: np.ndarray) -> list[int]:
        """
        Query points within coordinate range.
        
        Parameters:
        -----------
        min_coords, max_coords : np.ndarray
            Coordinate bounds
            
        Returns:
        --------
        list[int]
            Indices of points within range
        """
        pass


class BruteForceIndex(SpatialIndex):
    """
    Brute force spatial index (for small datasets or reference implementation).
    """
    
    def query_radius(self, center: np.ndarray, radius: float) -> tuple[list[int], list[float]]:
        """Query points within radius using brute force."""
        # Calculate distances to all points
        distances = np.linalg.norm(self.coordinates - center, axis=1)
        
        # Find points within radius
        within_radius = distances <= radius
        indices = np.where(within_radius)[0].tolist()
        radial_distances = distances[within_radius].tolist()
        
        return indices, radial_distances
    
    def query_knn(self, center: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """Query k nearest neighbors using brute force."""
        # Calculate distances to all points
        distances = np.linalg.norm(self.coordinates - center, axis=1)
        
        # Find k nearest
        k_actual = min(k, self.n_points)
        nearest_indices = np.argpartition(distances, k_actual-1)[:k_actual]
        
        # Sort by distance
        nearest_distances = distances[nearest_indices]
        sorted_order = np.argsort(nearest_distances)
        
        indices = nearest_indices[sorted_order].tolist()
        knn_distances = nearest_distances[sorted_order].tolist()
        
        return indices, knn_distances
    
    def query_range(self, min_coords: np.ndarray, max_coords: np.ndarray) -> list[int]:
        """Query points within coordinate range using brute force."""
        within_range = np.all(
            (self.coordinates >= min_coords) & (self.coordinates <= max_coords),
            axis=1
        )
        return np.where(within_range)[0].tolist()


class KDTreeIndex(SpatialIndex):
    """
    KD-tree spatial index for efficient nearest neighbor queries.
    """
    
    def __init__(self, coordinates: np.ndarray, data_indices: np.ndarray | None = None,
                 leaf_size: int = 30):
        """
        Initialize KD-tree index.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinate positions
        data_indices : np.ndarray, optional
            Associated data indices
        leaf_size : int
            Maximum number of points in leaf nodes
        """
        super().__init__(coordinates, data_indices)
        self.leaf_size = leaf_size
        
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(coordinates, leafsize=leaf_size)
            self._has_scipy = True
        except ImportError:
            # Fallback to sklearn
            try:
                from sklearn.neighbors import KDTree
                self.tree = KDTree(coordinates, leaf_size=leaf_size)
                self._has_scipy = False
            except ImportError:
                # Fallback to brute force
                print("Warning: Neither scipy nor sklearn available, using brute force")
                self._brute_force = BruteForceIndex(coordinates, data_indices)
                self.tree = None
                self._has_scipy = None
    
    def query_radius(self, center: np.ndarray, radius: float) -> tuple[list[int], list[float]]:
        """Query points within radius using KD-tree."""
        if self.tree is None:
            return self._brute_force.query_radius(center, radius)
        
        if self._has_scipy:
            # scipy cKDTree
            indices = self.tree.query_ball_point(center, radius)
            distances = [np.linalg.norm(self.coordinates[i] - center) for i in indices]
        else:
            # sklearn KDTree
            indices, distances = self.tree.query_radius([center], r=radius, return_distance=True)
            indices = indices[0].tolist()
            distances = distances[0].tolist()
        
        return indices, distances
    
    def query_knn(self, center: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """Query k nearest neighbors using KD-tree."""
        if self.tree is None:
            return self._brute_force.query_knn(center, k)
        
        k_actual = min(k, self.n_points)
        
        if self._has_scipy:
            # scipy cKDTree
            distances, indices = self.tree.query([center], k=k_actual)
            distances = distances[0].tolist()
            indices = indices[0].tolist()
        else:
            # sklearn KDTree
            distances, indices = self.tree.query([center], k=k_actual)
            distances = distances[0].tolist()
            indices = indices[0].tolist()
        
        return indices, distances
    
    def query_range(self, min_coords: np.ndarray, max_coords: np.ndarray) -> list[int]:
        """Query points within coordinate range."""
        if self.tree is None:
            return self._brute_force.query_range(min_coords, max_coords)
        
        # For range queries, fall back to brute force as KD-tree doesn't directly support this
        within_range = np.all(
            (self.coordinates >= min_coords) & (self.coordinates <= max_coords),
            axis=1
        )
        return np.where(within_range)[0].tolist()


class SpatialHashIndex(SpatialIndex):
    """
    Spatial hash index for fast range queries in uniform distributions.
    """
    
    def __init__(self, coordinates: np.ndarray, data_indices: np.ndarray | None = None,
                 cell_size: float | None = None):
        """
        Initialize spatial hash index.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinate positions
        data_indices : np.ndarray, optional
            Associated data indices
        cell_size : float, optional
            Size of hash grid cells (auto-determined if None)
        """
        super().__init__(coordinates, data_indices)
        
        # Determine cell size if not provided
        if cell_size is None:
            # Use average nearest neighbor distance as cell size estimate
            sample_size = min(100, self.n_points)
            sample_indices = np.random.choice(self.n_points, sample_size, replace=False)
            sample_coords = coordinates[sample_indices]
            
            distances = []
            for coord in sample_coords:
                dists = np.linalg.norm(coordinates - coord, axis=1)
                dists = dists[dists > 0]  # Exclude self
                if len(dists) > 0:
                    distances.append(np.min(dists))
            
            if distances:
                cell_size = np.mean(distances) * 2.0
            else:
                cell_size = 1.0
        
        self.cell_size = cell_size
        
        # Build hash table
        self.hash_table = defaultdict(list)
        self._build_hash_table()
    
    def _build_hash_table(self):
        """Build spatial hash table."""
        for i, coord in enumerate(self.coordinates):
            cell_key = self._get_cell_key(coord)
            self.hash_table[cell_key].append(i)
    
    def _get_cell_key(self, coord: np.ndarray) -> tuple[int, ...]:
        """Get hash cell key for coordinate."""
        return tuple(int(np.floor(c / self.cell_size)) for c in coord)
    
    def _get_neighbor_cells(self, center: np.ndarray, radius: float) -> list[tuple[int, ...]]:
        """Get hash cells that could contain points within radius."""
        center_cell = self._get_cell_key(center)
        cell_radius = int(np.ceil(radius / self.cell_size))
        
        neighbor_cells = []
        
        def generate_offsets(dim, max_offset):
            if dim == 0:
                yield []
            else:
                for offset in range(-max_offset, max_offset + 1):
                    for rest in generate_offsets(dim - 1, max_offset):
                        yield [offset] + rest
        
        for offset in generate_offsets(self.ndim, cell_radius):
            cell_key = tuple(center_cell[i] + offset[i] for i in range(self.ndim))
            neighbor_cells.append(cell_key)
        
        return neighbor_cells
    
    def query_radius(self, center: np.ndarray, radius: float) -> tuple[list[int], list[float]]:
        """Query points within radius using spatial hash."""
        candidate_indices = []
        
        # Get candidate points from neighboring cells
        neighbor_cells = self._get_neighbor_cells(center, radius)
        for cell_key in neighbor_cells:
            candidate_indices.extend(self.hash_table[cell_key])
        
        # Filter candidates by exact distance
        indices = []
        distances = []
        
        for idx in candidate_indices:
            coord = self.coordinates[idx]
            distance = np.linalg.norm(coord - center)
            if distance <= radius:
                indices.append(idx)
                distances.append(distance)
        
        return indices, distances
    
    def query_knn(self, center: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """Query k nearest neighbors using spatial hash."""
        # Start with a small radius and expand until we have k neighbors
        radius = self.cell_size
        max_radius = np.linalg.norm(np.max(self.coordinates, axis=0) - np.min(self.coordinates, axis=0))
        
        while radius <= max_radius:
            indices, distances = self.query_radius(center, radius)
            
            if len(indices) >= k:
                # Sort by distance and return k nearest
                sorted_pairs = sorted(zip(distances, indices))
                sorted_distances, sorted_indices = zip(*sorted_pairs[:k])
                return list(sorted_indices), list(sorted_distances)
            
            radius *= 2.0
        
        # If we still don't have k neighbors, return all we found
        if indices:
            sorted_pairs = sorted(zip(distances, indices))
            sorted_distances, sorted_indices = zip(*sorted_pairs)
            return list(sorted_indices), list(sorted_distances)
        else:
            return [], []
    
    def query_range(self, min_coords: np.ndarray, max_coords: np.ndarray) -> list[int]:
        """Query points within coordinate range using spatial hash."""
        # Determine cells that intersect with the range
        min_cell = self._get_cell_key(min_coords)
        max_cell = self._get_cell_key(max_coords)
        
        candidate_indices = []
        
        # Iterate through all cells in the range
        def iterate_cells(dim, current_cell):
            if dim == self.ndim:
                candidate_indices.extend(self.hash_table[tuple(current_cell)])
            else:
                for cell_idx in range(min_cell[dim], max_cell[dim] + 1):
                    current_cell[dim] = cell_idx
                    iterate_cells(dim + 1, current_cell)
        
        iterate_cells(0, [0] * self.ndim)
        
        # Filter candidates by exact range
        indices = []
        for idx in candidate_indices:
            coord = self.coordinates[idx]
            if np.all((coord >= min_coords) & (coord <= max_coords)):
                indices.append(idx)
        
        return indices


class GridSpatialIndex:
    """
    Spatial index specifically designed for regular grids.
    """
    
    def __init__(self, grid_manager: GridManager):
        """
        Initialize grid spatial index.
        
        Parameters:
        -----------
        grid_manager : GridManager
            Grid manager for coordinate transformations
        """
        self.grid_manager = grid_manager
        self.shape = grid_manager.shape
        self.ndim = grid_manager.ndim
    
    def query_radius_indices(self, center_index: tuple[int, ...], radius_indices: int) -> list[tuple[int, ...]]:
        """
        Query indices within radius in index space.
        
        Parameters:
        -----------
        center_index : tuple of int
            Center index position
        radius_indices : int
            Search radius in index units
            
        Returns:
        --------
        list[tuple of int]
            Indices within radius
        """
        # Generate candidate indices within bounding box
        candidates = []
        
        for dim in range(self.ndim):
            min_idx = max(0, center_index[dim] - radius_indices)
            max_idx = min(self.shape[dim], center_index[dim] + radius_indices + 1)
            candidates.append(range(min_idx, max_idx))
        
        # Filter by actual distance
        from itertools import product
        result = []
        
        for indices in product(*candidates):
            distance = np.linalg.norm(np.array(indices) - np.array(center_index))
            if distance <= radius_indices:
                result.append(indices)
        
        return result
    
    def query_radius_coordinates(self, center_coords: tuple[float, ...], radius: float) -> list[tuple[int, ...]]:
        """
        Query indices within radius in coordinate space.
        
        Parameters:
        -----------
        center_coords : tuple of float
            Center coordinate position
        radius : float
            Search radius in coordinate units
            
        Returns:
        --------
        list[tuple of int]
            Indices within radius
        """
        # Convert to index space
        center_index = self.grid_manager.coordinate_to_index(center_coords)
        
        # Convert radius to index space (use minimum scale factor)
        min_scale = min(self.grid_manager.mapping.scale)
        radius_indices = int(np.ceil(radius / min_scale))
        
        # Query in index space
        candidate_indices = self.query_radius_indices(center_index, radius_indices)
        
        # Filter by actual coordinate distance
        result = []
        center_array = np.array(center_coords)
        
        for indices in candidate_indices:
            coord = self.grid_manager.index_to_coordinate(indices)
            distance = np.linalg.norm(np.array(coord) - center_array)
            if distance <= radius:
                result.append(indices)
        
        return result
    
    def query_range(self, min_coords: tuple[float, ...], max_coords: tuple[float, ...]) -> list[tuple[int, ...]]:
        """
        Query indices within coordinate range.
        
        Parameters:
        -----------
        min_coords, max_coords : tuple of float
            Coordinate bounds
            
        Returns:
        --------
        list[tuple of int]
            Indices within range
        """
        # Convert to index bounds
        min_indices, max_indices = self.grid_manager.mapping.get_index_bounds(min_coords, max_coords)
        
        # Clip to grid bounds
        min_indices = self.grid_manager.clip_indices(min_indices)
        max_indices = self.grid_manager.clip_indices(max_indices)
        
        # Generate all indices in range
        from itertools import product
        ranges = [range(min_indices[i], max_indices[i] + 1) for i in range(self.ndim)]
        
        return list(product(*ranges))


class SpatialIndexFactory:
    """
    Factory for creating spatial indexes.
    """
    
    @staticmethod
    def create_index(index_type: str, coordinates: np.ndarray, **kwargs) -> SpatialIndex:
        """
        Create spatial index.
        
        Parameters:
        -----------
        index_type : str
            Type of index ('brute_force', 'kdtree', 'spatial_hash')
        coordinates : np.ndarray
            Coordinate positions
        **kwargs
            Index-specific parameters
            
        Returns:
        --------
        SpatialIndex
            Spatial index instance
        """
        data_indices = kwargs.get('data_indices', None)
        
        if index_type == 'brute_force':
            return BruteForceIndex(coordinates, data_indices)
        
        elif index_type == 'kdtree':
            leaf_size = kwargs.get('leaf_size', 30)
            return KDTreeIndex(coordinates, data_indices, leaf_size)
        
        elif index_type == 'spatial_hash':
            cell_size = kwargs.get('cell_size', None)
            return SpatialHashIndex(coordinates, data_indices, cell_size)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    @staticmethod
    def create_best_index(coordinates: np.ndarray, **kwargs) -> SpatialIndex:
        """
        Create the best spatial index for given data characteristics.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinate positions
        **kwargs
            Index parameters
            
        Returns:
        --------
        SpatialIndex
            Optimal spatial index instance
        """
        n_points = len(coordinates)
        ndim = coordinates.shape[1] if coordinates.ndim > 1 else 1
        
        # Choose index type based on data characteristics
        if n_points < 1000:
            # Small datasets: brute force is fine
            return SpatialIndexFactory.create_index('brute_force', coordinates, **kwargs)
        
        elif ndim <= 3 and n_points < 100000:
            # Medium datasets in low dimensions: KD-tree is optimal
            return SpatialIndexFactory.create_index('kdtree', coordinates, **kwargs)
        
        else:
            # Large datasets or high dimensions: spatial hash
            return SpatialIndexFactory.create_index('spatial_hash', coordinates, **kwargs)


def benchmark_spatial_indexes(coordinates: np.ndarray, 
                             query_points: np.ndarray,
                             query_radius: float) -> dict[str, dict[str, float]]:
    """
    Benchmark different spatial index types.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Database coordinates
    query_points : np.ndarray
        Query coordinate points
    query_radius : float
        Query radius
        
    Returns:
    --------
    dict[str, dict[str, float]]
        Timing results for each index type
    """
    import time
    
    index_types = ['brute_force', 'kdtree', 'spatial_hash']
    results = {}
    
    for index_type in index_types:
        try:
            # Create index
            start_time = time.time()
            index = SpatialIndexFactory.create_index(index_type, coordinates)
            build_time = time.time() - start_time
            
            # Perform queries
            start_time = time.time()
            for query_point in query_points:
                index.query_radius(query_point, query_radius)
            query_time = time.time() - start_time
            
            results[index_type] = {
                'build_time': build_time,
                'query_time': query_time,
                'total_time': build_time + query_time,
                'avg_query_time': query_time / len(query_points)
            }
            
        except Exception as e:
            print(f"Error benchmarking {index_type}: {e}")
            results[index_type] = {'error': str(e)}
    
    return results