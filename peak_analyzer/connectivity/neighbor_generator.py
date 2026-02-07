"""
Neighbor Generator

Efficient algorithms for generating neighbor indices in N-dimensional spaces
with support for various connectivity patterns and boundary conditions.
"""

from typing import Any, Iterator
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

from .connectivity_types import ConnectivityPattern


class NeighborGenerator(ABC):
    """
    Abstract base class for neighbor generation algorithms.
    """
    
    def __init__(self, connectivity: ConnectivityPattern):
        """
        Initialize neighbor generator.
        
        Parameters:
        -----------
        connectivity : ConnectivityPattern
            Connectivity pattern to use
        """
        self.connectivity = connectivity
        self.ndim = connectivity.ndim
    
    @abstractmethod
    def get_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Get neighbors for a single position.
        
        Parameters:
        -----------
        center : tuple of int
            Center position
        shape : tuple of int
            Array shape bounds
            
        Returns:
        --------
        list[tuple of int]
            Valid neighbor coordinates
        """
        pass
    
    @abstractmethod
    def get_neighbors_bulk(self, centers: list[tuple[int, ...]], shape: tuple[int, ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """
        Get neighbors for multiple positions efficiently.
        
        Parameters:
        -----------
        centers : list[tuple of int]
            List of center positions
        shape : tuple of int
            Array shape bounds
            
        Returns:
        --------
        dict[tuple of int, list[tuple of int]]
            Mapping from centers to their neighbors
        """
        pass


class StandardNeighborGenerator(NeighborGenerator):
    """
    Standard implementation using direct offset application.
    """
    
    def __init__(self, connectivity: ConnectivityPattern):
        """Initialize standard neighbor generator."""
        super().__init__(connectivity)
        self._offset_cache = {}
    
    def get_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get neighbors using standard offset method."""
        return self.connectivity.get_neighbors(center, shape)
    
    def get_neighbors_bulk(self, centers: list[tuple[int, ...]], shape: tuple[int, ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """Get neighbors for multiple centers."""
        result = {}
        
        for center in centers:
            result[center] = self.get_neighbors(center, shape)
        
        return result


class VectorizedNeighborGenerator(NeighborGenerator):
    """
    Vectorized implementation using NumPy operations for efficiency.
    """
    
    def __init__(self, connectivity: ConnectivityPattern):
        """Initialize vectorized neighbor generator."""
        super().__init__(connectivity)
        self._offsets_array = np.array(connectivity.get_neighbor_offsets())
    
    def get_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get neighbors using vectorized operations."""
        center_array = np.array(center)
        
        # Apply all offsets at once
        candidates = center_array + self._offsets_array
        
        # Filter valid neighbors
        shape_array = np.array(shape)
        valid_mask = np.all((candidates >= 0) & (candidates < shape_array), axis=1)
        
        valid_neighbors = candidates[valid_mask]
        
        return [tuple(neighbor) for neighbor in valid_neighbors]
    
    def get_neighbors_bulk(self, centers: list[tuple[int, ...]], shape: tuple[int, ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """Vectorized bulk neighbor generation."""
        if not centers:
            return {}
        
        centers_array = np.array(centers)  # Shape: (n_centers, ndim)
        offsets_array = self._offsets_array  # Shape: (n_offsets, ndim)
        
        # Broadcasting: (n_centers, 1, ndim) + (1, n_offsets, ndim)
        all_candidates = centers_array[:, np.newaxis, :] + offsets_array[np.newaxis, :, :]
        
        # Shape bounds
        shape_array = np.array(shape)
        
        # Valid mask for all candidates
        valid_mask = np.all((all_candidates >= 0) & (all_candidates < shape_array), axis=2)
        
        # Build result dictionary
        result = {}
        for i, center in enumerate(centers):
            neighbors = all_candidates[i][valid_mask[i]]
            result[center] = [tuple(neighbor) for neighbor in neighbors]
        
        return result


class CachedNeighborGenerator(NeighborGenerator):
    """
    Cached implementation that stores previously computed neighbors.
    """
    
    def __init__(self, connectivity: ConnectivityPattern, cache_size: int = 10000):
        """
        Initialize cached neighbor generator.
        
        Parameters:
        -----------
        connectivity : ConnectivityPattern
            Connectivity pattern
        cache_size : int
            Maximum number of cached entries
        """
        super().__init__(connectivity)
        self.cache_size = cache_size
        self._neighbor_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._base_generator = VectorizedNeighborGenerator(connectivity)
    
    def get_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get neighbors with caching."""
        cache_key = (center, shape)
        
        if cache_key in self._neighbor_cache:
            self._cache_hits += 1
            return self._neighbor_cache[cache_key]
        
        # Compute neighbors
        neighbors = self._base_generator.get_neighbors(center, shape)
        
        # Cache result
        self._cache_neighbor_result(cache_key, neighbors)
        self._cache_misses += 1
        
        return neighbors
    
    def get_neighbors_bulk(self, centers: list[tuple[int, ...]], shape: tuple[int, ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """Bulk generation with caching."""
        result = {}
        uncached_centers = []
        
        # Check cache first
        for center in centers:
            cache_key = (center, shape)
            if cache_key in self._neighbor_cache:
                result[center] = self._neighbor_cache[cache_key]
                self._cache_hits += 1
            else:
                uncached_centers.append(center)
        
        # Compute uncached neighbors
        if uncached_centers:
            uncached_result = self._base_generator.get_neighbors_bulk(uncached_centers, shape)
            
            for center, neighbors in uncached_result.items():
                cache_key = (center, shape)
                self._cache_neighbor_result(cache_key, neighbors)
                result[center] = neighbors
                self._cache_misses += 1
        
        return result
    
    def _cache_neighbor_result(self, cache_key: tuple, neighbors: list[tuple[int, ...]]):
        """Cache neighbor result with size management."""
        self._neighbor_cache[cache_key] = neighbors
        
        # Manage cache size (simple LRU approximation)
        if len(self._neighbor_cache) > self.cache_size:
            # Remove oldest entries
            num_to_remove = len(self._neighbor_cache) - self.cache_size + 100
            keys_to_remove = list(self._neighbor_cache.keys())[:num_to_remove]
            
            for key in keys_to_remove:
                del self._neighbor_cache[key]
    
    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._neighbor_cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """Clear the neighbor cache."""
        self._neighbor_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class RegionNeighborGenerator:
    """
    Specialized generator for computing neighbors within specific regions.
    """
    
    def __init__(self, connectivity: ConnectivityPattern):
        """Initialize region neighbor generator."""
        self.connectivity = connectivity
        self.ndim = connectivity.ndim
        self._base_generator = VectorizedNeighborGenerator(connectivity)
    
    def get_region_neighbors(self, 
                           region_indices: set[tuple[int, ...]], 
                           shape: tuple[int, ...],
                           include_external: bool = False) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """
        Get neighbors for all positions in a region.
        
        Parameters:
        -----------
        region_indices : set[tuple of int]
            Indices within the region
        shape : tuple of int
            Array shape bounds
        include_external : bool
            Whether to include neighbors outside the region
            
        Returns:
        --------
        dict[tuple of int, list[tuple of int]]
            Neighbors for each position in region
        """
        result = {}
        
        for center in region_indices:
            all_neighbors = self._base_generator.get_neighbors(center, shape)
            
            if include_external:
                # Include all valid neighbors
                result[center] = all_neighbors
            else:
                # Only include neighbors within the region
                region_neighbors = [n for n in all_neighbors if n in region_indices]
                result[center] = region_neighbors
        
        return result
    
    def get_boundary_neighbors(self, 
                              region_indices: set[tuple[int, ...]], 
                              shape: tuple[int, ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """
        Get external neighbors for boundary positions.
        
        Parameters:
        -----------
        region_indices : set[tuple of int]
            Indices within the region  
        shape : tuple of int
            Array shape bounds
            
        Returns:
        --------
        dict[tuple of int, list[tuple of int]]
            External neighbors for boundary positions
        """
        boundary_neighbors = {}
        
        for center in region_indices:
            all_neighbors = self._base_generator.get_neighbors(center, shape)
            external_neighbors = [n for n in all_neighbors if n not in region_indices]
            
            if external_neighbors:  # Position is on boundary
                boundary_neighbors[center] = external_neighbors
        
        return boundary_neighbors


class IterativeNeighborGenerator:
    """
    Generator for iterative neighbor exploration (BFS, DFS).
    """
    
    def __init__(self, connectivity: ConnectivityPattern):
        """Initialize iterative neighbor generator."""
        self.connectivity = connectivity
        self.ndim = connectivity.ndim
        self._base_generator = VectorizedNeighborGenerator(connectivity)
    
    def bfs_neighbors(self, start: tuple[int, ...], shape: tuple[int, ...], 
                     max_distance: int = float('inf'),
                     condition: callable | None = None) -> Iterator[tuple[tuple[int, ...], int]]:
        """
        Generate neighbors using breadth-first search.
        
        Parameters:
        -----------
        start : tuple of int
            Starting position
        shape : tuple of int
            Array shape bounds
        max_distance : int
            Maximum search distance
        condition : callable, optional
            Function to filter positions: condition(position) -> bool
            
        Yields:
        -------
        tuple of (position, distance)
            Positions in BFS order with their distances
        """
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        
        while queue:
            current, distance = queue.popleft()
            
            # Check distance limit
            if distance > max_distance:
                continue
            
            yield current, distance
            
            # Get neighbors
            neighbors = self._base_generator.get_neighbors(current, shape)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Apply condition if provided
                    if condition is None or condition(neighbor):
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
    
    def dfs_neighbors(self, start: tuple[int, ...], shape: tuple[int, ...],
                     max_depth: int = float('inf'),
                     condition: callable | None = None) -> Iterator[tuple[tuple[int, ...], int]]:
        """
        Generate neighbors using depth-first search.
        
        Parameters:
        -----------
        start : tuple of int
            Starting position
        shape : tuple of int
            Array shape bounds
        max_depth : int
            Maximum search depth
        condition : callable, optional
            Function to filter positions
            
        Yields:
        -------
        tuple of (position, depth)
            Positions in DFS order with their depths
        """
        visited = set()
        
        def dfs_recursive(position: tuple[int, ...], depth: int):
            if depth > max_depth or position in visited:
                return
            
            # Apply condition if provided
            if condition is not None and not condition(position):
                return
            
            visited.add(position)
            yield position, depth
            
            # Recursively visit neighbors
            neighbors = self._base_generator.get_neighbors(position, shape)
            for neighbor in neighbors:
                yield from dfs_recursive(neighbor, depth + 1)
        
        yield from dfs_recursive(start, 0)
    
    def radial_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...],
                        radius: int) -> list[list[tuple[int, ...]]]:
        """
        Get neighbors organized by radial distance from center.
        
        Parameters:
        -----------
        center : tuple of int
            Center position
        shape : tuple of int
            Array shape bounds  
        radius : int
            Maximum radius to consider
            
        Returns:
        --------
        list[list[tuple of int]]
            Neighbors grouped by distance (index = distance)
        """
        distance_groups = [[] for _ in range(radius + 1)]
        
        for position, distance in self.bfs_neighbors(center, shape, radius):
            if distance <= radius:
                distance_groups[distance].append(position)
        
        return distance_groups


class NeighborGeneratorFactory:
    """
    Factory for creating neighbor generators.
    """
    
    @staticmethod
    def create_generator(connectivity: ConnectivityPattern, 
                        generator_type: str = "vectorized",
                        **kwargs) -> NeighborGenerator:
        """
        Create a neighbor generator.
        
        Parameters:
        -----------
        connectivity : ConnectivityPattern
            Connectivity pattern to use
        generator_type : str
            Type of generator ('standard', 'vectorized', 'cached')
        **kwargs
            Additional parameters for specific generators
            
        Returns:
        --------
        NeighborGenerator
            Generator instance
        """
        if generator_type == "standard":
            return StandardNeighborGenerator(connectivity)
        elif generator_type == "vectorized":
            return VectorizedNeighborGenerator(connectivity)
        elif generator_type == "cached":
            cache_size = kwargs.get('cache_size', 10000)
            return CachedNeighborGenerator(connectivity, cache_size)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")


def benchmark_generators(connectivity: ConnectivityPattern, 
                        positions: list[tuple[int, ...]], 
                        shape: tuple[int, ...]) -> dict[str, dict[str, float]]:
    """
    Benchmark different neighbor generators.
    
    Parameters:
    -----------
    connectivity : ConnectivityPattern
        Connectivity pattern to test
    positions : list[tuple of int]
        Test positions
    shape : tuple of int
        Array shape bounds
        
    Returns:
    --------
    dict[str, dict[str, float]]
        Timing results for each generator
    """
    import time
    
    generators = {
        'standard': StandardNeighborGenerator(connectivity),
        'vectorized': VectorizedNeighborGenerator(connectivity),
        'cached': CachedNeighborGenerator(connectivity)
    }
    
    results = {}
    
    for name, generator in generators.items():
        # Single neighbor timing
        start_time = time.time()
        for pos in positions[:100]:  # Sample for single calls
            generator.get_neighbors(pos, shape)
        single_time = time.time() - start_time
        
        # Bulk neighbor timing
        start_time = time.time()
        generator.get_neighbors_bulk(positions, shape)
        bulk_time = time.time() - start_time
        
        results[name] = {
            'single_time': single_time,
            'bulk_time': bulk_time,
            'single_per_call': single_time / min(100, len(positions)),
            'bulk_per_position': bulk_time / len(positions) if positions else 0.0
        }
    
    return results