"""
Neighbor Generator

Simplified neighbor computation for N-dimensional grids using vectorized operations.
"""

from typing import Iterator, Callable
from functools import lru_cache
import numpy as np
from collections import deque

from .connectivity_types import Connectivity


@lru_cache(maxsize=1024)
def compute_neighbors(center: tuple[int, ...], shape: tuple[int, ...], 
                     connectivity_offsets: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """
    Compute valid neighbors for a single position.
    
    Parameters:
    -----------
    center : tuple of int
        Center position coordinates
    shape : tuple of int
        Array shape bounds
    connectivity_offsets : tuple of tuple of int
        Pre-computed neighbor offsets from connectivity pattern
        
    Returns:
    --------
    tuple of tuple of int
        Valid neighbor coordinates
    """
    center_array = np.array(center)
    offsets_array = np.array(connectivity_offsets)
    
    # Apply all offsets at once
    candidates = center_array + offsets_array
    
    # Filter valid neighbors
    shape_array = np.array(shape)
    valid_mask = np.all((candidates >= 0) & (candidates < shape_array), axis=1)
    
    valid_neighbors = candidates[valid_mask]
    return tuple(tuple(neighbor) for neighbor in valid_neighbors)


def compute_neighbors_bulk(centers: list[tuple[int, ...]], shape: tuple[int, ...],
                          connectivity_offsets: tuple[tuple[int, ...], ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """
    Compute neighbors for multiple positions efficiently.
    
    Parameters:
    -----------
    centers : list of tuple of int
        List of center positions
    shape : tuple of int
        Array shape bounds
    connectivity_offsets : tuple of tuple of int
        Pre-computed neighbor offsets
        
    Returns:
    --------
    dict[tuple of int, list of tuple of int]
        Mapping from centers to their neighbors
    """
    if not centers:
        return {}
    
    centers_array = np.array(centers)  # Shape: (n_centers, ndim)
    offsets_array = np.array(connectivity_offsets)  # Shape: (n_offsets, ndim)
    
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


class NeighborExplorer:
    """
    Generator for iterative neighbor exploration patterns (BFS, DFS, radial).
    """
    
    def __init__(self, connectivity: Connectivity):
        """Initialize with connectivity pattern."""
        self.connectivity = connectivity
        self._offsets = tuple(tuple(offset) for offset in connectivity.get_neighbor_offsets())
    
    def bfs_explore(self, start: tuple[int, ...], shape: tuple[int, ...], 
                   max_distance: int = float('inf'),
                   condition: Callable | None = None) -> Iterator[tuple[tuple[int, ...], int]]:
        """
        Breadth-first exploration of neighbors.
        
        Parameters:
        -----------
        start : tuple of int
            Starting position
        shape : tuple of int
            Array shape bounds
        max_distance : int
            Maximum search distance
        condition : Callable, optional
            Filter function: condition(position) -> bool
            
        Yields:
        -------
        tuple of (position, distance)
            Positions in BFS order with distances
        """
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        
        while queue:
            current, distance = queue.popleft()
            
            if distance > max_distance:
                continue
            
            yield current, distance
            
            # Get neighbors using core function
            neighbors = compute_neighbors(current, shape, self._offsets)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    if condition is None or condition(neighbor):
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
    
    def dfs_explore(self, start: tuple[int, ...], shape: tuple[int, ...],
                   max_depth: int = float('inf'),
                   condition: Callable | None = None) -> Iterator[tuple[tuple[int, ...], int]]:
        """
        Depth-first exploration of neighbors.
        
        Parameters:
        -----------
        start : tuple of int
            Starting position
        shape : tuple of int
            Array shape bounds
        max_depth : int
            Maximum search depth
        condition : Callable, optional
            Filter function
            
        Yields:
        -------
        tuple of (position, depth)
            Positions in DFS order with depths
        """
        visited = set()
        
        def dfs_recursive(position: tuple[int, ...], depth: int):
            if depth > max_depth or position in visited:
                return
            
            if condition is not None and not condition(position):
                return
            
            visited.add(position)
            yield position, depth
            
            # Get neighbors and recurse
            neighbors = compute_neighbors(position, shape, self._offsets)
            for neighbor in neighbors:
                yield from dfs_recursive(neighbor, depth + 1)
        
        yield from dfs_recursive(start, 0)
    
    def radial_groups(self, center: tuple[int, ...], shape: tuple[int, ...],
                     radius: int) -> list[list[tuple[int, ...]]]:
        """
        Get neighbors grouped by radial distance from center.
        
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
        list of list of tuple of int
            Neighbors grouped by distance (index = distance)
        """
        distance_groups = [[] for _ in range(radius + 1)]
        
        for position, distance in self.bfs_explore(center, shape, radius):
            if distance <= radius:
                distance_groups[distance].append(position)
        
        return distance_groups


def region_neighbors(region_indices: set[tuple[int, ...]], shape: tuple[int, ...],
                    connectivity_offsets: tuple[tuple[int, ...], ...],
                    include_external: bool = False) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """
    Compute neighbors for all positions in a region.
    
    Parameters:
    -----------
    region_indices : set of tuple of int
        Indices within the region
    shape : tuple of int
        Array shape bounds
    connectivity_offsets : tuple of tuple of int
        Connectivity offsets
    include_external : bool
        Whether to include neighbors outside the region
        
    Returns:
    --------
    dict[tuple of int, list of tuple of int]
        Neighbors for each position in region
    """
    result = {}
    
    for center in region_indices:
        all_neighbors = list(compute_neighbors(center, shape, connectivity_offsets))
        
        if include_external:
            result[center] = all_neighbors
        else:
            # Only include neighbors within the region
            region_neighbors_list = [n for n in all_neighbors if n in region_indices]
            result[center] = region_neighbors_list
    
    return result


def boundary_neighbors(region_indices: set[tuple[int, ...]], shape: tuple[int, ...],
                      connectivity_offsets: tuple[tuple[int, ...], ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """
    Get external neighbors for boundary positions.
    
    Parameters:
    -----------
    region_indices : set of tuple of int
        Indices within the region  
    shape : tuple of int
        Array shape bounds
    connectivity_offsets : tuple of tuple of int
        Connectivity offsets
        
    Returns:
    --------
    dict[tuple of int, list of tuple of int]
        External neighbors for boundary positions
    """
    boundary_neighbors_dict = {}
    
    for center in region_indices:
        all_neighbors = list(compute_neighbors(center, shape, connectivity_offsets))
        external_neighbors = [n for n in all_neighbors if n not in region_indices]
        
        if external_neighbors:  # Position is on boundary
            boundary_neighbors_dict[center] = external_neighbors
    
    return boundary_neighbors_dict