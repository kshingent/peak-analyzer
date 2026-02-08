"""
Virtual Peak Handler

Handles creation and management of virtual peaks for same-height connected peaks.
"""

import numpy as np
from dataclasses import dataclass

from ..models import Peak, VirtualPeak, SaddlePoint
from ..connectivity.path_finder import PathFinder


@dataclass
class PeakGroup:
    """Group of peaks at the same height that are connected."""
    peaks: list[Peak]
    common_height: float
    connection_paths: dict[tuple[int, int], list[tuple[int, ...]]]  # Paths between peak pairs
    saddle_points: list[tuple[int, ...]]


class VirtualPeakHandler:
    """
    Handles detection and creation of virtual peaks for same-height connected peaks.
    """
    
    def __init__(self, connectivity: str = 'face'):
        """
        Initialize virtual peak handler.
        
        Parameters:
        -----------
        connectivity : str
            Connectivity type for determining peak connections
        """
        self.connectivity = connectivity
        self.path_finder = PathFinder(connectivity)
        
    def detect_connected_same_height_peaks(self, peaks: list[Peak], data: np.ndarray) -> list[PeakGroup]:
        """
        Detect groups of peaks at the same height that are topographically connected.
        
        Parameters:
        -----------
        peaks : list[Peak]
            List of detected peaks
        data : np.ndarray
            Input data array
            
        Returns:
        --------
        list[PeakGroup]
            Groups of connected same-height peaks
        """
        # Group peaks by height
        height_groups = self._group_peaks_by_height(peaks)
        
        connected_groups = []
        
        # For each height level, find connected components
        for height, same_height_peaks in height_groups.items():
            if len(same_height_peaks) > 1:
                # Find connections between same-height peaks
                peak_connections = self._find_peak_connections(same_height_peaks, data, height)
                
                # Group connected peaks
                connected_components = self._find_connected_components(same_height_peaks, peak_connections)
                
                # Create PeakGroup for each connected component
                for component in connected_components:
                    if len(component) > 1:  # Only groups with multiple peaks
                        group = self._create_peak_group(component, height, peak_connections, data)
                        connected_groups.append(group)
        
        return connected_groups
    
    def create_virtual_peak(self, peak_group: PeakGroup, data: np.ndarray) -> VirtualPeak:
        """
        Create a virtual peak from a group of connected same-height peaks.
        
        Parameters:
        -----------
        peak_group : PeakGroup
            Group of connected peaks
        data : np.ndarray
            Input data array
            
        Returns:
        --------
        VirtualPeak
            Virtual peak representation
        """
        # Calculate virtual center as centroid of constituent peaks
        virtual_center = self._calculate_group_centroid(peak_group.peaks)
        
        # Calculate total area
        total_area = sum(len(peak.plateau_indices) for peak in peak_group.peaks)
        
        # Calculate virtual prominence
        virtual_prominence, virtual_prominence_base = self.calculate_virtual_prominence(
            peak_group, data
        )
        
        return VirtualPeak(
            constituent_peaks=peak_group.peaks,
            virtual_center=virtual_center,
            height=peak_group.common_height,
            virtual_prominence=virtual_prominence,
            virtual_prominence_base=virtual_prominence_base,
            total_area=total_area
        )
    
    def calculate_virtual_prominence(self, peak_group: PeakGroup, data: np.ndarray) -> tuple[float, tuple[int, ...]]:
        """
        Calculate prominence for a virtual peak representing connected same-height peaks.
        
        Parameters:
        -----------
        peak_group : PeakGroup
            Group of connected peaks
        data : np.ndarray
            Input data array
            
        Returns:
        --------
        tuple
            (virtual_prominence, virtual_prominence_base)
        """
        group_height = peak_group.common_height
        
        # Find all points that are part of the peak group (union of all plateaus)
        group_indices = set()
        for peak in peak_group.peaks:
            group_indices.update(peak.plateau_indices)
        
        # Search for higher terrain connected to the group
        higher_terrain_connection = self._find_higher_terrain_connection(
            group_indices, group_height, data
        )
        
        if higher_terrain_connection is not None:
            saddle_height, saddle_point = higher_terrain_connection
            virtual_prominence = group_height - saddle_height
            return virtual_prominence, saddle_point
        else:
            # No higher terrain found - use boundary-based calculation
            boundary_min = self._find_boundary_minimum(data)
            virtual_prominence = group_height - boundary_min
            boundary_point = self._find_boundary_minimum_location(data)
            return virtual_prominence, boundary_point
    
    def resolve_saddle_points(self, peak_group: PeakGroup, data: np.ndarray) -> list[SaddlePoint]:
        """
        Find saddle points between peaks in a group.
        
        Parameters:
        -----------
        peak_group : PeakGroup
            Group of connected peaks
        data : np.ndarray
            Input data array
            
        Returns:
        --------
        list[SaddlePoint]
            Saddle points between peaks
        """
        saddle_points = []
        peaks = peak_group.peaks
        
        # Find saddle points between each pair of peaks in the group
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                
                # Find the saddle point along the connection path
                if (i, j) in peak_group.connection_paths:
                    path = peak_group.connection_paths[(i, j)]
                    saddle = self._find_saddle_along_path(path, data)
                    
                    if saddle is not None:
                        saddle_point = SaddlePoint(
                            coordinates=saddle,
                            height=data[saddle],
                            connected_peaks=[i, j]
                        )
                        saddle_points.append(saddle_point)
        
        return saddle_points
    
    def _group_peaks_by_height(self, peaks: list[Peak]) -> dict[float, list[Peak]]:
        """Group peaks by their height values."""
        height_groups = {}
        
        for peak in peaks:
            height = peak.height
            if height not in height_groups:
                height_groups[height] = []
            height_groups[height].append(peak)
        
        return height_groups
    
    def _find_peak_connections(self, peaks: list[Peak], data: np.ndarray, height: float) -> dict[tuple[int, int], list[tuple[int, ...]]]:
        """
        Find connections between same-height peaks through terrain of equal or higher elevation.
        
        Returns dictionary mapping peak pairs to connection paths.
        """
        connections = {}
        
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                peak_i = peaks[i]
                peak_j = peaks[j]
                
                # Try to find a path between peaks through terrain >= height
                path = self.path_finder.find_minimum_height_path(
                    peak_i.center_indices,
                    peak_j.center_indices,
                    data,
                    min_height=height
                )
                
                if path is not None:
                    connections[(i, j)] = path
        
        return connections
    
    def _find_connected_components(self, peaks: list[Peak], connections: dict[tuple[int, int], list[tuple[int, ...]]]) -> list[list[Peak]]:
        """Find connected components among peaks using connection graph."""
        # Build adjacency list
        adjacency = {i: [] for i in range(len(peaks))}
        
        for (i, j) in connections:
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        for i in range(len(peaks)):
            if i not in visited:
                component = []
                self._dfs_component(i, adjacency, visited, component)
                component_peaks = [peaks[idx] for idx in component]
                components.append(component_peaks)
        
        return components
    
    def _dfs_component(self, node: int, adjacency: dict[int, list[int]], visited: set[int], component: list[int]):
        """Depth-first search to find connected component."""
        visited.add(node)
        component.append(node)
        
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                self._dfs_component(neighbor, adjacency, visited, component)
    
    def _create_peak_group(self, peaks: list[Peak], height: float, connections: dict[tuple[int, int], list[tuple[int, ...]]], data: np.ndarray) -> PeakGroup:
        """Create PeakGroup from connected peaks."""
        # Find saddle points between peaks
        saddle_points = []
        
        # Extract relevant connection paths for this group
        group_connections = {}
        
        for (i, j), path in connections.items():
            if i < len(peaks) and j < len(peaks):
                group_connections[(i, j)] = path
                
                # Find saddle point along this path
                saddle = self._find_saddle_along_path(path, data)
                if saddle is not None:
                    saddle_points.append(saddle)
        
        return PeakGroup(
            peaks=peaks,
            common_height=height,
            connection_paths=group_connections,
            saddle_points=saddle_points
        )
    
    def _calculate_group_centroid(self, peaks: list[Peak]) -> tuple[float, ...]:
        """Calculate centroid of peak group."""
        all_coordinates = []
        total_area = 0
        
        for peak in peaks:
            peak_area = len(peak.plateau_indices)
            total_area += peak_area
            
            # Weight by peak area
            for _ in range(peak_area):
                all_coordinates.append(peak.center_coordinates)
        
        if not all_coordinates:
            return peaks[0].center_coordinates
        
        # Calculate weighted centroid
        ndim = len(all_coordinates[0])
        centroid = []
        
        for dim in range(ndim):
            coord_sum = sum(coord[dim] for coord in all_coordinates)
            centroid.append(coord_sum / len(all_coordinates))
        
        return tuple(centroid)
    
    def _find_higher_terrain_connection(self, group_indices: set[tuple[int, ...]], group_height: float, data: np.ndarray) -> tuple[float, tuple[int, ...]]:
        """
        Find connection from peak group to higher terrain.
        
        Returns (saddle_height, saddle_point) if found, None otherwise.
        """
        # Start from boundary of group and search outward
        boundary_points = self._find_group_boundary(group_indices, data.shape)
        
        # Use breadth-first search to find higher terrain
        from collections import deque
        
        queue = deque()
        visited = set(group_indices)  # Start with group indices as visited
        
        # Add boundary points to queue
        for boundary_point in boundary_points:
            if data[boundary_point] <= group_height:  # Only consider points not higher than group
                queue.append((boundary_point, data[boundary_point]))
                visited.add(boundary_point)
        
        min_saddle_height = float('inf')
        best_saddle_point = None
        
        while queue:
            current_point, current_height = queue.popleft()
            
            # Check if we found higher terrain
            if current_height > group_height:
                if current_height < min_saddle_height:
                    min_saddle_height = current_height
                    best_saddle_point = current_point
                continue  # Don't explore beyond higher terrain
            
            # Explore neighbors
            neighbors = self._get_valid_neighbors(current_point, data.shape)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_height = data[neighbor]
                    
                    # Track minimum height along path
                    path_min_height = max(current_height, neighbor_height)
                    
                    if neighbor_height > group_height:
                        # Found higher terrain
                        if path_min_height < min_saddle_height:
                            min_saddle_height = path_min_height
                            best_saddle_point = neighbor
                    else:
                        queue.append((neighbor, path_min_height))
        
        if best_saddle_point is not None:
            return min_saddle_height, best_saddle_point
        else:
            return None
    
    def _find_group_boundary(self, group_indices: set[tuple[int, ...]], data_shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Find boundary points of a peak group."""
        boundary_points = []
        
        for point in group_indices:
            neighbors = self._get_valid_neighbors(point, data_shape)
            
            # If any neighbor is not in group, this is a boundary point
            for neighbor in neighbors:
                if neighbor not in group_indices:
                    boundary_points.append(neighbor)
                    break
        
        return list(set(boundary_points))  # Remove duplicates
    
    def _find_saddle_along_path(self, path: list[tuple[int, ...]], data: np.ndarray) -> tuple[int, ...]:
        """Find the saddle point (lowest point) along a path."""
        if len(path) < 2:
            return None
        
        min_height = float('inf')
        saddle_point = None
        
        for point in path:
            height = data[point]
            if height < min_height:
                min_height = height
                saddle_point = point
        
        return saddle_point
    
    def _get_valid_neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbors within data bounds."""
        neighbors = []
        ndim = len(point)
        
        # Generate neighbors based on face connectivity
        for dim in range(ndim):
            for delta in [-1, 1]:
                neighbor = list(point)
                neighbor[dim] += delta
                
                # Check bounds
                if 0 <= neighbor[dim] < shape[dim]:
                    neighbors.append(tuple(neighbor))
        
        return neighbors
    
    def _find_boundary_minimum(self, data: np.ndarray) -> float:
        """Find minimum value on data boundary."""
        min_val = float('inf')
        
        for dim in range(data.ndim):
            for side in [0, -1]:
                slices = [slice(None)] * data.ndim
                slices[dim] = side
                boundary_face = data[tuple(slices)]
                face_min = np.min(boundary_face)
                min_val = min(min_val, face_min)
        
        return min_val
    
    def _find_boundary_minimum_location(self, data: np.ndarray) -> tuple[int, ...]:
        """Find coordinates of minimum value on data boundary."""
        min_val = self._find_boundary_minimum(data)
        
        for dim in range(data.ndim):
            for side in [0, -1]:
                slices = [slice(None)] * data.ndim
                slices[dim] = side
                boundary_face = data[tuple(slices)]
                
                if np.min(boundary_face) == min_val:
                    min_location_in_face = np.unravel_index(np.argmin(boundary_face), boundary_face.shape)
                    full_coords = list(min_location_in_face)
                    if side == -1:
                        full_coords.insert(dim, data.shape[dim] - 1)
                    else:
                        full_coords.insert(dim, 0)
                    return tuple(full_coords)
        
        return tuple([0] * data.ndim)