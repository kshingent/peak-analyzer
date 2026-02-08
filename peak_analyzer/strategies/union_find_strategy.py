"""
Union-Find Strategy

Implements peak detection using Union-Find data structure with height-priority processing.
Prevents false peaks through wave-front expansion from processed terrain.
"""

from typing import Any
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass

from .base_strategy import BaseStrategy, StrategyConfig
from ..models import Peak
from ..core.union_find import GridUnionFind
from ..connectivity.connectivity_types import get_k_connectivity


@dataclass
class HeightGraph:
    """Graph structure for height-priority processing."""
    height_levels: dict[float, list[tuple[int, ...]]]
    adjacency: dict[tuple[int, ...], list[tuple[int, ...]]]
    heights: dict[tuple[int, ...], float]


@dataclass
class Component:
    """Connected component with metadata."""
    indices: list[tuple[int, ...]]
    height: float
    is_peak: bool
    prominence: float | None = None
    prominence_base: tuple[int, ...] | None = None


class UnionFindStrategy(BaseStrategy):
    """
    Peak detection strategy using Union-Find with height-priority processing.
    
    This strategy processes terrain by height levels from highest to lowest,
    using wave-front expansion to prevent false peak generation within plateaus.
    """
    
    def __init__(self, config: StrategyConfig | None = None, **kwargs):
        """
        Initialize Union-Find strategy.
        
        Parameters:
        -----------
        config : StrategyConfig, optional
            Strategy configuration
        **kwargs
            Additional parameters
        """
        super().__init__(config, **kwargs)
        self.connectivity_structure = None
        
    def detect_peaks(self, data: np.ndarray, **params) -> list[Peak]:
        """
        Detect peaks using Union-Find with height-priority processing.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        **params
            Additional parameters
            
        Returns:
        --------
        list[Peak]
            List of detected peaks
        """
        # Validate input
        self.validate_input(data)
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Build height graph
        height_graph = self._build_height_priority_graph(processed_data)
        
        # Detect peaks and calculate prominence
        peaks_with_prominence = self.detect_peaks_with_prominence(processed_data, height_graph)
        
        # Convert to Peak objects
        peaks = []
        for component in peaks_with_prominence:
            if component.is_peak:
                peak = self.create_peak_from_indices(
                    component.indices, 
                    processed_data
                )
                peaks.append(peak)
        
        # Postprocess peaks
        filtered_peaks = self.postprocess_peaks(peaks, data)
        
        return filtered_peaks
    
    def detect_peaks_with_prominence(self, data: np.ndarray, height_graph: HeightGraph | None = None) -> list[Component]:
        """
        Detect peaks and calculate prominence simultaneously.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        height_graph : HeightGraph, optional
            Pre-built height graph
            
        Returns:
        --------
        list[Component]
            Components with peak status and prominence
        """
        if height_graph is None:
            height_graph = self._build_height_priority_graph(data)
        
        # Initialize Union-Find structure
        grid_uf = GridUnionFind(data.shape)
        
        # Track processed points and components
        processed_points = set()
        peak_components = []
        
        # Process height levels from highest to lowest
        sorted_heights = sorted(height_graph.height_levels.keys(), reverse=True)
        
        for height in sorted_heights:
            current_batch = height_graph.height_levels[height]
            
            # Apply wave-front expansion
            newly_processed = self._wave_front_expansion(
                current_batch, processed_points, data, grid_uf
            )
            
            # Handle remaining unprocessed points (potential new peaks)
            remaining_points = set(current_batch) - newly_processed
            
            if remaining_points:
                # Find connected components among remaining points
                isolated_components = self._find_isolated_components(
                    list(remaining_points), data, grid_uf
                )
                
                # Register as peak candidates
                for component_indices in isolated_components:
                    component = Component(
                        indices=component_indices,
                        height=height,
                        is_peak=True
                    )
                    peak_components.append(component)
                    
                    # Union all points in this component
                    if len(component_indices) > 1:
                        for i in range(1, len(component_indices)):
                            grid_uf.union_coords(component_indices[0], component_indices[i])
            
            # Update processed points
            processed_points.update(current_batch)
        
        # Calculate prominence for detected peaks
        self._calculate_prominence_for_components(peak_components, data)
        
        return peak_components
    
    def _build_height_priority_graph(self, data: np.ndarray) -> HeightGraph:
        """
        Build height-priority graph for processing.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        HeightGraph
            Graph structure for height-priority processing
        """
        # Group points by height
        height_levels = defaultdict(list)
        heights = {}
        
        # Create iterator over all points
        it = np.nditer(data, flags=['multi_index'])
        while not it.finished:
            height = float(it[0])
            coords = it.multi_index
            
            height_levels[height].append(coords)
            heights[coords] = height
            
            it.iternext()
        
        # Build adjacency structure
        adjacency = self._build_adjacency_structure(data)
        
        return HeightGraph(
            height_levels=dict(height_levels),
            adjacency=adjacency,
            heights=heights
        )
    
    def _build_adjacency_structure(self, data: np.ndarray) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """Build adjacency structure based on connectivity."""
        adjacency = defaultdict(list)
        
        # Get connectivity structure
        if self.connectivity_structure is None:
            connectivity = self.config.connectivity
            self.connectivity_structure = get_k_connectivity(data.ndim, connectivity)
        
        # Generate neighbors for each point
        it = np.nditer(data, flags=['multi_index'])
        while not it.finished:
            coords = it.multi_index
            neighbors = self._get_valid_neighbors(coords, data.shape)
            adjacency[coords] = neighbors
            it.iternext()
        
        return dict(adjacency)
    
    def _wave_front_expansion(
        self, 
        current_batch: list[tuple[int, ...]], 
        processed_points: set[tuple[int, ...]], 
        data: np.ndarray,
        grid_uf: GridUnionFind
    ) -> set[tuple[int, ...]]:
        """
        Apply wave-front expansion from processed points.
        
        Parameters:
        -----------
        current_batch : list[tuple]
            Points at current height level
        processed_points : set[tuple]
            Previously processed points
        data : np.ndarray
            Input data
        grid_uf : GridUnionFind
            Union-Find structure
            
        Returns:
        --------
        set[tuple]
            Newly processed points from current batch
        """
        newly_processed = set()
        
        # Iterative wave-front expansion
        while True:
            temp_store = set()
            
            # Find unprocessed points connected to processed points
            for point in current_batch:
                if point not in newly_processed:
                    neighbors = self._get_valid_neighbors(point, data.shape)
                    
                    for neighbor in neighbors:
                        if neighbor in processed_points or neighbor in newly_processed:
                            # Point connects to already processed terrain
                            temp_store.add(point)
                            
                            # Perform union operations
                            self._perform_union_operations(point, neighbor, grid_uf)
                            break
            
            # No new connections found - terminate
            if not temp_store:
                break
            
            # Add newly connected points to processed set
            newly_processed.update(temp_store)
        
        return newly_processed
    
    def _perform_union_operations(self, point: tuple[int, ...], neighbor: tuple[int, ...], grid_uf: GridUnionFind):
        """Perform appropriate union operations."""
        # Check if point already has a region
        point_root = grid_uf.find_coord(point)
        neighbor_root = grid_uf.find_coord(neighbor)
        
        if point_root == point:
            # Point-to-region union (first connection)
            grid_uf.union_coords(point, neighbor)
        else:
            # Region-to-region union (subsequent connection)
            if point_root != neighbor_root:
                grid_uf.union_coords(point_root, neighbor_root)
    
    def _find_isolated_components(self, points: list[tuple[int, ...]], data: np.ndarray, grid_uf: GridUnionFind) -> list[list[tuple[int, ...]]]:
        """Find connected components among isolated points."""
        if not points:
            return []
        
        # Create temporary Union-Find for these points only
        temp_uf = GridUnionFind(data.shape)
        
        # Union points that are neighbors and have same height
        for point in points:
            neighbors = self._get_valid_neighbors(point, data.shape)
            
            for neighbor in neighbors:
                if neighbor in points and data[point] == data[neighbor]:
                    temp_uf.union_coords(point, neighbor)
        
        # Get connected components
        components = []
        visited = set()
        
        for point in points:
            if point not in visited:
                component = temp_uf.get_component_coords(point)
                # Filter to only include points from our original list
                component = [p for p in component if p in points]
                components.append(component)
                visited.update(component)
        
        return components
    
    def _calculate_prominence_for_components(self, components: list[Component], data: np.ndarray):
        """Calculate prominence for each peak component."""
        for component in components:
            if component.is_peak:
                prominence, prominence_base = self._calculate_component_prominence(component, data)
                component.prominence = prominence
                component.prominence_base = prominence_base
    
    def _calculate_component_prominence(self, component: Component, data: np.ndarray) -> tuple[float, tuple[int, ...]]:
        """Calculate prominence for a single component."""
        peak_height = component.height
        
        # Start descent search from component boundary
        boundary_points = self._get_component_boundary(component.indices, data.shape)
        
        # Use breadth-first search to find prominence base
        queue = deque([(point, peak_height) for point in boundary_points])
        visited = set(component.indices)  # Start with component as visited
        visited.update(boundary_points)
        
        min_height = peak_height
        prominence_base = boundary_points[0] if boundary_points else component.indices[0]
        
        while queue:
            current_point, path_min_height = queue.popleft()
            current_height = data[current_point]
            
            # Update minimum height along path
            path_min_height = min(path_min_height, current_height)
            
            if path_min_height < min_height:
                min_height = path_min_height
                prominence_base = current_point
            
            # Stop if we found higher terrain
            if current_height > peak_height:
                prominence = peak_height - min_height
                return prominence, prominence_base
            
            # Explore neighbors
            neighbors = self._get_valid_neighbors(current_point, data.shape)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path_min_height))
        
        # No higher terrain found - use boundary-based calculation
        boundary_min = self._find_boundary_minimum(data)
        prominence = peak_height - boundary_min
        return prominence, prominence_base
    
    def _get_component_boundary(self, component_indices: list[tuple[int, ...]], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get boundary points of a component."""
        component_set = set(component_indices)
        boundary_points = []
        
        for point in component_indices:
            neighbors = self._get_valid_neighbors(point, shape)
            
            for neighbor in neighbors:
                if neighbor not in component_set:
                    boundary_points.append(neighbor)
        
        return list(set(boundary_points))  # Remove duplicates
    
    def _get_valid_neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbors within bounds."""
        neighbors = []
        ndim = len(point)
        
        # Generate neighbors based on face connectivity (can be extended)
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
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray) -> dict[Peak, dict[str, Any]]:
        """Calculate features for detected peaks."""
        features = {}
        
        for peak in peaks:
            peak_features = {
                'height': peak.height,
                'area': len(peak.plateau_indices),
                'coordinates': peak.center_coordinates
            }
            features[peak] = peak_features
        
        return features
    
    @classmethod
    def estimate_performance(cls, data_shape: tuple[int, ...]) -> dict[str, float]:
        """Estimate performance for Union-Find strategy."""
        data_size = np.prod(data_shape)
        
        # Union-Find operations are nearly O(n) with path compression
        estimated_time = data_size * 1e-6  # seconds, rough estimate
        
        # Memory for Union-Find structure + height graph
        estimated_memory = data_size * 16 / (1024 * 1024)  # MB
        
        # Accuracy generally high for this strategy
        accuracy_score = 0.9
        
        # Scalability is good
        scalability_factor = 0.8
        
        return {
            "estimated_time": estimated_time,
            "estimated_memory": estimated_memory,
            "accuracy_score": accuracy_score,
            "scalability_factor": scalability_factor
        }
        