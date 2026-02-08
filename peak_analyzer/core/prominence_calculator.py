"""
Prominence Calculation Core

Implements prominence calculation following strict geomorphological definitions
extended to multidimensional spaces.
"""

import numpy as np
from dataclasses import dataclass
import heapq

from peak_analyzer.models import Peak
from peak_analyzer.connectivity.path_finder import PathFinder


@dataclass
class ProminenceResult:
    """Result of prominence calculation for a single peak."""
    peak: Peak
    prominence: float
    prominence_base: tuple[int, ...]  # Coordinates of prominence base
    descent_path: list[tuple[int, ...]]  # Path from peak to prominence base
    saddle_height: float


@dataclass
class Path:
    """Represents a path through the terrain."""
    points: list[tuple[int, ...]]
    heights: list[float]
    total_distance: float
    min_height: float
    max_height: float


class ProminenceCalculator:
    """
    Calculates prominence using height-priority terrain exploration.
    """
    
    def __init__(self, connectivity: int = 1, boundary_condition=None):
        """
        Initialize prominence calculator.
        
        Parameters:
        -----------
        connectivity : str or int
            Connectivity type for path finding
        boundary_condition : BoundaryCondition, optional
            Boundary condition for prominence calculation
        """
        self.connectivity = connectivity
        self.boundary_condition = boundary_condition
        self.path_finder = PathFinder(connectivity)
        
    def calculate_prominence(self, peak: Peak, data: np.ndarray, wlen: float | None = None) -> float:
        """
        Calculate prominence for a single peak.
        
        Parameters:
        -----------
        peak : Peak
            Peak to calculate prominence for
        data : np.ndarray
            Input data array
        wlen : float, optional
            Window length constraint for prominence calculation
            
        Returns:
        --------
        float
            Prominence value
        """
        result = self.calculate_prominence_detailed(peak, data, wlen)
        return result.prominence
    
    def calculate_prominence_detailed(self, peak: Peak, data: np.ndarray, wlen: float | None = None) -> ProminenceResult:
        """
        Calculate prominence with detailed path information.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Input data array
        wlen : float, optional
            Window length constraint
            
        Returns:
        --------
        ProminenceResult
            Detailed prominence calculation result
        """
        peak_height = peak.height
        peak_index = peak.center_indices
        
        # Check if this is the global maximum
        if self._is_global_maximum(peak, data):
            return self._handle_global_maximum(peak, data)
        
        # Find prominence base through descent path search
        prominence_base, descent_path, saddle_height = self.find_prominence_base(
            peak_index, data, wlen
        )
        
        prominence = peak_height - saddle_height
        
        return ProminenceResult(
            peak=peak,
            prominence=prominence,
            prominence_base=prominence_base,
            descent_path=descent_path,
            saddle_height=saddle_height
        )
    
    def find_prominence_base(self, start_point: tuple[int, ...], data: np.ndarray, wlen: float | None = None) -> tuple[tuple[int, ...], list[tuple[int, ...]], float]:
        """
        Find prominence base using height-priority search.
        
        Parameters:
        -----------
        start_point : tuple
            Starting point coordinates (peak location)
        data : np.ndarray
            Input data array
        wlen : float, optional
            Maximum search distance from start point
            
        Returns:
        --------
        tuple
            (prominence_base_coordinates, descent_path, saddle_height)
        """
        peak_height = data[start_point]
        
        # Initialize priority queue with (negative_height, coordinates, path)
        # Use negative height to make it a max-heap (highest elevation first)
        pq = [(-peak_height, start_point, [start_point])]
        visited = set()
        visited.add(start_point)
        
        # Track minimum height encountered during search
        min_height = peak_height
        min_height_point = start_point
        best_path = [start_point]
        
        while pq:
            neg_height, current_point, path = heapq.heappop(pq)
            current_height = -neg_height
            
            # Check window length constraint
            if wlen is not None:
                distance_from_start = self._calculate_distance(start_point, current_point)
                if distance_from_start > wlen:
                    continue
            
            # Update minimum height info
            if current_height < min_height:
                min_height = current_height
                min_height_point = current_point
                best_path = path.copy()
            
            # Check if we found higher terrain (prominence base found)
            if current_height > peak_height and current_point != start_point:
                # Found higher terrain - prominence base is last minimum
                return min_height_point, best_path, min_height
            
            # Explore neighbors
            neighbors = self._get_valid_neighbors(current_point, data.shape)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_height = data[neighbor]
                    new_path = path + [neighbor]
                    
                    heapq.heappush(pq, (-neighbor_height, neighbor, new_path))
        
        # If we reach here, no higher terrain was found (shouldn't happen for non-global maxima)
        return min_height_point, best_path, min_height
    
    def trace_descent_path(self, start_point: tuple[int, ...], data: np.ndarray, target_height: float | None = None) -> Path:
        """
        Trace steepest descent path from starting point.
        
        Parameters:
        -----------
        start_point : tuple
            Starting coordinates
        data : np.ndarray
            Input data array
        target_height : float, optional
            Stop when reaching this height
            
        Returns:
        --------
        Path
            Descent path with detailed information
        """
        path_points = [start_point]
        path_heights = [data[start_point]]
        total_distance = 0.0
        current_point = start_point
        
        max_iterations = np.prod(data.shape)  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            current_height = data[current_point]
            
            # Check target height condition
            if target_height is not None and current_height <= target_height:
                break
            
            # Find steepest descent neighbor
            neighbors = self._get_valid_neighbors(current_point, data.shape)
            
            if not neighbors:
                break  # No neighbors (shouldn't happen in interior)
            
            # Find neighbor with steepest descent
            best_neighbor = None
            max_descent = 0.0
            
            for neighbor in neighbors:
                neighbor_height = data[neighbor]
                height_diff = current_height - neighbor_height
                
                if height_diff > max_descent:
                    max_descent = height_diff
                    best_neighbor = neighbor
            
            # If no descent found, stop
            if best_neighbor is None or max_descent <= 0:
                break
            
            # Move to best neighbor
            step_distance = self._calculate_distance(current_point, best_neighbor)
            total_distance += step_distance
            current_point = best_neighbor
            
            path_points.append(current_point)
            path_heights.append(data[current_point])
        
        return Path(
            points=path_points,
            heights=path_heights,
            total_distance=total_distance,
            min_height=min(path_heights),
            max_height=max(path_heights)
        )
    
    def handle_boundary_cases(self, peak: Peak, data: np.ndarray) -> float:
        """
        Handle prominence calculation for peaks near data boundaries.
        
        Parameters:
        -----------
        peak : Peak
            Peak near boundary
        data : np.ndarray
            Input data array
            
        Returns:
        --------
        float
            Prominence value considering boundary effects
        """
        peak_height = peak.height
        
        # Find minimum height along data boundaries
        boundary_min = self._find_boundary_minimum(data)
        
        # For boundary peaks, prominence is difference to boundary minimum
        return peak_height - boundary_min
    
    def _is_global_maximum(self, peak: Peak, data: np.ndarray) -> bool:
        """Check if peak is the global maximum in the data."""
        peak_height = peak.height
        global_max = np.max(data)
        return np.isclose(peak_height, global_max)
    
    def _handle_global_maximum(self, peak: Peak, data: np.ndarray) -> ProminenceResult:
        """Handle prominence calculation for global maximum."""
        peak_height = peak.height
        boundary_min = self._find_boundary_minimum(data)
        
        prominence = peak_height - boundary_min
        
        # For global maximum, prominence base is at data boundary
        prominence_base = self._find_boundary_minimum_location(data)
        descent_path = [peak.center_indices, prominence_base]
        
        return ProminenceResult(
            peak=peak,
            prominence=prominence,
            prominence_base=prominence_base,
            descent_path=descent_path,
            saddle_height=boundary_min
        )
    
    def _find_boundary_minimum(self, data: np.ndarray) -> float:
        """Find boundary value based on boundary condition."""
        if self.boundary_condition is None:
            raise ValueError("Boundary condition must be specified for prominence calculation")
            
        # 境界条件に基づく境界値
        boundary_type = getattr(self.boundary_condition, 'boundary_type', None)
        if boundary_type is None:
            # boundary_conditionが文字列の場合
            boundary_type = self.boundary_condition
            
        if boundary_type == 'infinite_height':
            return float('inf')
        elif boundary_type == 'infinite_depth':
            return float('-inf')  
        else:
            raise ValueError(f"Invalid boundary type: {boundary_type}. "
                           "Only 'infinite_height' and 'infinite_depth' are supported.")
    
    def _get_actual_boundary_minimum(self, data: np.ndarray) -> float:
        """Get actual minimum value from data boundary."""
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
        
        # Search boundary faces for minimum location
        for dim in range(data.ndim):
            for side in [0, -1]:
                slices = [slice(None)] * data.ndim
                slices[dim] = side
                boundary_face = data[tuple(slices)]
                
                if np.min(boundary_face) == min_val:
                    # Find exact location within this face
                    min_location_in_face = np.unravel_index(np.argmin(boundary_face), boundary_face.shape)
                    
                    # Convert to full coordinates
                    full_coords = list(min_location_in_face)
                    if side == -1:
                        full_coords.insert(dim, data.shape[dim] - 1)
                    else:
                        full_coords.insert(dim, 0)
                    
                    return tuple(full_coords)
        
        # Fallback - return corner 
        return tuple([0] * data.ndim)
    
    def _get_valid_neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbors within data bounds."""
        neighbors = []
        ndim = len(point)
        
        # Generate neighbors based on connectivity
        # This is simplified - should use connectivity structure
        for dim in range(ndim):
            for delta in [-1, 1]:
                neighbor = list(point)
                neighbor[dim] += delta
                
                # Check bounds
                if 0 <= neighbor[dim] < shape[dim]:
                    neighbors.append(tuple(neighbor))
                    
        return neighbors
    
    def _calculate_distance(self, point1: tuple[int, ...], point2: tuple[int, ...]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))