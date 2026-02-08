"""
Path Finder

Implements pathfinding algorithms for N-dimensional spaces including BFS, 
Dijkstra's algorithm, A*, and specialized algorithms for topographic analysis.
"""

from typing import Callable, Dict
import numpy as np
from collections import deque
import heapq
from abc import ABC, abstractmethod

from .connectivity_types import Connectivity
from .neighbor_generator import VectorizedNeighborGenerator


class PathFinder(ABC):
    """
    Abstract base class for pathfinding algorithms.
    """
    
    def __init__(self, connectivity: Connectivity):
        """
        Initialize path finder.
        
        Parameters:
        -----------
        connectivity : Connectivity
            Connectivity pattern for neighbor relationships
        """
        self.connectivity = connectivity
        self.neighbor_generator = VectorizedNeighborGenerator(connectivity)
    
    @abstractmethod
    def find_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                 shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """
        Find path between start and goal positions.
        
        Parameters:
        -----------
        start : tuple of int
            Start position
        goal : tuple of int  
            Goal position
        shape : tuple of int
            Array shape bounds
        **kwargs
            Algorithm-specific parameters
            
        Returns:
        --------
        list[tuple of int | None]
            Path from start to goal, or None if no path exists
        """
        pass
    
    def find_shortest_paths(self, start: tuple[int, ...], goals: list[tuple[int, ...]], 
                           shape: tuple[int, ...], **kwargs) -> dict[tuple[int, ...], list[tuple[int, ...]] | None]:
        """
        Find shortest paths to multiple goals.
        
        Parameters:
        -----------
        start : tuple of int
            Start position
        goals : list[tuple of int]
            Goal positions
        shape : tuple of int
            Array shape bounds
            
        Returns:
        --------
        dict[tuple of int, list[tuple of int | None]]
            Paths to each goal
        """
        return {goal: self.find_path(start, goal, shape, **kwargs) for goal in goals}


class BreadthFirstSearchFinder(PathFinder):
    """
    Breadth-First Search pathfinder for unweighted graphs.
    """
    
    def find_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                 shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find shortest path using BFS."""
        if start == goal:
            return [start]
        
        # Optional obstacle function
        is_obstacle = kwargs.get('is_obstacle', None)
        max_distance = kwargs.get('max_distance', float('inf'))
        
        # BFS setup
        queue = deque([(start, 0)])
        visited = {start}
        parent = {start: None}
        
        while queue:
            current, distance = queue.popleft()
            
            # Check distance limit
            if distance >= max_distance:
                continue
            
            # Get neighbors
            neighbors = self.neighbor_generator.get_neighbors(current, shape)
            
            for neighbor in neighbors:
                # Skip if already visited
                if neighbor in visited:
                    continue
                
                # Skip if obstacle
                if is_obstacle and is_obstacle(neighbor):
                    continue
                
                # Mark as visited and set parent
                visited.add(neighbor)
                parent[neighbor] = current
                
                # Check if goal reached
                if neighbor == goal:
                    return self._reconstruct_path(parent, start, goal)
                
                # Add to queue
                queue.append((neighbor, distance + 1))
        
        return None  # No path found
    
    def _reconstruct_path(self, parent: Dict, start: tuple[int, ...], goal: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Reconstruct path from parent pointers."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path


class DijkstraFinder(PathFinder):
    """
    Dijkstra's algorithm pathfinder for weighted graphs.
    """
    
    def find_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                 shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find shortest path using Dijkstra's algorithm."""
        if start == goal:
            return [start]
        
        # Weight function (default: uniform weight)
        weight_func = kwargs.get('weight_func', lambda pos: 1.0)
        is_obstacle = kwargs.get('is_obstacle', None)
        max_cost = kwargs.get('max_cost', float('inf'))
        
        # Dijkstra setup
        distances = {start: 0.0}
        parent = {start: None}
        priority_queue = [(0.0, id(start), start)]
        visited = set()
        
        while priority_queue:
            current_distance, _, current = heapq.heappop(priority_queue)
            
            # Skip if already processed
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check if goal reached  
            if current == goal:
                return self._reconstruct_path(parent, start, goal)
            
            # Check cost limit
            if current_distance > max_cost:
                continue
            
            # Process neighbors
            neighbors = self.neighbor_generator.get_neighbors(current, shape)
            
            for neighbor in neighbors:
                # Skip if obstacle or already processed
                if neighbor in visited or (is_obstacle and is_obstacle(neighbor)):
                    continue
                
                # Calculate new distance
                edge_weight = weight_func(neighbor)
                new_distance = current_distance + edge_weight
                
                # Update if better path found
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parent[neighbor] = current
                    heapq.heappush(priority_queue, (new_distance, id(neighbor), neighbor))
        
        return None  # No path found
    
    def _reconstruct_path(self, parent: Dict, start: tuple[int, ...], goal: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Reconstruct path from parent pointers."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path


class AStarFinder(PathFinder):
    """
    A* algorithm pathfinder with heuristic optimization.
    """
    
    def __init__(self, connectivity: Connectivity, heuristic_p: float = 2.0):
        """
        Initialize A* finder.
        
        Parameters:
        -----------
        connectivity : ConnectivityPattern
            Connectivity pattern
        heuristic_p : float
            Heuristic Minkowski distance parameter (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)
        """
        super().__init__(connectivity)
        self.heuristic_p = heuristic_p
    
    def find_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                 shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find shortest path using A* algorithm."""
        if start == goal:
            return [start]
        
        # Parameters
        weight_func = kwargs.get('weight_func', lambda pos: 1.0)
        is_obstacle = kwargs.get('is_obstacle', None)
        max_cost = kwargs.get('max_cost', float('inf'))
        scale = kwargs.get('scale', [1.0] * len(start))
        
        # A* setup
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal, scale)}
        parent = {start: None}
        
        open_set = [(f_score[start], id(start), start)]
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if goal reached
            if current == goal:
                return self._reconstruct_path(parent, start, goal)
            
            # Check cost limit
            if g_score[current] > max_cost:
                continue
            
            # Process neighbors
            neighbors = self.neighbor_generator.get_neighbors(current, shape)
            
            for neighbor in neighbors:
                # Skip if obstacle or already processed
                if neighbor in closed_set or (is_obstacle and is_obstacle(neighbor)):
                    continue
                
                # Calculate tentative g score
                edge_weight = weight_func(neighbor)
                tentative_g = g_score[current] + edge_weight
                
                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal, scale)
                    parent[neighbor] = current
                    
                    heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
        
        return None  # No path found
    
    def _heuristic(self, pos1: tuple[int, ...], pos2: tuple[int, ...], scale: list[float]) -> float:
        """Calculate heuristic distance between positions."""
        diff = np.array(pos2) - np.array(pos1)
        scaled_diff = np.abs(diff * scale)
        
        if np.isinf(self.heuristic_p):
            # Chebyshev distance
            return float(np.max(scaled_diff))
        else:
            return float(np.power(np.sum(np.power(scaled_diff, self.heuristic_p)), 1.0 / self.heuristic_p))
    
    def _reconstruct_path(self, parent: Dict, start: tuple[int, ...], goal: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Reconstruct path from parent pointers."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path


class TopographicPathFinder(PathFinder):
    """
    Specialized pathfinder for topographic analysis.
    """
    
    def find_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                 shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find topographically optimal path."""
        data = kwargs.get('data')
        if data is None:
            raise ValueError("Topographic pathfinder requires 'data' parameter")
        
        path_type = kwargs.get('path_type', 'steepest_descent')
        
        if path_type == 'steepest_descent':
            return self._find_steepest_descent_path(start, goal, data, shape, **kwargs)
        elif path_type == 'ridge_line':
            return self._find_ridge_path(start, goal, data, shape, **kwargs)
        elif path_type == 'contour':
            return self._find_contour_path(start, goal, data, shape, **kwargs)
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def _find_steepest_descent_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                                   data: np.ndarray, shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find path following steepest descent."""
        max_steps = kwargs.get('max_steps', 1000)
        min_gradient = kwargs.get('min_gradient', 1e-6)
        
        path = [start]
        current = start
        
        for step in range(max_steps):
            if current == goal:
                return path
            
            # Calculate gradients to neighbors
            neighbors = self.neighbor_generator.get_neighbors(current, shape)
            
            if not neighbors:
                break
            
            # Find neighbor with steepest descent
            best_neighbor = None
            best_gradient = -float('inf')
            current_height = data[current]
            
            for neighbor in neighbors:
                neighbor_height = data[neighbor]
                gradient = current_height - neighbor_height  # Descent is positive
                
                if gradient > best_gradient:
                    best_gradient = gradient
                    best_neighbor = neighbor
            
            # Stop if no significant descent
            if best_neighbor is None or best_gradient < min_gradient:
                break
            
            current = best_neighbor
            path.append(current)
        
        return path if current == goal else None
    
    def _find_ridge_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                        data: np.ndarray, shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find path along ridge lines."""
        # Use A* with ridge-preference heuristic
        def ridge_weight(pos):
            # Higher weight for lower elevations (prefer ridges)
            base_weight = 1.0
            height_penalty = 1.0 / (data[pos] + 1.0)  # Avoid division by zero
            return base_weight + height_penalty
        
        kwargs['weight_func'] = ridge_weight
        
        # Use A* algorithm
        a_star = AStarFinder(self.connectivity)
        return a_star.find_path(start, goal, shape, **kwargs)
    
    def _find_contour_path(self, start: tuple[int, ...], goal: tuple[int, ...], 
                          data: np.ndarray, shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]] | None:
        """Find path along contour lines."""
        target_height = kwargs.get('target_height', data[start])
        height_tolerance = kwargs.get('height_tolerance', 0.1)
        
        def contour_weight(pos):
            height_diff = abs(data[pos] - target_height)
            if height_diff <= height_tolerance:
                return 1.0  # On contour
            else:
                return 1.0 + height_diff  # Penalty for leaving contour
        
        kwargs['weight_func'] = contour_weight
        
        # Use Dijkstra algorithm 
        dijkstra = DijkstraFinder(self.connectivity)
        return dijkstra.find_path(start, goal, shape, **kwargs)


class FloodFillFinder:
    """
    Flood fill algorithm for region connectivity analysis.
    """
    
    def __init__(self, connectivity: Connectivity):
        """Initialize flood fill finder."""
        self.connectivity = connectivity
        self.neighbor_generator = VectorizedNeighborGenerator(connectivity)
    
    def flood_fill(self, start: tuple[int, ...], shape: tuple[int, ...], 
                  condition: Callable[[tuple[int, ...]], bool]) -> set[tuple[int, ...]]:
        """
        Perform flood fill from start position.
        
        Parameters:
        -----------
        start : tuple of int
            Starting position
        shape : tuple of int
            Array shape bounds
        condition : callable
            Function to determine if position should be included
            
        Returns:
        --------
        set[tuple of int]
            All connected positions satisfying the condition
        """
        if not condition(start):
            return set()
        
        filled = set()
        queue = deque([start])
        filled.add(start)
        
        while queue:
            current = queue.popleft()
            neighbors = self.neighbor_generator.get_neighbors(current, shape)
            
            for neighbor in neighbors:
                if neighbor not in filled and condition(neighbor):
                    filled.add(neighbor)
                    queue.append(neighbor)
        
        return filled
    
    def find_connected_components(self, shape: tuple[int, ...], 
                                 condition: Callable[[tuple[int, ...]], bool]) -> list[set[tuple[int, ...]]]:
        """
        Find all connected components satisfying condition.
        
        Parameters:
        -----------
        shape : tuple of int
            Array shape bounds
        condition : callable
            Function to determine valid positions
            
        Returns:
        --------
        list[set[tuple of int]]
            List of connected components
        """
        # Generate all possible positions
        from itertools import product
        all_positions = set(product(*[range(s) for s in shape]))
        
        # Filter by condition
        valid_positions = {pos for pos in all_positions if condition(pos)}
        
        components = []
        unvisited = valid_positions.copy()
        
        while unvisited:
            # Start flood fill from arbitrary unvisited position
            start = next(iter(unvisited))
            component = self.flood_fill(start, shape, lambda pos: pos in unvisited)
            
            components.append(component)
            unvisited -= component
        
        return components


class WatershedPathFinder:
    """
    Watershed-based pathfinder for drainage analysis.
    """
    
    def __init__(self, connectivity: Connectivity):
        """Initialize watershed pathfinder."""
        self.connectivity = connectivity
        self.neighbor_generator = VectorizedNeighborGenerator(connectivity)
    
    def find_drainage_path(self, start: tuple[int, ...], data: np.ndarray, 
                          shape: tuple[int, ...], **kwargs) -> list[tuple[int, ...]]:
        """
        Find drainage path from start position flowing downward.
        
        Parameters:
        -----------
        start : tuple of int
            Starting position
        data : np.ndarray
            Height data
        shape : tuple of int
            Array shape bounds
        **kwargs
            Additional parameters
            
        Returns:
        --------
        list[tuple of int]
            Drainage path from start to sink
        """
        max_steps = kwargs.get('max_steps', 1000)
        min_descent = kwargs.get('min_descent', 1e-6)
        
        path = [start]
        current = start
        visited = {start}
        
        for step in range(max_steps):
            neighbors = self.neighbor_generator.get_neighbors(current, shape)
            
            # Remove already visited neighbors to avoid cycles
            neighbors = [n for n in neighbors if n not in visited]
            
            if not neighbors:
                break  # No unvisited neighbors
            
            # Find lowest neighbor
            current_height = data[current]
            best_neighbor = None
            best_height = current_height
            
            for neighbor in neighbors:
                neighbor_height = data[neighbor]
                if neighbor_height < best_height:
                    best_height = neighbor_height
                    best_neighbor = neighbor
            
            # Stop if no significant descent
            if best_neighbor is None or (current_height - best_height) < min_descent:
                break
            
            current = best_neighbor
            path.append(current)
            visited.add(current)
        
        return path
    
    def find_watershed_divide(self, peak1: tuple[int, ...], peak2: tuple[int, ...], 
                             data: np.ndarray, shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Find watershed divide between two peaks.
        
        Parameters:
        -----------
        peak1, peak2 : tuple of int
            Peak positions  
        data : np.ndarray
            Height data
        shape : tuple of int
            Array shape bounds
            
        Returns:
        --------
        list[tuple of int]
            Positions along watershed divide
        """
        
        # Find approximate divide line
        # This is a simplified implementation
        # A full implementation would require more sophisticated watershed analysis
        
        # Find midpoint between peaks
        midpoint = tuple((peak1[i] + peak2[i]) // 2 for i in range(len(peak1)))
        
        # Use A* to find high-elevation path between peaks
        def elevation_weight(pos):
            return 1.0 / (data[pos] + 1.0)  # Prefer higher elevations
        
        a_star = AStarFinder(self.connectivity)
        divide_path = a_star.find_path(peak1, peak2, shape, weight_func=elevation_weight)
        
        return divide_path if divide_path else [midpoint]


class PathFinderFactory:
    """
    Factory for creating pathfinder instances.
    """
    
    @staticmethod
    def create_pathfinder(algorithm: str, connectivity: Connectivity, **kwargs) -> PathFinder:
        """
        Create pathfinder instance.
        
        Parameters:
        -----------
        algorithm : str
            Algorithm type ('bfs', 'dijkstra', 'astar', 'topographic')
        connectivity : ConnectivityPattern
            Connectivity pattern
        **kwargs
            Algorithm-specific parameters
            
        Returns:
        --------
        PathFinder
            Pathfinder instance
        """
        if algorithm == 'bfs':
            return BreadthFirstSearchFinder(connectivity)
        elif algorithm == 'dijkstra':
            return DijkstraFinder(connectivity)
        elif algorithm == 'astar':
            heuristic_p = kwargs.get('heuristic_p', 2.0)
            return AStarFinder(connectivity, heuristic_p)
        elif algorithm == 'topographic':
            return TopographicPathFinder(connectivity)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")