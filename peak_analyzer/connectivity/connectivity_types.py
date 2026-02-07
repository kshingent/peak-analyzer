"""
Connectivity Types

Defines different connectivity patterns for N-dimensional spaces including
face connectivity, edge connectivity, vertex connectivity, and custom patterns.
"""

from typing import Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class ConnectivityType(Enum):
    """
    Enumeration of standard connectivity types.
    """
    FACE = "face"  # 4-connectivity in 2D, 6-connectivity in 3D
    EDGE = "edge"  # 8-connectivity in 2D, 18-connectivity in 3D  
    VERTEX = "vertex"  # 8-connectivity in 2D, 26-connectivity in 3D
    CUSTOM = "custom"  # User-defined connectivity pattern


class ConnectivityPattern(ABC):
    """
    Abstract base class for connectivity patterns.
    
    Defines how neighboring relationships are determined in N-dimensional space.
    """
    
    def __init__(self, ndim: int, name: str = ""):
        """
        Initialize connectivity pattern.
        
        Parameters:
        -----------
        ndim : int
            Number of dimensions
        name : str, optional
            Name of the connectivity pattern
        """
        self.ndim = ndim
        self.name = name
        self._neighbor_offsets = None
        self._distance_cache = {}
    
    @abstractmethod
    def get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """
        Get offset vectors for all neighbors.
        
        Returns:
        --------
        list[tuple[int, ...]]
            List of offset tuples relative to center position
        """
        pass
    
    @abstractmethod
    def get_neighbor_count(self) -> int:
        """
        Get expected number of neighbors.
        
        Returns:
        --------
        int
            Expected number of neighbors for interior points
        """
        pass
    
    def get_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Get valid neighbors for a center position within given bounds.
        
        Parameters:
        -----------
        center : tuple of int
            Center position
        shape : tuple of int
            Shape bounds for valid coordinates
            
        Returns:
        --------
        list[tuple of int]
            List of valid neighbor coordinates
        """
        if len(center) != self.ndim:
            raise ValueError(f"Center position must have {self.ndim} dimensions")
        
        neighbors = []
        offsets = self.get_neighbor_offsets()
        
        for offset in offsets:
            neighbor = tuple(center[i] + offset[i] for i in range(self.ndim))
            
            # Check bounds
            if all(0 <= neighbor[i] < shape[i] for i in range(self.ndim)):
                neighbors.append(neighbor)
        
        return neighbors
    
    def is_neighbor(self, pos1: tuple[int, ...], pos2: tuple[int, ...]) -> bool:
        """
        Check if two positions are neighbors under this connectivity.
        
        Parameters:
        -----------
        pos1, pos2 : tuple of int
            Positions to check
            
        Returns:
        --------
        bool
            True if positions are neighbors
        """
        if len(pos1) != self.ndim or len(pos2) != self.ndim:
            return False
        
        offset = tuple(pos2[i] - pos1[i] for i in range(self.ndim))
        return offset in self.get_neighbor_offsets()
    
    def get_distance_type(self) -> str:
        """
        Get the distance type associated with this connectivity.
        
        Returns:
        --------
        str
            Distance type ('manhattan', 'euclidean', 'chebyshev', 'custom')
        """
        return "custom"


class FaceConnectivity(ConnectivityPattern):
    """
    Face connectivity (4-connectivity in 2D, 6-connectivity in 3D).
    
    Only considers neighbors that share a face (edge in 2D).
    Manhattan distance = 1.
    """
    
    def __init__(self, ndim: int):
        """Initialize face connectivity."""
        super().__init__(ndim, f"face-{ndim}d")
    
    def get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """Get face neighbor offsets."""
        if self._neighbor_offsets is None:
            offsets = []
            
            for dim in range(self.ndim):
                # Positive direction
                offset_pos = [0] * self.ndim
                offset_pos[dim] = 1
                offsets.append(tuple(offset_pos))
                
                # Negative direction  
                offset_neg = [0] * self.ndim
                offset_neg[dim] = -1
                offsets.append(tuple(offset_neg))
            
            self._neighbor_offsets = offsets
        
        return self._neighbor_offsets
    
    def get_neighbor_count(self) -> int:
        """Get face neighbor count."""
        return 2 * self.ndim
    
    def get_distance_type(self) -> str:
        """Face connectivity uses Manhattan distance."""
        return "manhattan"


class EdgeConnectivity(ConnectivityPattern):
    """
    Edge connectivity (8-connectivity in 2D, 18-connectivity in 3D).
    
    Considers neighbors that share a face or edge.
    """
    
    def __init__(self, ndim: int):
        """Initialize edge connectivity.""" 
        super().__init__(ndim, f"edge-{ndim}d")
    
    def get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """Get edge neighbor offsets."""
        if self._neighbor_offsets is None:
            offsets = []
            
            # Generate all combinations where exactly 1 or 2 dimensions are non-zero
            from itertools import product
            
            for offset in product([-1, 0, 1], repeat=self.ndim):
                if offset == tuple([0] * self.ndim):  # Skip center
                    continue
                
                # Count non-zero dimensions
                non_zero_dims = sum(1 for x in offset if x != 0)
                
                # Include if 1 or 2 dimensions are non-zero
                if non_zero_dims <= 2:
                    offsets.append(offset)
            
            self._neighbor_offsets = offsets
        
        return self._neighbor_offsets
    
    def get_neighbor_count(self) -> int:
        """Get edge neighbor count."""
        if self.ndim == 2:
            return 8
        elif self.ndim == 3:
            return 18
        else:
            # General formula: sum of binomial coefficients
            face_count = 2 * self.ndim
            edge_count = 2 * self.ndim * (self.ndim - 1)  # Approximate
            return face_count + edge_count


class VertexConnectivity(ConnectivityPattern):
    """
    Vertex connectivity (8-connectivity in 2D, 26-connectivity in 3D).
    
    Considers all neighbors in the Moore neighborhood.
    Chebyshev distance = 1.
    """
    
    def __init__(self, ndim: int):
        """Initialize vertex connectivity."""
        super().__init__(ndim, f"vertex-{ndim}d")
    
    def get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """Get vertex neighbor offsets."""
        if self._neighbor_offsets is None:
            offsets = []
            
            from itertools import product
            
            # All combinations of {-1, 0, 1}^ndim except the center
            for offset in product([-1, 0, 1], repeat=self.ndim):
                if offset != tuple([0] * self.ndim):  # Skip center
                    offsets.append(offset)
            
            self._neighbor_offsets = offsets
        
        return self._neighbor_offsets
    
    def get_neighbor_count(self) -> int:
        """Get vertex neighbor count."""
        return 3**self.ndim - 1  # All 3^n positions except center
    
    def get_distance_type(self) -> str:
        """Vertex connectivity uses Chebyshev distance."""
        return "chebyshev"


class CustomConnectivity(ConnectivityPattern):
    """
    Custom user-defined connectivity pattern.
    """
    
    def __init__(self, ndim: int, offsets: list[tuple[int, ...]], name: str = "custom"):
        """
        Initialize custom connectivity.
        
        Parameters:
        -----------
        ndim : int
            Number of dimensions
        offsets : list[tuple[int, ...]]
            Custom neighbor offset vectors
        name : str
            Name for this connectivity pattern
        """
        super().__init__(ndim, name)
        
        # Validate offsets
        for offset in offsets:
            if len(offset) != ndim:
                raise ValueError(f"All offsets must have {ndim} dimensions")
            if offset == tuple([0] * ndim):
                raise ValueError("Offsets cannot include the center position (all zeros)")
        
        self._neighbor_offsets = list(offsets)
        self.custom_offsets = offsets
    
    def get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """Get custom neighbor offsets."""
        return self._neighbor_offsets
    
    def get_neighbor_count(self) -> int:
        """Get custom neighbor count."""
        return len(self._neighbor_offsets)


class AdaptiveConnectivity(ConnectivityPattern):
    """
    Adaptive connectivity that changes based on local conditions.
    
    Can switch between different connectivity patterns based on criteria
    such as local gradient, density, or custom conditions.
    """
    
    def __init__(self, 
                 ndim: int, 
                 base_connectivity: ConnectivityPattern,
                 adaptation_function: callable | None = None):
        """
        Initialize adaptive connectivity.
        
        Parameters:
        -----------
        ndim : int
            Number of dimensions
        base_connectivity : ConnectivityPattern
            Default connectivity pattern
        adaptation_function : callable, optional
            Function that determines connectivity based on local conditions
        """
        super().__init__(ndim, f"adaptive-{base_connectivity.name}")
        self.base_connectivity = base_connectivity
        self.adaptation_function = adaptation_function
        self.connectivity_cache = {}
    
    def get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """Get base neighbor offsets."""
        return self.base_connectivity.get_neighbor_offsets()
    
    def get_neighbor_count(self) -> int:
        """Get base neighbor count.""" 
        return self.base_connectivity.get_neighbor_count()
    
    def get_adaptive_neighbors(self, center: tuple[int, ...], shape: tuple[int, ...], 
                              data: np.ndarray, **kwargs) -> list[tuple[int, ...]]:
        """
        Get neighbors using adaptive connectivity.
        
        Parameters:
        -----------
        center : tuple of int
            Center position
        shape : tuple of int
            Shape bounds
        data : np.ndarray
            Data array for adaptation criteria
        **kwargs
            Additional parameters for adaptation function
            
        Returns:
        --------
        list[tuple of int]
            Adaptive neighbor list
        """
        if self.adaptation_function is None:
            return self.get_neighbors(center, shape)
        
        # Check cache
        cache_key = (center, tuple(shape))
        if cache_key in self.connectivity_cache:
            return self.connectivity_cache[cache_key]
        
        # Apply adaptation function
        adapted_connectivity = self.adaptation_function(center, data, **kwargs)
        
        if isinstance(adapted_connectivity, ConnectivityPattern):
            neighbors = adapted_connectivity.get_neighbors(center, shape)
        else:
            # Fallback to base connectivity
            neighbors = self.base_connectivity.get_neighbors(center, shape)
        
        # Cache result
        self.connectivity_cache[cache_key] = neighbors
        
        return neighbors


class ConnectivityFactory:
    """
    Factory for creating connectivity patterns.
    """
    
    @staticmethod
    def create_connectivity(connectivity_type: ConnectivityType, ndim: int, **kwargs) -> ConnectivityPattern:
        """
        Create connectivity pattern.
        
        Parameters:
        -----------
        connectivity_type : ConnectivityType
            Type of connectivity to create
        ndim : int
            Number of dimensions
        **kwargs
            Additional parameters for custom connectivity
            
        Returns:
        --------
        ConnectivityPattern
            Connectivity pattern instance
        """
        if connectivity_type == ConnectivityType.FACE:
            return FaceConnectivity(ndim)
        elif connectivity_type == ConnectivityType.EDGE:
            return EdgeConnectivity(ndim)
        elif connectivity_type == ConnectivityType.VERTEX:
            return VertexConnectivity(ndim)
        elif connectivity_type == ConnectivityType.CUSTOM:
            offsets = kwargs.get('offsets', [])
            name = kwargs.get('name', 'custom')
            return CustomConnectivity(ndim, offsets, name)
        else:
            raise ValueError(f"Unknown connectivity type: {connectivity_type}")
    
    @staticmethod
    def create_standard_connectivity(name: str, ndim: int) -> ConnectivityPattern:
        """
        Create standard connectivity by name.
        
        Parameters:
        -----------
        name : str
            Connectivity name ('4', '8', '6', '18', '26', 'face', 'edge', 'vertex')
        ndim : int
            Number of dimensions
            
        Returns:
        --------
        ConnectivityPattern
            Connectivity pattern instance
        """
        name = name.lower()
        
        # Standard 2D connectivities
        if name in ['4', 'face'] and ndim == 2:
            return FaceConnectivity(ndim)
        elif name in ['8', 'vertex'] and ndim == 2:
            return VertexConnectivity(ndim)
        
        # Standard 3D connectivities
        elif name in ['6', 'face'] and ndim == 3:
            return FaceConnectivity(ndim)
        elif name in ['18', 'edge'] and ndim == 3:
            return EdgeConnectivity(ndim)
        elif name in ['26', 'vertex'] and ndim == 3:
            return VertexConnectivity(ndim)
        
        # General case
        elif name in ['face']:
            return FaceConnectivity(ndim)
        elif name in ['edge']:
            return EdgeConnectivity(ndim)
        elif name in ['vertex']:
            return VertexConnectivity(ndim)
        
        else:
            raise ValueError(f"Unknown connectivity name: {name} for {ndim}D")


def get_connectivity_info(connectivity: ConnectivityPattern) -> dict[str, Any]:
    """
    Get information about a connectivity pattern.
    
    Parameters:
    -----------
    connectivity : ConnectivityPattern
        Connectivity pattern to analyze
        
    Returns:
    --------
    dict[str, Any]
        Connectivity information
    """
    return {
        'name': connectivity.name,
        'ndim': connectivity.ndim,
        'neighbor_count': connectivity.get_neighbor_count(),
        'distance_type': connectivity.get_distance_type(),
        'offsets': connectivity.get_neighbor_offsets(),
        'type': type(connectivity).__name__
    }


def visualize_connectivity_2d(connectivity: ConnectivityPattern) -> str:
    """
    Create ASCII visualization of 2D connectivity pattern.
    
    Parameters:
    -----------
    connectivity : ConnectivityPattern
        2D connectivity pattern to visualize
        
    Returns:
    --------
    str
        ASCII art representation
    """
    if connectivity.ndim != 2:
        return "Visualization only available for 2D connectivity"
    
    # Create 3x3 grid
    grid = [['.' for _ in range(3)] for _ in range(3)]
    grid[1][1] = 'C'  # Center
    
    # Mark neighbors
    offsets = connectivity.get_neighbor_offsets()
    for offset in offsets:
        row = 1 + offset[0]
        col = 1 + offset[1]
        if 0 <= row < 3 and 0 <= col < 3:
            grid[row][col] = 'N'
    
    # Convert to string
    lines = []
    lines.append(f"Connectivity: {connectivity.name}")
    lines.append("Pattern (C=center, N=neighbor):")
    for row in grid:
        lines.append(' '.join(row))
    
    return '\n'.join(lines)