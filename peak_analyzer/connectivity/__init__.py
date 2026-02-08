"""
Peak Analyzer Connectivity Layer

Provides connectivity patterns, neighbor generation, pathfinding algorithms,
and distance metrics for N-dimensional spatial analysis.

Classes:
--------
Connectivity : Abstract base for connectivity patterns
FaceConnectivity : Face connectivity (4-conn in 2D, 6-conn in 3D)
EdgeConnectivity : Edge connectivity (8-conn in 2D, 18-conn in 3D)
VertexConnectivity : Vertex connectivity (8-conn in 2D, 26-conn in 3D)
CustomConnectivity : User-defined connectivity patterns
AdaptiveConnectivity : Connectivity that adapts to local conditions

NeighborGenerator Functions : Core neighbor computation functions
get_neighbors : Compute valid neighbor coordinates with caching
get_iterator : Iterator interface for neighbor generation

Usage:
------
    from peak_analyzer.connectivity import (
        Connectivity, get_neighbors, get_iterator
    )
    
    # Create connectivity pattern
    connectivity = Connectivity(ndim=3, k=3)  # Full connectivity
    offsets = tuple(tuple(offset) for offset in connectivity.get_neighbor_offsets())
    
    # Get neighbors for a single position
    position = (10, 15, 5)
    neighbors = get_neighbors(position, shape=(100, 100, 50), offsets=offsets)
    
    # Use iterator interface for exploration loops
    for neighbor in get_iterator(position, (100, 100, 50), offsets):
        print(f"Neighbor: {neighbor}")
"""

from .connectivity_types import Connectivity

from .neighbor_generator import (
    get_neighbors,
    get_iterator
)

__all__ = [
    # Connectivity types
    'Connectivity',
    
    # Core neighbor generation functions
    'get_neighbors',
    'get_iterator',
]

# Version info
__version__ = '1.0.0'