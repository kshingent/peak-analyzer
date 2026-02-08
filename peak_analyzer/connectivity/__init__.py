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

NeighborGenerator : Abstract base for neighbor generation
StandardNeighborGenerator : Direct offset-based generation
VectorizedNeighborGenerator : NumPy vectorized generation
CachedNeighborGenerator : Cached neighbor generation
RegionNeighborGenerator : Specialized for region analysis

PathFinder : Abstract base for pathfinding algorithms
BreadthFirstSearchFinder : BFS pathfinding
DijkstraFinder : Dijkstra's algorithm
AStarFinder : A* algorithm with heuristics
TopographicPathFinder : Specialized topographic pathfinding
FloodFillFinder : Flood fill for region analysis
WatershedPathFinder : Watershed-based pathfinding

DistanceMetric : Abstract base for distance calculations
EuclideanDistance : Standard Euclidean distance
ManhattanDistance : L1 (cityblock) distance
ChebyshevDistance : L∞ (maximum) distance
MinkowskiDistance : General Lp distance
WeightedEuclideanDistance : Weighted Euclidean with custom weights
MahalanobisDistance : Mahalanobis distance with covariance
TopographicDistance : Distance incorporating elevation
GeodesicDistance : Distance along paths

Usage:
------
    from peak_analyzer.connectivity import (
        ConnectivityFactory, NeighborGeneratorFactory, 
        PathFinderFactory, DistanceMetricFactory
    )
    
    # Create connectivity pattern
    connectivity = ConnectivityFactory.create_connectivity(ConnectivityType.VERTEX, ndim=3)
    
    # Create neighbor generator
    neighbor_gen = NeighborGeneratorFactory.create_generator(
        connectivity, generator_type="vectorized"
    )
    
    # Get neighbors for a position
    neighbors = neighbor_gen.get_neighbors((10, 15, 5), shape=(100, 100, 50))
    
    # Create pathfinder
    pathfinder = PathFinderFactory.create_pathfinder("astar", connectivity)
    
    # Find path between positions
    path = pathfinder.find_path((0, 0, 0), (10, 10, 5), shape=(100, 100, 50))
    
    # Create distance metric
    distance_metric = DistanceMetricFactory.create_metric(
        DistanceType.MINKOWSKI, p=2.5, scale=[1.0, 1.0, 0.5]
    )
    
    # Calculate distance
    dist = distance_metric.calculate_distance((0, 0, 0), (10, 10, 5))
"""

from .connectivity_types import Connectivity

from .neighbor_generator import (
    NeighborGenerator,
    StandardNeighborGenerator,
    VectorizedNeighborGenerator,
    CachedNeighborGenerator,
    RegionNeighborGenerator,
    IterativeNeighborGenerator,
    NeighborGeneratorFactory,
    benchmark_generators
)

from .path_finder import (
    PathFinder,
    BreadthFirstSearchFinder,
    DijkstraFinder,
    AStarFinder,
    TopographicPathFinder,
    FloodFillFinder,
    WatershedPathFinder,
    PathFinderFactory
)

from .distance_metrics import (
    DistanceType,
    DistanceMetric,
    EuclideanDistance,
    ManhattanDistance,
    ChebyshevDistance,
    MinkowskiDistance,
    WeightedEuclideanDistance,
    MahalanobisDistance,
    TopographicDistance,
    GeodesicDistance,
    AdaptiveDistance,
    DistanceCalculator,
    DistanceMetricFactory,
    calculate_distance_statistics
)

__all__ = [
    # Connectivity types
    'Connectivity',
    
    # Neighbor generation
    'NeighborGenerator',
    'StandardNeighborGenerator',
    'VectorizedNeighborGenerator',
    'CachedNeighborGenerator',
    'RegionNeighborGenerator',
    'IterativeNeighborGenerator', 
    'NeighborGeneratorFactory',
    'benchmark_generators',
    
    # Pathfinding
    'PathFinder',
    'BreadthFirstSearchFinder',
    'DijkstraFinder',
    'AStarFinder',
    'TopographicPathFinder',
    'FloodFillFinder',
    'WatershedPathFinder',
    'PathFinderFactory',
    
    # Distance metrics
    'DistanceType',
    'DistanceMetric',
    'EuclideanDistance',
    'ManhattanDistance',
    'ChebyshevDistance',
    'MinkowskiDistance',
    'WeightedEuclideanDistance', 
    'MahalanobisDistance',
    'TopographicDistance',
    'GeodesicDistance',
    'AdaptiveDistance',
    'DistanceCalculator',
    'DistanceMetricFactory',
    'calculate_distance_statistics'
]

# Version info
__version__ = '1.0.0'