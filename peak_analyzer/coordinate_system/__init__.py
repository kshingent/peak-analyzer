"""
Coordinate System Layer

Provides coordinate transformation, grid management, and spatial indexing capabilities
for separating index space (ijk...) from coordinate space (xyz...).
"""

from .coordinate_mapping import (
    CoordinateMapping,
    CoordinateMappingBuilder,
    create_isotropic_mapping,
    create_anisotropic_mapping,
    create_physical_mapping
)

from .grid_manager import GridManager

from .spatial_indexing import (
    SpatialIndex,
    BruteForceIndex,
    KDTreeIndex,
    SpatialHashIndex,
    GridSpatialIndex,
    SpatialIndexFactory,
    benchmark_spatial_indexes
)

__all__ = [
    # Core coordinate mapping
    'CoordinateMapping',
    'CoordinateMappingBuilder',
    'create_isotropic_mapping',
    'create_anisotropic_mapping',
    'create_physical_mapping',
    
    # Grid management
    'GridManager',
    
    # Spatial indexing
    'SpatialIndex',
    'BruteForceIndex',
    'KDTreeIndex',
    'SpatialHashIndex',
    'GridSpatialIndex',
    'SpatialIndexFactory',
    'benchmark_spatial_indexes',
]

# Version information
__version__ = '1.0.0'