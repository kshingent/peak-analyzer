"""
Boundary Layer

Provides boundary handling, edge detection, and domain analysis capabilities
for peak analysis including boundary conditions and constraint enforcement.
"""

from .boundary_conditions import (
    BoundaryType,
    BoundaryCondition,
    NoBoundary,
    PeriodicBoundary,
    MirrorBoundary,
    ConstantBoundary,
    NearestBoundary,
    ZeroBoundary,
    LinearExtrapolateBoundary,
    BoundaryManager,
    PaddedArrayView,
    create_boundary_condition,
    analyze_boundary_effects
)

from .edge_detection import (
    EdgeMethod,
    EdgeDetector,
    BoundaryExtractor,
    detect_data_boundaries
)

__all__ = [
    # Boundary conditions
    'BoundaryType',
    'BoundaryCondition',
    'NoBoundary',
    'PeriodicBoundary',
    'MirrorBoundary',
    'ConstantBoundary',
    'NearestBoundary',
    'ZeroBoundary',
    'LinearExtrapolateBoundary',
    'BoundaryManager',
    'PaddedArrayView',
    'create_boundary_condition',
    'analyze_boundary_effects',
    
    # Edge detection
    'EdgeMethod',
    'EdgeDetector',
    'BoundaryExtractor',
    'detect_data_boundaries',
]

# Version information
__version__ = '1.0.0'