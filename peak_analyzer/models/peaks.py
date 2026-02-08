"""
Peak Detection Result Models

Data structures representing detected peaks, virtual peaks, and saddle points.
These are the primary output structures of the peak detection algorithms.
"""

from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Type Aliases
# =============================================================================

IndexTuple = tuple[int, ...]          # Index space coordinates (ijk...)
CoordTuple = tuple[float, ...]        # Coordinate space coordinates (xyz...)

# =============================================================================
# Peak Detection Results
# =============================================================================

@dataclass
class Peak:
    """
    Core peak data structure.
    
    Represents a detected peak with both computational (index) and 
    user-facing (coordinate) spatial information.
    """
    # Identification
    center_indices: tuple              # Internal processing indices (i, j, k, ...)
    center_coordinates: tuple          # User-facing coordinates (x, y, z, ...)
    plateau_indices: list[tuple]       # Region in index space
    height: float                      # Peak height value
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def coordinate_dict(self) -> dict[str, float]:
        """Return coordinates as dictionary {axis_name: value}."""
        return {f"axis_{i}": coord for i, coord in enumerate(self.center_coordinates)}


@dataclass
class VirtualPeak:
    """Virtual peak representing a group of connected same-height peaks."""
    constituent_peaks: list[Peak]
    virtual_center: tuple              # Centroid of all constituent peaks
    height: float
    virtual_prominence: float
    virtual_prominence_base: tuple
    total_area: int                    # Combined area of all constituent peaks


@dataclass
class SaddlePoint:
    """Represents a saddle point between peaks or peak groups."""
    coordinates: tuple
    height: float
    connected_peaks: list[int]         # Peak IDs connected by this saddle