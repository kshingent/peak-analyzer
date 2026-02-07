"""
Coordinate Mapping

Defines data structures and utilities for mapping between index space (ijk...)
and coordinate space (xyz...) with support for anisotropic scaling and offsets.
"""

from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CoordinateMapping:
    """
    Immutable mapping between index and coordinate spaces.
    
    Attributes:
    -----------
    scale : tuple of float
        Scaling factors for each dimension (index -> coordinate)
    origin : tuple of float
        Origin offset in coordinate space
    ndim : int
        Number of dimensions
    axis_labels : tuple of str
        Labels for each axis ('x', 'y', 'z', etc.)
    units : tuple of str
        Physical units for each dimension
    """
    scale: tuple[float, ...]
    origin: tuple[float, ...] = ()
    axis_labels: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    
    def __post_init__(self):
        """Validate and normalize mapping parameters."""
        # Ensure all scales are positive
        if any(s <= 0 for s in self.scale):
            raise ValueError("All scale factors must be positive")
        
        # Set default origin if not provided
        if not self.origin:
            object.__setattr__(self, 'origin', tuple(0.0 for _ in self.scale))
        
        # Validate dimensions match
        ndim = len(self.scale)
        if len(self.origin) != ndim:
            raise ValueError(f"Origin dimensions ({len(self.origin)}) must match scale dimensions ({ndim})")
        
        # Set default axis labels
        if not self.axis_labels:
            default_labels = ['x', 'y', 'z', 'w', 'v', 'u']
            labels = default_labels[:ndim] if ndim <= len(default_labels) else [f'dim_{i}' for i in range(ndim)]
            object.__setattr__(self, 'axis_labels', tuple(labels))
        elif len(self.axis_labels) != ndim:
            raise ValueError(f"Axis labels length ({len(self.axis_labels)}) must match dimensions ({ndim})")
        
        # Set default units
        if not self.units:
            object.__setattr__(self, 'units', tuple('unit' for _ in range(ndim)))
        elif len(self.units) != ndim:
            raise ValueError(f"Units length ({len(self.units)}) must match dimensions ({ndim})")
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.scale)
    
    def index_to_coordinate(self, indices: tuple[int, ...] | np.ndarray) -> tuple[float, ...] | np.ndarray:
        """
        Convert indices to coordinates.
        
        Parameters:
        -----------
        indices : tuple of int or np.ndarray
            Index position(s)
            
        Returns:
        --------
        tuple of float or np.ndarray
            Coordinate position(s)
        """
        if isinstance(indices, tuple):
            if len(indices) != self.ndim:
                raise ValueError(f"Index dimensions ({len(indices)}) must match mapping dimensions ({self.ndim})")
            
            coordinates = tuple(
                self.origin[i] + indices[i] * self.scale[i]
                for i in range(self.ndim)
            )
            return coordinates
        
        elif isinstance(indices, np.ndarray):
            if indices.shape[-1] != self.ndim:
                raise ValueError(f"Last dimension of indices array ({indices.shape[-1]}) must match mapping dimensions ({self.ndim})")
            
            # Broadcasting: indices * scale + origin
            scale_array = np.array(self.scale)
            origin_array = np.array(self.origin)
            
            coordinates = indices * scale_array + origin_array
            return coordinates
        
        else:
            raise TypeError("Indices must be tuple or numpy array")
    
    def coordinate_to_index(self, coordinates: tuple[float, ...] | np.ndarray, 
                           round_mode: str = 'nearest') -> tuple[int, ...] | np.ndarray:
        """
        Convert coordinates to indices.
        
        Parameters:
        -----------
        coordinates : tuple of float or np.ndarray
            Coordinate position(s)
        round_mode : str
            Rounding mode ('nearest', 'floor', 'ceil')
            
        Returns:
        --------
        tuple of int or np.ndarray
            Index position(s)
        """
        if isinstance(coordinates, tuple):
            if len(coordinates) != self.ndim:
                raise ValueError(f"Coordinate dimensions ({len(coordinates)}) must match mapping dimensions ({self.ndim})")
            
            # Convert to index space
            float_indices = tuple(
                (coordinates[i] - self.origin[i]) / self.scale[i]
                for i in range(self.ndim)
            )
            
            # Apply rounding
            if round_mode == 'nearest':
                indices = tuple(int(round(idx)) for idx in float_indices)
            elif round_mode == 'floor':
                indices = tuple(int(np.floor(idx)) for idx in float_indices)
            elif round_mode == 'ceil':
                indices = tuple(int(np.ceil(idx)) for idx in float_indices)
            else:
                raise ValueError(f"Unknown round_mode: {round_mode}")
            
            return indices
        
        elif isinstance(coordinates, np.ndarray):
            if coordinates.shape[-1] != self.ndim:
                raise ValueError(f"Last dimension of coordinates array ({coordinates.shape[-1]}) must match mapping dimensions ({self.ndim})")
            
            # Convert to index space
            scale_array = np.array(self.scale)
            origin_array = np.array(self.origin)
            
            float_indices = (coordinates - origin_array) / scale_array
            
            # Apply rounding
            if round_mode == 'nearest':
                indices = np.round(float_indices).astype(int)
            elif round_mode == 'floor':
                indices = np.floor(float_indices).astype(int)
            elif round_mode == 'ceil':
                indices = np.ceil(float_indices).astype(int)
            else:
                raise ValueError(f"Unknown round_mode: {round_mode}")
            
            return indices
        
        else:
            raise TypeError("Coordinates must be tuple or numpy array")
    
    def get_coordinate_bounds(self, shape: tuple[int, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """
        Get coordinate bounds for given index shape.
        
        Parameters:
        -----------
        shape : tuple of int
            Shape of index space
            
        Returns:
        --------
        tuple of (min_coords, max_coords)
            Coordinate bounds
        """
        if len(shape) != self.ndim:
            raise ValueError(f"Shape dimensions ({len(shape)}) must match mapping dimensions ({self.ndim})")
        
        # Minimum coordinates (index 0,0,...)
        min_coords = self.index_to_coordinate(tuple(0 for _ in range(self.ndim)))
        
        # Maximum coordinates (index shape-1)
        max_indices = tuple(s - 1 for s in shape)
        max_coords = self.index_to_coordinate(max_indices)
        
        return min_coords, max_coords
    
    def get_index_bounds(self, min_coords: tuple[float, ...], max_coords: tuple[float, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Get index bounds for given coordinate bounds.
        
        Parameters:
        -----------
        min_coords, max_coords : tuple of float
            Coordinate bounds
            
        Returns:
        --------
        tuple of (min_indices, max_indices)
            Index bounds
        """
        min_indices = self.coordinate_to_index(min_coords, round_mode='floor')
        max_indices = self.coordinate_to_index(max_coords, round_mode='ceil')
        
        return min_indices, max_indices
    
    def calculate_distance(self, coord1: tuple[float, ...], coord2: tuple[float, ...]) -> float:
        """
        Calculate Euclidean distance between coordinates.
        
        Parameters:
        -----------
        coord1, coord2 : tuple of float
            Coordinate positions
            
        Returns:
        --------
        float
            Euclidean distance
        """
        if len(coord1) != self.ndim or len(coord2) != self.ndim:
            raise ValueError("Coordinate dimensions must match mapping dimensions")
        
        diff_squared = sum((coord2[i] - coord1[i])**2 for i in range(self.ndim))
        return np.sqrt(diff_squared)
    
    def get_voxel_volume(self) -> float:
        """
        Get volume of a single voxel in coordinate space.
        
        Returns:
        --------
        float
            Voxel volume
        """
        return np.prod(self.scale)
    
    def get_axis_info(self, axis: int | str) -> dict[str, Any]:
        """
        Get information about a specific axis.
        
        Parameters:
        -----------
        axis : int or str
            Axis index or label
            
        Returns:
        --------
        dict[str, Any]
            Axis information
        """
        if isinstance(axis, str):
            if axis not in self.axis_labels:
                raise ValueError(f"Axis label '{axis}' not found in {self.axis_labels}")
            axis_idx = self.axis_labels.index(axis)
        else:
            axis_idx = axis
            if not (0 <= axis_idx < self.ndim):
                raise ValueError(f"Axis index {axis_idx} out of range [0, {self.ndim})")
        
        return {
            'index': axis_idx,
            'label': self.axis_labels[axis_idx],
            'scale': self.scale[axis_idx],
            'origin': self.origin[axis_idx],
            'unit': self.units[axis_idx]
        }
    
    def with_modified_scale(self, scale: tuple[float, ...]) -> 'CoordinateMapping':
        """
        Create new mapping with modified scale.
        
        Parameters:
        -----------
        scale : tuple of float
            New scale factors
            
        Returns:
        --------
        CoordinateMapping
            New mapping with modified scale
        """
        return CoordinateMapping(
            scale=scale,
            origin=self.origin,
            axis_labels=self.axis_labels,
            units=self.units
        )
    
    def with_modified_origin(self, origin: tuple[float, ...]) -> 'CoordinateMapping':
        """
        Create new mapping with modified origin.
        
        Parameters:
        -----------
        origin : tuple of float
            New origin offset
            
        Returns:
        --------
        CoordinateMapping
            New mapping with modified origin
        """
        return CoordinateMapping(
            scale=self.scale,
            origin=origin,
            axis_labels=self.axis_labels,
            units=self.units
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert mapping to dictionary representation."""
        return {
            'scale': list(self.scale),
            'origin': list(self.origin),
            'axis_labels': list(self.axis_labels),
            'units': list(self.units),
            'ndim': self.ndim
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CoordinateMapping':
        """
        Create mapping from dictionary representation.
        
        Parameters:
        -----------
        data : dict[str, Any]
            Dictionary containing mapping data
            
        Returns:
        --------
        CoordinateMapping
            Reconstructed mapping
        """
        return cls(
            scale=tuple(data['scale']),
            origin=tuple(data.get('origin', [])),
            axis_labels=tuple(data.get('axis_labels', [])),
            units=tuple(data.get('units', []))
        )
    
    def __str__(self) -> str:
        """String representation of coordinate mapping."""
        lines = [f"CoordinateMapping ({self.ndim}D):"]
        
        for i in range(self.ndim):
            lines.append(f"  {self.axis_labels[i]}: scale={self.scale[i]:.3f}, origin={self.origin[i]:.3f}, unit='{self.units[i]}'")
        
        return '\n'.join(lines)


class CoordinateMappingBuilder:
    """
    Builder for constructing coordinate mappings with validation.
    """
    
    def __init__(self, ndim: int):
        """
        Initialize mapping builder.
        
        Parameters:
        -----------
        ndim : int
            Number of dimensions
        """
        self.ndim = ndim
        self._scale = [1.0] * ndim
        self._origin = [0.0] * ndim
        self._axis_labels = []
        self._units = []
    
    def with_scale(self, scale: float | list[float, tuple[float, ...]]) -> 'CoordinateMappingBuilder':
        """
        Set scale factors.
        
        Parameters:
        -----------
        scale : float or list or tuple
            Scale factor(s). If single value, applies to all dimensions.
            
        Returns:
        --------
        CoordinateMappingBuilder
            Self for method chaining
        """
        if isinstance(scale, (int, float)):
            self._scale = [float(scale)] * self.ndim
        else:
            scale_list = list(scale)
            if len(scale_list) != self.ndim:
                raise ValueError(f"Scale must have {self.ndim} dimensions")
            self._scale = scale_list
        
        return self
    
    def with_origin(self, origin: float | list[float, tuple[float, ...]]) -> 'CoordinateMappingBuilder':
        """
        Set origin offset.
        
        Parameters:
        -----------
        origin : float or list or tuple
            Origin offset(s). If single value, applies to all dimensions.
            
        Returns:
        --------
        CoordinateMappingBuilder
            Self for method chaining
        """
        if isinstance(origin, (int, float)):
            self._origin = [float(origin)] * self.ndim
        else:
            origin_list = list(origin)
            if len(origin_list) != self.ndim:
                raise ValueError(f"Origin must have {self.ndim} dimensions")
            self._origin = origin_list
        
        return self
    
    def with_axis_labels(self, labels: list[str]) -> 'CoordinateMappingBuilder':
        """
        Set axis labels.
        
        Parameters:
        -----------
        labels : list[str]
            Axis labels
            
        Returns:
        --------
        CoordinateMappingBuilder
            Self for method chaining
        """
        if len(labels) != self.ndim:
            raise ValueError(f"Labels must have {self.ndim} dimensions")
        self._axis_labels = list(labels)
        return self
    
    def with_units(self, units: list[str]) -> 'CoordinateMappingBuilder':
        """
        Set physical units.
        
        Parameters:
        -----------
        units : list[str]
            Physical units
            
        Returns:
        --------
        CoordinateMappingBuilder
            Self for method chaining
        """
        if len(units) != self.ndim:
            raise ValueError(f"Units must have {self.ndim} dimensions")
        self._units = list(units)
        return self
    
    def build(self) -> CoordinateMapping:
        """
        Build the coordinate mapping.
        
        Returns:
        --------
        CoordinateMapping
            Constructed mapping
        """
        return CoordinateMapping(
            scale=tuple(self._scale),
            origin=tuple(self._origin),
            axis_labels=tuple(self._axis_labels) if self._axis_labels else (),
            units=tuple(self._units) if self._units else ()
        )


def create_isotropic_mapping(ndim: int, scale: float = 1.0, origin: float = 0.0) -> CoordinateMapping:
    """
    Create isotropic coordinate mapping (same scale in all dimensions).
    
    Parameters:
    -----------
    ndim : int
        Number of dimensions
    scale : float
        Uniform scale factor
    origin : float
        Uniform origin offset
        
    Returns:
    --------
    CoordinateMapping
        Isotropic coordinate mapping
    """
    return CoordinateMapping(
        scale=tuple(scale for _ in range(ndim)),
        origin=tuple(origin for _ in range(ndim))
    )


def create_anisotropic_mapping(scale: list[float], origin: list[float | None] = None) -> CoordinateMapping:
    """
    Create anisotropic coordinate mapping (different scales per dimension).
    
    Parameters:
    -----------
    scale : list[float]
        Scale factors for each dimension
    origin : list[float], optional
        Origin offsets for each dimension
        
    Returns:
    --------
    CoordinateMapping
        Anisotropic coordinate mapping
    """
    if origin is None:
        origin = [0.0] * len(scale)
    
    return CoordinateMapping(
        scale=tuple(scale),
        origin=tuple(origin)
    )


def create_physical_mapping(physical_scale: list[float], 
                          physical_units: list[str],
                          axis_labels: list[str | None] = None,
                          origin: list[float | None] = None) -> CoordinateMapping:
    """
    Create coordinate mapping with physical scales and units.
    
    Parameters:
    -----------
    physical_scale : list[float]
        Physical scale factors (e.g., [0.1, 0.1, 0.5] for mm/pixel)
    physical_units : list[str]
        Physical units (e.g., ['mm', 'mm', 'mm'])
    axis_labels : list[str], optional
        Axis labels (e.g., ['x', 'y', 'z'])
    origin : list[float], optional
        Origin offsets
        
    Returns:
    --------
    CoordinateMapping
        Physical coordinate mapping
    """
    ndim = len(physical_scale)
    
    if origin is None:
        origin = [0.0] * ndim
    
    if axis_labels is None:
        axis_labels = ['x', 'y', 'z', 'w', 'v', 'u'][:ndim]
    
    return CoordinateMapping(
        scale=tuple(physical_scale),
        origin=tuple(origin),
        axis_labels=tuple(axis_labels),
        units=tuple(physical_units)
    )