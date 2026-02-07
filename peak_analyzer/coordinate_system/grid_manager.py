"""
Grid Manager

Manages coordinate transformations between index space and coordinate space
with support for grid operations, boundary handling, and sampling.
"""

from typing import Any, Iterator
import numpy as np

from .coordinate_mapping import CoordinateMapping


class GridManager:
    """
    Manages coordinate grids and transformations between index and coordinate spaces.
    
    Provides high-level interface for coordinate operations including transformations,
    grid generation, sampling, and boundary handling.
    """
    
    def __init__(self, 
                 shape: tuple[int, ...], 
                 mapping: CoordinateMapping):
        """
        Initialize grid manager.
        
        Parameters:
        -----------
        shape : tuple of int
            Shape of the index space grid
        mapping : CoordinateMapping
            Coordinate mapping for transformations
        """
        self.shape = shape
        self.mapping = mapping
        
        if len(shape) != mapping.ndim:
            raise ValueError(f"Shape dimensions ({len(shape)}) must match mapping dimensions ({mapping.ndim})")
        
        # Cache frequently used values
        self._coord_bounds = None
        self._grid_coordinates = None
        self._index_mesh = None
        self._coordinate_mesh = None
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Total number of grid points."""
        return np.prod(self.shape)
    
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
        return self.mapping.index_to_coordinate(indices)
    
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
        return self.mapping.coordinate_to_index(coordinates, round_mode)
    
    def get_coordinate_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """
        Get coordinate bounds for the grid.
        
        Returns:
        --------
        tuple of (min_coords, max_coords)
            Coordinate bounds
        """
        if self._coord_bounds is None:
            self._coord_bounds = self.mapping.get_coordinate_bounds(self.shape)
        
        return self._coord_bounds
    
    def is_valid_index(self, indices: tuple[int, ...] | np.ndarray) -> bool | np.ndarray:
        """
        Check if indices are within grid bounds.
        
        Parameters:
        -----------
        indices : tuple of int or np.ndarray
            Index position(s) to check
            
        Returns:
        --------
        bool or np.ndarray
            True if indices are valid
        """
        if isinstance(indices, tuple):
            return all(0 <= indices[i] < self.shape[i] for i in range(self.ndim))
        
        elif isinstance(indices, np.ndarray):
            shape_array = np.array(self.shape)
            return np.all((indices >= 0) & (indices < shape_array), axis=-1)
        
        else:
            raise TypeError("Indices must be tuple or numpy array")
    
    def is_valid_coordinate(self, coordinates: tuple[float, ...] | np.ndarray) -> bool | np.ndarray:
        """
        Check if coordinates are within grid bounds.
        
        Parameters:
        -----------
        coordinates : tuple of float or np.ndarray
            Coordinate position(s) to check
            
        Returns:
        --------
        bool or np.ndarray
            True if coordinates are valid
        """
        indices = self.coordinate_to_index(coordinates, round_mode='nearest')
        return self.is_valid_index(indices)
    
    def clip_indices(self, indices: tuple[int, ...] | np.ndarray) -> tuple[int, ...] | np.ndarray:
        """
        Clip indices to grid bounds.
        
        Parameters:
        -----------
        indices : tuple of int or np.ndarray
            Index position(s) to clip
            
        Returns:
        --------
        tuple of int or np.ndarray
            Clipped indices
        """
        if isinstance(indices, tuple):
            return tuple(max(0, min(indices[i], self.shape[i] - 1)) for i in range(self.ndim))
        
        elif isinstance(indices, np.ndarray):
            shape_array = np.array(self.shape)
            return np.clip(indices, 0, shape_array - 1)
        
        else:
            raise TypeError("Indices must be tuple or numpy array")
    
    def clip_coordinates(self, coordinates: tuple[float, ...] | np.ndarray) -> tuple[float, ...] | np.ndarray:
        """
        Clip coordinates to grid bounds.
        
        Parameters:
        -----------
        coordinates : tuple of float or np.ndarray
            Coordinate position(s) to clip
            
        Returns:
        --------
        tuple of float or np.ndarray
            Clipped coordinates
        """
        min_coords, max_coords = self.get_coordinate_bounds()
        
        if isinstance(coordinates, tuple):
            return tuple(max(min_coords[i], min(coordinates[i], max_coords[i])) for i in range(self.ndim))
        
        elif isinstance(coordinates, np.ndarray):
            min_array = np.array(min_coords)
            max_array = np.array(max_coords)
            return np.clip(coordinates, min_array, max_array)
        
        else:
            raise TypeError("Coordinates must be tuple or numpy array")
    
    def generate_index_mesh(self) -> tuple[np.ndarray, ...]:
        """
        Generate meshgrid of all index positions.
        
        Returns:
        --------
        tuple of np.ndarray
            Meshgrid arrays for each dimension
        """
        if self._index_mesh is None:
            indices = [np.arange(s) for s in self.shape]
            self._index_mesh = np.meshgrid(*indices, indexing='ij')
        
        return self._index_mesh
    
    def generate_coordinate_mesh(self) -> tuple[np.ndarray, ...]:
        """
        Generate meshgrid of all coordinate positions.
        
        Returns:
        --------
        tuple of np.ndarray
            Coordinate meshgrid arrays for each dimension
        """
        if self._coordinate_mesh is None:
            index_mesh = self.generate_index_mesh()
            
            # Stack index arrays and convert to coordinates
            stacked_indices = np.stack(index_mesh, axis=-1)
            stacked_coords = self.mapping.index_to_coordinate(stacked_indices)
            
            # Split back into separate arrays
            self._coordinate_mesh = tuple(stacked_coords[..., i] for i in range(self.ndim))
        
        return self._coordinate_mesh
    
    def get_grid_coordinates(self) -> np.ndarray:
        """
        Get flattened array of all coordinate positions.
        
        Returns:
        --------
        np.ndarray
            Array of shape (total_points, ndim) with all coordinate positions
        """
        if self._grid_coordinates is None:
            coordinate_mesh = self.generate_coordinate_mesh()
            
            # Stack and reshape to flat array
            stacked = np.stack(coordinate_mesh, axis=-1)
            self._grid_coordinates = stacked.reshape(-1, self.ndim)
        
        return self._grid_coordinates
    
    def sample_coordinates(self, coordinates: np.ndarray, 
                          data: np.ndarray,
                          method: str = 'nearest',
                          fill_value: float = np.nan) -> np.ndarray:
        """
        Sample data at given coordinates using interpolation.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinate positions to sample at (shape: [..., ndim])
        data : np.ndarray
            Data array to sample from
        method : str
            Interpolation method ('nearest', 'linear')
        fill_value : float
            Value for out-of-bounds coordinates
            
        Returns:
        --------
        np.ndarray
            Sampled values
        """
        if data.shape != self.shape:
            raise ValueError(f"Data shape {data.shape} must match grid shape {self.shape}")
        
        # Convert coordinates to indices
        indices = self.coordinate_to_index(coordinates, round_mode='floor')
        
        if method == 'nearest':
            return self._sample_nearest(indices, data, fill_value)
        elif method == 'linear':
            return self._sample_linear(coordinates, data, fill_value)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def _sample_nearest(self, indices: np.ndarray, data: np.ndarray, fill_value: float) -> np.ndarray:
        """Sample using nearest neighbor interpolation."""
        # Round to nearest integer
        rounded_indices = np.round(indices).astype(int)
        
        # Check bounds
        valid_mask = self.is_valid_index(rounded_indices)
        
        # Initialize output
        output_shape = indices.shape[:-1]
        result = np.full(output_shape, fill_value, dtype=data.dtype)
        
        # Sample valid points
        if np.any(valid_mask):
            valid_indices = rounded_indices[valid_mask]
            
            # Convert to tuple for indexing
            index_tuples = tuple(valid_indices[..., i] for i in range(self.ndim))
            result[valid_mask] = data[index_tuples]
        
        return result
    
    def _sample_linear(self, coordinates: np.ndarray, data: np.ndarray, fill_value: float) -> np.ndarray:
        """Sample using linear interpolation."""
        # This is a simplified linear interpolation
        # For production use, consider scipy.ndimage.map_coordinates
        
        # Convert to floating point indices
        float_indices = self.coordinate_to_index(coordinates)
        
        # Get integer parts and fractional parts
        int_indices = np.floor(float_indices).astype(int)
        fractions = float_indices - int_indices
        
        # Check bounds
        valid_mask = self.is_valid_index(int_indices) & self.is_valid_index(int_indices + 1)
        
        # Initialize output
        output_shape = coordinates.shape[:-1]
        result = np.full(output_shape, fill_value, dtype=data.dtype)
        
        if not np.any(valid_mask):
            return result
        
        # Perform multi-dimensional linear interpolation
        # For simplicity, implement only for 2D and 3D cases
        if self.ndim == 2:
            result[valid_mask] = self._interpolate_2d(
                int_indices[valid_mask], fractions[valid_mask], data
            )
        elif self.ndim == 3:
            result[valid_mask] = self._interpolate_3d(
                int_indices[valid_mask], fractions[valid_mask], data
            )
        else:
            # Fall back to nearest neighbor for higher dimensions
            result = self._sample_nearest(int_indices, data, fill_value)
        
        return result
    
    def _interpolate_2d(self, int_indices: np.ndarray, fractions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """2D bilinear interpolation."""
        i0, j0 = int_indices[..., 0], int_indices[..., 1]
        i1, j1 = i0 + 1, j0 + 1
        
        fi, fj = fractions[..., 0], fractions[..., 1]
        
        # Sample corner values
        v00 = data[i0, j0]
        v01 = data[i0, j1]
        v10 = data[i1, j0]
        v11 = data[i1, j1]
        
        # Bilinear interpolation
        v0 = v00 * (1 - fj) + v01 * fj
        v1 = v10 * (1 - fj) + v11 * fj
        result = v0 * (1 - fi) + v1 * fi
        
        return result
    
    def _interpolate_3d(self, int_indices: np.ndarray, fractions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """3D trilinear interpolation."""
        i0, j0, k0 = int_indices[..., 0], int_indices[..., 1], int_indices[..., 2]
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        
        fi, fj, fk = fractions[..., 0], fractions[..., 1], fractions[..., 2]
        
        # Sample corner values
        v000 = data[i0, j0, k0]
        v001 = data[i0, j0, k1]
        v010 = data[i0, j1, k0]
        v011 = data[i0, j1, k1]
        v100 = data[i1, j0, k0]
        v101 = data[i1, j0, k1]
        v110 = data[i1, j1, k0]
        v111 = data[i1, j1, k1]
        
        # Trilinear interpolation
        v00 = v000 * (1 - fk) + v001 * fk
        v01 = v010 * (1 - fk) + v011 * fk
        v10 = v100 * (1 - fk) + v101 * fk
        v11 = v110 * (1 - fk) + v111 * fk
        
        v0 = v00 * (1 - fj) + v01 * fj
        v1 = v10 * (1 - fj) + v11 * fj
        
        result = v0 * (1 - fi) + v1 * fi
        
        return result
    
    def create_subgrid(self, 
                      index_slice: tuple[slice, ...],
                      update_origin: bool = True) -> 'GridManager':
        """
        Create subgrid from index slice.
        
        Parameters:
        -----------
        index_slice : tuple of slice
            Slice objects for each dimension
        update_origin : bool
            Whether to update origin to maintain coordinate alignment
            
        Returns:
        --------
        GridManager
            New grid manager for subgrid
        """
        if len(index_slice) != self.ndim:
            raise ValueError(f"Slice dimensions ({len(index_slice)}) must match grid dimensions ({self.ndim})")
        
        # Calculate new shape
        new_shape = tuple(
            len(range(*slice_obj.indices(self.shape[i]))) 
            for i, slice_obj in enumerate(index_slice)
        )
        
        # Calculate new mapping
        if update_origin:
            # Get starting indices
            start_indices = tuple(slice_obj.start or 0 for slice_obj in index_slice)
            
            # Convert to coordinates for new origin
            start_coords = self.index_to_coordinate(start_indices)
            new_mapping = self.mapping.with_modified_origin(start_coords)
        else:
            new_mapping = self.mapping
        
        return GridManager(new_shape, new_mapping)
    
    def create_scaled_grid(self, scale_factor: float | tuple[float, ...]) -> 'GridManager':
        """
        Create grid with scaled coordinates.
        
        Parameters:
        -----------
        scale_factor : float or tuple of float
            Scale factor(s) to apply
            
        Returns:
        --------
        GridManager
            New grid manager with scaled coordinates
        """
        if isinstance(scale_factor, (int, float)):
            new_scale = tuple(s * scale_factor for s in self.mapping.scale)
        else:
            if len(scale_factor) != self.ndim:
                raise ValueError("Scale factor dimensions must match grid dimensions")
            new_scale = tuple(self.mapping.scale[i] * scale_factor[i] for i in range(self.ndim))
        
        new_mapping = self.mapping.with_modified_scale(new_scale)
        return GridManager(self.shape, new_mapping)
    
    def get_distance_to_boundary(self, indices: tuple[int, ...] | np.ndarray) -> float | np.ndarray:
        """
        Calculate minimum distance to grid boundary.
        
        Parameters:
        -----------
        indices : tuple of int or np.ndarray
            Index position(s)
            
        Returns:
        --------
        float or np.ndarray
            Minimum distance to boundary in index units
        """
        if isinstance(indices, tuple):
            distances = []
            for i, idx in enumerate(indices):
                dist_to_lower = idx
                dist_to_upper = self.shape[i] - 1 - idx
                distances.append(min(dist_to_lower, dist_to_upper))
            return float(min(distances))
        
        elif isinstance(indices, np.ndarray):
            shape_array = np.array(self.shape) - 1
            distances_to_lower = indices
            distances_to_upper = shape_array - indices
            
            min_distances = np.minimum(distances_to_lower, distances_to_upper)
            return np.min(min_distances, axis=-1)
        
        else:
            raise TypeError("Indices must be tuple or numpy array")
    
    def iterate_indices(self, order: str = 'C') -> Iterator[tuple[int, ...]]:
        """
        Iterate over all index positions.
        
        Parameters:
        -----------
        order : str
            Iteration order ('C' for C-order, 'F' for Fortran-order)
            
        Yields:
        ------
        tuple of int
            Index positions
        """
        from itertools import product
        
        if order == 'C':
            ranges = [range(s) for s in self.shape]
        elif order == 'F':
            ranges = [range(s) for s in reversed(self.shape)]
        else:
            raise ValueError(f"Unknown order: {order}")
        
        for indices in product(*ranges):
            if order == 'F':
                indices = tuple(reversed(indices))
            yield indices
    
    def iterate_coordinates(self, order: str = 'C') -> Iterator[tuple[float, ...]]:
        """
        Iterate over all coordinate positions.
        
        Parameters:
        -----------
        order : str
            Iteration order ('C' for C-order, 'F' for Fortran-order)
            
        Yields:
        ------
        tuple of float
            Coordinate positions
        """
        for indices in self.iterate_indices(order):
            yield self.index_to_coordinate(indices)
    
    def get_grid_info(self) -> dict[str, Any]:
        """
        Get comprehensive grid information.
        
        Returns:
        --------
        dict[str, Any]
            Grid information dictionary
        """
        min_coords, max_coords = self.get_coordinate_bounds()
        
        return {
            'shape': self.shape,
            'ndim': self.ndim,
            'size': self.size,
            'mapping': self.mapping.to_dict(),
            'coordinate_bounds': {
                'min': min_coords,
                'max': max_coords
            },
            'voxel_volume': self.mapping.get_voxel_volume(),
            'total_volume': self.mapping.get_voxel_volume() * self.size
        }
    
    def clear_cache(self):
        """Clear cached data."""
        self._coord_bounds = None
        self._grid_coordinates = None
        self._index_mesh = None
        self._coordinate_mesh = None
    
    def __str__(self) -> str:
        """String representation of grid manager."""
        min_coords, max_coords = self.get_coordinate_bounds()
        
        lines = [
            f"GridManager ({self.ndim}D):",
            f"  Shape: {self.shape}",
            f"  Size: {self.size:,} points",
            f"  Coordinates: {min_coords} to {max_coords}",
            f"  Voxel volume: {self.mapping.get_voxel_volume():.6f}",
            f"  Mapping: {self.mapping.scale} scale, {self.mapping.origin} origin"
        ]
        
        return '\n'.join(lines)