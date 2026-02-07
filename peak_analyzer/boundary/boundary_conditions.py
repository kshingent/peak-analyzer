"""
Boundary Conditions

Provides boundary handling for peak analysis including edge detection,
extrapolation methods, and boundary constraint enforcement.
"""

from typing import Any
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


class BoundaryType(Enum):
    """Types of boundary conditions."""
    NONE = "none"              # No boundary handling
    PERIODIC = "periodic"      # Periodic boundaries
    MIRROR = "mirror"          # Mirror/reflect boundaries
    EXTRAPOLATE = "extrapolate"  # Extrapolate values
    CONSTANT = "constant"      # Constant value padding
    NEAREST = "nearest"        # Nearest value padding
    ZERO = "zero"             # Zero padding
    GRADIENT = "gradient"      # Linear gradient padding


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.
    """
    
    def __init__(self, boundary_type: BoundaryType):
        """
        Initialize boundary condition.
        
        Parameters:
        -----------
        boundary_type : BoundaryType
            Type of boundary condition
        """
        self.boundary_type = boundary_type
    
    @abstractmethod
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """
        Apply boundary condition for out-of-bounds access.
        
        Parameters:
        -----------
        data : np.ndarray
            Source data array
        indices : tuple of int
            Requested indices (may be out of bounds)
        shape : tuple of int
            Shape of the data array
            
        Returns:
        --------
        float or np.ndarray
            Value(s) at the specified indices with boundary condition applied
        """
        pass
    
    @abstractmethod
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """
        Extend array with boundary condition.
        
        Parameters:
        -----------
        data : np.ndarray
            Source data array
        padding : int or tuple of int
            Padding size (same for all dimensions or per dimension)
            
        Returns:
        --------
        np.ndarray
            Extended array with boundary padding
        """
        pass
    
    def is_in_bounds(self, indices: tuple[int, ...], shape: tuple[int, ...]) -> bool:
        """
        Check if indices are within array bounds.
        
        Parameters:
        -----------
        indices : tuple of int
            Indices to check
        shape : tuple of int
            Array shape
            
        Returns:
        --------
        bool
            True if indices are in bounds
        """
        return all(0 <= idx < size for idx, size in zip(indices, shape))


class NoBoundary(BoundaryCondition):
    """No boundary handling - raises error for out-of-bounds access."""
    
    def __init__(self):
        super().__init__(BoundaryType.NONE)
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply no boundary condition."""
        if not self.is_in_bounds(indices, shape):
            raise IndexError(f"Index {indices} is out of bounds for shape {shape}")
        return data[indices]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """No extension - return original array."""
        return data


class PeriodicBoundary(BoundaryCondition):
    """Periodic boundary condition - wraps around at edges."""
    
    def __init__(self):
        super().__init__(BoundaryType.PERIODIC)
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply periodic boundary condition."""
        wrapped_indices = tuple(idx % size for idx, size in zip(indices, shape))
        return data[wrapped_indices]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """Extend array with periodic boundary."""
        if isinstance(padding, int):
            padding = (padding,) * data.ndim
        
        extended = data
        for axis, pad in enumerate(padding):
            if pad > 0:
                # Wrap around for periodic boundary
                left_pad = data.take(range(-pad, 0), axis=axis)
                right_pad = data.take(range(pad), axis=axis)
                extended = np.concatenate([left_pad, extended, right_pad], axis=axis)
        
        return extended


class MirrorBoundary(BoundaryCondition):
    """Mirror boundary condition - reflects at edges."""
    
    def __init__(self):
        super().__init__(BoundaryType.MIRROR)
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply mirror boundary condition."""
        mirrored_indices = []
        
        for idx, size in zip(indices, shape):
            if idx < 0:
                # Mirror negative indices
                mirrored_idx = -idx - 1
            elif idx >= size:
                # Mirror indices beyond upper bound
                mirrored_idx = 2 * size - idx - 1
            else:
                mirrored_idx = idx
            
            # Ensure within bounds after mirroring
            mirrored_idx = np.clip(mirrored_idx, 0, size - 1)
            mirrored_indices.append(mirrored_idx)
        
        return data[tuple(mirrored_indices)]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """Extend array with mirror boundary."""
        if isinstance(padding, int):
            padding = (padding,) * data.ndim
        
        extended = data
        for axis, pad in enumerate(padding):
            if pad > 0:
                # Mirror for boundary extension
                size = extended.shape[axis]
                
                # Left padding (mirror from start)
                if pad >= size:
                    # Multiple reflections needed
                    left_indices = [i % (2 * size) for i in range(pad, 0, -1)]
                    left_indices = [idx if idx < size else 2 * size - idx - 1 
                                  for idx in left_indices]
                else:
                    left_indices = list(range(pad, 0, -1))
                
                left_pad = extended.take(left_indices, axis=axis)
                
                # Right padding (mirror from end)
                if pad >= size:
                    # Multiple reflections needed
                    right_indices = [i % (2 * size) for i in range(size - 1 - pad, size - 1)]
                    right_indices = [idx if idx >= 0 else -idx - 1 
                                   for idx in right_indices]
                else:
                    right_indices = list(range(size - 2, size - 2 - pad, -1))
                
                right_pad = extended.take(right_indices, axis=axis)
                
                extended = np.concatenate([left_pad, extended, right_pad], axis=axis)
        
        return extended


class ConstantBoundary(BoundaryCondition):
    """Constant boundary condition - fills with constant value."""
    
    def __init__(self, constant_value: float = 0.0):
        super().__init__(BoundaryType.CONSTANT)
        self.constant_value = constant_value
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply constant boundary condition."""
        if not self.is_in_bounds(indices, shape):
            return self.constant_value
        return data[indices]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """Extend array with constant boundary."""
        if isinstance(padding, int):
            padding = [(padding, padding)] * data.ndim
        else:
            padding = [(p, p) if isinstance(p, int) else p for p in padding]
        
        return np.pad(data, padding, mode='constant', constant_values=self.constant_value)


class NearestBoundary(BoundaryCondition):
    """Nearest boundary condition - extends with nearest edge values."""
    
    def __init__(self):
        super().__init__(BoundaryType.NEAREST)
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply nearest boundary condition."""
        clipped_indices = tuple(np.clip(idx, 0, size - 1) 
                               for idx, size in zip(indices, shape))
        return data[clipped_indices]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """Extend array with nearest boundary."""
        if isinstance(padding, int):
            padding = [(padding, padding)] * data.ndim
        else:
            padding = [(p, p) if isinstance(p, int) else p for p in padding]
        
        return np.pad(data, padding, mode='edge')


class ZeroBoundary(BoundaryCondition):
    """Zero boundary condition - fills with zeros."""
    
    def __init__(self):
        super().__init__(BoundaryType.ZERO)
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply zero boundary condition."""
        if not self.is_in_bounds(indices, shape):
            return 0.0
        return data[indices]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """Extend array with zero boundary."""
        if isinstance(padding, int):
            padding = [(padding, padding)] * data.ndim
        else:
            padding = [(p, p) if isinstance(p, int) else p for p in padding]
        
        return np.pad(data, padding, mode='constant', constant_values=0.0)


class LinearExtrapolateBoundary(BoundaryCondition):
    """Linear extrapolation boundary condition."""
    
    def __init__(self):
        super().__init__(BoundaryType.EXTRAPOLATE)
    
    def apply(self, data: np.ndarray, indices: tuple[int, ...], 
              shape: tuple[int, ...]) -> float | np.ndarray:
        """Apply linear extrapolation boundary condition."""
        if self.is_in_bounds(indices, shape):
            return data[indices]
        
        # Simple extrapolation using nearest edge and gradient
        clipped_indices = tuple(np.clip(idx, 0, size - 1) 
                               for idx, size in zip(indices, shape))
        
        # For now, fallback to nearest (can be improved with actual gradient calculation)
        return data[clipped_indices]
    
    def extend_array(self, data: np.ndarray, padding: int | tuple[int, ...]) -> np.ndarray:
        """Extend array with linear extrapolation."""
        if isinstance(padding, int):
            padding = [(padding, padding)] * data.ndim
        else:
            padding = [(p, p) if isinstance(p, int) else p for p in padding]
        
        return np.pad(data, padding, mode='linear_ramp')


class BoundaryManager:
    """
    Manages boundary conditions for multi-dimensional arrays.
    """
    
    def __init__(self, shape: tuple[int, ...], 
                 boundary_conditions: BoundaryCondition | list[BoundaryCondition | None] = None):
        """
        Initialize boundary manager.
        
        Parameters:
        -----------
        shape : tuple of int
            Shape of the array domain
        boundary_conditions : BoundaryCondition or list of BoundaryCondition, optional
            Boundary conditions (same for all dimensions or per dimension)
        """
        self.shape = shape
        self.ndim = len(shape)
        
        if boundary_conditions is None:
            # Default to no boundary
            self.boundary_conditions = [NoBoundary()] * self.ndim
        elif isinstance(boundary_conditions, BoundaryCondition):
            # Same boundary condition for all dimensions
            self.boundary_conditions = [boundary_conditions] * self.ndim
        else:
            # Per-dimension boundary conditions
            if len(boundary_conditions) != self.ndim:
                raise ValueError("Number of boundary conditions must match dimensionality")
            self.boundary_conditions = boundary_conditions
    
    def get_value(self, data: np.ndarray, indices: tuple[int, ...]) -> float:
        """
        Get value at indices with boundary handling.
        
        Parameters:
        -----------
        data : np.ndarray
            Source data array
        indices : tuple of int
            Requested indices
            
        Returns:
        --------
        float
            Value at indices with boundary condition applied
        """
        if not isinstance(indices, tuple):
            indices = tuple(indices)
        
        # Check if any dimension is out of bounds
        for dim, (idx, bc) in enumerate(zip(indices, self.boundary_conditions)):
            if not (0 <= idx < self.shape[dim]):
                # Use the appropriate boundary condition
                return bc.apply(data, indices, self.shape)
        
        # All indices are in bounds
        return data[indices]
    
    def extend_data(self, data: np.ndarray, 
                    padding: int | tuple[int, ...]) -> tuple[np.ndarray, tuple[slice, ...]]:
        """
        Extend data array with boundary conditions.
        
        Parameters:
        -----------
        data : np.ndarray
            Source data array
        padding : int or tuple of int
            Padding size
            
        Returns:
        --------
        tuple of (extended_data, original_slice)
            Extended data and slice to access original region
        """
        if isinstance(padding, int):
            padding = (padding,) * self.ndim
        
        # Apply boundary conditions sequentially for each dimension
        extended = data
        original_slices = []
        
        for dim, (pad, bc) in enumerate(zip(padding, self.boundary_conditions)):
            if pad > 0:
                extended = bc.extend_array(extended, [(0, 0)] * dim + [(pad, pad)] + [(0, 0)] * (self.ndim - dim - 1))
                original_slices.append(slice(pad, pad + self.shape[dim]))
            else:
                original_slices.append(slice(None))
        
        return extended, tuple(original_slices)
    
    def create_padded_view(self, data: np.ndarray, 
                          padding: int | tuple[int, ...]) -> 'PaddedArrayView':
        """
        Create a padded view of the array with boundary conditions.
        
        Parameters:
        -----------
        data : np.ndarray
            Source data array
        padding : int or tuple of int
            Padding size
            
        Returns:
        --------
        PaddedArrayView
            Padded array view with boundary handling
        """
        return PaddedArrayView(data, self, padding)


class PaddedArrayView:
    """
    Virtual padded view of an array with boundary conditions.
    """
    
    def __init__(self, data: np.ndarray, boundary_manager: BoundaryManager, 
                 padding: int | tuple[int, ...]):
        """
        Initialize padded array view.
        
        Parameters:
        -----------
        data : np.ndarray
            Source data array
        boundary_manager : BoundaryManager
            Boundary manager for handling out-of-bounds access
        padding : int or tuple of int
            Padding size
        """
        self.data = data
        self.boundary_manager = boundary_manager
        self.original_shape = data.shape
        
        if isinstance(padding, int):
            self.padding = (padding,) * data.ndim
        else:
            self.padding = padding
        
        self.padded_shape = tuple(s + 2 * p for s, p in zip(self.original_shape, self.padding))
    
    def __getitem__(self, indices: tuple[int, ...]) -> float:
        """
        Get item with boundary handling.
        
        Parameters:
        -----------
        indices : tuple of int
            Indices in the padded array
            
        Returns:
        --------
        float
            Value at indices
        """
        # Convert padded indices to original array indices
        original_indices = tuple(idx - pad for idx, pad in zip(indices, self.padding))
        
        return self.boundary_manager.get_value(self.data, original_indices)
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get padded array shape."""
        return self.padded_shape


def create_boundary_condition(boundary_type: str | BoundaryType, 
                            **kwargs) -> BoundaryCondition:
    """
    Factory function for creating boundary conditions.
    
    Parameters:
    -----------
    boundary_type : str or BoundaryType
        Type of boundary condition
    **kwargs
        Boundary condition specific parameters
        
    Returns:
    --------
    BoundaryCondition
        Boundary condition instance
    """
    if isinstance(boundary_type, str):
        boundary_type = BoundaryType(boundary_type.lower())
    
    if boundary_type == BoundaryType.NONE:
        return NoBoundary()
    elif boundary_type == BoundaryType.PERIODIC:
        return PeriodicBoundary()
    elif boundary_type == BoundaryType.MIRROR:
        return MirrorBoundary()
    elif boundary_type == BoundaryType.CONSTANT:
        constant_value = kwargs.get('constant_value', 0.0)
        return ConstantBoundary(constant_value)
    elif boundary_type == BoundaryType.NEAREST:
        return NearestBoundary()
    elif boundary_type == BoundaryType.ZERO:
        return ZeroBoundary()
    elif boundary_type == BoundaryType.EXTRAPOLATE:
        return LinearExtrapolateBoundary()
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")


def analyze_boundary_effects(data: np.ndarray, 
                           boundary_condition: BoundaryCondition,
                           analysis_size: int = 5) -> dict[str, Any]:
    """
    Analyze effects of boundary condition on data edges.
    
    Parameters:
    -----------
    data : np.ndarray
        Source data array
    boundary_condition : BoundaryCondition
        Boundary condition to analyze
    analysis_size : int
        Size of edge region to analyze
        
    Returns:
    --------
    dict[str, Any]
        Analysis results including edge values and gradients
    """
    extended = boundary_condition.extend_array(data, analysis_size)
    
    results = {
        'original_shape': data.shape,
        'extended_shape': extended.shape,
        'boundary_type': boundary_condition.boundary_type.value,
        'edge_analysis': {}
    }
    
    # Analyze each dimension's edges
    for dim in range(data.ndim):
        # Left edge
        left_slice = [slice(None)] * data.ndim
        left_slice[dim] = slice(0, analysis_size)
        left_values = extended[tuple(left_slice)]
        
        # Right edge  
        right_slice = [slice(None)] * data.ndim
        right_slice[dim] = slice(-analysis_size, None)
        right_values = extended[tuple(right_slice)]
        
        # Original edges for comparison
        orig_left_slice = [slice(None)] * data.ndim
        orig_left_slice[dim] = slice(0, min(analysis_size, data.shape[dim]))
        orig_left_values = data[tuple(orig_left_slice)]
        
        orig_right_slice = [slice(None)] * data.ndim
        orig_right_slice[dim] = slice(max(0, data.shape[dim] - analysis_size), None)
        orig_right_values = data[tuple(orig_right_slice)]
        
        results['edge_analysis'][f'dim_{dim}'] = {
            'left_boundary_values': left_values,
            'right_boundary_values': right_values,
            'original_left_values': orig_left_values,
            'original_right_values': orig_right_values,
            'left_mean': np.mean(left_values),
            'right_mean': np.mean(right_values),
            'left_std': np.std(left_values),
            'right_std': np.std(right_values)
        }
    
    return results