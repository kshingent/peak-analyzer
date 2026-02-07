"""
General Utilities

Provides general utility functions and helpers for peak analysis including
mathematical utilities, array operations, and common algorithms.
"""

from typing import Any, Callable
from collections.abc import Sequence
import numpy as np
from functools import wraps
import time
import warnings


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Parameters:
    -----------
    func : Callable
        Function to time
        
    Returns:
    --------
    Callable
        Decorated function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage (requires psutil).
    
    Parameters:
    -----------
    func : Callable
        Function to profile
        
    Returns:
    --------
    Callable
        Decorated function that prints memory usage
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            print(f"{func.__name__} memory usage: {mem_after:.2f} MB "
                  f"(change: {mem_diff:+.2f} MB)")
            
            return result
            
        except ImportError:
            warnings.warn("psutil not available for memory profiling")
            return func(*args, **kwargs)
    
    return wrapper


def validate_array(array: np.ndarray, 
                  min_dims: int | None = None,
                  max_dims: int | None = None,
                  dtype: type | np.dtype | None = None,
                  allow_empty: bool = False,
                  finite_only: bool = True) -> np.ndarray:
    """
    Validate and optionally convert array inputs.
    
    Parameters:
    -----------
    array : np.ndarray
        Array to validate
    min_dims : int, optional
        Minimum number of dimensions
    max_dims : int, optional
        Maximum number of dimensions
    dtype : type or np.dtype, optional
        Required data type
    allow_empty : bool
        Whether to allow empty arrays
    finite_only : bool
        Whether to require finite values only
        
    Returns:
    --------
    np.ndarray
        Validated array
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    # Convert to numpy array if needed
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    
    # Check emptiness
    if not allow_empty and array.size == 0:
        raise ValueError("Empty array not allowed")
    
    # Check dimensions
    if min_dims is not None and array.ndim < min_dims:
        raise ValueError(f"Array must have at least {min_dims} dimensions, got {array.ndim}")
    
    if max_dims is not None and array.ndim > max_dims:
        raise ValueError(f"Array must have at most {max_dims} dimensions, got {array.ndim}")
    
    # Check data type
    if dtype is not None:
        if not np.issubdtype(array.dtype, dtype):
            try:
                array = array.astype(dtype)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert array to type {dtype}")
    
    # Check for finite values
    if finite_only and array.size > 0:
        if not np.isfinite(array).all():
            raise ValueError("Array contains non-finite values (NaN or Inf)")
    
    return array


def ensure_list(obj: Any | Sequence[Any]) -> list[Any]:
    """
    Ensure object is a list.
    
    Parameters:
    -----------
    obj : Any or Sequence[Any]
        Object to convert to list
        
    Returns:
    --------
    list[Any]
        List containing the object(s)
    """
    if isinstance(obj, (list, tuple)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return [obj]


def ensure_tuple(obj: Any | Sequence[Any]) -> tuple[Any, ...]:
    """
    Ensure object is a tuple.
    
    Parameters:
    -----------
    obj : Any or Sequence[Any]
        Object to convert to tuple
        
    Returns:
    --------
    tuple[Any, ...]
        Tuple containing the object(s)
    """
    if isinstance(obj, tuple):
        return obj
    elif isinstance(obj, (list, np.ndarray)):
        return tuple(obj)
    else:
        return (obj,)


def normalize_array(array: np.ndarray, 
                   method: str = 'minmax',
                   axis: int | None = None) -> np.ndarray:
    """
    Normalize array values.
    
    Parameters:
    -----------
    array : np.ndarray
        Array to normalize
    method : str
        Normalization method ('minmax', 'zscore', 'robust', 'unit')
    axis : int, optional
        Axis along which to normalize
        
    Returns:
    --------
    np.ndarray
        Normalized array
    """
    array = validate_array(array)
    
    if method == 'minmax':
        # Min-max scaling to [0, 1]
        min_val = np.min(array, axis=axis, keepdims=True)
        max_val = np.max(array, axis=axis, keepdims=True)
        
        range_val = max_val - min_val
        # Avoid division by zero
        range_val = np.where(range_val == 0, 1, range_val)
        
        return (array - min_val) / range_val
    
    elif method == 'zscore':
        # Z-score normalization (zero mean, unit variance)
        mean_val = np.mean(array, axis=axis, keepdims=True)
        std_val = np.std(array, axis=axis, keepdims=True)
        
        # Avoid division by zero
        std_val = np.where(std_val == 0, 1, std_val)
        
        return (array - mean_val) / std_val
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median_val = np.median(array, axis=axis, keepdims=True)
        q75 = np.percentile(array, 75, axis=axis, keepdims=True)
        q25 = np.percentile(array, 25, axis=axis, keepdims=True)
        
        iqr = q75 - q25
        # Avoid division by zero
        iqr = np.where(iqr == 0, 1, iqr)
        
        return (array - median_val) / iqr
    
    elif method == 'unit':
        # Unit vector normalization (L2 norm)
        norm = np.linalg.norm(array, axis=axis, keepdims=True)
        # Avoid division by zero
        norm = np.where(norm == 0, 1, norm)
        
        return array / norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def pad_array(array: np.ndarray, 
              padding: int | tuple[int, ...], 
              mode: str = 'constant',
              **kwargs) -> np.ndarray:
    """
    Pad array with specified padding and mode.
    
    Parameters:
    -----------
    array : np.ndarray
        Array to pad
    padding : int or tuple of int
        Padding size (same for all dimensions or per dimension)
    mode : str
        Padding mode (see numpy.pad for options)
    **kwargs
        Additional arguments for numpy.pad
        
    Returns:
    --------
    np.ndarray
        Padded array
    """
    array = validate_array(array)
    
    if isinstance(padding, int):
        padding = ((padding, padding),) * array.ndim
    elif isinstance(padding, (tuple, list)):
        if len(padding) == array.ndim:
            # Per-dimension padding
            padding = tuple((p, p) if isinstance(p, int) else p for p in padding)
        else:
            raise ValueError(f"Padding length {len(padding)} doesn't match dimensions {array.ndim}")
    
    return np.pad(array, padding, mode=mode, **kwargs)


def rolling_window(array: np.ndarray, 
                  window_size: int | tuple[int, ...], 
                  step: int | tuple[int, ...] = 1) -> np.ndarray:
    """
    Create a rolling window view of an array.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array
    window_size : int or tuple of int
        Size of the rolling window
    step : int or tuple of int
        Step size for the window
        
    Returns:
    --------
    np.ndarray
        Array with additional dimensions for the windows
    """
    array = validate_array(array)
    
    if isinstance(window_size, int):
        window_size = (window_size,) * array.ndim
    
    if isinstance(step, int):
        step = (step,) * array.ndim
    
    if len(window_size) != array.ndim or len(step) != array.ndim:
        raise ValueError("Window size and step must match array dimensions")
    
    # Calculate output shape
    output_shape = []
    for i in range(array.ndim):
        size = (array.shape[i] - window_size[i]) // step[i] + 1
        if size <= 0:
            raise ValueError(f"Window too large for dimension {i}")
        output_shape.append(size)
    
    output_shape.extend(window_size)
    
    # Create strides for the rolling window
    input_strides = array.strides
    output_strides = tuple(s * step[i] for i, s in enumerate(input_strides)) + input_strides
    
    return np.lib.stride_tricks.as_strided(
        array, 
        shape=output_shape,
        strides=output_strides,
        writeable=False
    )


def find_local_maxima(array: np.ndarray, 
                     min_distance: int | tuple[int, ...] = 1,
                     threshold_abs: float | None = None,
                     threshold_rel: float | None = None,
                     num_peaks: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima in an array.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array
    min_distance : int or tuple of int
        Minimum distance between peaks
    threshold_abs : float, optional
        Absolute threshold for peak values
    threshold_rel : float, optional
        Relative threshold (fraction of max value)
    num_peaks : int, optional
        Maximum number of peaks to return
        
    Returns:
    --------
    tuple of (indices, values)
        Peak indices and values
    """
    array = validate_array(array, finite_only=True)
    
    if isinstance(min_distance, int):
        min_distance = (min_distance,) * array.ndim
    
    # Find candidates by comparing with neighbors
    maxima_mask = np.ones(array.shape, dtype=bool)
    
    for dim in range(array.ndim):
        for offset in [-1, 1]:
            # Shift array along this dimension
            shifted = np.roll(array, offset, axis=dim)
            
            # Create mask for valid comparisons (exclude wrapped edges)
            valid_mask = np.ones(array.shape, dtype=bool)
            if offset == -1:
                # Don't compare with right edge
                slc = [slice(None)] * array.ndim
                slc[dim] = slice(-1, None)
                valid_mask[tuple(slc)] = False
            else:
                # Don't compare with left edge
                slc = [slice(None)] * array.ndim
                slc[dim] = slice(0, 1)
                valid_mask[tuple(slc)] = False
            
            # Array must be greater than shifted version
            maxima_mask &= (array > shifted) | ~valid_mask
    
    # Apply thresholds
    if threshold_abs is not None:
        maxima_mask &= (array >= threshold_abs)
    
    if threshold_rel is not None:
        threshold_value = threshold_rel * np.max(array)
        maxima_mask &= (array >= threshold_value)
    
    # Get peak coordinates
    peak_indices = np.where(maxima_mask)
    peak_values = array[peak_indices]
    
    if len(peak_values) == 0:
        return np.array([]), np.array([])
    
    # Apply minimum distance constraint
    if any(d > 1 for d in min_distance):
        # Create coordinate array
        peak_coords = np.column_stack(peak_indices)
        
        # Sort by value (descending)
        sort_order = np.argsort(-peak_values)
        peak_coords = peak_coords[sort_order]
        peak_values = peak_values[sort_order]
        
        # Filter by minimum distance
        selected = [0]  # Always keep the highest peak
        
        for i in range(1, len(peak_coords)):
            candidate = peak_coords[i]
            
            # Check distance to all selected peaks
            keep = True
            for j in selected:
                reference = peak_coords[j]
                distances = np.abs(candidate - reference)
                
                if np.all(distances < min_distance):
                    keep = False
                    break
            
            if keep:
                selected.append(i)
        
        # Update arrays
        peak_coords = peak_coords[selected]
        peak_values = peak_values[selected]
        
        # Convert back to tuple of arrays
        peak_indices = tuple(peak_coords[:, i] for i in range(array.ndim))
    
    # Limit number of peaks
    if num_peaks is not None and len(peak_values) > num_peaks:
        # Sort by value and keep top N
        sort_order = np.argsort(-peak_values)[:num_peaks]
        peak_indices = tuple(indices[sort_order] for indices in peak_indices)
        peak_values = peak_values[sort_order]
    
    return peak_indices, peak_values


def interpolate_missing(array: np.ndarray, 
                       method: str = 'linear',
                       missing_value: float | None = None) -> np.ndarray:
    """
    Interpolate missing values in an array.
    
    Parameters:
    -----------
    array : np.ndarray
        Array with missing values
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')
    missing_value : float, optional
        Value considered as missing (default: NaN)
        
    Returns:
    --------
    np.ndarray
        Array with interpolated values
    """
    array = validate_array(array, finite_only=False)
    
    if missing_value is None:
        # Use NaN/Inf as missing values
        missing_mask = ~np.isfinite(array)
    else:
        # Use specific value as missing
        missing_mask = (array == missing_value)
    
    if not np.any(missing_mask):
        # No missing values
        return array
    
    result = array.copy()
    
    if array.ndim == 1:
        # 1D interpolation
        valid_indices = np.where(~missing_mask)[0]
        missing_indices = np.where(missing_mask)[0]
        
        if len(valid_indices) == 0:
            warnings.warn("No valid values for interpolation")
            return result
        
        valid_values = array[valid_indices]
        
        if method == 'linear':
            interpolated = np.interp(missing_indices, valid_indices, valid_values)
        elif method == 'nearest':
            # Simple nearest neighbor
            interpolated = []
            for idx in missing_indices:
                distances = np.abs(valid_indices - idx)
                nearest_idx = valid_indices[np.argmin(distances)]
                interpolated.append(array[nearest_idx])
            interpolated = np.array(interpolated)
        else:
            try:
                from scipy.interpolate import interp1d
                f = interp1d(valid_indices, valid_values, kind=method, 
                           bounds_error=False, fill_value='extrapolate')
                interpolated = f(missing_indices)
            except ImportError:
                warnings.warn("scipy not available, falling back to linear interpolation")
                interpolated = np.interp(missing_indices, valid_indices, valid_values)
        
        result[missing_indices] = interpolated
    
    else:
        # Multi-dimensional: interpolate along each axis
        for axis in range(array.ndim):
            # Move current axis to the end
            moved = np.moveaxis(result, axis, -1)
            original_shape = moved.shape
            
            # Reshape to 2D for easier processing
            reshaped = moved.reshape(-1, original_shape[-1])
            
            for i in range(reshaped.shape[0]):
                row = reshaped[i]
                row_missing = ~np.isfinite(row) if missing_value is None else (row == missing_value)
                
                if np.any(row_missing) and not np.all(row_missing):
                    valid_indices = np.where(~row_missing)[0]
                    missing_indices = np.where(row_missing)[0]
                    valid_values = row[valid_indices]
                    
                    interpolated = np.interp(missing_indices, valid_indices, valid_values)
                    row[missing_indices] = interpolated
                    reshaped[i] = row
            
            # Reshape back and move axis back
            moved = reshaped.reshape(original_shape)
            result = np.moveaxis(moved, -1, axis)
    
    return result


def downsample_array(array: np.ndarray, 
                    factors: int | tuple[int, ...], 
                    method: str = 'mean') -> np.ndarray:
    """
    Downsample array by given factors.
    
    Parameters:
    -----------
    array : np.ndarray
        Array to downsample
    factors : int or tuple of int
        Downsampling factors for each dimension
    method : str
        Downsampling method ('mean', 'max', 'min', 'median')
        
    Returns:
    --------
    np.ndarray
        Downsampled array
    """
    array = validate_array(array)
    
    if isinstance(factors, int):
        factors = (factors,) * array.ndim
    
    if len(factors) != array.ndim:
        raise ValueError("Number of factors must match array dimensions")
    
    # Check if downsampling is needed
    if all(f == 1 for f in factors):
        return array
    
    result = array
    
    for axis, factor in enumerate(factors):
        if factor <= 1:
            continue
        
        # Get the size along this axis
        axis_size = result.shape[axis]
        new_size = axis_size // factor
        
        if new_size == 0:
            raise ValueError(f"Factor {factor} too large for axis {axis} of size {axis_size}")
        
        # Trim array to be divisible by factor
        trim_size = new_size * factor
        slc = [slice(None)] * result.ndim
        slc[axis] = slice(0, trim_size)
        trimmed = result[tuple(slc)]
        
        # Reshape to separate the downsampling dimension
        old_shape = list(trimmed.shape)
        new_shape = old_shape.copy()
        new_shape[axis] = new_size
        new_shape.insert(axis + 1, factor)
        
        reshaped = trimmed.reshape(new_shape)
        
        # Apply reduction operation
        if method == 'mean':
            reduced = np.mean(reshaped, axis=axis + 1)
        elif method == 'max':
            reduced = np.max(reshaped, axis=axis + 1)
        elif method == 'min':
            reduced = np.min(reshaped, axis=axis + 1)
        elif method == 'median':
            reduced = np.median(reshaped, axis=axis + 1)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        result = reduced
    
    return result


def create_test_data(shape: tuple[int, ...], 
                    pattern: str = 'peaks',
                    noise_level: float = 0.1,
                    random_state: int | None = None) -> np.ndarray:
    """
    Create test data for peak analysis.
    
    Parameters:
    -----------
    shape : tuple of int
        Shape of the test array
    pattern : str
        Type of test pattern ('peaks', 'sinusoid', 'gaussian', 'random')
    noise_level : float
        Amount of noise to add
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Test data array
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if pattern == 'peaks':
        # Create multiple Gaussian peaks
        data = np.zeros(shape)
        
        # Add several peaks at random locations
        num_peaks = min(5, np.prod(shape) // 100 + 1)
        
        for _ in range(num_peaks):
            # Random center
            center = [np.random.randint(0, s) for s in shape]
            
            # Random amplitude and width
            amplitude = np.random.uniform(0.5, 1.0)
            width = [np.random.uniform(s * 0.05, s * 0.2) for s in shape]
            
            # Create meshgrid
            coordinates = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
            
            # Calculate Gaussian
            exponent = sum((coord - c) ** 2 / (2 * w ** 2) 
                          for coord, c, w in zip(coordinates, center, width))
            peak = amplitude * np.exp(-exponent)
            
            data += peak
    
    elif pattern == 'sinusoid':
        # Create sinusoidal pattern
        coordinates = np.meshgrid(*[np.linspace(0, 4 * np.pi, s) for s in shape], indexing='ij')
        
        data = np.zeros(shape)
        for i, coord in enumerate(coordinates):
            frequency = 1.0 + i * 0.5
            data += np.sin(frequency * coord)
        
        data = data / len(coordinates)  # Normalize
    
    elif pattern == 'gaussian':
        # Single large Gaussian
        center = [s // 2 for s in shape]
        width = [s * 0.3 for s in shape]
        
        coordinates = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        exponent = sum((coord - c) ** 2 / (2 * w ** 2) 
                      for coord, c, w in zip(coordinates, center, width))
        data = np.exp(-exponent)
    
    elif pattern == 'random':
        # Random data
        data = np.random.random(shape)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, shape)
        data += noise
    
    return data


def calculate_statistics(array: np.ndarray, 
                        axis: int | tuple[int, ...] | None = None) -> dict[str, float]:
    """
    Calculate comprehensive statistics for an array.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array
    axis : int or tuple of int, optional
        Axis or axes along which to calculate statistics
        
    Returns:
    --------
    dict[str, float]
        Dictionary of statistical measures
    """
    array = validate_array(array, finite_only=False)
    
    # Handle missing values
    finite_mask = np.isfinite(array)
    finite_array = array[finite_mask] if finite_mask.any() else array
    
    if finite_array.size == 0:
        return {'count': 0}
    
    stats = {
        'count': finite_array.size,
        'mean': np.mean(finite_array),
        'std': np.std(finite_array),
        'var': np.var(finite_array),
        'min': np.min(finite_array),
        'max': np.max(finite_array),
        'median': np.median(finite_array),
        'range': np.max(finite_array) - np.min(finite_array),
    }
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        stats[f'p{p}'] = np.percentile(finite_array, p)
    
    # Inter-quartile range
    stats['iqr'] = stats['p75'] - stats['p25']
    
    # Coefficient of variation
    if stats['mean'] != 0:
        stats['cv'] = stats['std'] / abs(stats['mean'])
    else:
        stats['cv'] = np.inf if stats['std'] > 0 else 0
    
    # Skewness and kurtosis (simplified versions)
    if stats['std'] > 0:
        centered = finite_array - stats['mean']
        normalized = centered / stats['std']
        stats['skewness'] = np.mean(normalized ** 3)
        stats['kurtosis'] = np.mean(normalized ** 4) - 3  # Excess kurtosis
    else:
        stats['skewness'] = 0
        stats['kurtosis'] = 0
    
    # Missing value information
    total_size = array.size
    missing_count = total_size - finite_array.size
    stats['missing_count'] = missing_count
    stats['missing_fraction'] = missing_count / total_size if total_size > 0 else 0
    
    return stats