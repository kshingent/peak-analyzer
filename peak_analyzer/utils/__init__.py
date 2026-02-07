"""
Utils Layer

Provides general utility functions, helpers, and common algorithms for peak analysis
including mathematical utilities, array operations, and performance tools.
"""

from .general import (
    timer,
    profile_memory,
    validate_array,
    ensure_list,
    ensure_tuple,
    normalize_array,
    pad_array,
    rolling_window,
    find_local_maxima,
    interpolate_missing,
    downsample_array,
    create_test_data,
    calculate_statistics
)

__all__ = [
    # Decorators and performance tools
    'timer',
    'profile_memory',
    
    # Data validation and conversion
    'validate_array',
    'ensure_list',
    'ensure_tuple',
    
    # Array operations
    'normalize_array',
    'pad_array',
    'rolling_window',
    'find_local_maxima',
    'interpolate_missing',
    'downsample_array',
    
    # Test data and analysis
    'create_test_data',
    'calculate_statistics',
]

# Version information
__version__ = '1.0.0'