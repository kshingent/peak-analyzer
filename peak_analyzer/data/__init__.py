"""
Data Layer

Provides data management, validation, and I/O capabilities for peak analysis
including data loading, saving, format conversion, and quality assessment.
"""

from .validation import (
    ValidationLevel,
    ValidationResult,
    DataValidator,
    BasicValidator,
    RangeValidator,
    ShapeValidator,
    StatisticalValidator,
    CoordinateValidator,
    CompositeValidator,
    create_validator,
    validate_peak_data,
    validate_coordinate_consistency,
    check_data_completeness
)

__all__ = [
    # Data validation
    'ValidationLevel',
    'ValidationResult',
    'DataValidator',
    'BasicValidator',
    'RangeValidator',
    'ShapeValidator',
    'StatisticalValidator',
    'CoordinateValidator',
    'CompositeValidator',
    'create_validator',
    'validate_peak_data',
    'validate_coordinate_consistency',
    'check_data_completeness',
    
    # Data I/O
    'DataFormat',
    'DataLoader',
    'DataSaver',
    'load_data',
    'save_data',
    'convert_format',
]

# Version information
__version__ = '1.0.0'