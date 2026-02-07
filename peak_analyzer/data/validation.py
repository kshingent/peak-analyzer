"""
Data Validation

Provides comprehensive data validation capabilities for peak analysis including
data type checking, range validation, shape consistency, and quality assessment.
"""

from typing import Any
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

from ..coordinate_system import CoordinateMapping, GridManager


class ValidationLevel(Enum):
    """Levels of validation strictness."""
    MINIMAL = "minimal"      # Basic type and shape checks
    STANDARD = "standard"    # Standard validation including ranges
    STRICT = "strict"        # Comprehensive validation
    CUSTOM = "custom"        # Custom validation rules


class ValidationResult:
    """
    Result of data validation.
    """
    
    def __init__(self, is_valid: bool = True, warnings: list[str | None] = None,
                 errors: list[str | None] = None, metadata: dict[str, Any | None] = None):
        """
        Initialize validation result.
        
        Parameters:
        -----------
        is_valid : bool
            Whether data passed validation
        warnings : list[str], optional
            Validation warnings
        errors : list[str], optional
            Validation errors
        metadata : dict[str, Any], optional
            Additional validation metadata
        """
        self.is_valid = is_valid
        self.warnings = warnings or []
        self.errors = errors or []
        self.metadata = metadata or {}
    
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)
    
    def add_error(self, message: str):
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def __bool__(self) -> bool:
        """Boolean conversion returns validation status."""
        return self.is_valid
    
    def __str__(self) -> str:
        """String representation of validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        result = f"Validation: {status}\n"
        
        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  - {error}\n"
        
        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  - {warning}\n"
        
        return result


class DataValidator(ABC):
    """
    Abstract base class for data validators.
    """
    
    def __init__(self, name: str):
        """
        Initialize data validator.
        
        Parameters:
        -----------
        name : str
            Name of the validator
        """
        self.name = name
    
    @abstractmethod
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """
        Validate data.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to validate
        **kwargs
            Additional validation parameters
            
        Returns:
        --------
        ValidationResult
            Validation result
        """
        pass


class BasicValidator(DataValidator):
    """
    Basic data validation (type, shape, finite values).
    """
    
    def __init__(self):
        super().__init__("BasicValidator")
    
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Validate basic data properties."""
        result = ValidationResult()
        
        # Check if input is numpy array
        if not isinstance(data, np.ndarray):
            result.add_error(f"Data must be numpy array, got {type(data)}")
            return result
        
        # Check if array is empty
        if data.size == 0:
            result.add_error("Data array is empty")
            return result
        
        # Check for finite values
        if not np.isfinite(data).all():
            num_invalid = np.sum(~np.isfinite(data))
            total = data.size
            result.add_error(f"Data contains {num_invalid}/{total} non-finite values (NaN/Inf)")
        
        # Check data type compatibility
        if not np.issubdtype(data.dtype, np.number):
            result.add_error(f"Data must be numeric, got dtype {data.dtype}")
        
        # Store metadata
        result.metadata.update({
            'shape': data.shape,
            'dtype': data.dtype,
            'size': data.size,
            'ndim': data.ndim,
            'finite_values': np.sum(np.isfinite(data)) if result.is_valid else 0
        })
        
        return result


class RangeValidator(DataValidator):
    """
    Validate data value ranges.
    """
    
    def __init__(self, min_value: float | None = None, max_value: float | None = None,
                 allow_equal: bool = True):
        """
        Initialize range validator.
        
        Parameters:
        -----------
        min_value : float, optional
            Minimum allowed value
        max_value : float, optional
            Maximum allowed value
        allow_equal : bool
            Whether to allow values equal to bounds
        """
        super().__init__("RangeValidator")
        self.min_value = min_value
        self.max_value = max_value
        self.allow_equal = allow_equal
    
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Validate data value ranges."""
        result = ValidationResult()
        
        if self.min_value is not None:
            if self.allow_equal:
                violates_min = data < self.min_value
            else:
                violates_min = data <= self.min_value
            
            if np.any(violates_min):
                num_violations = np.sum(violates_min)
                min_found = np.min(data)
                result.add_error(f"{num_violations} values below minimum {self.min_value} "
                               f"(minimum found: {min_found})")
        
        if self.max_value is not None:
            if self.allow_equal:
                violates_max = data > self.max_value
            else:
                violates_max = data >= self.max_value
            
            if np.any(violates_max):
                num_violations = np.sum(violates_max)
                max_found = np.max(data)
                result.add_error(f"{num_violations} values above maximum {self.max_value} "
                               f"(maximum found: {max_found})")
        
        # Store metadata
        result.metadata.update({
            'data_min': np.min(data),
            'data_max': np.max(data),
            'required_min': self.min_value,
            'required_max': self.max_value
        })
        
        return result


class ShapeValidator(DataValidator):
    """
    Validate array shape and dimensionality.
    """
    
    def __init__(self, expected_shape: tuple[int, ... ] = None,
                 expected_ndim: int | None = None,
                 min_size: int | None = None):
        """
        Initialize shape validator.
        
        Parameters:
        -----------
        expected_shape : tuple of int, optional
            Expected exact shape
        expected_ndim : int, optional
            Expected number of dimensions
        min_size : int, optional
            Minimum total array size
        """
        super().__init__("ShapeValidator")
        self.expected_shape = expected_shape
        self.expected_ndim = expected_ndim
        self.min_size = min_size
    
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Validate array shape."""
        result = ValidationResult()
        
        # Check exact shape
        if self.expected_shape is not None:
            if data.shape != self.expected_shape:
                result.add_error(f"Shape mismatch: expected {self.expected_shape}, "
                               f"got {data.shape}")
        
        # Check number of dimensions
        if self.expected_ndim is not None:
            if data.ndim != self.expected_ndim:
                result.add_error(f"Dimension mismatch: expected {self.expected_ndim}D, "
                               f"got {data.ndim}D")
        
        # Check minimum size
        if self.min_size is not None:
            if data.size < self.min_size:
                result.add_error(f"Array too small: minimum size {self.min_size}, "
                               f"got {data.size}")
        
        # Store metadata
        result.metadata.update({
            'shape': data.shape,
            'ndim': data.ndim,
            'size': data.size
        })
        
        return result


class StatisticalValidator(DataValidator):
    """
    Validate statistical properties of data.
    """
    
    def __init__(self, check_variance: bool = True, min_variance: float = 1e-12,
                 check_distribution: bool = False, expected_mean: float | None = None,
                 mean_tolerance: float = 0.1):
        """
        Initialize statistical validator.
        
        Parameters:
        -----------
        check_variance : bool
            Whether to check for sufficient variance
        min_variance : float
            Minimum required variance
        check_distribution : bool
            Whether to check distribution properties
        expected_mean : float, optional
            Expected mean value
        mean_tolerance : float
            Tolerance for mean comparison
        """
        super().__init__("StatisticalValidator")
        self.check_variance = check_variance
        self.min_variance = min_variance
        self.check_distribution = check_distribution
        self.expected_mean = expected_mean
        self.mean_tolerance = mean_tolerance
    
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Validate statistical properties."""
        result = ValidationResult()
        
        # Calculate statistics
        data_mean = np.mean(data)
        data_var = np.var(data)
        data_std = np.std(data)
        
        # Check variance
        if self.check_variance and data_var < self.min_variance:
            result.add_warning(f"Low variance detected: {data_var:.2e} < {self.min_variance:.2e} "
                              "(data may be constant or nearly constant)")
        
        # Check expected mean
        if self.expected_mean is not None:
            mean_diff = abs(data_mean - self.expected_mean)
            if mean_diff > self.mean_tolerance:
                result.add_warning(f"Mean differs from expected: {data_mean:.6f} vs "
                                 f"{self.expected_mean:.6f} (diff: {mean_diff:.6f})")
        
        # Basic distribution checks
        if self.check_distribution:
            # Check for outliers (simple 3-sigma rule)
            if data_std > 0:
                outliers = np.abs(data - data_mean) > 3 * data_std
                num_outliers = np.sum(outliers)
                if num_outliers > 0:
                    result.add_warning(f"Found {num_outliers} potential outliers "
                                     f"(>3σ from mean)")
        
        # Store metadata
        result.metadata.update({
            'mean': data_mean,
            'variance': data_var,
            'std': data_std,
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data)
        })
        
        return result


class CoordinateValidator(DataValidator):
    """
    Validate coordinate system consistency.
    """
    
    def __init__(self, coordinate_mapping: CoordinateMapping | None = None,
                 grid_manager: GridManager | None = None):
        """
        Initialize coordinate validator.
        
        Parameters:
        -----------
        coordinate_mapping : CoordinateMapping, optional
            Expected coordinate mapping
        grid_manager : GridManager, optional
            Grid manager for validation
        """
        super().__init__("CoordinateValidator")
        self.coordinate_mapping = coordinate_mapping
        self.grid_manager = grid_manager
    
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Validate coordinate system consistency."""
        result = ValidationResult()
        
        # Check shape consistency with coordinate mapping
        if self.coordinate_mapping is not None:
            expected_shape = self.coordinate_mapping.shape
            if data.shape != expected_shape:
                result.add_error(f"Shape inconsistent with coordinate mapping: "
                               f"expected {expected_shape}, got {data.shape}")
        
        # Check grid manager consistency
        if self.grid_manager is not None:
            expected_shape = self.grid_manager.shape
            if data.shape != expected_shape:
                result.add_error(f"Shape inconsistent with grid manager: "
                               f"expected {expected_shape}, got {data.shape}")
            
            # Validate coordinate bounds if possible
            try:
                coord_bounds = self.grid_manager.coordinate_bounds()
                result.metadata['coordinate_bounds'] = coord_bounds
            except Exception as e:
                result.add_warning(f"Could not determine coordinate bounds: {e}")
        
        return result


class CompositeValidator(DataValidator):
    """
    Composite validator that runs multiple validators.
    """
    
    def __init__(self, validators: list[DataValidator], stop_on_error: bool = False):
        """
        Initialize composite validator.
        
        Parameters:
        -----------
        validators : list[DataValidator]
            List of validators to run
        stop_on_error : bool
            Whether to stop on first error
        """
        super().__init__("CompositeValidator")
        self.validators = validators
        self.stop_on_error = stop_on_error
    
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Run all validators."""
        overall_result = ValidationResult()
        individual_results = {}
        
        for validator in self.validators:
            try:
                result = validator.validate(data, **kwargs)
                individual_results[validator.name] = result
                
                # Merge results
                overall_result.warnings.extend(result.warnings)
                overall_result.errors.extend(result.errors)
                overall_result.metadata[validator.name] = result.metadata
                
                if not result.is_valid:
                    overall_result.is_valid = False
                    if self.stop_on_error:
                        break
                        
            except Exception as e:
                error_msg = f"Validator {validator.name} failed: {e}"
                overall_result.add_error(error_msg)
                if self.stop_on_error:
                    break
        
        overall_result.metadata['individual_results'] = individual_results
        
        return overall_result


def create_validator(validation_level: str | ValidationLevel,
                    data_shape: tuple[int, ... ] = None,
                    coordinate_mapping: CoordinateMapping | None = None,
                    **kwargs) -> DataValidator:
    """
    Factory function for creating validators based on validation level.
    
    Parameters:
    -----------
    validation_level : str or ValidationLevel
        Level of validation
    data_shape : tuple of int, optional
        Expected data shape
    coordinate_mapping : CoordinateMapping, optional
        Coordinate mapping for validation
    **kwargs
        Additional validator parameters
        
    Returns:
    --------
    DataValidator
        Configured validator
    """
    if isinstance(validation_level, str):
        validation_level = ValidationLevel(validation_level.lower())
    
    validators = []
    
    if validation_level == ValidationLevel.MINIMAL:
        # Basic validation only
        validators.append(BasicValidator())
        
    elif validation_level == ValidationLevel.STANDARD:
        # Standard validation
        validators.append(BasicValidator())
        
        # Range validation if specified
        min_val = kwargs.get('min_value')
        max_val = kwargs.get('max_value')
        if min_val is not None or max_val is not None:
            validators.append(RangeValidator(min_val, max_val))
        
        # Shape validation if specified
        if data_shape is not None:
            validators.append(ShapeValidator(expected_shape=data_shape))
        
    elif validation_level == ValidationLevel.STRICT:
        # Comprehensive validation
        validators.append(BasicValidator())
        
        # Range validation
        min_val = kwargs.get('min_value')
        max_val = kwargs.get('max_value')
        if min_val is not None or max_val is not None:
            validators.append(RangeValidator(min_val, max_val))
        
        # Shape validation
        if data_shape is not None:
            validators.append(ShapeValidator(expected_shape=data_shape))
        
        # Statistical validation
        validators.append(StatisticalValidator())
        
        # Coordinate validation if mapping provided
        if coordinate_mapping is not None:
            validators.append(CoordinateValidator(coordinate_mapping))
    
    elif validation_level == ValidationLevel.CUSTOM:
        # Custom validators from kwargs
        custom_validators = kwargs.get('validators', [])
        if not custom_validators:
            raise ValueError("Custom validation level requires 'validators' parameter")
        validators.extend(custom_validators)
    
    # Return single validator or composite
    if len(validators) == 1:
        return validators[0]
    else:
        stop_on_error = kwargs.get('stop_on_error', False)
        return CompositeValidator(validators, stop_on_error)


def validate_peak_data(data: np.ndarray, 
                      validation_level: str | ValidationLevel = ValidationLevel.STANDARD,
                      coordinate_mapping: CoordinateMapping | None = None,
                      **kwargs) -> ValidationResult:
    """
    Comprehensive validation function for peak analysis data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to validate
    validation_level : str or ValidationLevel
        Level of validation to perform
    coordinate_mapping : CoordinateMapping, optional
        Coordinate mapping for validation
    **kwargs
        Additional validation parameters
        
    Returns:
    --------
    ValidationResult
        Validation result with detailed information
    """
    # Create appropriate validator
    validator = create_validator(validation_level, 
                               data_shape=data.shape,
                               coordinate_mapping=coordinate_mapping,
                               **kwargs)
    
    # Perform validation
    result = validator.validate(data, **kwargs)
    
    # Add overall data quality assessment
    if result.is_valid:
        quality_score = _calculate_quality_score(data, result.metadata)
        result.metadata['quality_score'] = quality_score
        
        if quality_score < 0.5:
            result.add_warning(f"Low data quality score: {quality_score:.2f}")
        elif quality_score > 0.9:
            result.metadata['high_quality'] = True
    
    return result


def _calculate_quality_score(data: np.ndarray, metadata: dict[str, Any]) -> float:
    """
    Calculate a simple data quality score.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    metadata : dict[str, Any]
        Validation metadata
        
    Returns:
    --------
    float
        Quality score between 0 and 1
    """
    score = 1.0
    
    # Penalize for non-finite values
    finite_ratio = np.sum(np.isfinite(data)) / data.size
    score *= finite_ratio
    
    # Penalize for very low variance (constant data)
    if 'StatisticalValidator' in metadata:
        stats = metadata['StatisticalValidator']
        variance = stats.get('variance', 0)
        data_range = stats.get('max', 0) - stats.get('min', 0)
        
        if data_range > 0:
            # Normalize variance by data range
            normalized_var = variance / (data_range ** 2)
            var_score = min(1.0, normalized_var * 100)  # Scale factor
            score *= var_score
        else:
            score *= 0.1  # Constant data gets low score
    
    # Bonus for good statistical properties
    if data.size > 1:
        # Check for reasonable distribution spread
        data_std = np.std(data)
        data_mean = np.mean(data)
        
        if data_std > 0 and abs(data_mean) > 0:
            cv = data_std / abs(data_mean)  # Coefficient of variation
            # Reasonable CV gets bonus
            if 0.01 < cv < 10:
                score *= 1.1
    
    return min(1.0, score)


def validate_coordinate_consistency(data: np.ndarray,
                                  coordinate_mapping: CoordinateMapping,
                                  tolerance: float = 1e-10) -> ValidationResult:
    """
    Validate consistency between data and coordinate mapping.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    coordinate_mapping : CoordinateMapping
        Coordinate mapping to validate against
    tolerance : float
        Numerical tolerance for comparisons
        
    Returns:
    --------
    ValidationResult
        Validation result focusing on coordinate consistency
    """
    result = ValidationResult()
    
    # Check shape consistency
    if data.shape != coordinate_mapping.shape:
        result.add_error(f"Data shape {data.shape} doesn't match "
                        f"coordinate mapping shape {coordinate_mapping.shape}")
    
    # Check coordinate bounds make sense
    try:
        coord_bounds = coordinate_mapping.coordinate_bounds()
        result.metadata['coordinate_bounds'] = coord_bounds
        
        # Check if bounds are reasonable
        for dim, (min_coord, max_coord) in enumerate(coord_bounds):
            if max_coord <= min_coord:
                result.add_error(f"Invalid coordinate bounds in dimension {dim}: "
                                f"[{min_coord}, {max_coord}]")
            
            coord_range = max_coord - min_coord
            if coord_range < tolerance:
                result.add_warning(f"Very small coordinate range in dimension {dim}: "
                                 f"{coord_range}")
                
    except Exception as e:
        result.add_error(f"Could not calculate coordinate bounds: {e}")
    
    # Check scale factors are reasonable
    scale = coordinate_mapping.scale
    for dim, s in enumerate(scale):
        if s <= 0:
            result.add_error(f"Invalid scale factor in dimension {dim}: {s}")
        elif s < tolerance:
            result.add_warning(f"Very small scale factor in dimension {dim}: {s}")
    
    return result


def check_data_completeness(data: np.ndarray, 
                          expected_coverage: float = 0.95) -> ValidationResult:
    """
    Check data completeness (non-NaN coverage).
    
    Parameters:
    -----------
    data : np.ndarray
        Data array to check
    expected_coverage : float
        Expected fraction of non-NaN values
        
    Returns:
    --------
    ValidationResult
        Validation result focusing on data completeness
    """
    result = ValidationResult()
    
    # Calculate coverage
    finite_mask = np.isfinite(data)
    coverage = np.sum(finite_mask) / data.size
    
    result.metadata['coverage'] = coverage
    result.metadata['missing_values'] = data.size - np.sum(finite_mask)
    result.metadata['total_values'] = data.size
    
    if coverage < expected_coverage:
        result.add_error(f"Insufficient data coverage: {coverage:.3f} < {expected_coverage:.3f}")
    elif coverage < 0.8:
        result.add_warning(f"Low data coverage: {coverage:.3f}")
    
    return result