"""
Parameter Validation for Peak Analysis

Provides validation functions for user inputs and algorithm parameters.
"""

from typing import Any
import warnings


class ParameterValidator:
    """
    Validates input parameters for peak analysis algorithms.
    """
    
    VALID_STRATEGIES = {'auto', 'union_find', 'plateau_first'}
    VALID_BOUNDARY_TYPES = {'infinite_height', 'infinite_depth', 'periodic', 'custom'}
    
    def __init__(self):
        """Initialize parameter validator."""
        pass
    
    def validate_strategy(self, strategy: str) -> str:
        """
        Validate detection strategy parameter.
        
        Parameters:
        -----------
        strategy : str
            Strategy name
            
        Returns:
        --------
        str
            Validated strategy name
            
        Raises:
        -------
        ValueError
            If strategy is not valid
        """
        if not isinstance(strategy, str):
            raise TypeError(f"Strategy must be a string, got {type(strategy)}")
            
        strategy = strategy.lower()
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {', '.join(self.VALID_STRATEGIES)}"
            )
        return strategy
    
    def validate_connectivity(self, connectivity: int) -> int:
        """
        Validate connectivity parameter.
        
        Parameters:
        -----------
        connectivity : int
            Integer k-connectivity value
            
        Returns:
        --------
        int
            Validated connectivity
            
        Raises:
        -------
        ValueError
            If connectivity is not valid
        """
        if connectivity < 1:
            raise ValueError("Connectivity level must be >= 1")
        return connectivity
    
    def validate_boundary(self, boundary: str) -> str:
        """
        Validate boundary condition parameter.
        
        Parameters:
        -----------
        boundary : str
            Boundary condition type
            
        Returns:
        --------
        str
            Validated boundary type
            
        Raises:
        -------
        ValueError
            If boundary type is not valid
        """
        if not isinstance(boundary, str):
            raise TypeError(f"Boundary must be a string, got {type(boundary)}")
            
        boundary = boundary.lower()
        if boundary not in self.VALID_BOUNDARY_TYPES:
            raise ValueError(
                f"Invalid boundary type '{boundary}'. "
                f"Valid options: {', '.join(self.VALID_BOUNDARY_TYPES)}"
            )
        return boundary
    
    def validate_scale(self, scale: float | list[float, None], data_ndim: int) -> list[float | None]:
        """
        Validate scale parameter.
        
        Parameters:
        -----------
        scale : float, list of float, or None
            Scale values for each dimension
        data_ndim : int
            Number of dimensions in data
            
        Returns:
        --------
        list of float or None
            Validated scale values
            
        Raises:
        -------
        ValueError
            If scale specification is invalid
        """
        if scale is None:
            return None
            
        if isinstance(scale, (int, float)):
            if scale <= 0:
                raise ValueError("Scale values must be positive")
            return [float(scale)] * data_ndim
            
        if isinstance(scale, (list, tuple)):
            if len(scale) != data_ndim:
                raise ValueError(
                    f"Scale list length ({len(scale)}) must match data dimensions ({data_ndim})"
                )
            scale_list = [float(s) for s in scale]
            if any(s <= 0 for s in scale_list):
                raise ValueError("All scale values must be positive")
            return scale_list
            
        raise TypeError(f"Scale must be number, list, or None, got {type(scale)}")
    
    def validate_minkowski_p(self, p: float) -> float:
        """
        Validate Minkowski distance parameter p.
        
        Parameters:
        -----------
        p : float
            Minkowski distance order (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)
            
        Returns:
        --------
        float
            Validated p value
            
        Raises:
        -------
        ValueError
            If p is not a positive number or is invalid for Minkowski distance
    """
        if not isinstance(p, (int, float)):
            raise TypeError(f"p must be a number, got {type(p)}")
    
        if p <= 0:
            raise ValueError("p must be positive for Minkowski distance")
    
        return float(p)
    
    def validate_filter_criteria(self, criteria: dict[str, Any]) -> dict[str, Any]:
        """
        Validate filter criteria parameters.
        
        Parameters:
        -----------
        criteria : dict
            Filter criteria dictionary
            
        Returns:
        --------
        dict
            Validated criteria
            
        Raises:
        -------
        ValueError
            If criteria format is invalid
        """
        valid_operators = {'gt', 'gte', 'lt', 'lte', 'eq', 'ne', 'in', 'nin'}
        validated_criteria = {}
        
        for key, value in criteria.items():
            # Parse key for feature and operator
            if '__' in key:
                feature, operator = key.split('__', 1)
                if operator not in valid_operators:
                    raise ValueError(
                        f"Invalid filter operator '{operator}'. "
                        f"Valid operators: {', '.join(valid_operators)}"
                    )
            else:
                feature, operator = key, 'eq'
            
            # Validate feature name (basic check)
            if not isinstance(feature, str) or not feature:
                raise ValueError("Feature name must be a non-empty string")
            
            validated_criteria[key] = value
            
        return validated_criteria
    
    def validate_wlen(self, wlen: float | None, data_shape: tuple) -> float | None:
        """
        Validate window length parameter for prominence calculation.
        
        Parameters:
        -----------
        wlen : float or None
            Window length constraint
        data_shape : tuple
            Shape of input data
            
        Returns:
        --------
        float or None
            Validated window length
            
        Raises:
        -------
        ValueError
            If wlen is invalid
        """
        if wlen is None:
            return None
            
        if not isinstance(wlen, (int, float)):
            raise TypeError(f"Window length must be a number, got {type(wlen)}")
            
        if wlen <= 0:
            raise ValueError("Window length must be positive")
            
        # Check if wlen is reasonable relative to data size
        max_dimension = max(data_shape)
        if wlen > max_dimension:
            warnings.warn(
                f"Window length ({wlen}) is larger than maximum data dimension ({max_dimension})"
            )
            
        return float(wlen)
    
    def validate_features_list(self, features: str | list[str]) -> list[str]:
        """
        Validate feature list parameter.
        
        Parameters:
        -----------
        features : str or list of str
            Feature specification
            
        Returns:
        --------
        list of str
            Validated feature list
            
        Raises:
        -------
        ValueError
            If features specification is invalid
        """
        if isinstance(features, str):
            if features.lower() == 'all':
                return ['all']  # Special case handled by feature manager
            else:
                return [features]
                
        if isinstance(features, (list, tuple)):
            if not all(isinstance(f, str) for f in features):
                raise ValueError("All features must be strings")
            return list(features)
            
        raise TypeError(f"Features must be string or list of strings, got {type(features)}")
    
    def validate_computation_parameters(self, **params) -> dict[str, Any]:
        """
        Validate parameters for feature computation.
        
        Parameters:
        -----------
        **params
            Computation parameters
            
        Returns:
        --------
        dict
            Validated parameters
            
        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        validated = {}
        
        for key, value in params.items():
            if key == 'min_area':
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError("min_area must be a non-negative number")
                validated[key] = value
                
            elif key == 'rel_height':
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError("rel_height must be between 0 and 1")
                validated[key] = value
                
            elif key == 'radius':
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError("radius must be positive")
                validated[key] = value
                
            else:
                # Allow other parameters to pass through
                validated[key] = value
                
        return validated