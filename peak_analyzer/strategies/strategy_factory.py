"""
Strategy Factory

Factory for creating and configuring peak detection strategies.
"""

from typing import Any, Type
from .base_strategy import BaseStrategy, StrategyConfig
from .union_find_strategy import UnionFindStrategy
from .plateau_first_strategy import PlateauFirstStrategy
from .hybrid_strategy import HybridStrategy
import numpy as np


class StrategyFactory:
    """
    Factory class for creating and configuring peak detection strategies.
    """
    
    # Registry of available strategies
    _strategies: dict[str, Type[BaseStrategy]] = {
        'union_find': UnionFindStrategy,
        'plateau_first': PlateauFirstStrategy,
        'hybrid': HybridStrategy
    }
    
    # Strategy aliases for convenience
    _aliases: dict[str, str] = {
        'uf': 'union_find',
        'unionfind': 'union_find',
        'plateau': 'plateau_first',
        'morphological': 'plateau_first',
        'adaptive': 'hybrid',
        'combined': 'hybrid'
    }
    
    @classmethod
    def create_strategy(
        self, 
        strategy_name: str, 
        config: StrategyConfig | None = None,
        **kwargs
    ) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy to create
        config : StrategyConfig, optional
            Configuration for the strategy
        **kwargs
            Additional strategy-specific parameters
            
        Returns:
        --------
        BaseStrategy
            Configured strategy instance
            
        Raises:
        -------
        ValueError
            If strategy name is not recognized
        """
        # Normalize strategy name
        normalized_name = self._normalize_strategy_name(strategy_name)
        
        if normalized_name not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available strategies: {list(self._strategies.keys())}"
            )
        
        # Get strategy class
        strategy_class = self._strategies[normalized_name]
        
        # Create and return instance
        return strategy_class(config=config, **kwargs)
    
    @classmethod
    def _normalize_strategy_name(cls, name: str) -> str:
        """Normalize strategy name, handling aliases."""
        name = name.lower().strip()
        
        # Check aliases first
        if name in cls._aliases:
            return cls._aliases[name]
        
        return name
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a new strategy class.
        
        Parameters:
        -----------
        name : str
            Name for the strategy
        strategy_class : Type[BaseStrategy]
            Strategy class to register
            
        Raises:
        -------
        TypeError
            If strategy_class is not a subclass of BaseStrategy
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError("Strategy class must inherit from BaseStrategy")
        
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def register_alias(cls, alias: str, strategy_name: str) -> None:
        """
        Register an alias for an existing strategy.
        
        Parameters:
        -----------
        alias : str
            Alias name
        strategy_name : str
            Name of existing strategy
            
        Raises:
        -------
        ValueError
            If strategy_name is not registered
        """
        normalized_name = cls._normalize_strategy_name(strategy_name)
        
        if normalized_name not in cls._strategies:
            raise ValueError(f"Strategy '{strategy_name}' is not registered")
        
        cls._aliases[alias.lower()] = normalized_name
    
    @classmethod
    def list_strategies(cls) -> dict[str, Type[BaseStrategy]]:
        """
        Get dictionary of all registered strategies.
        
        Returns:
        --------
        dict[str, Type[BaseStrategy]]
            Dictionary mapping strategy names to classes
        """
        return cls._strategies.copy()
    
    @classmethod
    def list_aliases(cls) -> dict[str, str]:
        """
        Get dictionary of all registered aliases.
        
        Returns:
        --------
        dict[str, str]
            Dictionary mapping aliases to strategy names
        """
        return cls._aliases.copy()
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> dict[str, Any]:
        """
        Get information about a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy
            
        Returns:
        --------
        dict[str, Any]
            Strategy information
            
        Raises:
        -------
        ValueError
            If strategy is not found
        """
        normalized_name = cls._normalize_strategy_name(strategy_name)
        
        if normalized_name not in cls._strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy_class = cls._strategies[normalized_name]
        
        return {
            'name': normalized_name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': strategy_class.__doc__,
            'aliases': [alias for alias, target in cls._aliases.items() if target == normalized_name]
        }
    
    @classmethod
    def create_default_config(cls, **overrides) -> StrategyConfig:
        """
        Create default strategy configuration with optional overrides.
        
        Parameters:
        -----------
        **overrides
            Configuration parameters to override defaults
            
        Returns:
        --------
        StrategyConfig
            Configuration object
        """
        return StrategyConfig(**overrides)
    
    @classmethod
    def recommend_strategy(
        cls, 
        data_shape: tuple, 
        characteristics: dict[str, Any | None] = None
    ) -> str:
        """
        Recommend optimal strategy based on data characteristics.
        
        Parameters:
        -----------
        data_shape : tuple
            Shape of input data
        characteristics : dict, optional
            Data characteristics (plateau_ratio, noise_level, etc.)
            
        Returns:
        --------
        str
            Recommended strategy name
        """
        data_size = np.prod(data_shape) if hasattr(np, 'prod') else 1
        for dim in data_shape:
            data_size *= dim
        
        # Default characteristics if not provided
        if characteristics is None:
            characteristics = {}
        
        plateau_ratio = characteristics.get('plateau_ratio', 0.1)
        noise_level = characteristics.get('noise_level', 0.1)
        peak_density = characteristics.get('peak_density', 0.01)
        
        # Simple recommendation heuristics
        
        # For very large data, use hybrid for memory efficiency
        if data_size > 10000000:  # 10M elements
            return 'hybrid'
        
        # For high plateau ratio, use plateau_first
        if plateau_ratio > 0.3:
            return 'plateau_first'
        
        # For very noisy data or high peak density, use union_find
        if noise_level > 0.2 or peak_density > 0.05:
            return 'union_find'
        
        # For medium-sized data with mixed characteristics, use hybrid
        if data_size > 1000000:  # 1M elements
            return 'hybrid'
        
        # Default to union_find for small to medium data
        return 'union_find'
    
    @classmethod
    def auto_configure(
        cls,
        data_shape: tuple,
        characteristics: dict[str, Any | None] = None,
        performance_requirements: dict[str, Any | None] = None
    ) -> BaseStrategy:
        """
        Automatically configure optimal strategy for given requirements.
        
        Parameters:
        -----------
        data_shape : tuple
            Shape of input data
        characteristics : dict, optional
            Data characteristics
        performance_requirements : dict, optional
            Performance requirements (max_memory, max_time, etc.)
            
        Returns:
        --------
        BaseStrategy
            Optimally configured strategy instance
        """
        # Recommend strategy
        strategy_name = cls.recommend_strategy(data_shape, characteristics)
        
        # Create base configuration
        config = StrategyConfig()
        strategy_kwargs = {}
        
        # Apply performance-based adjustments
        if performance_requirements:
            max_memory = performance_requirements.get('max_memory_mb')
            if max_memory:
                # Adjust chunk size for memory constraints
                if strategy_name == 'hybrid':
                    # Estimate chunk size to stay within memory limit
                    element_size = 8  # bytes
                    max_elements = max_memory * 1024 * 1024 / element_size / 2  # Safety factor
                    chunk_size = int(max_elements ** (1.0 / len(data_shape)))
                    strategy_kwargs['chunk_size'] = max(64, chunk_size)
            
            max_time = performance_requirements.get('max_time_seconds')
            if max_time:
                # For tight time constraints, prefer faster strategies
                if max_time < 10:  # 10 seconds
                    strategy_name = 'plateau_first'  # Generally faster for simple cases
                    
            min_accuracy = performance_requirements.get('min_accuracy', 0.8)
            if min_accuracy > 0.9:
                # High accuracy requirement - prefer union_find or hybrid
                if strategy_name == 'plateau_first':
                    strategy_name = 'hybrid'
        
        # Apply data characteristic adjustments
        if characteristics:
            # Adjust connectivity based on data dimensionality
            if len(data_shape) > 3:
                config.connectivity = 'face'  # Conservative for high dimensions
            
            # Adjust noise handling
            noise_level = characteristics.get('noise_level', 0.0)
            if noise_level > 0.1:
                config.noise_threshold = noise_level * 0.5
        
        # Create configured strategy
        return cls.create_strategy(strategy_name, config, **strategy_kwargs)


# Convenience functions

def create_union_find_strategy(**kwargs) -> UnionFindStrategy:
    """Create Union-Find strategy with parameters."""
    return StrategyFactory.create_strategy('union_find', **kwargs)

def create_plateau_first_strategy(**kwargs) -> PlateauFirstStrategy:
    """Create Plateau-First strategy with parameters."""
    return StrategyFactory.create_strategy('plateau_first', **kwargs)

def create_hybrid_strategy(**kwargs) -> HybridStrategy:
    """Create Hybrid strategy with parameters."""
    return StrategyFactory.create_strategy('hybrid', **kwargs)

def auto_strategy(data_shape: tuple, **requirements) -> BaseStrategy:
    """Create automatically configured strategy."""
    return StrategyFactory.auto_configure(data_shape, **requirements)