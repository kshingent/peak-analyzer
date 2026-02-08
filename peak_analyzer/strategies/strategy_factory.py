"""
Strategy Factory

Factory for creating and configuring peak detection strategies.
"""

from typing import Any, Type
from .base_strategy import BaseStrategy, StrategyConfig
from .union_find_strategy import UnionFindStrategy
from .plateau_first_strategy import PlateauFirstStrategy


class StrategyFactory:
    """
    Factory class for creating and configuring peak detection strategies.
    """
    
    # Registry of available strategies
    _strategies: dict[str, Type[BaseStrategy]] = {
        'union_find': UnionFindStrategy,
        'plateau_first': PlateauFirstStrategy
    }
    
    # Strategy aliases for convenience
    _aliases: dict[str, str] = {
        'uf': 'union_find',
        'unionfind': 'union_find',
        'plateau': 'plateau_first',
        'morphological': 'plateau_first'
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


# Convenience functions

def create_union_find_strategy(**kwargs) -> UnionFindStrategy:
    """Create Union-Find strategy with parameters."""
    return StrategyFactory.create_strategy('union_find', **kwargs)

def create_plateau_first_strategy(**kwargs) -> PlateauFirstStrategy:
    """Create Plateau-First strategy with parameters."""
    return StrategyFactory.create_strategy('plateau_first', **kwargs)


