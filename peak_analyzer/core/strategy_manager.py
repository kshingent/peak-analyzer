"""
Strategy Selection and Management

Handles automatic selection of optimal peak detection strategies based on data characteristics.
"""

import numpy as np
import warnings
from typing import Any

from peak_analyzer.models import DataCharacteristics, BenchmarkResults
from peak_analyzer.strategies.base_strategy import BaseStrategy, StrategyConfig
from peak_analyzer.strategies.union_find_strategy import UnionFindStrategy
from peak_analyzer.strategies.plateau_first_strategy import PlateauFirstStrategy
from peak_analyzer.core.data_analyzer import DataAnalyzer


class StrategyManager:
    """
    Manages selection and benchmarking of peak detection strategies.
    """
    
    def __init__(self):
        """Initialize strategy manager."""
        self.available_strategies = {
            'union_find': UnionFindStrategy,
            'plateau_first': PlateauFirstStrategy
        }
        self.strategy_cache = {}
        self.data_analyzer = DataAnalyzer()
        
    def select_optimal_strategy(self, data: np.ndarray, **kwargs) -> BaseStrategy:
        """
        Select optimal strategy based on data characteristics.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        **kwargs
            Additional parameters for strategy selection
            
        Returns:
        --------
        BaseStrategy
            Optimal strategy instance for the given data
        """
        # Analyze data characteristics using specialized analyzer
        characteristics = self.data_analyzer.analyze_characteristics(data)
        
        # Check if chunked processing is needed
        if self.data_analyzer.should_use_chunking(data):
            warnings.warn(
                f"Large dataset detected ({data.size} elements). "
                "Consider using smaller data chunks or optimized processing."
            )
        
        # Determine optimal strategy based on characteristics
        strategy_name = self._decide_strategy_name(characteristics, **kwargs)
        
        # Create or retrieve strategy instance
        return self._get_cached_strategy(strategy_name)
    
    def estimate_computational_cost(self, strategy_name: str, data_shape: tuple[int, ...]) -> dict[str, float]:
        """
        Estimate computational cost for a strategy and data size.
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy to evaluate
        data_shape : tuple
            Shape of input data
            
        Returns:
        --------
        dict[str, float]
            Dictionary with performance metrics containing keys:
            'estimated_time', 'estimated_memory', 'accuracy_score', 'scalability_factor'
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        strategy_class = self.available_strategies[strategy_name]
        
        # Get estimated performance from strategy
        return strategy_class.estimate_performance(data_shape)
    
    def benchmark_strategies(self, data: np.ndarray, strategies: list[str | None] = None) -> list[BenchmarkResults]:
        """
        Benchmark multiple strategies on given data.
        
        Parameters:
        -----------
        data : np.ndarray
            Test data for benchmarking
        strategies : list of str, optional
            List of strategy names to benchmark. If None, benchmarks all available.
            
        Returns:
        --------
        list of BenchmarkResults
            Benchmark results for each strategy
        """
        if strategies is None:
            strategies = list(self.available_strategies.keys())
            
        results = []
        
        for strategy_name in strategies:
            if strategy_name in self.available_strategies:
                result = self._benchmark_single_strategy(data, strategy_name)
                results.append(result)
                
        return results
    
    def configure_strategy(self, strategy_name: str, **params) -> BaseStrategy:
        """
        Configure and return strategy instance with specified parameters.
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy to configure
        **params
            Strategy-specific parameters
            
        Returns:
        --------
        BaseStrategy
            Configured strategy instance
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        strategy_class = self.available_strategies[strategy_name]       
        return strategy_class(**params)
    
    def auto_configure(
        self,
        data_shape: tuple,
        characteristics: DataCharacteristics | None = None,
        performance_requirements: dict[str, Any] | None = None
    ) -> BaseStrategy:
        """
        Automatically configure optimal strategy for given requirements.
        
        Parameters:
        -----------
        data_shape : tuple
            Shape of input data
        characteristics : DataCharacteristics, optional
            Data characteristics from DataAnalyzer
        performance_requirements : dict, optional
            Performance requirements (max_memory, max_time, etc.)
            
        Returns:
        --------
        BaseStrategy
            Optimally configured strategy instance
        """
        # Create dummy characteristics if not provided
        if characteristics is None:
            characteristics = self._create_dummy_characteristics(data_shape)
        
        # Determine optimal strategy using unified logic
        strategy_name = self._decide_strategy_name(
            characteristics, 
            performance_requirements=performance_requirements
        )
        
        # Build configuration based on characteristics and requirements
        config = self._build_config(characteristics, performance_requirements, data_shape)
        
        # Create configured strategy
        strategy_class = self.available_strategies[strategy_name]
        return strategy_class(config=config)
    
    def _decide_strategy_name(
        self, 
        characteristics: DataCharacteristics, 
        performance_requirements: dict[str, Any] | None = None,
        **kwargs
    ) -> str:
        """
        Unified strategy decision logic based on data characteristics and performance requirements.
        
        This is the single source of truth for strategy selection.
        
        Parameters:
        -----------
        characteristics : DataCharacteristics
            Data characteristics from DataAnalyzer
        performance_requirements : dict, optional
            Performance requirements (max_time_seconds, min_accuracy, etc.)
        **kwargs
            Additional parameters. Supports 'strategy_name' or 'force_strategy' for manual override.
            
        Returns:
        --------
        str
            Strategy name to use
        """
        # Manual strategy override - highest priority
        manual_strategy = kwargs.get('strategy_name') or kwargs.get('force_strategy')
        if manual_strategy:
            if manual_strategy not in self.available_strategies:
                raise ValueError(f"Unknown strategy: {manual_strategy}")
            return manual_strategy
        
        data_size = np.prod(characteristics.shape)
        plateau_ratio = characteristics.plateau_ratio
        noise_level = characteristics.noise_level
        peak_density = characteristics.peak_density_estimate
        
        # 1. Performance requirements override data-based decisions
        if performance_requirements:
            max_time = performance_requirements.get('max_time_seconds')
            min_accuracy = performance_requirements.get('min_accuracy', 0.8)
            
            # High accuracy requirement - prefer union_find
            if min_accuracy > 0.9:
                return 'union_find'
            
            # Time constraint - prefer faster plateau_first for simple cases
            if max_time and max_time < 10 and data_size < 100000 and plateau_ratio > 0.1:
                return 'plateau_first'
        
        # 2. Data-based strategy selection
        # For small data or very high plateau ratio, use plateau-first
        if data_size < 10000 or plateau_ratio > 0.4:
            return 'plateau_first'
        
        # For large data with low plateau ratio, use union-find
        if data_size > 100000 and plateau_ratio < 0.1:
            return 'union_find'
        
        # For high plateau ratio (moderate), prefer plateau-first
        if plateau_ratio > 0.2:
            return 'plateau_first'
        
        # For noisy data or high peak density, prefer union-find
        if (noise_level > characteristics.value_range[1] * 0.1 or 
              peak_density > 0.01):
            return 'union_find'
        
        # Default: for medium-sized data with balanced characteristics
        # Choose based on data size
        if data_size > 50000:
            return 'union_find'
        else:
            return 'plateau_first'
    
    def _benchmark_single_strategy(self, data: np.ndarray, strategy_name: str) -> BenchmarkResults:
        """Benchmark a single strategy."""
        import time
        import psutil
        import os
        
        strategy = self.available_strategies[strategy_name]()
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        peaks = strategy.detect_peaks(data)
        execution_time = time.time() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Basic quality metrics (can be expanded)
        quality_metrics = {
            'peaks_per_megapixel': len(peaks) / (data.size / 1e6),
            'average_peak_height': np.mean([p.height for p in peaks]) if peaks else 0.0
        }
        
        return BenchmarkResults(
            strategy_name=strategy_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            peaks_detected=len(peaks),
            quality_metrics=quality_metrics
        )
    
    def _get_cached_strategy(self, strategy_name: str) -> BaseStrategy:
        """
        Get cached strategy instance or create new one.
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy to retrieve
            
        Returns:
        --------
        BaseStrategy
            Strategy instance
        """
        if strategy_name not in self.strategy_cache:
            strategy_class = self.available_strategies[strategy_name]
            self.strategy_cache[strategy_name] = strategy_class()
            
        return self.strategy_cache[strategy_name]
    
    def _create_dummy_characteristics(self, data_shape: tuple) -> DataCharacteristics:
        """
        Create dummy characteristics when not provided.
        
        Parameters:
        -----------
        data_shape : tuple
            Shape of input data
            
        Returns:
        --------
        DataCharacteristics
            Dummy characteristics with conservative defaults
        """
        
        # Conservative defaults for unknown data
        return DataCharacteristics(
            shape=data_shape,
            value_range=(0.0, 1.0),
            plateau_ratio=0.1,  # Assume low plateau ratio
            noise_level=0.05,   # Assume low noise
            peak_density_estimate=0.005  # Assume moderate peak density
        )
    
    def _build_config(
        self, 
        characteristics: DataCharacteristics, 
        performance_requirements: dict[str, Any] | None,
        data_shape: tuple
    ) -> StrategyConfig:
        """
        Build strategy configuration based on characteristics and requirements.
        
        Parameters:
        -----------
        characteristics : DataCharacteristics
            Data characteristics
        performance_requirements : dict, optional
            Performance requirements
        data_shape : tuple
            Shape of input data
            
        Returns:
        --------
        StrategyConfig
            Configured strategy parameters
        """
        config = StrategyConfig()
        
        # Adjust connectivity based on data dimensionality
        if len(data_shape) > 3:
            config.connectivity = 1  # Conservative for high dimensions
        
        # Adjust noise handling based on data characteristics
        if characteristics.noise_level > 0.1:
            config.noise_threshold = characteristics.noise_level * 0.5
        
        # Performance-based adjustments
        if performance_requirements:
            max_memory = performance_requirements.get('max_memory_mb')
            if max_memory and max_memory < 1000:  # Less than 1GB
                config.chunk_size = min(config.chunk_size or 100000, 50000)
        
        return config