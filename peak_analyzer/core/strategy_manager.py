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
        strategy_name = self._choose_strategy(characteristics, **kwargs)
        
        # Create or retrieve strategy instance
        if strategy_name not in self.strategy_cache:
            strategy_class = self.available_strategies[strategy_name]
            self.strategy_cache[strategy_name] = strategy_class()
            
        return self.strategy_cache[strategy_name]
    
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
        # Recommend strategy based on data characteristics
        strategy_name = self._recommend_strategy_by_characteristics(data_shape, characteristics)
        
        # Create base configuration
        config = StrategyConfig()
        strategy_kwargs = {}
        
        # Apply performance-based adjustments
        if performance_requirements:
            max_time = performance_requirements.get('max_time_seconds')
            if max_time and max_time < 10:  # 10 seconds
                strategy_name = 'plateau_first'  # Generally faster for simple cases
                    
            min_accuracy = performance_requirements.get('min_accuracy', 0.8)
            if min_accuracy > 0.9:
                # High accuracy requirement - prefer union_find
                if strategy_name == 'plateau_first':
                    strategy_name = 'union_find'
        
        # Apply data characteristic adjustments
        if characteristics:
            # Adjust connectivity based on data dimensionality
            if len(data_shape) > 3:
                config.connectivity = 1  # Conservative for high dimensions
            
            # Adjust noise handling
            if characteristics.noise_level > 0.1:
                config.noise_threshold = characteristics.noise_level * 0.5
        
        # Create configured strategy
        strategy_class = self.available_strategies[strategy_name]
        return strategy_class(config=config, **strategy_kwargs)
    
    def _recommend_strategy_by_characteristics(
        self, 
        data_shape: tuple, 
        characteristics: DataCharacteristics | None = None
    ) -> str:
        """
        Recommend optimal strategy based on data characteristics.
        
        Parameters:
        -----------
        data_shape : tuple
            Shape of input data
        characteristics : DataCharacteristics, optional
            Data characteristics from DataAnalyzer
            
        Returns:
        --------
        str
            Recommended strategy name
        """
        data_size = np.prod(data_shape)
        
        if characteristics is None:
            # Default heuristics based on data size only
            if data_size > 1000000:  # 1M elements
                return 'union_find'
            else:
                return 'plateau_first'
        
        # Use detailed characteristics for decision
        plateau_ratio = characteristics.plateau_ratio
        noise_level = characteristics.noise_level
        peak_density = characteristics.peak_density_estimate
        
        # For very large data, use union_find for consistency
        if data_size > 10000000:  # 10M elements
            return 'union_find'
        
        # For high plateau ratio, use plateau_first
        if plateau_ratio > 0.3:
            return 'plateau_first'
        
        # For very noisy data or high peak density, use union_find
        if noise_level > 0.2 or peak_density > 0.05:
            return 'union_find'
        
        # For medium-sized data, use union_find as default
        if data_size > 1000000:  # 1M elements
            return 'union_find'
        
        # Default to plateau_first for smaller data
        return 'plateau_first'
    
    def _choose_strategy(self, characteristics: DataCharacteristics, **kwargs) -> str:
        """Choose optimal strategy based on data characteristics."""
        data_size = np.prod(characteristics.shape)
        plateau_ratio = characteristics.plateau_ratio
        noise_level = characteristics.noise_level
        peak_density = characteristics.peak_density_estimate
        
        # For small data or very high plateau ratio, use plateau-first
        if (data_size < 10000 or plateau_ratio > 0.4):
            return 'plateau_first'
        
        # For large data with low plateau ratio, use union-find
        elif (data_size > 100000 and plateau_ratio < 0.1):
            return 'union_find'
        
        # For high plateau ratio (moderate), prefer plateau-first
        elif plateau_ratio > 0.2:
            return 'plateau_first'
        
        # For noisy data or high peak density, prefer union-find
        elif (noise_level > characteristics.value_range[1] * 0.1 or 
              peak_density > 0.01):
            return 'union_find'
        
        # Default: for medium-sized data with balanced characteristics
        # Choose based on data size
        elif data_size > 50000:
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