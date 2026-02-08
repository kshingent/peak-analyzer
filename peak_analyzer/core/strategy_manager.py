"""
Strategy Selection and Management

Handles automatic selection of optimal peak detection strategies based on data characteristics.
"""

import numpy as np

from peak_analyzer.models import DataCharacteristics, BenchmarkResults
from peak_analyzer.strategies.base_strategy import BaseStrategy
from peak_analyzer.strategies.union_find_strategy import UnionFindStrategy
from peak_analyzer.strategies.plateau_first_strategy import PlateauFirstStrategy
from peak_analyzer.strategies.hybrid_strategy import HybridStrategy


class StrategyManager:
    """
    Manages selection and benchmarking of peak detection strategies.
    """
    
    def __init__(self):
        """Initialize strategy manager."""
        self.available_strategies = {
            'union_find': UnionFindStrategy,
            'plateau_first': PlateauFirstStrategy,
            'hybrid': HybridStrategy
        }
        self.strategy_cache = {}
        
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
        # Analyze data characteristics
        characteristics = self._analyze_data_characteristics(data)
        
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
    
    def _analyze_data_characteristics(self, data: np.ndarray) -> DataCharacteristics:
        """Analyze characteristics of input data."""
        # Basic properties
        shape = data.shape
        ndim = data.ndim
        data_type = str(data.dtype)
        value_range = (float(data.min()), float(data.max()))
        
        # Estimate plateau ratio
        plateau_ratio = self._estimate_plateau_ratio(data)
        
        # Estimate noise level
        noise_level = self._estimate_noise_level(data)
        
        # Estimate peak density
        peak_density_estimate = self._estimate_peak_density(data)
        
        return DataCharacteristics(
            shape=shape,
            ndim=ndim,
            data_type=data_type,
            value_range=value_range,
            plateau_ratio=plateau_ratio,
            noise_level=noise_level,
            peak_density_estimate=peak_density_estimate
        )
    
    def _estimate_plateau_ratio(self, data: np.ndarray) -> float:
        """Estimate percentage of data that forms plateaus."""
        # Simple implementation: count cells with identical neighbors
        plateau_cells = 0
        total_cells = data.size
        
        # Generate neighbor offsets for face connectivity
        offsets = self._get_face_connectivity_offsets(data.ndim)
        
        # Count cells with at least one identical neighbor
        it = np.nditer(data, flags=['multi_index'])
        while not it.finished:
            current_value = it[0]
            current_index = it.multi_index
            
            has_identical_neighbor = False
            for offset in offsets:
                neighbor_index = tuple(
                    current_index[i] + offset[i] for i in range(data.ndim)
                )
                
                # Check bounds
                if all(0 <= neighbor_index[i] < data.shape[i] for i in range(data.ndim)):
                    if data[neighbor_index] == current_value:
                        has_identical_neighbor = True
                        break
                        
            if has_identical_neighbor:
                plateau_cells += 1
                
            it.iternext()
            
        return plateau_cells / total_cells if total_cells > 0 else 0.0
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level in data."""
        # Simple implementation: standard deviation of gradient magnitude
        gradients = np.gradient(data)
        if isinstance(gradients, list):
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        else:
            gradient_magnitude = np.abs(gradients)
            
        return float(np.std(gradient_magnitude))
    
    def _estimate_peak_density(self, data: np.ndarray) -> float:
        """Estimate density of peaks in data."""
        # Simple implementation: count local maxima
        from scipy.ndimage import maximum_filter
        
        # Apply local maximum filter
        local_maxima = maximum_filter(data, size=3) == data
        
        # Count local maxima (excluding edges)
        core_slice = tuple(slice(1, -1) for _ in range(data.ndim))
        core_maxima = local_maxima[core_slice]
        
        peak_count = np.sum(core_maxima)
        core_volume = core_maxima.size
        
        return peak_count / core_volume if core_volume > 0 else 0.0
    
    def _choose_strategy(self, characteristics: DataCharacteristics, **kwargs) -> str:
        """Choose optimal strategy based on data characteristics."""
        # Simple heuristic-based selection
        
        # For small data or high plateau ratio, use plateau-first
        if (np.prod(characteristics.shape) < 10000 or  
            characteristics.plateau_ratio > 0.3):
            return 'plateau_first'
        
        # For large data with low plateau ratio, use union-find
        elif (np.prod(characteristics.shape) > 100000 and 
              characteristics.plateau_ratio < 0.1):
            return 'union_find'
        
        # Default to hybrid strategy
        else:
            return 'hybrid'
    
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
    
    def _get_face_connectivity_offsets(self, ndim: int) -> list[tuple[int, ...]]:
        """Get offsets for face connectivity in N dimensions."""
        offsets = []
        for dim in range(ndim):
            for direction in [-1, 1]:
                offset = [0] * ndim
                offset[dim] = direction
                offsets.append(tuple(offset))
        return offsets