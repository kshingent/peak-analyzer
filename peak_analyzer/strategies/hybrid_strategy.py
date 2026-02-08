"""
Hybrid Strategy

Combines Union-Find and Plateau-First strategies for optimal performance
across different data characteristics.
"""

from typing import Any
import numpy as np

from .base_strategy import BaseStrategy, StrategyConfig
from .union_find_strategy import UnionFindStrategy
from .plateau_first_strategy import PlateauFirstStrategy
from ..api.result_dataframe import Peak
from ..core.strategy_manager import DataCharacteristics


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy that adaptively chooses between Union-Find and Plateau-First
    approaches based on local data characteristics.
    
    This strategy analyzes data regions and applies the most suitable algorithm
    for each region, then merges the results.
    """
    
    def __init__(self, config: StrategyConfig | None = None, **kwargs):
        """
        Initialize Hybrid strategy.
        
        Parameters:
        -----------
        config : StrategyConfig, optional
            Strategy configuration
        **kwargs
            Additional parameters including:
            - chunk_size: Size of data chunks for adaptive processing
            - overlap_size: Overlap between chunks to handle boundary effects
            - plateau_threshold: Threshold for plateau ratio to choose strategy
        """
        super().__init__(config, **kwargs)
        
        # Initialize component strategies
        self.union_find_strategy = UnionFindStrategy(config, **kwargs)
        self.plateau_first_strategy = PlateauFirstStrategy(config, **kwargs)
        
        # Hybrid-specific parameters
        self.chunk_size = kwargs.get('chunk_size', None)
        self.overlap_size = kwargs.get('overlap_size', 2)
        self.plateau_threshold = kwargs.get('plateau_threshold', 0.2)
        self.performance_threshold = kwargs.get('performance_threshold', 1000000)  # 1M elements
        
    def detect_peaks(self, data: np.ndarray, **params) -> list[Peak]:
        """
        Detect peaks using hybrid approach.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        **params
            Additional parameters
            
        Returns:
        --------
        list[Peak]
            List of detected peaks
        """
        # Validate input
        self.validate_input(data)
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Decide on processing approach
        if self._should_use_chunking(processed_data):
            peaks = self._detect_peaks_chunked(processed_data, **params)
        else:
            peaks = self._detect_peaks_adaptive(processed_data, **params)
        
        # Postprocess peaks
        filtered_peaks = self.postprocess_peaks(peaks, data)
        
        return filtered_peaks
    
    def _should_use_chunking(self, data: np.ndarray) -> bool:
        """
        Decide whether to use chunked processing.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        bool
            True if chunked processing should be used
        """
        # Use chunking for very large datasets
        if data.size > self.performance_threshold:
            return True
        
        # Use chunking if explicitly configured
        if self.chunk_size is not None:
            return True
        
        return False
    
    def _detect_peaks_chunked(self, data: np.ndarray, **params) -> list[Peak]:
        """
        Detect peaks using chunked processing with overlaps.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        **params
            Additional parameters
            
        Returns:
        --------
        list[Peak]
            Combined peaks from all chunks
        """
        # Determine chunk size
        if self.chunk_size is None:
            # Adaptive chunk sizing
            chunk_size = self._calculate_adaptive_chunk_size(data)
        else:
            chunk_size = self.chunk_size
        
        # Generate chunk boundaries
        chunks = self._generate_chunks(data.shape, chunk_size, self.overlap_size)
        
        all_peaks = []
        
        # Process each chunk
        for chunk_slices in chunks:
            chunk_data = data[chunk_slices]
            
            # Detect peaks in chunk using adaptive strategy
            chunk_peaks = self._detect_peaks_adaptive(chunk_data, **params)
            
            # Transform peak coordinates to global space
            global_peaks = self._transform_peaks_to_global(chunk_peaks, chunk_slices)
            
            all_peaks.extend(global_peaks)
        
        # Remove duplicate peaks from overlapping regions
        deduplicated_peaks = self._remove_duplicate_peaks(all_peaks)
        
        return deduplicated_peaks
    
    def _detect_peaks_adaptive(self, data: np.ndarray, **params) -> list[Peak]:
        """
        Detect peaks using adaptive strategy selection.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        **params
            Additional parameters
            
        Returns:
        --------
        list[Peak]
            Detected peaks
        """
        # Analyze data characteristics
        characteristics = self._analyze_local_characteristics(data)
        
        # Choose strategy based on characteristics
        if self._should_use_union_find(characteristics):
            return self.union_find_strategy.detect_peaks(data, **params)
        else:
            return self.plateau_first_strategy.detect_peaks(data, **params)
    
    def _analyze_local_characteristics(self, data: np.ndarray) -> DataCharacteristics:
        """
        Analyze characteristics of data region.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to analyze
            
        Returns:
        --------
        DataCharacteristics
            Data characteristics
        """
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
        peak_density = self._estimate_peak_density(data)
        
        return DataCharacteristics(
            shape=shape,
            ndim=ndim,
            data_type=data_type,
            value_range=value_range,
            plateau_ratio=plateau_ratio,
            noise_level=noise_level,
            peak_density_estimate=peak_density
        )
    
    def _should_use_union_find(self, characteristics: DataCharacteristics) -> bool:
        """
        Decide whether to use Union-Find strategy.
        
        Parameters:
        -----------
        characteristics : DataCharacteristics
            Data characteristics
            
        Returns:
        --------
        bool
            True if Union-Find should be used
        """
        # Use Union-Find for data with low plateau ratio
        if characteristics.plateau_ratio < self.plateau_threshold:
            return True
        
        # Use Union-Find for noisy data
        if characteristics.noise_level > np.mean(characteristics.value_range) * 0.1:
            return True
        
        # Use Union-Find for high peak density
        if characteristics.peak_density_estimate > 0.01:  # 1% of points are peaks
            return True
        
        return False
    
    def _estimate_plateau_ratio(self, data: np.ndarray) -> float:
        """Estimate percentage of data forming plateaus."""
        # Simple implementation: count cells with identical neighbors
        plateau_cells = 0
        
        # Sample points for efficiency
        if data.size > 10000:
            indices = np.random.choice(data.size, 10000, replace=False)
            sample_coords = [np.unravel_index(idx, data.shape) for idx in indices]
        else:
            # Use all points for small data
            it = np.nditer(data, flags=['multi_index'])
            sample_coords = []
            while not it.finished:
                sample_coords.append(it.multi_index)
                it.iternext()
        
        for coord in sample_coords:
            current_value = data[coord]
            neighbors = self._get_valid_neighbors(coord, data.shape)
            
            # Check if any neighbor has same value
            has_identical_neighbor = any(data[neighbor] == current_value for neighbor in neighbors)
            
            if has_identical_neighbor:
                plateau_cells += 1
        
        return plateau_cells / len(sample_coords) if sample_coords else 0.0
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level using gradient statistics."""
        gradients = np.gradient(data)
        if isinstance(gradients, list):
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        else:
            gradient_magnitude = np.abs(gradients)
        
        return float(np.std(gradient_magnitude))
    
    def _estimate_peak_density(self, data: np.ndarray) -> float:
        """Estimate density of peaks using simple local maxima count."""
        from scipy.ndimage import maximum_filter
        
        # Small sample for efficiency
        if data.size > 100000:
            # Take central region sample
            center_slices = tuple(slice(s//4, 3*s//4) for s in data.shape)
            sample_data = data[center_slices]
        else:
            sample_data = data
        
        # Count local maxima
        local_maxima = maximum_filter(sample_data, size=3) == sample_data
        core_slice = tuple(slice(1, -1) for _ in range(sample_data.ndim))
        core_maxima = local_maxima[core_slice] if sample_data.ndim > 0 else local_maxima
        
        peak_count = np.sum(core_maxima)
        core_volume = core_maxima.size if hasattr(core_maxima, 'size') else 1
        
        return peak_count / core_volume if core_volume > 0 else 0.0
    
    def _calculate_adaptive_chunk_size(self, data: np.ndarray) -> int:
        """Calculate appropriate chunk size based on data characteristics."""
        # Target chunk size in elements
        target_elements = 1000000  # 1M elements per chunk
        
        # Calculate chunk size per dimension
        elements_per_dim = int((target_elements ** (1.0 / data.ndim)))
        
        # Ensure minimum chunk size
        min_chunk_size = 64
        
        return max(min_chunk_size, elements_per_dim)
    
    def _generate_chunks(self, shape: tuple[int, ...], chunk_size: int, overlap: int) -> list[tuple[slice, ...]]:
        """Generate chunk slice tuples with overlaps."""
        chunks = []
        
        # Calculate number of chunks per dimension
        chunks_per_dim = []
        for dim_size in shape:
            num_chunks = max(1, (dim_size + chunk_size - 1) // chunk_size)
            chunks_per_dim.append(num_chunks)
        
        # Generate all combinations of chunk indices
        import itertools
        for chunk_indices in itertools.product(*[range(n) for n in chunks_per_dim]):
            chunk_slices = []
            
            for dim, (chunk_idx, dim_size) in enumerate(zip(chunk_indices, shape)):
                start = max(0, chunk_idx * chunk_size - overlap)
                end = min(dim_size, (chunk_idx + 1) * chunk_size + overlap)
                chunk_slices.append(slice(start, end))
            
            chunks.append(tuple(chunk_slices))
        
        return chunks
    
    def _transform_peaks_to_global(self, peaks: list[Peak], chunk_slices: tuple[slice, ...]) -> list[Peak]:
        """Transform peak coordinates from chunk space to global space."""
        global_peaks = []
        
        # Calculate offset for each dimension
        offsets = tuple(s.start for s in chunk_slices)
        
        for peak in peaks:
            # Transform indices
            global_indices = tuple(
                idx + offset for idx, offset in zip(peak.center_indices, offsets)
            )
            
            # Transform coordinates (assuming they follow same offset pattern)
            global_coordinates = tuple(
                coord + offset for coord, offset in zip(peak.center_coordinates, offsets)
            )
            
            # Transform plateau indices
            global_plateau_indices = [
                tuple(idx + offset for idx, offset in zip(plateau_idx, offsets))
                for plateau_idx in peak.plateau_indices
            ]
            
            global_peak = Peak(
                center_indices=global_indices,
                center_coordinates=global_coordinates,
                plateau_indices=global_plateau_indices,
                height=peak.height
            )
            
            global_peaks.append(global_peak)
        
        return global_peaks
    
    def _remove_duplicate_peaks(self, peaks: list[Peak]) -> list[Peak]:
        """Remove duplicate peaks from overlapping chunk processing."""
        if not peaks:
            return peaks
        
        # Group peaks by height and proximity
        unique_peaks = []
        tolerance = 1.0  # Distance tolerance for considering peaks as duplicates
        
        for peak in peaks:
            is_duplicate = False
            
            for existing_peak in unique_peaks:
                # Check if peaks are at similar location and height
                if abs(peak.height - existing_peak.height) < 1e-10:
                    distance = np.sqrt(sum(
                        (p1 - p2) ** 2 
                        for p1, p2 in zip(peak.center_indices, existing_peak.center_indices)
                    ))
                    
                    if distance < tolerance:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_peaks.append(peak)
        
        return unique_peaks
    
    def _get_valid_neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbors within bounds."""
        neighbors = []
        ndim = len(point)
        
        for dim in range(ndim):
            for delta in [-1, 1]:
                neighbor = list(point)
                neighbor[dim] += delta
                
                if 0 <= neighbor[dim] < shape[dim]:
                    neighbors.append(tuple(neighbor))
        
        return neighbors
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray) -> dict[Peak, dict[str, Any]]:
        """Calculate features using the most appropriate strategy."""
        # Use Union-Find strategy for feature calculation by default
        return self.union_find_strategy.calculate_features(peaks, data)
    
    @classmethod
    def estimate_performance(cls, data_shape: tuple[int, ...]) -> dict[str, float]:
        """Estimate performance for Hybrid strategy."""
        # Get estimates from component strategies
        uf_metrics = UnionFindStrategy.estimate_performance(data_shape)
        pf_metrics = PlateauFirstStrategy.estimate_performance(data_shape)
        
        # Hybrid performance is between the two, slightly higher due to analysis overhead
        estimated_time = (uf_metrics["estimated_time"] + pf_metrics["estimated_time"]) / 2 * 1.1
        estimated_memory = max(uf_metrics["estimated_memory"], pf_metrics["estimated_memory"]) * 1.05
        
        # Accuracy should be better than individual strategies
        accuracy_score = max(uf_metrics["accuracy_score"], pf_metrics["accuracy_score"]) * 1.05
        accuracy_score = min(accuracy_score, 1.0)  # Cap at 1.0
        
        # Good scalability due to adaptive approach
        scalability_factor = 0.9
        
        return {
            "estimated_time": estimated_time,
            "estimated_memory": estimated_memory,
            "accuracy_score": accuracy_score,
            "scalability_factor": scalability_factor
        }