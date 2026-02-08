"""
Data Analysis for Peak Detection

Specialized class for analyzing data characteristics to support
optimal strategy selection and processing decisions.
"""

import numpy as np

from peak_analyzer.models import DataCharacteristics


class DataAnalyzer:
    """
    Analyzes data characteristics to support optimal peak detection strategy selection.
    
    This class is responsible for extracting relevant statistical and structural
    properties from input data that inform processing decisions.
    """
    
    def __init__(self):
        """Initialize data analyzer."""
        pass
    
    def analyze_characteristics(self, data: np.ndarray) -> DataCharacteristics:
        """
        Analyze comprehensive characteristics of input data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data to analyze
            
        Returns:
        --------
        DataCharacteristics
            Comprehensive data characteristics
        """
        # Basic properties
        shape = data.shape
        ndim = data.ndim
        data_type = str(data.dtype)
        value_range = (float(data.min()), float(data.max()))
        
        # Advanced statistical analysis
        plateau_ratio = self.estimate_plateau_ratio(data)
        noise_level = self.estimate_noise_level(data)
        peak_density_estimate = self.estimate_peak_density(data)
        
        return DataCharacteristics(
            shape=shape,
            ndim=ndim,
            data_type=data_type,
            value_range=value_range,
            plateau_ratio=plateau_ratio,
            noise_level=noise_level,
            peak_density_estimate=peak_density_estimate
        )
    
    def estimate_plateau_ratio(self, data: np.ndarray) -> float:
        """
        Estimate percentage of data that forms plateaus using efficient sampling.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        float
            Plateau ratio (0.0 to 1.0)
        """
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
    
    def estimate_noise_level(self, data: np.ndarray) -> float:
        """
        Estimate noise level using gradient statistics.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        float
            Estimated noise level
        """
        gradients = np.gradient(data)
        if isinstance(gradients, list):
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        else:
            gradient_magnitude = np.abs(gradients)
            
        return float(np.std(gradient_magnitude))
    
    def estimate_peak_density(self, data: np.ndarray) -> float:
        """
        Estimate density of peaks using central region sampling for efficiency.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        float
            Estimated peak density (peaks per unit volume)
        """
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
    
    def should_use_chunking(self, data: np.ndarray, performance_threshold: int = 1000000) -> bool:
        """
        Determine if chunked processing should be used based on data size.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        performance_threshold : int
            Size threshold for chunked processing
            
        Returns:
        --------
        bool
            True if chunked processing is recommended
        """
        return data.size > performance_threshold
    
    def calculate_adaptive_chunk_size(self, data: np.ndarray) -> int:
        """
        Calculate appropriate chunk size based on data characteristics.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        int
            Recommended chunk size per dimension
        """
        # Target chunk size in elements
        target_elements = 1000000  # 1M elements per chunk
        
        # Calculate chunk size per dimension
        elements_per_dim = int((target_elements ** (1.0 / data.ndim)))
        
        # Ensure minimum chunk size
        min_chunk_size = 64
        
        return max(min_chunk_size, elements_per_dim)
    
    def _get_valid_neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Get valid neighbors within bounds for face connectivity.
        
        Parameters:
        -----------
        point : tuple of int
            Point coordinates
        shape : tuple of int
            Data shape
            
        Returns:
        --------
        list of tuple
            Valid neighbor coordinates
        """
        neighbors = []
        ndim = len(point)
        
        for dim in range(ndim):
            for delta in [-1, 1]:
                neighbor = list(point)
                neighbor[dim] += delta
                
                if 0 <= neighbor[dim] < shape[dim]:
                    neighbors.append(tuple(neighbor))
        
        return neighbors