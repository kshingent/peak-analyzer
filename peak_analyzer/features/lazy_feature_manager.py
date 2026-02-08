"""
Lazy Feature Manager

Manages lazy evaluation of peak features with caching, selective computation,
and efficient memory usage for large datasets.
"""

from typing import Any, Callable
import numpy as np
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .geometric_calculator import GeometricCalculator
from .topographic_calculator import TopographicCalculator
from .morphological_calculator import MorphologicalCalculator
from .distance_calculator import DistanceCalculator
from peak_analyzer.models import Peak


class LazyFeatureManager:
    """
    Manager for lazy evaluation of peak features.
    
    Features are computed only when requested and cached for subsequent access.
    Supports selective computation, memory management, and parallel processing.
    """
    
    def __init__(self, 
                 peaks: list[Peak], 
                 data: np.ndarray,
                 scale: list[float | None] = None,
                 cache_size: int = 1000,
                 enable_parallel: bool = True,
                 max_workers: int | None = None):
        """
        Initialize lazy feature manager.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to compute features for
        data : np.ndarray
            Original data array
        scale : list of float, optional
            Physical scale for each dimension
        cache_size : int
            Maximum number of cached feature sets
        enable_parallel : bool
            Enable parallel computation
        max_workers : int, optional
            Maximum number of worker threads
        """
        self.peaks = peaks
        self.data = data
        self.scale = scale
        self.cache_size = cache_size
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Initialize calculators
        self.calculators = {
            'geometric': GeometricCalculator(scale),
            'topographic': TopographicCalculator(scale),
            'morphological': MorphologicalCalculator(scale),
            'distance': DistanceCalculator(scale)
        }
        
        # Feature cache
        self._feature_cache = {}
        self._cache_lock = threading.Lock()
        
        # Weak references to peaks for memory efficiency
        self._peak_refs = {id(peak): weakref.ref(peak) for peak in peaks}
        
        # Computation status tracking
        self._computation_status = {}
        self._computation_lock = threading.Lock()
    
    def get_features(
            self, 
            peak: Peak | int | list[Peak | int],
            feature_types: list[str | None] = None,
            specific_features: list[str | None] = None,
            **kwargs) -> dict[str, Any]:
        """
        Get features for one or more peaks.
        
        Parameters:
        -----------
        peak : Peak, int, or list
            Peak(s) to get features for (Peak object, index, or list of either)
        feature_types : list of str, optional
            Types of features to compute ('geometric', 'topographic', 'morphological', 'distance')
        specific_features : list of str, optional
            Specific feature names to compute
        **kwargs
            Additional parameters passed to calculators
            
        Returns:
        --------
        dict[str, Any]
            Features for the requested peaks
        """
        # Normalize input to list of Peak objects
        if isinstance(peak, (Peak, int)):
            peaks_to_process = [peak]
        else:
            peaks_to_process = peak
        
        peak_objects = []
        for p in peaks_to_process:
            if isinstance(p, int):
                peak_objects.append(self.peaks[p])
            else:
                peak_objects.append(p)
        
        # Default to all feature types if not specified
        if feature_types is None:
            feature_types = list(self.calculators.keys())
        
        # Compute features
        if len(peak_objects) == 1:
            return self._get_single_peak_features(peak_objects[0], feature_types, specific_features, **kwargs)
        else:
            return self._get_multiple_peak_features(peak_objects, feature_types, specific_features, **kwargs)
    
    def precompute_features(self, 
                           feature_types: list[str | None] = None,
                           batch_size: int = 10,
                           progress_callback: Callable | None = None) -> None:
        """
        Precompute features for all peaks.
        
        Parameters:
        -----------
        feature_types : list of str, optional
            Types of features to precompute
        batch_size : int
            Number of peaks to process in each batch
        progress_callback : callable, optional
            Callback function for progress updates
        """
        if feature_types is None:
            feature_types = list(self.calculators.keys())
        
        total_peaks = len(self.peaks)
        
        # Process in batches
        for batch_start in range(0, total_peaks, batch_size):
            batch_end = min(batch_start + batch_size, total_peaks)
            batch_peaks = self.peaks[batch_start:batch_end]
            
            # Compute features for batch
            self._compute_batch_features(batch_peaks, feature_types)
            
            # Progress callback
            if progress_callback:
                progress = batch_end / total_peaks
                progress_callback(progress, batch_end, total_peaks)
    
    def get_available_features(self, feature_type: str) -> list[str]:
        """
        Get list of available features for a calculator type.
        
        Parameters:
        -----------
        feature_type : str
            Calculator type ('geometric', 'topographic', etc.)
            
        Returns:
        --------
        list[str]
            Available feature names
        """
        if feature_type in self.calculators:
            return self.calculators[feature_type].get_available_features()
        else:
            return []
    
    def clear_cache(self, peak: Peak | None = None) -> None:
        """
        Clear feature cache.
        
        Parameters:
        -----------
        peak : Peak, optional
            Specific peak to clear cache for (if None, clears all)
        """
        with self._cache_lock:
            if peak is None:
                self._feature_cache.clear()
            else:
                peak_id = id(peak)
                keys_to_remove = [key for key in self._feature_cache.keys() if key[0] == peak_id]
                for key in keys_to_remove:
                    del self._feature_cache[key]
    
    def get_cache_statistics(self) -> dict[str, Any]:
        """
        Get cache usage statistics.
        
        Returns:
        --------
        dict[str, Any]
            Cache statistics
        """
        with self._cache_lock:
            cache_size = len(self._feature_cache)
            
            # Count features by type
            feature_type_counts = {}
            for (peak_id, feature_type, feature_name) in self._feature_cache.keys():
                if feature_type not in feature_type_counts:
                    feature_type_counts[feature_type] = 0
                feature_type_counts[feature_type] += 1
            
            return {
                'cached_features': cache_size,
                'max_cache_size': self.cache_size,
                'feature_type_counts': feature_type_counts,
                'cache_usage_ratio': cache_size / self.cache_size if self.cache_size > 0 else 0.0
            }
    
    def memory_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        # Clean up weak references
        dead_refs = []
        for peak_id, ref in self._peak_refs.items():
            if ref() is None:
                dead_refs.append(peak_id)
        
        for peak_id in dead_refs:
            del self._peak_refs[peak_id]
            # Remove cache entries for dead peaks
            with self._cache_lock:
                keys_to_remove = [key for key in self._feature_cache.keys() if key[0] == peak_id]
                for key in keys_to_remove:
                    del self._feature_cache[key]
    
    # Private methods
    
    def _get_single_peak_features(self, 
                                 peak: Peak, 
                                 feature_types: list[str],
                                 specific_features: list[str | None] = None,
                                 **kwargs) -> dict[str, Any]:
        """Get features for a single peak."""
        result = {}
        peak_id = id(peak)
        
        for feature_type in feature_types:
            if feature_type not in self.calculators:
                continue
            
            calculator = self.calculators[feature_type]
            
            if specific_features is None:
                # Get all features for this type
                available_features = calculator.get_available_features()
            else:
                # Filter to requested features
                available_features = [f for f in specific_features 
                                    if f in calculator.get_available_features()]
            
            type_features = {}
            
            for feature_name in available_features:
                # Check cache first
                cache_key = (peak_id, feature_type, feature_name)
                
                with self._cache_lock:
                    if cache_key in self._feature_cache:
                        type_features[feature_name] = self._feature_cache[cache_key]
                        continue
                
                # Compute feature if not cached
                feature_value = self._compute_single_feature(
                    peak, feature_type, feature_name, **kwargs
                )
                
                # Cache the result
                self._cache_feature(cache_key, feature_value)
                type_features[feature_name] = feature_value
            
            result[feature_type] = type_features
        
        return result
    
    def _get_multiple_peak_features(self, 
                                   peaks: list[Peak], 
                                   feature_types: list[str],
                                   specific_features: list[str | None] = None,
                                   **kwargs) -> dict[Peak, dict[str, Any]]:
        """Get features for multiple peaks."""
        if self.enable_parallel and len(peaks) > 1:
            return self._compute_parallel_features(peaks, feature_types, specific_features, **kwargs)
        else:
            return self._compute_sequential_features(peaks, feature_types, specific_features, **kwargs)
    
    def _compute_parallel_features(self, 
                                  peaks: list[Peak], 
                                  feature_types: list[str],
                                  specific_features: list[str | None] = None,
                                  **kwargs) -> dict[Peak, dict[str, Any]]:
        """Compute features in parallel."""
        result = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_peak = {
                executor.submit(
                    self._get_single_peak_features, 
                    peak, 
                    feature_types, 
                    specific_features, 
                    **kwargs
                ): peak for peak in peaks
            }
            
            # Collect results
            for future in as_completed(future_to_peak):
                peak = future_to_peak[future]
                try:
                    peak_features = future.result()
                    result[peak] = peak_features
                except Exception as e:
                    # Log error and continue
                    print(f"Error computing features for peak {id(peak)}: {e}")
                    result[peak] = {}
        
        return result
    
    def _compute_sequential_features(self, 
                                    peaks: list[Peak], 
                                    feature_types: list[str],
                                    specific_features: list[str | None] = None,
                                    **kwargs) -> dict[Peak, dict[str, Any]]:
        """Compute features sequentially."""
        result = {}
        
        for peak in peaks:
            try:
                peak_features = self._get_single_peak_features(
                    peak, feature_types, specific_features, **kwargs
                )
                result[peak] = peak_features
            except Exception as e:
                print(f"Error computing features for peak {id(peak)}: {e}")
                result[peak] = {}
        
        return result
    
    def _compute_single_feature(self, 
                               peak: Peak, 
                               feature_type: str, 
                               feature_name: str,
                               **kwargs) -> Any:
        """Compute a single feature for a peak."""
        calculator = self.calculators[feature_type]
        
        # Some features require all peaks (e.g., distance calculations)
        if feature_type == 'distance' or 'isolation' in feature_name:
            all_features = calculator.calculate_features(self.peaks, self.data, **kwargs)
            return all_features[peak].get(feature_name, None)
        else:
            # Features that can be computed independently
            peak_features = calculator.calculate_features([peak], self.data, **kwargs)
            return peak_features[peak].get(feature_name, None)
    
    def _compute_batch_features(self, peaks: list[Peak], feature_types: list[str]) -> None:
        """Compute features for a batch of peaks."""
        for feature_type in feature_types:
            if feature_type not in self.calculators:
                continue
            
            calculator = self.calculators[feature_type]
            
            try:
                # Compute all features for this type and batch
                batch_features = calculator.calculate_features(peaks, self.data)
                
                # Cache results
                for peak in peaks:
                    peak_id = id(peak)
                    if peak in batch_features:
                        for feature_name, feature_value in batch_features[peak].items():
                            cache_key = (peak_id, feature_type, feature_name)
                            self._cache_feature(cache_key, feature_value)
            
            except Exception as e:
                print(f"Error computing {feature_type} features for batch: {e}")
    
    def _cache_feature(self, cache_key: tuple, feature_value: Any) -> None:
        """Cache a feature value with size management."""
        with self._cache_lock:
            # Add to cache
            self._feature_cache[cache_key] = feature_value
            
            # Manage cache size
            if len(self._feature_cache) > self.cache_size:
                # Remove oldest entries (simple LRU approximation)
                num_to_remove = len(self._feature_cache) - self.cache_size + 1
                keys_to_remove = list(self._feature_cache.keys())[:num_to_remove]
                
                for key in keys_to_remove:
                    del self._feature_cache[key]
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.memory_cleanup()


class FeatureRequest:
    """
    Represents a feature computation request for lazy evaluation.
    """
    
    def __init__(self, 
                 peak: Peak,
                 feature_type: str,
                 feature_name: str,
                 priority: int = 0,
                 **kwargs):
        """
        Initialize feature request.
        
        Parameters:
        -----------
        peak : Peak
            Peak to compute feature for
        feature_type : str
            Type of feature calculator
        feature_name : str
            Name of specific feature
        priority : int
            Request priority (higher = more urgent)
        **kwargs
            Additional computation parameters
        """
        self.peak = peak
        self.feature_type = feature_type
        self.feature_name = feature_name
        self.priority = priority
        self.kwargs = kwargs
        self.timestamp = threading.current_thread().ident
    
    def __lt__(self, other):
        """Support priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


class AsyncFeatureComputer:
    """
    Asynchronous feature computer for background processing.
    """
    
    def __init__(self, feature_manager: LazyFeatureManager):
        """Initialize async computer."""
        self.feature_manager = feature_manager
        self.request_queue = []
        self.is_processing = False
        self.processing_thread = None
    
    def submit_request(self, request: FeatureRequest) -> None:
        """Submit a feature computation request."""
        # Add to priority queue
        import heapq
        heapq.heappush(self.request_queue, request)
        
        # Start processing if not already running
        if not self.is_processing:
            self._start_processing()
    
    def _start_processing(self) -> None:
        """Start background processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_requests)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def _process_requests(self) -> None:
        """Process requests from the queue."""
        import heapq
        
        self.is_processing = True
        
        try:
            while self.request_queue:
                request = heapq.heappop(self.request_queue)
                
                try:
                    self.feature_manager._compute_single_feature(
                        request.peak,
                        request.feature_type,
                        request.feature_name,
                        **request.kwargs
                    )
                except Exception as e:
                    print(f"Error processing async feature request: {e}")
        
        finally:
            self.is_processing = False