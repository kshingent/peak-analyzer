"""
Result Data Structures for Peak Analysis

Provides data containers for peak collections and lazy feature computation.
"""

import pandas as pd
import numpy as np
from typing import Any

from peak_analyzer.models import Peak, VirtualPeak
from peak_analyzer.coordinate_system.grid_manager import GridManager
from peak_analyzer.features.lazy_feature_manager import LazyFeatureManager


class PeakCollection:
    """
    Collection of detected peaks with filtering and feature computation capabilities.
    """
    
    def __init__(
        self, 
        peaks: list[Peak], 
        data: np.ndarray,
        grid_manager: GridManager,
        **initial_filters
    ):
        """
        Initialize peak collection.
        
        Parameters:
        -----------
        peaks : list[Peak]
            List of detected peaks
        data : np.ndarray
            Original data array
        grid_manager : GridManager
            Grid manager for coordinate transformations
        **initial_filters
            Initial filter criteria to apply
        """
        self.peaks = peaks
        self.data = data
        self.grid_manager = grid_manager
        self._lazy_features = LazyFeatureManager(self)
        
        # Apply initial filters if provided
        if initial_filters:
            self.peaks = self._apply_filters(self.peaks, **initial_filters)
    
    def filter(self, **criteria) -> 'PeakCollection':
        """
        Filter peaks based on specified criteria.
        
        Parameters:
        -----------
        **criteria
            Filter criteria (e.g., prominence__gt=0.5, area__gte=10)
            Operators: __gt, __gte, __lt, __lte, __eq, __ne
            
        Returns:
        --------
        PeakCollection
            New filtered peak collection
        """
        filtered_peaks = self._apply_filters(self.peaks, **criteria)
        return PeakCollection(
            peaks=filtered_peaks,
            data=self.data, 
            grid_manager=self.grid_manager
        )
    
    def sort_by(self, feature: str, ascending: bool = True) -> 'PeakCollection':
        """
        Sort peaks by specified feature.
        
        Parameters:
        -----------
        feature : str
            Feature name to sort by
        ascending : bool
            Sort order (True for ascending, False for descending)
            
        Returns:
        --------
        PeakCollection
            New sorted peak collection
        """
        # Compute feature values for sorting
        feature_values = self._lazy_features.compute_feature(feature)
        
        # Sort indices
        sort_indices = np.argsort(feature_values)
        if not ascending:
            sort_indices = sort_indices[::-1]
            
        # Create sorted peak list
        sorted_peaks = [self.peaks[i] for i in sort_indices]
        
        return PeakCollection(
            peaks=sorted_peaks,
            data=self.data,
            grid_manager=self.grid_manager
        )
    
    def get_features(self, features: list[str | None] = None) -> 'LazyDataFrame':
        """
        Get lazy dataframe with specified features.
        
        Parameters:
        -----------
        features : list of str, optional
            List of feature names to include. If None, includes all available features.
            
        Returns:
        --------
        LazyDataFrame
            Lazy dataframe for feature access
        """
        return LazyDataFrame(self, features)
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame with all computed features.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with peak coordinates and computed features
        """
        # Start with basic peak information
        data_dict = {
            'peak_id': list(range(len(self.peaks))),
            'height': [peak.height for peak in self.peaks]
        }
        
        # Add coordinate information
        if self.peaks:
            coord_names = [f"coord_{i}" for i in range(len(self.peaks[0].center_coordinates))]
            for i, coord_name in enumerate(coord_names):
                data_dict[coord_name] = [peak.center_coordinates[i] for peak in self.peaks]
        
        # Add computed features
        computed_features = self._lazy_features.get_computed_features()
        data_dict.update(computed_features)
        
        return pd.DataFrame(data_dict)
    
    def visualize(self, backend: str = 'matplotlib', **kwargs):
        """
        Visualize peaks and their properties.
        
        Parameters:
        -----------
        backend : str
            Visualization backend ('matplotlib', 'plotly', 'bokeh')
        **kwargs
            Backend-specific visualization parameters
        """
        # Implementation will be added based on visualization requirements
        pass
    
    def _apply_filters(self, peaks: list[Peak], **criteria) -> list[Peak]:
        """Apply filtering criteria to peak list."""
        if not criteria:
            return peaks
            
        filtered_peaks = []
        for peak in peaks:
            if self._peak_matches_criteria(peak, **criteria):
                filtered_peaks.append(peak)
                
        return filtered_peaks
    
    def _peak_matches_criteria(self, peak: Peak, **criteria) -> bool:
        """Check if a peak matches all filter criteria."""
        for criterion, value in criteria.items():
            if not self._check_criterion(peak, criterion, value):
                return False
        return True
    
    def _check_criterion(self, peak: Peak, criterion: str, value: Any) -> bool:
        """Check individual filter criterion."""
        # Parse criterion (feature__operator format)
        if '__' in criterion:
            feature_name, operator = criterion.split('__', 1)
        else:
            feature_name, operator = criterion, 'eq'
            
        # Get feature value (compute if necessary)
        feature_value = self._get_feature_value(peak, feature_name)
        
        # Apply operator
        if operator == 'gt':
            return feature_value > value
        elif operator == 'gte':
            return feature_value >= value
        elif operator == 'lt':
            return feature_value < value
        elif operator == 'lte':
            return feature_value <= value
        elif operator == 'eq':
            return feature_value == value
        elif operator == 'ne':
            return feature_value != value
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _get_feature_value(self, peak: Peak, feature_name: str) -> Any:
        """Get feature value for a specific peak."""
        # Basic properties
        if feature_name == 'height':
            return peak.height
        elif feature_name.startswith('coord_'):
            coord_idx = int(feature_name.split('_')[1])
            return peak.center_coordinates[coord_idx]
        else:
            # Computed features - get from lazy feature manager
            return self._lazy_features.get_peak_feature(peak, feature_name)
    
    def __len__(self) -> int:
        """Return number of peaks in collection."""
        return len(self.peaks)
    
    def __getitem__(self, index: int) -> Peak:
        """Get peak by index."""
        return self.peaks[index]
    
    def __iter__(self):
        """Iterate over peaks."""
        return iter(self.peaks)


class LazyDataFrame:
    """
    DataFrame-like interface with lazy feature computation.
    """
    
    def __init__(self, peak_collection: PeakCollection, features: list[str | None] = None):
        """
        Initialize lazy dataframe.
        
        Parameters:
        -----------
        peak_collection : PeakCollection
            Peak collection to wrap
        features : list of str, optional
            Specific features to include
        """
        self.peak_collection = peak_collection
        self.features = features or []
        self._cache = {}
    
    def compute_feature(self, feature_name: str, **params):
        """
        Compute specific feature for all peaks.
        
        Parameters:
        -----------
        feature_name : str
            Name of feature to compute
        **params
            Parameters for feature computation
        """
        return self.peak_collection._lazy_features.compute_feature(feature_name, **params)
    
    def __getitem__(self, key: str) -> np.ndarray:
        """
        Get feature values using lazy computation.
        
        Parameters:
        -----------
        key : str
            Feature name
            
        Returns:
        --------
        np.ndarray
            Array of feature values for all peaks
        """
        if key not in self._cache:
            self._cache[key] = self.compute_feature(key)
        return self._cache[key]
    
    def cache_features(self, features: list[str]):
        """Precompute and cache specified features."""
        for feature in features:
            self.__getitem__(feature)  # Trigger computation and caching
    
    def clear_cache(self):
        """Clear feature cache."""
        self._cache = {}
    
    @property
    def columns(self) -> list[str]:
        """Get available feature columns."""
        base_columns = ['height'] + [f"coord_{i}" for i in range(len(self.peak_collection.peaks[0].center_coordinates))]
        computed_columns = self.peak_collection._lazy_features.available_features
        return base_columns + computed_columns
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return self.peak_collection.to_pandas()


class VirtualPeakCollection:
    """
    Collection of virtual peaks representing same-height connected peaks.
    """
    
    def __init__(self, virtual_peaks: list['VirtualPeak']):
        """
        Initialize virtual peak collection.
        
        Parameters:
        -----------
        virtual_peaks : list[VirtualPeak]
            List of virtual peaks
        """
        self.virtual_peaks = virtual_peaks
    
    # Methods similar to PeakCollection but for virtual peaks
    # Implementation will be added as needed