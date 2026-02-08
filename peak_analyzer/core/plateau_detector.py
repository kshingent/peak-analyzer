"""
Plateau Detection Core

Implements the core logic for detecting and validating plateau regions in multidimensional data.
"""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import maximum_filter, binary_dilation, label

from peak_analyzer.connectivity.connectivity_types import Connectivity


@dataclass
class PlateauRegion:
    """
    Represents a detected plateau region.
    """
    indices: list[tuple[int, ...]]  # All indices in plateau
    height: float                   # Constant height of plateau
    centroid: tuple[float, ...]     # Geometric centroid
    is_valid: bool                  # Whether plateau passes validation
    boundary_indices: list[tuple[int, ...]] = None


class PlateauDetector:
    """
    Detects and validates plateau regions using morphological operations.
    """
    
    def __init__(self, connectivity: int = 1):
        """
        Initialize plateau detector.
        
        Parameters:
        -----------
        connectivity : str or int
            Connectivity type for neighbor detection
        """
        self.connectivity = connectivity
        self._connectivity_structure = None
        
    def detect_plateaus(self, data: np.ndarray, connectivity: str | int | None = None) -> list[PlateauRegion]:
        """
        Detect all plateau regions in data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        connectivity : str or int, optional
            Override default connectivity
            
        Returns:
        --------
        list[PlateauRegion]
            List of detected plateau regions
        """
        connectivity = connectivity or self.connectivity
        conn = Connectivity(data.ndim, connectivity)
        self._connectivity_structure = conn.structure
        
        # Step 1: Apply local maximum filter
        local_maxima_mask = self._apply_local_maximum_filter(data)
        
        # Step 2: Find connected components of same height
        plateau_candidates = self._find_connected_components(data, local_maxima_mask)
        
        # Step 3: Validate each plateau using dilation test
        validated_plateaus = []
        for candidate in plateau_candidates:
            if self.validate_plateau(candidate, data, connectivity):
                candidate.is_valid = True
                validated_plateaus.append(candidate)
            else:
                candidate.is_valid = False
                
        return validated_plateaus
    
    def validate_plateau(self, region: PlateauRegion, data: np.ndarray, connectivity: str | int) -> bool:
        """
        Validate plateau region using dilation test.
        
        A true peak plateau should have all boundary cells strictly lower
        than the plateau height after dilation.
        
        Parameters:
        -----------
        region : PlateauRegion
            Plateau region to validate
        data : np.ndarray
            Input data array
        connectivity : str or int
            Connectivity for dilation
            
        Returns:
        --------
        bool
            True if plateau is a valid peak, False otherwise
        """
        # Create binary mask for plateau region
        plateau_mask = np.zeros(data.shape, dtype=bool)
        for idx in region.indices:
            plateau_mask[idx] = True
            
        # Apply morphological dilation
        conn = Connectivity(data.ndim, connectivity)
        dilated_mask = binary_dilation(plateau_mask, structure=conn.structure)
        
        # Find boundary (dilated - original)
        boundary_mask = dilated_mask & (~plateau_mask)
        
        # Get boundary indices
        boundary_indices = list(zip(*np.where(boundary_mask)))
        region.boundary_indices = boundary_indices
        
        # Check if all boundary cells are strictly lower
        plateau_height = region.height
        for boundary_idx in boundary_indices:
            if data[boundary_idx] >= plateau_height:
                return False  # Found boundary cell with same or higher height
                
        return True  # All boundary cells are strictly lower
    
    def merge_connected_plateaus(self, plateaus: list[PlateauRegion]) -> list[PlateauRegion]:
        """
        Merge plateaus that are connected and at same height.
        
        Parameters:
        -----------
        plateaus : list[PlateauRegion]
            List of plateau regions to potentially merge
            
        Returns:
        --------
        list[PlateauRegion]
            List with merged plateaus
        """
        if not plateaus:
            return []
            
        # Group plateaus by height
        height_groups = {}
        for plateau in plateaus:
            height = plateau.height
            if height not in height_groups:
                height_groups[height] = []
            height_groups[height].append(plateau)
            
        merged_plateaus = []
        
        # Process each height group
        for height, same_height_plateaus in height_groups.items():
            if len(same_height_plateaus) == 1:
                merged_plateaus.extend(same_height_plateaus)
            else:
                # Find connected components among same-height plateaus
                merged = self._merge_connected_same_height(same_height_plateaus)
                merged_plateaus.extend(merged)
                
        return merged_plateaus
    
    def filter_noise_plateaus(self, plateaus: list[PlateauRegion], min_area: float) -> list[PlateauRegion]:
        """
        Filter out small plateaus likely to be noise.
        
        Parameters:
        -----------
        plateaus : list[PlateauRegion]
            List of plateau regions
        min_area : float
            Minimum area (number of cells) for valid plateau
            
        Returns:
        --------
        list[PlateauRegion]
            Filtered list of plateaus
        """
        filtered = []
        for plateau in plateaus:
            if len(plateau.indices) >= min_area:
                filtered.append(plateau)
                
        return filtered
    
    def _apply_local_maximum_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply local maximum filter to identify potential peak cells.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray
            Boolean mask of potential peak cells
        """
        # Apply maximum filter with connectivity structure
        max_filtered = maximum_filter(data, footprint=self._connectivity_structure)
        
        # Local maxima are cells equal to their filtered value
        local_maxima = (data == max_filtered)
        
        return local_maxima
    
    def _find_connected_components(self, data: np.ndarray, mask: np.ndarray) -> list[PlateauRegion]:
        """
        Find connected components of same height within mask.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        mask : np.ndarray
            Boolean mask of candidate cells
            
        Returns:
        --------
        list[PlateauRegion]
            List of plateau regions
        """
        plateaus = []
        
        # Get unique heights in masked region
        masked_data = data * mask
        unique_heights = np.unique(masked_data[mask])
        
        for height in unique_heights:
            if height == 0 and not mask.all():
                continue  # Skip background
                
            # Create mask for current height
            height_mask = (data == height) & mask
            
            if not height_mask.any():
                continue
                
            # Find connected components at this height
            labeled_array, num_features = label(height_mask, structure=self._connectivity_structure)
            
            for component_id in range(1, num_features + 1):
                component_mask = (labeled_array == component_id)
                indices = list(zip(*np.where(component_mask)))
                
                # Calculate centroid
                centroid = tuple(np.mean([list(idx) for idx in indices], axis=0))
                
                plateau = PlateauRegion(
                    indices=indices,
                    height=float(height),
                    centroid=centroid,
                    is_valid=False  # Will be determined during validation
                )
                
                plateaus.append(plateau)
                
        return plateaus
    
    def _merge_connected_same_height(self, plateaus: list[PlateauRegion]) -> list[PlateauRegion]:
        """
        Merge plateaus of same height that are spatially connected.
        
        Parameters:
        -----------
        plateaus : list[PlateauRegion]
            Plateaus at same height to potentially merge
            
        Returns:
        --------
        list[PlateauRegion]
            Merged plateaus
        """
        if len(plateaus) <= 1:
            return plateaus
            
        # Build adjacency graph
        adjacency = {}
        for i, plateau_i in enumerate(plateaus):
            adjacency[i] = set()
            for j, plateau_j in enumerate(plateaus):
                if i != j and self._are_plateaus_adjacent(plateau_i, plateau_j):
                    adjacency[i].add(j)
                    
        # Find connected components in adjacency graph
        visited = set()
        merged_groups = []
        
        for start_idx in range(len(plateaus)):
            if start_idx not in visited:
                group = self._dfs_adjacency(start_idx, adjacency, visited)
                merged_groups.append(group)
                
        # Create merged plateau for each group
        merged_plateaus = []
        for group in merged_groups:
            if len(group) == 1:
                merged_plateaus.append(plateaus[group[0]])
            else:
                # Merge plateaus in group
                merged_plateau = self._create_merged_plateau([plateaus[i] for i in group])
                merged_plateaus.append(merged_plateau)
                
        return merged_plateaus
    
    def _are_plateaus_adjacent(self, plateau1: PlateauRegion, plateau2: PlateauRegion) -> bool:
        """
        Check if two plateaus are spatially adjacent.
        
        Parameters:
        -----------
        plateau1, plateau2 : PlateauRegion
            Plateaus to check adjacency
            
        Returns:
        --------
        bool
            True if plateaus are adjacent
        """
        indices1 = set(plateau1.indices)
        
        # Check if any point in plateau2 is adjacent to any point in plateau1
        for idx2 in plateau2.indices:
            # Generate neighbors of idx2
            neighbors = self._get_neighbors(idx2, self._connectivity_structure.shape)
            
            # Check if any neighbor is in plateau1
            for neighbor in neighbors:
                if neighbor in indices1:
                    return True
                    
        return False
    
    def _get_neighbors(self, index: tuple[int, ...], data_shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Get valid neighbors of a given index."""
        neighbors = []
        ndim = len(index)
        
        # Generate neighbor offsets from connectivity structure
        # This is a simplified version - should use actual connectivity structure
        for dim in range(ndim):
            for direction in [-1, 1]:
                neighbor = list(index)
                neighbor[dim] += direction
                
                # Check bounds
                if 0 <= neighbor[dim] < data_shape[dim]:
                    neighbors.append(tuple(neighbor))
                    
        return neighbors
    
    def _dfs_adjacency(self, start_idx: int, adjacency: dict, visited: set) -> list[int]:
        """Depth-first search on adjacency graph."""
        stack = [start_idx]
        group = []
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                group.append(current)
                
                # Add unvisited neighbors to stack
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                        
        return group
    
    def _create_merged_plateau(self, plateaus: list[PlateauRegion]) -> PlateauRegion:
        """Create a single plateau from multiple connected plateaus."""
        # Combine all indices
        all_indices = []
        for plateau in plateaus:
            all_indices.extend(plateau.indices)
            
        # Use height from first plateau (they should all be the same)
        height = plateaus[0].height
        
        # Recalculate centroid
        centroid = tuple(np.mean([list(idx) for idx in all_indices], axis=0))
        
        # Check validity - if any component was valid, merged is valid
        is_valid = any(plateau.is_valid for plateau in plateaus)
        
        return PlateauRegion(
            indices=all_indices,
            height=height,
            centroid=centroid,
            is_valid=is_valid
        )