"""
Morphological Feature Calculator

Calculates morphological properties of peaks using mathematical morphology operations
including erosion, dilation, opening, closing, and watershed transforms.
"""

from typing import Any
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, binary_erosion, binary_dilation
from skimage.morphology import watershed, disk, ball
from skimage.segmentation import find_boundaries

from .base_calculator import BaseCalculator
from ..models import Peak


class MorphologicalCalculator(BaseCalculator):
    """
    Calculator for morphological features of peaks.
    
    Uses mathematical morphology operations to analyze peak shape,
    connectivity patterns, and structural characteristics.
    """
    
    def __init__(self, scale: list[float | None] = None):
        """
        Initialize morphological calculator.
        
        Parameters:
        -----------
        scale : list of float, optional
            Physical scale for each dimension
        """
        super().__init__(scale)
    
    def calculate_features(self, peaks: list[Peak], data: np.ndarray, **kwargs) -> dict[Peak, dict[str, Any]]:
        """
        Calculate morphological features for peaks.
        
        Parameters:
        -----------
        peaks : list[Peak]
            Peaks to calculate features for
        data : np.ndarray
            Original data array
        **kwargs
            Additional parameters including:
            - kernel_size: Size of morphological kernel
            - watershed_markers: Number of watershed markers
            
        Returns:
        --------
        dict[Peak, dict[str, Any]]
            Morphological features for each peak
        """
        self.validate_inputs(peaks, data)
        
        kernel_size = kwargs.get('kernel_size', 3)
        watershed_markers = kwargs.get('watershed_markers', 10)
        
        features = {}
        
        for peak in peaks:
            peak_features = {}
            
            # Create peak mask
            peak_mask = self._create_peak_mask(peak, data.shape)
            
            # Basic morphological operations
            peak_features['erosion_resistance'] = self.calculate_erosion_resistance(peak_mask, kernel_size)
            peak_features['dilation_expansion'] = self.calculate_dilation_expansion(peak_mask, kernel_size)
            peak_features['opening_response'] = self.calculate_opening_response(peak_mask, kernel_size)
            peak_features['closing_response'] = self.calculate_closing_response(peak_mask, kernel_size)
            
            # Shape characterization
            peak_features['convex_hull_ratio'] = self.calculate_convex_hull_ratio(peak_mask)
            peak_features['solidity'] = self.calculate_solidity(peak_mask)
            peak_features['concavity_analysis'] = self.calculate_concavity_analysis(peak_mask)
            
            # Connectivity analysis
            peak_features['connectivity_pattern'] = self.analyze_connectivity_pattern(peak_mask)
            peak_features['skeleton_analysis'] = self.analyze_skeleton(peak_mask)
            
            # Watershed analysis
            peak_features['watershed_properties'] = self.analyze_watershed_properties(
                peak, data, num_markers=watershed_markers
            )
            
            # Topological features
            peak_features['euler_number'] = self.calculate_euler_number(peak_mask)
            peak_features['boundary_complexity'] = self.calculate_boundary_complexity(peak_mask)
            
            features[peak] = peak_features
        
        return features
    
    def get_available_features(self) -> list[str]:
        """Get list of available morphological features."""
        return [
            'erosion_resistance',
            'dilation_expansion', 
            'opening_response',
            'closing_response',
            'convex_hull_ratio',
            'solidity',
            'concavity_analysis',
            'connectivity_pattern',
            'skeleton_analysis',
            'watershed_properties',
            'euler_number',
            'boundary_complexity'
        ]
    
    def calculate_erosion_resistance(self, mask: np.ndarray, kernel_size: int) -> dict[str, float]:
        """
        Calculate resistance to morphological erosion.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
        kernel_size : int
            Size of erosion kernel
            
        Returns:
        --------
        dict[str, float]
            Erosion resistance measures
        """
        original_area = np.sum(mask)
        
        if original_area == 0:
            return {'resistance_ratio': 0.0, 'iterations_to_disappear': 0}
        
        # Create structuring element
        if mask.ndim == 2:
            kernel = disk(kernel_size)
        elif mask.ndim == 3:
            kernel = ball(kernel_size)
        else:
            # For higher dimensions, use cube
            kernel = np.ones([kernel_size * 2 + 1] * mask.ndim)
        
        # Perform erosion
        eroded_mask = binary_erosion(mask, kernel)
        remaining_area = np.sum(eroded_mask)
        
        resistance_ratio = remaining_area / original_area if original_area > 0 else 0.0
        
        # Count iterations until disappearance
        iterations = 0
        current_mask = mask.copy()
        small_kernel = np.ones([3] * mask.ndim)  # Small kernel for iteration counting
        
        while np.sum(current_mask) > 0 and iterations < 50:  # Limit iterations
            current_mask = binary_erosion(current_mask, small_kernel)
            iterations += 1
        
        return {
            'resistance_ratio': resistance_ratio,
            'iterations_to_disappear': iterations
        }
    
    def calculate_dilation_expansion(self, mask: np.ndarray, kernel_size: int) -> dict[str, float]:
        """
        Calculate dilation expansion characteristics.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
        kernel_size : int
            Size of dilation kernel
            
        Returns:
        --------
        dict[str, float]
            Dilation expansion measures
        """
        original_area = np.sum(mask)
        
        # Create structuring element
        if mask.ndim == 2:
            kernel = disk(kernel_size)
        elif mask.ndim == 3:
            kernel = ball(kernel_size)
        else:
            kernel = np.ones([kernel_size * 2 + 1] * mask.ndim)
        
        # Perform dilation
        dilated_mask = binary_dilation(mask, kernel)
        expanded_area = np.sum(dilated_mask)
        
        expansion_ratio = expanded_area / original_area if original_area > 0 else 0.0
        expansion_factor = expanded_area - original_area
        
        return {
            'expansion_ratio': expansion_ratio,
            'expansion_factor': expansion_factor
        }
    
    def calculate_opening_response(self, mask: np.ndarray, kernel_size: int) -> dict[str, float]:
        """
        Calculate morphological opening response.
        
        Opening removes small objects and smooths boundaries.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
        kernel_size : int
            Size of opening kernel
            
        Returns:
        --------
        dict[str, float]
            Opening response measures
        """
        original_area = np.sum(mask)
        
        # Create structuring element
        if mask.ndim == 2:
            kernel = disk(kernel_size)
        elif mask.ndim == 3:
            kernel = ball(kernel_size)
        else:
            kernel = np.ones([kernel_size * 2 + 1] * mask.ndim)
        
        # Perform opening (erosion followed by dilation)
        opened_mask = binary_dilation(binary_erosion(mask, kernel), kernel)
        remaining_area = np.sum(opened_mask)
        
        opening_ratio = remaining_area / original_area if original_area > 0 else 0.0
        smoothing_effect = original_area - remaining_area
        
        return {
            'opening_ratio': opening_ratio,
            'smoothing_effect': smoothing_effect
        }
    
    def calculate_closing_response(self, mask: np.ndarray, kernel_size: int) -> dict[str, float]:
        """
        Calculate morphological closing response.
        
        Closing fills small holes and connects nearby objects.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
        kernel_size : int
            Size of closing kernel
            
        Returns:
        --------
        dict[str, float]
            Closing response measures
        """
        original_area = np.sum(mask)
        
        # Create structuring element
        if mask.ndim == 2:
            kernel = disk(kernel_size)
        elif mask.ndim == 3:
            kernel = ball(kernel_size)
        else:
            kernel = np.ones([kernel_size * 2 + 1] * mask.ndim)
        
        # Perform closing (dilation followed by erosion)
        closed_mask = binary_erosion(binary_dilation(mask, kernel), kernel)
        final_area = np.sum(closed_mask)
        
        closing_ratio = final_area / original_area if original_area > 0 else 0.0
        filling_effect = final_area - original_area
        
        return {
            'closing_ratio': closing_ratio,
            'filling_effect': filling_effect
        }
    
    def calculate_convex_hull_ratio(self, mask: np.ndarray) -> float:
        """
        Calculate ratio of object area to its convex hull area.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        float
            Convex hull ratio (0 to 1)
        """
        if np.sum(mask) == 0:
            return 0.0
        
        # For 2D, use scipy's convex hull
        if mask.ndim == 2:
            try:
                # Get coordinates of mask points
                coords = np.column_stack(np.where(mask))
                
                if len(coords) < 3:  # Need at least 3 points for convex hull
                    return 1.0
            
                
                # Create convex hull mask
                hull_mask = np.zeros_like(mask)
                
                # Fill convex hull region (simplified approximation)
                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)
                
                # Bounding box approximation
                hull_mask[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1] = 1
                
                original_area = np.sum(mask)
                hull_area = np.sum(hull_mask)
                
                return original_area / hull_area if hull_area > 0 else 0.0
                
            except ImportError:
                # Fallback to bounding box ratio
                coords = np.column_stack(np.where(mask))
                if len(coords) == 0:
                    return 0.0
                
                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)
                
                bbox_area = np.prod(max_coords - min_coords + 1)
                original_area = np.sum(mask)
                
                return original_area / bbox_area if bbox_area > 0 else 0.0
        
        else:
            # For higher dimensions, use bounding box approximation
            coords = np.column_stack([np.where(mask)[i] for i in range(mask.ndim)])
            
            if len(coords) == 0:
                return 0.0
            
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            
            bbox_volume = np.prod(max_coords - min_coords + 1)
            original_volume = np.sum(mask)
            
            return original_volume / bbox_volume if bbox_volume > 0 else 0.0
    
    def calculate_solidity(self, mask: np.ndarray) -> float:
        """
        Calculate solidity (ratio of area to convex hull area).
        
        This is similar to convex_hull_ratio but uses a different calculation method.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        float
            Solidity measure
        """
        return self.calculate_convex_hull_ratio(mask)
    
    def calculate_concavity_analysis(self, mask: np.ndarray) -> dict[str, float]:
        """
        Analyze concavity features of the peak shape.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        dict[str, float]
            Concavity measures
        """
        if np.sum(mask) == 0:
            return {'concavity_index': 0.0, 'concave_points': 0}
        
        # Calculate boundary
        boundary = find_boundaries(mask, mode='inner')
        boundary_coords = np.column_stack(np.where(boundary))
        
        if len(boundary_coords) < 3:
            return {'concavity_index': 0.0, 'concave_points': 0}
        
        # For 2D, analyze boundary curvature
        if mask.ndim == 2:
            concave_points = 0
            total_points = len(boundary_coords)
            
            # Simple curvature analysis
            for i in range(1, total_points - 1):
                # Calculate vectors to previous and next points
                prev_vec = boundary_coords[i] - boundary_coords[i-1]
                next_vec = boundary_coords[i+1] - boundary_coords[i]
                
                # Cross product indicates curvature direction
                if len(prev_vec) >= 2 and len(next_vec) >= 2:
                    cross_product = prev_vec[0] * next_vec[1] - prev_vec[1] * next_vec[0]
                    if cross_product < 0:  # Concave point
                        concave_points += 1
            
            concavity_index = concave_points / total_points if total_points > 0 else 0.0
            
            return {
                'concavity_index': concavity_index,
                'concave_points': concave_points
            }
        
        else:
            # For higher dimensions, use simpler measures
            return {
                'concavity_index': 1.0 - self.calculate_convex_hull_ratio(mask),
                'concave_points': 0
            }
    
    def analyze_connectivity_pattern(self, mask: np.ndarray) -> dict[str, Any]:
        """
        Analyze connectivity patterns within the peak region.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        dict[str, Any]
            Connectivity analysis results
        """
        if np.sum(mask) == 0:
            return {'connected_components': 0, 'connectivity_type': 'disconnected'}
        
        # Label connected components
        if mask.ndim == 2:
            structure = np.ones((3, 3))  # 8-connectivity
        else:
            structure = np.ones([3] * mask.ndim)  # Full connectivity
        
        labeled_array, num_components = label(mask, structure=structure)
        
        # Analyze component sizes
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            component_sizes.append(size)
        
        # Determine connectivity type
        if num_components == 0:
            connectivity_type = 'empty'
        elif num_components == 1:
            connectivity_type = 'connected'
        else:
            connectivity_type = 'disconnected'
        
        # Calculate connectivity metrics
        if component_sizes:
            largest_component_ratio = max(component_sizes) / np.sum(mask)
            size_variance = np.var(component_sizes) if len(component_sizes) > 1 else 0.0
        else:
            largest_component_ratio = 0.0
            size_variance = 0.0
        
        return {
            'connected_components': num_components,
            'connectivity_type': connectivity_type,
            'component_sizes': component_sizes,
            'largest_component_ratio': largest_component_ratio,
            'size_variance': size_variance
        }
    
    def analyze_skeleton(self, mask: np.ndarray) -> dict[str, Any]:
        """
        Analyze morphological skeleton of the peak region.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        dict[str, Any]
            Skeleton analysis results
        """
        if np.sum(mask) == 0:
            return {'skeleton_length': 0, 'branch_points': 0, 'end_points': 0}
        
        try:
            # Compute morphological skeleton
            if mask.ndim == 2:
                from skimage.morphology import skeletonize
                skeleton = skeletonize(mask)
            else:
                # For higher dimensions, use thinning approximation
                skeleton = mask.copy()
                # Simple thinning - not as sophisticated as 2D skeletonization
                for _ in range(3):
                    skeleton = binary_erosion(skeleton, np.ones([3] * mask.ndim))
            
            skeleton_length = np.sum(skeleton)
            
            # Analyze skeleton structure for 2D case
            if mask.ndim == 2 and skeleton_length > 0:
                # Find branch points and end points
                branch_points = self._count_branch_points(skeleton)
                end_points = self._count_end_points(skeleton)
            else:
                branch_points = 0
                end_points = 0
            
            # Calculate skeleton ratio
            original_area = np.sum(mask)
            skeleton_ratio = skeleton_length / original_area if original_area > 0 else 0.0
            
            return {
                'skeleton_length': int(skeleton_length),
                'skeleton_ratio': skeleton_ratio,
                'branch_points': branch_points,
                'end_points': end_points
            }
            
        except ImportError:
            # Fallback if skimage not available
            return {
                'skeleton_length': 0,
                'skeleton_ratio': 0.0,
                'branch_points': 0,
                'end_points': 0
            }
    
    def analyze_watershed_properties(self, peak: Peak, data: np.ndarray, num_markers: int = 10) -> dict[str, Any]:
        """
        Analyze watershed properties around the peak.
        
        Parameters:
        -----------
        peak : Peak
            Peak to analyze
        data : np.ndarray
            Height data
        num_markers : int
            Number of watershed markers
            
        Returns:
        --------
        dict[str, Any]
            Watershed analysis results
        """
        try:
            # Create region around peak for watershed analysis
            peak_center = self.calculate_centroid(peak.plateau_indices)
            radius = 20  # Analysis radius
            
            # Define region bounds
            region_slices = []
            for i, center in enumerate(peak_center):
                start = max(0, int(center - radius))
                end = min(data.shape[i], int(center + radius))
                region_slices.append(slice(start, end))
            
            region_data = data[tuple(region_slices)]
            
            if region_data.size == 0:
                return {'watershed_regions': 0, 'drainage_area': 0.0}
            
            # Only perform watershed for 2D data (most common case)
            if region_data.ndim == 2:
                # Create markers (use local minima)
                from skimage.feature import peak_local_maxima
                
                # Invert for minima detection
                inverted_data = -region_data
                
                # Find local maxima in inverted data (i.e., minima in original)
                minima_coords = peak_local_maxima(inverted_data, min_distance=5, num_peaks=num_markers)
                
                if len(minima_coords[0]) > 0:
                    # Create markers image
                    markers = np.zeros_like(region_data, dtype=int)
                    for i, (row, col) in enumerate(zip(minima_coords[0], minima_coords[1])):
                        markers[row, col] = i + 1
                    
                    # Perform watershed
                    watershed_result = watershed(region_data, markers)
                    
                    num_regions = len(np.unique(watershed_result)) - 1  # Exclude background
                    
                    # Calculate drainage area as approximation
                    peak_region_size = len(peak.plateau_indices)
                    
                    return {
                        'watershed_regions': num_regions,
                        'drainage_area': float(peak_region_size),
                        'watershed_complexity': num_regions / (region_data.size / 100)
                    }
            
            # Fallback for non-2D or when watershed fails
            return {
                'watershed_regions': 1,
                'drainage_area': float(len(peak.plateau_indices)),
                'watershed_complexity': 0.1
            }
            
        except ImportError:
            # Fallback if sklearn not available
            return {
                'watershed_regions': 1,
                'drainage_area': float(len(peak.plateau_indices))
            }
    
    def calculate_euler_number(self, mask: np.ndarray) -> int:
        """
        Calculate Euler number (topological invariant).
        
        For 2D: Euler number = connected_components - holes
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        int
            Euler number
        """
        if mask.ndim == 2:
            # Use ndimage to calculate Euler number
            euler_number = ndimage.label(mask)[1] - self._count_holes_2d(mask)
            return euler_number
        else:
            # For higher dimensions, approximate with connected components
            return ndimage.label(mask)[1]
    
    def calculate_boundary_complexity(self, mask: np.ndarray) -> dict[str, float]:
        """
        Calculate boundary complexity measures.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask of peak region
            
        Returns:
        --------
        dict[str, float]
            Boundary complexity measures
        """
        if np.sum(mask) == 0:
            return {'boundary_length': 0.0, 'fractal_dimension': 0.0}
        
        # Calculate boundary
        boundary = find_boundaries(mask, mode='inner')
        boundary_length = np.sum(boundary)
        
        # Calculate area
        area = np.sum(mask)
        
        # Boundary complexity as perimeter-to-area ratio
        if area > 0:
            complexity = boundary_length / np.sqrt(area)
        else:
            complexity = 0.0
        
        # Approximate fractal dimension using box-counting method (simplified)
        fractal_dim = self._estimate_fractal_dimension(boundary)
        
        return {
            'boundary_length': float(boundary_length),
            'complexity_ratio': complexity,
            'fractal_dimension': fractal_dim
        }
    
    # Helper methods
    
    def _create_peak_mask(self, peak: Peak, data_shape: tuple[int, ...]) -> np.ndarray:
        """Create binary mask for peak region."""
        mask = np.zeros(data_shape, dtype=bool)
        for idx in peak.plateau_indices:
            mask[idx] = True
        return mask
    
    def _count_branch_points(self, skeleton: np.ndarray) -> int:
        """Count branch points in 2D skeleton."""
        # A branch point has more than 2 neighbors
        branch_count = 0
        
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j]:
                    # Count neighbors
                    neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - 1  # Exclude center
                    if neighbors > 2:
                        branch_count += 1
        
        return branch_count
    
    def _count_end_points(self, skeleton: np.ndarray) -> int:
        """Count end points in 2D skeleton."""
        # An end point has exactly 1 neighbor
        end_count = 0
        
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j]:
                    # Count neighbors
                    neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - 1  # Exclude center
                    if neighbors == 1:
                        end_count += 1
        
        return end_count
    
    def _count_holes_2d(self, mask: np.ndarray) -> int:
        """Count holes in 2D binary mask."""
        # Invert mask and label connected components
        # Holes are background components that don't touch the border
        inverted = ~mask
        labeled, num_labels = label(inverted)
        
        # Count background components that don't touch border
        holes = 0
        for label_id in range(1, num_labels + 1):
            component = (labeled == label_id)
            
            # Check if component touches border
            touches_border = (
                np.any(component[0, :]) or np.any(component[-1, :]) or
                np.any(component[:, 0]) or np.any(component[:, -1])
            )
            
            if not touches_border:
                holes += 1
        
        return holes
    
    def _estimate_fractal_dimension(self, boundary: np.ndarray) -> float:
        """Estimate fractal dimension using simplified box-counting."""
        if np.sum(boundary) == 0:
            return 0.0
        
        # Get boundary coordinates
        coords = np.column_stack(np.where(boundary))
        
        if len(coords) < 2:
            return 1.0
        
        # Simple fractal dimension estimation
        # Use the range in each dimension
        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
        
        if boundary.ndim == 2:
            # For 2D, estimate based on boundary irregularity
            perimeter = len(coords)
            bounding_perimeter = 2 * np.sum(ranges)
            
            if bounding_perimeter > 0:
                irregularity = perimeter / bounding_perimeter
                # Map to fractal dimension between 1 and 2
                fractal_dim = 1 + min(irregularity, 1.0)
            else:
                fractal_dim = 1.0
        else:
            # For higher dimensions, use simplified formula
            fractal_dim = 1.0 + np.mean(ranges) / np.max(ranges) if np.max(ranges) > 0 else 1.0
        
        return fractal_dim