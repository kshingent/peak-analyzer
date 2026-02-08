"""
Edge Detection

Provides edge and boundary detection capabilities for peak analysis including
gradient-based edge detection, boundary extraction, and domain analysis.
"""

from typing import Any
import numpy as np
from enum import Enum

from peak_analyzer.coordinate_system import GridManager
from .boundary_conditions import BoundaryManager


class EdgeMethod(Enum):
    """Methods for edge detection."""
    GRADIENT = "gradient"        # Gradient-based edge detection
    SOBEL = "sobel"             # Sobel operator
    PREWITT = "prewitt"         # Prewitt operator
    ROBERTS = "roberts"         # Roberts cross-gradient
    LAPLACIAN = "laplacian"     # Laplacian edge detection
    CANNY = "canny"             # Canny edge detector
    THRESHOLD = "threshold"      # Threshold-based edge detection


class EdgeDetector:
    """
    Edge detection for multi-dimensional arrays.
    """
    
    def __init__(self, method: str | EdgeMethod = EdgeMethod.GRADIENT,
                 boundary_manager: BoundaryManager | None = None):
        """
        Initialize edge detector.
        
        Parameters:
        -----------
        method : str or EdgeMethod
            Edge detection method
        boundary_manager : BoundaryManager, optional
            Boundary conditions for gradient calculations
        """
        if isinstance(method, str):
            method = EdgeMethod(method.lower())
        
        self.method = method
        self.boundary_manager = boundary_manager
    
    def detect_edges(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect edges in data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        **kwargs
            Method-specific parameters
            
        Returns:
        --------
        np.ndarray
            Edge detection result
        """
        if self.method == EdgeMethod.GRADIENT:
            return self._gradient_edges(data, **kwargs)
        elif self.method == EdgeMethod.SOBEL:
            return self._sobel_edges(data, **kwargs)
        elif self.method == EdgeMethod.PREWITT:
            return self._prewitt_edges(data, **kwargs)
        elif self.method == EdgeMethod.ROBERTS:
            return self._roberts_edges(data, **kwargs)
        elif self.method == EdgeMethod.LAPLACIAN:
            return self._laplacian_edges(data, **kwargs)
        elif self.method == EdgeMethod.CANNY:
            return self._canny_edges(data, **kwargs)
        elif self.method == EdgeMethod.THRESHOLD:
            return self._threshold_edges(data, **kwargs)
        else:
            raise ValueError(f"Unknown edge detection method: {self.method}")
    
    def _gradient_edges(self, data: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Calculate gradient-based edges."""
        gradients = np.gradient(data)
        
        # Calculate gradient magnitude
        if isinstance(gradients, list):
            # Multi-dimensional gradient
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        else:
            # 1D gradient
            gradient_magnitude = np.abs(gradients)
        
        if threshold is not None:
            gradient_magnitude = gradient_magnitude > threshold
        
        return gradient_magnitude
    
    def _sobel_edges(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate Sobel edges."""
        if data.ndim == 1:
            # 1D Sobel approximation using finite differences
            kernel = np.array([-1, 0, 1])
            padded = np.pad(data, 1, mode='edge')
            result = np.abs(np.convolve(padded, kernel, mode='valid')[1:-1])
            return result
        
        elif data.ndim == 2:
            # 2D Sobel operators
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Apply boundary handling
            if self.boundary_manager:
                extended, _ = self.boundary_manager.extend_data(data, 1)
            else:
                extended = np.pad(data, 1, mode='edge')
            
            # Convolve with Sobel kernels
            from scipy import ndimage
            try:
                grad_x = ndimage.convolve(extended, sobel_x)[1:-1, 1:-1]
                grad_y = ndimage.convolve(extended, sobel_y)[1:-1, 1:-1]
                return np.sqrt(grad_x**2 + grad_y**2)
            except ImportError:
                # Fallback to manual convolution
                return self._manual_convolve_2d(data, [sobel_x, sobel_y])
        
        else:
            # Multi-dimensional: use gradient approximation
            return self._gradient_edges(data, **kwargs)
    
    def _prewitt_edges(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate Prewitt edges."""
        if data.ndim == 2:
            # 2D Prewitt operators
            prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            
            return self._manual_convolve_2d(data, [prewitt_x, prewitt_y])
        else:
            # Fallback to gradient
            return self._gradient_edges(data, **kwargs)
    
    def _roberts_edges(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate Roberts cross-gradient edges."""
        if data.ndim == 2:
            # Roberts cross-gradient operators
            roberts_x = np.array([[1, 0], [0, -1]])
            roberts_y = np.array([[0, 1], [-1, 0]])
            
            return self._manual_convolve_2d(data, [roberts_x, roberts_y])
        else:
            # Fallback to gradient
            return self._gradient_edges(data, **kwargs)
    
    def _laplacian_edges(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate Laplacian edges."""
        if data.ndim == 1:
            # 1D Laplacian: second derivative
            kernel = np.array([1, -2, 1])
            padded = np.pad(data, 1, mode='edge')
            result = np.abs(np.convolve(padded, kernel, mode='valid')[1:-1])
            return result
        
        elif data.ndim == 2:
            # 2D Laplacian kernel
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            
            if self.boundary_manager:
                extended, _ = self.boundary_manager.extend_data(data, 1)
            else:
                extended = np.pad(data, 1, mode='edge')
            
            try:
                from scipy import ndimage
                result = ndimage.convolve(extended, laplacian)[1:-1, 1:-1]
                return np.abs(result)
            except ImportError:
                return self._manual_convolve_2d(data, [laplacian], take_magnitude=False)
        
        else:
            # Multi-dimensional: use divergence of gradient
            gradients = np.gradient(data)
            if isinstance(gradients, list):
                laplacian = sum(np.gradient(g, axis=i) 
                              for i, g in enumerate(gradients))
            else:
                laplacian = np.gradient(gradients)
            
            return np.abs(laplacian)
    
    def _canny_edges(self, data: np.ndarray, sigma: float = 1.0,
                     low_threshold: float = 0.1, high_threshold: float = 0.2) -> np.ndarray:
        """Calculate Canny edges (simplified version)."""
        try:
            from scipy import ndimage
            
            # Gaussian smoothing
            smoothed = ndimage.gaussian_filter(data, sigma)
            
            # Gradient calculation
            gradients = np.gradient(smoothed)
            
            if data.ndim == 1:
                gradient_magnitude = np.abs(gradients)
            else:
                gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
            
            # Thresholding (simplified - no non-maximum suppression)
            high_edges = gradient_magnitude > high_threshold
            
            # Simple edge linking (connect high edges through low edges)
            edges = high_edges.copy()
            
            # This is a simplified version - full Canny would include
            # non-maximum suppression and proper edge linking
            
            return edges.astype(float)
            
        except ImportError:
            # Fallback to gradient-based edge detection
            return self._gradient_edges(data, threshold=low_threshold)
    
    def _threshold_edges(self, data: np.ndarray, threshold: float,
                        comparison: str = 'greater') -> np.ndarray:
        """Calculate threshold-based edges."""
        if comparison == 'greater':
            return (data > threshold).astype(float)
        elif comparison == 'less':
            return (data < threshold).astype(float)
        elif comparison == 'equal':
            return (np.abs(data - threshold) < 1e-10).astype(float)
        else:
            raise ValueError(f"Unknown comparison: {comparison}")
    
    def _manual_convolve_2d(self, data: np.ndarray, kernels: list[np.ndarray],
                           take_magnitude: bool = True) -> np.ndarray:
        """Manual 2D convolution for edge detection."""
        if self.boundary_manager:
            extended, _ = self.boundary_manager.extend_data(data, 1)
        else:
            extended = np.pad(data, 1, mode='edge')
        
        results = []
        
        for kernel in kernels:
            kh, kw = kernel.shape
            result = np.zeros(data.shape)
            
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    region = extended[i:i+kh, j:j+kw]
                    result[i, j] = np.sum(region * kernel)
            
            results.append(result)
        
        if take_magnitude and len(results) > 1:
            return np.sqrt(sum(r**2 for r in results))
        elif len(results) == 1:
            return np.abs(results[0]) if take_magnitude else results[0]
        else:
            return np.array(results)


class BoundaryExtractor:
    """
    Extract boundaries and domain edges from data.
    """
    
    def __init__(self, grid_manager: GridManager | None = None):
        """
        Initialize boundary extractor.
        
        Parameters:
        -----------
        grid_manager : GridManager, optional
            Grid manager for coordinate transformations
        """
        self.grid_manager = grid_manager
    
    def extract_domain_boundary(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """
        Extract domain boundary (edges of the data array).
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
            
        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary with boundary faces/edges
        """
        boundary_faces = {}
        
        for dim in range(data.ndim):
            # Lower boundary (face at index 0 along dimension)
            lower_slice = [slice(None)] * data.ndim
            lower_slice[dim] = 0
            boundary_faces[f'lower_{dim}'] = data[tuple(lower_slice)]
            
            # Upper boundary (face at last index along dimension)
            upper_slice = [slice(None)] * data.ndim
            upper_slice[dim] = -1
            boundary_faces[f'upper_{dim}'] = data[tuple(upper_slice)]
        
        return boundary_faces
    
    def extract_value_boundary(self, data: np.ndarray, threshold: float,
                             method: str = 'contour') -> list[np.ndarray]:
        """
        Extract boundaries based on value thresholds.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array
        threshold : float
            Value threshold for boundary detection
        method : str
            Boundary extraction method ('contour', 'gradient')
            
        Returns:
        --------
        list[np.ndarray]
            List of boundary coordinates
        """
        if method == 'contour':
            return self._extract_contour_boundary(data, threshold)
        elif method == 'gradient':
            return self._extract_gradient_boundary(data, threshold)
        else:
            raise ValueError(f"Unknown boundary extraction method: {method}")
    
    def _extract_contour_boundary(self, data: np.ndarray, threshold: float) -> list[np.ndarray]:
        """Extract contour-based boundaries."""
        if data.ndim == 1:
            # 1D: find crossing points
            crossings = []
            for i in range(len(data) - 1):
                if (data[i] <= threshold < data[i+1]) or (data[i] >= threshold > data[i+1]):
                    # Linear interpolation for crossing point
                    t = (threshold - data[i]) / (data[i+1] - data[i])
                    crossing = i + t
                    crossings.append(crossing)
            
            return [np.array(crossings)] if crossings else []
        
        elif data.ndim == 2:
            try:
                # Use matplotlib for contour extraction
                import matplotlib.pyplot as plt
                
                # Create coordinate grids
                if self.grid_manager:
                    x_coords, y_coords = self.grid_manager.coordinate_mesh()
                else:
                    y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]
                
                # Extract contours
                contours = plt.contour(x_coords, y_coords, data, levels=[threshold])
                
                boundary_lines = []
                for collection in contours.collections:
                    for path in collection.get_paths():
                        boundary_lines.append(path.vertices)
                
                plt.close()  # Clean up figure
                
                return boundary_lines
                
            except ImportError:
                # Fallback to simple threshold-based extraction
                boundary_mask = np.abs(data - threshold) < (np.max(data) - np.min(data)) * 0.01
                boundary_indices = np.where(boundary_mask)
                
                if len(boundary_indices[0]) > 0:
                    if self.grid_manager:
                        boundary_coords = []
                        for i, j in zip(*boundary_indices):
                            coord = self.grid_manager.index_to_coordinate((i, j))
                            boundary_coords.append(coord)
                        return [np.array(boundary_coords)]
                    else:
                        return [np.column_stack(boundary_indices)]
                else:
                    return []
        
        else:
            # Multi-dimensional: use threshold mask
            boundary_mask = np.abs(data - threshold) < (np.max(data) - np.min(data)) * 0.01
            boundary_indices = np.where(boundary_mask)
            
            if len(boundary_indices[0]) > 0:
                return [np.column_stack(boundary_indices)]
            else:
                return []
    
    def _extract_gradient_boundary(self, data: np.ndarray, threshold: float) -> list[np.ndarray]:
        """Extract gradient-based boundaries."""
        # Use edge detection to find high-gradient regions
        edge_detector = EdgeDetector(method=EdgeMethod.GRADIENT)
        edges = edge_detector.detect_edges(data)
        
        # Threshold the edge response
        boundary_mask = edges > threshold
        boundary_indices = np.where(boundary_mask)
        
        if len(boundary_indices[0]) > 0:
            if self.grid_manager:
                boundary_coords = []
                for indices in zip(*boundary_indices):
                    coord = self.grid_manager.index_to_coordinate(indices)
                    boundary_coords.append(coord)
                return [np.array(boundary_coords)]
            else:
                return [np.column_stack(boundary_indices)]
        else:
            return []
    
    def analyze_boundary_properties(self, boundaries: list[np.ndarray]) -> dict[str, Any]:
        """
        Analyze properties of extracted boundaries.
        
        Parameters:
        -----------
        boundaries : list[np.ndarray]
            List of boundary coordinate arrays
            
        Returns:
        --------
        dict[str, Any]
            Boundary analysis results
        """
        if not boundaries:
            return {'num_boundaries': 0}
        
        analysis = {
            'num_boundaries': len(boundaries),
            'boundary_lengths': [],
            'boundary_stats': []
        }
        
        for i, boundary in enumerate(boundaries):
            if len(boundary) == 0:
                continue
            
            # Calculate boundary length/perimeter
            if boundary.ndim == 1:
                length = len(boundary)
            else:
                # Calculate path length
                if len(boundary) > 1:
                    diffs = np.diff(boundary, axis=0)
                    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                    length = np.sum(segment_lengths)
                else:
                    length = 0.0
            
            analysis['boundary_lengths'].append(length)
            
            # Basic statistics
            boundary_stats = {
                'num_points': len(boundary),
                'length': length,
                'centroid': np.mean(boundary, axis=0) if len(boundary) > 0 else None,
                'bounding_box': {
                    'min': np.min(boundary, axis=0) if len(boundary) > 0 else None,
                    'max': np.max(boundary, axis=0) if len(boundary) > 0 else None
                }
            }
            
            analysis['boundary_stats'].append(boundary_stats)
        
        # Overall statistics
        if analysis['boundary_lengths']:
            analysis['total_boundary_length'] = sum(analysis['boundary_lengths'])
            analysis['average_boundary_length'] = np.mean(analysis['boundary_lengths'])
            analysis['max_boundary_length'] = max(analysis['boundary_lengths'])
            analysis['min_boundary_length'] = min(analysis['boundary_lengths'])
        
        return analysis


def detect_data_boundaries(data: np.ndarray, 
                          method: str | EdgeMethod = EdgeMethod.GRADIENT,
                          grid_manager: GridManager | None = None,
                          **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Comprehensive boundary detection for data arrays.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    method : str or EdgeMethod
        Edge detection method
    grid_manager : GridManager, optional
        Grid manager for coordinate transformations
    **kwargs
        Method-specific parameters
        
    Returns:
    --------
    tuple of (edges, analysis)
        Edge detection result and analysis
    """
    # Create edge detector
    edge_detector = EdgeDetector(method=method)
    
    # Detect edges
    edges = edge_detector.detect_edges(data, **kwargs)
    
    # Create boundary extractor
    boundary_extractor = BoundaryExtractor(grid_manager)
    
    # Extract domain boundaries
    domain_boundaries = boundary_extractor.extract_domain_boundary(data)
    
    # Analyze edges
    edge_stats = {
        'edge_method': method.value if isinstance(method, EdgeMethod) else method,
        'edge_intensity_stats': {
            'mean': np.mean(edges),
            'std': np.std(edges),
            'min': np.min(edges),
            'max': np.max(edges),
            'num_edge_pixels': np.sum(edges > 0)
        },
        'domain_boundaries': domain_boundaries
    }
    
    return edges, edge_stats