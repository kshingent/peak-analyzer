# API Reference

_← [Back to Algorithm](../algorithms/algorithm.md) | [Architecture](../architecture/architecture.md) | [日本語版](api_reference_ja.md) →_

## Core Classes and Functions

### **core/peak_detector.py**
```python
class PeakAnalyzer:
    def __init__(strategy, connectivity, boundary_condition, **params)
    def find_peaks(data) -> LazyPeakDataFrame
    def _select_strategy(data) -> Strategy
    def _validate_input(data) -> None
```

### **core/union_find.py**
```python
class UnionFind:
    def __init__(size)
    def union(x, y)
    def find(x) -> int
    def get_components() -> dict[int, list[int]]
    def connected(x, y) -> bool
```

## Strategy Classes

### **strategies/union_find_strategy.py**
```python
class UnionFindStrategy:
    def detect_peaks_and_prominence(data, connectivity, boundary) -> list[Peak]
    def _build_height_priority_graph(data) -> Graph
    def _trace_prominence_paths(peaks, graph) -> dict[Peak, float]
```

### **strategies/plateau_first_strategy.py**
```python
class PlateauFirstStrategy:
    def detect_peaks(data, connectivity, boundary) -> list[Peak]
    def _detect_plateaus(data) -> list[PlateauRegion]
    def _calculate_prominence_batch(plateaus) -> dict[PlateauRegion, float]
```

### **strategies/base_strategy.py**
```python
class BaseStrategy(ABC):
    @abstractmethod
    def detect_peaks(self, data, **params) -> list[Peak]
    @abstractmethod
    def calculate_features(self, peaks, data) -> dict[Peak, Features]
    @abstractmethod
    def estimate_performance(self, data_shape) -> PerformanceMetrics
    
    def preprocess_data(self, data) -> np.ndarray
    def postprocess_peaks(self, peaks) -> list[Peak]
```

## Data Models

### **models/peaks.py**
```python
@dataclass
class Peak:
    position: IndexTuple | CoordTuple
    height: float
    area: int
    prominence: float | None = None

@dataclass 
class VirtualPeak:
    position: IndexTuple | CoordTuple
    height: float
    is_boundary_artifact: bool = False

@dataclass
class SaddlePoint:
    position: IndexTuple | CoordTuple
    height: float
```

## API Layer Classes

### **api/peak_detector.py**
```python
class PeakAnalyzer:
    def __init__(self, strategy='auto', connectivity=1, 
                 boundary='infinite_height', scale=None, 
                 minkowski_p=2.0, **kwargs)
    def find_peaks(self, data, **filters) -> PeakCollection
    def analyze_prominence(self, peaks, wlen=None) -> ProminenceResults
    def calculate_features(self, peaks, features='all') -> FeatureDataFrame
    def filter_peaks(self, peaks, **criteria) -> PeakCollection
    def get_virtual_peaks(self, peaks) -> VirtualPeakCollection
```

### **api/result_dataframe.py**
```python
class PeakCollection:
    def filter(self, **criteria) -> 'PeakCollection'
    def sort_by(self, feature, ascending=True) -> 'PeakCollection'
    def get_features(self, features=None) -> LazyDataFrame
    def to_pandas(self) -> pd.DataFrame
    def visualize(self, backend='matplotlib', **kwargs)

class LazyDataFrame:
    def compute_feature(self, feature_name, **params)
    def __getitem__(self, key)  # Trigger lazy computation
    def cache_features(self, features)
    def clear_cache(self)
```

## Core Algorithm Classes

### **core/strategy_manager.py**
```python
class StrategyManager:
    def select_optimal_strategy(self, data, **kwargs) -> Strategy
        # **kwargs supports 'strategy_name' or 'force_strategy' for manual override
    def estimate_computational_cost(self, strategy_name, data_shape) -> dict[str, float]
    def benchmark_strategies(self, data, strategies=None) -> list[BenchmarkResults]
    def configure_strategy(self, strategy_name, **params) -> Strategy
    def auto_configure(self, data_shape, characteristics=None, 
                      performance_requirements=None) -> Strategy
        # Automatically configure optimal strategy with performance requirements
```

### **core/plateau_detector.py**
```python
class PlateauDetector:
    def detect_plateaus(self, data, connectivity) -> list[PlateauRegion]
    def validate_plateau(self, region, data, connectivity) -> bool
    def merge_connected_plateaus(self, plateaus) -> list[PlateauRegion]
    def filter_noise_plateaus(self, plateaus, min_area) -> list[PlateauRegion]
```

### **core/prominence_calculator.py**
```python
class ProminenceCalculator:
    def calculate_prominence(self, peak, data, wlen=None) -> float
    def find_prominence_base(self, peak, data) -> Coordinate
    def trace_descent_path(self, start_point, data) -> Path
    def handle_boundary_cases(self, peak, data) -> float
```

### **core/virtual_peak_handler.py**
```python
class VirtualPeakHandler:
    def detect_connected_same_height_peaks(self, peaks) -> list[PeakGroup]
    def create_virtual_peak(self, peak_group) -> VirtualPeak
    def calculate_virtual_prominence(self, virtual_peak, data) -> float
    def resolve_saddle_points(self, peak_group) -> list[SaddlePoint]
```

## Feature Calculation Classes

### **features/geometric_calculator.py**
```python
class GeometricCalculator(BaseCalculator):
    def calculate_area(self, peak_region, scale) -> float
    def calculate_volume(self, peak_region, data, scale) -> float
    def calculate_centroid(self, peak_region, data) -> Coordinate
    def calculate_aspect_ratio(self, peak_region) -> dict[str, float]
    def calculate_bounding_box(self, peak_region) -> BoundingBox
```

### **features/topographic_calculator.py**
```python
class TopographicCalculator(BaseCalculator):
    def calculate_isolation(self, peak, all_peaks, minkowski_p) -> float
    def calculate_relative_height(self, peak, data, neighborhood_radius) -> float
    def calculate_topographic_position_index(self, peak, data) -> float
    def calculate_width_at_relative_height(self, peak, data, rel_height) -> float
```

### **features/distance_calculator.py**
```python
class DistanceCalculator(BaseCalculator):
    def calculate_minkowski_distance(self, point1, point2, p, scale) -> float
    def find_nearest_higher_peak(self, peak, all_peaks) -> tuple[Peak, float]
    def calculate_peak_density(self, center, peaks, radius) -> float
    def calculate_watershed_distance(self, peak1, peak2, data) -> float
```

## Connectivity and Coordinate System Classes

### **connectivity/connectivity_types.py**
```python
class Connectivity:
    def __init__(self, ndim: int, k: int)
    def get_neighbors(self, center: tuple, shape: tuple) -> np.ndarray
    @property
    def neighbor_count(self) -> int
# N-dimensional k-connectivity implementation
# k=1: face connectivity, k=ndim: full (Moore) connectivity
```

### **coordinate_system/grid_manager.py**
```python
class GridManager:
    def __init__(self, mapping: CoordinateMapping, connectivity_level: int)
    def indices_to_coordinates(self, indices) -> tuple | np.ndarray
    def coordinates_to_indices(self, coordinates) -> tuple | np.ndarray
    def calculate_distance(self, coord1, coord2, p=2.0) -> float
    def get_neighbors_coordinates(self, center_coordinates) -> list[tuple]
    def find_neighbors_in_radius(self, center, radius, metric) -> list[tuple]

class Peak:
    center_indices: tuple[int, ...]      # Internal processing indices
    center_coordinates: tuple[float, ...] # User-facing coordinates
    plateau_indices: list[tuple[int, ...]] # Region in index space
    @property
    def coordinate_dict(self) -> dict     # {x: 1.5, y: 2.3, z: 4.1}
```

### **coordinate_system/coordinate_mapping.py**
```python
@dataclass
class CoordinateMapping:
    indices_shape: tuple[int, ...]     # Data array shape (I, J, K, ...)
    coordinate_origin: tuple[float, ...] # Real-world origin (x0, y0, z0, ...)
    coordinate_spacing: tuple[float, ...] # Physical spacing (dx, dy, dz, ...)
    axis_names: tuple[str, ...] = ('x', 'y', 'z') # Coordinate axis names
```

## Boundary Processing

Boundary handling is implemented with simple padding:

- **infinite_height**: Pad data with infinite values (default)
- **infinite_depth**: Pad data with negative infinite values

```python
# Simple padding implementation
padded_data = np.pad(data, 1, mode='constant', constant_values=pad_value)
```

Boundary value handling:
- During prominence calculation, infinite values are skipped for efficiency
- Padding is always fixed at 1 pixel

## Boundary Conditions and Prominence Calculation Constraints

### **Boundary Condition Selection Constraints**

**Only 2 boundary conditions are practical:**

**1. infinite_height (Recommended・Default)**
- Prominence calculation possible for all peaks
- Treats boundary as infinitely high walls
- Most natural interpretation in topographical analysis

**2. infinite_depth**
- Prominence calculation possible for all peaks except the global maximum
- Global maximum prominence = global maximum height - global minimum height
- Treats boundary as infinitely deep valleys

**Problems with periodic, constant, mirror, nearest, etc.:**
- Create "artificial terrain" beyond the boundary
- Prominence calculation becomes dependent on boundary condition assumptions
- Results lack topographical meaning
- **Not applicable for prominence calculation**

### **Global Maximum Peak Handling in infinite_depth**
```python
# Global maximum peak handling in infinite_depth case
global_max_peak = find_global_maximum(peaks)
global_min_height = find_global_minimum(data)

for peak in peaks:
    if peak == global_max_peak:
        # Global maximum prominence = height difference from lowest point
        peak.prominence = peak.height - global_min_height
    else:
        # Other peaks use standard calculation
        peak.prominence = calculate_standard_prominence(peak)
```

### **Boundary Condition Selection Guidelines**
- **Default**: `infinite_height` - Most safe and general approach
- **Special cases**: `infinite_depth` - When emphasizing peaks near boundaries
- **Others**: Used only for peak detection, prominence calculation disabled

## Data Management Classes

### **data/lazy_dataframe.py**
```python
class LazyDataFrame:
    def __init__(self, peak_collection, feature_calculators)
    def __getitem__(self, feature_name)  # Lazy computation
    def compute_batch(self, feature_names) -> dict[str, np.ndarray]
    def cache_strategy(self, strategy='lru', max_size=None)
    def memory_usage(self) -> MemoryReport
```

## Feature Calculation Specifications

### **Geometric Features**
- **Peak Plateau Area**: Number of cells in plateau region
- **Peak Volume**: Sum of height values in plateau region  
- **Centroid Position**: Weighted center of mass coordinates
- **Boundary Length/Area**: Perimeter of plateau region
- **Aspect Ratio**: Elongation measures in each dimension
- **Bounding Box**: Min/max coordinates in each dimension

### **Topographic Features**  
- **Peak Height**: Maximum elevation in region
- **Prominence**: Vertical distance to lowest enclosing contour
- **Isolation**: Distance to nearest higher peak
- **Relative Height**: Height relative to local surroundings
- **Topographic Position Index**: Position relative to neighborhood elevation
- **Width at Relative Height (rel_height)**: Area of the peak region at a specified relative height between prominence_base and peak_height (e.g., rel_height=0.5 measures width at half-prominence level)

### **Morphological Features**
- **Sharpness**: Local curvature around peak centroid
- **Average Slope**: Mean gradient magnitude in plateau region
- **Slope Variability**: Standard deviation of gradients
- **Directional Slopes**: Gradients in each coordinate direction
- **Laplacian**: Second-order derivative measures

### **Distance and Connectivity Features**
- **Nearest Higher Peak Distance**: Configurable Minkowski distance (L_p norm) to closest higher peak
- **Nearest Similar Peak Distance**: Distance to peak of similar height using specified distance metric
- **Peak Density**: Number of peaks within specified radius (accounting for pixel scale)
- **Connectivity Order**: Hierarchical relationship to other peaks
- **Watershed Distance**: Distance following steepest descent paths
- **Scaled Distances**: All distance measurements respect pixel scale parameters for accurate physical measurements
- **Minkowski Distance Options**: Support for L1 (Manhattan), L2 (Euclidean), L∞ (Chebyshev), and custom Lp norms