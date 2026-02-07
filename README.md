# peak-analyzer

## Core Algorithm: Topography-Aware Multidimensional Peak Detection

### The Limitation of 1D Logic in Higher Dimensions

While `scipy.signal.find_peaks` is a robust tool for 1D signal processing, its underlying logic is specialized for one-dimensional topography. In 1D, a peak is simply a point higher than its immediate left and right neighbors. In 2D or N-dimensional spaces, this definition becomes insufficient.

Existing 2D peak detection algorithms frequently encounter two primary issues:

1. **Plateau Misidentification:** Areas of constant height (plateaus) are often incorrectly flagged as multiple individual peaks.
2. **Lack of Feature-Based Filtering:** Most tools lack a mechanism to filter peaks based on geomorphological features such as prominence or isolation, as they treat peaks as isolated points rather than structural components of the data.

**PeakAnalyzer** addresses these issues by redefining a peak not as a "point," but as a **topographic region**.

---

### Topographic Features Extracted

PeakAnalyzer quantifies the geometric properties of peaks to allow for meaningful selection and filtering:

* **Peak Coordinates:** The centroid or representative point of the peak region.
* **Height:** The absolute value at the peak.
* **Prominence:** The vertical distance between the peak and its lowest contour line.
* **Area:** The spatial extent of the peak region.
* **Sharpness:** The local curvature or rate of height change.
* **Isolation:** The distance to the nearest higher peak.
* **Distance:** Spatial relationship metrics between detected features.

#### Pixel Scale and Distance Metrics

* **Pixel Scale Parameter:** Allows specification of real-world scale for each dimension (e.g., `scale=[1.0, 0.5]` for different x/y resolution).
* **Minkowski Distance:** Distance calculations support configurable Minkowski distance metrics with parameter `p` (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev).
* **Distance Scale:** Physical units can be preserved through scale parameters, ensuring accurate real-world distance measurements.
* **Relative Height (rel_height):** Defines peak width as the area enclosed between prominence_base and peak_height at a relative height ratio.

#### Coordinate System Architecture

**Index Space vs. Coordinate Space Separation**

PeakAnalyzer maintains a clear distinction between two spatial representations:

* **Index Space (i,j,...):** Internal data array indexing for computational efficiency
  - Used for: Algorithm processing, memory access, connectivity analysis
  - Format: Integer indices matching data array dimensions
  - Example: `peak_indices = (127, 89)` for 2D array access

* **Coordinate Space (x,y,...):** Real-world physical coordinates for user interaction (when applicable)
  - Used for: User input/output, visualization, spatial analysis, filtering
  - Format: Float coordinates with physical units
  - Example: `peak_coordinates = (-120.5, 36.2)` for longitude/latitude, data value = 1450.0 (arbitrary physical units)

**Physical Meaning and Real-World Units**
* **Distance & Area Calculations:** All measurements computed in actual physical units
* **Anisotropic Resolution Support:** Each axis can have different scales (`scale=[1.0, 1.0, 0.5]`)
* **Unit Preservation:** Maintains consistent physical units throughout analysis pipeline
* **Scale-Aware Features:** Geometric properties respect real-world dimensions

**Spatial Analysis Capabilities**
* **GIS-Compatible Operations:** Coordinate-based filtering, buffering, and spatial queries
* **Geographic Integration:** Native support for geographic coordinate systems
* **Multi-Scale Analysis:** Seamless handling of different resolution data
* **Projection Support:** Flexible coordinate system transformations

**Visualization and User Interface Principles**
* **User-Centric Coordinates:** All user interactions in intuitive xyz... coordinate space
* **Real-World Visualization:** Plots and displays use physical coordinates with proper units
* **Index Abstraction:** Internal ijk... indexing hidden from user interface
* **Intuitive Filtering:** Spatial filters operate on meaningful coordinate ranges

---

### Rigorous Plateau Handling

The core of multidimensional peak detection lies in the correct identification of plateaus. PeakAnalyzer treats a peak as a "region of equal height" to prevent the generation of artifacts within flat surfaces.

#### Plateau Detection Procedure

1. **Maximum Filtering:** Apply a local maximum filter to the dataset.
2. **Dilation:** Perform a morphological dilation on the identified regions of constant height .
3. **Validation:** If no cells with height  exist in the dilated boundary that were not part of the original region, the area is classified as a **true peak plateau**.

This logic ensures that internal cells of a large flat top are not misidentified as separate peaks.

---

### Prominence Calculation

PeakAnalyzer adheres strictly to the geomorphological definition of prominence, extending it to multidimensional spaces:

1. A height-priority neighborhood search begins from the peak region.
2. The search continues until it encounters a point with a higher value than the current peak.
3. The lowest elevation reached during this search is defined as prominence_base.
4. The prominence is calculated as: `prominence = peak_height - prominence_base`

#### Window Length (wlen) Parameter

The `wlen` parameter constrains prominence calculation to a specified distance:
* When `wlen` is specified, prominence calculation stops at points beyond the given distance from the peak
* This effectively treats the analysis region as a windowed subset of the full dataset
* Useful for focusing on local prominence within specific spatial scales

#### Highest Peak Prominence Handling

For peaks that are globally highest (no higher peaks exist):
* **Global Maximum Rule:** Prominence equals the height difference to the lowest point on the data boundary
* **Boundary Condition:** If infinite boundary is used, prominence calculation traces to actual data edges

#### Same-Height Connected Peaks with Virtual Peak Creation

When multiple peaks share the same elevation and are topographically connected:
* **Connected Same-Height Peaks at Any Elevation:**
  - When multiple peaks A and B exist at the same height `h` and are connected through terrain of equal or higher elevation
  - **Step 1**: Detect all peaks at the same elevation `h` that are connected via terrain at height ≥ `h`
  - **Step 2**: Find saddle points between connected same-height peaks A and B
  - **Step 3**: Set prominence_base for individual peaks A and B to the height of their connecting saddle point
  - **Step 4**: Create a virtual peak encompassing all connected peaks at height `h`
  - **Step 5**: If higher terrain at height `h' > h` connects to any peak in the virtual peak group via a saddle point, set the virtual peak's prominence_base to that saddle height
  - **Step 6**: The virtual peak's prominence = `h - saddle_height_to_higher_terrain`
  - **Step 7**: If no higher terrain exists, use boundary-based prominence calculation
  - This approach ensures consistent prominence calculation for any elevation level with connected peaks

By implementing this via path-finding logic, the algorithm remains dimension-agnostic.

---

### Calculation Strategies

The library provides two strategies depending on the "topography" of the data:

| Strategy | Logic | Best Use Case |
| --- | --- | --- |
| **Independent Calculation** | Identifies all plateaus first, then calculates prominence for each individually. | Data with high contrast and isolated, sharp peaks. |
| **Batch Heap Search** | Uses a heap-priority queue to explore the terrain by height. | Smooth terrains or data with expansive, complex plateau structures. |

---

### Efficiency via Lazy Evaluation

Calculating all topographic features for every potential peak is computationally expensive. PeakAnalyzer utilizes a **LazyDataFrame** approach to optimize performance.

* **On-Demand Computation:** Features are only calculated when explicitly requested or required for a filtering operation.
* **Dynamic Process:** If a user filters by `prominence > 0.5`, the algorithm calculates prominence only. If a subsequent filter for `area` is added, it then computes the area for the remaining candidates.
* **Memory Efficiency:** This prevents the overhead of maintaining large feature matrices for peaks that are ultimately discarded.

---

### Design Principles

1. **Topographic Focus:** Treat peaks as regions, not dimensionless points.
2. **Strict Plateau Logic:** Ensure flat surfaces are handled without artifact generation.
3. **Geomorphological Accuracy:** Extend 1D prominence definitions to N-dimensions without simplification.
4. **Optimized Search:** Utilize heap-priority structures to maintain logical consistency across dimensions.
5. **Computational Economy:** Use lazy evaluation to minimize unnecessary calculations on large datasets.
6. **Coordinate System Separation:** Maintain clear distinction between index space (ijk...) for internal processing and coordinate space (xyz...) for user interaction.
7. **Physical Meaning Priority:** All user-facing measurements and visualizations use real-world units and coordinates.
8. **Anisotropic Compatibility:** Native support for different resolutions and scales across dimensions.
9. **Spatial Analysis Integration:** Enable GIS-like coordinate-based analysis and filtering capabilities.

---

## Software Architecture Design

### Architecture Overview

PeakAnalyzer employs a layered, modular architecture that provides a comprehensive framework for topographic peak detection. Each layer has clearly separated responsibilities, ensuring extensibility and maintainability.

```
              User API Layer
┌─────────────────────────────────────────┐
│           PeakAnalyzer                  │  <- Main Interface
│        (peak_detector.py)               │
└─────────────┬───────────────────────────┘
              │
           Core Control Layer
┌─────────────┴───────────────────────────┐
│      StrategySelector & Manager         │  <- Algorithm Selection & Coordination
│    (strategy_selector.py)               │
└─────────────┬───────────────────────────┘
              │
          Algorithm Layer
┌─────────────┴───────────────────────────┐
│  UnionFindStrategy  │  PlateauFirst     │  <- Detection Strategy Implementation
│                     │  Strategy         │
└─────────────┬───────┴───────────────────┘
              │
           Feature Computation Layer
┌─────────────┴───────────────────────────┐
│  Geometric   │ Topographic │ Distance  │  <- Topographic Feature Calculation
│  Features    │ Features    │ Features   │
└─────────────┬───────────────────────────┘
              │
          Foundation Services Layer
┌─────────────┴───────────────────────────┐
│ Connectivity │ Boundary │ Validation   │  <- Core Functions & Utilities
│ & Neighbors  │ Handling │ & Memory     │
└─────────────────────────────────────────┘
```

### Directory Structure and Responsibility Separation

```
peak_analyzer/
├── peak_analyzer/                    # Main package
│   ├── __init__.py                   # Package entry point
│   ├── api/                         # User API Layer
│   │   ├── peak_detector.py         # Main analysis class
│   │   ├── result_dataframe.py      # Result data structures
│   │   └── parameter_validation.py  # Parameter validation
│   │
│   ├── core/                        # Core Algorithm Layer
│   │   ├── strategy_manager.py      # Strategy selection & management
│   │   ├── plateau_detector.py      # Plateau detection core
│   │   ├── prominence_calculator.py # Prominence calculation core
│   │   ├── virtual_peak_handler.py  # Virtual peak processing
│   │   └── union_find.py           # Connected component data structure
│   │
│   ├── strategies/                  # Detection Strategy Implementation Layer
│   │   ├── base_strategy.py         # Strategy base class
│   │   ├── union_find_strategy.py   # Union-Find strategy
│   │   ├── plateau_first_strategy.py # Plateau-first strategy
│   │   ├── hybrid_strategy.py       # Hybrid strategy
│   │   └── strategy_factory.py      # Strategy factory
│   │
│   ├── features/                     # Feature Calculation Layer
│   │   ├── base_calculator.py       # Calculation base class
│   │   ├── geometric_calculator.py  # Geometric features
│   │   ├── topographic_calculator.py # Topographic features
│   │   ├── morphological_calculator.py # Morphological features
│   │   ├── distance_calculator.py   # Distance & connectivity features
│   │   └── lazy_feature_manager.py  # Lazy computation manager
│   │
│   ├── connectivity/                 # Connectivity Definition Layer
│   │   ├── connectivity_types.py    # N-dimensional connectivity definitions
│   │   ├── neighbor_generator.py    # Neighbor generator
│   │   ├── path_finder.py          # Path finding algorithms
│   │   └── distance_metrics.py      # Distance metric implementations
│   │
│   ├── coordinate_system/           # Coordinate System Layer
│   │   ├── grid_manager.py         # Index ↔ Coordinate conversion & spatial operations
│   │   ├── coordinate_mapping.py   # Mapping definitions and validation
│   │   └── spatial_indexing.py     # Spatial indexing and search acceleration
│   │
│   ├── boundary/                    # Boundary Processing Layer
│   │   ├── boundary_handler.py      # Boundary condition processing
│   │   ├── edge_detector.py        # Edge effect detection
│   │   ├── padding_strategies.py   # Padding strategies
│   │   └── artifact_filter.py      # Artifact removal
│   │
│   ├── data/                       # Data Management Layer
│   │   ├── lazy_dataframe.py       # Lazy evaluation dataframe
│   │   ├── peak_collection.py      # Peak collection management
│   │   ├── cache_manager.py        # Cache management
│   │   └── memory_optimizer.py     # Memory optimization
│   │
│   └── utils/                      # Utility Layer
│       ├── validation.py           # Input validation
│       ├── performance_profiler.py # Performance profiling
│       ├── error_handling.py       # Error handling
│       ├── logging_config.py       # Logging configuration
│       └── type_definitions.py     # Type definitions
│
├── tests/                          # Test Suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── performance/               # Performance tests
│   └── fixtures/                  # Test data
│
├── examples/                       # Usage Examples & Demos
│   ├── basic_usage.py             # Basic usage
│   ├── advanced_features.py       # Advanced features
│   ├── custom_strategies.py       # Custom strategies
│   └── benchmarking.py           # Benchmark examples
│
├── docs/                          # Documentation
│   ├── api/                      # API specifications
│   ├── tutorials/                # Tutorials
│   ├── algorithms/               # Algorithm details
│   └── examples/                 # Detailed examples
│
└── benchmarks/                    # Performance Benchmarks
    ├── synthetic_data/           # Synthetic data tests
    ├── real_world_data/          # Real-world data tests
    └── comparison_studies/       # Comparative studies
```

### Layer-by-Layer Detailed Specifications

#### 1. **api/** - User API Layer
Provides unified interface for topographic peak analysis

**peak_detector.py**: Main analysis engine
```python
class PeakAnalyzer:
    def __init__(self, strategy='auto', connectivity='face', 
                 boundary='infinite_height', scale=None, 
                 distance_metric='euclidean', **kwargs)
    def find_peaks(self, data, **filters) -> PeakCollection
    def analyze_prominence(self, peaks, wlen=None) -> ProminenceResults
    def calculate_features(self, peaks, features='all') -> FeatureDataFrame
    def filter_peaks(self, peaks, **criteria) -> PeakCollection
    def get_virtual_peaks(self, peaks) -> VirtualPeakCollection
```

**result_dataframe.py**: Result data structures
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

#### 2. **core/** - Core Algorithm Layer
Implements core algorithms for topographic analysis

**strategy_manager.py**: Strategy selection & management
```python
class StrategyManager:
    def select_optimal_strategy(self, data_characteristics) -> Strategy
    def estimate_computational_cost(self, strategy, data_shape)
    def benchmark_strategies(self, data, strategies) -> BenchmarkResults
    def configure_strategy(self, strategy_name, **params) -> Strategy
```

**plateau_detector.py**: Plateau detection core
```python
class PlateauDetector:
    def detect_plateaus(self, data, connectivity) -> List[PlateauRegion]
    def validate_plateau(self, region, data, connectivity) -> bool
    def merge_connected_plateaus(self, plateaus) -> List[PlateauRegion]
    def filter_noise_plateaus(self, plateaus, min_area) -> List[PlateauRegion]
```

**prominence_calculator.py**: Prominence calculation core
```python
class ProminenceCalculator:
    def calculate_prominence(self, peak, data, wlen=None) -> float
    def find_prominence_base(self, peak, data) -> Coordinate
    def trace_descent_path(self, start_point, data) -> Path
    def handle_boundary_cases(self, peak, data) -> float
```

**virtual_peak_handler.py**: Virtual peak processing
```python
class VirtualPeakHandler:
    def detect_connected_same_height_peaks(self, peaks) -> List[PeakGroup]
    def create_virtual_peak(self, peak_group) -> VirtualPeak
    def calculate_virtual_prominence(self, virtual_peak, data) -> float
    def resolve_saddle_points(self, peak_group) -> List[SaddlePoint]
```

#### 3. **strategies/** - Detection Strategy Implementation Layer
Optimized detection algorithms for different data characteristics

**base_strategy.py**: Common strategy interface
```python
class BaseStrategy(ABC):
    @abstractmethod
    def detect_peaks(self, data, **params) -> List[Peak]
    @abstractmethod
    def calculate_features(self, peaks, data) -> Dict[Peak, Features]
    @abstractmethod
    def estimate_performance(self, data_shape) -> PerformanceMetrics
    
    def preprocess_data(self, data) -> np.ndarray
    def postprocess_peaks(self, peaks) -> List[Peak]
```

**union_find_strategy.py**: Union-Find strategy
```python
class UnionFindStrategy(BaseStrategy):
    def detect_peaks_with_prominence(self, data) -> Tuple[List[Peak], Dict[Peak, float]]
    def build_height_graph(self, data) -> HeightGraph
    def process_height_level(self, height, points) -> List[Component]
    def wave_front_expansion(self, batch, processed) -> Set[Point]
```

**plateau_first_strategy.py**: Plateau-first strategy
```python
class PlateauFirstStrategy(BaseStrategy):
    def detect_plateaus_then_prominence(self, data) -> Tuple[List[Peak], Dict[Peak, float]]
    def apply_local_maximum_filter(self, data) -> np.ndarray
    def validate_plateaus_by_dilation(self, candidates) -> List[PlateauRegion]
    def batch_prominence_calculation(self, plateaus, data) -> Dict[Peak, float]
```

#### 4. **features/** - Feature Calculation Layer
Comprehensive computation framework for topographic features

**geometric_calculator.py**: Geometric features
```python
class GeometricCalculator(BaseCalculator):
    def calculate_area(self, peak_region, scale) -> float
    def calculate_volume(self, peak_region, data, scale) -> float
    def calculate_centroid(self, peak_region, data) -> Coordinate
    def calculate_aspect_ratio(self, peak_region) -> Dict[str, float]
    def calculate_bounding_box(self, peak_region) -> BoundingBox
```

**topographic_calculator.py**: Topographic features
```python
class TopographicCalculator(BaseCalculator):
    def calculate_isolation(self, peak, all_peaks, distance_metric) -> float
    def calculate_relative_height(self, peak, data, neighborhood_radius) -> float
    def calculate_topographic_position_index(self, peak, data) -> float
    def calculate_width_at_relative_height(self, peak, data, rel_height) -> float
```

**distance_calculator.py**: Distance & connectivity features
```python
class DistanceCalculator(BaseCalculator):
    def calculate_minkowski_distance(self, point1, point2, p, scale) -> float
    def find_nearest_higher_peak(self, peak, all_peaks) -> Tuple[Peak, float]
    def calculate_peak_density(self, center, peaks, radius) -> float
    def calculate_watershed_distance(self, peak1, peak2, data) -> float
```

#### 5. **connectivity/** - Connectivity Definition Layer
Connectivity and path finding in N-dimensional space

#### 6. **coordinate_system/** - Coordinate System Layer
Unified management of index space and coordinate space transformations

**grid_manager.py**: Index ↔ Coordinate conversion and spatial operations
```python
class GridManager:
    def __init__(self, mapping: CoordinateMapping, connectivity_level: int)
    def indices_to_coordinates(self, indices) -> Union[Tuple, np.ndarray]
    def coordinates_to_indices(self, coordinates) -> Union[Tuple, np.ndarray]
    def calculate_distance(self, coord1, coord2, metric='euclidean') -> float
    def get_neighbors_coordinates(self, center_coordinates) -> List[Tuple]
    def find_neighbors_in_radius(self, center, radius, metric) -> List[Tuple]

class Peak:
    center_indices: Tuple[int, ...]      # Internal processing indices
    center_coordinates: Tuple[float, ...] # User-facing coordinates
    plateau_indices: List[Tuple[int, ...]] # Region in index space
    @property
    def coordinate_dict(self) -> dict     # {x: 1.5, y: 2.3, z: 4.1}
```

**coordinate_mapping.py**: Mapping definitions
```python
@dataclass
class CoordinateMapping:
    indices_shape: Tuple[int, ...]     # Data array shape (I, J, K, ...)
    coordinate_origin: Tuple[float, ...] # Real-world origin (x0, y0, z0, ...)
    coordinate_spacing: Tuple[float, ...] # Physical spacing (dx, dy, dz, ...)
    axis_names: Tuple[str, ...] = ('x', 'y', 'z') # Coordinate axis names
```

#### 7. **boundary/** - Boundary Processing Layer
Proper handling at data boundaries

**boundary_handler.py**: Boundary condition processing
```python
class BoundaryHandler:
    def __init__(self, boundary_type, boundary_value=None)
    def extend_data_with_boundary(self, data) -> np.ndarray
    def remove_boundary_artifacts(self, peaks, min_distance) -> List[Peak]
    def handle_edge_effects(self, peaks, data_shape) -> List[Peak]

def infinite_height_boundary(data, pad_width=1) -> np.ndarray
def periodic_boundary(data, pad_width=1) -> np.ndarray
def custom_boundary(data, boundary_value, pad_width=1) -> np.ndarray
```

#### 7. **data/** - Data Management Layer
Efficient data processing and memory management

**lazy_dataframe.py**: Lazy evaluation dataframe
```python
class LazyDataFrame:
    def __init__(self, peak_collection, feature_calculators)
    def __getitem__(self, feature_name)  # Lazy computation
    def compute_batch(self, feature_names) -> Dict[str, np.ndarray]
    def cache_strategy(self, strategy='lru', max_size=None)
    def memory_usage(self) -> MemoryReport
```

### Data Flow and Interactions

```
1. Data Input & Preprocessing
   ├── Input validation (validation.py)
   ├── Boundary handling (boundary_handler.py)
   └── Data augmentation & normalization

2. Strategy Selection & Initialization
   ├── Data characteristics analysis (strategy_manager.py)
   ├── Optimal strategy selection
   └── Parameter tuning

3. Peak Detection Execution
   ├── Union-Find strategy OR Plateau-first strategy
   ├── Plateau detection & validation
   ├── Prominence calculation
   └── Virtual peak processing

4. Feature Calculation & Filtering
   ├── Lazy feature computation (lazy_feature_manager.py)
   ├── User-specified filter application
   └── Result data structure construction

5. Result Output & Visualization
   ├── DataFrame conversion
   ├── Statistical information generation
   └── Visualization & export
```

### Performance Optimization Features

- **Adaptive Strategy Selection**: Automatic optimal algorithm selection based on data characteristics
- **Lazy Evaluation**: Efficient memory usage by computing only required features
- **Multi-threading**: Parallelization of CPU-intensive computations
- **Caching System**: Intelligent caching of computational results
- **Memory Optimization**: Chunk processing for large-scale datasets
- **Profiling**: Real-time performance monitoring and bottleneck identification

### Detailed Function Specifications

#### **core/peak_detector.py**
```python
class PeakAnalyzer:
    def __init__(strategy, connectivity, boundary_condition, **params)
    def find_peaks(data) -> LazyPeakDataFrame
    def _select_strategy(data) -> Strategy
    def _validate_input(data) -> None
```

#### **core/union_find.py**
```python
class UnionFind:
    def __init__(size)
    def union(x, y)
    def find(x) -> int
    def get_components() -> Dict[int, List[int]]
    def connected(x, y) -> bool
```

#### **strategies/union_find_strategy.py**
```python
class UnionFindStrategy:
    def detect_peaks_and_prominence(data, connectivity, boundary) -> List[Peak]
    def _build_height_priority_graph(data) -> Graph
    def _trace_prominence_paths(peaks, graph) -> Dict[Peak, float]
```

#### **strategies/plateau_first_strategy.py**
```python
class PlateauFirstStrategy:
    def detect_peaks(data, connectivity, boundary) -> List[Peak]
    def _detect_plateaus(data) -> List[PlateauRegion]
    def _calculate_prominence_batch(plateaus) -> Dict[PlateauRegion, float]
```

#### **connectivity/connectivity_types.py**
```python
def get_k_connectivity(ndim: int, k: int) -> np.ndarray
def generate_k_connected_offsets(ndim: int, k: int) -> np.ndarray
# k=1: Face sharing (2n neighbors)
# k=2: Face + edge sharing 
# k=3: Face + edge + vertex sharing
# ...
# k=ndim: All boundary sharing (3^n-1 neighbors)
```

#### **boundary/boundary_conditions.py**
```python
class BoundaryHandler:
    def __init__(boundary_type: str, boundary_value: Optional[float])
    def extend_data_with_boundary(data) -> np.ndarray
    def remove_boundary_artifacts(peaks) -> List[Peak]
    
def infinite_height_boundary(data) -> np.ndarray
def infinite_depth_boundary(data) -> np.ndarray
def periodic_boundary(data) -> np.ndarray
```

### Feature Calculation Specifications

#### **Geometric Features**
- **Peak Plateau Area**: Number of cells in plateau region
- **Peak Volume**: Sum of height values in plateau region  
- **Centroid Position**: Weighted center of mass coordinates
- **Boundary Length/Area**: Perimeter of plateau region
- **Aspect Ratio**: Elongation measures in each dimension
- **Bounding Box**: Min/max coordinates in each dimension

#### **Topographic Features**  
- **Peak Height**: Maximum elevation in region
- **Prominence**: Vertical distance to lowest enclosing contour
- **Isolation**: Distance to nearest higher peak
- **Relative Height**: Height relative to local surroundings
- **Topographic Position Index**: Position relative to neighborhood elevation
- **Width at Relative Height (rel_height)**: Area of the peak region at a specified relative height between prominence_base and peak_height (e.g., rel_height=0.5 measures width at half-prominence level)

#### **Morphological Features**
- **Sharpness**: Local curvature around peak centroid
- **Average Slope**: Mean gradient magnitude in plateau region
- **Slope Variability**: Standard deviation of gradients
- **Directional Slopes**: Gradients in each coordinate direction
- **Laplacian**: Second-order derivative measures

#### **Distance and Connectivity Features**
- **Nearest Higher Peak Distance**: Configurable Minkowski distance (L_p norm) to closest higher peak
- **Nearest Similar Peak Distance**: Distance to peak of similar height using specified distance metric
- **Peak Density**: Number of peaks within specified radius (accounting for pixel scale)
- **Connectivity Order**: Hierarchical relationship to other peaks
- **Watershed Distance**: Distance following steepest descent paths
- **Scaled Distances**: All distance measurements respect pixel scale parameters for accurate physical measurements
- **Minkowski Distance Options**: Support for L1 (Manhattan), L2 (Euclidean), L∞ (Chebyshev), and custom Lp norms

### Algorithm Implementation Details

#### **Union-Find Strategy Algorithm**

##### **Core Challenge**: Height-Priority Processing Without False Peaks

**Step 1: Priority Queue Initialization**
- Create priority queue with all data points as (height, coordinates)
- Use max-heap to process from highest to lowest elevation
- Initialize Union-Find structure for all data points

**Step 2: Height-Level Batch Processing**
- **Critical Insight**: Process ALL points of same height simultaneously
- Extract all points with current maximum height from queue
- Create temporary batch of same-height points for unified processing

**Step 3: Same-Height Connectivity Analysis with Progressive Union**
- **Critical Issue**: Naive union within same-height batch creates false peaks
- **Solution**: Wave-front expansion from already-processed points

**Wave-Front Expansion Algorithm:**
```
processed_points = set()  # From previous height levels
current_batch = get_same_height_points(current_height)
newly_processed = set()  # Points processed in current iteration

# Iterative wave-front expansion
while True:
    temp_store = set()  # Temporarily store newly connected points
    
    # Find unprocessed points connected to processed points
    for point in current_batch:
        if point not in newly_processed:
            for neighbor in get_k_neighbors(point):
                if neighbor in processed_points or neighbor in newly_processed:
                    # Point connects to already processed terrain
                    temp_store.add(point)
                    
                    # Union operations
                    if not has_region(point):
                        # Point-to-region union (first connection)
                        neighbor_region = find_region(neighbor)
                        union_point_to_region(point, neighbor_region)
                    else:
                        # Region-to-region union (subsequent connection)
                        point_region = find_region(point)
                        neighbor_region = find_region(neighbor)
                        if point_region != neighbor_region:
                            union_regions(point_region, neighbor_region)
                    break
    
    # No new connections found - terminate
    if not temp_store:
        break
    
    # Add newly connected points to processed set
    newly_processed.update(temp_store)

# Handle remaining unprocessed points (potential new peaks)
remaining_points = current_batch - newly_processed
isolated_components = find_connected_components(remaining_points, k_connectivity)
for component in isolated_components:
    register_as_new_peak_candidate(component)
```

**Key Benefits:**
- **Prevents False Peaks**: No interior plateau points processed independently
- **Maintains Connectivity**: Proper region merging when multiple processed neighbors exist
- **Wave-Front Processing**: Mimics natural water-flow expansion from higher terrain

**Step 4: Region Validation and Peak Detection**
- **Seed-Connected Regions**: Mark as non-peak (connected to higher terrain)
- **Isolated Regions**: Mark as peak candidates (no connection to higher terrain)
- **Orphaned Points**: Individual points not connected to any seeds → potential new peaks

**Step 4: Union-Find Integration**
- For each plateau component:
  1. Union all points within the component
  2. Check connectivity to previously processed higher plateaus
  3. If connected to higher terrain, mark as non-peak plateau
  4. If isolated from higher terrain, mark as candidate peak plateau

**Step 5: Prominence Calculation During Traversal**
- As we descend through height levels:
  1. Track saddle points for each peak plateau
  2. When a peak connects to higher terrain, record saddle elevation
  3. Calculate prominence as (peak_height - saddle_height)

**Step 6: False Peak Prevention**
- **Key Strategy**: Never process individual points of same-height plateaus separately
- Always process entire same-height connected components as atomic units
- Reject any component that connects to equal-or-higher elevation terrain

##### **Queue Management Strategy**
```
while priority_queue not empty:
    current_height = peek_max_height(queue)
    same_height_batch = extract_all_with_height(queue, current_height)
    
    # Process entire batch atomically
    components = find_connected_components(same_height_batch, k_connectivity)
    
    for component in components:
        if is_isolated_from_higher_terrain(component):
            register_as_peak(component)
        else:
            mark_as_non_peak(component)
        
        union_all_points_in_component(component)
```

##### **Union-Find Integration Logic**
- **Progressive Union Strategy**: Start from processed points, expand into unprocessed same-height points
- **Two-Phase Union**:
  1. **Point-to-Region**: Unprocessed point joins existing region (first connection)
  2. **Region-to-Region**: Existing regions merge when connected via unprocessed point (subsequent connections)
- **Seed-Based Processing**: Only points connected to higher processed terrain act as integration seeds
- **Prominence Tracking**: Maintain saddle elevation for each region root during expansion

#### **Plateau-First Strategy Algorithm**  

##### **Phase 1: Plateau Detection Logic**

**Step 1: Local Maximum Identification**
- For each cell (i,j,...), apply local maximum filter using specified k-connectivity
- Cell is candidate if `data[i,j,...] >= max(all_k_connected_neighbors)`
- This creates binary mask of potential peak cells
- **Issue**: Non-peak plateaus are also detected by this filter

**Step 2: Connected Component Analysis**
- Among candidate cells, group those with identical height values
- Use Union-Find or flood-fill to find connected components using k-connectivity
- Each component represents a potential plateau region of constant height

**Step 3: Plateau Validation (Dilation Test)**
- **Key Insight**: True peak plateaus vs. non-peak plateaus behave differently under dilation
- For each connected component of height `h`:
  1. Create binary mask of the component
  2. Apply morphological dilation using k-connectivity structuring element
  3. Check dilated boundary: `dilated_mask AND NOT original_mask`
  4. **Critical Logic**: If ANY boundary cell has height = `h` (same height), reject as non-peak plateau
  5. If ALL boundary cells have height < `h` (strictly lower), accept as true peak plateau

**Reasoning**: 
- True peak plateaus: Dilation boundary will always be strictly lower
- Non-peak plateaus: Dilation boundary will contain cells of same height (connected to higher terrain). Note that dilation boundary may include both original plateau interior points (missed by local maximum filter due to connection to higher regions) and external points

##### **Phase 2: Prominence Calculation**
- For each validated plateau, perform breadth-first search from boundary
- Track minimum elevation until reaching higher terrain
- Calculate prominence as height difference

#### **Edge/Boundary Handling**
- **Infinite Height Boundary**: Pad data edges with maximum float value
- **Infinite Depth Boundary**: Pad data edges with minimum float value  
- **Periodic Boundary**: Wrap data edges with opposite edge values
- **Custom Boundary**: User-specified constant values at edges
- **Artifact Removal**: Filter out peaks too close to data boundaries
1-Connectivity**: Face sharing (2n neighbors)
- **2-Connectivity**: Face + edge sharing
- **3-Connectivity**: Face + edge + vertex sharing  
- **...**
- **n-Connectivity**: All boundary sharing (3^n-1 neighbors)
- **Efficiency**: Precomputed offset arrays for each connectivity level
- **Custom Patterns**: User-defined neighbor offset patterns
- **Efficiency**: Precomputed offset arrays for fast neighbor generation

### Performance and Memory Considerations
- **Lazy Evaluation**: Features computed only when requested
- **Memory Mapping**: Large arrays handled via memory-mapped files
- **Chunk Processing**: Data divided into overlapping chunks for memory efficiency
- **Parallel Processing**: Multi-threaded feature calculation
- **Caching**: Intelligent caching of expensive computations