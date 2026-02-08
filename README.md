# Status: Work in Progress (WIP)

This project is currently a prototype under development. Algorithm implementations and refactoring are ongoing, and the structure may undergo significant changes.

# peak-analyzer

A multidimensional peak detection library that treats peaks as topographic regions rather than isolated points, addressing fundamental limitations in existing 2D peak detection algorithms.

## Why PeakAnalyzer?

Existing 2D peak detection algorithms frequently encounter critical issues:

1. **Plateau Misidentification**: Areas of constant height (plateaus) are incorrectly flagged as multiple individual peaks.
2. **Lack of Feature-Based Filtering**: Most tools cannot filter peaks based on geomorphological features such as prominence or isolation.

**PeakAnalyzer** addresses these issues by redefining peaks as **topographic regions** with comprehensive feature extraction and filtering capabilities.

## Key Features

### Topographic Feature Extraction
- **Peak Coordinates**: Centroid or representative point of the peak region
- **Height**: Absolute value at the peak
- **Prominence**: Vertical distance between peak and its lowest contour line
- **Area**: Spatial extent of the peak region
- **Sharpness**: Local curvature or rate of height change
- **Isolation**: Distance to the nearest higher peak
- **Distance**: Spatial relationship metrics between detected features

### Rigorous Plateau Handling
- Treats peaks as "regions of equal height" to prevent artifact generation
- Uses morphological dilation validation to distinguish true peak plateaus from non-peak plateaus

### Coordinate System Support
- Clear separation between **Index Space** (i,j,...) for internal processing and **Coordinate Space** (x,y,...) for user interaction
- Real-world physical coordinates with pixel scale parameters
- Support for anisotropic resolutions and GIS-compatible operations

### Lazy Feature Calculation
- Features computed only when requested or required for filtering
- Memory-efficient processing for large datasets

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv add ".[dev,examples]"

# Or using pip
pip install -e ".[dev,examples]"
```

### Basic Usage

```python
from peak_analyzer import PeakAnalyzer
import numpy as np

# Generate sample 2D data
data = np.random.randn(100, 100) + 5

# Initialize analyzer
analyzer = PeakAnalyzer(
    strategy='auto',
    connectivity='face', 
    boundary='infinite_height'
)

# Find peaks
peaks = analyzer.find_peaks(data)

# Filter by prominence and area
significant_peaks = peaks.filter(
    prominence=lambda p: p > 0.5,
    area=lambda a: a > 5
)

# Access features
print(f"Found {len(significant_peaks)} significant peaks")
features_df = significant_peaks.to_pandas()
print(features_df[['coordinates', 'height', 'prominence', 'area']])
```

### Advanced Usage with Physical Coordinates

```python
# Configure real-world coordinates
analyzer = PeakAnalyzer(
    scale=[1.0, 0.5],  # Different x/y resolution
    distance_metric='euclidean'
)

# Find peaks with feature calculation
peaks = analyzer.find_peaks(data, 
                          prominence_threshold=0.3,
                          min_area=3)

# Calculate additional features
features = peaks.get_features(['isolation', 'sharpness', 'aspect_ratio'])

# Spatial filtering by coordinate range
regional_peaks = peaks.filter_by_coordinates(
    x_range=(-10, 10),
    y_range=(20, 40)
)
```

## Documentation

- **[Algorithm Details](docs/algorithms/algorithm.md)**: Core algorithm theory and implementation details
- **[Development Guide](DEVELOPMENT.md)**: Setup and contribution guidelines
- **[API Reference](docs/api/)**: Comprehensive API documentation
- **[Examples](examples/)**: Usage examples and demonstrations
- **[日本語版README](README_japanese.md)**: Japanese version of this documentation

## Performance Considerations

- **Adaptive Strategy Selection**: Automatic optimal algorithm selection based on data characteristics
- **Memory Optimization**: Chunk processing and memory mapping for large datasets
- **Parallel Processing**: Multi-threaded feature calculation
- **Intelligent Caching**: Strategic caching of expensive computations

## License

[Specify license here]

## Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions and contribution guidelines.