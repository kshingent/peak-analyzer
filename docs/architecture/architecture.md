# Software Architecture Design

_← [Back to Algorithm](../algorithms/algorithm.md) | [日本語版](architecture_ja.md) →_

## Architecture Overview

PeakAnalyzer employs a layered, modular architecture that provides a comprehensive framework for topographic peak detection. Each layer has clearly separated responsibilities, ensuring extensibility and maintainability.

```mermaid
flowchart TD
    A["User API Layer<br/>PeakAnalyzer<br/>(peak_detector.py)"] --> B["Core Control Layer<br/>StrategyManager<br/>(strategy_manager.py)"]
    B --> C["Algorithm Layer<br/>UnionFindStrategy | PlateauFirst Strategy"]
    C --> D["Feature Computation Layer<br/>Geometric Features | Topographic Features | Distance Features"]
    D --> E["Foundation Services Layer<br/>Connectivity & Neighbors | Boundary Handling | Validation & Memory"]
    
    classDef apiLayer fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef coreLayer fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef algorithmLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef featureLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef foundationLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A apiLayer
    class B coreLayer
    class C algorithmLayer
    class D featureLayer
    class E foundationLayer
```

## Directory Structure and Responsibility Separation

```
peak_analyzer/
├── peak_analyzer/                    # Main package
│   ├── __init__.py                   # Package entry point
│   ├── models/                      # Central data structure definitions
│   │   ├── __init__.py              # Package entry point
│   │   ├── peaks.py                 # Peak detection results
│   │   └── data_analysis.py         # Analysis metadata structures
│   ├── api/                         # User API Layer
│   │   ├── peak_detector.py         # Main analysis class
│   │   ├── result_dataframe.py      # Result dataframe processing
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
│   │   └── neighbor_generator.py    # Simplified neighbor generation
│   │
│   ├── coordinate_system/           # Coordinate System Layer
│   │   ├── grid_manager.py         # Index ↔ Coordinate conversion & spatial operations
│   │   ├── coordinate_mapping.py   # Mapping definitions and validation
│   │   └── spatial_indexing.py     # Spatial indexing and search acceleration
│   │
   ├── data/                       # Data Management Layer
   │   └── validation.py           # Input validation
   │
   └── utils/                      # Utility Layer
       └── general.py              # General utility functions
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

## Layer-by-Layer Detailed Specifications

### 1. **api/** - User API Layer
Provides unified interface for topographic peak analysis

**models/**: Central data structure definitions
- **peaks.py**: Peak detection result structures (Peak, VirtualPeak, SaddlePoint)
- **data_analysis.py**: Analysis metadata structures (DataCharacteristics, BenchmarkResults)

### 2. **core/** - Core Algorithm Layer
Implements core algorithms for topographic analysis

**strategy_manager.py**: Strategy selection & management
**plateau_detector.py**: Plateau detection core
**prominence_calculator.py**: Prominence calculation core
**virtual_peak_handler.py**: Virtual peak processing

### 3. **strategies/** - Detection Strategy Implementation Layer
Optimized detection algorithms for different data characteristics

**base_strategy.py**: Common strategy interface
**union_find_strategy.py**: Union-Find strategy
**plateau_first_strategy.py**: Plateau-first strategy

### 4. **features/** - Feature Calculation Layer
Comprehensive computation framework for topographic features

**geometric_calculator.py**: Geometric features
**topographic_calculator.py**: Topographic features
**distance_calculator.py**: Distance & connectivity features

### 5. **connectivity/** - Connectivity Definition Layer
Connectivity in N-dimensional space

### 6. **coordinate_system/** - Coordinate System Layer
Unified management of index space and coordinate space transformations

**grid_manager.py**: Index ↔ Coordinate conversion and spatial operations
**coordinate_mapping.py**: Mapping definitions
**spatial_indexing.py**: Spatial indexing and search acceleration

### 7. **data/** - Data Management Layer
Input validation and data preprocessing

**validation.py**: Input validation and data preprocessing

### 8. **utils/** - Utility Layer
General utility functions

**general.py**: Common utility functions and helpers

## Data Flow and Interactions

```mermaid
flowchart TD
    A[Data Input & Preprocessing] --> A1[Input validation<br/>validation.py]
    A --> A3[Data augmentation & normalization]
    
    A1 --> B[Strategy Selection & Initialization]
    A3 --> B
    
    B --> B1[Data characteristics analysis<br/>strategy_manager.py]
    B --> B2[Optimal strategy selection]
    B --> B3[Parameter tuning]
    
    B1 --> C[Peak Detection Execution]
    B2 --> C
    B3 --> C
    
    C --> C1[Union-Find strategy OR<br/>Plateau-first strategy]
    C --> C2[Plateau detection & validation]
    C --> C3[Prominence calculation]
    C --> C4[Virtual peak processing]
    
    C1 --> D[Feature Calculation & Filtering]
    C2 --> D
    C3 --> D
    C4 --> D
    
    D --> D1[Lazy feature computation<br/>lazy_feature_manager.py]
    D --> D2[User-specified filter application]
    D --> D3[Result data structure construction]
    
    D1 --> E[Result Output & Visualization]
    D2 --> E
    D3 --> E
    
    E --> E1[DataFrame conversion]
    E --> E2[Statistical information generation]
    E --> E3[Visualization & export]
    
    classDef inputPhase fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef strategyPhase fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef detectionPhase fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef featurePhase fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef outputPhase fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A,A1,A2,A3 inputPhase
    class B,B1,B2,B3 strategyPhase
    class C,C1,C2,C3,C4 detectionPhase
    class D,D1,D2,D3 featurePhase
    class E,E1,E2,E3 outputPhase
```

## Performance Optimization Features

- **Adaptive Strategy Selection**: Automatic optimal algorithm selection based on data characteristics
- **Lazy Evaluation**: Efficient memory usage by computing only required features
- **Multi-threading**: Parallelization of CPU-intensive computations
- **Caching System**: Intelligent caching of computational results
- **Memory Optimization**: Chunk processing for large-scale datasets
- **Profiling**: Real-time performance monitoring and bottleneck identification