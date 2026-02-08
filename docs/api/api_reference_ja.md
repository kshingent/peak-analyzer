# APIリファレンス

_← [アルゴリズム](../algorithms/algorithm_ja.md) | [アーキテクチャ](../architecture/architecture_ja.md) | [English Version](api_reference.md) →_

## コアクラスと関数

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

## 戦略クラス

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

## データモデル

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

## APIレイヤークラス

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
    def __getitem__(self, key)  # 遅延計算をトリガー
    def cache_features(self, features)
    def clear_cache(self)
```

## コアアルゴリズムクラス

### **core/strategy_manager.py**
```python
class StrategyManager:
    def select_optimal_strategy(self, data, **kwargs) -> Strategy
        # **kwargs は手動指定のため 'strategy_name' または 'force_strategy' をサポート
    def estimate_computational_cost(self, strategy_name, data_shape) -> dict[str, float]
    def benchmark_strategies(self, data, strategies=None) -> list[BenchmarkResults]
    def configure_strategy(self, strategy_name, **params) -> Strategy
    def auto_configure(self, data_shape, characteristics=None, 
                      performance_requirements=None) -> Strategy
        # パフォーマンス要件に基づいて最適な戦略を自動設定
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

## 特徴量計算クラス

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

## 接続性と座標系クラス

### **connectivity/connectivity_types.py**
```python
class Connectivity:
    def __init__(self, ndim: int, k: int)
    def get_neighbors(self, center: tuple, shape: tuple) -> np.ndarray
    @property
    def neighbor_count(self) -> int
# N次元k接続性実装
# k=1: 面共有接続性, k=ndim: 完全 (Moore) 接続性
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
    center_indices: tuple[int, ...]      # 内部処理インデックス
    center_coordinates: tuple[float, ...] # ユーザー向け座標
    plateau_indices: list[tuple[int, ...]] # インデックス空間の領域
    @property
    def coordinate_dict(self) -> dict     # {x: 1.5, y: 2.3, z: 4.1}
```

### **coordinate_system/coordinate_mapping.py**
```python
@dataclass
class CoordinateMapping:
    indices_shape: tuple[int, ...]     # データ配列形状 (I, J, K, ...)
    coordinate_origin: tuple[float, ...] # 実世界原点 (x0, y0, z0, ...)
    coordinate_spacing: tuple[float, ...] # 物理的間隔 (dx, dy, dz, ...)
    axis_names: tuple[str, ...] = ('x', 'y', 'z') # 座標軸名
```

## 境界処理

境界処理はシンプルなパディングで実装されます：

- **infinite_height**: データを無限値でパディング（デフォルト）
- **infinite_depth**: データを負の無限値でパディング

```python
# シンプルなパディング実装
padded_data = np.pad(data, 1, mode='constant', constant_values=pad_value)
```

境界値の扱い：
- プロミネンス計算中、無限値は効率のためスキップされる
- パディングは常に1ピクセル固定

## 境界条件とプロミネンス計算の制約

### **境界条件の選択制約**

**実用的な境界条件は2種類のみ:**

**1. infinite_height（推奨・デフォルト）**
- 全てのピークでプロミネンス計算が可能
- 境界を無限に高い壁として扱う
- 地形学的に最も自然な解釈

**2. infinite_depth**  
- 最高ピーク以外でプロミネンス計算が可能
- 最高ピークのプロミネンス = 最高ピーク高度 - 最低地点高度
- 境界を無限に深い谷として扱う

**periodic, constant, mirror, nearest等の問題:**
- 境界の向こう側に「仮想的な地形」を作り出す
- プロミネンス計算が境界条件の仮定に依存してしまう  
- 地形学的意味を持たない結果になる
- **プロミネンス計算には適用不可**

### **infinite_depthでの最高ピーク処理**
```python
# infinite_depthの場合の最高ピーク処理
global_max_peak = find_global_maximum(peaks)
global_min_height = find_global_minimum(data)

for peak in peaks:
    if peak == global_max_peak:
        # 最高ピークのプロミネンス = 最低地点からの高さ
        peak.prominence = peak.height - global_min_height
    else:
        # 他のピークは通常計算
        peak.prominence = calculate_standard_prominence(peak)
```

### **境界条件選択の指針**
- **デフォルト**: `infinite_height` - 最も安全で一般的
- **特殊ケース**: `infinite_depth` - 境界近くのピークも重視する場合
- **その他**: ピーク検出のみで使用、プロミネンス計算は無効化される

## データ管理クラス

### **data/lazy_dataframe.py**
```python
class LazyDataFrame:
    def __init__(self, peak_collection, feature_calculators)
    def __getitem__(self, feature_name)  # 遅延計算
    def compute_batch(self, feature_names) -> dict[str, np.ndarray]
    def cache_strategy(self, strategy='lru', max_size=None)
    def memory_usage(self) -> MemoryReport
```

## 特徴量計算仕様

### **幾何学的特徴量**
- **ピークプラトー面積**: プラトー領域内のセル数
- **ピーク体積**: プラトー領域内の高度値の合計
- **重心位置**: 質量中心座標の重み付き平均
- **境界長/面積**: プラトー領域の周囲長
- **アスペクト比**: 各次元における伸長度指標
- **境界ボックス**: 各次元の最小/最大座標

### **地形学的特徴量**  
- **ピーク高度**: 領域内の最大標高
- **プロミネンス**: 最低囲繞等高線までの垂直距離
- **孤立度**: 最も近い高いピークまでの距離
- **相対高度**: 局所周辺に対する相対的高度
- **地形位置指数**: 近傍標高に対する相対位置
- **相対高度幅 (rel_height)**: prominence_baseとpeak_heightの間の指定相対高度におけるピーク領域面積 (例: rel_height=0.5で半プロミネンスレベルでの幅を測定)

### **形態学的特徴量**
- **鋭さ**: ピーク重心周辺の局所曲率
- **平均傾斜**: プラトー領域における勾配大きさの平均
- **傾斜変動性**: 勾配の標準偏差
- **方向別傾斜**: 各座標方向の勾配
- **ラプラシアン**: 二階微分指標

### **距離・接続性特徴量**
- **最近高ピーク距離**: 最も近い高いピークまでの設定可能ミンコフスキー距離 (L_p ノルム)
- **類似ピーク距離**: 指定距離メトリックを使用した類似高度ピークまでの距離
- **ピーク密度**: 指定半径内のピーク数 (ピクセルスケール考慮)
- **接続次数**: 他ピークとの階層関係
- **流域距離**: 最急降下経路に沿った距離
- **スケール距離**: 全ての距離測定はピクセルスケールパラメータを尊重し、正確な物理距離測定を実現
- **ミンコフスキー距離オプション**: L1 (マンハッタン), L2 (ユークリッド), L∞ (チェビシェフ), カスタムLpノルムをサポート