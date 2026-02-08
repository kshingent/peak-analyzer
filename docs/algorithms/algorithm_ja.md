# コアアルゴリズム: 地形学的多次元ピーク検出

_← [メインREADMEに戻る](../../README_japanese.md) | [English Version](algorithm.md) →_

## 高次元における1Dロジックの限界

`scipy.signal.find_peaks` は非常に優れた 1 次元ピーク抽出アルゴリズムですが、そのロジック自体が 1 次元地形に特化して設計されています。

1D ではピークは「左右より高い点」として定義できますが、2D 以上では事情がまったく異なります。

既存の 2D ピーク抽出アルゴリズムには、特に次の 2 つの重大な問題があります。

1. **プラトー (plateau) をピークと誤認識する**
   * 同じ高さが広がる領域をピークと誤判定してしまう
2. **ピークの選別ができない**
   * プロミネンスや isolation といった地誌学的特徴量を計算しても、それを用いてピークをフィルタリングする機構が存在しない

これらはすべて、「ピークを点として扱っている」ことに起因します。

PeakAnalyzer はこれを根本から見直し、

> **ピークを "点" ではなく "地形構造 (topographic region)" として扱う**

ことで、多次元データに適用可能なピーク抽出を実現します。

---

## 抽出される地形学的特徴

PeakAnalyzer は単なるピーク位置検出ではなく、ピークの地形的性質を定量化します。

* **ピーク座標**: ピーク領域の重心または代表点
* **高さ**: ピークでの絶対値
* **プロミネンス**: ピークとその最低等高線との垂直距離
* **面積**: ピーク領域の空間的範囲
* **鋭さ**: 局所曲率または高度変化率
* **アイソレーション**: 最も近い高いピークまでの距離
* **距離**: 検出された特徴間の空間関係メトリック

### ピクセルスケールと距離メトリック

* **ピクセルスケールパラメータ**: 各次元の実世界スケールを指定可能（例：`scale=[1.0, 0.5]` でx/y解像度の違いを設定）
* **ミンコフスキー距離**: 距離計算ではパラメータ `p` でミンコフスキー距離メトリックを設定可能（p=1: マンハッタン、p=2: ユークリッド、p=∞: チェビシェフ）
* **距離スケール**: スケールパラメータにより物理単位が保持され、正確な実世界距離測定を保証
* **相対高度 (rel_height)**: prominence_baseとpeak_heightの間の相対高度比でピーク幅（面積）を定義

### 座標系アーキテクチャ

**インデックス空間と座標空間の分離**

PeakAnalyzerは2つの空間表現を明確に区別して管理します：

* **インデックス空間 (i,j,...)**: 計算効率のための内部データ配列インデックス
  - 用途: アルゴリズム処理、メモリアクセス、接続性解析
  - 形式: データ配列次元に対応する整数インデックス
  - 例: `peak_indices = (127, 89)` （2次元配列アクセス用）

* **座標空間 (x,y,...)**: ユーザー操作のための実世界物理座標（存在する場合）
  - 用途: ユーザー入出力、可視化、空間解析、フィルタリング
  - 形式: 物理単位を持つ浮動小数点座標
  - 例: `peak_coordinates = (-120.5, 36.2)` （経度/緯度）、データ値 = 1450.0（任意の物理単位）

**物理的意味と実世界単位**
* **距離・面積計算**: すべての測定が実際の物理単位で計算される
* **異方性解像度対応**: 各軸で異なるスケール設定可能（`scale=[1.0, 1.0, 0.5]`）
* **単位保持**: 解析パイプライン全体で一貫した物理単位を維持
* **スケール考慮特徴**: 幾何学的特性が実世界の寸法を尊重

**空間解析機能**
* **GIS互換操作**: 座標ベースのフィルタリング、バッファリング、空間クエリ
* **地理的統合**: 地理座標系のネイティブサポート
* **マルチスケール解析**: 異なる解像度データのシームレス処理
* **投影サポート**: 柔軟な座標系変換

**可視化とユーザーインターフェース原則**
* **ユーザー中心座標**: すべてのユーザー操作を直感的なxyz...座標空間で実行
* **実世界可視化**: プロットと表示は適切な単位を持つ物理座標を使用
* **インデックス抽象化**: 内部ijk...インデックスをユーザーインターフェースから隠蔽
* **直感的フィルタリング**: 空間フィルターが意味のある座標範囲で動作

これらの特徴により、「検出」ではなく **意味のあるピークの選別** が可能になります。

---

## 厳密なプラトー処理

多くの 2D アルゴリズムが失敗する最大の理由は **プラトーを正しく扱えていない** ことです。

PeakAnalyzer では、まずピークを「点」ではなく「同じ高さを持つ領域」として扱います。

### プラトー検出手順

1. **最大フィルタリング**: データセットに局所最大フィルタを適用
2. **膨張処理**: 識別された一定高度領域に形態学的膨張を実行
3. **検証**: 膨張境界に同じ高度のセルが存在しない場合、その領域を **真のピークプラトー** として分類

この処理が欠けているために、多くの 2D 手法ではプラトー内部に誤ったピークが生成されます。

---

## プロミネンス計算

PeakAnalyzer はプロミネンスの地形学的定義に厳密に従い、それを多次元空間に拡張します：

1. ピーク領域から高度優先近傍探索を開始
2. 現在のピークより高い値を持つ点に遭遇するまで探索を続行
3. この探索中に到達した最低標高をprominence_baseと定義
4. プロミネンスを計算: `prominence = peak_height - prominence_base`

### ウィンドウ長 (wlen) パラメータ

`wlen` パラメータはプロミネンス計算を指定距離内に制限します：
* `wlen` が指定されると、ピークから指定距離を超える点でプロミネンス計算が停止
* これにより解析領域を完全データセットのウィンドウ化サブセットとして扱う
* 特定の空間スケール内でのローカルプロミネンスに焦点を当てる際に有用

### 最高標高の場合のプロミネンス処理

大域的に最高であるピーク（より高いピークが存在しない）の場合：
* **大域最大規則**: プロミネンスはデータ境界の最低点との高度差に等しい
* **境界条件**: 無限境界を使用する場合、プロミネンス計算は実際のデータエッジまで追跡

### 仮想ピーク作成による同高度接続ピーク処理

同じ標高を持ち地形的に接続している複数のピークが存在する場合：
* **任意の標高での接続同高度ピーク**:
  - 同じ高さ`h`の複数のピークA、Bが存在し、高さ≥`h`の地形を通じて接続している場合
  - **ステップ1**: 高さ≥`h`の地形を介して接続した同じ標高`h`のすべてのピークを検出
  - **ステップ2**: 接続した同高度ピークA、B間の鞍点を発見
  - **ステップ3**: 個別ピークAとBのprominence_baseを接続する鞍点の高さに設定
  - **ステップ4**: 高さ`h`のすべての接続ピークを包含する仮想ピークを作成
  - **ステップ5**: より高い高さ`h' > h`の地形が仮想ピークグループ内のいずれかのピークに鞍点を介して接続する場合、仮想ピークのprominence_baseをその鞍点の高さに設定
  - **ステップ6**: 仮想ピークのプロミネンス = `h - より高い地形への鞍点高度`
  - **ステップ7**: より高い地形が存在しない場合、境界ベースのプロミネンス計算を使用
  - このアプローチにより、接続ピークが存在する任意の標高レベルで一貫したプロミネンス計算を保証

経路探索ロジックを通じて実装することで、アルゴリズムは次元に依存しません。

---

## 計算戦略

ライブラリは、データの「地形」に応じて2つの戦略を提供します：

| 戦略 | ロジック | 最適な使用ケース |
| --- | --- | --- |
| **独立計算** | 最初にすべてのプラトーを識別し、次に各々について個別にプロミネンスを計算 | 高コントラストで孤立した鋭いピークを持つデータ |
| **バッチヒープ探索** | ヒープ優先キューを使用して高度順に地形を探索 | 滑らかな地形または広範囲で複雑なプラトー構造を持つデータ |

---

## 遅延評価による効率性

すべての潜在的ピークに対してすべての地形学的特徴を計算することは、計算コストが高くなります。PeakAnalyzer は性能を最適化するため、**LazyDataFrame** アプローチを利用します。

* **オンデマンド計算**: 特徴は明示的に要求されるか、フィルタリング操作に必要な場合にのみ計算される
* **動的プロセス**: ユーザーが `prominence > 0.5` でフィルタリングする場合、アルゴリズムはプロミネンスのみを計算。その後 `area` フィルタが追加されれば、残りの候補に対して面積を計算
* **メモリ効率**: 最終的に破棄されるピークに対する大きな特徴行列の維持オーバーヘッドを防ぐ

---

## 設計原則

1. **地形学的焦点**: ピークを次元のない点ではなく、領域として扱う
2. **厳密なプラトーロジック**: 平坦な表面がアーティファクトの生成なしに処理されることを保証
3. **地形学的精度**: 1Dプロミネンス定義を簡略化なしにN次元に拡張
4. **最適化された探索**: ヒープ優先構造を利用して次元間の論理的一貫性を維持
5. **計算経済性**: 大規模データセットでの不要な計算を最小化するため遅延評価を使用
6. **座標系分離**: 内部処理用インデックス空間(ijk...)とユーザー操作用座標空間(xyz...)の明確な区別を維持
7. **物理的意味優先**: すべてのユーザー向け測定と可視化で実世界単位と座標を使用
8. **異方性互換**: 次元間での異なる解像度とスケールのネイティブサポート
9. **空間解析統合**: GIS的な座標ベース解析とフィルタリング機能を有効化

---

## ソフトウェアアーキテクチャ設計

### アーキテクチャ概観

PeakAnalyzerは階層化されたモジュラー設計を採用し、地形学的ピーク検出のための包括的なフレームワークを提供します。各層は明確に分離された責任を持ち、拡張性と保守性を確保しています。

```
              ユーザーAPI層
┌─────────────────────────────────────────┐
│           PeakAnalyzer                  │  <- メインインターフェース
│        (peak_detector.py)               │
└─────────────┬───────────────────────────┘
              │
           コア制御層
┌─────────────┴───────────────────────────┐
│      StrategySelector & Manager         │  <- アルゴリズム選択・調整
│    (strategy_selector.py)               │
└─────────────┬───────────────────────────┘
              │
          アルゴリズム層
┌─────────────┴───────────────────────────┐
│  UnionFindStrategy  │  PlateauFirst     │  <- 検出戦略実装
│                     │  Strategy         │
└─────────────┬───────┴───────────────────┘
              │
           特徴計算層
┌─────────────┴───────────────────────────┐
│  Geometric   │ Topographic │ Distance  │  <- 地形特徴計算
│  Features    │ Features    │ Features   │
└─────────────┬───────────────────────────┘
              │
          基盤サービス層
┌─────────────┴───────────────────────────┐
│ Connectivity │ Boundary │ Validation   │  <- 基本機能・ユーティリティ
│ & Neighbors  │ Handling │ & Memory     │
└─────────────────────────────────────────┘
```

### ディレクトリ構造と責任分離

```
peak_analyzer/
├── peak_analyzer/                    # メインパッケージ
│   ├── __init__.py                   # パッケージエントリポイント
│   ├── api/                         # ユーザーAPI層
│   │   ├── peak_detector.py         # メイン分析クラス
│   │   ├── result_dataframe.py      # 結果データ構造
│   │   └── parameter_validation.py  # パラメータ検証
│   │
│   ├── core/                        # コアアルゴリズム層
│   │   ├── strategy_manager.py      # 戦略選択・管理
│   │   ├── plateau_detector.py      # プラトー検出コア
│   │   ├── prominence_calculator.py # プロミネンス計算コア
│   │   ├── virtual_peak_handler.py  # 仮想ピーク処理
│   │   └── union_find.py           # 連結成分データ構造
│   │
│   ├── strategies/                  # 検出戦略実装層
│   │   ├── base_strategy.py         # 戦略ベースクラス
│   │   ├── union_find_strategy.py   # Union-Find戦略
│   │   ├── plateau_first_strategy.py # プラトー優先戦略
│   │   ├── hybrid_strategy.py       # ハイブリッド戦略
│   │   └── strategy_factory.py      # 戦略ファクトリー
│   │
│   ├── features/                     # 特徴計算層
│   │   ├── base_calculator.py       # 計算ベースクラス
│   │   ├── geometric_calculator.py  # 幾何学的特徴
│   │   ├── topographic_calculator.py # 地形学的特徴
│   │   ├── morphological_calculator.py # 形態学的特徴
│   │   ├── distance_calculator.py   # 距離・接続性特徴
│   │   └── lazy_feature_manager.py  # 遅延計算マネージャー
│   │
│   ├── connectivity/                 # 接続性定義層
│   │   ├── connectivity_types.py    # N次元接続性定義
│   │   ├── neighbor_generator.py    # 近傍生成器
│   │   ├── path_finder.py          # 経路探索アルゴリズム
│   │   └── distance_metrics.py      # 距離メトリック実装
│   │
│   ├── coordinate_system/           # 座標系層
│   │   ├── grid_manager.py         # インデックス ↔ 座標変換・空間操作
│   │   ├── coordinate_mapping.py   # マッピング定義と検証
│   │   └── spatial_indexing.py     # 空間インデックスと検索高速化
│   │
│   ├── boundary/                    # 境界処理層
│   │   ├── boundary_handler.py      # 境界条件処理
│   │   ├── edge_detector.py        # エッジ効果検出
│   │   ├── padding_strategies.py   # パディング戦略
│   │   └── artifact_filter.py      # アーティファクト除去
│   │
│   ├── data/                       # データ管理層
│   │   ├── lazy_dataframe.py       # 遅延評価データフレーム
│   │   ├── peak_collection.py      # ピーク集合管理
│   │   ├── cache_manager.py        # キャッシュ管理
│   │   └── memory_optimizer.py     # メモリ最適化
│   │
│   └── utils/                      # ユーティリティ層
│       ├── validation.py           # 入力検証
│       ├── performance_profiler.py # 性能プロファイリング
│       ├── error_handling.py       # エラー処理
│       ├── logging_config.py       # ログ設定
│       └── type_definitions.py     # 型定義
│
├── tests/                          # テストスイート
│   ├── unit/                      # 単体テスト
│   ├── integration/               # 統合テスト
│   ├── performance/               # 性能テスト
│   └── fixtures/                  # テストデータ
│
├── examples/                       # 使用例・デモ
│   ├── basic_usage.py             # 基本使用法
│   ├── advanced_features.py       # 高度な機能
│   ├── custom_strategies.py       # カスタム戦略
│   └── benchmarking.py           # ベンチマーク例
│
├── docs/                          # ドキュメント
│   ├── api/                      # API仕様
│   ├── tutorials/                # チュートリアル
│   ├── algorithms/               # アルゴリズム詳細
│   └── examples/                 # 使用例詳細
│
└── benchmarks/                    # 性能ベンチマーク
    ├── synthetic_data/           # 合成データテスト
    ├── real_world_data/          # 実世界データテスト
    └── comparison_studies/       # 他手法との比較
```

### 層別詳細仕様

#### 1. **api/** - ユーザーAPI層
地形学的ピーク分析のための統一インターフェースを提供

**peak_detector.py**: メイン分析エンジン
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

**result_dataframe.py**: 結果データ構造
```python
class PeakCollection:
    def filter(self, **criteria) -> 'PeakCollection'
    def sort_by(self, feature, ascending=True) -> 'PeakCollection'
    def get_features(self, features=None) -> LazyDataFrame
    def to_pandas(self) -> pd.DataFrame
    def visualize(self, backend='matplotlib', **kwargs)

class LazyDataFrame:
    def compute_feature(self, feature_name, **params)
    def __getitem__(self, key)  # 遅延計算トリガー
    def cache_features(self, features)
    def clear_cache(self)
```

#### 2. **core/** - コアアルゴリズム層
地形学的分析の核となるアルゴリズムを実装

**strategy_manager.py**: 戦略選択・管理
```python
class StrategyManager:
    def select_optimal_strategy(self, data_characteristics) -> Strategy
    def estimate_computational_cost(self, strategy, data_shape)
    def benchmark_strategies(self, data, strategies) -> BenchmarkResults
    def configure_strategy(self, strategy_name, **params) -> Strategy
```

**plateau_detector.py**: プラトー検出コア
```python
class PlateauDetector:
    def detect_plateaus(self, data, connectivity) -> List[PlateauRegion]
    def validate_plateau(self, region, data, connectivity) -> bool
    def merge_connected_plateaus(self, plateaus) -> List[PlateauRegion]
    def filter_noise_plateaus(self, plateaus, min_area) -> List[PlateauRegion]
```

**prominence_calculator.py**: プロミネンス計算コア
```python
class ProminenceCalculator:
    def calculate_prominence(self, peak, data, wlen=None) -> float
    def find_prominence_base(self, peak, data) -> Coordinate
    def trace_descent_path(self, start_point, data) -> Path
    def handle_boundary_cases(self, peak, data) -> float
```

**virtual_peak_handler.py**: 仮想ピーク処理
```python
class VirtualPeakHandler:
    def detect_connected_same_height_peaks(self, peaks) -> List[PeakGroup]
    def create_virtual_peak(self, peak_group) -> VirtualPeak
    def calculate_virtual_prominence(self, virtual_peak, data) -> float
    def resolve_saddle_points(self, peak_group) -> List[SaddlePoint]
```

#### 3. **strategies/** - 検出戦略実装層
異なるデータ特性に最適化された検出アルゴリズム

**base_strategy.py**: 戦略共通インターフェース
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

**union_find_strategy.py**: Union-Find戦略
```python
class UnionFindStrategy(BaseStrategy):
    def detect_peaks_with_prominence(self, data) -> Tuple[List[Peak], Dict[Peak, float]]
    def build_height_graph(self, data) -> HeightGraph
    def process_height_level(self, height, points) -> List[Component]
    def wave_front_expansion(self, batch, processed) -> Set[Point]
```

**plateau_first_strategy.py**: プラトー優先戦略
```python
class PlateauFirstStrategy(BaseStrategy):
    def detect_plateaus_then_prominence(self, data) -> Tuple[List[Peak], Dict[Peak, float]]
    def apply_local_maximum_filter(self, data) -> np.ndarray
    def validate_plateaus_by_dilation(self, candidates) -> List[PlateauRegion]
    def batch_prominence_calculation(self, plateaus, data) -> Dict[Peak, float]
```

#### 4. **features/** - 特徴計算層
地形学的特徴の包括的計算フレームワーク

**geometric_calculator.py**: 幾何学的特徴
```python
class GeometricCalculator(BaseCalculator):
    def calculate_area(self, peak_region, scale) -> float
    def calculate_volume(self, peak_region, data, scale) -> float
    def calculate_centroid(self, peak_region, data) -> Coordinate
    def calculate_aspect_ratio(self, peak_region) -> Dict[str, float]
    def calculate_bounding_box(self, peak_region) -> BoundingBox
```

**topographic_calculator.py**: 地形学的特徴
```python
class TopographicCalculator(BaseCalculator):
    def calculate_isolation(self, peak, all_peaks, distance_metric) -> float
    def calculate_relative_height(self, peak, data, neighborhood_radius) -> float
    def calculate_topographic_position_index(self, peak, data) -> float
    def calculate_width_at_relative_height(self, peak, data, rel_height) -> float
```

**distance_calculator.py**: 距離・接続性特徴
```python
class DistanceCalculator(BaseCalculator):
    def calculate_minkowski_distance(self, point1, point2, p, scale) -> float
    def find_nearest_higher_peak(self, peak, all_peaks) -> Tuple[Peak, float]
    def calculate_peak_density(self, center, peaks, radius) -> float
    def calculate_watershed_distance(self, peak1, peak2, data) -> float
```

#### 5. **connectivity/** - 接続性定義層
N次元空間での接続性とパス探索

#### 6. **coordinate_system/** - 座標系層
インデックス空間と座標空間の変換統合管理

**grid_manager.py**: インデックス ↔ 座標変換と空間操作
```python
class GridManager:
    def __init__(self, mapping: CoordinateMapping, connectivity_level: int)
    def indices_to_coordinates(self, indices) -> Union[Tuple, np.ndarray]
    def coordinates_to_indices(self, coordinates) -> Union[Tuple, np.ndarray]
    def calculate_distance(self, coord1, coord2, metric='euclidean') -> float
    def get_neighbors_coordinates(self, center_coordinates) -> List[Tuple]
    def find_neighbors_in_radius(self, center, radius, metric) -> List[Tuple]

class Peak:
    center_indices: Tuple[int, ...]      # 内部処理用インデックス
    center_coordinates: Tuple[float, ...] # ユーザー向け座標
    plateau_indices: List[Tuple[int, ...]] # インデックス空間での領域
    @property
    def coordinate_dict(self) -> dict     # {x: 1.5, y: 2.3, z: 4.1}
```

**coordinate_mapping.py**: マッピング定義
```python
@dataclass
class CoordinateMapping:
    indices_shape: Tuple[int, ...]     # データ配列形状 (I, J, K, ...)
    coordinate_origin: Tuple[float, ...] # 実世界原点 (x0, y0, z0, ...)
    coordinate_spacing: Tuple[float, ...] # 物理スペーシング (dx, dy, dz, ...)
    axis_names: Tuple[str, ...] = ('x', 'y', 'z') # 座標軸名
```

#### 7. **boundary/** - 境界処理層
データ境界での適切な処理

**boundary_handler.py**: 境界条件処理
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

#### 7. **data/** - データ管理層
効率的なデータ処理とメモリ管理

**lazy_dataframe.py**: 遅延評価データフレーム
```python
class LazyDataFrame:
    def __init__(self, peak_collection, feature_calculators)
    def __getitem__(self, feature_name)  # 遅延計算
    def compute_batch(self, feature_names) -> Dict[str, np.ndarray]
    def cache_strategy(self, strategy='lru', max_size=None)
    def memory_usage(self) -> MemoryReport
```

### データフローと相互作用

```
1. データ入力・前処理
   ├── 入力検証 (validation.py)
   ├── 境界処理 (boundary_handler.py)
   └── データ拡張・正規化

2. 戦略選択・初期化
   ├── データ特性解析 (strategy_manager.py)
   ├── 最適戦略選択
   └── パラメータ調整

3. ピーク検出実行
   ├── Union-Find戦略 OR プラトー優先戦略
   ├── プラトー検出・検証
   ├── プロミネンス計算
   └── 仮想ピーク処理

4. 特徴計算・フィルタリング
   ├── 遅延特徴計算 (lazy_feature_manager.py)
   ├── ユーザー指定フィルタ適用
   └── 結果データ構造構築

5. 結果出力・可視化
   ├── データフレーム変換
   ├── 統計情報生成
   └── 可視化・エクスポート
```

### 性能最適化機能

- **適応的戦略選択**: データ特性に基づく最適アルゴリズム自動選択
- **遅延評価**: 必要な特徴のみを計算する効率的なメモリ使用
- **マルチスレッド処理**: CPU集約的計算の並列化
- **キャッシュシステム**: 計算結果の知的キャッシング
- **メモリ最適化**: 大規模データセット用のチャンク処理
- **プロファイリング**: リアルタイム性能監視とボトルネック特定

### 詳細関数仕様

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
# k=1: 面共有（2n近傍）
# k=2: 面＋辺共有
# k=3: 面＋辺＋頂点共有
# ...
# k=ndim: すべての境界共有（3^n-1近傍）
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

### 特徴計算仕様

#### **幾何学的特徴**
- **ピークプラトー面積**: プラトー領域内のセル数
- **ピーク体積**: プラトー領域内の高度値の合計
- **重心位置**: 質量中心の重み付き座標
- **境界長・面積**: プラトー領域の周囲長
- **アスペクト比**: 各次元での伸長度測定
- **境界ボックス**: 各次元での最小・最大座標

#### **地形学的特徴**
- **ピーク高度**: 領域内の最大標高
- **プロミネンス**: 最低包囲等高線への垂直距離
- **アイソレーション**: 最も近い高いピークまでの距離
- **相対高度**: 局所環境に対する高度
- **地形位置指数**: 近傍標高に対する位置
- **相対高度でのピーク幅 (rel_height)**: prominence_baseとpeak_heightの間の指定相対高度でのピーク領域面積（例：rel_height=0.5で半プロミネンスレベルでの幅を測定）

#### **形態学的特徴**
- **鋭さ**: ピーク重心周辺の局所曲率
- **平均傾斜**: プラトー領域での勾配の平均値
- **傾斜変動**: 勾配の標準偏差
- **方向別傾斜**: 各座標方向での勾配
- **ラプラシアン**: 二次微分測定

#### **距離・接続性特徴**
- **最近高ピーク距離**: 最も近い高いピークへの設定可能なミンコフスキー距離（Lp ノルム）
- **類似ピーク距離**: 指定された距離メトリックを使用した類似高度のピークまでの距離
- **ピーク密度**: 指定半径内のピーク数（ピクセルスケールを考慮）
- **接続次数**: 他のピークとの階層的関係
- **流域距離**: 最急降下経路に沿った距離
- **スケール化距離**: すべての距離測定はピクセルスケールパラメータを尊重し正確な物理測定を実現
- **ミンコフスキー距離オプション**: L1（マンハッタン）、L2（ユークリッド）、L∞（チェビシェフ）、カスタムLpノルムに対応

### アルゴリズム実装詳細

#### **Union-Find戦略アルゴリズム**

##### **核心的挑戦**: 偽ピークなしの高度優先処理

**ステップ1: 優先キュー初期化**
- すべてのデータ点を（高度、座標）として優先キューを作成
- 最高から最低標高への処理のため最大ヒープを使用
- すべてのデータ点でUnion-Find構造を初期化

**ステップ2: 高度レベルバッチ処理**
- **重要な洞察**: 同じ高度のすべての点を同時に処理
- キューから現在の最大高度を持つすべての点を抽出
- 統一処理のため同高度点の一時バッチを作成

**ステップ3: 波面拡張による同高度接続性解析**
- **重要な問題**: 同高度バッチ内での素朴なunionは偽ピークを作成
- **解決策**: 既処理点からの波面拡張

**波面拡張アルゴリズム:**
```
processed_points = set()  # 前の高度レベルから
current_batch = get_same_height_points(current_height)
newly_processed = set()  # 現在の反復で処理された点

# 反復的波面拡張
while True:
    temp_store = set()  # 新しく接続された点を一時保存
    
    # 処理済み点に接続する未処理点を発見
    for point in current_batch:
        if point not in newly_processed:
            for neighbor in get_k_neighbors(point):
                if neighbor in processed_points or neighbor in newly_processed:
                    # 点が既処理地形に接続
                    temp_store.add(point)
                    
                    # Union操作
                    if not has_region(point):
                        # 点対領域union（最初の接続）
                        neighbor_region = find_region(neighbor)
                        union_point_to_region(point, neighbor_region)
                    else:
                        # 領域対領域union（以降の接続）
                        point_region = find_region(point)
                        neighbor_region = find_region(neighbor)
                        if point_region != neighbor_region:
                            union_regions(point_region, neighbor_region)
                    break
    
    # 新しい接続が見つからない場合 - 終了
    if not temp_store:
        break
    
    # 新しく接続された点を処理済みセットに追加
    newly_processed.update(temp_store)

# 残りの未処理点を処理（潜在的新ピーク）
remaining_points = current_batch - newly_processed
isolated_components = find_connected_components(remaining_points, k_connectivity)
for component in isolated_components:
    register_as_new_peak_candidate(component)
```

**主要利点:**
- **偽ピーク防止**: プラトー内部点が独立して処理されない
- **接続性維持**: 複数の処理済み近傍が存在する場合の適切な領域統合
- **波面処理**: 高地形からの自然な水流拡張を模倣

**ステップ4: 領域検証とピーク検出**
- **種接続領域**: 非ピークとしてマーク（高地形に接続）
- **孤立領域**: ピーク候補としてマーク（高地形への接続なし）
- **孤立点**: 種に接続されない個別点→潜在的新ピーク

**ステップ5: 横断中のプロミネンス計算**
- 高度レベルを下降するにつれて:
  1. 各ピークプラトーの鞍部点を追跡
  2. ピークが高地形に接続する際、鞍部標高を記録
  3. プロミネンスを（ピーク高度 - 鞍部高度）として計算

**ステップ6: 偽ピーク防止**
- **主要戦略**: 同高度プラトーの個別点を個別に処理しない
- 常に同高度連結成分全体を原子単位として処理
- 等高度以上の地形に接続する成分を拒否

##### **Union-Find統合ロジック**
- **累進Union戦略**: 処理済み点から開始し、未処理同高度点へ拡張
- **二段階Union**:
  1. **点対領域**: 未処理点が既存領域に参加（最初の接続）
  2. **領域対領域**: 未処理点を介して接続された既存領域の統合（以降の接続）
- **種ベース処理**: 高い処理済み地形に接続する点のみが統合種として機能
- **プロミネンス追跡**: 拡張中の各領域ルートの鞍部標高を維持

#### **プラトー優先戦略アルゴリズム**

##### **第1段階: プラトー検出ロジック**

**ステップ1: 局所最大値識別**
- 各セル (i,j,...) に対し、指定されたk-connectivityを使用して局所最大フィルタを適用
- `data[i,j,...] >= max(all_k_connected_neighbors)` の場合、セルは候補
- これにより潜在的ピークセルのバイナリマスクを作成
- **問題**: 非ピークプラトーもこのフィルタで検出される

**ステップ2: 連結成分解析**
- 候補セル中で、同一高度値を持つものをグループ化
- k-connectivityを使用してUnion-Findまたは洪水充填で連結成分を発見
- 各成分は潜在的な一定高度プラトー領域を表す

**ステップ3: プラトー検証（膨張テスト）**
- **主要洞察**: 真のピークプラトーと非ピークプラトーは膨張下で異なる振る舞い
- 高度 `h` の各連結成分に対して:
  1. 成分のバイナリマスクを作成
  2. k-connectivity構造要素を使用して形態学的膨張を適用
  3. 膨張境界をチェック: `dilated_mask AND NOT original_mask`
  4. **重要ロジック**: 境界セルが高度 = `h`（同じ高度）の場合、非ピークプラトーとして拒否
  5. すべての境界セルが高度 < `h`（厳密に低い）の場合、真のピークプラトーとして承認

**推論**:
- 真のピークプラトー: 膨張境界は常に厳密に低くなる
- 非ピークプラトー: 膨張境界に同じ高度のセルが含まれる（高地形に接続）。なお、膨張境界には元のプラトー内部点（高い領域への接続により局所最大フィルタで取りこぼされた点）と外部点の両方が含まれる場合がある

##### **第2段階: プロミネンス計算**
- 検証された各プラトーに対し、境界から幅優先探索を実行
- 高地形に到達するまで最低標高を追跡
- プロミネンスを高度差として計算

#### **エッジ・境界処理**
- **無限高度境界**: データエッジを最大浮動小数点値でパディング
- **無限深度境界**: データエッジを最小浮動小数点値でパディング
- **周期境界**: データエッジを反対側エッジ値で包む
- **カスタム境界**: エッジでユーザー指定の定数値
- **アーティファクト除去**: データ境界に近すぎるピークをフィルタリング

#### **N次元接続性**
- **1-connectivity**: 面共有（2n近傍）
- **2-connectivity**: 面＋辺共有
- **3-connectivity**: 面＋辺＋頂点共有
- **...**
- **n-connectivity**: すべての境界共有（3^n-1近傍）
- **効率性**: 各接続レベルの事前計算済みオフセット配列

### 性能・メモリ考慮事項
- **遅延評価**: 要求された場合にのみ特徴を計算
- **メモリマッピング**: メモリマップファイルを介した大配列処理
- **チャンク処理**: メモリ効率のためデータを重複チャンクに分割
- **並列処理**: マルチスレッド特徴計算
- **キャッシング**: 高価な計算の知的キャッシング