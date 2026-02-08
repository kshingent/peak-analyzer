# Status: Work in Progress (WIP)
このプロジェクトは現在開発中のプロトタイプです。アルゴリズムの実装やリファクタリングを順次進めており、構成が大きく変更される可能性があります。

# peak-analyzer

ピークを孤立した点ではなく地形領域として扱う多次元ピーク検出ライブラリです。既存の2Dピーク検出アルゴリズムの根本的な制限に対処します。

## なぜPeakAnalyzerが必要か？

既存の2Dピーク検出アルゴリズムには重要な問題があります：

1. **プラトーの誤認識**: 一定の高さの領域（プラトー）が複数の個別ピークとして誤って識別される
2. **特徴ベースフィルタリングの欠如**: プロミネンスやアイソレーションなどの地形学的特徴に基づくピークフィルタリング機能がない

**PeakAnalyzer** は、ピークを **地形領域** として再定義し、包括的な特徴抽出とフィルタリング機能により、これらの問題に対処します。

## 主要機能

### 地形学的特徴抽出
- **ピーク座標**: ピーク領域の重心または代表点
- **高さ**: ピークでの絶対値
- **プロミネンス**: ピークとその最低等高線との垂直距離
- **面積**: ピーク領域の空間的範囲
- **鋭さ**: 局所曲率または高度変化率
- **アイソレーション**: 最も近い高いピークまでの距離
- **距離**: 検出された特徴間の空間関係メトリック

### 厳密なプラトー処理
- ピークを「同じ高さの領域」として扱い、アーティファクト生成を防止
- 形態学的膨張検証を使用して真のピークプラトーと非ピークプラトーを区別

### 座標系サポート
- 内部処理用の **インデックス空間** (i,j,...) とユーザー操作用の **座標空間** (x,y,...) の明確な分離
- ピクセルスケールパラメータによる実世界物理座標
- 異方性解像度とGIS互換操作のサポート

### 遅延特徴計算
- 要求時または フィルタリングに必要な場合にのみ特徴を計算
- 大規模データセット向けのメモリ効率的処理

## クイックスタート

### インストール

```bash
# uv使用（推奨）
uv add ".[dev,examples]"

# pipを使用
pip install -e ".[dev,examples]"
```

### 基本使用法

```python
from peak_analyzer import PeakAnalyzer
import numpy as np

# サンプル2Dデータ生成
data = np.random.randn(100, 100) + 5

# アナライザー初期化
analyzer = PeakAnalyzer(
    strategy='auto',
    connectivity='face', 
    boundary='infinite_height'
)

# ピーク検出
peaks = analyzer.find_peaks(data)

# プロミネンスと面積でフィルタリング
significant_peaks = peaks.filter(
    prominence=lambda p: p > 0.5,
    area=lambda a: a > 5
)

# 特徴アクセス
print(f"有意なピークを {len(significant_peaks)} 個発見")
features_df = significant_peaks.to_pandas()
print(features_df[['coordinates', 'height', 'prominence', 'area']])
```

### 物理座標を使用した高度な使用法

```python
# 実世界座標の設定
analyzer = PeakAnalyzer(
    scale=[1.0, 0.5],  # 異なるx/y解像度
    distance_metric='euclidean'
)

# 特徴計算を伴うピーク検出
peaks = analyzer.find_peaks(data, 
                          prominence_threshold=0.3,
                          min_area=3)

# 追加特徴の計算
features = peaks.get_features(['isolation', 'sharpness', 'aspect_ratio'])

# 座標範囲による空間フィルタリング
regional_peaks = peaks.filter_by_coordinates(
    x_range=(-10, 10),
    y_range=(20, 40)
)
```

## ドキュメント

- **[アルゴリズム詳細](docs/algorithms/algorithm_ja.md)**: コアアルゴリズム理論と実装詳細
- **[開発ガイド](DEVELOPMENT.md)**: セットアップと貢献ガイドライン  
- **[API リファレンス](docs/api/)**: 包括的なAPI ドキュメント
- **[例](examples/)**: 使用例とデモンストレーション
- **[English README](README.md)**: このドキュメントの英語版

## 性能考慮事項

- **適応的戦略選択**: データ特性に基づく最適アルゴリズムの自動選択
- **メモリ最適化**: 大規模データセット向けのチャンク処理とメモリマッピング 
- **並列処理**: CPU集約的計算のマルチスレッド化
- **知的キャッシング**: 高価な計算の戦略的キャッシング

## ライセンス

[ライセンスを指定]

## 貢献

セットアップ手順と貢献ガイドラインについては [DEVELOPMENT.md](DEVELOPMENT.md) を参照してください。