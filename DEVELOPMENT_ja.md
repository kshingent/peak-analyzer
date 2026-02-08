# 開発セットアップガイド

## クイックスタート

1. **プロジェクトディレクトリにクローンまたは移動**
   ```bash
   cd peak_analyzer
   ```

2. **開発モードでインストール**
   ```bash
   # uv を使用（推奨）
   uv add --dev ".[dev,examples]"
   
   # または pip を使用
   pip install -e ".[dev,examples]"
   ```

3. **デモンストレーションの実行**
   ```bash
   python main.py
   ```

4. **テストの実行**
   ```bash
   pytest
   ```

## 詳細セットアップ

### 前提条件

- Python 3.10 以上
- パッケージ管理用の uv（推奨）または pip

### インストールオプション

#### オプション1: 開発インストール（推奨）
```bash
# 全ての依存関係を含む編集可能モードでパッケージをインストール
uv add --dev ".[dev,examples]"

# または pip を使用
pip install -e ".[dev,examples]"
```

#### オプション2: 基本インストール
```bash
# コア依存関係のみをインストール
uv add .

# または pip を使用
pip install .
```

#### オプション3: ユーザーインストール
```bash
# PyPI からインストール（公開時）
pip install peak-analyzer
```

### 開発環境のセットアップ

#### 1. コード品質ツール
```bash
# Black でコードをフォーマット
black peak_analyzer/ tests/ examples/

# flake8 でコードを lint
flake8 peak_analyzer/ tests/ examples/

# 全ての品質チェックを実行
black --check peak_analyzer/ && flake8 peak_analyzer/
```

#### 2. テストセットアップ
```bash
# 全テストを実行
pytest

# カバレッジ付きでテストを実行
pytest --cov=peak_analyzer

# 高速テストのみを実行
pytest -m "not slow"

# 特定のテストファイルを実行
pytest tests/test_peak_analyzer.py

# 詳細出力でテストを実行
pytest -v
```

#### 3. 例の実行
```bash
# 基本使用例
python examples/basic_usage.py

# 高度な機能例
python examples/advanced_usage.py

# メインデモンストレーション
python main.py
```

## 開発用プロジェクト構造

```
peak_analyzer/
├── peak_analyzer/          # ソースコード
├── tests/                  # テストファイル
├── examples/               # 使用例
├── main.py                 # デモスクリプト
├── pyproject.toml         # プロジェクト設定
├── ARCHITECTURE.md        # アーキテクチャドキュメント
└── README.md              # メインドキュメント
```

## 開発ワークフロー

### 1. 新機能の追加

```bash
# 機能ブランチを作成
git checkout -b feature/new-calculator

# 適切なモジュールにコードを追加
# 例: peak_analyzer/features/new_calculator.py

# 対応するテストを追加
# 例: tests/test_new_calculator.py

# テストを実行
pytest tests/test_new_calculator.py

# フォーマットと lint
black peak_analyzer/ tests/
flake8 peak_analyzer/ tests/

# 変更をコミット
git commit -m "新しい計算機能を追加"
```

### 2. 変更のテスト

```bash
# 全テストを実行
pytest

# カバレッジレポート付きでテストを実行
pytest --cov=peak_analyzer --cov-report=html

# 統合テストを実行
pytest -m integration

# 特定の機能をテスト
pytest tests/test_peak_analyzer.py::TestPeakAnalyzer::test_feature_name
```

### 3. ドキュメントの更新

```bash
# ユーザー向け変更については README.md を更新
# 設計変更については ARCHITECTURE.md を更新
# examples/ ディレクトリに例を追加
# コード内の docstring を更新
```

## 一般的な開発タスク

### 新しい特徴量計算器の追加

1. 計算器モジュールを作成:
   ```python
   # peak_analyzer/features/my_calculator.py
   class MyCalculator:
       def calculate(self, peak_region):
           # 実装をここに記述
           return result
   ```

2. features の `__init__.py` に追加:
   ```python
   from .my_calculator import MyCalculator
   __all__.append("MyCalculator")
   ```

3. LazyDataFrame と統合:
   ```python
   # peak_analyzer/core/lazy_dataframe.py で
   elif feature_name == "my_feature":
       calculator = MyCalculator()
       self._feature_cache[feature_name] = np.array([
           calculator.calculate(region) for region in self.peak_regions
       ])
   ```

4. テストを追加:
   ```python
   # tests/test_my_calculator.py
   def test_my_calculator():
       # テスト実装
   ```

### 新しい検出戦略の追加

1. 戦略モジュールを作成:
   ```python
   # peak_analyzer/strategies/my_strategy.py
   class MyStrategy:
       def process_peaks(self, peak_regions, prominence_calculator):
           # 実装をここに記述
           return processed_regions
   ```

2. PeakAnalyzer に登録:
   ```python
   # peak_analyzer/core/peak_detector.py で
   def _get_strategy(self, strategy_name):
       strategies = {
           "my_strategy": MyStrategy(),
           # ... 既存の戦略
       }
   ```

### デバッグのヒント

#### デバッグログを有効化
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# ピーク検出コードをここに記述
```

#### パフォーマンスプロファイル
```python
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()

# ピーク検出コードをここに記述

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(10)
```

#### メモリ使用量
```python
import tracemalloc

tracemalloc.start()

# コードをここに記述

current, peak = tracemalloc.get_traced_memory()
print(f"現在のメモリ使用量: {current / 1024 / 1024:.1f} MB")
print(f"ピーク時メモリ使用量: {peak / 1024 / 1024:.1f} MB")
```

## トラブルシューティング

### よくある問題

1. **インポートエラー**: `-e` フラグを使用して開発モードでインストールしたことを確認
2. **依存関係の不足**: `uv add` または `pip install -e ".[dev,examples]"` を実行
3. **テスト失敗**: 必要なパッケージが全てインストールされているか確認
4. **パフォーマンス問題**: データタイプに適した戦略を使用

### サポートを受ける

1. `examples/` ディレクトリの例を確認
2. 使用パターンについては `tests/` のテストを確認
3. `ARCHITECTURE.md` のアーキテクチャドキュメントを読む
4. プロジェクトリポジトリで issue を開く

## 貢献ガイドライン

1. **コードスタイル**: Black フォーマッターを使用し、PEP 8 に従う
2. **テスト**: 全ての新機能にテストを追加
3. **ドキュメント**: 関連するドキュメントを更新
4. **パフォーマンス**: 遅延評価とメモリ効率を考慮
5. **互換性**: N次元一般化を維持