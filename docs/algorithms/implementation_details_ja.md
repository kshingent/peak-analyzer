# アルゴリズム実装詳細

_← [アルゴリズム](algorithm_ja.md) | [アーキテクチャ](../architecture/architecture_ja.md) | [APIリファレンス](../api/api_reference_ja.md) | [English Version](implementation_details.md) →_

## Union-Find戦略アルゴリズム

### **コアチャレンジ**: 偽ピークを避ける高度優先処理

**ステップ1: 優先キューの初期化**
- 全データポイントを(高度, 座標)として優先キューを作成
- 最高から最低標高への処理のためmaxヒープを使用
- 全データポイントに対してUnion-Find構造を初期化

**ステップ2: 高度レベルバッチ処理**
- **重要な洞察**: 同じ高度の全ポイントを同時に処理
- キューから現在の最大高度の全ポイントを抽出
- 統一処理のため同一高度ポイントの一時バッチを作成

**ステップ3: プログレッシブユニオンによる同一高度接続性解析**
- **重要な問題**: 同一高度バッチ内の単純なユニオンは偽ピークを生成
- **解決策**: 既に処理されたポイントからの波面拡張

**波面拡張アルゴリズム:**
```
processed_points = set()  # 前の高度レベルから
current_batch = get_same_height_points(current_height)
newly_processed = set()  # 現在の反復で処理されたポイント

# 反復的波面拡張
while True:
    temp_store = set()  # 新しく接続されたポイントを一時保存
    
    # 処理済みポイントに接続された未処理ポイントを見つける
    for point in current_batch:
        if point not in newly_processed:
            for neighbor in get_k_neighbors(point):
                if neighbor in processed_points or neighbor in newly_processed:
                    # ポイントが既に処理された地形に接続
                    temp_store.add(point)
                    
                    # ユニオン操作
                    if not has_region(point):
                        # ポイント-リージョンユニオン（初回接続）
                        neighbor_region = find_region(neighbor)
                        union_point_to_region(point, neighbor_region)
                    else:
                        # リージョン-リージョンユニオン（後続接続）
                        point_region = find_region(point)
                        neighbor_region = find_region(neighbor)
                        if point_region != neighbor_region:
                            union_regions(point_region, neighbor_region)
                    break
    
    # 新しい接続が見つからない - 終了
    if not temp_store:
        break
    
    # 新しく接続されたポイントを処理済みセットに追加
    newly_processed.update(temp_store)

# 残りの未処理ポイントを処理（潜在的な新ピーク）
remaining_points = current_batch - newly_processed
isolated_components = find_connected_components(remaining_points, k_connectivity)
for component in isolated_components:
    register_as_new_peak_candidate(component)
```

**主な利点:**
- **偽ピーク防止**: プラトー内部ポイントが独立して処理されない
- **接続性維持**: 複数の処理済み近傍が存在する場合の適切な領域統合
- **波面処理**: より高い地形からの自然な水流拡張を模倣

**ステップ4: 領域検証とピーク検出**
- **シード接続領域**: 非ピークとしてマーク（高い地形に接続）
- **分離領域**: ピーク候補としてマーク（高い地形への接続なし）
- **孤立ポイント**: シードに接続されていない個別ポイント → 潜在的新ピーク

**ステップ4: Union-Find統合**
- 各プラトー成分について:
  1. 成分内の全ポイントをユニオン
  2. 以前に処理された高いプラトーへの接続性をチェック
  3. 高い地形に接続されている場合、非ピークプラトーとしてマーク
  4. 高い地形から分離されている場合、候補ピークプラトーとしてマーク

**ステップ5: 横断中のプロミネンス計算**
- 高度レベルを降下しながら:
  1. 各ピークプラトーのサドルポイントを追跡
  2. ピークが高い地形に接続する時、サドル標高を記録
  3. プロミネンスを (peak_height - saddle_height) として計算

**ステップ6: 偽ピーク防止**
- **重要な戦略**: 同一高度プラトーの個別ポイントを決して別々に処理しない
- 同一高度接続成分全体を常に原子単位として処理
- 等しいまたはより高い標高地形に接続する成分を拒否

### **キュー管理戦略**
```
while priority_queue not empty:
    current_height = peek_max_height(queue)
    same_height_batch = extract_all_with_height(queue, current_height)
    
    # バッチ全体を原子的に処理
    components = find_connected_components(same_height_batch, k_connectivity)
    
    for component in components:
        if is_isolated_from_higher_terrain(component):
            register_as_peak(component)
        else:
            mark_as_non_peak(component)
        
        union_all_points_in_component(component)
```

### **Union-Find統合ロジック**
- **プログレッシブユニオン戦略**: 処理済みポイントから開始し、未処理同一高度ポイントに拡張
- **二段階ユニオン**:
  1. **ポイント-リージョン**: 未処理ポイントが既存リージョンに参加（初回接続）
  2. **リージョン-リージョン**: 未処理ポイント経由で既存リージョンが統合（後続接続）
- **シードベース処理**: より高い処理済み地形に接続されたポイントのみが統合シードとして機能
- **プロミネンス追跡**: 拡張中に各リージョンルートのサドル標高を維持

## プラトー優先戦略アルゴリズム  

### **フェーズ1: プラトー検出ロジック**

**ステップ1: 局所最大値特定**
- 各セル(i,j,...)に対し、指定されたk接続性を使用して局所最大フィルタを適用
- `data[i,j,...] >= max(all_k_connected_neighbors)` の場合、セルは候補
- これによって潜在的ピークセルのバイナリマスクを作成
- **問題**: このフィルタでは非ピークプラトーも検出される

**ステップ2: 連結成分解析**
- 候補セルの中から、同じ高度値を持つものをグループ化
- k接続性を使用してUnion-Findまたはflood-fillで連結成分を見つける
- 各成分は一定高度の潜在的プラトー領域を表す

**ステップ3: プラトー検証（膨張テスト）**
- **重要な洞察**: 真のピークプラトー vs. 非ピークプラトーは膨張下で異なる動作を示す
- 高度`h`の各連結成分について:
  1. 成分のバイナリマスクを作成
  2. k接続性構造要素を使用して形態学的膨張を適用
  3. 膨張境界をチェック: `dilated_mask AND NOT original_mask`
  4. **重要ロジック**: 境界セルのいずれかが高度 = `h`（同じ高度）の場合、非ピークプラトーとして拒否
  5. 全境界セルが高度 < `h`（厳密に低い）の場合、真のピークプラトーとして受諾

**理由**: 
- 真のピークプラトー: 膨張境界は常に厳密に低い
- 非ピークプラトー: 膨張境界には同じ高度のセルが含まれる（高い地形に接続）。注意：膨張境界には、より高い領域への接続により局所最大フィルタが見落とした元のプラトー内部ポイント（外部ポイントも）の両方が含まれる可能性がある

### **フェーズ2: プロミネンス計算**
- 検証された各プラトーについて、境界から幅優先検索を実行
- より高い地形に到達するまで最小標高を追跡
- プロミネンスを高度差として計算

## エッジ/境界処理
- **無限高度境界**: データエッジを無限値でパッド（デフォルト）
  - 境界領域がピークプロミネンス計算に一切干渉しないことを保証
  - 全ピーク検出およびプロミネンスアルゴリズムに対して数学的に健全
- **無限深度境界**: データエッジを負の無限値でパッド
  - 境界隣接ピークを強調する必要がある特殊ケースで使用
  - 全体最大ピークのプロミネンスは全体最小値より上の高さとして計算
- **アーティファクト除去**: データ境界に近すぎるピークをフィルタ

## N次元接続性
- **1-接続性**: 面共有 (2n 近傍)
- **2-接続性**: 面 + エッジ共有
- **3-接続性**: 面 + エッジ + 頂点共有
- **...**
- **n-接続性**: 全境界共有 (3^n-1 近傍)
- **効率**: 各接続レベル用の事前計算オフセット配列
- **カスタムパターン**: ユーザー定義近傍オフセットパターン
- **効率**: 高速近傍生成のための事前計算オフセット配列

## パフォーマンスとメモリの考慮事項
- **遅延評価**: 要求時のみ特徴量を計算
- **メモリマッピング**: メモリマップファイル経由で大きな配列を処理
- **チャンク処理**: メモリ効率のため重複チャンクにデータを分割
- **並列処理**: マルチスレッド特徴量計算
- **キャッシュ**: 高価な計算の知的キャッシング