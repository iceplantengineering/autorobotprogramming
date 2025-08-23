# Robot Teaching Application

## 概要

Visual Componentsとの連携による産業用ロボット教示・自動化システム。TCP/UDP通信を使用したPythonベースの外部制御アプリケーションです。

## プロジェクト進捗

### Phase 1: 基盤通信システム構築 ✅ 完了

- [x] **外部Pythonアプリケーション構造設計** - RobotTeachingAppクラス実装完了
- [x] **TCP/UDP通信インターフェース** - JSON形式メッセージング、エラー回復機能付き
- [x] **Visual ComponentsロボットコンポーネントPythonスクリプト作成** - vcScript統合対応
- [x] **ツールコントローラー実装** - 溶接ガン・グリッパー制御完了
- [x] **基本I/O制御とメッセージハンドリング** - 15標準I/O信号対応
- [x] **Phase 2準備** - 基本ハンドリング作業用設定ファイル構造設計完了

### Phase 2: 基本ハンドリング作業実装 ✅ 完了

- [x] **基本ピック&プレース作業フロー** - 5段階ワークフロー実装完了
- [x] **高度な軌道生成システム** - 衝突検査・複数軌道タイプ対応
- [x] **統合安全監視システム** - 3層安全チェック・リアルタイム監視
- [x] **作業教示インターフェース** - CLI・Web・API対応
- [x] **Visual Components統合テスト** - 包括的テストスイート完成

### Phase 3: 高度な自動化機能 ✅ 完了

- [x] **コンベア追従制御システム** - リアルタイム追従・動的軌道修正
- [x] **視覚認識統合システム** - OpenCV基盤ワークピース検出・認識
- [x] **複数ロボット協調制御** - 分散制御・衝突回避・タスクスケジューリング
- [x] **生産管理システム連携** - MES/ERP連携・リアルタイム生産データ管理

## 技術仕様

### システム要件
- Python 3.8+
- Visual Components Professional (推奨)
- TCP/IP ネットワーク接続

### アーキテクチャ
```
Robot Teaching Application
├── 通信レイヤー (TCP/UDP)
├── エラー回復システム
├── ツールコントローラー
├── I/O制御システム
├── 設定管理システム
└── Visual Components統合
```

## ファイル構成

```
autorobotprogramming/
├── README.md
├── robot_teaching_app.py             # メインアプリケーション
├── tcp_communication.py             # TCP通信モジュール
├── vc_robot_controller.py           # ロボットコントローラー
├── vc_tool_controller.py            # ツールコントローラー
├── io_message_handler.py            # I/O制御・メッセージ処理
├── error_recovery.py                # エラー回復システム
├── config_manager.py                # 設定ファイル管理
├── basic_handling_workflow.py       # 基本ハンドリングワークフロー
├── trajectory_generation.py         # 軌道生成システム
├── integrated_safety_system.py      # 統合安全システム
├── work_teaching_interface.py       # 作業教示インターフェース
├── vc_integration_test.py           # 統合テストスイート
├── test_error_handling.py           # エラーハンドリングテスト
├── conveyor_tracking_system.py      # コンベア追従制御システム (Phase 3)
├── vision_integration_system.py     # 視覚認識統合システム (Phase 3)
├── multi_robot_coordination.py      # 複数ロボット協調制御 (Phase 3)
├── production_management_integration.py  # 生産管理システム連携 (Phase 3)
├── config/
│   ├── application_config.json      # 基本設定
│   ├── handling_operations.json     # 作業操作定義
│   └── work_templates.json          # ワークテンプレート
├── taught_works/                    # 教示作業保存（自動生成）
├── logs/                            # ログファイル（自動生成）
└── production.db                    # 生産データベース（自動生成）
```

## 主要機能

### 1. 通信システム
- **TCP/UDP双方向通信**: JSON形式メッセージング
- **自動再接続機能**: 指数バックオフ付きリトライ
- **エラー回復**: サーキットブレーカーパターン
- **接続監視**: ヘルスチェック機能

### 2. ツール制御
#### 溶接ガンコントローラー
- スポット溶接実行
- 電極摩耗管理
- 溶接パラメーター調整
- TCP（Tool Center Point）キャリブレーション

#### グリッパーコントローラー  
- 開閉制御（位置・力制御）
- ワーク検出機能
- 把持力調整
- TCP キャリブレーション

### 3. 基本ハンドリングワークフロー（Phase 2）
- **5段階ワークフロー**: 準備→ピック→移動→プレース→完了
- **SafetyMonitor**: リアルタイム安全監視システム
- **パフォーマンス追跡**: サイクル時間・成功率測定
- **エラー回復**: 自動診断・復旧機能

### 4. 高度な軌道生成システム（Phase 2）
- **複数軌道タイプ**: 直線・円弧・スプライン・ジョイント
- **衝突検査**: 3D空間リアルタイム衝突回避
- **マルチポイント最適化**: TSPアルゴリズムによる経路最適化
- **コンベア追従**: 移動コンベア対応軌道生成

### 5. 統合安全システム（Phase 2）
- **3層安全チェック**: 基本安全・ワークスペース・軌道検証
- **安全ゾーン管理**: 監視・制限・禁止の3種類エリア
- **リアルタイム監視**: 100ms間隔での状態確認
- **安全イベント管理**: 履歴記録・コールバック対応

### 6. 作業教示インターフェース（Phase 2）
- **CLI教示**: 対話式コマンドライン操作
- **Web教示**: ブラウザベースGUI（ポート8080）
- **教示モード**: 手動・ガイド・テンプレート・インポート
- **セッション管理**: 複数同時教示・保存・読み込み

### 7. I/O制御システム
- **標準I/O信号**: 15の産業用ロボット標準信号対応
- **リアルタイム監視**: 0.1秒間隔でのI/O状態監視
- **履歴管理**: 信号変化履歴記録（最新100件）
- **コールバック機能**: 信号変化時の自動処理

#### 対応I/O信号
**入力信号**
- START_BUTTON, E_STOP, PAUSE_BUTTON, RESUME_BUTTON
- MODE_SELECT, PART_PRESENT, JIG_CLAMPED, DOOR_CLOSED
- AIR_PRESSURE_OK, WELD_ENABLE, WELDER_READY
- SUPPLY_PART_OK, DISCHARGE_READY

**出力信号**  
- WORK_COMPLETE, ERROR_OCCURRED, READY_LAMP, WORKING_LAMP
- STEP_NUMBER, ERROR_CODE, WELD_EXECUTE, WELD_COMPLETE
- PICK_COMPLETE, PLACE_COMPLETE

### 8. 設定管理システム
- **JSON設定ファイル**: 柔軟な設定変更
- **設定検証機能**: 起動時自動検証
- **ワークテンプレート**: 業界別作業パターン
- **操作定義**: 標準作業フローテンプレート

#### 対応業界テンプレート
- **自動車産業**: 部品ハンドリング、スポット溶接
- **電子機器組立**: 精密部品実装
- **一般製造業**: 汎用ピック&プレース

## Phase 3: 高度な自動化機能

### 9. コンベア追従制御システム
- **リアルタイム追従**: 動的ワークピース位置予測・軌道修正
- **エンコーダー連携**: 高精度位置検出・速度同期
- **ピックアップゾーン管理**: 柔軟なゾーン設定・自動切り替え
- **予測アルゴリズム**: 2秒先行予測による最適タイミング制御

### 10. 視覚認識統合システム
- **OpenCV画像処理**: リアルタイム物体検出・認識
- **複数検出手法**: 輪郭・テンプレート・色・特徴点マッチング
- **カメラキャリブレーション**: 画像座標⇔実座標変換
- **品質検査**: 寸法測定・欠陥検出・合否判定

#### 対応検出方式
- **輪郭ベース検出**: 形状による物体識別
- **テンプレートマッチング**: 既知パターンとの照合
- **色ベース検出**: HSV色空間フィルタリング
- **特徴点検出**: SIFT/ORB特徴量マッチング

### 11. 複数ロボット協調制御
- **分散制御アーキテクチャ**: マスター・スレーブ協調制御
- **衝突予測・回避**: リアルタイム軌道衝突判定・自動回避
- **タスクスケジューリング**: 優先度ベース最適割り当て
- **ワークスペース管理**: 動的予約システム・競合解決

#### 協調機能
- **同期作業**: 複数ロボットによる協調組み立て
- **リレー作業**: ワークピース受け渡し制御
- **負荷分散**: 動的作業配分・効率最適化
- **故障時冗長性**: 自動代替ロボット切り替え

### 12. 生産管理システム連携
- **MES/ERP統合**: REST API・データベース連携
- **リアルタイム生産追跡**: 進捗・品質・効率監視
- **作業指示システム**: 動的作業オーダー管理
- **品質管理**: 検査記録・トレーサビリティ

#### 生産データ管理
- **生産オーダー**: 製品・数量・納期管理
- **作業実績**: サイクル時間・稼働率・OEE計算
- **品質記録**: 検査結果・不具合履歴・改善追跡
- **予防保全**: 稼働時間・エラー履歴・メンテナンス計画

## インストール・使用方法

### 1. 基本セットアップ
```bash
git clone https://github.com/iceplantengineering/autorobotprogramming.git
cd autorobotprogramming
pip install -r requirements.txt  # 作成予定
```

### 2. アプリケーション起動
```bash
# メインアプリケーション起動
python robot_teaching_app.py

# ツールコントローラー起動（グリッパー）
python vc_tool_controller.py gripper

# ツールコントローラー起動（溶接ガン）  
python vc_tool_controller.py welding

# 作業教示インターフェース起動
python work_teaching_interface.py

# 統合テスト実行（Phase 3機能含む）
python vc_integration_test.py

# Phase 3 個別システム起動
python conveyor_tracking_system.py     # コンベア追従システム
python vision_integration_system.py    # 視覚認識システム  
python multi_robot_coordination.py     # 複数ロボット協調
python production_management_integration.py  # 生産管理連携
```

### 3. 設定カスタマイズ
```bash
# 設定ファイル編集
config/application_config.json    # 基本設定
config/handling_operations.json  # 作業定義
config/work_templates.json       # ワークテンプレート
```

## API仕様

### メッセージ形式
```json
{
  "message_id": "timestamp_command",
  "timestamp": "2024-01-01T10:00:00.000Z",
  "command_type": "robot_move",
  "target_component": "robot_1",
  "parameters": {
    "position": [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
    "speed": 50
  },
  "response_required": true
}
```

### 主要コマンド
- `robot_move`: ロボット移動
- `tool_control`: ツール制御  
- `weld_execute`: 溶接実行
- `io_read/write`: I/O制御
- `status_request`: 状態要求
- `program_upload/execute`: プログラム実行

## エラーハンドリング

### 実装済みエラー対応
1. **接続タイムアウト**: 自動再試行（最大10回）
2. **APIエラー**: 指数バックオフでリトライ
3. **ネットワーク障害**: サーキットブレーカーで保護
4. **サービス停止**: ヘルスチェック監視

### ログ出力
```
2024-01-01 10:00:00 - INFO - TCP Server started on localhost:8888
2024-01-01 10:00:01 - WARNING - Connection failed (attempt 2/10): Connection refused
2024-01-01 10:00:02 - INFO - Retrying in 2.1 seconds...
2024-01-01 10:00:05 - INFO - Connected successfully (attempt 3)
```

## テスト

### エラーハンドリングテスト
```bash
python test_error_handling.py
```

### I/O システムテスト  
```bash
python io_message_handler.py
```

## Phase 2 新機能詳細

### 1. 作業教示システム
```bash
# CLI教示インターフェース
python work_teaching_interface.py

# 教示セッション開始
teach> start manual
teach> position pick      # ピック位置教示
teach> position place     # プレース位置教示  
teach> workpiece part1 plastic 1.5 50  # ワークピース定義
teach> generate           # 軌道生成
teach> execute           # タスク実行
```

### 2. Web教示インターフェース
- **URL**: http://localhost:8080
- **機能**: ブラウザでの視覚的教示操作
- **対応**: リアルタイムステータス表示、軌道生成、安全監視

### 3. 高度な軌道制御
```python
from trajectory_generation import generate_handling_trajectory

# 基本軌道生成
trajectory = generate_handling_trajectory({
    "operation_type": "basic_pick_place",
    "pick_position": [100, -200, 150, 0, 0, 0],
    "place_position": [300, 100, 150, 0, 0, 0],
    "workpiece": {...},
    "parameters": {"safety_height": 60.0}
})

# マルチポイント軌道
trajectory = generate_handling_trajectory({
    "operation_type": "multi_point_handling", 
    "pick_points": [...],
    "place_points": [...],
    "parameters": {"cycle_mode": "optimized"}
})
```

### 4. 統合安全システム
```python
from integrated_safety_system import safety_system

# 安全状態確認
status = safety_system.check_overall_safety()
print(f"System safe: {status['overall_safe']}")

# 位置安全性チェック
position = Position(100, 100, 150, 0, 0, 0)
safe = safety_system.is_safe_to_move(position)

# 軌道安全検証  
validation = safety_system.validate_trajectory_safety(trajectory_points)
```

## 開発計画

### Phase 3: 高度な自動化機能（次期計画）
- **コンベア追従制御**: リアルタイム位置補正
- **視覚認識統合**: OpenCV・機械学習対応
- **複数ロボット協調**: 分散制御・衝突回避
- **生産管理システム連携**: MES・ERPデータ連携
- **予知保全**: 機械学習による故障予測

## ライセンス

MIT License

## 貢献

プルリクエスト歓迎です。大きな変更については、まずissueで議論してください。

## サポート

- Issues: https://github.com/iceplantengineering/autorobotprogramming/issues
- Email: support@example.com

---

© 2024 Ice Plant Engineering. All rights reserved.