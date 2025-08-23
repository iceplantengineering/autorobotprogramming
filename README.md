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

### Phase 2: 基本ハンドリング作業実装 🔄 準備完了

- [ ] 基本ピック&プレース作業フロー
- [ ] 安全監視システム統合
- [ ] 作業教示インターフェース開発
- [ ] Visual Components実機連携テスト

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
├── robot_teaching_app.py          # メインアプリケーション
├── tcp_communication.py          # TCP通信モジュール
├── vc_robot_controller.py        # ロボットコントローラー
├── vc_tool_controller.py         # ツールコントローラー
├── io_message_handler.py         # I/O制御・メッセージ処理
├── error_recovery.py             # エラー回復システム
├── config_manager.py             # 設定ファイル管理
├── test_error_handling.py        # エラーハンドリングテスト
├── config/
│   ├── application_config.json   # 基本設定
│   ├── handling_operations.json  # 作業操作定義
│   └── work_templates.json       # ワークテンプレート
└── logs/                         # ログファイル（自動生成）
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

### 3. I/O制御システム
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

### 4. 設定管理システム
- **JSON設定ファイル**: 柔軟な設定変更
- **設定検証機能**: 起動時自動検証
- **ワークテンプレート**: 業界別作業パターン
- **操作定義**: 標準作業フローテンプレート

#### 対応業界テンプレート
- **自動車産業**: 部品ハンドリング、スポット溶接
- **電子機器組立**: 精密部品実装
- **一般製造業**: 汎用ピック&プレース

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

## 開発計画

### Phase 2: 基本ハンドリング作業実装
- 基本ピック&プレース軌道生成
- 安全監視システム統合  
- 作業教示GUI開発
- Visual Components実機連携

### Phase 3: 高度な自動化機能
- コンベア追従制御
- 視覚認識統合
- 複数ロボット協調
- 生産管理システム連携

## ライセンス

MIT License

## 貢献

プルリクエスト歓迎です。大きな変更については、まずissueで議論してください。

## サポート

- Issues: https://github.com/iceplantengineering/autorobotprogramming/issues
- Email: support@example.com

---

© 2024 Ice Plant Engineering. All rights reserved.