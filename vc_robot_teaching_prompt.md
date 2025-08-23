# Visual Componentsロボット教示データ自動生成アプリ開発プロンプト

## 1. プロジェクト概要

Visual Components（VC）の標準ロボットを使用して、スポット溶接とハンドリング作業の教示データを自動生成するアプリケーションを開発する。

### 対象アプリケーション
- **スポット溶接**: 溶接ガンを用いた自動車部品等の溶接作業
- **荷役作業**: クランプ式ハンドリングツールによるピック&プレース作業

### 技術制約
- VCモデル全体に対するPythonコンソールは使用不可
- 個別コンポーネント（ロボット、モジュール）レベルでのPythonスクリプトのみ有効
- 外部アプリケーションとの通信はTCP/UDP経由で実現

## 2. システム要件定義

### 2.1 ロボット基本仕様
- **ロボット種類**: 6軸垂直多関節ロボット（標準）
- **可動範囲**: 各軸の動作限界角度・位置
- **ペイロード**: 最大搬送重量（kg）
- **到達距離**: 作業半径（mm）
- **位置精度**: 繰り返し位置決め精度（±mm）
- **最大速度**: 各軸最大動作速度
- **安全機能**: 緊急停止、速度制限、干渉チェック

### 2.2 ツール仕様

#### スポット溶接ガン
- **TCP設定**: ツール座標系（X,Y,Z,Rx,Ry,Rz）
- **ガン仕様**: ヘッド寸法、開閉ストローク、電極位置
- **制御パラメータ**: 加圧力、溶接時間、冷却制御
- **センサ**: 電極摩耗検出、加圧力センサ

#### クランプ式ハンドリングツール
- **TCP設定**: グリップ中心座標
- **把持仕様**: グリップ力、開閉ストローク、把持範囲
- **センサ**: 把持確認、力覚センサ、位置検出
- **制御**: 開閉速度、把持力制御

### 2.3 I/O要件

#### 入力信号（外部システムから）
```
- START: 作業開始信号
- E_STOP: 緊急停止信号
- PAUSE/RESUME: 一時停止・再開信号
- MODE_SELECT: 作業モード選択（AUTO/MANUAL/TEACH）
- PART_PRESENT: ワーク検出センサ
- JIG_CLAMPED: 治具クランプ完了
- DOOR_CLOSED: 安全扉状態
- AIR_PRESSURE_OK: エアー圧力正常
- WELD_ENABLE: 溶接許可信号（スポット溶接用）
- WELDER_READY: 溶接機準備完了（スポット溶接用）
- SUPPLY_PART_OK: 供給部ワーク有無（荷役用）
- DISCHARGE_READY: 排出部空き確認（荷役用）
```

#### 出力信号（外部システムへ）
```
- WORK_COMPLETE: 作業完了信号
- ERROR_OCCURRED: 異常発生信号
- READY: 待機中信号
- WORKING: 作業中信号
- STEP_NUMBER: 現在作業ステップ
- ERROR_CODE: エラーコード出力
- WELD_EXECUTE: 溶接実行指令（スポット溶接用）
- WELD_COMPLETE: 溶接完了確認（スポット溶接用）
- PICK_COMPLETE: ピック完了（荷役用）
- PLACE_COMPLETE: プレース完了（荷役用）
```

## 3. 開発フェーズ計画

### Phase 1: 基盤通信システム（最優先）
#### 目的
外部アプリケーションとVisual Components間のデータ送受信基盤を構築

#### 実装内容
- **通信プロトコル**: TCP/UDP通信インターフェース
- **データ形式**: JSON形式での標準化メッセージング
- **接続管理**: 接続状態監視・自動再接続機能
- **基本コマンド**: ping/pong、状態取得、基本制御

#### メッセージフォーマット例
```json
{
  "message_id": "unique_identifier",
  "timestamp": "2025-08-23T10:30:00.000Z",
  "command_type": "robot_move|io_control|status_request|tool_control",
  "target_component": "robot_1|tool_1|fixture_1|controller",
  "parameters": {
    "position": [x, y, z, rx, ry, rz],
    "speed": 100,
    "io_data": {"output_1": true, "output_2": false},
    "tool_command": "open|close|weld_start|weld_stop"
  },
  "response_required": true
}
```

### Phase 2: シンプルなハンドリング作業（実証優先）
#### 目的
クランプ式ツールでの基本ピック&プレース機能を実装

#### 実装内容
- **固定位置間搬送**: 予め設定された位置間での基本動作
- **基本軌道生成**: 直線補間による安全な移動経路
- **把持制御**: ツール開閉・把持確認
- **基本I/O制御**: 把持確認、完了通知信号

#### 作業シーケンス
```
1. 初期位置待機
2. 外部START信号受信
3. ピック位置へ移動
4. ツール開放
5. ワーク把持
6. 把持確認
7. プレース位置へ移動
8. ワーク放置
9. ツール開放確認
10. 初期位置復帰
11. 完了信号出力
```

### Phase 3: I/Oインターロック拡張
#### 目的
安全機能と外部機器協調動作を実装

#### 実装内容
- **安全インターロック**: 緊急停止、安全扉監視、エアー圧監視
- **エラーハンドリング**: 異常検出、エラーコード生成、復旧処理
- **協調制御**: 外部コンベア、治具との協調動作
- **タイムアウト制御**: 各工程での異常時間監視

### Phase 4: スポット溶接適用
#### 目的
溶接特有の高精度制御と品質管理機能を追加

#### 実装内容
- **高精度TCP制御**: 溶接点への正確なアプローチ
- **溶接シーケンス**: 溶接条件設定、実行、確認
- **品質管理**: 溶接結果監視、不良検出
- **溶接点最適化**: 効率的な溶接順序計算

## 4. システム構成

### 外部アプリケーション（Python推奨）
```python
# 主要機能
class RobotTeachingApp:
    def __init__(self):
        self.tcp_server = TCPServer()
        self.command_processor = CommandProcessor()
        self.trajectory_generator = TrajectoryGenerator()
        self.io_controller = IOController()
    
    def generate_teaching_data(self, work_data, application_type):
        # CADデータから教示データ生成
        pass
    
    def send_robot_program(self, program_data):
        # VCへプログラム送信
        pass
    
    def monitor_execution(self):
        # 実行状況監視
        pass
```

### Visual Components側実装

#### ロボットコンポーネントスクリプト
```python
# ロボット制御スクリプト（VCコンポーネント内）
import socket
import json
from vcScript import *

class RobotController:
    def __init__(self):
        self.tcp_client = socket.socket()
        self.current_program = []
        self.execution_state = "READY"
    
    def connect_external_app(self):
        # 外部アプリとの接続確立
        pass
    
    def execute_program(self, program_data):
        # 受信プログラムの実行
        pass
    
    def send_status(self):
        # 状態情報送信
        pass
```

#### ツールコンポーネントスクリプト
```python
# ツール制御スクリプト（VCコンポーネント内）
class ToolController:
    def __init__(self, tool_type):
        self.tool_type = tool_type  # "welding_gun" or "gripper"
        self.tcp_offset = [0, 0, 0, 0, 0, 0]
        self.tool_state = "READY"
    
    def execute_tool_command(self, command):
        # ツール固有コマンド実行
        pass
    
    def update_tcp(self):
        # TCP補正計算
        pass
```

## 5. 開発要求仕様

### 5.1 機能要件
- 外部CADデータからの自動教示データ生成
- リアルタイム実行制御・監視
- 複数ロボット・ツールの統合制御
- 安全機能の確実な実装
- エラー発生時の適切な処理・復旧

### 5.2 性能要件
- 通信レイテンシ: 100ms以下
- 位置精度: ±0.1mm（スポット溶接）、±0.5mm（ハンドリング）
- サイクルタイム: 作業内容に応じた最適化
- 可用性: 99%以上の稼働率

### 5.3 拡張要件
- 他のアプリケーション（アーク溶接、塗装等）への展開可能性
- 複数ロボットシステムへの拡張対応
- AIによる軌道最適化機能の組み込み余地

## 6. 開発ガイドライン

### コーディング規約
- 可読性を重視したコメント記載
- エラーハンドリングの徹底
- ログ出力による動作トレース
- 設定の外部ファイル化

### テスト方針
- 単体テスト: 各コンポーネント単独での動作確認
- 結合テスト: 通信・協調動作の確認
- システムテスト: 実際の作業での動作検証
- 安全テスト: 異常時の安全停止確認

### ドキュメント要件
- システム設計書
- API仕様書
- 操作マニュアル
- トラブルシューティングガイド

---

このプロンプトに基づいて、Phase 1から順次開発を進めることで、Visual Componentsを活用した実用的なロボット教示データ自動生成システムが構築できます。