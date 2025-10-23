"""
リモート監視サービス (Phase 4-2)
リアルタイム生産監視ダッシュボードとWeb API
WebSocketによるリアルタイム通信とREST API実装
"""

import json
import time
import logging
import asyncio
import websockets
import threading
import queue
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid
import sqlite3

# Webサーバー関連
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from production_management_integration import ProductionManagementSystem, ProductionMetrics
from cloud_connector import CloudDataSynchronizer, CloudConnector
from multi_robot_coordination import RobotState, RobotInfo

logger = logging.getLogger(__name__)

class MonitoringEventType(Enum):
    """監視イベントタイプ"""
    ROBOT_STATUS_CHANGE = "robot_status_change"
    PRODUCTION_UPDATE = "production_update"
    QUALITY_ALERT = "quality_alert"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_METRICS = "performance_metrics"
    MAINTENANCE_ALERT = "maintenance_alert"

@dataclass
class MonitoringEvent:
    """監視イベント"""
    event_id: str
    event_type: MonitoringEventType
    timestamp: datetime
    source: str  # イベントソース
    data: Dict[str, Any]
    severity: str = "info"  # info, warning, error, critical
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

@dataclass
class RealTimeMetrics:
    """リアルタイムメトリクス"""
    timestamp: datetime
    robot_id: str
    production_order_id: str
    work_order_id: str
    cycle_time: float
    throughput: float
    quality_score: float
    oee: float
    availability: float
    performance: float
    quality_rate: float
    energy_consumption: float
    maintenance_alerts: int

class MonitoringWebSocketManager:
    """WebSocket接続マネージャー"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_subscriptions: Dict[WebSocket, List[str]] = {}
        self.connection_lock = threading.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        """WebSocket接続受付"""
        await websocket.accept()

        with self.connection_lock:
            self.active_connections.add(websocket)
            self.connection_subscriptions[websocket] = []

        logger.info(f"WebSocket client connected: {client_id}")

        # 接続確認メッセージ送信
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        """WebSocket接続切断"""
        with self.connection_lock:
            self.active_connections.discard(websocket)
            if websocket in self.connection_subscriptions:
                del self.connection_subscriptions[websocket]

        logger.info("WebSocket client disconnected")

    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """個人メッセージ送信"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")

    async def broadcast(self, message: Dict[str, Any], event_type: str = None):
        """ブロードキャスト送信"""
        if not self.active_connections:
            return

        message_str = json.dumps(message, default=str)
        disconnected = []

        with self.connection_lock:
            connections_copy = self.active_connections.copy()

        for connection in connections_copy:
            try:
                # サブスクリプションフィルタリング
                if event_type and self.connection_subscriptions.get(connection):
                    if event_type not in self.connection_subscriptions[connection]:
                        continue

                await connection.send_text(message_str)

            except Exception as e:
                logger.error(f"Failed to send broadcast message: {e}")
                disconnected.append(connection)

        # 切断された接続をクリーンアップ
        for connection in disconnected:
            self.disconnect(connection)

    def subscribe(self, websocket: WebSocket, event_types: List[str]):
        """イベントタイプをサブスクライブ"""
        with self.connection_lock:
            if websocket in self.connection_subscriptions:
                self.connection_subscriptions[websocket] = event_types

    def get_connection_count(self) -> int:
        """接続数取得"""
        with self.connection_lock:
            return len(self.active_connections)

class RealTimeDataCollector:
    """リアルタイムデータ収集器"""

    def __init__(self, production_system: ProductionManagementSystem, cloud_synchronizer: CloudDataSynchronizer = None):
        self.production_system = production_system
        self.cloud_synchronizer = cloud_synchronizer

        # データ収集スレッド制御
        self.running = False
        self.data_collection_thread: Optional[threading.Thread] = None
        self.alert_monitoring_thread: Optional[threading.Thread] = None

        # データ収集間隔
        self.collection_interval = 5.0  # 5秒間隔
        self.alert_check_interval = 10.0  # 10秒間隔

        # データキュー
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.event_queue = queue.Queue(maxsize=1000)

        # リアルタイムデータキャッシュ
        self.current_metrics: Dict[str, RealTimeMetrics] = {}
        self.recent_events: List[MonitoringEvent] = []
        self.robot_status_cache: Dict[str, RobotState] = {}

        # しきい値設定
        self.thresholds = {
            'oee_warning': 0.70,
            'oee_critical': 0.50,
            'quality_warning': 0.90,
            'quality_critical': 0.80,
            'cycle_time_warning': 120.0,  # 秒
            'cycle_time_critical': 180.0,
            'availability_warning': 0.85,
            'availability_critical': 0.70
        }

        # コールバック
        self.on_metrics_collected: Optional[Callable[[RealTimeMetrics], None]] = None
        self.on_alert_generated: Optional[Callable[[MonitoringEvent], None]] = None

    def start(self) -> bool:
        """データ収集開始"""
        try:
            if self.running:
                logger.warning("Real-time data collector already running")
                return False

            self.running = True

            # データ収集スレッド開始
            self.data_collection_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
            self.data_collection_thread.start()

            # アラート監視スレッド開始
            self.alert_monitoring_thread = threading.Thread(target=self._alert_monitoring_loop, daemon=True)
            self.alert_monitoring_thread.start()

            logger.info("Real-time data collector started")
            return True

        except Exception as e:
            logger.error(f"Failed to start real-time data collector: {e}")
            return False

    def stop(self):
        """データ収集停止"""
        self.running = False

        if self.data_collection_thread and self.data_collection_thread.is_alive():
            self.data_collection_thread.join(timeout=5.0)

        if self.alert_monitoring_thread and self.alert_monitoring_thread.is_alive():
            self.alert_monitoring_thread.join(timeout=5.0)

        logger.info("Real-time data collector stopped")

    def _data_collection_loop(self):
        """データ収集ループ"""
        logger.info("Data collection loop started")

        while self.running:
            try:
                self._collect_production_metrics()
                self._collect_robot_status()
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Data collection error: {e}")
                time.sleep(10.0)

        logger.info("Data collection loop ended")

    def _alert_monitoring_loop(self):
        """アラート監視ループ"""
        logger.info("Alert monitoring loop started")

        while self.running:
            try:
                self._check_for_alerts()
                time.sleep(self.alert_check_interval)

            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                time.sleep(10.0)

        logger.info("Alert monitoring loop ended")

    def _collect_production_metrics(self):
        """生産メトリクス収集"""
        try:
            # 生産システムからメトリクス取得
            dashboard_data = self.production_system.get_production_dashboard()

            # リアルタイムメトリクス生成
            current_time = datetime.now()

            # 各ロボットのメトリクスを収集（模擬）
            for robot_id in ['robot_001', 'robot_002', 'robot_003']:
                metrics = RealTimeMetrics(
                    timestamp=current_time,
                    robot_id=robot_id,
                    production_order_id="PO_CURRENT",
                    work_order_id=f"WO_{robot_id}_ACTIVE",
                    cycle_time=45.0 + (hash(robot_id) % 20) - 10,
                    throughput=60.0,
                    quality_score=0.95 - (hash(robot_id) % 10) * 0.01,
                    oee=0.80 + (hash(robot_id) % 20) * 0.01,
                    availability=0.90,
                    performance=0.92,
                    quality_rate=0.95,
                    energy_consumption=5.5,
                    maintenance_alerts=0
                )

                self.current_metrics[robot_id] = metrics

                # メトリクスキューに追加
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    logger.warning("Metrics queue full, dropping metrics")

                # コールバック実行
                if self.on_metrics_collected:
                    self.on_metrics_collected(metrics)

            # クラウド同期
            if self.cloud_synchronizer:
                for robot_id, metrics in self.current_metrics.items():
                    self.cloud_synchronizer.queue_sync_data('production_metrics', asdict(metrics))

        except Exception as e:
            logger.error(f"Production metrics collection failed: {e}")

    def _collect_robot_status(self):
        """ロボットステータス収集"""
        try:
            # ロボット状態収集（模擬）
            for robot_id in ['robot_001', 'robot_002', 'robot_003']:
                # 実際の実装ではロボットから直接状態を取得
                robot_info = RobotInfo(
                    robot_id=robot_id,
                    robot_type="6_axis_robot",
                    current_position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    status=RobotState.IDLE,
                    payload=0.0,
                    workspace_bounds=[[-1000, -1000, 0], [1000, 1000, 2000]]
                )

                self.robot_status_cache[robot_id] = RobotState.IDLE

        except Exception as e:
            logger.error(f"Robot status collection failed: {e}")

    def _check_for_alerts(self):
        """アラートチェック"""
        try:
            for robot_id, metrics in self.current_metrics.items():
                # OEEアラートチェック
                if metrics.oee < self.thresholds['oee_critical']:
                    self._generate_alert(
                        event_type=MonitoringEventType.SYSTEM_ALERT,
                        source=robot_id,
                        severity="critical",
                        data={
                            "metric": "oee",
                            "value": metrics.oee,
                            "threshold": self.thresholds['oee_critical'],
                            "message": f"OEE critically low for {robot_id}"
                        }
                    )
                elif metrics.oee < self.thresholds['oee_warning']:
                    self._generate_alert(
                        event_type=MonitoringEventType.SYSTEM_ALERT,
                        source=robot_id,
                        severity="warning",
                        data={
                            "metric": "oee",
                            "value": metrics.oee,
                            "threshold": self.thresholds['oee_warning'],
                            "message": f"OEE below warning threshold for {robot_id}"
                        }
                    )

                # 品質アラートチェック
                if metrics.quality_score < self.thresholds['quality_critical']:
                    self._generate_alert(
                        event_type=MonitoringEventType.QUALITY_ALERT,
                        source=robot_id,
                        severity="critical",
                        data={
                            "metric": "quality_score",
                            "value": metrics.quality_score,
                            "threshold": self.thresholds['quality_critical'],
                            "message": f"Quality critically low for {robot_id}"
                        }
                    )

                # サイクルタイムアラートチェック
                if metrics.cycle_time > self.thresholds['cycle_time_critical']:
                    self._generate_alert(
                        event_type=MonitoringEventType.PERFORMANCE_METRICS,
                        source=robot_id,
                        severity="warning",
                        data={
                            "metric": "cycle_time",
                            "value": metrics.cycle_time,
                            "threshold": self.thresholds['cycle_time_critical'],
                            "message": f"Cycle time too slow for {robot_id}"
                        }
                    )

        except Exception as e:
            logger.error(f"Alert checking failed: {e}")

    def _generate_alert(self, event_type: MonitoringEventType, source: str,
                       severity: str, data: Dict[str, Any]):
        """アラート生成"""
        event = MonitoringEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            data=data,
            severity=severity
        )

        # イベントキューに追加
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping alert")

        # 最近のイベントリストに追加
        self.recent_events.append(event)

        # 古いイベントを削除（最新100件を保持）
        if len(self.recent_events) > 100:
            self.recent_events = self.recent_events[-100:]

        # コールバック実行
        if self.on_alert_generated:
            self.on_alert_generated(event)

        logger.warning(f"Alert generated: {severity} - {data.get('message', 'Unknown alert')}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """現在のメトリクス取得"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {robot_id: asdict(metrics) for robot_id, metrics in self.current_metrics.items()},
            "robot_status": self.robot_status_cache.copy()
        }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """最近のイベント取得"""
        events = sorted(self.recent_events, key=lambda e: e.timestamp, reverse=True)
        return [asdict(event) for event in events[:limit]]

    def acknowledge_event(self, event_id: str, acknowledged_by: str) -> bool:
        """イベント確認"""
        for event in self.recent_events:
            if event.event_id == event_id:
                event.acknowledged = True
                event.acknowledged_by = acknowledged_by
                event.acknowledged_at = datetime.now()
                return True
        return False

# FastAPIモデル（使用可能な場合）
if FASTAPI_AVAILABLE:
    class MetricsResponse(BaseModel):
        timestamp: str
        metrics: Dict[str, Dict[str, Any]]
        robot_status: Dict[str, str]

    class EventRequest(BaseModel):
        event_type: str
        source: str
        severity: str
        data: Dict[str, Any]

    class SubscriptionRequest(BaseModel):
        event_types: List[str]

class RemoteMonitoringService:
    """リモート監視サービス"""

    def __init__(self, production_system: ProductionManagementSystem,
                 cloud_synchronizer: CloudDataSynchronizer = None, port: int = 8090):
        self.production_system = production_system
        self.cloud_synchronizer = cloud_synchronizer
        self.port = port

        # WebSocketマネージャー
        self.websocket_manager = MonitoringWebSocketManager()

        # リアルタイムデータ収集器
        self.data_collector = RealTimeDataCollector(production_system, cloud_synchronizer)

        # FastAPIアプリケーション
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available. Remote monitoring service disabled.")

        # サーバースレッド
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        # 設定
        self.cors_origins = [
            "http://localhost:3000",  # React開発サーバー
            "http://localhost:8080",  # 既存Web UI
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ]

    def _create_fastapi_app(self) -> FastAPI:
        """FastAPIアプリケーション作成"""
        app = FastAPI(
            title="Remote Monitoring Service",
            description="Real-time production monitoring API",
            version="1.0.0"
        )

        # CORSミドルウェア
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # データ収集器コールバック設定
        self.data_collector.on_metrics_collected = self._on_metrics_collected
        self.data_collector.on_alert_generated = self._on_alert_generated

        # APIルート定義
        self._setup_api_routes(app)

        return app

    def _setup_api_routes(self, app: FastAPI):
        """APIルート設定"""

        @app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """監視ダッシュボードHTML"""
            return self._get_dashboard_html()

        @app.get("/api/health")
        async def health_check():
            """ヘルスチェック"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "remote_monitoring",
                "version": "1.0.0"
            }

        @app.get("/api/metrics", response_model=MetricsResponse)
        async def get_current_metrics():
            """現在のメトリクス取得"""
            return self.data_collector.get_current_metrics()

        @app.get("/api/events")
        async def get_recent_events(limit: int = Query(50, le=100)):
            """最近のイベント取得"""
            return self.data_collector.get_recent_events(limit)

        @app.post("/api/events/acknowledge/{event_id}")
        async def acknowledge_event(event_id: str, user: str = Query(...)):
            """イベント確認"""
            success = self.data_collector.acknowledge_event(event_id, user)
            if success:
                return {"status": "acknowledged", "event_id": event_id}
            else:
                raise HTTPException(status_code=404, detail="Event not found")

        @app.get("/api/production/summary")
        async def get_production_summary():
            """生産サマリー取得"""
            return self.production_system.get_production_dashboard()

        @app.get("/api/robots/status")
        async def get_robot_status():
            """ロボット状態取得"""
            return {
                "robots": [
                    {
                        "robot_id": robot_id,
                        "status": status.value if isinstance(status, Enum) else status,
                        "last_updated": datetime.now().isoformat()
                    }
                    for robot_id, status in self.data_collector.robot_status_cache.items()
                ]
            }

        @app.get("/api/cloud/status")
        async def get_cloud_status():
            """クラウド同期状態取得"""
            if self.cloud_synchronizer:
                return self.cloud_synchronizer.get_sync_status()
            else:
                return {"status": "not_configured"}

        @app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocketエンドポイント"""
            await self.websocket_manager.connect(websocket, client_id)

            try:
                while True:
                    # クライアントからのメッセージ受信
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # メッセージタイプに応じた処理
                    if message.get("type") == "subscribe":
                        event_types = message.get("event_types", [])
                        self.websocket_manager.subscribe(websocket, event_types)
                        await self.websocket_manager.send_personal_message(websocket, {
                            "type": "subscription_confirmed",
                            "event_types": event_types
                        })

                    elif message.get("type") == "get_metrics":
                        metrics = self.data_collector.get_current_metrics()
                        await self.websocket_manager.send_personal_message(websocket, {
                            "type": "metrics_response",
                            "data": metrics
                        })

            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)

        @app.post("/api/manual/event")
        async def create_manual_event(event: EventRequest):
            """手動イベント作成"""
            monitoring_event = MonitoringEvent(
                event_id=str(uuid.uuid4()),
                event_type=MonitoringEventType(event.event_type),
                timestamp=datetime.now(),
                source=event.source,
                severity=event.severity,
                data=event.data
            )

            self.data_collector.recent_events.append(monitoring_event)

            # WebSocketブロードキャスト
            await self.websocket_manager.broadcast({
                "type": "new_event",
                "event": asdict(monitoring_event)
            }, event.event_type)

            return {"status": "created", "event_id": monitoring_event.event_id}

        @app.get("/api/statistics")
        async def get_statistics():
            """統計情報取得"""
            current_metrics = self.data_collector.get_current_metrics()
            recent_events = self.data_collector.get_recent_events()

            # 基本統計計算
            active_robots = len(current_metrics["metrics"])
            recent_alerts = len([e for e in recent_events if e["severity"] in ["warning", "error", "critical"]])

            # 平均メトリクス計算
            if current_metrics["metrics"]:
                avg_oee = sum(m["oee"] for m in current_metrics["metrics"].values()) / len(current_metrics["metrics"])
                avg_quality = sum(m["quality_score"] for m in current_metrics["metrics"].values()) / len(current_metrics["metrics"])
                avg_cycle_time = sum(m["cycle_time"] for m in current_metrics["metrics"].values()) / len(current_metrics["metrics"])
            else:
                avg_oee = avg_quality = avg_cycle_time = 0

            return {
                "timestamp": datetime.now().isoformat(),
                "active_robots": active_robots,
                "websocket_connections": self.websocket_manager.get_connection_count(),
                "recent_alerts_24h": recent_alerts,
                "average_oee": round(avg_oee, 3),
                "average_quality": round(avg_quality, 3),
                "average_cycle_time": round(avg_cycle_time, 1),
                "data_collector_running": self.data_collector.running
            }

    def _get_dashboard_html(self) -> str:
        """監視ダッシュボードHTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Production Monitoring Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .dashboard { max-width: 1400px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
                .metric-label { color: #7f8c8d; margin-top: 5px; }
                .events-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .event { padding: 10px; border-left: 4px solid #3498db; margin: 5px 0; background: #f8f9fa; }
                .event.warning { border-left-color: #f39c12; }
                .event.error { border-left-color: #e74c3c; }
                .event.critical { border-left-color: #c0392b; }
                .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
                .status-connected { background: #27ae60; }
                .status-disconnected { background: #e74c3c; }
                .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .refresh-btn:hover { background: #2980b9; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>Production Monitoring Dashboard</h1>
                    <div>
                        <span class="status-indicator status-connected"></span>
                        <span id="connection-status">Connected</span>
                        <button class="refresh-btn" onclick="refreshData()">Refresh</button>
                    </div>
                </div>

                <div class="metrics-grid" id="metrics-grid">
                    <!-- メトリクスカードがここに表示される -->
                </div>

                <div class="events-section">
                    <h2>Recent Events & Alerts</h2>
                    <div id="events-list">
                        <!-- イベントリストがここに表示される -->
                    </div>
                </div>
            </div>

            <script>
                const clientId = 'dashboard_' + Date.now();
                let ws;

                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;

                    ws = new WebSocket(wsUrl);

                    ws.onopen = function() {
                        console.log('WebSocket connected');
                        document.getElementById('connection-status').textContent = 'Connected';
                        document.querySelector('.status-indicator').className = 'status-indicator status-connected';

                        // すべてのイベントタイプをサブスクライブ
                        ws.send(JSON.stringify({
                            type: 'subscribe',
                            event_types: ['production_update', 'quality_alert', 'system_alert', 'performance_metrics']
                        }));

                        // 初期データ要求
                        ws.send(JSON.stringify({ type: 'get_metrics' }));
                    };

                    ws.onmessage = function(event) {
                        const message = JSON.parse(event.data);
                        handleWebSocketMessage(message);
                    };

                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        document.getElementById('connection-status').textContent = 'Disconnected';
                        document.querySelector('.status-indicator').className = 'status-indicator status-disconnected';

                        // 5秒後に再接続
                        setTimeout(connectWebSocket, 5000);
                    };

                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };
                }

                function handleWebSocketMessage(message) {
                    switch(message.type) {
                        case 'metrics_response':
                            updateMetrics(message.data);
                            break;
                        case 'new_event':
                            addEvent(message.event);
                            break;
                        case 'connection_established':
                            console.log('Connection established:', message.client_id);
                            break;
                    }
                }

                function updateMetrics(metricsData) {
                    const metricsGrid = document.getElementById('metrics-grid');
                    metricsGrid.innerHTML = '';

                    for (const [robotId, metrics] of Object.entries(metricsData.metrics)) {
                        const card = createMetricCard(robotId, metrics);
                        metricsGrid.appendChild(card);
                    }
                }

                function createMetricCard(robotId, metrics) {
                    const card = document.createElement('div');
                    card.className = 'metric-card';

                    const oeeColor = metrics.oee > 0.8 ? '#27ae60' : metrics.oee > 0.6 ? '#f39c12' : '#e74c3c';
                    const qualityColor = metrics.quality_score > 0.9 ? '#27ae60' : metrics.quality_score > 0.8 ? '#f39c12' : '#e74c3c';

                    card.innerHTML = `
                        <h3>${robotId}</h3>
                        <div class="metric-value" style="color: ${oeeColor}">${(metrics.oee * 100).toFixed(1)}%</div>
                        <div class="metric-label">OEE</div>
                        <div class="metric-value" style="color: ${qualityColor}; font-size: 1.5em">${(metrics.quality_score * 100).toFixed(1)}%</div>
                        <div class="metric-label">Quality</div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #7f8c8d;">
                            Cycle Time: ${metrics.cycle_time.toFixed(1)}s<br>
                            Throughput: ${metrics.throughput.toFixed(1)}/hr<br>
                            Availability: ${(metrics.availability * 100).toFixed(1)}%
                        </div>
                    `;

                    return card;
                }

                function addEvent(event) {
                    const eventsList = document.getElementById('events-list');
                    const eventDiv = document.createElement('div');
                    eventDiv.className = `event ${event.severity}`;

                    const timestamp = new Date(event.timestamp).toLocaleString();
                    eventDiv.innerHTML = `
                        <strong>${event.source}</strong> - ${event.data.message || event.event_type}
                        <br><small>${timestamp} - Severity: ${event.severity}</small>
                    `;

                    eventsList.insertBefore(eventDiv, eventsList.firstChild);

                    // 最新10件のみを保持
                    while (eventsList.children.length > 10) {
                        eventsList.removeChild(eventsList.lastChild);
                    }
                }

                async function refreshData() {
                    try {
                        const response = await fetch('/api/metrics');
                        const data = await response.json();
                        updateMetrics(data);

                        const eventsResponse = await fetch('/api/events?limit=10');
                        const events = await eventsResponse.json();
                        const eventsList = document.getElementById('events-list');
                        eventsList.innerHTML = '';
                        events.forEach(event => addEvent(event));

                    } catch (error) {
                        console.error('Failed to refresh data:', error);
                    }
                }

                // 初期化
                connectWebSocket();

                // 30秒ごとにデータ更新
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """

    async def _on_metrics_collected(self, metrics: RealTimeMetrics):
        """メトリクス収集時コールバック"""
        # WebSocketブロードキャスト
        await self.websocket_manager.broadcast({
            "type": "metrics_update",
            "robot_id": metrics.robot_id,
            "metrics": asdict(metrics)
        }, "performance_metrics")

    async def _on_alert_generated(self, event: MonitoringEvent):
        """アラート生成時コールバック"""
        # WebSocketブロードキャスト
        await self.websocket_manager.broadcast({
            "type": "new_event",
            "event": asdict(event)
        }, event.event_type.value)

    def start(self) -> bool:
        """サービス開始"""
        try:
            if not FASTAPI_AVAILABLE:
                logger.error("FastAPI not available. Cannot start remote monitoring service.")
                return False

            if self.running:
                logger.warning("Remote monitoring service already running")
                return False

            self.running = True

            # データ収集器開始
            if not self.data_collector.start():
                logger.error("Failed to start data collector")
                return False

            # FastAPIサーバースレッド開始
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()

            logger.info(f"Remote monitoring service started on port {self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start remote monitoring service: {e}")
            return False

    def _run_server(self):
        """サーバー実行"""
        try:
            uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"Server error: {e}")

    def stop(self):
        """サービス停止"""
        self.running = False

        # データ収集器停止
        self.data_collector.stop()

        logger.info("Remote monitoring service stopped")

    def get_service_status(self) -> Dict[str, Any]:
        """サービス状態取得"""
        return {
            "running": self.running,
            "port": self.port,
            "websocket_connections": self.websocket_manager.get_connection_count(),
            "data_collector_running": self.data_collector.running,
            "current_metrics_count": len(self.data_collector.current_metrics),
            "recent_events_count": len(self.data_collector.recent_events),
            "service_url": f"http://localhost:{self.port}"
        }

# グローバルインスタンス
remote_monitoring_service: Optional[RemoteMonitoringService] = None

def initialize_remote_monitoring(production_system: ProductionManagementSystem,
                               cloud_synchronizer: CloudDataSynchronizer = None,
                               port: int = 8090) -> RemoteMonitoringService:
    """リモート監視サービス初期化"""
    global remote_monitoring_service
    remote_monitoring_service = RemoteMonitoringService(production_system, cloud_synchronizer, port)
    return remote_monitoring_service

def get_remote_monitoring_service() -> Optional[RemoteMonitoringService]:
    """リモート監視サービス取得"""
    return remote_monitoring_service

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Remote Monitoring Service...")

    try:
        # モック生産管理システム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())

        # リモート監視サービス初期化
        service = initialize_remote_monitoring(mock_pms)

        if service.start():
            print(f"Remote monitoring service started successfully!")
            print(f"Dashboard URL: http://localhost:{service.port}")
            print("Press Ctrl+C to stop...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                service.stop()

        else:
            print("Failed to start remote monitoring service")

    except Exception as e:
        print(f"Remote monitoring service test failed: {e}")