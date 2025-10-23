"""
モバイルアプリ (Phase 4-3)
スマートフォン・タブレット対応の軽量監視アプリケーション
PWA対応・オフライン機能・プッシュ通知
"""

import json
import time
import logging
import threading
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import sqlite3
import hashlib
import base64

# Webサーバー関連
try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from production_management_integration import ProductionManagementSystem
from remote_monitoring_service import RealTimeMetrics, MonitoringEvent

logger = logging.getLogger(__name__)

class MobileDeviceType(Enum):
    """モバイルデバイスタイプ"""
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    DESKTOP = "desktop"

class NotificationType(Enum):
    """通知タイプ"""
    ALERT = "alert"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"

@dataclass
class MobileDevice:
    """モバイルデバイス情報"""
    device_id: str
    device_type: MobileDeviceType
    user_agent: str
    last_seen: datetime
    push_token: Optional[str] = None
    notification_preferences: Dict[str, bool] = None
    is_active: bool = True

@dataclass
class MobileNotification:
    """モバイル通知"""
    notification_id: str
    device_id: str
    notification_type: NotificationType
    title: str
    body: str
    data: Dict[str, Any]
    created_at: datetime
    sent_at: Optional[datetime] = None
    read: bool = False

class MobileAppCache:
    """モバイルアプリキャッシュ"""

    def __init__(self, cache_dir: str = "mobile_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # キャッシュデータベース
        self.db_path = self.cache_dir / "mobile_cache.db"
        self._initialize_cache_db()

    def _initialize_cache_db(self):
        """キャッシュデータベース初期化"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cached_data (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS offline_actions (
                    action_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    synced BOOLEAN DEFAULT FALSE
                );

                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cached_data (expires_at);
                CREATE INDEX IF NOT EXISTS idx_offline_synced ON offline_actions (synced);
            """)

    def set_cache(self, key: str, data: Any, ttl_seconds: int = 300):
        """キャッシュ設定"""
        timestamp = datetime.now()
        expires_at = timestamp + timedelta(seconds=ttl_seconds)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cached_data (key, data, timestamp, expires_at)
                VALUES (?, ?, ?, ?)
            """, (
                key,
                json.dumps(data, default=str),
                timestamp.isoformat(),
                expires_at.isoformat()
            ))

    def get_cache(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT data, expires_at FROM cached_data
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (key, datetime.now().isoformat()))

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def store_offline_action(self, device_id: str, action_type: str, action_data: Dict[str, Any]) -> str:
        """オフラインアクション保存"""
        action_id = hashlib.md5(f"{device_id}{action_type}{time.time()}".encode()).hexdigest()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO offline_actions (action_id, device_id, action_type, action_data, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                action_id,
                device_id,
                action_type,
                json.dumps(action_data),
                datetime.now().isoformat()
            ))

        return action_id

    def get_pending_actions(self, device_id: str) -> List[Dict[str, Any]]:
        """未同期アクション取得"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT action_id, action_type, action_data, created_at
                FROM offline_actions
                WHERE device_id = ? AND synced = FALSE
                ORDER BY created_at
            """, (device_id,))

            actions = []
            for row in cursor.fetchall():
                actions.append({
                    'action_id': row[0],
                    'action_type': row[1],
                    'action_data': json.loads(row[2]),
                    'created_at': row[3]
                })

            return actions

    def mark_action_synced(self, action_id: str):
        """アクション同期済みマーク"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                UPDATE offline_actions SET synced = TRUE WHERE action_id = ?
            """, (action_id,))

    def cleanup_expired_cache(self):
        """期限切れキャッシュクリーンアップ"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                DELETE FROM cached_data WHERE expires_at < ?
            """, (datetime.now().isoformat(),))

class MobileNotificationService:
    """モバイル通知サービス"""

    def __init__(self):
        self.devices: Dict[str, MobileDevice] = {}
        self.notifications: List[MobileNotification] = []
        self.notification_queue = []

        # プッシュ通知設定（模擬）
        self.push_service_available = False

    def register_device(self, device_id: str, device_type: MobileDeviceType,
                       user_agent: str, push_token: str = None) -> MobileDevice:
        """デバイス登録"""
        device = MobileDevice(
            device_id=device_id,
            device_type=device_type,
            user_agent=user_agent,
            last_seen=datetime.now(),
            push_token=push_token,
            notification_preferences={
                "alerts": True,
                "production_updates": True,
                "quality_alerts": True,
                "system_notifications": False
            }
        )

        self.devices[device_id] = device
        logger.info(f"Mobile device registered: {device_id} ({device_type.value})")
        return device

    def update_device_activity(self, device_id: str):
        """デバイスアクティビティ更新"""
        if device_id in self.devices:
            self.devices[device_id].last_seen = datetime.now()
            self.devices[device_id].is_active = True

    def send_notification(self, device_id: str, notification_type: NotificationType,
                         title: str, body: str, data: Dict[str, Any] = None) -> str:
        """通知送信"""
        notification = MobileNotification(
            notification_id=str(uuid.uuid4()),
            device_id=device_id,
            notification_type=notification_type,
            title=title,
            body=body,
            data=data or {},
            created_at=datetime.now()
        )

        self.notifications.append(notification)
        self.notification_queue.append(notification)

        # 実際のプッシュ通知送信（模擬）
        if self.push_service_available and device_id in self.devices:
            device = self.devices[device_id]
            if device.push_token and self._should_send_notification(device, notification_type):
                self._send_push_notification(device, notification)

        logger.info(f"Notification sent to {device_id}: {title}")
        return notification.notification_id

    def _should_send_notification(self, device: MobileDevice, notification_type: NotificationType) -> bool:
        """通知送信べきか判断"""
        if not device.notification_preferences:
            return False

        preferences = device.notification_preferences
        if notification_type == NotificationType.WARNING and preferences.get("alerts"):
            return True
        elif notification_type == NotificationType.INFO and preferences.get("production_updates"):
            return True
        elif notification_type == NotificationType.ALERT and preferences.get("quality_alerts"):
            return True

        return False

    def _send_push_notification(self, device: MobileDevice, notification: MobileNotification):
        """プッシュ通知送信（模擬実装）"""
        # 実際の実装ではFirebase Cloud MessagingやApple Push Notification Serviceを使用
        logger.info(f"Push notification sent to device {device.device_id}: {notification.title}")
        notification.sent_at = datetime.now()

    def get_device_notifications(self, device_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """デバイス通知取得"""
        device_notifications = [
            notif for notif in self.notifications
            if notif.device_id == device_id
        ]

        # 時間順にソート
        device_notifications.sort(key=lambda n: n.created_at, reverse=True)

        return [asdict(notif) for notif in device_notifications[:limit]]

    def mark_notification_read(self, device_id: str, notification_id: str) -> bool:
        """通知既読マーク"""
        for notification in self.notifications:
            if (notification.device_id == device_id and
                notification.notification_id == notification_id):
                notification.read = True
                return True
        return False

class MobileAppService:
    """モバイルアプリサービス"""

    def __init__(self, production_system: ProductionManagementSystem, port: int = 8091):
        self.production_system = production_system
        self.port = port

        # サービスコンポーネント
        self.cache = MobileAppCache()
        self.notification_service = MobileNotificationService()

        # FastAPIアプリケーション
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available. Mobile app service disabled.")

        # サーバースレッド
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        # 定期タスク
        self.cleanup_thread: Optional[threading.Thread] = None

    def _create_fastapi_app(self) -> FastAPI:
        """FastAPIアプリケーション作成"""
        app = FastAPI(
            title="Mobile App Service",
            description="Mobile monitoring and control application",
            version="1.0.0"
        )

        # CORSミドルウェア
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # モバイルアプリからのアクセスを許可
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 静的ファイル
        app.mount("/static", StaticFiles(directory="mobile_static"), name="static")

        # APIルート定義
        self._setup_api_routes(app)

        return app

    def _setup_api_routes(self, app: FastAPI):
        """APIルート設定"""

        @app.get("/", response_class=HTMLResponse)
        async def get_mobile_app():
            """モバイルアプリHTML"""
            return self._get_mobile_app_html()

        @app.get("/api/mobile/health")
        async def mobile_health_check():
            """モバイルアプリヘルスチェック"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "mobile_app",
                "version": "1.0.0",
                "features": {
                    "offline_mode": True,
                    "push_notifications": self.notification_service.push_service_available,
                    "cache_enabled": True
                }
            }

        @app.post("/api/mobile/register")
        async def register_device(request: Request):
            """デバイス登録"""
            device_data = await request.json()

            # デバイス情報取得
            device_id = device_data.get("device_id") or str(uuid.uuid4())
            user_agent = request.headers.get("user-agent", "")

            # デバイスタイプ判定（簡易）
            device_type = self._detect_device_type(user_agent)

            # デバイス登録
            device = self.notification_service.register_device(
                device_id=device_id,
                device_type=device_type,
                user_agent=user_agent,
                push_token=device_data.get("push_token")
            )

            return {
                "device_id": device.device_id,
                "device_type": device.device_type.value,
                "registered_at": device.last_seen.isoformat()
            }

        @app.get("/api/mobile/dashboard")
        async def get_mobile_dashboard(request: Request, device_id: str = Query(...)):
            """モバイルダッシュボードデータ取得"""
            # デバイスアクティビティ更新
            self.notification_service.update_device_activity(device_id)

            # キャッシュ確認
            cache_key = f"dashboard_{device_id}"
            cached_data = self.cache.get_cache(cache_key)

            if cached_data:
                return cached_data

            # データ取得
            dashboard_data = self.production_system.get_production_dashboard()

            # モバイル用に最適化
            mobile_dashboard = self._optimize_for_mobile(dashboard_data)

            # キャッシュ保存
            self.cache.set_cache(cache_key, mobile_dashboard, ttl_seconds=60)

            return mobile_dashboard

        @app.get("/api/mobile/metrics")
        async def get_mobile_metrics(request: Request, device_id: str = Query(...)):
            """モバイル用メトリクス取得"""
            self.notification_service.update_device_activity(device_id)

            # 簡略化メトリクス
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "active_robots": 3,
                    "total_orders_today": 15,
                    "completion_rate": 0.87,
                    "average_oee": 0.82
                },
                "alerts": [
                    {
                        "id": "alert_001",
                        "type": "warning",
                        "message": "Robot 001 OEE below threshold",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }

        @app.get("/api/mobile/notifications")
        async def get_notifications(device_id: str = Query(...), limit: int = Query(20, le=50)):
            """通知取得"""
            self.notification_service.update_device_activity(device_id)
            return self.notification_service.get_device_notifications(device_id, limit)

        @app.post("/api/mobile/notifications/{notification_id}/read")
        async def mark_notification_read(notification_id: str, device_id: str = Query(...)):
            """通知既読"""
            success = self.notification_service.mark_notification_read(device_id, notification_id)
            return {"success": success}

        @app.post("/api/mobile/offline-action")
        async def store_offline_action(request: Request):
            """オフラインアクション保存"""
            action_data = await request.json()

            action_id = self.cache.store_offline_action(
                device_id=action_data["device_id"],
                action_type=action_data["action_type"],
                action_data=action_data["data"]
            )

            return {"action_id": action_id, "status": "stored"}

        @app.get("/api/mobile/sync-pending")
        async def sync_pending_actions(device_id: str = Query(...)):
            """保留中のアクション同期"""
            self.notification_service.update_device_activity(device_id)
            pending_actions = self.cache.get_pending_actions(device_id)

            return {
                "pending_actions": pending_actions,
                "count": len(pending_actions)
            }

        @app.post("/api/mobile/sync-complete/{action_id}")
        async def mark_action_synced(action_id: str):
            """アクション同期完了"""
            self.cache.mark_action_synced(action_id)
            return {"success": True}

        @app.get("/api/mobile/production-summary")
        async def get_production_summary(request: Request, device_id: str = Query(...)):
            """生産サマリー取得"""
            self.notification_service.update_device_activity(device_id)

            # 現在のサマリー
            dashboard = self.production_system.get_production_dashboard()

            return {
                "today": {
                    "orders_completed": dashboard["today_summary"]["completed_orders"],
                    "orders_total": dashboard["today_summary"]["total_orders"],
                    "quantity_produced": dashboard["today_summary"]["total_quantity"],
                    "average_oee": dashboard["today_summary"]["avg_oee"],
                    "average_quality": dashboard["today_summary"]["avg_quality"]
                },
                "shift": {
                    "start_time": "08:00",
                    "end_time": "16:00",
                    "progress": 0.75
                }
            }

        @app.get("/manifest.json")
        async def get_pwa_manifest():
            """PWAマニフェスト"""
            return {
                "name": "Robot Production Monitor",
                "short_name": "RoboMonitor",
                "description": "Mobile robot production monitoring app",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#ffffff",
                "theme_color": "#2c3e50",
                "icons": [
                    {
                        "src": "/static/icon-192.png",
                        "sizes": "192x192",
                        "type": "image/png"
                    },
                    {
                        "src": "/static/icon-512.png",
                        "sizes": "512x512",
                        "type": "image/png"
                    }
                ]
            }

        @app.get("/service-worker.js")
        async def get_service_worker():
            """サービスワーカー"""
            return Response(
                content=self._get_service_worker_js(),
                media_type="application/javascript"
            )

    def _detect_device_type(self, user_agent: str) -> MobileDeviceType:
        """デバイスタイプ検出"""
        user_agent_lower = user_agent.lower()

        if "mobile" in user_agent_lower or "android" in user_agent_lower or "iphone" in user_agent_lower:
            return MobileDeviceType.SMARTPHONE
        elif "tablet" in user_agent_lower or "ipad" in user_agent_lower:
            return MobileDeviceType.TABLET
        else:
            return MobileDeviceType.DESKTOP

    def _optimize_for_mobile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """モバイル用データ最適化"""
        return {
            "timestamp": data["timestamp"],
            "active_robots": data["active_production_orders"],
            "today_summary": {
                "completed_orders": data["today_summary"]["completed_orders"],
                "total_orders": data["today_summary"]["total_orders"],
                "completion_rate": data["today_summary"]["completion_rate"],
                "average_oee": data["today_summary"]["avg_oee"]
            },
            "alerts_count": 1,  # 簡略化
            "system_status": "running"
        }

    def _get_mobile_app_html(self) -> str:
        """モバイルアプリHTML"""
        return """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="default">
            <meta name="theme-color" content="#2c3e50">

            <title>Robot Production Monitor</title>

            <!-- PWAマニフェスト -->
            <link rel="manifest" href="/manifest.json">

            <!-- iOSアイコン -->
            <link rel="apple-touch-icon" href="/static/icon-192.png">

            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f5f7fa;
                    color: #2c3e50;
                    line-height: 1.6;
                }

                .app-container {
                    max-width: 100%;
                    min-height: 100vh;
                    background: #ffffff;
                }

                .header {
                    background: linear-gradient(135deg, #2c3e50, #3498db);
                    color: white;
                    padding: 15px 20px;
                    position: sticky;
                    top: 0;
                    z-index: 100;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }

                .header h1 {
                    font-size: 1.5em;
                    font-weight: 600;
                    margin-bottom: 5px;
                }

                .header .status {
                    font-size: 0.9em;
                    opacity: 0.9;
                }

                .status-indicator {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: #27ae60;
                    border-radius: 50%;
                    margin-right: 5px;
                }

                .main-content {
                    padding: 20px 15px;
                }

                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-bottom: 20px;
                }

                .metric-card {
                    background: white;
                    border-radius: 12px;
                    padding: 20px 15px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    text-align: center;
                    border: 1px solid #e1e8ed;
                }

                .metric-value {
                    font-size: 2em;
                    font-weight: 700;
                    color: #2c3e50;
                    margin-bottom: 5px;
                }

                .metric-label {
                    font-size: 0.85em;
                    color: #7f8c8d;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .metric-change {
                    font-size: 0.8em;
                    margin-top: 5px;
                }

                .metric-change.positive {
                    color: #27ae60;
                }

                .metric-change.negative {
                    color: #e74c3c;
                }

                .alerts-section {
                    background: white;
                    border-radius: 12px;
                    padding: 20px 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }

                .section-title {
                    font-size: 1.2em;
                    font-weight: 600;
                    margin-bottom: 15px;
                    color: #2c3e50;
                }

                .alert-item {
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    border-left: 4px solid;
                    background: #f8f9fa;
                }

                .alert-item.warning {
                    border-left-color: #f39c12;
                    background: #fef9e7;
                }

                .alert-item.error {
                    border-left-color: #e74c3c;
                    background: #fdedec;
                }

                .alert-item.info {
                    border-left-color: #3498db;
                    background: #e3f2fd;
                }

                .alert-title {
                    font-weight: 600;
                    margin-bottom: 4px;
                }

                .alert-time {
                    font-size: 0.8em;
                    color: #7f8c8d;
                }

                .production-section {
                    background: white;
                    border-radius: 12px;
                    padding: 20px 15px;
            margin-bottom: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }

                .progress-bar {
                    width: 100%;
                    height: 8px;
                    background: #ecf0f1;
                    border-radius: 4px;
                    overflow: hidden;
                    margin: 10px 0;
                }

                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #27ae60, #2ecc71);
                    transition: width 0.3s ease;
                }

                .stats-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #ecf0f1;
                }

                .stats-row:last-child {
                    border-bottom: none;
                }

                .stat-label {
                    color: #7f8c8d;
                    font-size: 0.9em;
                }

                .stat-value {
                    font-weight: 600;
                    color: #2c3e50;
                }

                .refresh-button {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 56px;
                    height: 56px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    border: none;
                    box-shadow: 0 4px 16px rgba(52, 152, 219, 0.3);
                    cursor: pointer;
                    font-size: 1.5em;
                    z-index: 1000;
                    transition: transform 0.2s ease;
                }

                .refresh-button:active {
                    transform: scale(0.95);
                }

                .loading {
                    text-align: center;
                    padding: 40px;
                    color: #7f8c8d;
                }

                .offline-indicator {
                    background: #e74c3c;
                    color: white;
                    padding: 8px 15px;
                    text-align: center;
                    font-size: 0.9em;
                }

                .online-indicator {
                    background: #27ae60;
                    color: white;
                    padding: 8px 15px;
                    text-align: center;
                    font-size: 0.9em;
                }

                /* レスポンシブデザイン */
                @media (min-width: 768px) {
                    .metrics-grid {
                        grid-template-columns: repeat(4, 1fr);
                        gap: 20px;
                    }

                    .main-content {
                        padding: 30px;
                        max-width: 800px;
                        margin: 0 auto;
                    }
                }

                /* ダークモード対応 */
                @media (prefers-color-scheme: dark) {
                    body {
                        background: #1a1a1a;
                        color: #ecf0f1;
                    }

                    .metric-card,
                    .alerts-section,
                    .production-section {
                        background: #2c3e50;
                        border-color: #34495e;
                    }

                    .alert-item {
                        background: #34495e;
                    }
                }

                /* タッチ対応 */
                .metric-card:active,
                .alert-item:active {
                    transform: scale(0.98);
                    transition: transform 0.1s ease;
                }
            </style>
        </head>
        <body>
            <div class="app-container">
                <div class="header">
                    <h1>🤖 Robot Monitor</h1>
                    <div class="status">
                        <span class="status-indicator"></span>
                        <span id="connection-status">Connecting...</span>
                    </div>
                </div>

                <div id="offline-indicator" class="offline-indicator" style="display: none;">
                    ⚠️ Offline Mode - Limited functionality
                </div>

                <div class="main-content">
                    <div id="loading" class="loading">
                        <div>Loading production data...</div>
                    </div>

                    <div id="content" style="display: none;">
                        <!-- メトリクスグリッド -->
                        <div class="metrics-grid" id="metrics-grid">
                            <!-- メトリクスカードが動的に表示される -->
                        </div>

                        <!-- 生産概要 -->
                        <div class="production-section">
                            <div class="section-title">📊 Production Summary</div>
                            <div class="stats-row">
                                <span class="stat-label">Today's Progress</span>
                                <span class="stat-value" id="completion-rate">0%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                            </div>
                            <div class="stats-row">
                                <span class="stat-label">Orders Completed</span>
                                <span class="stat-value" id="orders-completed">0/0</span>
                            </div>
                            <div class="stats-row">
                                <span class="stat-label">Average OEE</span>
                                <span class="stat-value" id="average-oee">0%</span>
                            </div>
                            <div class="stats-row">
                                <span class="stat-label">Quality Rate</span>
                                <span class="stat-value" id="quality-rate">0%</span>
                            </div>
                        </div>

                        <!-- アラートセクション -->
                        <div class="alerts-section">
                            <div class="section-title">🚨 Recent Alerts</div>
                            <div id="alerts-list">
                                <!-- アラートが動的に表示される -->
                            </div>
                        </div>
                    </div>
                </div>

                <button class="refresh-button" onclick="refreshData()" id="refresh-btn">
                    🔄
                </button>
            </div>

            <script>
                // グローバル変数
                let deviceId = localStorage.getItem('device_id') || generateDeviceId();
                let isOnline = navigator.onLine;
                let lastSyncTime = null;

                // デバイスID生成
                function generateDeviceId() {
                    return 'mobile_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                }

                // アプリ初期化
                async function initializeApp() {
                    try {
                        // デバイス登録
                        await registerDevice();

                        // オフライン/オンライン状態監視
                        setupConnectivityListeners();

                        // データ読み込み
                        await loadDashboardData();

                        // 定期更新
                        setInterval(refreshData, 30000); // 30秒ごとに更新

                    } catch (error) {
                        console.error('App initialization failed:', error);
                        showError('Failed to initialize app');
                    }
                }

                // デバイス登録
                async function registerDevice() {
                    try {
                        const response = await fetch('/api/mobile/register', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                device_id: deviceId,
                                push_token: null // TODO: プッシュトークン実装
                            })
                        });

                        if (response.ok) {
                            const data = await response.json();
                            deviceId = data.device_id;
                            localStorage.setItem('device_id', deviceId);
                            console.log('Device registered:', deviceId);
                        }
                    } catch (error) {
                        console.error('Device registration failed:', error);
                    }
                }

                // 接続状態リスナー設定
                function setupConnectivityListeners() {
                    window.addEventListener('online', () => {
                        isOnline = true;
                        updateConnectionStatus();
                        syncOfflineActions();
                    });

                    window.addEventListener('offline', () => {
                        isOnline = false;
                        updateConnectionStatus();
                    });

                    updateConnectionStatus();
                }

                // 接続状態更新
                function updateConnectionStatus() {
                    const statusElement = document.getElementById('connection-status');
                    const offlineIndicator = document.getElementById('offline-indicator');

                    if (isOnline) {
                        statusElement.textContent = 'Connected';
                        offlineIndicator.style.display = 'none';
                    } else {
                        statusElement.textContent = 'Offline';
                        offlineIndicator.style.display = 'block';
                    }
                }

                // ダッシュボードデータ読み込み
                async function loadDashboardData() {
                    try {
                        if (isOnline) {
                            const response = await fetch(`/api/mobile/dashboard?device_id=${deviceId}`);

                            if (response.ok) {
                                const data = await response.json();
                                updateDashboard(data);
                                lastSyncTime = new Date();
                                hideLoading();
                            }
                        } else {
                            // オフライン時はキャッシュから読み込み
                            loadCachedData();
                        }
                    } catch (error) {
                        console.error('Failed to load dashboard data:', error);
                        if (isOnline) {
                            showError('Failed to load data');
                        }
                    }
                }

                // ダッシュボード更新
                function updateDashboard(data) {
                    // メトリクス更新
                    updateMetrics(data);

                    // 生産サマリー更新
                    updateProductionSummary(data);

                    // アラート更新
                    updateAlerts(data);

                    // データをキャッシュ
                    cacheData(data);
                }

                // メトリクス更新
                function updateMetrics(data) {
                    const metricsGrid = document.getElementById('metrics-grid');

                    const metrics = [
                        {
                            label: 'Active Robots',
                            value: data.active_robots || 0,
                            change: '+1',
                            changeType: 'positive'
                        },
                        {
                            label: 'OEE',
                            value: ((data.today_summary?.average_oee || 0) * 100).toFixed(1) + '%',
                            change: '+2.3%',
                            changeType: 'positive'
                        },
                        {
                            label: 'Quality',
                            value: ((data.today_summary?.average_quality || 0) * 100).toFixed(1) + '%',
                            change: '-0.5%',
                            changeType: 'negative'
                        },
                        {
                            label: 'Efficiency',
                            value: '87%',
                            change: '+1.2%',
                            changeType: 'positive'
                        }
                    ];

                    metricsGrid.innerHTML = metrics.map(metric => `
                        <div class="metric-card">
                            <div class="metric-value">${metric.value}</div>
                            <div class="metric-label">${metric.label}</div>
                            <div class="metric-change ${metric.changeType}">
                                ${metric.change}
                            </div>
                        </div>
                    `).join('');
                }

                // 生産サマリー更新
                function updateProductionSummary(data) {
                    const completionRate = data.today_summary?.completion_rate || 0;
                    const completedOrders = data.today_summary?.completed_orders || 0;
                    const totalOrders = data.today_summary?.total_orders || 0;
                    const avgOee = (data.today_summary?.average_oee || 0) * 100;
                    const avgQuality = (data.today_summary?.average_quality || 0) * 100;

                    document.getElementById('completion-rate').textContent = (completionRate * 100).toFixed(1) + '%';
                    document.getElementById('progress-fill').style.width = (completionRate * 100) + '%';
                    document.getElementById('orders-completed').textContent = `${completedOrders}/${totalOrders}`;
                    document.getElementById('average-oee').textContent = avgOee.toFixed(1) + '%';
                    document.getElementById('quality-rate').textContent = avgQuality.toFixed(1) + '%';
                }

                // アラート更新
                function updateAlerts(data) {
                    const alertsList = document.getElementById('alerts-list');

                    // 模擬アラートデータ
                    const alerts = [
                        {
                            type: 'warning',
                            title: 'Robot 001 OEE Below Threshold',
                            time: '2 minutes ago'
                        },
                        {
                            type: 'info',
                            title: 'Shift Change in 30 minutes',
                            time: '15 minutes ago'
                        }
                    ];

                    if (alerts.length === 0) {
                        alertsList.innerHTML = '<div style="text-align: center; color: #27ae60;">✅ No active alerts</div>';
                    } else {
                        alertsList.innerHTML = alerts.map(alert => `
                            <div class="alert-item ${alert.type}">
                                <div class="alert-title">${alert.title}</div>
                                <div class="alert-time">${alert.time}</div>
                            </div>
                        `).join('');
                    }
                }

                // データキャッシュ
                function cacheData(data) {
                    try {
                        localStorage.setItem('dashboard_cache', JSON.stringify({
                            data: data,
                            timestamp: new Date().toISOString()
                        }));
                    } catch (error) {
                        console.error('Failed to cache data:', error);
                    }
                }

                // キャッシュデータ読み込み
                function loadCachedData() {
                    try {
                        const cached = localStorage.getItem('dashboard_cache');
                        if (cached) {
                            const { data, timestamp } = JSON.parse(cached);
                            const cacheAge = new Date() - new Date(timestamp);

                            // 5分以内のキャッシュを使用
                            if (cacheAge < 5 * 60 * 1000) {
                                updateDashboard(data);
                                hideLoading();
                                return;
                            }
                        }
                    } catch (error) {
                        console.error('Failed to load cached data:', error);
                    }

                    hideLoading();
                    showError('No cached data available');
                }

                // データ更新
                async function refreshData() {
                    const refreshBtn = document.getElementById('refresh-btn');

                    // リフレッシュボタンアニメーション
                    refreshBtn.style.transform = 'rotate(360deg)';
                    setTimeout(() => {
                        refreshBtn.style.transform = 'rotate(0deg)';
                    }, 500);

                    await loadDashboardData();
                }

                // オフラインアクション同期
                async function syncOfflineActions() {
                    try {
                        // 保留中のアクションを同期
                        const response = await fetch(`/api/mobile/sync-pending?device_id=${deviceId}`);
                        if (response.ok) {
                            const data = await response.json();

                            // 各アクションを処理
                            for (const action of data.pending_actions) {
                                await processOfflineAction(action);
                            }
                        }
                    } catch (error) {
                        console.error('Failed to sync offline actions:', error);
                    }
                }

                // オフラインアクション処理
                async function processOfflineAction(action) {
                    try {
                        // アクションタイプに応じて処理
                        switch (action.action_type) {
                            case 'acknowledge_alert':
                                // アラート確認処理
                                break;
                            case 'update_settings':
                                // 設定更新処理
                                break;
                        }

                        // 同期完了を通知
                        await fetch(`/api/mobile/sync-complete/${action.action_id}`, {
                            method: 'POST'
                        });

                    } catch (error) {
                        console.error('Failed to process offline action:', error);
                    }
                }

                // ローディング非表示
                function hideLoading() {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('content').style.display = 'block';
                }

                // エラー表示
                function showError(message) {
                    const content = document.getElementById('content');
                    content.innerHTML = `
                        <div style="text-align: center; padding: 40px; color: #e74c3c;">
                            <div style="font-size: 3em; margin-bottom: 20px;">⚠️</div>
                            <div>${message}</div>
                            <button onclick="location.reload()" style="
                                margin-top: 20px;
                                padding: 12px 24px;
                                background: #3498db;
                                color: white;
                                border: none;
                                border-radius: 8px;
                                cursor: pointer;
                            ">Retry</button>
                        </div>
                    `;
                }

                // アプリ起動
                document.addEventListener('DOMContentLoaded', initializeApp);

                // PWAインストールプロンプト（省略）
                let deferredPrompt;
                window.addEventListener('beforeinstallprompt', (e) => {
                    e.preventDefault();
                    deferredPrompt = e;
                });
            </script>
        </body>
        </html>
        """

    def _get_service_worker_js(self) -> str:
        """サービスワーカーJavaScript"""
        return """
        const CACHE_NAME = 'robot-monitor-v1';
const urlsToCache = [
    '/',
    '/static/icon-192.png',
    '/static/icon-512.png'
];

// インストール
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

// フェッチ
self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // キャッシュがあれば返す
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});

// プッシュ通知（省略）
self.addEventListener('push', (event) => {
    const options = {
        body: event.data.text(),
        icon: '/static/icon-192.png',
        badge: '/static/icon-192.png'
    };

    event.waitUntil(
        self.registration.showNotification('Robot Monitor', options)
    );
});
        """

    def start(self) -> bool:
        """サービス開始"""
        try:
            if not FASTAPI_AVAILABLE:
                logger.error("FastAPI not available. Cannot start mobile app service.")
                return False

            if self.running:
                logger.warning("Mobile app service already running")
                return False

            self.running = True

            # クリーンアップスレッド開始
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()

            # FastAPIサーバースレッド開始
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()

            logger.info(f"Mobile app service started on port {self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start mobile app service: {e}")
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
            logger.error(f"Mobile app server error: {e}")

    def _cleanup_loop(self):
        """クリーンアップループ"""
        while self.running:
            try:
                self.cache.cleanup_expired_cache()
                time.sleep(3600)  # 1時間ごとにクリーンアップ
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(300)  # エラー時は5分後に再試行

    def stop(self):
        """サービス停止"""
        self.running = False

        logger.info("Mobile app service stopped")

    def get_service_status(self) -> Dict[str, Any]:
        """サービス状態取得"""
        return {
            "running": self.running,
            "port": self.port,
            "registered_devices": len(self.notification_service.devices),
            "active_devices": len([d for d in self.notification_service.devices.values() if d.is_active]),
            "total_notifications": len(self.notification_service.notifications),
            "service_url": f"http://localhost:{self.port}",
            "features": {
                "pwa_support": True,
                "offline_mode": True,
                "push_notifications": self.notification_service.push_service_available,
                "device_detection": True
            }
        }

# グローバルインスタンス
mobile_app_service: Optional[MobileAppService] = None

def initialize_mobile_app(production_system: ProductionManagementSystem, port: int = 8091) -> MobileAppService:
    """モバイルアプリサービス初期化"""
    global mobile_app_service
    mobile_app_service = MobileAppService(production_system, port)
    return mobile_app_service

def get_mobile_app_service() -> Optional[MobileAppService]:
    """モバイルアプリサービス取得"""
    return mobile_app_service

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Mobile App Service...")

    try:
        # モック生産管理システム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())

        # モバイルアプリサービス初期化
        service = initialize_mobile_app(mock_pms)

        if service.start():
            print(f"Mobile app service started successfully!")
            print(f"Mobile app URL: http://localhost:{service.port}")
            print("Features: PWA support, Offline mode, Push notifications")
            print("Press Ctrl+C to stop...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                service.stop()

        else:
            print("Failed to start mobile app service")

    except Exception as e:
        print(f"Mobile app service test failed: {e}")
