"""
生産管理システム連携 (Phase 3)
MES (Manufacturing Execution System) および ERP との連携
リアルタイム生産データ管理・分析・報告システム
"""

import time
import json
import threading
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor

from basic_handling_workflow import Position, WorkPiece
from multi_robot_coordination import CoordinationTask, RobotInfo, RobotState
from config_manager import config_manager

logger = logging.getLogger(__name__)

class ProductionStatus(Enum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

class WorkOrderStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

class QualityStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    REWORK = "rework"
    PENDING = "pending"

@dataclass
class ProductionOrder:
    """生産オーダー"""
    order_id: str
    product_id: str
    product_name: str
    quantity: int
    priority: int
    due_date: datetime
    created_date: datetime = field(default_factory=datetime.now)
    status: ProductionStatus = ProductionStatus.PLANNED
    progress: float = 0.0  # 0.0-1.0
    assigned_resources: List[str] = field(default_factory=list)  # ロボットID等
    estimated_duration: float = 0.0  # 分
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    specifications: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

@dataclass
class WorkOrder:
    """作業オーダー"""
    work_order_id: str
    production_order_id: str
    operation_id: str
    operation_name: str
    sequence: int
    robot_id: Optional[str] = None
    workstation_id: Optional[str] = None
    status: WorkOrderStatus = WorkOrderStatus.PENDING
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    standard_time: float = 0.0  # 標準時間（分）
    actual_time: float = 0.0    # 実績時間（分）
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProductionMetrics:
    """生産メトリクス"""
    timestamp: datetime
    robot_id: str
    production_order_id: str
    work_order_id: str
    cycle_time: float  # サイクル時間（秒）
    throughput: float  # スループット（個/時）
    quality_score: float  # 品質スコア（0.0-1.0）
    oee: float  # Overall Equipment Effectiveness
    availability: float  # 稼働率
    performance: float   # 性能率
    quality_rate: float  # 品質率
    energy_consumption: float = 0.0  # エネルギー消費（kWh）
    maintenance_alerts: int = 0  # メンテナンスアラート数

@dataclass
class QualityRecord:
    """品質記録"""
    record_id: str
    production_order_id: str
    work_order_id: str
    product_id: str
    inspection_time: datetime
    inspector: str  # ロボットIDまたは検査員ID
    status: QualityStatus
    measurements: Dict[str, float] = field(default_factory=dict)
    defects: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    notes: str = ""

class MESConnector(ABC):
    """MES接続基底クラス"""
    
    @abstractmethod
    def send_production_data(self, data: Dict[str, Any]) -> bool:
        """生産データ送信"""
        pass
    
    @abstractmethod
    def receive_work_orders(self) -> List[WorkOrder]:
        """作業オーダー受信"""
        pass
    
    @abstractmethod
    def update_work_order_status(self, work_order_id: str, status: WorkOrderStatus) -> bool:
        """作業オーダー状態更新"""
        pass

class RESTMESConnector(MESConnector):
    """REST API MES接続"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def send_production_data(self, data: Dict[str, Any]) -> bool:
        try:
            response = self.session.post(
                f"{self.base_url}/api/production-data",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.debug(f"Production data sent successfully: {data.get('order_id', 'unknown')}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send production data: {e}")
            return False
    
    def receive_work_orders(self) -> List[WorkOrder]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/work-orders/pending",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            work_orders = []
            for order_data in response.json():
                work_order = WorkOrder(
                    work_order_id=order_data["work_order_id"],
                    production_order_id=order_data["production_order_id"],
                    operation_id=order_data["operation_id"],
                    operation_name=order_data["operation_name"],
                    sequence=order_data["sequence"],
                    scheduled_start=datetime.fromisoformat(order_data["scheduled_start"]) if order_data.get("scheduled_start") else None,
                    scheduled_end=datetime.fromisoformat(order_data["scheduled_end"]) if order_data.get("scheduled_end") else None,
                    standard_time=order_data.get("standard_time", 0.0),
                    parameters=order_data.get("parameters", {})
                )
                work_orders.append(work_order)
            
            logger.info(f"Received {len(work_orders)} work orders from MES")
            return work_orders
            
        except requests.RequestException as e:
            logger.error(f"Failed to receive work orders: {e}")
            return []
    
    def update_work_order_status(self, work_order_id: str, status: WorkOrderStatus) -> bool:
        try:
            data = {
                "work_order_id": work_order_id,
                "status": status.value,
                "timestamp": datetime.now().isoformat()
            }
            
            response = self.session.put(
                f"{self.base_url}/api/work-orders/{work_order_id}/status",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.debug(f"Work order status updated: {work_order_id} -> {status.value}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update work order status: {e}")
            return False

class MockMESConnector(MESConnector):
    """モック MES接続（テスト用）"""
    
    def __init__(self):
        self.sent_data: List[Dict[str, Any]] = []
        self.work_orders: List[WorkOrder] = []
        self.order_statuses: Dict[str, WorkOrderStatus] = {}
        
        # テスト用作業オーダー生成
        self._generate_test_work_orders()
    
    def send_production_data(self, data: Dict[str, Any]) -> bool:
        self.sent_data.append(data.copy())
        logger.info(f"Mock MES received production data: {data.get('order_id', 'unknown')}")
        return True
    
    def receive_work_orders(self) -> List[WorkOrder]:
        # 未処理の作業オーダーを返す
        pending_orders = [wo for wo in self.work_orders if wo.status == WorkOrderStatus.PENDING]
        return pending_orders
    
    def update_work_order_status(self, work_order_id: str, status: WorkOrderStatus) -> bool:
        self.order_statuses[work_order_id] = status
        
        # 作業オーダーの状態更新
        for wo in self.work_orders:
            if wo.work_order_id == work_order_id:
                wo.status = status
                break
        
        logger.info(f"Mock MES updated work order: {work_order_id} -> {status.value}")
        return True
    
    def _generate_test_work_orders(self):
        """テスト用作業オーダー生成"""
        base_time = datetime.now()
        
        for i in range(5):
            work_order = WorkOrder(
                work_order_id=f"WO_{i+1:03d}",
                production_order_id=f"PO_001",
                operation_id=f"OP_{i+1:02d}",
                operation_name=f"Assembly Operation {i+1}",
                sequence=i+1,
                scheduled_start=base_time + timedelta(minutes=i*30),
                scheduled_end=base_time + timedelta(minutes=i*30+20),
                standard_time=20.0,
                parameters={"part_type": "component_A", "quality_level": "high"}
            )
            self.work_orders.append(work_order)

class ProductionDatabase:
    """生産データベース"""
    
    def __init__(self, db_path: str = "production.db"):
        self.db_path = db_path
        self.connection_lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS production_orders (
                    order_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    due_date TEXT NOT NULL,
                    created_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.0,
                    assigned_resources TEXT,
                    estimated_duration REAL DEFAULT 0.0,
                    actual_start_time TEXT,
                    actual_end_time TEXT,
                    specifications TEXT,
                    notes TEXT
                );
                
                CREATE TABLE IF NOT EXISTS work_orders (
                    work_order_id TEXT PRIMARY KEY,
                    production_order_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    operation_name TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    robot_id TEXT,
                    workstation_id TEXT,
                    status TEXT NOT NULL,
                    scheduled_start TEXT,
                    actual_start TEXT,
                    scheduled_end TEXT,
                    actual_end TEXT,
                    standard_time REAL DEFAULT 0.0,
                    actual_time REAL DEFAULT 0.0,
                    parameters TEXT,
                    quality_data TEXT,
                    FOREIGN KEY (production_order_id) REFERENCES production_orders (order_id)
                );
                
                CREATE TABLE IF NOT EXISTS production_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    production_order_id TEXT NOT NULL,
                    work_order_id TEXT NOT NULL,
                    cycle_time REAL NOT NULL,
                    throughput REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    oee REAL NOT NULL,
                    availability REAL NOT NULL,
                    performance REAL NOT NULL,
                    quality_rate REAL NOT NULL,
                    energy_consumption REAL DEFAULT 0.0,
                    maintenance_alerts INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS quality_records (
                    record_id TEXT PRIMARY KEY,
                    production_order_id TEXT NOT NULL,
                    work_order_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    inspection_time TEXT NOT NULL,
                    inspector TEXT NOT NULL,
                    status TEXT NOT NULL,
                    measurements TEXT,
                    defects TEXT,
                    images TEXT,
                    notes TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_production_orders_status ON production_orders (status);
                CREATE INDEX IF NOT EXISTS idx_work_orders_status ON work_orders (status);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON production_metrics (timestamp);
                CREATE INDEX IF NOT EXISTS idx_quality_inspection_time ON quality_records (inspection_time);
            """)
        
        logger.info("Production database initialized")
    
    def save_production_order(self, order: ProductionOrder):
        """生産オーダー保存"""
        with self.connection_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO production_orders 
                    (order_id, product_id, product_name, quantity, priority, due_date, 
                     created_date, status, progress, assigned_resources, estimated_duration, 
                     actual_start_time, actual_end_time, specifications, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.order_id, order.product_id, order.product_name, order.quantity,
                    order.priority, order.due_date.isoformat(), order.created_date.isoformat(),
                    order.status.value, order.progress, json.dumps(order.assigned_resources),
                    order.estimated_duration,
                    order.actual_start_time.isoformat() if order.actual_start_time else None,
                    order.actual_end_time.isoformat() if order.actual_end_time else None,
                    json.dumps(order.specifications), order.notes
                ))
    
    def save_work_order(self, work_order: WorkOrder):
        """作業オーダー保存"""
        with self.connection_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO work_orders 
                    (work_order_id, production_order_id, operation_id, operation_name, sequence,
                     robot_id, workstation_id, status, scheduled_start, actual_start,
                     scheduled_end, actual_end, standard_time, actual_time, parameters, quality_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    work_order.work_order_id, work_order.production_order_id, work_order.operation_id,
                    work_order.operation_name, work_order.sequence, work_order.robot_id,
                    work_order.workstation_id, work_order.status.value,
                    work_order.scheduled_start.isoformat() if work_order.scheduled_start else None,
                    work_order.actual_start.isoformat() if work_order.actual_start else None,
                    work_order.scheduled_end.isoformat() if work_order.scheduled_end else None,
                    work_order.actual_end.isoformat() if work_order.actual_end else None,
                    work_order.standard_time, work_order.actual_time,
                    json.dumps(work_order.parameters), json.dumps(work_order.quality_data)
                ))
    
    def save_production_metrics(self, metrics: ProductionMetrics):
        """生産メトリクス保存"""
        with self.connection_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO production_metrics 
                    (timestamp, robot_id, production_order_id, work_order_id, cycle_time,
                     throughput, quality_score, oee, availability, performance, quality_rate,
                     energy_consumption, maintenance_alerts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(), metrics.robot_id, metrics.production_order_id,
                    metrics.work_order_id, metrics.cycle_time, metrics.throughput,
                    metrics.quality_score, metrics.oee, metrics.availability, metrics.performance,
                    metrics.quality_rate, metrics.energy_consumption, metrics.maintenance_alerts
                ))
    
    def save_quality_record(self, record: QualityRecord):
        """品質記録保存"""
        with self.connection_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO quality_records 
                    (record_id, production_order_id, work_order_id, product_id, inspection_time,
                     inspector, status, measurements, defects, images, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id, record.production_order_id, record.work_order_id,
                    record.product_id, record.inspection_time.isoformat(), record.inspector,
                    record.status.value, json.dumps(record.measurements),
                    json.dumps(record.defects), json.dumps(record.images), record.notes
                ))
    
    def get_production_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """生産サマリー取得"""
        with self.connection_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 基本統計
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_orders,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_orders,
                        AVG(progress) as average_progress,
                        SUM(quantity) as total_quantity
                    FROM production_orders 
                    WHERE created_date BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                
                basic_stats = cursor.fetchone()
                
                # メトリクス統計
                cursor.execute("""
                    SELECT 
                        AVG(cycle_time) as avg_cycle_time,
                        AVG(throughput) as avg_throughput,
                        AVG(oee) as avg_oee,
                        AVG(quality_score) as avg_quality
                    FROM production_metrics 
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                
                metrics_stats = cursor.fetchone()
                
                return {
                    "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                    "total_orders": basic_stats[0] if basic_stats else 0,
                    "completed_orders": basic_stats[1] if basic_stats else 0,
                    "completion_rate": basic_stats[1] / basic_stats[0] if basic_stats and basic_stats[0] > 0 else 0,
                    "average_progress": basic_stats[2] if basic_stats else 0,
                    "total_quantity": basic_stats[3] if basic_stats else 0,
                    "avg_cycle_time": metrics_stats[0] if metrics_stats else 0,
                    "avg_throughput": metrics_stats[1] if metrics_stats else 0,
                    "avg_oee": metrics_stats[2] if metrics_stats else 0,
                    "avg_quality": metrics_stats[3] if metrics_stats else 0
                }

class ProductionManagementSystem:
    """生産管理システムメイン"""
    
    def __init__(self, mes_connector: MESConnector = None, database: ProductionDatabase = None):
        self.mes_connector = mes_connector or MockMESConnector()
        self.database = database or ProductionDatabase()
        
        # 内部状態
        self.active_production_orders: Dict[str, ProductionOrder] = {}
        self.active_work_orders: Dict[str, WorkOrder] = {}
        self.robot_assignments: Dict[str, List[str]] = {}  # robot_id -> work_order_ids
        
        # スレッド制御
        self.running = False
        self.mes_sync_thread: Optional[threading.Thread] = None
        self.metrics_collection_thread: Optional[threading.Thread] = None
        
        # データ収集間隔
        self.mes_sync_interval = 60.0  # 60秒間隔でMES同期
        self.metrics_interval = 30.0   # 30秒間隔でメトリクス収集
        
        # コールバック
        self.on_work_order_received: Optional[Callable[[WorkOrder], None]] = None
        self.on_production_completed: Optional[Callable[[str], None]] = None
        self.on_quality_alert: Optional[Callable[[QualityRecord], None]] = None
        
        # パフォーマンス追跡
        self.session_start_time = datetime.now()
        self.total_processed_orders = 0
        
    def start(self) -> bool:
        """システム開始"""
        try:
            if self.running:
                logger.warning("Production management system already running")
                return False
            
            self.running = True
            
            # MES同期スレッド開始
            self.mes_sync_thread = threading.Thread(target=self._mes_sync_loop, daemon=True)
            self.mes_sync_thread.start()
            
            # メトリクス収集スレッド開始
            self.metrics_collection_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
            self.metrics_collection_thread.start()
            
            logger.info("Production management system started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start production management system: {e}")
            return False
    
    def stop(self):
        """システム停止"""
        self.running = False
        
        if self.mes_sync_thread and self.mes_sync_thread.is_alive():
            self.mes_sync_thread.join(timeout=2.0)
        
        if self.metrics_collection_thread and self.metrics_collection_thread.is_alive():
            self.metrics_collection_thread.join(timeout=2.0)
        
        logger.info("Production management system stopped")
    
    def create_production_order(self, product_id: str, product_name: str, 
                              quantity: int, due_date: datetime,
                              priority: int = 5, specifications: Dict[str, Any] = None) -> str:
        """生産オーダー作成"""
        order_id = f"PO_{uuid.uuid4().hex[:8].upper()}"
        
        order = ProductionOrder(
            order_id=order_id,
            product_id=product_id,
            product_name=product_name,
            quantity=quantity,
            priority=priority,
            due_date=due_date,
            specifications=specifications or {}
        )
        
        self.active_production_orders[order_id] = order
        self.database.save_production_order(order)
        
        logger.info(f"Production order created: {order_id}")
        return order_id
    
    def assign_work_order_to_robot(self, work_order_id: str, robot_id: str) -> bool:
        """作業オーダーをロボットに割り当て"""
        if work_order_id not in self.active_work_orders:
            logger.error(f"Work order not found: {work_order_id}")
            return False
        
        work_order = self.active_work_orders[work_order_id]
        work_order.robot_id = robot_id
        work_order.status = WorkOrderStatus.IN_PROGRESS
        work_order.actual_start = datetime.now()
        
        # ロボット割り当て追跡
        if robot_id not in self.robot_assignments:
            self.robot_assignments[robot_id] = []
        self.robot_assignments[robot_id].append(work_order_id)
        
        # データベース更新
        self.database.save_work_order(work_order)
        
        # MES更新
        self.mes_connector.update_work_order_status(work_order_id, WorkOrderStatus.IN_PROGRESS)
        
        logger.info(f"Work order {work_order_id} assigned to robot {robot_id}")
        return True
    
    def complete_work_order(self, work_order_id: str, actual_time: float = None,
                          quality_data: Dict[str, Any] = None) -> bool:
        """作業オーダー完了"""
        if work_order_id not in self.active_work_orders:
            logger.error(f"Work order not found: {work_order_id}")
            return False
        
        work_order = self.active_work_orders[work_order_id]
        work_order.status = WorkOrderStatus.COMPLETED
        work_order.actual_end = datetime.now()
        
        if actual_time is not None:
            work_order.actual_time = actual_time
        elif work_order.actual_start:
            work_order.actual_time = (work_order.actual_end - work_order.actual_start).total_seconds() / 60.0
        
        if quality_data:
            work_order.quality_data = quality_data
        
        # ロボット割り当て解除
        if work_order.robot_id and work_order.robot_id in self.robot_assignments:
            if work_order_id in self.robot_assignments[work_order.robot_id]:
                self.robot_assignments[work_order.robot_id].remove(work_order_id)
        
        # 生産オーダー進捗更新
        self._update_production_order_progress(work_order.production_order_id)
        
        # データベース更新
        self.database.save_work_order(work_order)
        
        # MES更新
        self.mes_connector.update_work_order_status(work_order_id, WorkOrderStatus.COMPLETED)
        
        # メトリクス記録
        self._record_work_order_metrics(work_order)
        
        self.total_processed_orders += 1
        logger.info(f"Work order completed: {work_order_id}")
        return True
    
    def record_quality_inspection(self, production_order_id: str, work_order_id: str,
                                product_id: str, inspector: str, status: QualityStatus,
                                measurements: Dict[str, float] = None,
                                defects: List[str] = None) -> str:
        """品質検査記録"""
        record_id = f"QR_{uuid.uuid4().hex[:8].upper()}"
        
        record = QualityRecord(
            record_id=record_id,
            production_order_id=production_order_id,
            work_order_id=work_order_id,
            product_id=product_id,
            inspection_time=datetime.now(),
            inspector=inspector,
            status=status,
            measurements=measurements or {},
            defects=defects or []
        )
        
        self.database.save_quality_record(record)
        
        # 品質アラート
        if status == QualityStatus.FAIL and self.on_quality_alert:
            self.on_quality_alert(record)
        
        logger.info(f"Quality inspection recorded: {record_id} - {status.value}")
        return record_id
    
    def get_production_dashboard(self) -> Dict[str, Any]:
        """生産ダッシュボードデータ取得"""
        current_time = datetime.now()
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 本日のサマリー
        today_summary = self.database.get_production_summary(today_start, current_time)
        
        # リアルタイム状態
        active_orders = len(self.active_production_orders)
        active_work_orders = len([wo for wo in self.active_work_orders.values() 
                                 if wo.status == WorkOrderStatus.IN_PROGRESS])
        
        # ロボット状態（模擬）
        robot_utilization = {}
        for robot_id, work_orders in self.robot_assignments.items():
            robot_utilization[robot_id] = len(work_orders)
        
        return {
            "timestamp": current_time.isoformat(),
            "session_uptime": (current_time - self.session_start_time).total_seconds(),
            "active_production_orders": active_orders,
            "active_work_orders": active_work_orders,
            "total_processed_orders": self.total_processed_orders,
            "today_summary": today_summary,
            "robot_utilization": robot_utilization
        }
    
    def _mes_sync_loop(self):
        """MES同期ループ"""
        logger.info("MES sync loop started")
        
        while self.running:
            try:
                # 新規作業オーダー取得
                new_work_orders = self.mes_connector.receive_work_orders()
                
                for work_order in new_work_orders:
                    if work_order.work_order_id not in self.active_work_orders:
                        self.active_work_orders[work_order.work_order_id] = work_order
                        self.database.save_work_order(work_order)
                        
                        logger.info(f"Received new work order from MES: {work_order.work_order_id}")
                        
                        if self.on_work_order_received:
                            self.on_work_order_received(work_order)
                
                # 生産データ送信
                self._send_production_data_to_mes()
                
                time.sleep(self.mes_sync_interval)
                
            except Exception as e:
                logger.error(f"MES sync error: {e}")
                time.sleep(10.0)  # エラー時は短い間隔で再試行
        
        logger.info("MES sync loop ended")
    
    def _metrics_collection_loop(self):
        """メトリクス収集ループ"""
        logger.info("Metrics collection loop started")
        
        while self.running:
            try:
                self._collect_robot_metrics()
                time.sleep(self.metrics_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(10.0)
        
        logger.info("Metrics collection loop ended")
    
    def _update_production_order_progress(self, production_order_id: str):
        """生産オーダー進捗更新"""
        if production_order_id not in self.active_production_orders:
            return
        
        production_order = self.active_production_orders[production_order_id]
        
        # 関連作業オーダー取得
        related_work_orders = [
            wo for wo in self.active_work_orders.values()
            if wo.production_order_id == production_order_id
        ]
        
        if not related_work_orders:
            return
        
        # 進捗計算
        completed_orders = sum(1 for wo in related_work_orders if wo.status == WorkOrderStatus.COMPLETED)
        total_orders = len(related_work_orders)
        
        production_order.progress = completed_orders / total_orders
        
        # 完了チェック
        if production_order.progress >= 1.0:
            production_order.status = ProductionStatus.COMPLETED
            production_order.actual_end_time = datetime.now()
            
            if self.on_production_completed:
                self.on_production_completed(production_order_id)
        
        self.database.save_production_order(production_order)
    
    def _record_work_order_metrics(self, work_order: WorkOrder):
        """作業オーダーメトリクス記録"""
        if not work_order.robot_id or not work_order.actual_start or not work_order.actual_end:
            return
        
        # 基本メトリクス計算
        cycle_time = work_order.actual_time * 60  # 分を秒に変換
        
        # 模擬メトリクス（実際の実装では各ロボットから取得）
        throughput = 60.0 / cycle_time if cycle_time > 0 else 0.0  # 個/時
        quality_score = 0.95  # 95%品質スコア（模擬）
        availability = 0.90   # 90%稼働率（模擬）
        performance = min(1.0, work_order.standard_time / work_order.actual_time if work_order.actual_time > 0 else 0.0)
        quality_rate = quality_score
        oee = availability * performance * quality_rate
        
        metrics = ProductionMetrics(
            timestamp=work_order.actual_end,
            robot_id=work_order.robot_id,
            production_order_id=work_order.production_order_id,
            work_order_id=work_order.work_order_id,
            cycle_time=cycle_time,
            throughput=throughput,
            quality_score=quality_score,
            oee=oee,
            availability=availability,
            performance=performance,
            quality_rate=quality_rate
        )
        
        self.database.save_production_metrics(metrics)
    
    def _send_production_data_to_mes(self):
        """生産データをMESに送信"""
        # 完了した生産オーダーのデータ送信
        for order_id, order in self.active_production_orders.items():
            if order.status == ProductionStatus.COMPLETED:
                production_data = {
                    "order_id": order.order_id,
                    "product_id": order.product_id,
                    "quantity": order.quantity,
                    "status": order.status.value,
                    "progress": order.progress,
                    "actual_start_time": order.actual_start_time.isoformat() if order.actual_start_time else None,
                    "actual_end_time": order.actual_end_time.isoformat() if order.actual_end_time else None,
                    "timestamp": datetime.now().isoformat()
                }
                
                if self.mes_connector.send_production_data(production_data):
                    # 送信成功したオーダーを削除
                    del self.active_production_orders[order_id]
    
    def _collect_robot_metrics(self):
        """ロボットメトリクス収集"""
        # 実際の実装では各ロボットからメトリクスを取得
        # ここでは模擬データを生成
        current_time = datetime.now()
        
        for robot_id, work_order_ids in self.robot_assignments.items():
            if work_order_ids:  # アクティブな作業がある場合のみ
                # 模擬メトリクス生成
                metrics = ProductionMetrics(
                    timestamp=current_time,
                    robot_id=robot_id,
                    production_order_id="PO_CURRENT",
                    work_order_id=work_order_ids[0] if work_order_ids else "WO_UNKNOWN",
                    cycle_time=45.0 + (hash(robot_id) % 20) - 10,  # 35-55秒の範囲
                    throughput=60.0,
                    quality_score=0.95,
                    oee=0.80,
                    availability=0.90,
                    performance=0.92,
                    quality_rate=0.95
                )
                
                self.database.save_production_metrics(metrics)

# グローバルインスタンス
production_system: Optional[ProductionManagementSystem] = None

def initialize_production_system(mes_connector: MESConnector = None) -> ProductionManagementSystem:
    """生産管理システム初期化"""
    global production_system
    production_system = ProductionManagementSystem(mes_connector)
    return production_system

def get_production_system() -> Optional[ProductionManagementSystem]:
    """生産管理システム取得"""
    return production_system

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    
    # モックMES接続を使用
    mock_mes = MockMESConnector()
    
    # 生産管理システム初期化
    pms = initialize_production_system(mock_mes)
    
    if pms.start():
        logger.info("Production management system started successfully")
        
        # コールバック設定
        def on_work_order_received(work_order: WorkOrder):
            logger.info(f"New work order received: {work_order.work_order_id}")
            # 自動的にロボットに割り当て（テスト）
            pms.assign_work_order_to_robot(work_order.work_order_id, "robot_001")
        
        def on_production_completed(order_id: str):
            logger.info(f"Production order completed: {order_id}")
        
        pms.on_work_order_received = on_work_order_received
        pms.on_production_completed = on_production_completed
        
        try:
            # テスト生産オーダー作成
            due_date = datetime.now() + timedelta(hours=8)
            order_id = pms.create_production_order(
                product_id="PROD_001",
                product_name="Test Assembly",
                quantity=10,
                due_date=due_date,
                priority=5
            )
            
            # テスト実行
            time.sleep(5.0)
            
            # いくつかの作業オーダーを完了
            for wo_id in list(pms.active_work_orders.keys())[:2]:
                pms.complete_work_order(wo_id, actual_time=25.5)
                
                # 品質検査記録
                pms.record_quality_inspection(
                    production_order_id=order_id,
                    work_order_id=wo_id,
                    product_id="PROD_001",
                    inspector="robot_001",
                    status=QualityStatus.PASS,
                    measurements={"dimension_x": 100.2, "dimension_y": 50.1}
                )
            
            time.sleep(3.0)
            
            # ダッシュボードデータ取得
            dashboard = pms.get_production_dashboard()
            logger.info(f"Production dashboard: {json.dumps(dashboard, indent=2)}")
            
        finally:
            pms.stop()
    
    else:
        logger.error("Failed to start production management system")