"""
エッジコンピューティングノード (Phase 5-4)
ローカルデータ処理・低遅延制御・分散コンピューティング
リアルタイム処理・オフライン機能・エッジAI推論
"""

import json
import time
import logging
import threading
import asyncio
import queue
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import pickle
import hashlib
import os
import signal
import subprocess
from pathlib import Path
from collections import defaultdict, deque

# 機械学習推論
try:
    import onnxruntime as ort
    import tensorflow as tf
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

# ネットワーキング
import socket
import http.server
import socketserver
from urllib.parse import urlparse
import requests

from production_management_integration import ProductionManagementSystem
from digital_twin_integration import DigitalTwinCore

logger = logging.getLogger(__name__)

class EdgeNodeType(Enum):
    """エッジノードタイプ"""
    GATEWAY = "gateway"  # ゲートウェイノード
    COMPUTING = "computing"  # コンピューティングノード
    SENSOR = "sensor"  # セサーノード
    ACTUATOR = "actuator"  # アクチュエータノード
    STORAGE = "storage"  # ストレージノード

class ProcessingMode(Enum):
    """処理モード"""
    REAL_TIME = "real_time"  # リアルタイム処理
    BATCH = "batch"  # バッチ処理
    STREAMING = "streaming"  # ストリーミング処理
    OFFLINE = "offline"  # オフライン処理

class TaskPriority(Enum):
    """タスク優先度"""
    CRITICAL = 1  # 臨界
    HIGH = 2  # 高
    NORMAL = 3  # 通常
    LOW = 4  # 低
    BACKGROUND = 5  # バックグラウンド

@dataclass
class EdgeNodeInfo:
    """エッジノード情報"""
    node_id: str
    node_type: EdgeNodeType
    hostname: str
    ip_address: str
    port: int
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    capabilities: List[str]
    location: str
    last_heartbeat: datetime
    status: str = "active"

@dataclass
class ProcessingTask:
    """処理タスク"""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Any
    processing_function: str
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class DataPipeline:
    """データパイプライン"""
    pipeline_id: str
    name: str
    input_source: str
    output_destination: str
    processing_steps: List[Dict[str, Any]]
    batch_size: int
    processing_interval: float  # 秒
    enabled: bool = True
    last_run: Optional[datetime] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeModel:
    """エッジAIモデル"""
    model_id: str
    model_name: str
    model_type: str  # classification, regression, detection, forecasting
    model_path: str
    input_shape: List[int]
    output_shape: List[int]
    preprocessing: Dict[str, Any]
    postprocessing: Dict[str, Any]
    inference_time: float = 0.0
    accuracy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class TaskScheduler:
    """タスクスケジューラ"""

    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = queue.PriorityQueue()
        self.running_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: List[ProcessingTask] = []
        self.task_functions: Dict[str, Callable] = {}
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

    def register_function(self, name: str, function: Callable):
        """処理関数登録"""
        self.task_functions[name] = function
        logger.info(f"Registered task function: {name}")

    def submit_task(self, task_type: str, data: Any, priority: TaskPriority = TaskPriority.NORMAL,
                   parameters: Dict[str, Any] = None) -> str:
        """タスク送信"""
        task = ProcessingTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            data=data,
            processing_function=task_type,
            parameters=parameters or {},
            created_at=datetime.now()
        )

        # 優先度キューに追加（優先度が小さいほど高優先度）
        self.task_queue.put((priority.value, task))
        logger.info(f"Submitted task: {task.task_id} ({task_type})")
        return task.task_id

    def start_scheduler(self) -> bool:
        """スケジューラ起動"""
        try:
            if self.is_running:
                logger.warning("Task scheduler already running")
                return False

            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()

            logger.info("Task scheduler started")
            return True

        except Exception as e:
            logger.error(f"Failed to start task scheduler: {e}")
            return False

    def stop_scheduler(self):
        """スケジューラ停止"""
        self.is_running = False

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)

        logger.info("Task scheduler stopped")

    def _scheduler_loop(self):
        """スケジューラループ"""
        logger.info("Task scheduler loop started")

        while self.is_running:
            try:
                # 実行中タスク数をチェック
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    # 新規タスク取得
                    try:
                        priority, task = self.task_queue.get(timeout=1.0)
                        self._execute_task(task)
                    except queue.Empty:
                        continue

                # 完了したタスクをクリーンアップ
                self._cleanup_completed_tasks()

                time.sleep(0.1)  # 100ms

            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                time.sleep(1.0)

        logger.info("Task scheduler loop ended")

    def _execute_task(self, task: ProcessingTask):
        """タスク実行"""
        try:
            if task.processing_function not in self.task_functions:
                task.status = "failed"
                task.error = f"Function not found: {task.processing_function}"
                self.completed_tasks.append(task)
                return

            task.status = "running"
            task.started_at = datetime.now()
            self.running_tasks[task.task_id] = task

            # 別スレッドで実行
            execution_thread = threading.Thread(
                target=self._run_task_function,
                args=(task,),
                daemon=True
            )
            execution_thread.start()

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.completed_tasks.append(task)
            logger.error(f"Task execution setup failed: {e}")

    def _run_task_function(self, task: ProcessingTask):
        """タスク関数実行"""
        try:
            function = self.task_functions[task.processing_function]

            # タイムアウト付き実行
            timeout = task.parameters.get('timeout', 300)  # 5分デフォルト
            result = function(task.data, **task.parameters)

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()

        except Exception as e:
            task.error = str(e)
            task.status = "failed"

            # リトライ処理
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                task.error = f"Retry {task.retry_count}/{task.max_retries}: {str(e)}"
                self.task_queue.put((task.priority.value, task))
                logger.warning(f"Task retry scheduled: {task.task_id}")

        finally:
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks.append(task)

    def _cleanup_completed_tasks(self):
        """完了タスククリーンアップ"""
        # 最新1000件を保持
        if len(self.completed_tasks) > 1000:
            self.completed_tasks = self.completed_tasks[-1000:]

    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """タスク状態取得"""
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]

        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        total_tasks = len(self.completed_tasks)
        completed = len([t for t in self.completed_tasks if t.status == "completed"])
        failed = len([t for t in self.completed_tasks if t.status == "failed"])

        return {
            "running_tasks": len(self.running_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "total_completed": total_tasks,
            "success_rate": completed / total_tasks if total_tasks > 0 else 0,
            "average_execution_time": self._calculate_avg_execution_time()
        }

    def _calculate_avg_execution_time(self) -> float:
        """平均実行時間計算"""
        completed_tasks = [t for t in self.completed_tasks if t.status == "completed" and t.started_at and t.completed_at]

        if not completed_tasks:
            return 0.0

        total_time = sum((t.completed_at - t.started_at).total_seconds() for t in completed_tasks)
        return total_time / len(completed_tasks)

class EdgeInferenceEngine:
    """エッジ推論エンジン"""

    def __init__(self):
        self.models: Dict[str, EdgeModel] = {}
        self.inference_sessions: Dict[str, Any] = {}
        self.model_cache = {}
        self.inference_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

    def load_model(self, model: EdgeModel) -> bool:
        """モデル読み込み"""
        try:
            if not INFERENCE_AVAILABLE:
                logger.warning("Inference libraries not available")
                return False

            # ONNXモデル読み込み
            if model.model_path.endswith('.onnx'):
                session = ort.InferenceSession(model.model_path)
                self.inference_sessions[model.model_id] = session

            # TensorFlowモデル読み込み
            elif model.model_path.endswith(('.h5', '.pb')):
                tf_model = tf.keras.models.load_model(model.model_path)
                self.inference_sessions[model.model_id] = tf_model

            self.models[model.model_id] = model
            logger.info(f"Loaded edge model: {model.model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model.model_id}: {e}")
            return False

    def predict(self, model_id: str, input_data: Any) -> Optional[Any]:
        """推論実行"""
        try:
            if model_id not in self.models:
                logger.error(f"Model not found: {model_id}")
                return None

            if model_id not in self.inference_sessions:
                logger.error(f"Inference session not found: {model_id}")
                return None

            start_time = time.time()
            model = self.models[model_id]
            session = self.inference_sessions[model_id]

            # 前処理
            processed_input = self._preprocess_input(input_data, model.preprocessing)

            # 推論実行
            if isinstance(session, ort.InferenceSession):
                # ONNX推論
                input_name = session.get_inputs()[0].name
                result = session.run(None, {input_name: processed_input})
            else:
                # TensorFlow推論
                result = session.predict(processed_input)

            # 後処理
            processed_output = self._postprocess_output(result, model.postprocessing)

            # 統計更新
            inference_time = time.time() - start_time
            self.inference_stats[model_id]["count"] += 1
            self.inference_stats[model_id]["total_time"] += inference_time

            return processed_output

        except Exception as e:
            logger.error(f"Inference failed for model {model_id}: {e}")
            return None

    def _preprocess_input(self, input_data: Any, preprocessing: Dict[str, Any]) -> Any:
        """入力前処理"""
        if not preprocessing:
            return input_data

        processed = input_data

        # 正規化
        if "normalize" in preprocessing:
            norm_params = preprocessing["normalize"]
            if isinstance(processed, np.ndarray):
                processed = (processed - norm_params["mean"]) / norm_params["std"]

        # リサイズ
        if "resize" in preprocessing:
            resize_params = preprocessing["resize"]
            if isinstance(processed, np.ndarray):
                # 実際のリサイズ処理
                pass

        return processed

    def _postprocess_output(self, output: Any, postprocessing: Dict[str, Any]) -> Any:
        """出力後処理"""
        if not postprocessing:
            return output

        processed = output

        # シグモイド活性化
        if "sigmoid" in postprocessing and postprocessing["sigmoid"]:
            if isinstance(processed, (list, tuple)) and len(processed) > 0:
                processed = 1 / (1 + np.exp(-processed[0]))

        # ソフトマックス
        if "softmax" in postprocessing and postprocessing["softmax"]:
            if isinstance(processed, np.ndarray):
                exp_values = np.exp(processed - np.max(processed))
                processed = exp_values / np.sum(exp_values)

        return processed

    def get_model_statistics(self) -> Dict[str, Any]:
        """モデル統計取得"""
        return {
            "loaded_models": len(self.models),
            "models": {
                model_id: {
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "inference_count": self.inference_stats[model_id]["count"],
                    "average_inference_time": (
                        self.inference_stats[model_id]["total_time"] /
                        self.inference_stats[model_id]["count"]
                        if self.inference_stats[model_id]["count"] > 0 else 0.0
                    )
                }
                for model_id, model in self.models.items()
            }
        }

class DataBuffer:
    """データバッファ"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_data(self, data: Any, timestamp: datetime = None):
        """データ追加"""
        with self.lock:
            self.buffer.append({
                "data": data,
                "timestamp": timestamp or datetime.now()
            })

    def get_latest(self, count: int = 1) -> List[Any]:
        """最新データ取得"""
        with self.lock:
            return [item["data"] for item in list(self.buffer)[-count:]]

    def get_data_since(self, since_time: datetime) -> List[Any]:
        """指定時刻以降のデータ取得"""
        with self.lock:
            return [item["data"] for item in self.buffer if item["timestamp"] >= since_time]

    def clear(self):
        """バッファクリア"""
        with self.lock:
            self.buffer.clear()

    def size(self) -> int:
        """バッファサイズ取得"""
        return len(self.buffer)

class LocalStorage:
    """ローカルストレージ"""

    def __init__(self, storage_path: str = "edge_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.db_path = self.storage_path / "edge_data.db"

        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    processed BOOLEAN DEFAULT FALSE
                );

                CREATE TABLE IF NOT EXISTS processing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    result TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processing_time REAL
                );

                CREATE TABLE IF NOT EXISTS model_cache (
                    model_id TEXT PRIMARY KEY,
                    model_data TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_data(timestamp);
                CREATE INDEX IF NOT EXISTS idx_sensor_id ON sensor_data(sensor_id);
            """)

    def store_sensor_data(self, sensor_id: str, data: Any, timestamp: datetime = None):
        """センサーデータ保存"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO sensor_data (sensor_id, timestamp, data)
                VALUES (?, ?, ?)
            """, (
                sensor_id,
                (timestamp or datetime.now()).isoformat(),
                json.dumps(data, default=str)
            ))

    def get_sensor_data(self, sensor_id: str, start_time: datetime = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
        """センサーデータ取得"""
        with sqlite3.connect(str(self.db_path)) as conn:
            query = "SELECT * FROM sensor_data WHERE sensor_id = ?"
            params = [sensor_id]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "sensor_id": row[1],
                    "timestamp": datetime.fromisoformat(row[2]),
                    "data": json.loads(row[3]),
                    "processed": bool(row[4])
                }
                for row in rows
            ]

    def store_processing_result(self, task_id: str, result: Any, processing_time: float):
        """処理結果保存"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO processing_results (task_id, result, timestamp, processing_time)
                VALUES (?, ?, ?, ?)
            """, (
                task_id,
                json.dumps(result, default=str),
                datetime.now().isoformat(),
                processing_time
            ))

    def cleanup_old_data(self, retention_days: int = 7):
        """古いデータクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        with sqlite3.connect(str(self.db_path)) as conn:
            # センサーデータ削除
            conn.execute("""
                DELETE FROM sensor_data WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))

            # 処理結果削除
            conn.execute("""
                DELETE FROM processing_results WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))

        logger.info(f"Cleaned up data older than {retention_days} days")

class EdgeComputingNode:
    """エッジコンピューティングノード"""

    def __init__(self, node_id: str, node_type: EdgeNodeType, port: int = 8093):
        self.node_id = node_id
        self.node_type = node_type
        self.port = port

        # コンポーネント
        self.task_scheduler = TaskScheduler()
        self.inference_engine = EdgeInferenceEngine()
        self.local_storage = LocalStorage(f"edge_storage_{node_id}")
        self.data_buffers: Dict[str, DataBuffer] = {}

        # ノード情報
        self.node_info = self._get_node_info()
        self.connected_nodes: Dict[str, EdgeNodeInfo] = {}
        self.data_pipelines: Dict[str, DataPipeline] = {}

        # 実行制御
        self.running = False
        self.main_thread: Optional[threading.Thread] = None

        # 統計
        self.statistics = {
            "tasks_processed": 0,
            "data_processed": 0,
            "uptime": 0.0,
            "start_time": datetime.now()
        }

        # コールバック
        self.on_data_processed: Optional[Callable[[str, Any], None]] = None
        self.on_node_connected: Optional[Callable[[EdgeNodeInfo], None]] = None

        # 処理関数登録
        self._register_processing_functions()

    def _get_node_info(self) -> EdgeNodeInfo:
        """ノード情報取得"""
        import psutil

        return EdgeNodeInfo(
            node_id=self.node_id,
            node_type=self.node_type,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=self.port,
            cpu_cores=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            storage_gb=psutil.disk_usage('/').total / (1024**3),
            capabilities=self._detect_capabilities(),
            location="local",
            last_heartbeat=datetime.now()
        )

    def _get_local_ip(self) -> str:
        """ローカルIP取得"""
        try:
            # 外部に接続してローカルIPを取得
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _detect_capabilities(self) -> List[str]:
        """機能検出"""
        capabilities = ["basic_processing"]

        if INFERENCE_AVAILABLE:
            capabilities.append("ai_inference")

        try:
            import cv2
            capabilities.append("computer_vision")
        except ImportError:
            pass

        try:
            import tensorflow
            capabilities.append("deep_learning")
        except ImportError:
            pass

        return capabilities

    def _register_processing_functions(self):
        """処理関数登録"""
        self.task_scheduler.register_function("data_validation", self._validate_data)
        self.task_scheduler.register_function("data_transformation", self._transform_data)
        self.task_scheduler.register_function("anomaly_detection", self._detect_anomalies)
        self.task_scheduler.register_function("data_aggregation", self._aggregate_data)
        self.task_scheduler.register_function("ai_inference", self._run_ai_inference)

    def start_node(self) -> bool:
        """ノード起動"""
        try:
            if self.running:
                logger.warning("Edge node already running")
                return False

            self.running = True
            self.statistics["start_time"] = datetime.now()

            # メインループスレッド開始
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()

            # タスクスケジューラ起動
            self.task_scheduler.start_scheduler()

            logger.info(f"Edge computing node started: {self.node_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start edge node {self.node_id}: {e}")
            return False

    def stop_node(self):
        """ノード停止"""
        self.running = False

        # タスクスケジューラ停止
        self.task_scheduler.stop_scheduler()

        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)

        logger.info(f"Edge computing node stopped: {self.node_id}")

    def _main_loop(self):
        """メインループ"""
        logger.info(f"Edge node main loop started: {self.node_id}")

        while self.running:
            try:
                # データパイプライン処理
                self._process_data_pipelines()

                # 統計更新
                self._update_statistics()

                # 接続ノードヘルスチェック
                self._check_connected_nodes()

                time.sleep(1.0)  # 1秒間隔

            except Exception as e:
                logger.error(f"Edge node main loop error: {e}")
                time.sleep(5.0)

        logger.info(f"Edge node main loop ended: {self.node_id}")

    def _process_data_pipelines(self):
        """データパイプライン処理"""
        current_time = datetime.now()

        for pipeline_id, pipeline in self.data_pipelines.items():
            if not pipeline.enabled:
                continue

            # 処理間隔チェック
            if (pipeline.last_run and
                (current_time - pipeline.last_run).total_seconds() < pipeline.processing_interval):
                continue

            # データ取得
            input_data = self._get_pipeline_input(pipeline.input_source)
            if not input_data:
                continue

            # タスク送信
            task_id = self.task_scheduler.submit_task(
                "data_transformation",
                {
                    "pipeline_id": pipeline_id,
                    "data": input_data,
                    "steps": pipeline.processing_steps
                },
                TaskPriority.NORMAL,
                {"batch_size": pipeline.batch_size}
            )

            pipeline.last_run = current_time

    def _get_pipeline_input(self, source: str) -> Any:
        """パイプライン入力取得"""
        if source.startswith("sensor:"):
            sensor_id = source.split(":")[1]
            if sensor_id in self.data_buffers:
                return self.data_buffers[sensor_id].get_latest(pipeline.batch_size)

        elif source.startswith("storage:"):
            # ストレージからデータ取得
            pass

        return None

    def _update_statistics(self):
        """統計情報更新"""
        if self.running:
            self.statistics["uptime"] = (datetime.now() - self.statistics["start_time"]).total_seconds()

        scheduler_stats = self.task_scheduler.get_statistics()
        self.statistics["tasks_processed"] = scheduler_stats["total_completed"]

    def _check_connected_nodes(self):
        """接続ノードヘルスチェック"""
        current_time = datetime.now()
        timeout = timedelta(seconds=30)

        inactive_nodes = [
            node_id for node_id, node_info in self.connected_nodes.items()
            if current_time - node_info.last_heartbeat > timeout
        ]

        for node_id in inactive_nodes:
            del self.connected_nodes[node_id]
            logger.info(f"Removed inactive node: {node_id}")

    def add_sensor_data(self, sensor_id: str, data: Any, timestamp: datetime = None):
        """センサーデータ追加"""
        # データバッファに追加
        if sensor_id not in self.data_buffers:
            self.data_buffers[sensor_id] = DataBuffer()

        self.data_buffers[sensor_id].add_data(data, timestamp)

        # ローカルストレージに保存
        self.local_storage.store_sensor_data(sensor_id, data, timestamp)

        self.statistics["data_processed"] += 1

        # コールバック実行
        if self.on_data_processed:
            self.on_data_processed(sensor_id, data)

    def submit_processing_task(self, task_type: str, data: Any,
                             priority: TaskPriority = TaskPriority.NORMAL,
                             parameters: Dict[str, Any] = None) -> str:
        """処理タスク送信"""
        return self.task_scheduler.submit_task(task_type, data, priority, parameters)

    def create_data_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """データパイプライン作成"""
        pipeline = DataPipeline(
            pipeline_id=str(uuid.uuid4()),
            name=pipeline_config["name"],
            input_source=pipeline_config["input_source"],
            output_destination=pipeline_config["output_destination"],
            processing_steps=pipeline_config["processing_steps"],
            batch_size=pipeline_config.get("batch_size", 100),
            processing_interval=pipeline_config.get("processing_interval", 5.0)
        )

        self.data_pipelines[pipeline.pipeline_id] = pipeline
        logger.info(f"Created data pipeline: {pipeline.pipeline_id}")
        return pipeline.pipeline_id

    def connect_to_node(self, node_info: EdgeNodeInfo) -> bool:
        """ノードに接続"""
        try:
            # 実際の接続処理
            self.connected_nodes[node_info.node_id] = node_info
            logger.info(f"Connected to edge node: {node_info.node_id}")

            if self.on_node_connected:
                self.on_node_connected(node_info)

            return True

        except Exception as e:
            logger.error(f"Failed to connect to node {node_info.node_id}: {e}")
            return False

    def get_node_status(self) -> Dict[str, Any]:
        """ノード状態取得"""
        return {
            "node_info": asdict(self.node_info),
            "running": self.running,
            "statistics": self.statistics.copy(),
            "task_scheduler": self.task_scheduler.get_statistics(),
            "inference_engine": self.inference_engine.get_model_statistics(),
            "connected_nodes": len(self.connected_nodes),
            "data_pipelines": len(self.data_pipelines),
            "data_buffers": {
                sensor_id: {"size": buffer.size(), "max_size": buffer.max_size}
                for sensor_id, buffer in self.data_buffers.items()
            }
        }

    # 処理関数実装
    def _validate_data(self, data: Any, **kwargs) -> Dict[str, Any]:
        """データ検証"""
        try:
            if isinstance(data, (list, dict)):
                return {
                    "valid": True,
                    "data_type": type(data).__name__,
                    "size": len(data) if hasattr(data, '__len__') else 0
                }
            else:
                return {
                    "valid": True,
                    "data_type": type(data).__name__,
                    "value": data
                }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _transform_data(self, data: Any, **kwargs) -> Any:
        """データ変換"""
        try:
            pipeline_id = kwargs.get("pipeline_id")
            steps = kwargs.get("steps", [])

            transformed_data = data

            for step in steps:
                step_type = step.get("type")
                if step_type == "filter":
                    # フィルタ処理
                    pass
                elif step_type == "aggregate":
                    # 集約処理
                    pass
                elif step_type == "normalize":
                    # 正規化処理
                    pass

            return transformed_data

        except Exception as e:
            logger.error(f"Data transformation error: {e}")
            return data

    def _detect_anomalies(self, data: Any, **kwargs) -> Dict[str, Any]:
        """異常検知"""
        try:
            # 簡単な異常検知（統計的異常検知）
            if isinstance(data, list) and len(data) > 0:
                values = [float(item) for item in data if isinstance(item, (int, float))]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    anomalies = []
                    for i, value in enumerate(values):
                        z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                        if z_score > 3:  # 3σルール
                            anomalies.append({"index": i, "value": value, "z_score": z_score})

                    return {
                        "anomalies_detected": len(anomalies) > 0,
                        "anomaly_count": len(anomalies),
                        "anomalies": anomalies,
                        "statistics": {"mean": mean_val, "std": std_val}
                    }

            return {"anomalies_detected": False, "anomaly_count": 0}

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {"anomalies_detected": False, "error": str(e)}

    def _aggregate_data(self, data: Any, **kwargs) -> Dict[str, Any]:
        """データ集約"""
        try:
            if isinstance(data, list) and len(data) > 0:
                values = [float(item) for item in data if isinstance(item, (int, float))]
                if values:
                    return {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "sum": np.sum(values)
                    }

            return {"count": 0, "error": "No numeric data to aggregate"}

        except Exception as e:
            logger.error(f"Data aggregation error: {e}")
            return {"count": 0, "error": str(e)}

    def _run_ai_inference(self, data: Any, **kwargs) -> Any:
        """AI推論実行"""
        model_id = kwargs.get("model_id")
        if not model_id:
            return {"error": "Model ID not provided"}

        return self.inference_engine.predict(model_id, data)

# グローバルインスタンス
edge_nodes: Dict[str, EdgeComputingNode] = {}

def create_edge_node(node_id: str, node_type: EdgeNodeType, port: int = None) -> EdgeComputingNode:
    """エッジノード作成"""
    if port is None:
        # ポート番号自動割り当て
        port = 8093 + len(edge_nodes)

    node = EdgeComputingNode(node_id, node_type, port)
    edge_nodes[node_id] = node
    return node

def get_edge_node(node_id: str) -> Optional[EdgeComputingNode]:
    """エッジノード取得"""
    return edge_nodes.get(node_id)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Edge Computing Node System...")

    try:
        # ゲートウェイノード作成
        gateway_node = create_edge_node("gateway_001", EdgeNodeType.GATEWAY)

        # コンピューティングノード作成
        compute_node = create_edge_node("compute_001", EdgeNodeType.COMPUTING)

        # ノード起動
        if gateway_node.start_node() and compute_node.start_node():
            print("Edge computing nodes started successfully!")

            # センサーデータ追加
            for i in range(10):
                sensor_data = {
                    "temperature": 20 + np.random.normal(0, 2),
                    "pressure": 100 + np.random.normal(0, 5),
                    "timestamp": time.time()
                }
                gateway_node.add_sensor_data("temp_sensor_001", sensor_data)

            # 処理タスク送信
            task_id = gateway_node.submit_processing_task(
                "anomaly_detection",
                [25, 23, 24, 26, 22, 30, 15, 20, 21, 19]
            )

            # データパイプライン作成
            pipeline_config = {
                "name": "Temperature Monitoring Pipeline",
                "input_source": "sensor:temp_sensor_001",
                "output_destination": "local_storage",
                "processing_steps": [
                    {"type": "filter", "field": "temperature"},
                    {"type": "aggregate", "function": "mean"}
                ],
                "batch_size": 5,
                "processing_interval": 2.0
            }

            pipeline_id = gateway_node.create_data_pipeline(pipeline_config)
            print(f"Created data pipeline: {pipeline_id}")

            # テスト実行
            time.sleep(3)

            # タスク状態確認
            task_status = gateway_node.task_scheduler.get_task_status(task_id)
            if task_status:
                print(f"Task status: {task_status.status}")

            # ノード状態確認
            gateway_status = gateway_node.get_node_status()
            print(f"Gateway node status: {gateway_status['running']}")

            time.sleep(2)
            gateway_node.stop_node()
            compute_node.stop_node()

        else:
            print("Failed to start edge computing nodes")

    except Exception as e:
        print(f"Edge computing node test failed: {e}")