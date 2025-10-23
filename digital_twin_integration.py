"""
デジタルツイン統合システム (Phase 5-1)
Visual Components連携 + OPC-UAプロトコル実装
リアルタイム双方向同期・物理シミュレーション・仮想デバッグ
"""

import json
import time
import logging
import threading
import asyncio
import queue
import struct
import socket
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import numpy as np
from collections import defaultdict, deque
import xml.etree.ElementTree as ET

# OPC-UAライブラリ
try:
    from asyncua import Server, Client, ua
    from asyncua.common.methods import uamethod
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False

# 3D処理ライブラリ
try:
    import trimesh
    import open3d as o3d
    from scipy.spatial.transform import Rotation
    MESH_AVAILABLE = True
except ImportError:
    MESH_AVAILABLE = False

# 通信ライブラリ
import requests
import websockets
from concurrent.futures import ThreadPoolExecutor

from production_management_integration import ProductionManagementSystem
from multi_robot_coordination import RobotInfo, RobotState
from vc_robot_controller import VCRobotController

logger = logging.getLogger(__name__)

class DigitalTwinState(Enum):
    """デジタルツイン状態"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    SYNCHRONIZING = "synchronizing"
    SYNCHRONIZED = "synchronized"
    SIMULATION = "simulation"
    ERROR = "error"

class SyncDirection(Enum):
    """同期方向"""
    PHYSICAL_TO_VIRTUAL = "physical_to_virtual"
    VIRTUAL_TO_PHYSICAL = "virtual_to_physical"
    BIDIRECTIONAL = "bidirectional"

class OPCUADataType(Enum):
    """OPC-UAデータタイプ"""
    BOOLEAN = "Boolean"
    SBYTE = "SByte"
    BYTE = "Byte"
    INT16 = "Int16"
    UINT16 = "UInt16"
    INT32 = "Int32"
    UINT32 = "UInt32"
    INT64 = "Int64"
    UINT64 = "UInt64"
    FLOAT = "Float"
    DOUBLE = "Double"
    STRING = "String"
    DATETIME = "DateTime"

@dataclass
class VirtualRobot:
    """仮想ロボットモデル"""
    robot_id: str
    robot_type: str
    virtual_position: np.ndarray  # [x, y, z, rx, ry, rz]
    virtual_joints: Dict[str, float]
    virtual_speed: float
    virtual_payload: float
    workspace_bounds: List[List[float]]
    collision_mesh: Optional[Any] = None
    last_sync_time: Optional[datetime] = None
    sync_quality: float = 1.0  # 0-1

@dataclass
class DigitalTwinObject:
    """デジタルツインオブジェクト"""
    object_id: str
    object_type: str  # robot, tool, workpiece, environment
    physical_id: str  # 物理デバイスID
    virtual_id: str   # 仮想モデルID
    opcua_node_id: str  # OPC-UAノードID
    sync_direction: SyncDirection
    sync_frequency: float  # Hz
    last_sync: Optional[datetime] = None
    sync_errors: int = 0
    data_buffer: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class SimulationParameters:
    """シミュレーションパラメータ"""
    time_scale: float = 1.0  # 時間スケール
    physics_enabled: bool = True
    collision_detection: bool = True
    gravity: float = 9.81
    simulation_step: float = 0.016  # 60 FPS
    max_simulation_time: float = 3600.0  # 1時間

@dataclass
class TwinSyncEvent:
    """ツイン同期イベント"""
    event_id: str
    timestamp: datetime
    object_id: str
    event_type: str  # sync_request, sync_complete, error, collision_detected
    data: Dict[str, Any]
    physical_value: Any = None
    virtual_value: Any = None
    delta: Any = None

class OPCUAServerManager:
    """OPC-UAサーバーマネージャー"""

    def __init__(self, endpoint_url: str = "opc.tcp://0.0.0.0:4840/robot-digital-twin/"):
        self.endpoint_url = endpoint_url
        self.server = None
        self.is_running = False
        self.namespace_index = 2
        self.nodes = {}

    async def start_server(self) -> bool:
        """OPC-UAサーバー起動"""
        try:
            if not OPCUA_AVAILABLE:
                logger.error("OPC-UA library not available")
                return False

            self.server = Server()

            # サーバー設定
            await self.server.init()
            self.server.set_endpoint(self.endpoint_url)
            self.server.set_server_name("Robot Digital Twin Server")

            # セキュリティ設定
            self.server.set_security_policy([
                ua.SecurityPolicyType.NoSecurity,
                ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt
            ])

            # 名前空間設定
            uri = "http://robot-digital-tinct.com"
            self.namespace_index = await self.server.register_namespace(uri)

            # オブジェクトノード作成
            objects = await self.server.get_objects_node()
            twin_objects = await objects.add_object(self.namespace_index, "DigitalTwinObjects")

            # デフォルトノード作成
            await self._create_default_nodes(twin_objects)

            # サーバー起動
            async with self.server:
                self.is_running = True
                logger.info(f"OPC-UA Server started at {self.endpoint_url}")

                # 無限ループでサーバーを維持
                while self.is_running:
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start OPC-UA server: {e}")
            return False

    async def stop_server(self):
        """OPC-UAサーバー停止"""
        if self.server:
            self.is_running = False
            await self.server.stop()
            logger.info("OPC-UA Server stopped")

    async def _create_default_nodes(self, parent_node):
        """デフォルトノード作成"""
        # システムステータスノード
        system_status = await parent_node.add_object(self.namespace_index, "SystemStatus")

        await system_status.add_variable(self.namespace_index, "IsConnected", False)
        await system_status.add_variable(self.namespace_index, "LastSyncTime", datetime.now())
        await system_status.add_variable(self.namespace_index, "SyncQuality", 1.0)

        # ロボットノード作成メソッド
        self.nodes['system_status'] = system_status

    async def add_robot_node(self, robot_id: str, parent_node=None) -> Dict[str, Any]:
        """ロボットノード追加"""
        try:
            if not parent_node:
                objects = await self.server.get_objects_node()
                parent_node = await objects.add_object(self.namespace_index, "DigitalTwinObjects")

            # ロボットオブジェクト作成
            robot_node = await parent_node.add_object(self.namespace_index, f"Robot_{robot_id}")

            # 位置ノード
            position_node = await robot_node.add_object(self.namespace_index, "Position")
            x_node = await position_node.add_variable(self.namespace_index, "X", 0.0)
            y_node = await position_node.add_variable(self.namespace_index, "Y", 0.0)
            z_node = await position_node.add_variable(self.namespace_index, "Z", 0.0)
            rx_node = await position_node.add_variable(self.namespace_index, "Rx", 0.0)
            ry_node = await position_node.add_variable(self.namespace_index, "Ry", 0.0)
            rz_node = await position_node.add_variable(self.namespace_index, "Rz", 0.0)

            # 状態ノード
            status_node = await robot_node.add_object(self.namespace_index, "Status")
            state_node = await status_node.add_variable(self.namespace_index, "State", "IDLE")
            speed_node = await status_node.add_variable(self.namespace_index, "Speed", 0.0)
            payload_node = await status_node.add_variable(self.namespace_index, "Payload", 0.0)

            # 関節ノード
            joints_node = await robot_node.add_object(self.namespace_index, "Joints")
            joint_nodes = {}
            for i in range(6):  # 6軸ロボット想定
                joint_node = await joints_node.add_variable(
                    self.namespace_index, f"Joint_{i+1}", 0.0
                )
                joint_nodes[f"joint_{i+1}"] = joint_node

            # 同期ノード
            sync_node = await robot_node.add_object(self.namespace_index, "Sync")
            last_sync_node = await sync_node.add_variable(self.namespace_index, "LastSync", datetime.now())
            quality_node = await sync_node.add_variable(self.namespace_index, "Quality", 1.0)
            direction_node = await sync_node.add_variable(self.namespace_index, "Direction", "BIDIRECTIONAL")

            nodes = {
                'robot': robot_node,
                'position': {
                    'x': x_node, 'y': y_node, 'z': z_node,
                    'rx': rx_node, 'ry': ry_node, 'rz': rz_node
                },
                'status': {
                    'state': state_node, 'speed': speed_node, 'payload': payload_node
                },
                'joints': joint_nodes,
                'sync': {
                    'last_sync': last_sync_node, 'quality': quality_node, 'direction': direction_node
                }
            }

            self.nodes[robot_id] = nodes
            return nodes

        except Exception as e:
            logger.error(f"Failed to add robot node {robot_id}: {e}")
            return {}

    async def update_robot_position(self, robot_id: str, position: np.ndarray):
        """ロボット位置更新"""
        if robot_id not in self.nodes:
            return

        nodes = self.nodes[robot_id]
        if 'position' in nodes:
            await nodes['position']['x'].write_value(float(position[0]))
            await nodes['position']['y'].write_value(float(position[1]))
            await nodes['position']['z'].write_value(float(position[2]))
            await nodes['position']['rx'].write_value(float(position[3]))
            await nodes['position']['ry'].write_value(float(position[4]))
            await nodes['position']['rz'].write_value(float(position[5]))

    async def update_robot_joints(self, robot_id: str, joints: Dict[str, float]):
        """ロボット関節更新"""
        if robot_id not in self.nodes:
            return

        nodes = self.nodes[robot_id]
        if 'joints' in nodes:
            for joint_name, joint_value in joints.items():
                if joint_name in nodes['joints']:
                    await nodes['joints'][joint_name].write_value(joint_value)

    async def update_robot_status(self, robot_id: str, status: Dict[str, Any]):
        """ロボットステータス更新"""
        if robot_id not in self.nodes:
            return

        nodes = self.nodes[robot_id]
        if 'status' in nodes:
            if 'state' in status:
                await nodes['status']['state'].write_value(str(status['state']))
            if 'speed' in status:
                await nodes['status']['speed'].write_value(float(status['speed']))
            if 'payload' in status:
                await nodes['status']['payload'].write_value(float(status['payload']))

class OPCUAClientManager:
    """OPC-UAクライアントマネージャー"""

    def __init__(self):
        self.clients: Dict[str, Client] = {}
        self.connection_status: Dict[str, bool] = {}

    async def connect_client(self, client_id: str, endpoint_url: str) -> bool:
        """OPC-UAクライアント接続"""
        try:
            if not OPCUA_AVAILABLE:
                logger.error("OPC-UA library not available")
                return False

            client = Client(url=endpoint_url)
            await client.connect()

            self.clients[client_id] = client
            self.connection_status[client_id] = True

            logger.info(f"OPC-UA client {client_id} connected to {endpoint_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect OPC-UA client {client_id}: {e}")
            self.connection_status[client_id] = False
            return False

    async def disconnect_client(self, client_id: str):
        """OPC-UAクライアント切断"""
        if client_id in self.clients:
            try:
                await self.clients[client_id].disconnect()
                del self.clients[client_id]
                self.connection_status[client_id] = False
                logger.info(f"OPC-UA client {client_id} disconnected")
            except Exception as e:
                logger.error(f"Failed to disconnect OPC-UA client {client_id}: {e}")

    async def read_node_value(self, client_id: str, node_id: str) -> Any:
        """ノード値読み取り"""
        if client_id not in self.clients:
            return None

        try:
            client = self.clients[client_id]
            node = client.get_node(node_id)
            value = await node.read_value()
            return value
        except Exception as e:
            logger.error(f"Failed to read node {node_id} from client {client_id}: {e}")
            return None

    async def write_node_value(self, client_id: str, node_id: str, value: Any) -> bool:
        """ノード値書き込み"""
        if client_id not in self.clients:
            return False

        try:
            client = self.clients[client_id]
            node = client.get_node(node_id)
            await node.write_value(value)
            return True
        except Exception as e:
            logger.error(f"Failed to write node {node_id} to client {client_id}: {e}")
            return False

class VisualComponentsInterface:
    """Visual Componentsインターフェース"""

    def __init__(self, vc_host: str = "localhost", vc_port: int = 8000):
        self.vc_host = vc_host
        self.vc_port = vc_port
        self.base_url = f"http://{vc_host}:{vc_port}"
        self.session_id = None
        self.is_connected = False

    async def connect(self) -> bool:
        """Visual Componentsに接続"""
        try:
            # セッション確立
            response = requests.post(
                f"{self.base_url}/api/session/create",
                json={"client": "digital_twin", "version": "1.0"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                self.is_connected = True
                logger.info(f"Connected to Visual Components. Session: {self.session_id}")
                return True
            else:
                logger.error(f"Failed to connect to Visual Components: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Visual Components connection error: {e}")
            return False

    async def disconnect(self):
        """Visual Componentsから切断"""
        if self.is_connected and self.session_id:
            try:
                requests.post(
                    f"{self.base_url}/api/session/close",
                    json={"session_id": self.session_id},
                    timeout=5
                )
                self.is_connected = False
                self.session_id = None
                logger.info("Disconnected from Visual Components")
            except Exception as e:
                logger.error(f"Error disconnecting from Visual Components: {e}")

    async def get_robot_position(self, robot_id: str) -> Optional[np.ndarray]:
        """ロボット位置取得"""
        try:
            response = requests.get(
                f"{self.base_url}/api/robots/{robot_id}/position",
                params={"session_id": self.session_id},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                position = data.get("position", [])
                if len(position) >= 6:
                    return np.array(position[:6])
            return None

        except Exception as e:
            logger.error(f"Failed to get robot position from Visual Components: {e}")
            return None

    async def set_robot_position(self, robot_id: str, position: np.ndarray) -> bool:
        """ロボット位置設定"""
        try:
            response = requests.post(
                f"{self.base_url}/api/robots/{robot_id}/position",
                json={
                    "session_id": self.session_id,
                    "position": position.tolist()[:6]
                },
                timeout=5
            )
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to set robot position in Visual Components: {e}")
            return False

    async def simulate_operation(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """操作シミュレーション実行"""
        try:
            response = requests.post(
                f"{self.base_url}/api/simulation/run",
                json={
                    "session_id": self.session_id,
                    "operation": operation_data
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            return {}

        except Exception as e:
            logger.error(f"Failed to simulate operation in Visual Components: {e}")
            return {}

    async def get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """シミュレーション結果取得"""
        try:
            response = requests.get(
                f"{self.base_url}/api/simulation/{simulation_id}/results",
                params={"session_id": self.session_id},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            return {}

        except Exception as e:
            logger.error(f"Failed to get simulation results from Visual Components: {e}")
            return {}

class PhysicsEngine:
    """物理エンジン"""

    def __init__(self):
        self.gravity = np.array([0, 0, -9.81])
        self.collision_objects = []
        self.simulation_time = 0.0

    def check_collision(self, obj1_pos: np.ndarray, obj1_bounds: List[List[float]],
                       obj2_pos: np.ndarray, obj2_bounds: List[List[float]]) -> bool:
        """衝突検出（AABB）"""
        try:
            # バウンディングボックスを計算
            obj1_min = obj1_pos + np.array(obj1_bounds[0])
            obj1_max = obj1_pos + np.array(obj1_bounds[1])

            obj2_min = obj2_pos + np.array(obj2_bounds[0])
            obj2_max = obj2_pos + np.array(obj2_bounds[1])

            # AABB衝突判定
            return (obj1_min[0] <= obj2_max[0] and obj1_max[0] >= obj2_min[0] and
                    obj1_min[1] <= obj2_max[1] and obj1_max[1] >= obj2_min[1] and
                    obj1_min[2] <= obj2_max[2] and obj1_max[2] >= obj2_min[2])

        except Exception as e:
            logger.error(f"Collision detection error: {e}")
            return False

    def simulate_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray,
                          max_velocity: float, waypoints: List[np.ndarray] = None) -> List[np.ndarray]:
        """軌道シミュレーション"""
        try:
            if waypoints is None:
                waypoints = []

            trajectory = [start_pos]
            current_pos = start_pos.copy()

            # ウェイポイントを追加
            for waypoint in waypoints:
                trajectory.append(waypoint)

            trajectory.append(end_pos)

            # 軌道補間
            smooth_trajectory = []
            for i in range(len(trajectory) - 1):
                start = trajectory[i]
                end = trajectory[i + 1]

                # 線形補間
                steps = max(10, int(np.linalg.norm(end - start) / max_velocity * 100))
                for t in np.linspace(0, 1, steps):
                    interpolated = start + t * (end - start)
                    smooth_trajectory.append(interpolated)

            return smooth_trajectory

        except Exception as e:
            logger.error(f"Trajectory simulation error: {e}")
            return [start_pos, end_pos]

class DigitalTwinCore:
    """デジタルツインコアシステム"""

    def __init__(self, production_system: ProductionManagementSystem):
        self.production_system = production_system

        # コンポーネント
        self.opcua_server = OPCUAServerManager()
        self.opcua_client = OPCUAClientManager()
        self.vc_interface = VisualComponentsInterface()
        self.physics_engine = PhysicsEngine()

        # デジタルツインオブジェクト
        self.virtual_robots: Dict[str, VirtualRobot] = {}
        self.digital_objects: Dict[str, DigitalTwinObject] = {}

        # 同期管理
        self.sync_events: deque = deque(maxlen=1000)
        self.sync_statistics = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'average_sync_time': 0.0
        }

        # シミュレーション管理
        self.simulation_params = SimulationParameters()
        self.is_simulating = False
        self.simulation_thread: Optional[threading.Thread] = None

        # 実行制御
        self.running = False
        self.state = DigitalTwinState.DISCONNECTED
        self.sync_thread: Optional[threading.Thread] = None

        # 同期設定
        self.sync_frequency = 10.0  # Hz
        self.sync_tolerance = 0.01  # 1% tolerance
        self.max_sync_attempts = 3

        # コールバック
        self.on_sync_event: Optional[Callable[[TwinSyncEvent], None]] = None
        self.on_collision_detected: Optional[Callable[[str, str], None]] = None

    async def start(self) -> bool:
        """デジタルツインシステム起動"""
        try:
            if self.running:
                logger.warning("Digital twin system already running")
                return False

            self.running = True
            self.state = DigitalTwinState.CONNECTING

            # Visual Components接続
            if not await self.vc_interface.connect():
                logger.warning("Failed to connect to Visual Components")

            # OPC-UAサーバー起動
            opcua_task = asyncio.create_task(self.opcua_server.start_server())

            # 初期ロボット登録
            await self._register_initial_robots()

            # 同期スレッド起動
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()

            self.state = DigitalTwinState.SYNCHRONIZING
            logger.info("Digital twin system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start digital twin system: {e}")
            self.state = DigitalTwinState.ERROR
            return False

    async def stop(self):
        """デジタルツインシステム停止"""
        self.running = False
        self.state = DigitalTwinState.DISCONNECTED

        # Visual Components切断
        await self.vc_interface.disconnect()

        # OPC-UAサーバー停止
        await self.opcua_server.stop_server()

        # スレッド停止待機
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)

        logger.info("Digital twin system stopped")

    async def _register_initial_robots(self):
        """初期ロボット登録"""
        try:
            # 生産システムからロボット情報取得
            # ここでは模擬データを使用
            initial_robots = [
                {"robot_id": "robot_001", "robot_type": "6_axis_robot"},
                {"robot_id": "robot_002", "robot_type": "6_axis_robot"},
                {"robot_id": "robot_003", "robot_type": "scara_robot"}
            ]

            for robot_info in initial_robots:
                await self.register_robot(robot_info["robot_id"], robot_info["robot_type"])

        except Exception as e:
            logger.error(f"Failed to register initial robots: {e}")

    async def register_robot(self, robot_id: str, robot_type: str,
                           physical_id: str = None) -> bool:
        """ロボット登録"""
        try:
            # 仮想ロボット作成
            virtual_robot = VirtualRobot(
                robot_id=robot_id,
                robot_type=robot_type,
                virtual_position=np.zeros(6),
                virtual_joints={f"joint_{i+1}": 0.0 for i in range(6)},
                virtual_speed=0.0,
                virtual_payload=0.0,
                workspace_bounds=[[-1000, -1000, 0], [1000, 1000, 2000]]
            )
            self.virtual_robots[robot_id] = virtual_robot

            # OPC-UAノード作成
            await self.opcua_server.add_robot_node(robot_id)

            # デジタルツインオブジェクト作成
            digital_object = DigitalTwinObject(
                object_id=f"robot_{robot_id}",
                object_type="robot",
                physical_id=physical_id or robot_id,
                virtual_id=robot_id,
                opcua_node_id=f"ns=2;s=Robot_{robot_id}",
                sync_direction=SyncDirection.BIDIRECTIONAL,
                sync_frequency=self.sync_frequency
            )
            self.digital_objects[robot_id] = digital_object

            logger.info(f"Robot registered in digital twin: {robot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register robot {robot_id}: {e}")
            return False

    def _sync_loop(self):
        """同期ループ"""
        logger.info("Digital twin sync loop started")

        while self.running:
            try:
                start_time = time.time()

                # ロボット同期
                for robot_id in list(self.virtual_robots.keys()):
                    self._sync_robot(robot_id)

                # 同期統計更新
                sync_time = time.time() - start_time
                self._update_sync_statistics(sync_time)

                # 待機
                time.sleep(1.0 / self.sync_frequency)

            except Exception as e:
                logger.error(f"Digital twin sync error: {e}")
                time.sleep(1.0)

        logger.info("Digital twin sync loop ended")

    def _sync_robot(self, robot_id: str):
        """ロボット同期"""
        try:
            if robot_id not in self.virtual_robots:
                return

            virtual_robot = self.virtual_robots[robot_id]

            # Visual Componentsから物理位置取得
            physical_position = asyncio.run(self.vc_interface.get_robot_position(robot_id))

            if physical_position is not None:
                # 位置差分計算
                delta = np.linalg.norm(physical_position - virtual_robot.virtual_position)

                # 同品質判定
                sync_quality = max(0, 1.0 - delta / 100.0)  # 100mmで品質0
                virtual_robot.sync_quality = sync_quality

                # 仮想位置更新
                virtual_robot.virtual_position = physical_position.copy()
                virtual_robot.last_sync_time = datetime.now()

                # OPC-UAノード更新
                asyncio.run(self.opcua_server.update_robot_position(robot_id, physical_position))

                # 同期イベント記録
                event = TwinSyncEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    object_id=robot_id,
                    event_type="sync_complete",
                    data={
                        "sync_quality": sync_quality,
                        "delta": float(delta),
                        "sync_direction": "physical_to_virtual"
                    },
                    physical_value=physical_position.tolist(),
                    virtual_value=virtual_robot.virtual_position.tolist(),
                    delta=float(delta)
                )

                self.sync_events.append(event)
                self.sync_statistics['total_syncs'] += 1
                self.sync_statistics['successful_syncs'] += 1

                # コールバック実行
                if self.on_sync_event:
                    self.on_sync_event(event)

            else:
                self.sync_statistics['failed_syncs'] += 1

        except Exception as e:
            logger.error(f"Robot sync failed for {robot_id}: {e}")
            self.sync_statistics['failed_syncs'] += 1

    def _update_sync_statistics(self, sync_time: float):
        """同期統計更新"""
        if self.sync_statistics['total_syncs'] > 0:
            total_time = self.sync_statistics['average_sync_time'] * (self.sync_statistics['total_syncs'] - 1) + sync_time
            self.sync_statistics['average_sync_time'] = total_time / self.sync_statistics['total_syncs']

    async def execute_virtual_simulation(self, robot_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """仮想シミュレーション実行"""
        try:
            if robot_id not in self.virtual_robots:
                return {"error": "Robot not found"}

            virtual_robot = self.virtual_robots[robot_id]

            # Visual Componentsでシミュレーション実行
            simulation_result = await self.vc_interface.simulate_operation(operation)

            if simulation_result:
                # 物理エンジンで追加検証
                trajectory = simulation_result.get("trajectory", [])
                if trajectory:
                    collision_detected = False
                    for i, point in enumerate(trajectory):
                        # 衝突検出
                        for other_robot_id, other_robot in self.virtual_robots.items():
                            if other_robot_id != robot_id:
                                if self.physics_engine.check_collision(
                                    point, virtual_robot.workspace_bounds,
                                    other_robot.virtual_position, other_robot.workspace_bounds
                                ):
                                    collision_detected = True
                                    if self.on_collision_detected:
                                        self.on_collision_detected(robot_id, other_robot_id)
                                    break

                        if collision_detected:
                            break

                    simulation_result["collision_detected"] = collision_detected

            return simulation_result

        except Exception as e:
            logger.error(f"Virtual simulation failed for {robot_id}: {e}")
            return {"error": str(e)}

    async def sync_to_physical(self, robot_id: str, virtual_data: Dict[str, Any]) -> bool:
        """仮想→物理同期"""
        try:
            if robot_id not in self.virtual_robots:
                return False

            virtual_robot = self.virtual_robots[robot_id]

            # 仮想データを物理デバイスに適用
            if "position" in virtual_data:
                new_position = np.array(virtual_data["position"])

                # Visual Componentsに位置反映
                success = await self.vc_interface.set_robot_position(robot_id, new_position)

                if success:
                    virtual_robot.virtual_position = new_position
                    logger.info(f"Synced virtual position to physical for {robot_id}")

                return success

            return False

        except Exception as e:
            logger.error(f"Virtual to physical sync failed for {robot_id}: {e}")
            return False

    def get_twin_status(self) -> Dict[str, Any]:
        """ツイン状態取得"""
        return {
            "state": self.state.value,
            "is_running": self.running,
            "vc_connected": self.vc_interface.is_connected,
            "opcua_running": self.opcua_server.is_running,
            "registered_robots": len(self.virtual_robots),
            "digital_objects": len(self.digital_objects),
            "sync_statistics": self.sync_statistics.copy(),
            "is_simulating": self.is_simulating,
            "virtual_robots": {
                robot_id: {
                    "robot_type": robot.robot_type,
                    "last_sync": robot.last_sync_time.isoformat() if robot.last_sync_time else None,
                    "sync_quality": robot.sync_quality,
                    "position": robot.virtual_position.tolist()
                }
                for robot_id, robot in self.virtual_robots.items()
            }
        }

    def get_recent_sync_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """最近の同期イベント取得"""
        events = list(self.sync_events)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return [asdict(event) for event in events[:limit]]

# グローバルインスタンス
digital_twin_core: Optional[DigitalTwinCore] = None

async def initialize_digital_twin(production_system: ProductionManagementSystem) -> DigitalTwinCore:
    """デジタルツイン初期化"""
    global digital_twin_core
    digital_twin_core = DigitalTwinCore(production_system)
    return digital_twin_core

def get_digital_twin() -> Optional[DigitalTwinCore]:
    """デジタルツイン取得"""
    return digital_twin_core

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Digital Twin Integration System...")

    async def test_digital_twin():
        try:
            # モック生産管理システム
            from production_management_integration import ProductionManagementSystem, MockMESConnector
            mock_pms = ProductionManagementSystem(MockMESConnector())

            # デジタルツイン初期化
            dt_core = await initialize_digital_twin(mock_pms)

            if await dt_core.start():
                print("Digital twin system started successfully!")

                # テスト実行
                await asyncio.sleep(2)

                # 仮想シミュレーション実行
                test_operation = {
                    "robot_id": "robot_001",
                    "operation_type": "pick_and_place",
                    "start_position": [100, 100, 100, 0, 0, 0],
                    "end_position": [200, 200, 200, 0, 0, 0],
                    "speed": 50.0
                }

                result = await dt_core.execute_virtual_simulation("robot_001", test_operation)
                print(f"Simulation result: {result}")

                # ステータス確認
                status = dt_core.get_twin_status()
                print(f"Twin status: {status['state']}")
                print(f"Registered robots: {status['registered_robots']}")

                await asyncio.sleep(2)
                await dt_core.stop()

            else:
                print("Failed to start digital twin system")

        except Exception as e:
            print(f"Digital twin test failed: {e}")

    # 非同期テスト実行
    asyncio.run(test_digital_twin())