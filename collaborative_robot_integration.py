"""
コラボレーションロボット統合 (Phase 5-3)
人間協調ロボット（コボット）システム統合
安全性監視・協調作業・力制御・タスクスケジューリング
"""

import json
import time
import logging
import threading
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import queue
import math
from collections import defaultdict, deque

from multi_robot_coordination import RobotInfo, RobotState, CoordinationTask
from digital_twin_integration import DigitalTwinCore
from production_management_integration import ProductionManagementSystem

logger = logging.getLogger(__name__)

class SafetyZoneType(Enum):
    """安全ゾーンタイプ"""
    MONITORED = "monitored"  # 監視ゾーン
    RESTRICTED = "restricted"  # 制限ゾーン
    FORBIDDEN = "forbidden"  # 禁止ゾーン
    APPROACH = "approach"  # 接近ゾーン

class CollaborationMode(Enum):
    """協調モード"""
    HAND_GUIDING = "hand_guiding"  # 手誘導
    POWER_LIMITING = "power_limiting"  # 出力制限
    SPEED_LIMITING = "speed_limiting"  # 速度制限
    STANDSTILL_MONITORING = "standstill_monitoring"  # 静止監視
    COLLISION_AVOIDANCE = "collision_avoidance"  # 衝突回避

class ForceControlMode(Enum):
    """力制御モード"""
    IMPEDANCE = "impedance"  # インピーダンス制御
    ADMITTANCE = "admittance"  # アドミッタンス制御
    HYBRID = "hybrid"  # ハイブリッド制御
    COMPLIANT = "compliant"  * コンプライアンス制御

@dataclass
class SafetyZone:
    """安全ゾーン"""
    zone_id: str
    zone_type: SafetyZoneType
    zone_shape: str  # sphere, box, cylinder, custom
    position: np.ndarray  # [x, y, z]
    dimensions: np.ndarray  # [width, height, depth] or [radius]
    orientation: np.ndarray  # [rx, ry, rz]
    speed_limit: float  # m/s
    force_limit: float  # N
    allowed_objects: List[str]  # 許可オブジェクト
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class HumanPresence:
    """人間存在情報"""
    human_id: str
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    last_detected: datetime
    detection_source: str  # camera, sensor, wearable
    personal_safety_equipment: Dict[str, bool] = field(default_factory=dict)

@dataclass
class ForceSensorData:
    """力センサーデータ"""
    sensor_id: str
    timestamp: datetime
    forces: np.ndarray  # [fx, fy, fz]
    torques: np.ndarray  # [tx, ty, tz]
    raw_values: np.ndarray
    calibrated: bool
    temperature: float

@dataclass
class CollaborativeTask:
    """協調タスク"""
    task_id: str
    task_name: str
    participants: List[str]  # human_ids + robot_ids
    task_type: str  # assembly, handling, inspection, maintenance
    collaboration_mode: CollaborationMode
    safety_requirements: Dict[str, Any]
    workflow_steps: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "pending"  # pending, active, paused, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0

@dataclass
class SafetyEvent:
    """安全イベント"""
    event_id: str
    timestamp: datetime
    event_type: str  # collision_detected, human_approach, force_exceeded, safety_zone_breach
    severity: str  # low, medium, high, critical
    robot_id: str
    human_id: Optional[str]
    position: np.ndarray
    description: str
    immediate_action: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class SafetyMonitor:
    """安全監視システム"""

    def __init__(self):
        self.safety_zones: Dict[str, SafetyZone] = {}
        self.active_humans: Dict[str, HumanPresence] = {}
        self.safety_events: List[SafetyEvent] = []
        self.force_sensors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # 安全設定
        self.emergency_stop_enabled = True
        self.safety_scan_frequency = 20.0  # Hz
        self.max_approach_speed = 0.25  # m/s
        self.max_contact_force = 50.0  # N
        self.min_safety_distance = 0.3  # m

        # コールバック
        self.on_safety_event: Optional[Callable[[SafetyEvent], None]] = None
        self.on_emergency_stop: Optional[Callable[[str], None]] = None

    def add_safety_zone(self, zone: SafetyZone):
        """安全ゾーン追加"""
        self.safety_zones[zone.zone_id] = zone
        logger.info(f"Added safety zone: {zone.zone_id} ({zone.zone_type.value})")

    def update_human_presence(self, human_id: str, presence: HumanPresence):
        """人間存在情報更新"""
        self.active_humans[human_id] = presence

    def add_force_sensor_data(self, sensor_id: str, data: ForceSensorData):
        """力センサーデータ追加"""
        self.force_sensors[sensor_id].append(data)

    def perform_safety_scan(self, robot_positions: Dict[str, np.ndarray]) -> List[SafetyEvent]:
        """安全スキャン実行"""
        events = []
        current_time = datetime.now()

        # 古い人間存在情報をクリーンアップ
        self._cleanup_old_human_presence(current_time)

        for robot_id, robot_pos in robot_positions.items():
            # 人間接近検知
            human_events = self._check_human_proximity(robot_id, robot_pos, current_time)
            events.extend(human_events)

            # 安全ゾーン侵入検知
            zone_events = self._check_safety_zones(robot_id, robot_pos, current_time)
            events.extend(zone_events)

            # 力制限値超過検知
            force_events = self._check_force_limits(robot_id, current_time)
            events.extend(force_events)

        # 新しいイベントを記録
        for event in events:
            self.safety_events.append(event)
            if self.on_safety_event:
                self.on_safety_event(event)

        # 緊急停止が必要なイベントを処理
        critical_events = [e for e in events if e.severity == "critical"]
        for event in critical_events:
            if self.on_emergency_stop:
                self.on_emergency_stop(event.robot_id)

        return events

    def _cleanup_old_human_presence(self, current_time: datetime):
        """古い人間存在情報クリーンアップ"""
        timeout = timedelta(seconds=5.0)  # 5秒以上更新がない情報を削除

        expired_humans = [
            human_id for human_id, presence in self.active_humans.items()
            if current_time - presence.last_detected > timeout
        ]

        for human_id in expired_humans:
            del self.active_humans[human_id]
            logger.debug(f"Removed expired human presence: {human_id}")

    def _check_human_proximity(self, robot_id: str, robot_pos: np.ndarray,
                              current_time: datetime) -> List[SafetyEvent]:
        """人間接近チェック"""
        events = []

        for human_id, human_presence in self.active_humans.items():
            distance = np.linalg.norm(robot_pos[:3] - human_presence.position[:3])
            relative_velocity = np.linalg.norm(human_presence.velocity[:3])

            # 非常に近い場合
            if distance < self.min_safety_distance:
                event = SafetyEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=current_time,
                    event_type="human_approach",
                    severity="critical",
                    robot_id=robot_id,
                    human_id=human_id,
                    position=robot_pos,
                    description=f"Human dangerously close to robot ({distance:.3f}m)",
                    immediate_action="emergency_stop"
                )
                events.append(event)

            # 高速接近の場合
            elif distance < 0.5 and relative_velocity > self.max_approach_speed:
                event = SafetyEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=current_time,
                    event_type="human_approach",
                    severity="high",
                    robot_id=robot_id,
                    human_id=human_id,
                    position=robot_pos,
                    description=f"Human approaching too fast ({relative_velocity:.3f}m/s)",
                    immediate_action="speed_limit"
                )
                events.append(event)

        return events

    def _check_safety_zones(self, robot_id: str, robot_pos: np.ndarray,
                           current_time: datetime) -> List[SafetyEvent]:
        """安全ゾーンチェック"""
        events = []

        for zone_id, zone in self.safety_zones.items():
            if not zone.active:
                continue

            in_zone = self._is_position_in_zone(robot_pos, zone)

            if in_zone:
                if zone.zone_type == SafetyZone.FORBIDDEN:
                    event = SafetyEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=current_time,
                        event_type="safety_zone_breach",
                        severity="critical",
                        robot_id=robot_id,
                        human_id=None,
                        position=robot_pos,
                        description=f"Robot entered forbidden zone: {zone_id}",
                        immediate_action="emergency_stop"
                    )
                    events.append(event)

                elif zone.zone_type == SafetyZone.RESTRICTED:
                    event = SafetyEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=current_time,
                        event_type="safety_zone_breach",
                        severity="medium",
                        robot_id=robot_id,
                        human_id=None,
                        position=robot_pos,
                        description=f"Robot entered restricted zone: {zone_id}",
                        immediate_action="speed_limit"
                    )
                    events.append(event)

        return events

    def _is_position_in_zone(self, position: np.ndarray, zone: SafetyZone) -> bool:
        """位置がゾーン内にあるかチェック"""
        try:
            rel_pos = position[:3] - zone.position

            if zone.zone_shape == "sphere":
                distance = np.linalg.norm(rel_pos)
                return distance <= zone.dimensions[0]

            elif zone.zone_shape == "box":
                # 簡易的なAABB判定
                half_dims = zone.dimensions[:3] / 2
                return (abs(rel_pos[0]) <= half_dims[0] and
                        abs(rel_pos[1]) <= half_dims[1] and
                        abs(rel_pos[2]) <= half_dims[2])

            elif zone.zone_shape == "cylinder":
                radius = zone.dimensions[0]
                height = zone.dimensions[1]
                radial_dist = np.linalg.norm(rel_pos[:2])
                return radial_dist <= radius and abs(rel_pos[2]) <= height / 2

        except Exception as e:
            logger.error(f"Zone checking error: {e}")

        return False

    def _check_force_limits(self, robot_id: str, current_time: datetime) -> List[SafetyEvent]:
        """力制限値チェック"""
        events = []

        for sensor_id, sensor_data_queue in self.force_sensors.items():
            if len(sensor_data_queue) < 1:
                continue

            latest_data = sensor_data_queue[-1]
            total_force = np.linalg.norm(latest_data.forces)

            if total_force > self.max_contact_force:
                event = SafetyEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=current_time,
                    event_type="force_exceeded",
                    severity="high",
                    robot_id=robot_id,
                    human_id=None,
                    position=latest_data.forces,
                    description=f"Contact force exceeded limit: {total_force:.1f}N > {self.max_contact_force}N",
                    immediate_action="force_limit"
                )
                events.append(event)

        return events

class ForceController:
    """力制御システム"""

    def __init__(self):
        self.control_mode = ForceControlMode.IMPEDANCE
        self.impedance_params = {
            'stiffness': np.array([1000, 1000, 1000, 100, 100, 100]),  # N/m or Nm/rad
            'damping': np.array([50, 50, 50, 5, 5, 5])  # Ns/m or Nms/rad
        }
        self.force_limit = 50.0  # N
        self.velocity_limit = 0.5  # m/s
        self.is_enabled = False

    def enable_force_control(self, control_mode: ForceControlMode, params: Dict[str, Any] = None):
        """力制御有効化"""
        self.control_mode = control_mode
        self.is_enabled = True

        if params:
            if 'stiffness' in params:
                self.impedance_params['stiffness'] = np.array(params['stiffness'])
            if 'damping' in params:
                self.impedance_params['damping'] = np.array(params['damping'])

        logger.info(f"Force control enabled: {control_mode.value}")

    def disable_force_control(self):
        """力制御無効化"""
        self.is_enabled = False
        logger.info("Force control disabled")

    def calculate_commanded_force(self, current_position: np.ndarray,
                                 desired_position: np.ndarray,
                                 current_velocity: np.ndarray,
                                 external_force: np.ndarray) -> np.ndarray:
        """指令力計算"""
        if not self.is_enabled:
            return np.zeros(6)

        try:
            # 位置偏差
            position_error = desired_position - current_position

            if self.control_mode == ForceControlMode.IMPEDANCE:
                # インピーダンス制御: F = K*error + D*(-velocity)
                force_command = (
                    self.impedance_params['stiffness'] * position_error -
                    self.impedance_params['damping'] * current_velocity
                )
            elif self.control_mode == ForceControlMode.ADMITTANCE:
                # アドミッタンス制御
                velocity_command = (
                    self.impedance_params['stiffness'] * position_error +
                    self.impedance_params['damping'] * current_velocity
                )
                force_command = velocity_command  # 簡略化

            # 力制限値でクリップ
            force_magnitude = np.linalg.norm(force_command)
            if force_magnitude > self.force_limit:
                force_command = force_command * (self.force_limit / force_magnitude)

            return force_command

        except Exception as e:
            logger.error(f"Force calculation error: {e}")
            return np.zeros(6)

class HandGuidingSystem:
    """手誘導システム"""

    def __init__(self):
        self.is_guiding = False
        self.guiding_robot_id: Optional[str] = None
        self.guiding_session_id: Optional[str] = None
        self.force_threshold = 5.0  # N
        self.max_guiding_speed = 0.3  # m/s
        self.guiding_trajectory: List[np.ndarray] = []

    def start_guiding(self, robot_id: str) -> str:
        """手誘導開始"""
        self.is_guiding = True
        self.guiding_robot_id = robot_id
        self.guiding_session_id = str(uuid.uuid4())
        self.guiding_trajectory = []

        logger.info(f"Hand guiding started for robot: {robot_id}")
        return self.guiding_session_id

    def stop_guiding(self) -> Optional[List[np.ndarray]]:
        """手誘導停止"""
        trajectory = self.guiding_trajectory.copy()

        self.is_guiding = False
        self.guiding_robot_id = None
        session_id = self.guiding_session_id
        self.guiding_session_id = None
        self.guiding_trajectory = []

        logger.info(f"Hand guiding stopped. Session: {session_id}")
        return trajectory

    def process_guiding_force(self, force_data: ForceSensorData, current_position: np.ndarray) -> Optional[np.ndarray]:
        """手誘導力処理"""
        if not self.is_guiding or not self.guiding_robot_id:
            return None

        total_force = np.linalg.norm(force_data.forces)

        if total_force > self.force_threshold:
            # 力に基づいて速度指令を計算
            force_direction = force_data.forces / total_force
            velocity_command = force_direction * min(total_force / 10.0, self.max_guiding_speed)

            # 位置更新（簡易的）
            new_position = current_position + velocity_command * 0.01  # 10ms Δt
            self.guiding_trajectory.append(new_position)

            return velocity_command

        return None

class CollaborativeRobotController:
    """コラボレーションロボットコントローラ"""

    def __init__(self, robot_id: str, initial_position: np.ndarray):
        self.robot_id = robot_id
        self.current_position = initial_position.copy()
        self.current_velocity = np.zeros(6)
        self.target_position = initial_position.copy()
        self.target_velocity = np.zeros(6)

        # 協動機能
        self.safety_monitor = SafetyMonitor()
        self.force_controller = ForceController()
        self.hand_guiding = HandGuidingSystem()

        # 状態管理
        self.is_collaborative = True
        self.emergency_stopped = False
        self.last_safety_scan = datetime.now()
        self.safety_scan_interval = 0.05  # 20Hz

        # 設定
        self.max_speed = 1.0  # m/s
        self.max_acceleration = 2.0  # m/s²
        self.default_safety_zones = self._create_default_safety_zones()

        # コールバック
        self.on_position_update: Optional[Callable[[np.ndarray], None]] = None
        self.on_safety_event: Optional[Callable[[SafetyEvent], None]] = None

        # デフォルト安全ゾーン設定
        for zone in self.default_safety_zones:
            self.safety_monitor.add_safety_zone(zone)

    def _create_default_safety_zones(self) -> List[SafetyZone]:
        """デフォルト安全ゾーン作成"""
        zones = []

        # 禁止ゾーン（ロボットベース周り）
        zones.append(SafetyZone(
            zone_id="base_forbidden",
            zone_type=SafetyZoneType.FORBIDDEN,
            zone_shape="cylinder",
            position=np.array([0, 0, 0]),
            dimensions=np.array([0.3, 0.5]),  # radius, height
            orientation=np.zeros(3),
            speed_limit=0.0,
            force_limit=0.0,
            allowed_objects=[]
        ))

        # 監視ゾーン（作業領域）
        zones.append(SafetyZone(
            zone_id="workspace_monitored",
            zone_type=SafetyZoneType.MONITORED,
            zone_shape="box",
            position=np.array([0.5, 0, 0.5]),
            dimensions=np.array([1.0, 1.0, 1.0]),
            orientation=np.zeros(3),
            speed_limit=0.5,
            force_limit=30.0,
            allowed_objects=["robot", "tool"]
        ))

        return zones

    def update_position(self, position: np.ndarray, velocity: np.ndarray = None):
        """位置更新"""
        self.current_position = position.copy()
        if velocity is not None:
            self.current_velocity = velocity.copy()

        if self.on_position_update:
            self.on_position_update(position)

    def set_target_position(self, target: np.ndarray, max_speed: float = None):
        """目標位置設定"""
        self.target_position = target.copy()
        if max_speed:
            self.max_speed = max_speed

    def move_to_target(self, dt: float) -> bool:
        """目標位置に移動"""
        try:
            # 安全スキャン
            current_time = datetime.now()
            if (current_time - self.last_safety_scan).total_seconds() > self.safety_scan_interval:
                safety_events = self.safety_monitor.perform_safety_scan({self.robot_id: self.current_position})
                self.last_safety_scan = current_time

                # 緊急停止処理
                if any(event.severity == "critical" for event in safety_events):
                    self.emergency_stopped = True
                    logger.warning(f"Emergency stop triggered for {self.robot_id}")
                    return False

            # 緊急停止中は移動しない
            if self.emergency_stopped:
                return False

            # 手誘導中は強制優先
            if self.hand_guiding.is_guiding:
                return True

            # 速度計算
            position_error = self.target_position - self.current_position
            distance = np.linalg.norm(position_error[:3])

            if distance < 0.001:  # 1mm tolerance
                return True

            # 単純なP制御
            kp = 5.0  # 比例ゲイン
            desired_velocity = kp * position_error

            # 速度制限
            speed = np.linalg.norm(desired_velocity[:3])
            if speed > self.max_speed:
                desired_velocity = desired_velocity * (self.max_speed / speed)

            # 力制御が有効な場合は適用
            if self.force_controller.is_enabled:
                external_force = np.zeros(6)  # 実際はセンサーから取得
                force_command = self.force_controller.calculate_commanded_force(
                    self.current_position,
                    self.target_position,
                    self.current_velocity,
                    external_force
                )
                # 力指令を速度指令に変換（簡略化）
                desired_velocity += force_command * 0.01

            # 位置更新
            self.current_position += desired_velocity * dt
            self.current_velocity = desired_velocity

            if self.on_position_update:
                self.on_position_update(self.current_position)

            return False

        except Exception as e:
            logger.error(f"Movement error for {self.robot_id}: {e}")
            return False

    def reset_emergency_stop(self):
        """非常停止リセット"""
        self.emergency_stopped = False
        logger.info(f"Emergency stop reset for {self.robot_id}")

    def enable_collaborative_mode(self, mode: CollaborationMode, params: Dict[str, Any] = None):
        """協調モード有効化"""
        if mode == CollaborationMode.HAND_GUIDING:
            self.hand_guiding.start_guiding(self.robot_id)
        elif mode == CollaborationMode.POWER_LIMITING:
            self.force_controller.enable_force_control(ForceControlMode.IMPEDANCE, params)
        elif mode == CollaborationMode.SPEED_LIMITING:
            self.max_speed = params.get('speed_limit', 0.3) if params else 0.3

        logger.info(f"Collaborative mode enabled for {self.robot_id}: {mode.value}")

    def disable_collaborative_mode(self):
        """協調モード無効化"""
        if self.hand_guiding.is_guiding:
            self.hand_guiding.stop_guiding()
        self.force_controller.disable_force_control()
        self.max_speed = 1.0

        logger.info(f"Collaborative mode disabled for {self.robot_id}")

    def get_robot_status(self) -> Dict[str, Any]:
        """ロボット状態取得"""
        return {
            "robot_id": self.robot_id,
            "position": self.current_position.tolist(),
            "velocity": self.current_velocity.tolist(),
            "is_collaborative": self.is_collaborative,
            "emergency_stopped": self.emergency_stopped,
            "hand_guiding_active": self.hand_guiding.is_guiding,
            "force_control_enabled": self.force_controller.is_enabled,
            "active_safety_zones": len([z for z in self.safety_monitor.safety_zones.values() if z.active]),
            "recent_safety_events": len([e for e in self.safety_monitor.safety_events
                                       if (datetime.now() - e.timestamp).total_seconds() < 60])
        }

class CollaborativeTaskManager:
    """協調タスクマネージャ"""

    def __init__(self):
        self.active_tasks: Dict[str, CollaborativeTask] = []
        self.task_history: List[CollaborativeTask] = []
        self.assigned_robots: Dict[str, str] = {}  # robot_id -> task_id

    def create_collaborative_task(self, task_data: Dict[str, Any]) -> str:
        """協調タスク作成"""
        task = CollaborativeTask(
            task_id=str(uuid.uuid4()),
            task_name=task_data["task_name"],
            participants=task_data["participants"],
            task_type=task_data["task_type"],
            collaboration_mode=CollaborationMode(task_data["collaboration_mode"]),
            safety_requirements=task_data.get("safety_requirements", {}),
            workflow_steps=task_data.get("workflow_steps", [])
        )

        self.active_tasks.append(task)
        logger.info(f"Created collaborative task: {task.task_id}")
        return task.task_id

    def assign_robot_to_task(self, robot_id: str, task_id: str) -> bool:
        """ロボットをタスクに割り当て"""
        task = self._get_task_by_id(task_id)
        if not task:
            return False

        if robot_id not in task.participants:
            task.participants.append(robot_id)

        self.assigned_robots[robot_id] = task_id
        logger.info(f"Assigned robot {robot_id} to task {task_id}")
        return True

    def start_task(self, task_id: str) -> bool:
        """タスク開始"""
        task = self._get_task_by_id(task_id)
        if not task:
            return False

        task.status = "active"
        task.started_at = datetime.now()
        task.current_step = 0

        logger.info(f"Started collaborative task: {task_id}")
        return True

    def complete_task_step(self, task_id: str) -> bool:
        """タスクステップ完了"""
        task = self._get_task_by_id(task_id)
        if not task or task.status != "active":
            return False

        task.current_step += 1
        task.progress = task.current_step / len(task.workflow_steps)

        if task.current_step >= len(task.workflow_steps):
            task.status = "completed"
            task.completed_at = datetime.now()
            task.progress = 1.0

            # 履歴に移動
            self.active_tasks.remove(task)
            self.task_history.append(task)

            # ロボット割り当てを解除
            robots_to_remove = [rid for rid, tid in self.assigned_robots.items() if tid == task_id]
            for robot_id in robots_to_remove:
                del self.assigned_robots[robot_id]

        logger.info(f"Task step completed: {task_id} (step {task.current_step})")
        return True

    def _get_task_by_id(self, task_id: str) -> Optional[CollaborativeTask]:
        """タスクIDでタスク取得"""
        for task in self.active_tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_active_tasks(self) -> List[CollaborativeTask]:
        """アクティブタスク取得"""
        return self.active_tasks.copy()

    def get_robot_task(self, robot_id: str) -> Optional[CollaborativeTask]:
        """ロボットの割り当てタスク取得"""
        task_id = self.assigned_robots.get(robot_id)
        return self._get_task_by_id(task_id) if task_id else None

class CollaborativeRobotSystem:
    """コラボレーションロボットシステム"""

    def __init__(self, digital_twin: DigitalTwinCore, production_system: ProductionManagementSystem):
        self.digital_twin = digital_twin
        self.production_system = production_system

        # コンポーネント
        self.collaborative_robots: Dict[str, CollaborativeRobotController] = {}
        self.task_manager = CollaborativeTaskManager()

        # 実行制御
        self.running = False
        self.control_thread: Optional[threading.Thread] = None
        self.control_frequency = 100.0  # Hz

        # 統計
        self.system_stats = {
            'total_collaboration_time': 0.0,
            'completed_tasks': 0,
            'safety_events': 0,
            'human_interactions': 0
        }

        # コールバック
        self.on_task_completed: Optional[Callable[[CollaborativeTask], None]] = None
        self.on_safety_alert: Optional[Callable[[SafetyEvent], None]] = None

    def register_collaborative_robot(self, robot_id: str, initial_position: np.ndarray) -> bool:
        """コラボレーションロボット登録"""
        try:
            controller = CollaborativeRobotController(robot_id, initial_position)

            # 安全監視コールバック設定
            controller.safety_monitor.on_safety_event = self._handle_safety_event

            self.collaborative_robots[robot_id] = controller
            logger.info(f"Registered collaborative robot: {robot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register collaborative robot {robot_id}: {e}")
            return False

    def start_system(self) -> bool:
        """システム起動"""
        try:
            if self.running:
                logger.warning("Collaborative robot system already running")
                return False

            self.running = True

            # 制御ループスレッド開始
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

            logger.info("Collaborative robot system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start collaborative robot system: {e}")
            return False

    def stop_system(self):
        """システム停止"""
        self.running = False

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        logger.info("Collaborative robot system stopped")

    def _control_loop(self):
        """制御ループ"""
        logger.info("Collaborative robot control loop started")
        dt = 1.0 / self.control_frequency

        while self.running:
            try:
                current_time = datetime.now()

                # 全ロボットの制御更新
                for robot_id, controller in self.collaborative_robots.items():
                    # 手誘導処理
                    if controller.hand_guiding.is_guiding:
                        # 力センサーデータ取得（模擬）
                        force_data = self._simulate_force_sensor_data(robot_id)
                        if force_data:
                            controller.hand_guiding.process_guiding_force(force_data, controller.current_position)

                    # 位置制御
                    reached_target = controller.move_to_target(dt)
                    if reached_target:
                        # タスク進行
                        task = self.task_manager.get_robot_task(robot_id)
                        if task and task.status == "active":
                            self.task_manager.complete_task_step(task.task_id)

                # 統計更新
                self._update_statistics(current_time)

                # 制御周期待機
                time.sleep(dt)

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                time.sleep(0.1)

        logger.info("Collaborative robot control loop ended")

    def _simulate_force_sensor_data(self, robot_id: str) -> Optional[ForceSensorData]:
        """力センサーデータシミュレーション"""
        # 実際の実装ではセンサーからデータ取得
        import random

        return ForceSensorData(
            sensor_id=f"{robot_id}_force_sensor",
            timestamp=datetime.now(),
            forces=np.array([
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-5, 5),
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(-1, 1)
            ]),
            torques=np.zeros(6),
            raw_values=np.random.randn(12),
            calibrated=True,
            temperature=25.0
        )

    def _update_statistics(self, current_time: datetime):
        """統計情報更新"""
        # 協調時間更新
        active_robots = len(self.collaborative_robots)
        self.system_stats['total_collaboration_time'] += active_robots * (1.0 / self.control_frequency)

    def _handle_safety_event(self, event: SafetyEvent):
        """安全イベント処理"""
        self.system_stats['safety_events'] += 1

        if self.on_safety_alert:
            self.on_safety_alert(event)

        # 重大なイベントはログ記録
        if event.severity in ["high", "critical"]:
            logger.warning(f"Safety event in {event.robot_id}: {event.description}")

    def create_collaborative_task(self, task_data: Dict[str, Any]) -> str:
        """協調タスク作成"""
        task_id = self.task_manager.create_collaborative_task(task_data)

        # ロボット自動割り当て
        for participant in task_data["participants"]:
            if participant in self.collaborative_robots:
                self.task_manager.assign_robot_to_task(participant, task_id)

        return task_id

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        active_tasks = self.task_manager.get_active_tasks()

        return {
            "running": self.running,
            "registered_robots": len(self.collaborative_robots),
            "active_tasks": len(active_tasks),
            "completed_tasks": self.system_stats['completed_tasks'],
            "safety_events": self.system_stats['safety_events'],
            "total_collaboration_time": self.system_stats['total_collaboration_time'],
            "robots": {
                robot_id: controller.get_robot_status()
                for robot_id, controller in self.collaborative_robots.items()
            },
            "tasks": [asdict(task) for task in active_tasks]
        }

# グローバルインスタンス
collaborative_robot_system: Optional[CollaborativeRobotSystem] = None

def initialize_collaborative_robot_system(digital_twin: DigitalTwinCore,
                                         production_system: ProductionManagementSystem) -> CollaborativeRobotSystem:
    """コラボレーションロボットシステム初期化"""
    global collaborative_robot_system
    collaborative_robot_system = CollaborativeRobotSystem(digital_twin, production_system)
    return collaborative_robot_system

def get_collaborative_robot_system() -> Optional[CollaborativeRobotSystem]:
    """コラボレーションロボットシステム取得"""
    return collaborative_robot_system

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Collaborative Robot Integration System...")

    try:
        # モックシステム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())
        mock_dt = None  # デジタルツインは別途初期化

        # コラボレーションロボットシステム初期化
        cobot_system = initialize_collaborative_robot_system(mock_dt, mock_pms)

        # コラボレーションロボット登録
        cobot_system.register_collaborative_robot("cobot_001", np.zeros(6))
        cobot_system.register_collaborative_robot("cobot_002", np.array([0.5, 0, 0, 0, 0, 0]))

        # システム起動
        if cobot_system.start_system():
            print("Collaborative robot system started successfully!")

            # 協調タスク作成
            task_data = {
                "task_name": "協調組み立て作業",
                "participants": ["cobot_001", "cobot_002"],
                "task_type": "assembly",
                "collaboration_mode": "hand_guiding",
                "safety_requirements": {
                    "max_force": 30.0,
                    "max_speed": 0.3
                },
                "workflow_steps": [
                    {"step": 1, "description": "部品をピック"},
                    {"step": 2, "description": "組み立て位置へ移動"},
                    {"step": 3, "description": "組み立て実行"}
                ]
            }

            task_id = cobot_system.create_collaborative_task(task_data)
            print(f"Created collaborative task: {task_id}")

            # テスト実行
            time.sleep(2)

            # ステータス確認
            status = cobot_system.get_system_status()
            print(f"System status:")
            print(f"  Registered robots: {status['registered_robots']}")
            print(f"  Active tasks: {status['active_tasks']}")

            time.sleep(2)
            cobot_system.stop_system()

        else:
            print("Failed to start collaborative robot system")

    except Exception as e:
        print(f"Collaborative robot system test failed: {e}")