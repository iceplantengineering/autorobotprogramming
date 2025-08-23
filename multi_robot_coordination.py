"""
複数ロボット協調制御システム (Phase 3)
複数のロボットが協調して作業を行うための分散制御・衝突回避システム
リアルタイム通信と協調アルゴリズムによる効率的な作業実行
"""

import time
import threading
import logging
import math
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from basic_handling_workflow import Position, WorkPiece
from integrated_safety_system import safety_system
from trajectory_generation import TrajectoryPoint, AdvancedTrajectoryGenerator
from tcp_communication import TCPCommunicationManager
from config_manager import config_manager

logger = logging.getLogger(__name__)

class RobotState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class CoordinationState(Enum):
    STANDALONE = "standalone"
    COORDINATING = "coordinating"
    SYNCHRONIZED = "synchronized"
    CONFLICT_RESOLUTION = "conflict_resolution"
    ERROR = "error"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class RobotInfo:
    """ロボット情報"""
    robot_id: str
    name: str
    position: Position
    state: RobotState
    capabilities: List[str]  # 能力リスト（溶接、組み立て、搬送など）
    workspace_bounds: Dict[str, float]  # 作業領域
    max_payload: float  # 最大ペイロード
    max_reach: float   # 最大リーチ
    current_task_id: Optional[str] = None
    last_update: float = field(default_factory=time.time)
    communication_endpoint: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class CoordinationTask:
    """協調タスク"""
    task_id: str
    task_type: str
    description: str
    required_robots: List[str]  # 必要なロボットID
    required_capabilities: List[str]
    priority: TaskPriority
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)  # 依存タスクID
    parameters: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_robots: List[str] = field(default_factory=list)
    status: str = "pending"

@dataclass
class WorkspaceReservation:
    """ワークスペース予約"""
    reservation_id: str
    robot_id: str
    workspace_area: Dict[str, float]  # x_min, x_max, y_min, y_max, z_min, z_max
    start_time: float
    end_time: float
    priority: TaskPriority

class CollisionPredictor:
    """衝突予測システム"""
    
    def __init__(self):
        self.prediction_horizon = 5.0  # 5秒先まで予測
        self.safety_margin = 50.0  # 50mm安全マージン
        self.time_step = 0.1  # 100ms間隔での予測
    
    def predict_collision(self, robot1_trajectory: List[TrajectoryPoint], 
                         robot2_trajectory: List[TrajectoryPoint],
                         robot1_bounds: Dict[str, float],
                         robot2_bounds: Dict[str, float]) -> List[Tuple[float, Position, Position]]:
        """衝突予測"""
        potential_collisions = []
        
        # 軌道の時間同期
        max_duration = max(
            len(robot1_trajectory) * self.time_step,
            len(robot2_trajectory) * self.time_step
        )
        
        for t in range(int(max_duration / self.time_step)):
            time_offset = t * self.time_step
            
            # 各ロボットの予測位置取得
            pos1 = self._get_position_at_time(robot1_trajectory, time_offset)
            pos2 = self._get_position_at_time(robot2_trajectory, time_offset)
            
            if pos1 and pos2:
                # 衝突チェック
                if self._check_robot_collision(pos1, pos2, robot1_bounds, robot2_bounds):
                    potential_collisions.append((time_offset, pos1, pos2))
        
        return potential_collisions
    
    def _get_position_at_time(self, trajectory: List[TrajectoryPoint], 
                             time_offset: float) -> Optional[Position]:
        """指定時間での位置取得"""
        if not trajectory:
            return None
        
        # 簡単な線形補間
        trajectory_index = int(time_offset / self.time_step)
        
        if trajectory_index >= len(trajectory):
            return trajectory[-1].position if trajectory else None
        
        return trajectory[trajectory_index].position
    
    def _check_robot_collision(self, pos1: Position, pos2: Position,
                              bounds1: Dict[str, float], bounds2: Dict[str, float]) -> bool:
        """ロボット間衝突チェック"""
        # 簡単な球体近似による衝突チェック
        distance = math.sqrt(
            (pos2.x - pos1.x)**2 +
            (pos2.y - pos1.y)**2 +
            (pos2.z - pos1.z)**2
        )
        
        # ロボットの最大リーチを衝突判定半径とする
        robot1_radius = bounds1.get("max_reach", 600.0)  # デフォルト600mm
        robot2_radius = bounds2.get("max_reach", 600.0)
        
        collision_threshold = robot1_radius + robot2_radius + self.safety_margin
        
        return distance < collision_threshold

class TaskScheduler:
    """タスクスケジューラー"""
    
    def __init__(self):
        self.pending_tasks: List[CoordinationTask] = []
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: List[CoordinationTask] = []
        self.task_lock = threading.Lock()
    
    def add_task(self, task: CoordinationTask):
        """タスク追加"""
        with self.task_lock:
            self.pending_tasks.append(task)
            self._sort_tasks_by_priority()
            logger.info(f"Added task {task.task_id} with priority {task.priority.name}")
    
    def get_next_task(self, available_robots: List[str], 
                      available_capabilities: List[str]) -> Optional[CoordinationTask]:
        """次のタスク取得"""
        with self.task_lock:
            for task in self.pending_tasks:
                # 依存関係チェック
                if not self._check_dependencies_satisfied(task):
                    continue
                
                # 必要な能力チェック
                if not all(cap in available_capabilities for cap in task.required_capabilities):
                    continue
                
                # 必要なロボット数チェック
                if len(available_robots) < len(task.required_robots):
                    continue
                
                # タスクを活性化
                self.pending_tasks.remove(task)
                task.assigned_robots = available_robots[:len(task.required_robots)]
                task.start_time = time.time()
                task.status = "active"
                self.active_tasks[task.task_id] = task
                
                return task
        
        return None
    
    def complete_task(self, task_id: str):
        """タスク完了"""
        with self.task_lock:
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                task.completion_time = time.time()
                task.status = "completed"
                self.completed_tasks.append(task)
                logger.info(f"Task {task_id} completed")
    
    def _check_dependencies_satisfied(self, task: CoordinationTask) -> bool:
        """依存関係チェック"""
        for dep_task_id in task.dependencies:
            # 完了タスクから検索
            if not any(t.task_id == dep_task_id for t in self.completed_tasks):
                return False
        return True
    
    def _sort_tasks_by_priority(self):
        """優先順位ソート"""
        self.pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)

class WorkspaceManager:
    """ワークスペース管理システム"""
    
    def __init__(self):
        self.reservations: List[WorkspaceReservation] = []
        self.reservation_lock = threading.Lock()
    
    def request_reservation(self, robot_id: str, workspace_area: Dict[str, float],
                          duration: float, priority: TaskPriority = TaskPriority.NORMAL) -> Optional[str]:
        """ワークスペース予約要求"""
        with self.reservation_lock:
            start_time = time.time()
            end_time = start_time + duration
            
            # 競合チェック
            if self._check_workspace_conflict(workspace_area, start_time, end_time):
                logger.warning(f"Workspace conflict for robot {robot_id}")
                return None
            
            # 予約作成
            reservation_id = str(uuid.uuid4())
            reservation = WorkspaceReservation(
                reservation_id=reservation_id,
                robot_id=robot_id,
                workspace_area=workspace_area,
                start_time=start_time,
                end_time=end_time,
                priority=priority
            )
            
            self.reservations.append(reservation)
            logger.info(f"Workspace reserved: {reservation_id} for robot {robot_id}")
            return reservation_id
    
    def release_reservation(self, reservation_id: str):
        """ワークスペース予約解放"""
        with self.reservation_lock:
            self.reservations = [r for r in self.reservations if r.reservation_id != reservation_id]
            logger.info(f"Workspace reservation released: {reservation_id}")
    
    def _check_workspace_conflict(self, requested_area: Dict[str, float], 
                                start_time: float, end_time: float) -> bool:
        """ワークスペース競合チェック"""
        for reservation in self.reservations:
            # 時間重複チェック
            if not (end_time <= reservation.start_time or start_time >= reservation.end_time):
                # 空間重複チェック
                if self._areas_overlap(requested_area, reservation.workspace_area):
                    return True
        return False
    
    def _areas_overlap(self, area1: Dict[str, float], area2: Dict[str, float]) -> bool:
        """エリア重複チェック"""
        return not (
            area1["x_max"] < area2["x_min"] or area1["x_min"] > area2["x_max"] or
            area1["y_max"] < area2["y_min"] or area1["y_min"] > area2["y_max"] or
            area1["z_max"] < area2["z_min"] or area1["z_min"] > area2["z_max"]
        )

class MultiRobotCoordinator:
    """複数ロボット協調制御メインクラス"""
    
    def __init__(self):
        self.robots: Dict[str, RobotInfo] = {}
        self.task_scheduler = TaskScheduler()
        self.workspace_manager = WorkspaceManager()
        self.collision_predictor = CollisionPredictor()
        
        # 通信管理
        self.communication_managers: Dict[str, TCPCommunicationManager] = {}
        
        # 制御状態
        self.coordination_state = CoordinationState.STANDALONE
        self.coordination_enabled = False
        self.master_robot_id: Optional[str] = None
        
        # スレッド制御
        self.coordination_thread: Optional[threading.Thread] = None
        self.status_monitor_thread: Optional[threading.Thread] = None
        self._running = False
        
        # パフォーマンス追跡
        self.coordination_metrics = {
            "tasks_completed": 0,
            "collisions_avoided": 0,
            "average_task_time": 0.0,
            "workspace_utilization": 0.0
        }
        
        # コールバック
        self.on_task_assigned: Optional[Callable[[str, CoordinationTask], None]] = None
        self.on_collision_detected: Optional[Callable[[str, str], None]] = None
        self.on_coordination_error: Optional[Callable[[str], None]] = None
        
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            # 設定読み込み
            multi_robot_config = config_manager.get_config_value("APPLICATION", "multi_robot", {})
            
            self.coordination_enabled = multi_robot_config.get("enabled", False)
            if not self.coordination_enabled:
                logger.info("Multi-robot coordination disabled in config")
                return True
            
            # ロボット設定読み込み
            robots_config = multi_robot_config.get("robots", {})
            for robot_id, robot_config in robots_config.items():
                robot_info = RobotInfo(
                    robot_id=robot_id,
                    name=robot_config.get("name", robot_id),
                    position=Position.from_list(robot_config.get("initial_position", [0, 0, 0, 0, 0, 0])),
                    state=RobotState.IDLE,
                    capabilities=robot_config.get("capabilities", []),
                    workspace_bounds=robot_config.get("workspace_bounds", {}),
                    max_payload=robot_config.get("max_payload", 10.0),
                    max_reach=robot_config.get("max_reach", 600.0),
                    communication_endpoint=robot_config.get("endpoint", f"localhost:{8888 + hash(robot_id) % 1000}")
                )
                
                self.robots[robot_id] = robot_info
                logger.info(f"Registered robot: {robot_id}")
            
            # マスターロボット決定
            if self.robots:
                self.master_robot_id = list(self.robots.keys())[0]  # 最初のロボットをマスターに
                logger.info(f"Master robot: {self.master_robot_id}")
            
            # 通信初期化
            self._initialize_communication()
            
            self.coordination_state = CoordinationState.STANDALONE
            return True
            
        except Exception as e:
            logger.error(f"Multi-robot coordinator initialization failed: {e}")
            self.coordination_state = CoordinationState.ERROR
            return False
    
    def start_coordination(self) -> bool:
        """協調制御開始"""
        if not self.coordination_enabled or not self.robots:
            logger.warning("Cannot start coordination - not configured or no robots")
            return False
        
        if self._running:
            logger.warning("Coordination already running")
            return False
        
        try:
            self._running = True
            
            # 協調制御スレッド開始
            self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
            self.coordination_thread.start()
            
            # ステータス監視スレッド開始
            self.status_monitor_thread = threading.Thread(target=self._status_monitor_loop, daemon=True)
            self.status_monitor_thread.start()
            
            self.coordination_state = CoordinationState.COORDINATING
            logger.info("Multi-robot coordination started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start coordination: {e}")
            self.coordination_state = CoordinationState.ERROR
            return False
    
    def stop_coordination(self):
        """協調制御停止"""
        self._running = False
        
        if self.coordination_thread and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=2.0)
        
        if self.status_monitor_thread and self.status_monitor_thread.is_alive():
            self.status_monitor_thread.join(timeout=2.0)
        
        self.coordination_state = CoordinationState.STANDALONE
        logger.info("Multi-robot coordination stopped")
    
    def add_robot(self, robot_info: RobotInfo) -> bool:
        """ロボット追加"""
        try:
            self.robots[robot_info.robot_id] = robot_info
            
            # 通信初期化
            if robot_info.communication_endpoint:
                host, port = robot_info.communication_endpoint.split(":")
                comm_manager = TCPCommunicationManager(host, int(port))
                self.communication_managers[robot_info.robot_id] = comm_manager
            
            logger.info(f"Robot added: {robot_info.robot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add robot {robot_info.robot_id}: {e}")
            return False
    
    def remove_robot(self, robot_id: str) -> bool:
        """ロボット削除"""
        if robot_id in self.robots:
            # 進行中タスクのチェック
            active_tasks_with_robot = [
                task for task in self.task_scheduler.active_tasks.values()
                if robot_id in task.assigned_robots
            ]
            
            if active_tasks_with_robot:
                logger.warning(f"Cannot remove robot {robot_id} - active tasks exist")
                return False
            
            # ロボット削除
            del self.robots[robot_id]
            
            # 通信管理削除
            if robot_id in self.communication_managers:
                del self.communication_managers[robot_id]
            
            # マスターロボット再選定
            if self.master_robot_id == robot_id and self.robots:
                self.master_robot_id = list(self.robots.keys())[0]
                logger.info(f"New master robot: {self.master_robot_id}")
            
            logger.info(f"Robot removed: {robot_id}")
            return True
        
        return False
    
    def add_coordination_task(self, task: CoordinationTask) -> bool:
        """協調タスク追加"""
        try:
            # 必要ロボット存在チェック
            for robot_id in task.required_robots:
                if robot_id not in self.robots:
                    logger.error(f"Required robot {robot_id} not found")
                    return False
            
            # 必要能力チェック
            available_capabilities = set()
            for robot_id in task.required_robots:
                available_capabilities.update(self.robots[robot_id].capabilities)
            
            if not all(cap in available_capabilities for cap in task.required_capabilities):
                logger.error(f"Required capabilities not available for task {task.task_id}")
                return False
            
            self.task_scheduler.add_task(task)
            logger.info(f"Coordination task added: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add task {task.task_id}: {e}")
            return False
    
    def update_robot_position(self, robot_id: str, position: Position):
        """ロボット位置更新"""
        if robot_id in self.robots:
            self.robots[robot_id].position = position
            self.robots[robot_id].last_update = time.time()
    
    def update_robot_state(self, robot_id: str, state: RobotState):
        """ロボット状態更新"""
        if robot_id in self.robots:
            self.robots[robot_id].state = state
            self.robots[robot_id].last_update = time.time()
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """協調状態取得"""
        return {
            "coordination_state": self.coordination_state.value,
            "master_robot": self.master_robot_id,
            "total_robots": len(self.robots),
            "active_robots": len([r for r in self.robots.values() if r.state != RobotState.ERROR]),
            "pending_tasks": len(self.task_scheduler.pending_tasks),
            "active_tasks": len(self.task_scheduler.active_tasks),
            "completed_tasks": len(self.task_scheduler.completed_tasks),
            "metrics": self.coordination_metrics.copy(),
            "workspace_reservations": len(self.workspace_manager.reservations)
        }
    
    def get_robot_status(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """ロボット状態取得"""
        if robot_id in self.robots:
            robot = self.robots[robot_id]
            return {
                "robot_id": robot.robot_id,
                "name": robot.name,
                "position": robot.position.to_list(),
                "state": robot.state.value,
                "capabilities": robot.capabilities,
                "current_task": robot.current_task_id,
                "last_update": robot.last_update,
                "performance_metrics": robot.performance_metrics
            }
        return None
    
    def _initialize_communication(self):
        """通信初期化"""
        for robot_id, robot_info in self.robots.items():
            if robot_info.communication_endpoint:
                try:
                    host, port = robot_info.communication_endpoint.split(":")
                    comm_manager = TCPCommunicationManager(host, int(port))
                    self.communication_managers[robot_id] = comm_manager
                    logger.info(f"Communication initialized for robot {robot_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize communication for robot {robot_id}: {e}")
    
    def _coordination_loop(self):
        """協調制御ループ"""
        logger.info("Coordination loop started")
        
        while self._running:
            try:
                # アクティブなロボットを取得
                available_robots = [
                    robot_id for robot_id, robot in self.robots.items()
                    if robot.state == RobotState.IDLE
                ]
                
                if available_robots:
                    # 利用可能な能力を集計
                    available_capabilities = set()
                    for robot_id in available_robots:
                        available_capabilities.update(self.robots[robot_id].capabilities)
                    
                    # 次のタスクを取得
                    next_task = self.task_scheduler.get_next_task(
                        available_robots, list(available_capabilities)
                    )
                    
                    if next_task:
                        # タスク実行
                        success = self._execute_coordination_task(next_task)
                        
                        if success:
                            self.coordination_metrics["tasks_completed"] += 1
                        else:
                            logger.error(f"Task execution failed: {next_task.task_id}")
                
                # 衝突チェック
                self._check_collision_risks()
                
                time.sleep(0.1)  # 10Hz制御
                
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                if self.on_coordination_error:
                    self.on_coordination_error(str(e))
        
        logger.info("Coordination loop ended")
    
    def _status_monitor_loop(self):
        """ステータス監視ループ"""
        logger.info("Status monitor loop started")
        
        while self._running:
            try:
                current_time = time.time()
                
                # ロボット通信状態チェック
                for robot_id, robot in self.robots.items():
                    if current_time - robot.last_update > 5.0:  # 5秒以上更新なし
                        if robot.state != RobotState.ERROR:
                            logger.warning(f"Robot {robot_id} communication timeout")
                            robot.state = RobotState.ERROR
                
                # ワークスペース予約期限チェック
                expired_reservations = [
                    res for res in self.workspace_manager.reservations
                    if current_time > res.end_time
                ]
                
                for reservation in expired_reservations:
                    self.workspace_manager.release_reservation(reservation.reservation_id)
                    logger.info(f"Auto-released expired reservation: {reservation.reservation_id}")
                
                time.sleep(1.0)  # 1Hz監視
                
            except Exception as e:
                logger.error(f"Status monitor error: {e}")
        
        logger.info("Status monitor loop ended")
    
    def _execute_coordination_task(self, task: CoordinationTask) -> bool:
        """協調タスク実行"""
        try:
            logger.info(f"Executing coordination task: {task.task_id}")
            
            # ワークスペース予約
            workspace_reservations = []
            for robot_id in task.assigned_robots:
                robot = self.robots[robot_id]
                workspace_area = robot.workspace_bounds.copy()
                
                reservation_id = self.workspace_manager.request_reservation(
                    robot_id, workspace_area, task.estimated_duration, task.priority
                )
                
                if reservation_id:
                    workspace_reservations.append(reservation_id)
                else:
                    # 予約失敗 - 既存予約を解放
                    for res_id in workspace_reservations:
                        self.workspace_manager.release_reservation(res_id)
                    logger.error(f"Workspace reservation failed for task {task.task_id}")
                    return False
            
            # ロボット状態更新
            for robot_id in task.assigned_robots:
                self.robots[robot_id].state = RobotState.WORKING
                self.robots[robot_id].current_task_id = task.task_id
            
            # タスクコールバック実行
            if self.on_task_assigned:
                for robot_id in task.assigned_robots:
                    self.on_task_assigned(robot_id, task)
            
            # 実際のタスク実行（模擬）
            # 実装では、各ロボットに具体的なコマンドを送信
            time.sleep(min(task.estimated_duration / 10, 2.0))  # 模擬実行時間
            
            # タスク完了
            self.task_scheduler.complete_task(task.task_id)
            
            # ワークスペース予約解放
            for reservation_id in workspace_reservations:
                self.workspace_manager.release_reservation(reservation_id)
            
            # ロボット状態をアイドルに戻す
            for robot_id in task.assigned_robots:
                self.robots[robot_id].state = RobotState.IDLE
                self.robots[robot_id].current_task_id = None
            
            logger.info(f"Coordination task completed: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task execution error {task.task_id}: {e}")
            
            # エラー時のクリーンアップ
            for robot_id in task.assigned_robots:
                if robot_id in self.robots:
                    self.robots[robot_id].state = RobotState.ERROR
                    self.robots[robot_id].current_task_id = None
            
            return False
    
    def _check_collision_risks(self):
        """衝突リスクチェック"""
        try:
            working_robots = [
                robot_id for robot_id, robot in self.robots.items()
                if robot.state == RobotState.WORKING
            ]
            
            # ペア毎に衝突チェック
            for i, robot1_id in enumerate(working_robots):
                for robot2_id in working_robots[i+1:]:
                    
                    robot1 = self.robots[robot1_id]
                    robot2 = self.robots[robot2_id]
                    
                    # 簡単な距離ベース衝突チェック
                    distance = math.sqrt(
                        (robot2.position.x - robot1.position.x)**2 +
                        (robot2.position.y - robot1.position.y)**2 +
                        (robot2.position.z - robot1.position.z)**2
                    )
                    
                    # 安全距離チェック
                    safe_distance = (robot1.max_reach + robot2.max_reach + 
                                   self.collision_predictor.safety_margin)
                    
                    if distance < safe_distance:
                        logger.warning(f"Collision risk detected: {robot1_id} and {robot2_id}")
                        
                        if self.on_collision_detected:
                            self.on_collision_detected(robot1_id, robot2_id)
                        
                        self.coordination_metrics["collisions_avoided"] += 1
        
        except Exception as e:
            logger.error(f"Collision check error: {e}")

# グローバルインスタンス
multi_robot_coordinator = MultiRobotCoordinator()

def create_coordination_task(task_type: str, description: str, 
                           required_robots: List[str], required_capabilities: List[str],
                           priority: TaskPriority = TaskPriority.NORMAL,
                           estimated_duration: float = 60.0,
                           **kwargs) -> CoordinationTask:
    """協調タスク作成ヘルパー"""
    return CoordinationTask(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        description=description,
        required_robots=required_robots,
        required_capabilities=required_capabilities,
        priority=priority,
        estimated_duration=estimated_duration,
        parameters=kwargs
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    
    # テストロボット追加
    robot1 = RobotInfo(
        robot_id="robot_001",
        name="Assembly Robot 1",
        position=Position(0, 0, 300, 0, 0, 0),
        state=RobotState.IDLE,
        capabilities=["pick_place", "assembly"],
        workspace_bounds={"x_min": -500, "x_max": 500, "y_min": -500, "y_max": 500, "z_min": 0, "z_max": 800},
        max_payload=5.0,
        max_reach=600.0
    )
    
    robot2 = RobotInfo(
        robot_id="robot_002", 
        name="Welding Robot 1",
        position=Position(800, 0, 300, 0, 0, 0),
        state=RobotState.IDLE,
        capabilities=["welding", "material_handling"],
        workspace_bounds={"x_min": 300, "x_max": 1300, "y_min": -500, "y_max": 500, "z_min": 0, "z_max": 800},
        max_payload=10.0,
        max_reach=700.0
    )
    
    # コーディネーター初期化
    if multi_robot_coordinator.initialize():
        multi_robot_coordinator.add_robot(robot1)
        multi_robot_coordinator.add_robot(robot2)
        
        # テストタスク追加
        task1 = create_coordination_task(
            task_type="assembly_line",
            description="Collaborative assembly operation",
            required_robots=["robot_001"],
            required_capabilities=["pick_place", "assembly"],
            priority=TaskPriority.NORMAL,
            estimated_duration=30.0
        )
        
        task2 = create_coordination_task(
            task_type="welding_operation",
            description="Welding task with material handling",
            required_robots=["robot_002"],
            required_capabilities=["welding"],
            priority=TaskPriority.HIGH,
            estimated_duration=45.0
        )
        
        multi_robot_coordinator.add_coordination_task(task1)
        multi_robot_coordinator.add_coordination_task(task2)
        
        # コールバック設定
        def on_task_assigned(robot_id: str, task: CoordinationTask):
            logger.info(f"Task {task.task_id} assigned to robot {robot_id}")
        
        def on_collision_detected(robot1_id: str, robot2_id: str):
            logger.warning(f"Collision detected between {robot1_id} and {robot2_id}")
        
        multi_robot_coordinator.on_task_assigned = on_task_assigned
        multi_robot_coordinator.on_collision_detected = on_collision_detected
        
        try:
            # 協調制御開始
            multi_robot_coordinator.start_coordination()
            
            # テスト実行
            time.sleep(10.0)
            
            # ステータス確認
            status = multi_robot_coordinator.get_coordination_status()
            logger.info(f"Coordination status: {status}")
            
            # ロボットステータス確認
            for robot_id in ["robot_001", "robot_002"]:
                robot_status = multi_robot_coordinator.get_robot_status(robot_id)
                logger.info(f"Robot {robot_id} status: {robot_status}")
            
        finally:
            multi_robot_coordinator.stop_coordination()
    
    else:
        logger.error("Failed to initialize multi-robot coordinator")