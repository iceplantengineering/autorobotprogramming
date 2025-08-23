"""
コンベア追従制御システム (Phase 3)
リアルタイムでコンベア上のワークピース位置を追従し、
動的な軌道修正を行う高度な制御システム
"""

import time
import math
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from basic_handling_workflow import Position, WorkPiece
from trajectory_generation import TrajectoryPoint, AdvancedTrajectoryGenerator
from integrated_safety_system import safety_system
from config_manager import config_manager

logger = logging.getLogger(__name__)

class ConveyorState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    ERROR = "error"

class TrackingState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    TRACKING = "tracking"
    PICKING = "picking"
    PLACING = "placing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConveyorConfig:
    """コンベア設定"""
    name: str
    speed: float  # mm/s
    direction: List[float]  # 方向ベクトル [x, y, z]
    origin: List[float]  # コンベア原点 [x, y, z]
    length: float  # コンベア長さ (mm)
    width: float   # コンベア幅 (mm)
    encoder_resolution: float = 1.0  # エンコーダー分解能 (mm/pulse)
    pickup_zone_start: float = 0.0  # ピックアップ開始位置 (mm)
    pickup_zone_end: float = 200.0  # ピックアップ終了位置 (mm)

@dataclass 
class WorkpieceDetection:
    """ワークピース検出情報"""
    id: str
    position: Position  # 現在位置
    velocity: List[float]  # 速度ベクトル
    size: List[float]  # サイズ [length, width, height]
    detection_time: float
    confidence: float = 1.0
    predicted_positions: List[Tuple[float, Position]] = field(default_factory=list)

class ConveyorSensor(ABC):
    """コンベアセンサー基底クラス"""
    
    @abstractmethod
    def get_encoder_position(self) -> float:
        """エンコーダー位置取得"""
        pass
    
    @abstractmethod
    def get_speed(self) -> float:
        """コンベア速度取得"""
        pass
    
    @abstractmethod
    def is_workpiece_present(self, position: float) -> bool:
        """ワークピース存在チェック"""
        pass

class MockConveyorSensor(ConveyorSensor):
    """モックコンベアセンサー（テスト用）"""
    
    def __init__(self):
        self._encoder_position = 0.0
        self._speed = 50.0  # mm/s
        self._start_time = time.time()
        self._workpiece_positions = [100.0, 300.0, 500.0]  # テスト用ワークピース位置
    
    def get_encoder_position(self) -> float:
        # 時間経過に基づく仮想エンコーダー位置
        elapsed = time.time() - self._start_time
        self._encoder_position = (elapsed * self._speed) % 1000  # 1mサイクル
        return self._encoder_position
    
    def get_speed(self) -> float:
        return self._speed
    
    def is_workpiece_present(self, position: float) -> bool:
        # テスト用：特定位置にワークピースが存在
        for wp_pos in self._workpiece_positions:
            if abs(position - wp_pos) < 25.0:  # 25mm範囲
                return True
        return False

class WorkpieceTracker:
    """ワークピース追跡システム"""
    
    def __init__(self, conveyor_config: ConveyorConfig):
        self.conveyor_config = conveyor_config
        self.active_workpieces: Dict[str, WorkpieceDetection] = {}
        self.prediction_horizon = 2.0  # 2秒先まで予測
        self.update_interval = 0.05  # 50ms更新間隔
        
        self._tracking = False
        self._thread: Optional[threading.Thread] = None
    
    def start_tracking(self):
        """追跡開始"""
        if not self._tracking:
            self._tracking = True
            self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._thread.start()
            logger.info("Workpiece tracking started")
    
    def stop_tracking(self):
        """追跡停止"""
        self._tracking = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Workpiece tracking stopped")
    
    def add_workpiece(self, workpiece_id: str, initial_position: Position, velocity: List[float] = None):
        """ワークピース追加"""
        if velocity is None:
            velocity = [self.conveyor_config.speed * self.conveyor_config.direction[0],
                       self.conveyor_config.speed * self.conveyor_config.direction[1],
                       0.0]
        
        detection = WorkpieceDetection(
            id=workpiece_id,
            position=initial_position,
            velocity=velocity,
            size=[50.0, 50.0, 25.0],  # デフォルトサイズ
            detection_time=time.time()
        )
        
        # 予測位置計算
        self._calculate_predicted_positions(detection)
        
        self.active_workpieces[workpiece_id] = detection
        logger.info(f"Added workpiece {workpiece_id} for tracking")
    
    def get_workpiece_position(self, workpiece_id: str, time_offset: float = 0.0) -> Optional[Position]:
        """ワークピース位置取得"""
        if workpiece_id not in self.active_workpieces:
            return None
        
        workpiece = self.active_workpieces[workpiece_id]
        
        if time_offset == 0.0:
            return workpiece.position
        
        # 予測位置から最適な位置を取得
        target_time = time.time() + time_offset
        
        for pred_time, pred_position in workpiece.predicted_positions:
            if abs(pred_time - target_time) < 0.1:  # 100ms以内
                return pred_position
        
        # 線形予測
        current_time = time.time()
        dt = target_time - current_time
        
        predicted_x = workpiece.position.x + workpiece.velocity[0] * dt
        predicted_y = workpiece.position.y + workpiece.velocity[1] * dt
        predicted_z = workpiece.position.z + workpiece.velocity[2] * dt
        
        return Position(predicted_x, predicted_y, predicted_z,
                       workpiece.position.rx, workpiece.position.ry, workpiece.position.rz)
    
    def remove_workpiece(self, workpiece_id: str):
        """ワークピース削除"""
        if workpiece_id in self.active_workpieces:
            del self.active_workpieces[workpiece_id]
            logger.info(f"Removed workpiece {workpiece_id} from tracking")
    
    def _tracking_loop(self):
        """追跡ループ"""
        while self._tracking:
            try:
                self._update_workpiece_positions()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Tracking loop error: {e}")
                time.sleep(0.1)
    
    def _update_workpiece_positions(self):
        """ワークピース位置更新"""
        current_time = time.time()
        
        for workpiece_id, workpiece in list(self.active_workpieces.items()):
            dt = current_time - workpiece.detection_time
            
            # 位置更新
            workpiece.position.x += workpiece.velocity[0] * self.update_interval
            workpiece.position.y += workpiece.velocity[1] * self.update_interval
            workpiece.position.z += workpiece.velocity[2] * self.update_interval
            
            # 予測位置更新
            self._calculate_predicted_positions(workpiece)
            
            # コンベア範囲外チェック
            if self._is_out_of_range(workpiece.position):
                logger.debug(f"Workpiece {workpiece_id} out of range, removing")
                self.remove_workpiece(workpiece_id)
    
    def _calculate_predicted_positions(self, workpiece: WorkpieceDetection):
        """予測位置計算"""
        workpiece.predicted_positions.clear()
        current_time = time.time()
        
        for i in range(int(self.prediction_horizon / 0.1)):  # 100ms間隔
            pred_time = current_time + (i + 1) * 0.1
            dt = pred_time - current_time
            
            pred_x = workpiece.position.x + workpiece.velocity[0] * dt
            pred_y = workpiece.position.y + workpiece.velocity[1] * dt
            pred_z = workpiece.position.z + workpiece.velocity[2] * dt
            
            pred_position = Position(pred_x, pred_y, pred_z,
                                   workpiece.position.rx, workpiece.position.ry, workpiece.position.rz)
            
            workpiece.predicted_positions.append((pred_time, pred_position))
    
    def _is_out_of_range(self, position: Position) -> bool:
        """コンベア範囲外チェック"""
        origin = self.conveyor_config.origin
        direction = self.conveyor_config.direction
        
        # コンベア軸上の投影位置計算
        rel_pos = [
            position.x - origin[0],
            position.y - origin[1],
            position.z - origin[2]
        ]
        
        projection = (
            rel_pos[0] * direction[0] +
            rel_pos[1] * direction[1] +
            rel_pos[2] * direction[2]
        )
        
        return projection < 0 or projection > self.conveyor_config.length

class ConveyorTrackingController:
    """コンベア追従制御メインクラス"""
    
    def __init__(self, conveyor_config: ConveyorConfig, sensor: ConveyorSensor = None):
        self.conveyor_config = conveyor_config
        self.sensor = sensor or MockConveyorSensor()
        self.workpiece_tracker = WorkpieceTracker(conveyor_config)
        self.trajectory_generator = AdvancedTrajectoryGenerator()
        
        self.state = TrackingState.IDLE
        self.current_workpiece_id: Optional[str] = None
        self.robot_position = Position(0, 0, 300, 0, 0, 0)  # 現在のロボット位置
        
        # 制御パラメータ
        self.max_tracking_speed = 200.0  # mm/s
        self.prediction_lead_time = 0.5  # 予測先行時間
        self.pickup_tolerance = 5.0  # ピックアップ許容誤差
        
        # コールバック
        self.on_workpiece_detected: Optional[Callable] = None
        self.on_tracking_started: Optional[Callable] = None
        self.on_picking_completed: Optional[Callable] = None
        self.on_error_occurred: Optional[Callable] = None
        
        # ステータス監視
        self.status_lock = threading.Lock()
        self.last_update_time = time.time()
    
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            # センサー初期化確認
            encoder_pos = self.sensor.get_encoder_position()
            speed = self.sensor.get_speed()
            
            logger.info(f"Conveyor sensor initialized - Position: {encoder_pos:.2f}, Speed: {speed:.2f}")
            
            # 追跡システム開始
            self.workpiece_tracker.start_tracking()
            
            # 安全システム設定
            self._setup_safety_zones()
            
            self.state = TrackingState.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Conveyor tracking system initialization failed: {e}")
            self.state = TrackingState.ERROR
            return False
    
    def start_detection(self, workpiece_template: WorkPiece = None):
        """ワークピース検出開始"""
        if self.state != TrackingState.IDLE:
            logger.warning(f"Cannot start detection in state: {self.state}")
            return False
        
        self.state = TrackingState.DETECTING
        
        # 検出ループを別スレッドで開始
        detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detection_thread.start()
        
        logger.info("Workpiece detection started")
        return True
    
    def track_workpiece(self, workpiece_id: str, target_position: Position) -> bool:
        """ワークピース追従開始"""
        if workpiece_id not in self.workpiece_tracker.active_workpieces:
            logger.error(f"Workpiece {workpiece_id} not found for tracking")
            return False
        
        self.current_workpiece_id = workpiece_id
        self.state = TrackingState.TRACKING
        
        # 追従制御を別スレッドで開始
        tracking_thread = threading.Thread(
            target=self._tracking_control_loop,
            args=(workpiece_id, target_position),
            daemon=True
        )
        tracking_thread.start()
        
        if self.on_tracking_started:
            self.on_tracking_started(workpiece_id)
        
        logger.info(f"Started tracking workpiece {workpiece_id}")
        return True
    
    def perform_dynamic_pickup(self, workpiece_id: str, place_position: Position) -> bool:
        """動的ピックアップ実行"""
        if self.state != TrackingState.TRACKING:
            logger.error("Cannot perform pickup - not in tracking state")
            return False
        
        try:
            self.state = TrackingState.PICKING
            
            # 現在のワークピース位置取得
            current_pos = self.workpiece_tracker.get_workpiece_position(workpiece_id)
            if not current_pos:
                raise ValueError(f"Cannot get position for workpiece {workpiece_id}")
            
            # ピック位置予測
            pickup_time_estimate = self._estimate_pickup_time(current_pos)
            predicted_pickup_pos = self.workpiece_tracker.get_workpiece_position(
                workpiece_id, pickup_time_estimate
            )
            
            if not predicted_pickup_pos:
                raise ValueError("Cannot predict pickup position")
            
            # 動的軌道生成
            dynamic_trajectory = self._generate_dynamic_pickup_trajectory(
                predicted_pickup_pos, place_position
            )
            
            # 軌道実行（ここでは実際のロボット制御は模擬）
            success = self._execute_dynamic_trajectory(dynamic_trajectory, workpiece_id)
            
            if success:
                self.state = TrackingState.PLACING
                logger.info(f"Dynamic pickup completed for workpiece {workpiece_id}")
                
                if self.on_picking_completed:
                    self.on_picking_completed(workpiece_id, predicted_pickup_pos)
                
                # プレース完了後
                self.workpiece_tracker.remove_workpiece(workpiece_id)
                self.current_workpiece_id = None
                self.state = TrackingState.COMPLETED
                
                return True
            else:
                self.state = TrackingState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Dynamic pickup failed: {e}")
            self.state = TrackingState.ERROR
            if self.on_error_occurred:
                self.on_error_occurred(f"Pickup failed: {e}")
            return False
    
    def get_conveyor_status(self) -> Dict[str, Any]:
        """コンベア状態取得"""
        with self.status_lock:
            return {
                "conveyor_speed": self.sensor.get_speed(),
                "encoder_position": self.sensor.get_encoder_position(),
                "tracking_state": self.state.value,
                "active_workpieces": len(self.workpiece_tracker.active_workpieces),
                "current_workpiece": self.current_workpiece_id,
                "last_update": self.last_update_time
            }
    
    def stop(self):
        """システム停止"""
        self.state = TrackingState.IDLE
        self.workpiece_tracker.stop_tracking()
        self.current_workpiece_id = None
        logger.info("Conveyor tracking system stopped")
    
    def _detection_loop(self):
        """ワークピース検出ループ"""
        detection_count = 0
        
        while self.state == TrackingState.DETECTING:
            try:
                encoder_pos = self.sensor.get_encoder_position()
                
                # ピックアップゾーン内でワークピース検出
                if (self.conveyor_config.pickup_zone_start <= encoder_pos <= 
                    self.conveyor_config.pickup_zone_end):
                    
                    if self.sensor.is_workpiece_present(encoder_pos):
                        # 新規ワークピース検出
                        workpiece_id = f"workpiece_{detection_count:03d}"
                        detection_count += 1
                        
                        # 実世界座標に変換
                        world_pos = self._encoder_to_world_position(encoder_pos)
                        
                        self.workpiece_tracker.add_workpiece(workpiece_id, world_pos)
                        
                        logger.info(f"Detected workpiece {workpiece_id} at position {world_pos.to_list()}")
                        
                        if self.on_workpiece_detected:
                            self.on_workpiece_detected(workpiece_id, world_pos)
                
                time.sleep(0.1)  # 100ms間隔
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                self.state = TrackingState.ERROR
                break
    
    def _tracking_control_loop(self, workpiece_id: str, target_position: Position):
        """追従制御ループ"""
        control_frequency = 20  # 20Hz制御
        
        while (self.state == TrackingState.TRACKING and 
               workpiece_id in self.workpiece_tracker.active_workpieces):
            
            try:
                # ワークピース位置予測
                predicted_pos = self.workpiece_tracker.get_workpiece_position(
                    workpiece_id, self.prediction_lead_time
                )
                
                if predicted_pos:
                    # ロボット位置更新（実際の制御システムでは現在位置を取得）
                    self._update_robot_position_for_tracking(predicted_pos)
                
                time.sleep(1.0 / control_frequency)
                
            except Exception as e:
                logger.error(f"Tracking control error: {e}")
                break
        
        logger.debug(f"Tracking control loop ended for {workpiece_id}")
    
    def _generate_dynamic_pickup_trajectory(self, pickup_position: Position, 
                                          place_position: Position) -> List[TrajectoryPoint]:
        """動的ピックアップ軌道生成"""
        
        # コンベア追従軌道設定
        conveyor_config = {
            "speed": self.conveyor_config.speed,
            "direction": self.conveyor_config.direction,
            "pick_position": pickup_position.to_list()
        }
        
        # 追従パラメータ
        parameters = {
            "tracking_distance": 150.0,
            "tracking_speed": min(self.max_tracking_speed, self.conveyor_config.speed * 1.2),
            "approach_height": 80.0,
            "safety_height": 50.0
        }
        
        trajectory = self.trajectory_generator.generate_conveyor_tracking_trajectory(
            conveyor_config, place_position, WorkPiece(name="dynamic_pickup"), parameters
        )
        
        return trajectory
    
    def _execute_dynamic_trajectory(self, trajectory: List[TrajectoryPoint], 
                                   workpiece_id: str) -> bool:
        """動的軌道実行（模擬）"""
        try:
            logger.info(f"Executing dynamic trajectory with {len(trajectory)} points")
            
            for i, point in enumerate(trajectory):
                # 実際のロボット制御では、ここでロボットに位置指令を送信
                
                # 軌道実行中の安全チェック
                if not safety_system.is_safe_to_move(point.position):
                    logger.error(f"Safety violation at trajectory point {i}")
                    return False
                
                # 位置更新（模擬）
                self.robot_position = point.position
                
                # 実行時間計算（模擬）
                if i > 0:
                    prev_point = trajectory[i-1]
                    distance = math.sqrt(
                        (point.position.x - prev_point.position.x)**2 +
                        (point.position.y - prev_point.position.y)**2 +
                        (point.position.z - prev_point.position.z)**2
                    )
                    
                    # 速度に基づく実行時間
                    execution_time = distance / max(point.speed, 1.0)
                    time.sleep(min(execution_time / 100, 0.1))  # 模擬実行（実際の1/100時間）
                
                logger.debug(f"Executed trajectory point {i}: {point.description}")
            
            logger.info("Dynamic trajectory execution completed")
            return True
            
        except Exception as e:
            logger.error(f"Trajectory execution failed: {e}")
            return False
    
    def _estimate_pickup_time(self, current_position: Position) -> float:
        """ピック時間推定"""
        # ロボット移動時間 + コンベア移動時間を考慮
        robot_move_distance = math.sqrt(
            (current_position.x - self.robot_position.x)**2 +
            (current_position.y - self.robot_position.y)**2 +
            (current_position.z - self.robot_position.z)**2
        )
        
        # 平均ロボット速度を50mm/sと仮定
        robot_move_time = robot_move_distance / 50.0
        
        # 余裕を持って1.5倍
        return robot_move_time * 1.5
    
    def _update_robot_position_for_tracking(self, target_position: Position):
        """追従のためのロボット位置更新"""
        # 実際の実装では、ロボットコントローラーに追従指令を送信
        # ここでは位置を徐々に更新（模擬）
        
        dx = target_position.x - self.robot_position.x
        dy = target_position.y - self.robot_position.y
        dz = target_position.z - self.robot_position.z
        
        # 追従速度制限
        max_step = self.max_tracking_speed * 0.05  # 50ms間隔での最大移動量
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance > max_step:
            scale = max_step / distance
            dx *= scale
            dy *= scale
            dz *= scale
        
        self.robot_position.x += dx
        self.robot_position.y += dy
        self.robot_position.z += dz
        
        with self.status_lock:
            self.last_update_time = time.time()
    
    def _encoder_to_world_position(self, encoder_position: float) -> Position:
        """エンコーダー位置を実世界座標に変換"""
        origin = self.conveyor_config.origin
        direction = self.conveyor_config.direction
        
        world_x = origin[0] + encoder_position * direction[0]
        world_y = origin[1] + encoder_position * direction[1]
        world_z = origin[2] + encoder_position * direction[2]
        
        return Position(world_x, world_y, world_z, 0, 0, 0)
    
    def _setup_safety_zones(self):
        """安全ゾーン設定"""
        # コンベア周辺の安全ゾーン定義
        conveyor_safety_zone = {
            "type": "monitoring",
            "name": "conveyor_area",
            "center": self.conveyor_config.origin,
            "dimensions": [
                self.conveyor_config.length + 200,  # 前後100mmずつ余裕
                self.conveyor_config.width + 200,   # 左右100mmずつ余裕
                200  # 高さ200mm
            ]
        }
        
        try:
            safety_system.add_safety_zone(conveyor_safety_zone)
            logger.info("Conveyor safety zone configured")
        except Exception as e:
            logger.warning(f"Failed to setup safety zone: {e}")

# グローバルインスタンス
_conveyor_controllers: Dict[str, ConveyorTrackingController] = {}

def create_conveyor_controller(name: str, conveyor_config: ConveyorConfig, 
                             sensor: ConveyorSensor = None) -> ConveyorTrackingController:
    """コンベアコントローラー作成"""
    controller = ConveyorTrackingController(conveyor_config, sensor)
    _conveyor_controllers[name] = controller
    return controller

def get_conveyor_controller(name: str) -> Optional[ConveyorTrackingController]:
    """コンベアコントローラー取得"""
    return _conveyor_controllers.get(name)

def shutdown_all_controllers():
    """全コントローラー停止"""
    for controller in _conveyor_controllers.values():
        controller.stop()
    _conveyor_controllers.clear()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    test_conveyor_config = ConveyorConfig(
        name="test_conveyor",
        speed=50.0,
        direction=[1.0, 0.0, 0.0],
        origin=[0.0, -300.0, 100.0],
        length=1000.0,
        width=200.0,
        pickup_zone_start=100.0,
        pickup_zone_end=800.0
    )
    
    # コントローラー作成・初期化
    controller = create_conveyor_controller("test", test_conveyor_config)
    
    if controller.initialize():
        logger.info("Conveyor tracking system initialized successfully")
        
        # コールバック設定
        controller.on_workpiece_detected = lambda wp_id, pos: logger.info(f"Detected: {wp_id} at {pos.to_list()}")
        controller.on_picking_completed = lambda wp_id, pos: logger.info(f"Pickup completed: {wp_id}")
        
        # 検出開始
        controller.start_detection()
        
        # テスト実行
        try:
            time.sleep(2.0)  # 2秒待機
            
            # 手動でワークピースを追加（テスト用）
            test_position = Position(200.0, -300.0, 125.0, 0, 0, 0)
            controller.workpiece_tracker.add_workpiece("test_wp_001", test_position)
            
            # 追従開始
            target_place_position = Position(400.0, 200.0, 100.0, 0, 0, 0)
            controller.track_workpiece("test_wp_001", target_place_position)
            
            time.sleep(3.0)  # 追従テスト
            
            # 動的ピックアップ実行
            pickup_success = controller.perform_dynamic_pickup("test_wp_001", target_place_position)
            logger.info(f"Dynamic pickup result: {pickup_success}")
            
            time.sleep(2.0)
            
            # ステータス確認
            status = controller.get_conveyor_status()
            logger.info(f"Final status: {status}")
            
        finally:
            controller.stop()
            shutdown_all_controllers()
    
    else:
        logger.error("Failed to initialize conveyor tracking system")