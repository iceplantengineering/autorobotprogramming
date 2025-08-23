import threading
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

from config_manager import config_manager
from io_message_handler import io_controller, message_processor
from error_recovery import (
    error_recovery_manager,
    ErrorType,
    ErrorSeverity,
    with_retry_and_circuit_breaker
)

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    PICKING = "picking"
    MOVING = "moving"
    PLACING = "placing"
    COMPLETING = "completing"
    ERROR = "error"
    PAUSED = "paused"
    EMERGENCY_STOPPED = "emergency_stopped"

class WorkflowResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class Position:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]
    
    @classmethod
    def from_list(cls, pos_list: List[float]) -> 'Position':
        if len(pos_list) != 6:
            raise ValueError("Position list must contain exactly 6 values")
        return cls(*pos_list)
    
    def add_offset(self, offset: 'Position') -> 'Position':
        return Position(
            self.x + offset.x,
            self.y + offset.y,
            self.z + offset.z,
            self.rx + offset.rx,
            self.ry + offset.ry,
            self.rz + offset.rz
        )

@dataclass 
class WorkPiece:
    name: str
    part_type: str
    weight: float
    dimensions: List[float]  # [length, width, height]
    material: str
    grip_force: float = 50.0
    special_handling: Dict[str, Any] = None

@dataclass
class HandlingTask:
    task_id: str
    pick_position: Position
    place_position: Position
    workpiece: WorkPiece
    approach_speed: int = 80
    work_speed: int = 30
    safety_height: float = 50.0
    grip_force: Optional[float] = None
    quality_check: bool = False
    intermediate_positions: List[Position] = None

class SafetyMonitor:
    """安全監視システム"""
    
    def __init__(self):
        self.is_monitoring = False
        self.safety_violations: List[Dict[str, Any]] = []
        self.emergency_stop_active = False
        self.door_interlock_active = True
        
    def start_monitoring(self):
        """安全監視開始"""
        self.is_monitoring = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """安全監視停止"""
        self.is_monitoring = False
        logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """安全監視ループ"""
        while self.is_monitoring:
            try:
                self._check_safety_conditions()
                time.sleep(0.1)  # 100ms間隔でチェック
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_safety_conditions(self):
        """安全条件チェック"""
        # 緊急停止チェック
        e_stop = io_controller.get_signal_value("E_STOP")
        if e_stop:
            if not self.emergency_stop_active:
                self.emergency_stop_active = True
                self._record_safety_violation("EMERGENCY_STOP", "Emergency stop activated")
        else:
            self.emergency_stop_active = False
        
        # ドアインターロックチェック
        door_closed = io_controller.get_signal_value("DOOR_CLOSED")
        air_pressure = io_controller.get_signal_value("AIR_PRESSURE_OK")
        
        safety_ok = door_closed and air_pressure and not e_stop
        
        if not safety_ok and not self._has_recent_violation("SAFETY_INTERLOCK"):
            violation_details = {
                "door_closed": door_closed,
                "air_pressure_ok": air_pressure,
                "e_stop": e_stop
            }
            self._record_safety_violation("SAFETY_INTERLOCK", "Safety interlock conditions not met", violation_details)
    
    def _record_safety_violation(self, violation_type: str, message: str, details: Dict[str, Any] = None):
        """安全違反記録"""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "type": violation_type,
            "message": message,
            "details": details or {}
        }
        self.safety_violations.append(violation)
        
        # 履歴を最新50件に制限
        if len(self.safety_violations) > 50:
            self.safety_violations = self.safety_violations[-50:]
        
        # エラー回復システムに記録
        error_recovery_manager.record_error(
            ErrorType.UNKNOWN_ERROR,
            f"Safety violation: {message}",
            ErrorSeverity.CRITICAL,
            {"violation_type": violation_type, "details": details}
        )
        
        logger.error(f"Safety violation: {violation_type} - {message}")
    
    def _has_recent_violation(self, violation_type: str, seconds: int = 5) -> bool:
        """最近の違反チェック"""
        cutoff_time = datetime.now().timestamp() - seconds
        for violation in self.safety_violations:
            if violation["type"] == violation_type:
                violation_time = datetime.fromisoformat(violation["timestamp"]).timestamp()
                if violation_time >= cutoff_time:
                    return True
        return False
    
    def is_safe_to_operate(self) -> bool:
        """動作安全確認"""
        return (
            not self.emergency_stop_active and
            io_controller.get_signal_value("DOOR_CLOSED") and
            io_controller.get_signal_value("AIR_PRESSURE_OK") and
            not io_controller.get_signal_value("E_STOP")
        )
    
    def get_safety_status(self) -> Dict[str, Any]:
        """安全状態取得"""
        return {
            "is_safe": self.is_safe_to_operate(),
            "emergency_stop_active": self.emergency_stop_active,
            "door_interlock_active": self.door_interlock_active,
            "recent_violations": [v for v in self.safety_violations if 
                               datetime.now().timestamp() - datetime.fromisoformat(v["timestamp"]).timestamp() < 300]
        }

class BasicHandlingWorkflow:
    """基本ハンドリングワークフロー"""
    
    def __init__(self, tcp_server=None):
        self.tcp_server = tcp_server
        self.current_state = WorkflowState.IDLE
        self.current_task: Optional[HandlingTask] = None
        self.safety_monitor = SafetyMonitor()
        
        self.workflow_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "average_cycle_time": 0.0,
            "total_runtime": 0.0
        }
        
        self.is_running = False
        self.is_paused = False
        self._workflow_lock = threading.Lock()
        
        # 設定読み込み
        self.robot_config = config_manager.get_config_value("APPLICATION", "robot.default_settings", {})
        self.gripper_config = config_manager.get_tool_config("gripper")
        
    def initialize_workflow(self) -> bool:
        """ワークフロー初期化"""
        try:
            # 安全監視開始
            self.safety_monitor.start_monitoring()
            
            # I/O初期状態設定
            io_controller.set_signal_value("READY_LAMP", True)
            io_controller.set_signal_value("WORKING_LAMP", False)
            io_controller.set_signal_value("ERROR_OCCURRED", False)
            
            self.current_state = WorkflowState.IDLE
            
            logger.info("Basic handling workflow initialized")
            return True
            
        except Exception as e:
            logger.error(f"Workflow initialization failed: {e}")
            return False
    
    def create_handling_task(self, task_config: Dict[str, Any]) -> HandlingTask:
        """ハンドリングタスク作成"""
        try:
            # 位置情報取得
            pick_pos = Position.from_list(task_config["pick_position"])
            place_pos = Position.from_list(task_config["place_position"])
            
            # ワークピース情報
            workpiece_data = task_config.get("workpiece", {})
            workpiece = WorkPiece(
                name=workpiece_data.get("name", "unknown"),
                part_type=workpiece_data.get("part_type", "generic"),
                weight=workpiece_data.get("weight", 1.0),
                dimensions=workpiece_data.get("dimensions", [100, 100, 50]),
                material=workpiece_data.get("material", "unknown"),
                grip_force=workpiece_data.get("grip_force", 50.0)
            )
            
            # タスク作成
            task = HandlingTask(
                task_id=task_config.get("task_id", f"task_{int(time.time())}"),
                pick_position=pick_pos,
                place_position=place_pos,
                workpiece=workpiece,
                approach_speed=task_config.get("approach_speed", 80),
                work_speed=task_config.get("work_speed", 30),
                safety_height=task_config.get("safety_height", 50.0),
                grip_force=task_config.get("grip_force"),
                quality_check=task_config.get("quality_check", False)
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to create handling task: {e}")
            raise ValueError(f"Invalid task configuration: {e}")
    
    def execute_handling_task(self, task: HandlingTask) -> WorkflowResult:
        """ハンドリングタスク実行"""
        with self._workflow_lock:
            if not self.safety_monitor.is_safe_to_operate():
                logger.error("Cannot execute task: Safety conditions not met")
                return WorkflowResult.FAILED
            
            self.current_task = task
            start_time = time.time()
            
            try:
                logger.info(f"Starting handling task: {task.task_id}")
                
                # ワークフロー実行
                result = self._execute_workflow_steps(task)
                
                # 実行時間計測
                cycle_time = time.time() - start_time
                
                # メトリクス更新
                self._update_performance_metrics(result, cycle_time)
                
                # 履歴記録
                self._record_workflow_history(task, result, cycle_time)
                
                logger.info(f"Handling task completed: {task.task_id} - Result: {result.value}")
                return result
                
            except Exception as e:
                logger.error(f"Workflow execution error: {e}")
                self.current_state = WorkflowState.ERROR
                return WorkflowResult.FAILED
            finally:
                self.current_task = None
                self.current_state = WorkflowState.IDLE
    
    def _execute_workflow_steps(self, task: HandlingTask) -> WorkflowResult:
        """ワークフロー実行ステップ"""
        
        # 1. 準備段階
        self.current_state = WorkflowState.PREPARING
        if not self._prepare_for_operation():
            return WorkflowResult.FAILED
        
        # 2. ピック動作
        self.current_state = WorkflowState.PICKING
        if not self._execute_pick_sequence(task):
            return WorkflowResult.FAILED
        
        # 3. 移動動作
        self.current_state = WorkflowState.MOVING
        if not self._execute_move_sequence(task):
            return WorkflowResult.FAILED
        
        # 4. プレース動作
        self.current_state = WorkflowState.PLACING
        if not self._execute_place_sequence(task):
            return WorkflowResult.FAILED
        
        # 5. 完了処理
        self.current_state = WorkflowState.COMPLETING
        if not self._complete_operation():
            return WorkflowResult.FAILED
        
        return WorkflowResult.SUCCESS
    
    def _prepare_for_operation(self) -> bool:
        """動作準備"""
        try:
            # 開始条件チェック
            if not self._check_start_conditions():
                return False
            
            # ステータス更新
            io_controller.set_signal_value("WORKING_LAMP", True)
            io_controller.set_signal_value("READY_LAMP", False)
            
            # グリッパー初期化
            if not self._send_gripper_command("open"):
                return False
            
            time.sleep(0.5)  # 安定化待機
            return True
            
        except Exception as e:
            logger.error(f"Operation preparation failed: {e}")
            return False
    
    def _execute_pick_sequence(self, task: HandlingTask) -> bool:
        """ピック動作シーケンス"""
        try:
            # ピック接近位置に移動
            pick_approach = task.pick_position.add_offset(
                Position(0, 0, task.safety_height, 0, 0, 0)
            )
            
            if not self._send_robot_move(pick_approach.to_list(), task.approach_speed):
                return False
            
            # グリッパー開放確認
            if not self._send_gripper_command("open"):
                return False
            
            # ピック位置に移動
            if not self._send_robot_move(task.pick_position.to_list(), task.work_speed):
                return False
            
            # ワークピース把持
            grip_force = task.grip_force or task.workpiece.grip_force
            if not self._send_gripper_command("close", {"grip_force": grip_force}):
                return False
            
            # 把持確認
            time.sleep(0.5)
            if not self._verify_grip():
                logger.error("Grip verification failed")
                return False
            
            # ピック完了信号
            io_controller.set_signal_value("PICK_COMPLETE", True)
            
            # 接近位置に戻る
            if not self._send_robot_move(pick_approach.to_list(), task.work_speed):
                return False
            
            logger.info("Pick sequence completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pick sequence failed: {e}")
            return False
    
    def _execute_move_sequence(self, task: HandlingTask) -> bool:
        """移動動作シーケンス"""
        try:
            # プレース接近位置計算
            place_approach = task.place_position.add_offset(
                Position(0, 0, task.safety_height, 0, 0, 0)
            )
            
            # 中間位置があれば経由
            if task.intermediate_positions:
                for i, intermediate_pos in enumerate(task.intermediate_positions):
                    if not self._send_robot_move(intermediate_pos.to_list(), task.approach_speed):
                        return False
                    logger.debug(f"Passed intermediate position {i+1}")
            
            # プレース接近位置に移動
            if not self._send_robot_move(place_approach.to_list(), task.approach_speed):
                return False
            
            logger.info("Move sequence completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Move sequence failed: {e}")
            return False
    
    def _execute_place_sequence(self, task: HandlingTask) -> bool:
        """プレース動作シーケンス"""
        try:
            # プレース位置に移動
            if not self._send_robot_move(task.place_position.to_list(), task.work_speed):
                return False
            
            # ワークピース開放
            if not self._send_gripper_command("open"):
                return False
            
            time.sleep(0.3)  # 開放安定化
            
            # プレース完了信号
            io_controller.set_signal_value("PLACE_COMPLETE", True)
            
            # プレース接近位置に戻る
            place_approach = task.place_position.add_offset(
                Position(0, 0, task.safety_height, 0, 0, 0)
            )
            if not self._send_robot_move(place_approach.to_list(), task.work_speed):
                return False
            
            logger.info("Place sequence completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Place sequence failed: {e}")
            return False
    
    def _complete_operation(self) -> bool:
        """動作完了処理"""
        try:
            # 完了信号設定
            io_controller.set_signal_value("WORK_COMPLETE", True)
            io_controller.set_signal_value("WORKING_LAMP", False)
            io_controller.set_signal_value("READY_LAMP", True)
            
            # 信号リセット（短時間後）
            def reset_completion_signals():
                time.sleep(2.0)
                io_controller.set_signal_value("WORK_COMPLETE", False)
                io_controller.set_signal_value("PICK_COMPLETE", False)
                io_controller.set_signal_value("PLACE_COMPLETE", False)
            
            threading.Thread(target=reset_completion_signals, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Operation completion failed: {e}")
            return False
    
    def _check_start_conditions(self) -> bool:
        """開始条件チェック"""
        required_signals = [
            ("PART_PRESENT", True),
            ("JIG_CLAMPED", True),
            ("DOOR_CLOSED", True),
            ("AIR_PRESSURE_OK", True),
            ("E_STOP", False)
        ]
        
        for signal_name, expected_value in required_signals:
            actual_value = io_controller.get_signal_value(signal_name)
            if actual_value != expected_value:
                logger.error(f"Start condition not met: {signal_name} = {actual_value}, expected {expected_value}")
                return False
        
        return True
    
    @with_retry_and_circuit_breaker("robot_communication", ErrorType.API_TIMEOUT)
    def _send_robot_move(self, position: List[float], speed: int) -> bool:
        """ロボット移動指令送信"""
        if not self.tcp_server:
            logger.warning("TCP server not available, simulating robot move")
            time.sleep(1.0)  # 移動時間シミュレーション
            return True
        
        try:
            message = {
                "message_id": f"{int(time.time() * 1000)}_robot_move",
                "timestamp": datetime.now().isoformat() + "Z",
                "command_type": "robot_move",
                "target_component": "robot_1",
                "parameters": {
                    "position": position,
                    "speed": speed
                },
                "response_required": True
            }
            
            return self.tcp_server.send_message(message)
            
        except Exception as e:
            logger.error(f"Robot move command failed: {e}")
            raise e
    
    @with_retry_and_circuit_breaker("tool_communication", ErrorType.API_TIMEOUT) 
    def _send_gripper_command(self, command: str, parameters: Dict[str, Any] = None) -> bool:
        """グリッパー指令送信"""
        if not self.tcp_server:
            logger.warning("TCP server not available, simulating gripper command")
            time.sleep(0.5)  # 動作時間シミュレーション
            return True
        
        try:
            message = {
                "message_id": f"{int(time.time() * 1000)}_gripper_control",
                "timestamp": datetime.now().isoformat() + "Z",
                "command_type": "tool_control",
                "target_component": "gripper_1",
                "parameters": {
                    "tool_command": command,
                    **(parameters or {})
                },
                "response_required": True
            }
            
            return self.tcp_server.send_message(message)
            
        except Exception as e:
            logger.error(f"Gripper command failed: {e}")
            raise e
    
    def _verify_grip(self) -> bool:
        """把持確認"""
        # 実際の実装ではグリッパーのセンサー値を確認
        # ここではシミュレーション
        grip_confirm = io_controller.get_signal_value("GRIPPER_CLOSE_CONFIRM")
        return grip_confirm if grip_confirm is not None else True
    
    def _update_performance_metrics(self, result: WorkflowResult, cycle_time: float):
        """パフォーマンスメトリクス更新"""
        self.performance_metrics["total_cycles"] += 1
        
        if result == WorkflowResult.SUCCESS:
            self.performance_metrics["successful_cycles"] += 1
        else:
            self.performance_metrics["failed_cycles"] += 1
        
        # 平均サイクル時間更新
        total_time = self.performance_metrics["total_runtime"] + cycle_time
        self.performance_metrics["total_runtime"] = total_time
        self.performance_metrics["average_cycle_time"] = total_time / self.performance_metrics["total_cycles"]
    
    def _record_workflow_history(self, task: HandlingTask, result: WorkflowResult, cycle_time: float):
        """ワークフロー履歴記録"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "result": result.value,
            "cycle_time": cycle_time,
            "workpiece_type": task.workpiece.part_type,
            "pick_position": task.pick_position.to_list(),
            "place_position": task.place_position.to_list(),
            "approach_speed": task.approach_speed,
            "work_speed": task.work_speed
        }
        
        self.workflow_history.append(history_entry)
        
        # 履歴を最新100件に制限
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[-100:]
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """ワークフロー状態取得"""
        return {
            "current_state": self.current_state.value,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_task_id": self.current_task.task_id if self.current_task else None,
            "safety_status": self.safety_monitor.get_safety_status(),
            "performance_metrics": self.performance_metrics,
            "recent_history": self.workflow_history[-10:] if self.workflow_history else []
        }
    
    def pause_workflow(self):
        """ワークフロー一時停止"""
        self.is_paused = True
        self.current_state = WorkflowState.PAUSED
        logger.info("Workflow paused")
    
    def resume_workflow(self):
        """ワークフロー再開"""
        if self.safety_monitor.is_safe_to_operate():
            self.is_paused = False
            self.current_state = WorkflowState.IDLE
            logger.info("Workflow resumed")
        else:
            logger.error("Cannot resume: Safety conditions not met")
    
    def emergency_stop(self):
        """緊急停止"""
        self.current_state = WorkflowState.EMERGENCY_STOPPED
        io_controller.set_signal_value("ERROR_OCCURRED", True)
        logger.error("Emergency stop activated")
    
    def shutdown(self):
        """ワークフローシャットダウン"""
        self.safety_monitor.stop_monitoring()
        self.is_running = False
        self.current_state = WorkflowState.IDLE
        logger.info("Basic handling workflow shutdown")

# グローバルインスタンス
basic_workflow = BasicHandlingWorkflow()

def initialize_handling_workflow(tcp_server=None) -> bool:
    """ハンドリングワークフロー初期化"""
    try:
        basic_workflow.tcp_server = tcp_server
        return basic_workflow.initialize_workflow()
    except Exception as e:
        logger.error(f"Failed to initialize handling workflow: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    if initialize_handling_workflow():
        logger.info("Handling workflow initialized successfully")
        
        # テスト用タスク設定
        test_task_config = {
            "task_id": "test_pick_place_001",
            "pick_position": [100.0, -200.0, 150.0, 0.0, 0.0, 0.0],
            "place_position": [300.0, 100.0, 150.0, 0.0, 0.0, 0.0],
            "workpiece": {
                "name": "test_part",
                "part_type": "test_component",
                "weight": 1.5,
                "dimensions": [50, 50, 25],
                "material": "plastic",
                "grip_force": 30.0
            },
            "approach_speed": 70,
            "work_speed": 25,
            "safety_height": 60.0
        }
        
        try:
            # 開始条件設定（テスト用）
            io_controller.set_signal_value("PART_PRESENT", True)
            io_controller.set_signal_value("JIG_CLAMPED", True)
            io_controller.set_signal_value("DOOR_CLOSED", True)
            io_controller.set_signal_value("AIR_PRESSURE_OK", True)
            io_controller.set_signal_value("E_STOP", False)
            
            # タスク作成・実行
            task = basic_workflow.create_handling_task(test_task_config)
            result = basic_workflow.execute_handling_task(task)
            
            logger.info(f"Test task result: {result.value}")
            
            # ステータス確認
            status = basic_workflow.get_workflow_status()
            logger.info(f"Final workflow status: {status}")
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
        
        finally:
            basic_workflow.shutdown()
    else:
        logger.error("Failed to initialize handling workflow")