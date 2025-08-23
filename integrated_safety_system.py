import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from config_manager import config_manager
from io_message_handler import io_controller
from basic_handling_workflow import Position
from error_recovery import error_recovery_manager, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

class SafetyZoneType(Enum):
    RESTRICTED = "restricted"
    MONITORED = "monitored"
    FORBIDDEN = "forbidden"

@dataclass
class SafetyZone:
    name: str
    zone_type: SafetyZoneType
    boundaries: Dict[str, float]  # x_min, x_max, y_min, y_max, z_min, z_max
    safety_level: SafetyLevel
    description: str = ""
    active: bool = True

@dataclass
class SafetyEvent:
    timestamp: datetime
    event_type: str
    safety_level: SafetyLevel
    description: str
    affected_zones: List[str]
    robot_position: Optional[Position] = None
    io_states: Optional[Dict[str, Any]] = None
    resolved: bool = False

class WorkspaceMonitor:
    """ワークスペース監視システム"""
    
    def __init__(self):
        self.safety_zones: Dict[str, SafetyZone] = {}
        self.current_robot_position: Optional[Position] = None
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self._initialize_default_zones()
    
    def _initialize_default_zones(self):
        """デフォルト安全ゾーン初期化"""
        # 基本ワークスペース
        workspace_limits = config_manager.get_robot_limits().get("workspace_limits", {})
        
        # 作業領域
        work_zone = SafetyZone(
            name="work_area",
            zone_type=SafetyZoneType.MONITORED,
            boundaries={
                "x_min": workspace_limits.get("x_min", -800),
                "x_max": workspace_limits.get("x_max", 800),
                "y_min": workspace_limits.get("y_min", -800),
                "y_max": workspace_limits.get("y_max", 800),
                "z_min": workspace_limits.get("z_min", 0),
                "z_max": workspace_limits.get("z_max", 800)
            },
            safety_level=SafetyLevel.SAFE,
            description="Normal robot work area"
        )
        self.add_safety_zone(work_zone)
        
        # 制限領域（人間の作業エリア）
        human_zone = SafetyZone(
            name="human_work_area",
            zone_type=SafetyZoneType.RESTRICTED,
            boundaries={
                "x_min": -200, "x_max": 200,
                "y_min": 400, "y_max": 800,
                "z_min": 0, "z_max": 2000
            },
            safety_level=SafetyLevel.DANGER,
            description="Human operator work area - restricted access"
        )
        self.add_safety_zone(human_zone)
        
        # 禁止領域（機械の固定部分）
        machine_zone = SafetyZone(
            name="machine_base",
            zone_type=SafetyZoneType.FORBIDDEN,
            boundaries={
                "x_min": -100, "x_max": 100,
                "y_min": -100, "y_max": 100,
                "z_min": -200, "z_max": 50
            },
            safety_level=SafetyLevel.CRITICAL,
            description="Machine base - forbidden zone"
        )
        self.add_safety_zone(machine_zone)
    
    def add_safety_zone(self, zone: SafetyZone):
        """安全ゾーン追加"""
        self.safety_zones[zone.name] = zone
        logger.info(f"Added safety zone: {zone.name} ({zone.zone_type.value})")
    
    def remove_safety_zone(self, zone_name: str) -> bool:
        """安全ゾーン削除"""
        if zone_name in self.safety_zones:
            del self.safety_zones[zone_name]
            logger.info(f"Removed safety zone: {zone_name}")
            return True
        return False
    
    def update_robot_position(self, position: Position):
        """ロボット位置更新"""
        self.current_robot_position = position
    
    def check_position_safety(self, position: Position) -> Dict[str, Any]:
        """位置安全性チェック"""
        safety_result = {
            "is_safe": True,
            "safety_level": SafetyLevel.SAFE,
            "violated_zones": [],
            "warnings": []
        }
        
        for zone_name, zone in self.safety_zones.items():
            if not zone.active:
                continue
            
            if self._position_in_zone(position, zone):
                if zone.zone_type == SafetyZoneType.FORBIDDEN:
                    safety_result["is_safe"] = False
                    safety_result["safety_level"] = SafetyLevel.CRITICAL
                    safety_result["violated_zones"].append(zone_name)
                
                elif zone.zone_type == SafetyZoneType.RESTRICTED:
                    # 制限エリアは条件付きで許可
                    if not self._check_restricted_zone_conditions(zone_name):
                        safety_result["is_safe"] = False
                        safety_result["safety_level"] = SafetyLevel.DANGER
                        safety_result["violated_zones"].append(zone_name)
                
                elif zone.zone_type == SafetyZoneType.MONITORED:
                    safety_result["warnings"].append(f"Entering monitored zone: {zone_name}")
        
        return safety_result
    
    def _position_in_zone(self, position: Position, zone: SafetyZone) -> bool:
        """位置がゾーン内にあるかチェック"""
        bounds = zone.boundaries
        return (
            bounds["x_min"] <= position.x <= bounds["x_max"] and
            bounds["y_min"] <= position.y <= bounds["y_max"] and
            bounds["z_min"] <= position.z <= bounds["z_max"]
        )
    
    def _check_restricted_zone_conditions(self, zone_name: str) -> bool:
        """制限ゾーン進入条件チェック"""
        if zone_name == "human_work_area":
            # 人間作業エリアは人検出がない場合のみ許可
            human_present = io_controller.get_signal_value("HUMAN_PRESENT")
            return not human_present if human_present is not None else True
        
        return True
    
    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Workspace monitoring started")
    
    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Workspace monitoring stopped")
    
    def _monitoring_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                if self.current_robot_position:
                    safety_check = self.check_position_safety(self.current_robot_position)
                    if not safety_check["is_safe"]:
                        self._handle_safety_violation(safety_check)
                
                time.sleep(0.1)  # 100ms間隔
            except Exception as e:
                logger.error(f"Workspace monitoring error: {e}")
                time.sleep(1.0)
    
    def _handle_safety_violation(self, safety_check: Dict[str, Any]):
        """安全違反処理"""
        for zone_name in safety_check["violated_zones"]:
            logger.error(f"Safety violation in zone: {zone_name}")
            
            # エラー記録
            error_recovery_manager.record_error(
                ErrorType.UNKNOWN_ERROR,
                f"Safety zone violation: {zone_name}",
                ErrorSeverity.CRITICAL,
                {"zone": zone_name, "safety_level": safety_check["safety_level"].value}
            )

class IntegratedSafetySystem:
    """統合安全システム"""
    
    def __init__(self):
        self.workspace_monitor = WorkspaceMonitor()
        self.safety_events: List[SafetyEvent] = []
        self.safety_callbacks: List[Callable[[SafetyEvent], None]] = []
        
        self.emergency_stop_active = False
        self.safety_system_enabled = True
        self.last_safety_check = None
        
        # 設定読み込み
        self.safety_config = config_manager.get_safety_config()
        self._initialize_safety_monitoring()
    
    def _initialize_safety_monitoring(self):
        """安全監視初期化"""
        # I/O信号コールバック登録
        io_controller.register_callback("E_STOP", self._handle_emergency_stop)
        io_controller.register_callback("DOOR_CLOSED", self._handle_door_interlock)
        io_controller.register_callback("AIR_PRESSURE_OK", self._handle_air_pressure)
        
        # ワークスペース監視開始
        self.workspace_monitor.start_monitoring()
        
        logger.info("Integrated safety system initialized")
    
    def register_safety_callback(self, callback: Callable[[SafetyEvent], None]):
        """安全イベントコールバック登録"""
        self.safety_callbacks.append(callback)
    
    def check_overall_safety(self) -> Dict[str, Any]:
        """総合安全状態チェック"""
        safety_status = {
            "overall_safe": True,
            "safety_level": SafetyLevel.SAFE,
            "active_violations": [],
            "system_status": {
                "emergency_stop": self.emergency_stop_active,
                "door_closed": io_controller.get_signal_value("DOOR_CLOSED"),
                "air_pressure": io_controller.get_signal_value("AIR_PRESSURE_OK"),
                "human_present": io_controller.get_signal_value("HUMAN_PRESENT")
            },
            "workspace_safe": True
        }
        
        # 緊急停止チェック
        if self.emergency_stop_active:
            safety_status["overall_safe"] = False
            safety_status["safety_level"] = SafetyLevel.CRITICAL
            safety_status["active_violations"].append("Emergency stop active")
        
        # ドアインターロック
        if not io_controller.get_signal_value("DOOR_CLOSED"):
            safety_status["overall_safe"] = False
            safety_status["safety_level"] = SafetyLevel.DANGER
            safety_status["active_violations"].append("Safety door open")
        
        # エア圧力
        if not io_controller.get_signal_value("AIR_PRESSURE_OK"):
            safety_status["overall_safe"] = False
            safety_status["safety_level"] = SafetyLevel.WARNING
            safety_status["active_violations"].append("Air pressure low")
        
        # ワークスペース安全性
        if self.workspace_monitor.current_robot_position:
            workspace_check = self.workspace_monitor.check_position_safety(
                self.workspace_monitor.current_robot_position
            )
            if not workspace_check["is_safe"]:
                safety_status["overall_safe"] = False
                safety_status["workspace_safe"] = False
                safety_status["active_violations"].extend([
                    f"Workspace violation: {zone}" for zone in workspace_check["violated_zones"]
                ])
        
        self.last_safety_check = safety_status
        return safety_status
    
    def is_safe_to_move(self, target_position: Position) -> bool:
        """移動安全確認"""
        if not self.safety_system_enabled:
            return True
        
        # 基本安全条件
        basic_safe = (
            not self.emergency_stop_active and
            io_controller.get_signal_value("DOOR_CLOSED") and
            io_controller.get_signal_value("AIR_PRESSURE_OK")
        )
        
        if not basic_safe:
            return False
        
        # 目標位置の安全性
        position_check = self.workspace_monitor.check_position_safety(target_position)
        return position_check["is_safe"]
    
    def validate_trajectory_safety(self, trajectory_points: List[Position]) -> Dict[str, Any]:
        """軌道安全検証"""
        validation_result = {
            "is_safe": True,
            "unsafe_points": [],
            "warnings": [],
            "critical_violations": []
        }
        
        for i, position in enumerate(trajectory_points):
            safety_check = self.workspace_monitor.check_position_safety(position)
            
            if not safety_check["is_safe"]:
                validation_result["is_safe"] = False
                validation_result["unsafe_points"].append(i)
                
                if safety_check["safety_level"] == SafetyLevel.CRITICAL:
                    validation_result["critical_violations"].extend(
                        safety_check["violated_zones"]
                    )
            
            validation_result["warnings"].extend(safety_check["warnings"])
        
        return validation_result
    
    def _handle_emergency_stop(self, signal):
        """緊急停止処理"""
        if signal.value:
            self.emergency_stop_active = True
            self._create_safety_event(
                "EMERGENCY_STOP",
                SafetyLevel.CRITICAL,
                "Emergency stop button pressed",
                ["all_zones"]
            )
            logger.critical("EMERGENCY STOP ACTIVATED")
        else:
            self.emergency_stop_active = False
            logger.info("Emergency stop released")
    
    def _handle_door_interlock(self, signal):
        """ドアインターロック処理"""
        if not signal.value:  # ドアが開いた
            self._create_safety_event(
                "DOOR_OPEN",
                SafetyLevel.DANGER,
                "Safety door opened",
                ["work_area"]
            )
            logger.warning("Safety door opened")
    
    def _handle_air_pressure(self, signal):
        """エア圧力処理"""
        if not signal.value:  # 圧力低下
            self._create_safety_event(
                "AIR_PRESSURE_LOW",
                SafetyLevel.WARNING,
                "Air pressure below minimum",
                ["work_area"]
            )
            logger.warning("Air pressure low")
    
    def _create_safety_event(self, event_type: str, safety_level: SafetyLevel, 
                           description: str, affected_zones: List[str]):
        """安全イベント作成"""
        event = SafetyEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            safety_level=safety_level,
            description=description,
            affected_zones=affected_zones,
            robot_position=self.workspace_monitor.current_robot_position,
            io_states=io_controller.get_all_signals_status()
        )
        
        self.safety_events.append(event)
        
        # 履歴を最新100件に制限
        if len(self.safety_events) > 100:
            self.safety_events = self.safety_events[-100:]
        
        # コールバック実行
        for callback in self.safety_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Safety callback error: {e}")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """安全状態取得"""
        return {
            "system_enabled": self.safety_system_enabled,
            "emergency_stop_active": self.emergency_stop_active,
            "overall_safety": self.check_overall_safety(),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "level": event.safety_level.value,
                    "description": event.description,
                    "resolved": event.resolved
                }
                for event in self.safety_events[-10:]
            ],
            "active_zones": len([z for z in self.workspace_monitor.safety_zones.values() if z.active])
        }
    
    def acknowledge_safety_event(self, event_index: int) -> bool:
        """安全イベント確認"""
        if 0 <= event_index < len(self.safety_events):
            self.safety_events[event_index].resolved = True
            logger.info(f"Safety event {event_index} acknowledged")
            return True
        return False
    
    def enable_safety_system(self):
        """安全システム有効化"""
        self.safety_system_enabled = True
        logger.info("Safety system enabled")
    
    def disable_safety_system(self):
        """安全システム無効化（テスト用）"""
        logger.warning("Safety system disabled - FOR TESTING ONLY")
        self.safety_system_enabled = False
    
    def shutdown(self):
        """安全システム終了"""
        self.workspace_monitor.stop_monitoring()
        logger.info("Integrated safety system shutdown")

# グローバルインスタンス
safety_system = IntegratedSafetySystem()

def initialize_safety_system() -> bool:
    """安全システム初期化"""
    try:
        # 追加の初期化処理があれば実行
        logger.info("Safety system initialization completed")
        return True
    except Exception as e:
        logger.error(f"Safety system initialization failed: {e}")
        return False

def is_safe_to_operate() -> bool:
    """動作安全確認（簡易インターフェース）"""
    return safety_system.check_overall_safety()["overall_safe"]

def validate_position_safety(position: Position) -> bool:
    """位置安全確認（簡易インターフェース）"""
    return safety_system.workspace_monitor.check_position_safety(position)["is_safe"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    if initialize_safety_system():
        logger.info("Safety system test started")
        
        # テスト用位置
        test_positions = [
            Position(100, 100, 100, 0, 0, 0),  # 安全領域
            Position(0, 500, 100, 0, 0, 0),    # 制限領域
            Position(0, 0, -100, 0, 0, 0)      # 禁止領域
        ]
        
        for i, pos in enumerate(test_positions):
            safety_check = safety_system.workspace_monitor.check_position_safety(pos)
            logger.info(f"Position {i+1}: {pos.to_list()} - Safe: {safety_check['is_safe']}")
        
        # 安全状態表示
        status = safety_system.get_safety_status()
        logger.info(f"Safety system status: {status}")
        
        # 5秒間監視テスト
        time.sleep(5)
        
        safety_system.shutdown()
    else:
        logger.error("Safety system initialization failed")