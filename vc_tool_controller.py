import socket
import json
import threading
import time
import logging
from typing import Dict, Any, Optional
from enum import Enum
from error_recovery import (
    error_recovery_manager,
    with_retry_and_circuit_breaker, 
    ErrorType,
    ErrorSeverity
)

logger = logging.getLogger(__name__)

class ToolType(Enum):
    WELDING_GUN = "welding_gun"
    GRIPPER = "gripper"

class ToolState(Enum):
    READY = "READY"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    GRIPPING = "GRIPPING"
    WELDING = "WELDING"
    ERROR = "ERROR"

class WeldingGunController:
    """スポット溶接ガン制御クラス"""
    
    def __init__(self, tool_name: str = "welding_gun_1"):
        self.tool_name = tool_name
        self.state = ToolState.READY
        self.tcp_offset = [0.0, 0.0, 150.0, 0.0, 0.0, 0.0]
        self.electrode_position = 0.0
        self.pressure = 0.0
        self.current_settings = {
            "weld_time": 1.0,
            "pressure": 100.0,
            "current": 5000.0,
            "electrode_force": 200.0
        }
        self.electrode_wear = 0.0
        self.cycle_count = 0
        
        try:
            from vcScript import *
            self.vc_app = getApplication()
            self.tool_component = None
            self._initialize_welding_gun()
            logger.info("Welding gun VC Script environment initialized")
        except ImportError:
            logger.warning("vcScript not available - running in simulation mode")
            self.vc_app = None
            self.tool_component = None
    
    def _initialize_welding_gun(self):
        """溶接ガンコンポーネント初期化"""
        try:
            if self.vc_app:
                simulation = self.vc_app.Simulation
                components = simulation.Components
                
                for component in components:
                    if "weld" in component.Name.lower() or "gun" in component.Name.lower():
                        self.tool_component = component
                        logger.info(f"Welding gun component found: {component.Name}")
                        break
                
        except Exception as e:
            logger.error(f"Welding gun initialization error: {e}")
    
    def execute_welding(self, weld_parameters: Dict[str, Any]) -> bool:
        """溶接実行"""
        try:
            self.state = ToolState.WELDING
            
            weld_time = weld_parameters.get("weld_time", self.current_settings["weld_time"])
            pressure = weld_parameters.get("pressure", self.current_settings["pressure"])
            current = weld_parameters.get("current", self.current_settings["current"])
            electrode_force = weld_parameters.get("electrode_force", self.current_settings["electrode_force"])
            
            logger.info(f"Starting welding: time={weld_time}s, pressure={pressure}kPa, current={current}A")
            
            self._apply_electrode_pressure(electrode_force)
            
            time.sleep(0.2)
            
            self._execute_weld_cycle(weld_time, current)
            
            self._release_electrode_pressure()
            
            self.cycle_count += 1
            self.electrode_wear += 0.001
            
            self.state = ToolState.READY
            
            logger.info(f"Welding completed. Cycle count: {self.cycle_count}")
            return True
            
        except Exception as e:
            logger.error(f"Welding execution error: {e}")
            self.state = ToolState.ERROR
            return False
    
    def _apply_electrode_pressure(self, force: float):
        """電極加圧"""
        if self.tool_component:
            try:
                pass
            except Exception as e:
                logger.error(f"Electrode pressure application error: {e}")
        
        self.pressure = force
        time.sleep(0.1)
    
    def _execute_weld_cycle(self, weld_time: float, current: float):
        """溶接サイクル実行"""
        logger.info(f"Executing weld cycle: {weld_time}s at {current}A")
        time.sleep(weld_time)
    
    def _release_electrode_pressure(self):
        """電極開放"""
        if self.tool_component:
            try:
                pass
            except Exception as e:
                logger.error(f"Electrode pressure release error: {e}")
        
        self.pressure = 0.0
        time.sleep(0.1)
    
    def check_electrode_wear(self) -> Dict[str, Any]:
        """電極摩耗チェック"""
        wear_limit = 0.5
        wear_status = "OK" if self.electrode_wear < wear_limit else "WARNING"
        
        return {
            "wear_amount": self.electrode_wear,
            "wear_limit": wear_limit,
            "status": wear_status,
            "cycles_remaining": int((wear_limit - self.electrode_wear) / 0.001)
        }
    
    def calibrate_tcp(self):
        """TCP キャリブレーション"""
        logger.info("TCP calibration started")
        time.sleep(1.0)
        logger.info("TCP calibration completed")
    
    def get_status(self) -> Dict[str, Any]:
        """溶接ガンステータス取得"""
        return {
            "tool_name": self.tool_name,
            "tool_type": ToolType.WELDING_GUN.value,
            "state": self.state.value,
            "tcp_offset": self.tcp_offset,
            "electrode_position": self.electrode_position,
            "current_pressure": self.pressure,
            "current_settings": self.current_settings,
            "electrode_wear": self.electrode_wear,
            "cycle_count": self.cycle_count,
            "wear_check": self.check_electrode_wear()
        }

class GripperController:
    """クランプ式グリッパー制御クラス"""
    
    def __init__(self, tool_name: str = "gripper_1"):
        self.tool_name = tool_name
        self.state = ToolState.OPEN
        self.tcp_offset = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
        self.grip_position = 0.0
        self.grip_force = 0.0
        self.max_grip_force = 100.0
        self.grip_range = [0.0, 80.0]
        self.grip_speed = 50.0
        self.force_sensor_reading = 0.0
        self.position_sensor_reading = 0.0
        
        try:
            from vcScript import *
            self.vc_app = getApplication()
            self.tool_component = None
            self._initialize_gripper()
            logger.info("Gripper VC Script environment initialized")
        except ImportError:
            logger.warning("vcScript not available - running in simulation mode")
            self.vc_app = None
            self.tool_component = None
    
    def _initialize_gripper(self):
        """グリッパーコンポーネント初期化"""
        try:
            if self.vc_app:
                simulation = self.vc_app.Simulation
                components = simulation.Components
                
                for component in components:
                    if "grip" in component.Name.lower() or "clamp" in component.Name.lower():
                        self.tool_component = component
                        logger.info(f"Gripper component found: {component.Name}")
                        break
                
        except Exception as e:
            logger.error(f"Gripper initialization error: {e}")
    
    def open_gripper(self, speed: Optional[float] = None) -> bool:
        """グリッパー開放"""
        try:
            open_speed = speed or self.grip_speed
            
            logger.info(f"Opening gripper at speed {open_speed}%")
            
            if self.tool_component:
                pass
            
            start_position = self.grip_position
            target_position = self.grip_range[1]
            
            steps = 20
            for i in range(steps):
                self.grip_position = start_position + (target_position - start_position) * (i + 1) / steps
                time.sleep(0.05)
            
            self.grip_force = 0.0
            self.state = ToolState.OPEN
            
            logger.info("Gripper opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Gripper open error: {e}")
            self.state = ToolState.ERROR
            return False
    
    def close_gripper(self, target_force: Optional[float] = None, speed: Optional[float] = None) -> bool:
        """グリッパー閉じる"""
        try:
            grip_force = target_force or 50.0
            close_speed = speed or self.grip_speed
            
            if grip_force > self.max_grip_force:
                grip_force = self.max_grip_force
            
            logger.info(f"Closing gripper with force {grip_force}N at speed {close_speed}%")
            
            if self.tool_component:
                pass
            
            start_position = self.grip_position
            target_position = self.grip_range[0]
            
            steps = 20
            for i in range(steps):
                self.grip_position = start_position + (target_position - start_position) * (i + 1) / steps
                time.sleep(0.05)
                
                if self._detect_workpiece():
                    break
            
            self.grip_force = grip_force
            
            if self._verify_grip():
                self.state = ToolState.GRIPPING
                logger.info("Workpiece gripped successfully")
                return True
            else:
                self.state = ToolState.CLOSED
                logger.warning("No workpiece detected or grip failed")
                return False
                
        except Exception as e:
            logger.error(f"Gripper close error: {e}")
            self.state = ToolState.ERROR
            return False
    
    def _detect_workpiece(self) -> bool:
        """ワーク検出"""
        self.force_sensor_reading = self.grip_force * 0.8 + (time.time() % 1) * 5
        return self.force_sensor_reading > 10.0
    
    def _verify_grip(self) -> bool:
        """把持確認"""
        force_ok = self.force_sensor_reading > 5.0
        position_ok = self.grip_position < self.grip_range[1] * 0.8
        
        return force_ok and position_ok
    
    def release_workpiece(self) -> bool:
        """ワーク開放"""
        return self.open_gripper()
    
    def adjust_grip_force(self, new_force: float) -> bool:
        """把持力調整"""
        try:
            if new_force > self.max_grip_force:
                new_force = self.max_grip_force
            
            self.grip_force = new_force
            logger.info(f"Grip force adjusted to {new_force}N")
            return True
            
        except Exception as e:
            logger.error(f"Grip force adjustment error: {e}")
            return False
    
    def calibrate_tcp(self):
        """TCP キャリブレーション"""
        logger.info("Gripper TCP calibration started")
        time.sleep(1.0)
        logger.info("Gripper TCP calibration completed")
    
    def get_status(self) -> Dict[str, Any]:
        """グリッパーステータス取得"""
        return {
            "tool_name": self.tool_name,
            "tool_type": ToolType.GRIPPER.value,
            "state": self.state.value,
            "tcp_offset": self.tcp_offset,
            "grip_position": self.grip_position,
            "grip_force": self.grip_force,
            "max_grip_force": self.max_grip_force,
            "grip_range": self.grip_range,
            "grip_speed": self.grip_speed,
            "force_sensor_reading": self.force_sensor_reading,
            "position_sensor_reading": self.position_sensor_reading,
            "workpiece_detected": self._detect_workpiece()
        }

class ToolController:
    """統合ツールコントローラー"""
    
    def __init__(self, tool_type: ToolType, tool_name: str = None, 
                 server_host: str = "localhost", server_port: int = 8888):
        self.tool_type = tool_type
        self.tool_name = tool_name or f"{tool_type.value}_1"
        self.server_host = server_host
        self.server_port = server_port
        
        self.tcp_client = None
        self.is_connected = False
        self.message_handlers = {}
        
        if tool_type == ToolType.WELDING_GUN:
            self.tool = WeldingGunController(self.tool_name)
        elif tool_type == ToolType.GRIPPER:
            self.tool = GripperController(self.tool_name)
        else:
            raise ValueError(f"Unsupported tool type: {tool_type}")
        
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """メッセージハンドラー設定"""
        self.message_handlers.update({
            "tool_control": self._handle_tool_control,
            "tool_status": self._handle_tool_status,
            "tool_calibration": self._handle_tool_calibration,
            "weld_execute": self._handle_weld_execute,
            "grip_control": self._handle_grip_control,
            "force_adjustment": self._handle_force_adjustment
        })
    
    @with_retry_and_circuit_breaker("tool_controller", ErrorType.CONNECTION_TIMEOUT)
    def connect_external_app(self) -> bool:
        """外部アプリケーション接続"""
        try:
            self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_client.settimeout(10.0)
            self.tcp_client.connect((self.server_host, self.server_port))
            self.is_connected = True
            
            logger.info(f"Tool controller {self.tool_name} connected to {self.server_host}:{self.server_port}")
            
            receive_thread = threading.Thread(target=self._receive_messages)
            receive_thread.daemon = True
            receive_thread.start()
            
            self._send_connection_status()
            return True
            
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            error_recovery_manager.record_error(
                ErrorType.CONNECTION_TIMEOUT if isinstance(e, socket.timeout) else ErrorType.NETWORK_ERROR,
                f"Tool controller connection failed: {str(e)}",
                ErrorSeverity.HIGH,
                {"tool_name": self.tool_name, "host": self.server_host, "port": self.server_port}
            )
            self.is_connected = False
            raise e
        except Exception as e:
            error_recovery_manager.record_error(
                ErrorType.UNKNOWN_ERROR,
                f"Tool controller connection error: {str(e)}",
                ErrorSeverity.CRITICAL,
                {"tool_name": self.tool_name}
            )
            self.is_connected = False
            raise e
    
    def _receive_messages(self):
        """メッセージ受信ループ"""
        buffer = ""
        try:
            while self.is_connected:
                data = self.tcp_client.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_message(line.strip())
                        
        except Exception as e:
            if self.is_connected:
                logger.error(f"Message receive error: {e}")
        finally:
            self.is_connected = False
    
    def _process_message(self, message: str):
        """メッセージ処理"""
        try:
            msg_data = json.loads(message)
            command_type = msg_data.get("command_type")
            target = msg_data.get("target_component")
            
            if target == self.tool_name or target == "all":
                if command_type in self.message_handlers:
                    response = self.message_handlers[command_type](msg_data)
                    if msg_data.get("response_required", False):
                        self._send_response(response)
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _handle_tool_control(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """ツール制御コマンド処理"""
        try:
            parameters = message_data.get("parameters", {})
            command = parameters.get("tool_command", "")
            
            if self.tool_type == ToolType.GRIPPER:
                if command == "open":
                    result = self.tool.open_gripper()
                elif command == "close":
                    grip_force = parameters.get("grip_force", 50.0)
                    result = self.tool.close_gripper(grip_force)
                else:
                    return self._create_error_response(message_data, f"Unknown gripper command: {command}")
            
            elif self.tool_type == ToolType.WELDING_GUN:
                if command == "weld":
                    result = self.tool.execute_welding(parameters)
                else:
                    return self._create_error_response(message_data, f"Unknown welding gun command: {command}")
            
            else:
                return self._create_error_response(message_data, f"Unsupported tool type: {self.tool_type}")
            
            if result:
                return self._create_success_response(
                    message_data, 
                    {"message": f"Command '{command}' executed successfully"}
                )
            else:
                return self._create_error_response(message_data, f"Command '{command}' failed")
                
        except Exception as e:
            logger.error(f"Tool control error: {e}")
            return self._create_error_response(message_data, str(e))
    
    def _handle_tool_status(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """ツールステータス要求処理"""
        status_data = self.tool.get_status()
        return self._create_success_response(message_data, status_data)
    
    def _handle_tool_calibration(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """ツールキャリブレーション処理"""
        try:
            self.tool.calibrate_tcp()
            return self._create_success_response(
                message_data, 
                {"message": "TCP calibration completed"}
            )
        except Exception as e:
            return self._create_error_response(message_data, str(e))
    
    def _handle_weld_execute(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """溶接実行処理"""
        if self.tool_type != ToolType.WELDING_GUN:
            return self._create_error_response(message_data, "Not a welding gun")
        
        try:
            parameters = message_data.get("parameters", {})
            result = self.tool.execute_welding(parameters)
            
            if result:
                return self._create_success_response(
                    message_data, 
                    {"message": "Welding executed successfully", "cycle_count": self.tool.cycle_count}
                )
            else:
                return self._create_error_response(message_data, "Welding execution failed")
                
        except Exception as e:
            return self._create_error_response(message_data, str(e))
    
    def _handle_grip_control(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """グリップ制御処理"""
        if self.tool_type != ToolType.GRIPPER:
            return self._create_error_response(message_data, "Not a gripper")
        
        try:
            parameters = message_data.get("parameters", {})
            action = parameters.get("action", "")
            
            if action == "grip":
                force = parameters.get("force", 50.0)
                result = self.tool.close_gripper(force)
            elif action == "release":
                result = self.tool.open_gripper()
            else:
                return self._create_error_response(message_data, f"Unknown grip action: {action}")
            
            return self._create_success_response(
                message_data, 
                {"message": f"Grip action '{action}' completed", "result": result}
            )
            
        except Exception as e:
            return self._create_error_response(message_data, str(e))
    
    def _handle_force_adjustment(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """力調整処理"""
        if self.tool_type != ToolType.GRIPPER:
            return self._create_error_response(message_data, "Not a gripper")
        
        try:
            parameters = message_data.get("parameters", {})
            new_force = parameters.get("force", 50.0)
            
            result = self.tool.adjust_grip_force(new_force)
            return self._create_success_response(
                message_data, 
                {"message": f"Force adjusted to {new_force}N", "result": result}
            )
            
        except Exception as e:
            return self._create_error_response(message_data, str(e))
    
    def _send_connection_status(self):
        """接続状態通知"""
        status_message = {
            "message_id": f"{int(time.time() * 1000)}_tool_connection",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "command_type": "connection_status",
            "target_component": "controller",
            "parameters": {
                "component_name": self.tool_name,
                "component_type": "tool_controller",
                "tool_type": self.tool_type.value,
                "status": "connected"
            }
        }
        
        self._send_message(status_message)
    
    def _send_message(self, message_data: Dict[str, Any]) -> bool:
        """メッセージ送信"""
        if self.is_connected and self.tcp_client:
            try:
                message_json = json.dumps(message_data) + '\n'
                self.tcp_client.send(message_json.encode('utf-8'))
                return True
            except Exception as e:
                logger.error(f"Message send error: {e}")
                return False
        return False
    
    def _send_response(self, response_data: Dict[str, Any]):
        """レスポンス送信"""
        self._send_message(response_data)
    
    def _create_success_response(self, original_message: Dict[str, Any], 
                               data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """成功レスポンス作成"""
        return {
            "message_id": f"response_{original_message.get('message_id')}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "response_to": original_message.get("message_id"),
            "status": "success",
            "component_name": self.tool_name,
            "data": data or {}
        }
    
    def _create_error_response(self, original_message: Dict[str, Any], 
                              error_message: str) -> Dict[str, Any]:
        """エラーレスポンス作成"""
        return {
            "message_id": f"response_{original_message.get('message_id')}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "response_to": original_message.get("message_id"),
            "status": "error",
            "component_name": self.tool_name,
            "error": error_message
        }
    
    def start_tool_controller(self):
        """ツールコントローラー開始"""
        logger.info(f"Starting tool controller: {self.tool_name} ({self.tool_type.value})")
        
        if self.connect_external_app():
            logger.info(f"Tool controller {self.tool_name} started successfully")
            return True
        else:
            logger.error(f"Failed to start tool controller {self.tool_name}")
            return False
    
    def stop_tool_controller(self):
        """ツールコントローラー停止"""
        self.is_connected = False
        
        if self.tcp_client:
            self.tcp_client.close()
        
        logger.info(f"Tool controller {self.tool_name} stopped")

def initialize_welding_gun_controller(tool_name: str = "welding_gun_1"):
    """溶接ガンコントローラー初期化関数"""
    controller = ToolController(ToolType.WELDING_GUN, tool_name)
    if controller.start_tool_controller():
        return controller
    else:
        return None

def initialize_gripper_controller(tool_name: str = "gripper_1"):
    """グリッパーコントローラー初期化関数"""
    controller = ToolController(ToolType.GRIPPER, tool_name)
    if controller.start_tool_controller():
        return controller
    else:
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "gripper":
            tool_controller = ToolController(ToolType.GRIPPER)
        elif sys.argv[1] == "welding":
            tool_controller = ToolController(ToolType.WELDING_GUN)
        else:
            print("Usage: python vc_tool_controller.py [gripper|welding]")
            sys.exit(1)
    else:
        tool_controller = ToolController(ToolType.GRIPPER)
    
    try:
        tool_controller.start_tool_controller()
        
        while tool_controller.is_connected:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Stopping tool controller...")
        tool_controller.stop_tool_controller()