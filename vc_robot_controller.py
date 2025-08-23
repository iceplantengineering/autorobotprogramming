import socket
import json
import threading
import time
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class RobotState(Enum):
    READY = "READY"
    MOVING = "MOVING"
    WORKING = "WORKING"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class RobotController:
    """
    Visual Components Robot Component Script
    このクラスはVCのロボットコンポーネント内で実行されるPythonスクリプト用
    """
    
    def __init__(self, component_name: str = "robot_1", server_host: str = "localhost", server_port: int = 8888):
        self.component_name = component_name
        self.server_host = server_host
        self.server_port = server_port
        
        self.tcp_client = None
        self.is_connected = False
        self.current_program = []
        self.execution_state = RobotState.READY
        self.current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.target_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.message_handlers = {}
        self._setup_message_handlers()
        
        self.execution_thread = None
        self.stop_execution = False
        
        try:
            from vcScript import *
            self.vc_app = getApplication()
            self.robot_component = None
            self._initialize_vc_robot()
            logger.info("VC Script environment initialized successfully")
        except ImportError:
            logger.warning("vcScript not available - running in simulation mode")
            self.vc_app = None
            self.robot_component = None
    
    def _initialize_vc_robot(self):
        """Visual Componentsのロボットコンポーネント初期化"""
        try:
            if self.vc_app:
                simulation = self.vc_app.Simulation
                components = simulation.Components
                
                for component in components:
                    if hasattr(component, 'Joints') and len(component.Joints) >= 6:
                        self.robot_component = component
                        logger.info(f"Robot component found: {component.Name}")
                        break
                
                if not self.robot_component:
                    logger.error("No 6-axis robot component found in simulation")
                    
        except Exception as e:
            logger.error(f"Robot initialization error: {e}")
    
    def _setup_message_handlers(self):
        """メッセージハンドラーの設定"""
        self.message_handlers.update({
            "robot_move": self._handle_robot_move,
            "program_execute": self._handle_program_execute,
            "program_pause": self._handle_program_pause,
            "program_stop": self._handle_program_stop,
            "status_request": self._handle_status_request,
            "emergency_stop": self._handle_emergency_stop,
            "home_position": self._handle_home_position,
            "get_current_position": self._handle_get_current_position
        })
    
    def connect_external_app(self) -> bool:
        """外部アプリケーションとの接続確立"""
        try:
            self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_client.connect((self.server_host, self.server_port))
            self.is_connected = True
            
            logger.info(f"Connected to external app at {self.server_host}:{self.server_port}")
            
            receive_thread = threading.Thread(target=self._receive_messages)
            receive_thread.daemon = True
            receive_thread.start()
            
            self._send_connection_status()
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False
    
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
            logger.info("Disconnected from external app")
    
    def _process_message(self, message: str):
        """受信メッセージの処理"""
        try:
            msg_data = json.loads(message)
            command_type = msg_data.get("command_type")
            target = msg_data.get("target_component")
            
            if target == self.component_name or target == "all":
                if command_type in self.message_handlers:
                    response = self.message_handlers[command_type](msg_data)
                    if msg_data.get("response_required", False):
                        self._send_response(response)
                else:
                    logger.warning(f"Unknown command type: {command_type}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _handle_robot_move(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """ロボット移動コマンド処理"""
        try:
            parameters = message_data.get("parameters", {})
            position = parameters.get("position", [])
            speed = parameters.get("speed", 100)
            
            if len(position) != 6:
                return self._create_error_response(message_data, "Invalid position format")
            
            self.target_position = position
            result = self._move_robot(position, speed)
            
            if result:
                return self._create_success_response(
                    message_data, 
                    {"message": "Move completed", "position": self.current_position}
                )
            else:
                return self._create_error_response(message_data, "Move failed")
                
        except Exception as e:
            logger.error(f"Robot move error: {e}")
            return self._create_error_response(message_data, str(e))
    
    def _handle_program_execute(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """プログラム実行コマンド処理"""
        try:
            parameters = message_data.get("parameters", {})
            program = parameters.get("program", [])
            
            self.current_program = program
            
            if self.execution_thread and self.execution_thread.is_alive():
                return self._create_error_response(message_data, "Program already executing")
            
            self.stop_execution = False
            self.execution_thread = threading.Thread(target=self._execute_program)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            
            return self._create_success_response(
                message_data, 
                {"message": f"Program execution started with {len(program)} steps"}
            )
            
        except Exception as e:
            logger.error(f"Program execution error: {e}")
            return self._create_error_response(message_data, str(e))
    
    def _handle_program_pause(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """プログラム一時停止処理"""
        self.execution_state = RobotState.READY
        return self._create_success_response(message_data, {"message": "Program paused"})
    
    def _handle_program_stop(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """プログラム停止処理"""
        self.stop_execution = True
        self.execution_state = RobotState.READY
        return self._create_success_response(message_data, {"message": "Program stopped"})
    
    def _handle_status_request(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """ステータス要求処理"""
        status_data = {
            "component_name": self.component_name,
            "state": self.execution_state.value,
            "current_position": self.current_position,
            "target_position": self.target_position,
            "is_connected": self.is_connected,
            "program_loaded": len(self.current_program) > 0,
            "program_steps": len(self.current_program)
        }
        
        return self._create_success_response(message_data, status_data)
    
    def _handle_emergency_stop(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """緊急停止処理"""
        self.stop_execution = True
        self.execution_state = RobotState.EMERGENCY_STOP
        
        if self.robot_component:
            try:
                self.robot_component.stop()
            except:
                pass
        
        return self._create_success_response(message_data, {"message": "Emergency stop activated"})
    
    def _handle_home_position(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """ホームポジション移動処理"""
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = self._move_robot(home_position, 50)
        
        if result:
            return self._create_success_response(
                message_data, 
                {"message": "Moved to home position", "position": self.current_position}
            )
        else:
            return self._create_error_response(message_data, "Failed to move to home position")
    
    def _handle_get_current_position(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """現在位置取得処理"""
        self._update_current_position()
        return self._create_success_response(
            message_data, 
            {"current_position": self.current_position}
        )
    
    def _move_robot(self, position: List[float], speed: int) -> bool:
        """ロボット移動実行"""
        try:
            self.execution_state = RobotState.MOVING
            
            if self.robot_component:
                joints = self.robot_component.Joints
                if len(joints) >= 6:
                    for i, angle in enumerate(position[:6]):
                        joints[i].CurrentValue = angle
                    
                    self.robot_component.update()
                    time.sleep(0.1)
            else:
                time.sleep(1.0)
            
            self.current_position = position[:]
            self.execution_state = RobotState.READY
            
            logger.info(f"Robot moved to position: {position} at speed {speed}%")
            return True
            
        except Exception as e:
            logger.error(f"Robot move execution error: {e}")
            self.execution_state = RobotState.ERROR
            return False
    
    def _execute_program(self):
        """プログラム実行ループ"""
        try:
            self.execution_state = RobotState.WORKING
            
            for step_index, step in enumerate(self.current_program):
                if self.stop_execution:
                    break
                
                step_type = step.get("type", "")
                
                if step_type == "move":
                    position = step.get("position", [])
                    speed = step.get("speed", 100)
                    
                    if not self._move_robot(position, speed):
                        logger.error(f"Failed to execute move step {step_index}")
                        break
                
                elif step_type == "tool_control":
                    self._send_tool_command(step)
                
                elif step_type == "weld":
                    self._send_weld_command(step)
                
                else:
                    logger.warning(f"Unknown step type: {step_type}")
                
                time.sleep(0.1)
            
            if not self.stop_execution:
                self.execution_state = RobotState.READY
                logger.info("Program execution completed")
            else:
                logger.info("Program execution stopped")
                
        except Exception as e:
            logger.error(f"Program execution error: {e}")
            self.execution_state = RobotState.ERROR
    
    def _send_tool_command(self, step: Dict[str, Any]):
        """ツール制御コマンド送信"""
        tool_message = {
            "message_id": f"{int(time.time() * 1000)}_tool_command",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "command_type": "tool_control",
            "target_component": "tool_1",
            "parameters": {
                "tool_command": step.get("command", ""),
                "grip_force": step.get("grip_force", 50.0)
            }
        }
        
        self._send_message(tool_message)
    
    def _send_weld_command(self, step: Dict[str, Any]):
        """溶接コマンド送信"""
        weld_message = {
            "message_id": f"{int(time.time() * 1000)}_weld_command",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "command_type": "weld_execute",
            "target_component": "welder_1",
            "parameters": {
                "weld_time": step.get("weld_time", 1.0),
                "pressure": step.get("pressure", 100.0),
                "current": step.get("current", 5000.0),
                "electrode_force": step.get("electrode_force", 200.0)
            }
        }
        
        self._send_message(weld_message)
    
    def _update_current_position(self):
        """現在位置更新"""
        if self.robot_component:
            try:
                joints = self.robot_component.Joints
                if len(joints) >= 6:
                    self.current_position = [joint.CurrentValue for joint in joints[:6]]
            except Exception as e:
                logger.error(f"Position update error: {e}")
    
    def _send_connection_status(self):
        """接続状態通知"""
        status_message = {
            "message_id": f"{int(time.time() * 1000)}_connection_status",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "command_type": "connection_status",
            "target_component": "controller",
            "parameters": {
                "component_name": self.component_name,
                "status": "connected",
                "robot_type": "6_axis_robot"
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
            "component_name": self.component_name,
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
            "component_name": self.component_name,
            "error": error_message
        }
    
    def start_robot_controller(self):
        """ロボットコントローラー開始"""
        logger.info(f"Starting robot controller: {self.component_name}")
        
        if self.connect_external_app():
            logger.info("Robot controller started successfully")
            return True
        else:
            logger.error("Failed to start robot controller")
            return False
    
    def stop_robot_controller(self):
        """ロボットコントローラー停止"""
        self.stop_execution = True
        self.is_connected = False
        
        if self.tcp_client:
            self.tcp_client.close()
        
        logger.info("Robot controller stopped")

def initialize_robot_controller():
    """VC環境でのロボットコントローラー初期化関数"""
    controller = RobotController()
    if controller.start_robot_controller():
        return controller
    else:
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    robot_controller = RobotController()
    
    try:
        robot_controller.start_robot_controller()
        
        while robot_controller.is_connected:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Stopping robot controller...")
        robot_controller.stop_robot_controller()