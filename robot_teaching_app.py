import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from tcp_communication import TCPServer, MessageBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ApplicationType(Enum):
    SPOT_WELDING = "spot_welding"
    HANDLING = "handling"

class RobotState(Enum):
    READY = "READY"
    WORKING = "WORKING"
    ERROR = "ERROR"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"

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

@dataclass
class WeldPoint:
    position: Position
    weld_time: float
    pressure: float
    current: float
    electrode_force: float

@dataclass
class HandlingPoint:
    pick_position: Position
    place_position: Position
    grip_force: float
    approach_speed: int
    work_speed: int

class TrajectoryGenerator:
    def __init__(self):
        self.safety_height = 50.0
        
    def generate_welding_trajectory(self, weld_points: List[WeldPoint]) -> List[Dict[str, Any]]:
        trajectory = []
        
        for i, point in enumerate(weld_points):
            approach_pos = Position(
                point.position.x, point.position.y, 
                point.position.z + self.safety_height,
                point.position.rx, point.position.ry, point.position.rz
            )
            
            trajectory.extend([
                {
                    "type": "move",
                    "position": approach_pos.to_list(),
                    "speed": 50,
                    "description": f"Approach weld point {i+1}"
                },
                {
                    "type": "move",
                    "position": point.position.to_list(),
                    "speed": 10,
                    "description": f"Move to weld point {i+1}"
                },
                {
                    "type": "weld",
                    "weld_time": point.weld_time,
                    "pressure": point.pressure,
                    "current": point.current,
                    "electrode_force": point.electrode_force,
                    "description": f"Execute welding at point {i+1}"
                },
                {
                    "type": "move",
                    "position": approach_pos.to_list(),
                    "speed": 30,
                    "description": f"Retract from weld point {i+1}"
                }
            ])
            
        return trajectory
        
    def generate_handling_trajectory(self, handling_points: List[HandlingPoint]) -> List[Dict[str, Any]]:
        trajectory = []
        
        for i, point in enumerate(handling_points):
            pick_approach = Position(
                point.pick_position.x, point.pick_position.y,
                point.pick_position.z + self.safety_height,
                point.pick_position.rx, point.pick_position.ry, point.pick_position.rz
            )
            
            place_approach = Position(
                point.place_position.x, point.place_position.y,
                point.place_position.z + self.safety_height,
                point.place_position.rx, point.place_position.ry, point.place_position.rz
            )
            
            trajectory.extend([
                {
                    "type": "move",
                    "position": pick_approach.to_list(),
                    "speed": point.approach_speed,
                    "description": f"Approach pick position {i+1}"
                },
                {
                    "type": "tool_control",
                    "command": "open",
                    "description": "Open gripper"
                },
                {
                    "type": "move",
                    "position": point.pick_position.to_list(),
                    "speed": point.work_speed,
                    "description": f"Move to pick position {i+1}"
                },
                {
                    "type": "tool_control",
                    "command": "close",
                    "grip_force": point.grip_force,
                    "description": "Close gripper and pick workpiece"
                },
                {
                    "type": "move",
                    "position": pick_approach.to_list(),
                    "speed": point.work_speed,
                    "description": "Lift workpiece"
                },
                {
                    "type": "move",
                    "position": place_approach.to_list(),
                    "speed": point.approach_speed,
                    "description": f"Move to place approach {i+1}"
                },
                {
                    "type": "move",
                    "position": point.place_position.to_list(),
                    "speed": point.work_speed,
                    "description": f"Move to place position {i+1}"
                },
                {
                    "type": "tool_control",
                    "command": "open",
                    "description": "Open gripper and release workpiece"
                },
                {
                    "type": "move",
                    "position": place_approach.to_list(),
                    "speed": point.work_speed,
                    "description": "Retract from place position"
                }
            ])
            
        return trajectory

class IOController:
    def __init__(self):
        self.input_signals = {
            "START": False,
            "E_STOP": False,
            "PAUSE": False,
            "RESUME": False,
            "MODE_SELECT": "AUTO",
            "PART_PRESENT": False,
            "JIG_CLAMPED": False,
            "DOOR_CLOSED": False,
            "AIR_PRESSURE_OK": False,
            "WELD_ENABLE": False,
            "WELDER_READY": False,
            "SUPPLY_PART_OK": False,
            "DISCHARGE_READY": False
        }
        
        self.output_signals = {
            "WORK_COMPLETE": False,
            "ERROR_OCCURRED": False,
            "READY": False,
            "WORKING": False,
            "STEP_NUMBER": 0,
            "ERROR_CODE": 0,
            "WELD_EXECUTE": False,
            "WELD_COMPLETE": False,
            "PICK_COMPLETE": False,
            "PLACE_COMPLETE": False
        }
    
    def update_input_signal(self, signal_name: str, value: Any):
        if signal_name in self.input_signals:
            self.input_signals[signal_name] = value
            logger.info(f"Input signal {signal_name} updated to {value}")
        else:
            logger.warning(f"Unknown input signal: {signal_name}")
    
    def set_output_signal(self, signal_name: str, value: Any):
        if signal_name in self.output_signals:
            self.output_signals[signal_name] = value
            logger.info(f"Output signal {signal_name} set to {value}")
        else:
            logger.warning(f"Unknown output signal: {signal_name}")
    
    def check_safety_conditions(self) -> bool:
        return (
            not self.input_signals["E_STOP"] and
            self.input_signals["DOOR_CLOSED"] and
            self.input_signals["AIR_PRESSURE_OK"]
        )
    
    def check_start_conditions(self, application_type: ApplicationType) -> bool:
        base_conditions = (
            self.input_signals["START"] and
            self.input_signals["PART_PRESENT"] and
            self.input_signals["JIG_CLAMPED"] and
            self.check_safety_conditions()
        )
        
        if application_type == ApplicationType.SPOT_WELDING:
            return (
                base_conditions and
                self.input_signals["WELD_ENABLE"] and
                self.input_signals["WELDER_READY"]
            )
        elif application_type == ApplicationType.HANDLING:
            return (
                base_conditions and
                self.input_signals["SUPPLY_PART_OK"] and
                self.input_signals["DISCHARGE_READY"]
            )
        
        return base_conditions

class CommandProcessor:
    def __init__(self):
        self.command_queue = []
        
    def add_command(self, command: Dict[str, Any]):
        self.command_queue.append(command)
        
    def get_next_command(self) -> Optional[Dict[str, Any]]:
        if self.command_queue:
            return self.command_queue.pop(0)
        return None
    
    def clear_queue(self):
        self.command_queue.clear()

class RobotTeachingApp:
    def __init__(self, tcp_port: int = 8888):
        self.tcp_server = TCPServer(port=tcp_port)
        self.command_processor = CommandProcessor()
        self.trajectory_generator = TrajectoryGenerator()
        self.io_controller = IOController()
        
        self.current_state = RobotState.READY
        self.current_program = []
        self.current_step = 0
        self.application_type = ApplicationType.HANDLING
        
        self._setup_message_handlers()
        
    def _setup_message_handlers(self):
        self.tcp_server.register_handler("status_request", self._handle_status_request)
        self.tcp_server.register_handler("program_upload", self._handle_program_upload)
        self.tcp_server.register_handler("program_execute", self._handle_program_execute)
        self.tcp_server.register_handler("program_pause", self._handle_program_pause)
        self.tcp_server.register_handler("program_resume", self._handle_program_resume)
        self.tcp_server.register_handler("program_stop", self._handle_program_stop)
        self.tcp_server.register_handler("io_update", self._handle_io_update)
        self.tcp_server.register_handler("emergency_stop", self._handle_emergency_stop)
        
    def _handle_status_request(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        status_data = {
            "robot_state": self.current_state.value,
            "current_step": self.current_step,
            "total_steps": len(self.current_program),
            "application_type": self.application_type.value,
            "input_signals": self.io_controller.input_signals,
            "output_signals": self.io_controller.output_signals,
            "connection_status": self.tcp_server.connection_status
        }
        
        return MessageBuilder.create_response(message_data, "success", status_data)
    
    def _handle_program_upload(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            program_data = message_data.get("parameters", {}).get("program", [])
            self.current_program = program_data
            self.current_step = 0
            
            logger.info(f"Program uploaded: {len(program_data)} steps")
            return MessageBuilder.create_response(
                message_data, "success", 
                {"message": f"Program uploaded with {len(program_data)} steps"}
            )
        except Exception as e:
            logger.error(f"Program upload error: {e}")
            return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
    
    def _handle_program_execute(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.current_program:
                return MessageBuilder.create_response(
                    message_data, "error", {"error": "No program loaded"}
                )
            
            if not self.io_controller.check_start_conditions(self.application_type):
                return MessageBuilder.create_response(
                    message_data, "error", {"error": "Start conditions not met"}
                )
            
            self.current_state = RobotState.WORKING
            self.io_controller.set_output_signal("WORKING", True)
            self.io_controller.set_output_signal("READY", False)
            
            self._execute_program()
            
            return MessageBuilder.create_response(
                message_data, "success", {"message": "Program execution started"}
            )
        except Exception as e:
            logger.error(f"Program execution error: {e}")
            self.current_state = RobotState.ERROR
            self.io_controller.set_output_signal("ERROR_OCCURRED", True)
            return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
    
    def _handle_program_pause(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        self.current_state = RobotState.PAUSED
        self.io_controller.set_output_signal("WORKING", False)
        return MessageBuilder.create_response(message_data, "success", {"message": "Program paused"})
    
    def _handle_program_resume(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == RobotState.PAUSED:
            self.current_state = RobotState.WORKING
            self.io_controller.set_output_signal("WORKING", True)
            return MessageBuilder.create_response(message_data, "success", {"message": "Program resumed"})
        else:
            return MessageBuilder.create_response(
                message_data, "error", {"error": "Program not in paused state"}
            )
    
    def _handle_program_stop(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        self.current_state = RobotState.READY
        self.current_step = 0
        self.io_controller.set_output_signal("WORKING", False)
        self.io_controller.set_output_signal("READY", True)
        return MessageBuilder.create_response(message_data, "success", {"message": "Program stopped"})
    
    def _handle_io_update(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            io_data = message_data.get("parameters", {}).get("io_data", {})
            for signal_name, value in io_data.items():
                self.io_controller.update_input_signal(signal_name, value)
            
            return MessageBuilder.create_response(message_data, "success", {"message": "I/O updated"})
        except Exception as e:
            return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
    
    def _handle_emergency_stop(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        self.current_state = RobotState.EMERGENCY_STOP
        self.io_controller.set_output_signal("ERROR_OCCURRED", True)
        self.io_controller.set_output_signal("WORKING", False)
        logger.warning("Emergency stop activated")
        return MessageBuilder.create_response(message_data, "success", {"message": "Emergency stop activated"})
    
    def _execute_program(self):
        for step_index, step in enumerate(self.current_program):
            if self.current_state != RobotState.WORKING:
                break
                
            self.current_step = step_index + 1
            self.io_controller.set_output_signal("STEP_NUMBER", self.current_step)
            
            self._send_step_to_vc(step)
            
            time.sleep(0.1)
        
        if self.current_state == RobotState.WORKING:
            self.current_state = RobotState.READY
            self.io_controller.set_output_signal("WORK_COMPLETE", True)
            self.io_controller.set_output_signal("WORKING", False)
            self.io_controller.set_output_signal("READY", True)
    
    def _send_step_to_vc(self, step: Dict[str, Any]):
        step_type = step.get("type")
        
        if step_type == "move":
            message = MessageBuilder.create_robot_move_command(
                "robot_1", 
                step.get("position", []),
                step.get("speed", 100)
            )
        elif step_type == "tool_control":
            message = MessageBuilder.create_tool_control_command(
                "tool_1",
                step.get("command", "")
            )
        elif step_type == "weld":
            message = MessageBuilder.create_message(
                "weld_execute",
                "welder_1",
                {
                    "weld_time": step.get("weld_time", 1.0),
                    "pressure": step.get("pressure", 100.0),
                    "current": step.get("current", 5000.0),
                    "electrode_force": step.get("electrode_force", 200.0)
                }
            )
        else:
            logger.warning(f"Unknown step type: {step_type}")
            return
        
        self.tcp_server.send_message(message)
    
    def generate_teaching_data(self, work_data: Dict[str, Any], application_type: ApplicationType):
        self.application_type = application_type
        
        if application_type == ApplicationType.SPOT_WELDING:
            weld_points = []
            for point_data in work_data.get("weld_points", []):
                weld_point = WeldPoint(
                    position=Position(**point_data.get("position", {})),
                    weld_time=point_data.get("weld_time", 1.0),
                    pressure=point_data.get("pressure", 100.0),
                    current=point_data.get("current", 5000.0),
                    electrode_force=point_data.get("electrode_force", 200.0)
                )
                weld_points.append(weld_point)
            
            trajectory = self.trajectory_generator.generate_welding_trajectory(weld_points)
            
        elif application_type == ApplicationType.HANDLING:
            handling_points = []
            for point_data in work_data.get("handling_points", []):
                handling_point = HandlingPoint(
                    pick_position=Position(**point_data.get("pick_position", {})),
                    place_position=Position(**point_data.get("place_position", {})),
                    grip_force=point_data.get("grip_force", 50.0),
                    approach_speed=point_data.get("approach_speed", 80),
                    work_speed=point_data.get("work_speed", 30)
                )
                handling_points.append(handling_point)
            
            trajectory = self.trajectory_generator.generate_handling_trajectory(handling_points)
        
        else:
            raise ValueError(f"Unsupported application type: {application_type}")
        
        self.current_program = trajectory
        logger.info(f"Generated teaching data for {application_type.value}: {len(trajectory)} steps")
        
        return trajectory
    
    def send_robot_program(self, program_data: List[Dict[str, Any]]):
        self.current_program = program_data
        
        message = MessageBuilder.create_message(
            "program_upload",
            "controller",
            {"program": program_data}
        )
        
        return self.tcp_server.send_message(message)
    
    def monitor_execution(self):
        status_request = MessageBuilder.create_status_request("robot_1")
        self.tcp_server.send_message(status_request)
    
    def start_application(self):
        logger.info("Starting Robot Teaching Application")
        self.io_controller.set_output_signal("READY", True)
        self.tcp_server.start_server()

if __name__ == "__main__":
    app = RobotTeachingApp()
    
    try:
        app.start_application()
    except KeyboardInterrupt:
        logger.info("Shutting down application...")
        app.tcp_server.stop_server()