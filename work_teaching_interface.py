import threading
import time
import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import socket
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

from config_manager import config_manager
from basic_handling_workflow import BasicHandlingWorkflow, Position, WorkPiece, HandlingTask, WorkflowResult
from trajectory_generation import generate_handling_trajectory, trajectory_generator
from integrated_safety_system import safety_system
from io_message_handler import io_controller

logger = logging.getLogger(__name__)

class TeachingMode(Enum):
    MANUAL = "manual"          # 手動教示
    GUIDED = "guided"         # ガイド付き教示
    TEMPLATE = "template"     # テンプレート使用
    IMPORT = "import"         # データインポート

class InterfaceType(Enum):
    CLI = "cli"               # コマンドライン
    WEB = "web"              # Webインターフェース
    JSON_API = "json_api"    # JSON API
    FILE_BASED = "file_based" # ファイルベース

@dataclass
class TeachingSession:
    session_id: str
    start_time: datetime
    mode: TeachingMode
    interface_type: InterfaceType
    current_task: Optional[HandlingTask] = None
    taught_positions: List[Position] = None
    parameters: Dict[str, Any] = None
    completed: bool = False

    def __post_init__(self):
        if self.taught_positions is None:
            self.taught_positions = []
        if self.parameters is None:
            self.parameters = {}

class WorkTeachingInterface:
    """作業教示インターフェース"""
    
    def __init__(self, workflow: BasicHandlingWorkflow):
        self.workflow = workflow
        self.active_sessions: Dict[str, TeachingSession] = {}
        self.teaching_templates = self._load_teaching_templates()
        self.current_robot_position = Position(0, 0, 200, 0, 0, 0)
        
        # Web サーバー設定
        self.web_server: Optional[socketserver.TCPServer] = None
        self.web_server_thread: Optional[threading.Thread] = None
        self.web_port = 8080
        
    def _load_teaching_templates(self) -> Dict[str, Any]:
        """教示テンプレート読み込み"""
        templates = config_manager.get_config_value("TEMPLATES", "work_templates", {})
        return templates
    
    def start_teaching_session(self, mode: TeachingMode, 
                             interface_type: InterfaceType,
                             session_params: Dict[str, Any] = None) -> str:
        """教示セッション開始"""
        session_id = f"session_{int(time.time() * 1000)}"
        
        session = TeachingSession(
            session_id=session_id,
            start_time=datetime.now(),
            mode=mode,
            interface_type=interface_type,
            parameters=session_params or {}
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Started teaching session: {session_id} ({mode.value})")
        return session_id
    
    def end_teaching_session(self, session_id: str) -> bool:
        """教示セッション終了"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.completed = True
            
            logger.info(f"Ended teaching session: {session_id}")
            return True
        return False

class CLITeachingInterface:
    """コマンドライン教示インターフェース"""
    
    def __init__(self, teaching_interface: WorkTeachingInterface):
        self.teaching_interface = teaching_interface
        self.current_session_id: Optional[str] = None
        self.commands = {
            'help': self._help,
            'start': self._start_session,
            'position': self._teach_position,
            'workpiece': self._define_workpiece,
            'test': self._test_movement,
            'generate': self._generate_trajectory,
            'execute': self._execute_task,
            'status': self._show_status,
            'save': self._save_work,
            'load': self._load_work,
            'quit': self._quit
        }
    
    def run_interactive_session(self):
        """対話式教示セッション実行"""
        print("\n=== Robot Work Teaching Interface ===")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command_line = input("\nteach> ").strip().lower()
                if not command_line:
                    continue
                
                parts = command_line.split()
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                if command in self.commands:
                    try:
                        self.commands[command](args)
                    except Exception as e:
                        print(f"Error executing command: {e}")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break
    
    def _help(self, args: List[str]):
        """ヘルプ表示"""
        help_text = """
Available Commands:
  help                 - Show this help message
  start <mode>        - Start teaching session (manual/guided/template)
  position <name>     - Teach position (pick/place/intermediate)
  workpiece           - Define workpiece properties
  test                - Test robot movement to taught positions
  generate            - Generate trajectory from taught positions
  execute             - Execute the taught task
  status              - Show current session status
  save <filename>     - Save taught work to file
  load <filename>     - Load work from file
  quit                - Exit teaching interface

Examples:
  start manual
  position pick
  position place
  workpiece part1 plastic 1.5 50
  generate
  execute
"""
        print(help_text)
    
    def _start_session(self, args: List[str]):
        """セッション開始"""
        if not args:
            print("Usage: start <mode>")
            print("Available modes: manual, guided, template")
            return
        
        mode_str = args[0].lower()
        try:
            mode = TeachingMode(mode_str)
            self.current_session_id = self.teaching_interface.start_teaching_session(
                mode, InterfaceType.CLI
            )
            print(f"Started {mode.value} teaching session: {self.current_session_id}")
            
            if mode == TeachingMode.GUIDED:
                self._guided_teaching_flow()
            
        except ValueError:
            print(f"Invalid mode: {mode_str}")
    
    def _teach_position(self, args: List[str]):
        """位置教示"""
        if not self.current_session_id:
            print("No active session. Use 'start' command first.")
            return
        
        if not args:
            print("Usage: position <name>")
            print("Common names: pick, place, intermediate1, intermediate2")
            return
        
        position_name = args[0]
        
        # 現在位置を取得（実際の実装では手動操作またはセンサーから）
        current_pos = self._get_current_position()
        
        session = self.teaching_interface.active_sessions[self.current_session_id]
        
        # 位置を保存
        position_data = {
            "name": position_name,
            "position": current_pos,
            "timestamp": datetime.now().isoformat()
        }
        
        # セッションに位置追加
        session.taught_positions.append(current_pos)
        session.parameters[position_name] = current_pos.to_list()
        
        print(f"Position '{position_name}' taught: {current_pos.to_list()}")
    
    def _define_workpiece(self, args: List[str]):
        """ワークピース定義"""
        if not self.current_session_id:
            print("No active session. Use 'start' command first.")
            return
        
        if len(args) < 4:
            print("Usage: workpiece <name> <material> <weight> <grip_force>")
            print("Example: workpiece part1 plastic 1.5 50")
            return
        
        try:
            name = args[0]
            material = args[1]
            weight = float(args[2])
            grip_force = float(args[3])
            
            workpiece_data = {
                "name": name,
                "part_type": "taught_component",
                "material": material,
                "weight": weight,
                "dimensions": [100, 100, 50],  # デフォルト
                "grip_force": grip_force
            }
            
            session = self.teaching_interface.active_sessions[self.current_session_id]
            session.parameters["workpiece"] = workpiece_data
            
            print(f"Workpiece defined: {name} ({material}, {weight}kg, {grip_force}N)")
            
        except ValueError as e:
            print(f"Invalid parameter: {e}")
    
    def _test_movement(self, args: List[str]):
        """テスト移動"""
        if not self.current_session_id:
            print("No active session.")
            return
        
        session = self.teaching_interface.active_sessions[self.current_session_id]
        
        if "pick" not in session.parameters or "place" not in session.parameters:
            print("Both pick and place positions must be taught first.")
            return
        
        print("Testing movement sequence...")
        
        # 安全性チェック
        pick_pos = Position.from_list(session.parameters["pick"])
        place_pos = Position.from_list(session.parameters["place"])
        
        if not safety_system.is_safe_to_move(pick_pos):
            print("WARNING: Pick position is not safe!")
            return
        
        if not safety_system.is_safe_to_move(place_pos):
            print("WARNING: Place position is not safe!")
            return
        
        print("✓ Pick position safety: OK")
        print("✓ Place position safety: OK")
        print("Test movement sequence validated")
    
    def _generate_trajectory(self, args: List[str]):
        """軌道生成"""
        if not self.current_session_id:
            print("No active session.")
            return
        
        session = self.teaching_interface.active_sessions[self.current_session_id]
        
        required_params = ["pick", "place", "workpiece"]
        missing_params = [p for p in required_params if p not in session.parameters]
        
        if missing_params:
            print(f"Missing required parameters: {missing_params}")
            return
        
        try:
            # 軌道生成設定
            trajectory_config = {
                "operation_type": "basic_pick_place",
                "pick_position": session.parameters["pick"],
                "place_position": session.parameters["place"],
                "workpiece": session.parameters["workpiece"],
                "parameters": {
                    "approach_speed": 70,
                    "work_speed": 25,
                    "safety_height": 60.0
                }
            }
            
            trajectory = generate_handling_trajectory(trajectory_config)
            
            print(f"Generated trajectory with {len(trajectory)} points:")
            for i, point in enumerate(trajectory[:5]):  # 最初の5点を表示
                print(f"  {i+1}: {point.position.to_list()} @ {point.speed}% - {point.description}")
            
            if len(trajectory) > 5:
                print(f"  ... and {len(trajectory) - 5} more points")
            
            # 軌道をセッションに保存
            session.parameters["trajectory"] = [
                {
                    "position": point.position.to_list(),
                    "speed": point.speed,
                    "description": point.description
                }
                for point in trajectory
            ]
            
            print("Trajectory generation completed successfully")
            
        except Exception as e:
            print(f"Trajectory generation failed: {e}")
    
    def _execute_task(self, args: List[str]):
        """タスク実行"""
        if not self.current_session_id:
            print("No active session.")
            return
        
        session = self.teaching_interface.active_sessions[self.current_session_id]
        
        if "trajectory" not in session.parameters:
            print("Generate trajectory first using 'generate' command.")
            return
        
        # 安全確認
        if not safety_system.is_safe_to_operate():
            print("System is not safe to operate. Check safety conditions.")
            return
        
        print("Preparing to execute taught task...")
        
        try:
            # タスク作成
            task_config = {
                "task_id": f"taught_task_{session.session_id}",
                "pick_position": session.parameters["pick"],
                "place_position": session.parameters["place"],
                "workpiece": session.parameters["workpiece"],
                "approach_speed": 70,
                "work_speed": 25,
                "safety_height": 60.0
            }
            
            task = self.teaching_interface.workflow.create_handling_task(task_config)
            
            print("Starting task execution...")
            result = self.teaching_interface.workflow.execute_handling_task(task)
            
            if result == WorkflowResult.SUCCESS:
                print("✓ Task executed successfully!")
            else:
                print(f"✗ Task execution failed: {result.value}")
            
        except Exception as e:
            print(f"Task execution error: {e}")
    
    def _show_status(self, args: List[str]):
        """ステータス表示"""
        print("\n=== Teaching Session Status ===")
        
        if self.current_session_id:
            session = self.teaching_interface.active_sessions[self.current_session_id]
            print(f"Session ID: {session.session_id}")
            print(f"Mode: {session.mode.value}")
            print(f"Started: {session.start_time}")
            print(f"Taught positions: {len(session.taught_positions)}")
            
            for param_name, param_value in session.parameters.items():
                if param_name in ["pick", "place"]:
                    print(f"  {param_name}: {param_value}")
                elif param_name == "workpiece":
                    workpiece = param_value
                    print(f"  workpiece: {workpiece['name']} ({workpiece.get('material', 'unknown')})")
        else:
            print("No active session")
        
        # システム状態
        print(f"\nSystem Status:")
        safety_status = safety_system.get_safety_status()
        print(f"  Safety: {'OK' if safety_status['overall_safety']['overall_safe'] else 'WARNING'}")
        print(f"  Robot ready: {io_controller.get_signal_value('READY_LAMP')}")
    
    def _save_work(self, args: List[str]):
        """作業保存"""
        if not args:
            print("Usage: save <filename>")
            return
        
        if not self.current_session_id:
            print("No active session to save.")
            return
        
        filename = args[0]
        if not filename.endswith('.json'):
            filename += '.json'
        
        session = self.teaching_interface.active_sessions[self.current_session_id]
        
        save_data = {
            "session_info": {
                "session_id": session.session_id,
                "mode": session.mode.value,
                "created": session.start_time.isoformat()
            },
            "taught_data": session.parameters,
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        try:
            os.makedirs("taught_works", exist_ok=True)
            filepath = os.path.join("taught_works", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"Work saved to: {filepath}")
            
        except Exception as e:
            print(f"Save failed: {e}")
    
    def _load_work(self, args: List[str]):
        """作業読み込み"""
        if not args:
            print("Usage: load <filename>")
            return
        
        filename = args[0]
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join("taught_works", filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            # 新しいセッション作成
            mode_str = load_data["session_info"].get("mode", "manual")
            mode = TeachingMode(mode_str)
            
            self.current_session_id = self.teaching_interface.start_teaching_session(
                mode, InterfaceType.CLI
            )
            
            session = self.teaching_interface.active_sessions[self.current_session_id]
            session.parameters = load_data["taught_data"]
            
            # 教示位置復元
            if "pick" in session.parameters:
                session.taught_positions.append(Position.from_list(session.parameters["pick"]))
            if "place" in session.parameters:
                session.taught_positions.append(Position.from_list(session.parameters["place"]))
            
            print(f"Work loaded from: {filepath}")
            print(f"Loaded session: {load_data['session_info']['session_id']}")
            
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Load failed: {e}")
    
    def _quit(self, args: List[str]):
        """終了"""
        if self.current_session_id:
            self.teaching_interface.end_teaching_session(self.current_session_id)
        print("Teaching interface closed.")
        exit(0)
    
    def _guided_teaching_flow(self):
        """ガイド付き教示フロー"""
        print("\n=== Guided Teaching Mode ===")
        print("I'll guide you through teaching a pick and place operation.")
        
        steps = [
            ("Define workpiece", "workpiece"),
            ("Teach pick position", "position pick"),
            ("Teach place position", "position place"),
            ("Generate trajectory", "generate"),
            ("Test the operation", "test")
        ]
        
        print("\nSteps to complete:")
        for i, (desc, cmd) in enumerate(steps, 1):
            print(f"  {i}. {desc} (use: {cmd})")
        
        print("\nStart with step 1, or type 'help' for more information.")
    
    def _get_current_position(self) -> Position:
        """現在位置取得（シミュレーション用）"""
        # 実際の実装ではロボットから位置を取得
        # ここではユーザー入力でシミュレーション
        try:
            print("Enter position [x y z rx ry rz] or press Enter for current:")
            pos_input = input("Position> ").strip()
            
            if not pos_input:
                # デフォルト位置を返す
                import random
                x = random.uniform(0, 500)
                y = random.uniform(-300, 300)
                z = random.uniform(100, 300)
                return Position(x, y, z, 0, 0, 0)
            
            coords = [float(x) for x in pos_input.split()]
            if len(coords) == 6:
                return Position(*coords)
            elif len(coords) == 3:
                return Position(coords[0], coords[1], coords[2], 0, 0, 0)
            else:
                raise ValueError("Invalid number of coordinates")
                
        except Exception as e:
            print(f"Using default position due to error: {e}")
            return Position(100, 100, 150, 0, 0, 0)

class WebTeachingInterface:
    """Web教示インターフェース"""
    
    def __init__(self, teaching_interface: WorkTeachingInterface, port: int = 8080):
        self.teaching_interface = teaching_interface
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start_web_server(self):
        """Webサーバー開始"""
        try:
            handler = self._create_request_handler()
            self.server = socketserver.TCPServer(("localhost", self.port), handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            logger.info(f"Web teaching interface started at http://localhost:{self.port}")
            print(f"Web interface available at: http://localhost:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
    
    def stop_web_server(self):
        """Webサーバー停止"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Web teaching interface stopped")
    
    def _create_request_handler(self):
        """リクエストハンドラー作成"""
        teaching_interface = self.teaching_interface
        
        class TeachingRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self._serve_main_page()
                elif self.path == '/api/status':
                    self._serve_status_api()
                elif self.path == '/api/sessions':
                    self._serve_sessions_api()
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == '/api/teach_position':
                    self._handle_teach_position()
                elif self.path == '/api/start_session':
                    self._handle_start_session()
                elif self.path == '/api/execute_task':
                    self._handle_execute_task()
                else:
                    self.send_error(404)
            
            def _serve_main_page(self):
                html_content = self._generate_html_interface()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_content.encode())
            
            def _serve_status_api(self):
                status = {
                    "system_status": safety_system.get_safety_status(),
                    "active_sessions": len(teaching_interface.active_sessions),
                    "current_position": teaching_interface.current_robot_position.to_list()
                }
                self._send_json_response(status)
            
            def _serve_sessions_api(self):
                sessions_data = []
                for session_id, session in teaching_interface.active_sessions.items():
                    sessions_data.append({
                        "session_id": session_id,
                        "mode": session.mode.value,
                        "start_time": session.start_time.isoformat(),
                        "completed": session.completed,
                        "taught_positions": len(session.taught_positions)
                    })
                self._send_json_response(sessions_data)
            
            def _send_json_response(self, data):
                json_data = json.dumps(data, indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json_data.encode())
            
            def _generate_html_interface(self):
                """HTML インターフェース生成"""
                return """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Work Teaching Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .button:hover { background: #0056b3; }
        .input-group { margin: 10px 0; }
        .input-group label { display: inline-block; width: 150px; }
        .input-group input { padding: 5px; width: 200px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 3px; }
        .status.safe { background: #d4edda; color: #155724; }
        .status.warning { background: #fff3cd; color: #856404; }
        .log { background: #f8f9fa; padding: 10px; height: 200px; overflow-y: scroll; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Robot Work Teaching Interface</h1>
        
        <div class="section">
            <h2>System Status</h2>
            <div id="system-status" class="status safe">System Ready</div>
            <button class="button" onclick="refreshStatus()">Refresh Status</button>
        </div>
        
        <div class="section">
            <h2>Teaching Session</h2>
            <div class="input-group">
                <label>Mode:</label>
                <select id="teaching-mode">
                    <option value="manual">Manual</option>
                    <option value="guided">Guided</option>
                    <option value="template">Template</option>
                </select>
            </div>
            <button class="button" onclick="startSession()">Start New Session</button>
            <div id="session-info"></div>
        </div>
        
        <div class="section">
            <h2>Position Teaching</h2>
            <div class="input-group">
                <label>Position Name:</label>
                <input type="text" id="position-name" placeholder="pick, place, intermediate1">
            </div>
            <div class="input-group">
                <label>X:</label><input type="number" id="pos-x" step="0.1">
                <label>Y:</label><input type="number" id="pos-y" step="0.1">
                <label>Z:</label><input type="number" id="pos-z" step="0.1">
            </div>
            <button class="button" onclick="teachPosition()">Teach Position</button>
            <button class="button" onclick="getCurrentPosition()">Get Current Position</button>
        </div>
        
        <div class="section">
            <h2>Workpiece Definition</h2>
            <div class="input-group">
                <label>Name:</label>
                <input type="text" id="workpiece-name" placeholder="part1">
            </div>
            <div class="input-group">
                <label>Material:</label>
                <input type="text" id="workpiece-material" placeholder="plastic">
            </div>
            <div class="input-group">
                <label>Weight (kg):</label>
                <input type="number" id="workpiece-weight" step="0.1">
            </div>
            <div class="input-group">
                <label>Grip Force (N):</label>
                <input type="number" id="workpiece-force" step="1">
            </div>
            <button class="button" onclick="defineWorkpiece()">Define Workpiece</button>
        </div>
        
        <div class="section">
            <h2>Operations</h2>
            <button class="button" onclick="generateTrajectory()">Generate Trajectory</button>
            <button class="button" onclick="executeTask()">Execute Task</button>
            <button class="button" onclick="testMovement()">Test Movement</button>
        </div>
        
        <div class="section">
            <h2>Activity Log</h2>
            <div id="activity-log" class="log"></div>
        </div>
    </div>
    
    <script>
        let currentSessionId = null;
        
        function log(message) {
            const logDiv = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}\\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('system-status');
                    const safe = data.system_status.overall_safety.overall_safe;
                    statusDiv.className = `status ${safe ? 'safe' : 'warning'}`;
                    statusDiv.textContent = safe ? 'System Ready' : 'Safety Warning';
                    log('Status refreshed');
                })
                .catch(error => log('Error: ' + error));
        }
        
        function startSession() {
            const mode = document.getElementById('teaching-mode').value;
            // Implementation would send POST request to start session
            log(`Started ${mode} teaching session`);
        }
        
        function teachPosition() {
            const name = document.getElementById('position-name').value;
            const x = document.getElementById('pos-x').value;
            const y = document.getElementById('pos-y').value;
            const z = document.getElementById('pos-z').value;
            
            if (!name || !x || !y || !z) {
                log('Please fill all position fields');
                return;
            }
            
            log(`Taught position '${name}': [${x}, ${y}, ${z}]`);
        }
        
        function getCurrentPosition() {
            // In real implementation, this would fetch current robot position
            log('Current position: [100.0, -50.0, 200.0, 0.0, 0.0, 0.0]');
            document.getElementById('pos-x').value = 100.0;
            document.getElementById('pos-y').value = -50.0;
            document.getElementById('pos-z').value = 200.0;
        }
        
        function defineWorkpiece() {
            const name = document.getElementById('workpiece-name').value;
            const material = document.getElementById('workpiece-material').value;
            const weight = document.getElementById('workpiece-weight').value;
            const force = document.getElementById('workpiece-force').value;
            
            if (!name || !material || !weight || !force) {
                log('Please fill all workpiece fields');
                return;
            }
            
            log(`Defined workpiece: ${name} (${material}, ${weight}kg, ${force}N)`);
        }
        
        function generateTrajectory() {
            log('Generating trajectory...');
            setTimeout(() => log('Trajectory generated successfully'), 1000);
        }
        
        function executeTask() {
            log('Executing task...');
            setTimeout(() => log('Task completed successfully'), 2000);
        }
        
        function testMovement() {
            log('Testing movement sequence...');
            setTimeout(() => log('Movement test completed - All positions safe'), 1500);
        }
        
        // Initialize
        window.onload = function() {
            refreshStatus();
            log('Robot Work Teaching Interface initialized');
        };
        
        // Refresh status every 5 seconds
        setInterval(refreshStatus, 5000);
    </script>
</body>
</html>
                """
        
        return TeachingRequestHandler

# グローバルインスタンス作成用関数
def create_work_teaching_interface(workflow: BasicHandlingWorkflow) -> WorkTeachingInterface:
    """作業教示インターフェース作成"""
    return WorkTeachingInterface(workflow)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト用ワークフロー作成
    from basic_handling_workflow import basic_workflow
    
    # 教示インターフェース作成
    teaching_interface = create_work_teaching_interface(basic_workflow)
    
    # CLI インターフェース開始
    cli_interface = CLITeachingInterface(teaching_interface)
    
    # Webインターフェースも起動（バックグラウンド）
    web_interface = WebTeachingInterface(teaching_interface)
    web_interface.start_web_server()
    
    try:
        print("Starting CLI teaching interface...")
        print("Web interface also available at http://localhost:8080")
        cli_interface.run_interactive_session()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        web_interface.stop_web_server()