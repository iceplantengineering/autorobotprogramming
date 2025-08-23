#!/usr/bin/env python3
"""
Integrated Web Server with Visual Components Communication
Visual Components連携統合WebUI
"""

import http.server
import socketserver
import json
import threading
import time
import logging
import socket
from urllib.parse import urlparse, parse_qs
import os
from datetime import datetime

# 既存モジュールをインポート
from tcp_communication import TCPServer, MessageBuilder

logger = logging.getLogger(__name__)

class IntegratedRobotSystem:
    def __init__(self, web_port=8080, tcp_port=8888):
        self.web_port = web_port
        self.tcp_port = tcp_port
        
        # Web UI サーバー
        self.web_server = None
        self.web_thread = None
        
        # TCP サーバー (Visual Components通信用)
        self.tcp_server = TCPServer("localhost", tcp_port)
        
        # システム状態管理
        self.robot_status = {
            "connected": False,
            "vc_connected": False,
            "position": {"x": 0, "y": 0, "z": 200, "rx": 0, "ry": 0, "rz": 0},
            "joints": {"J1": 0, "J2": 0, "J3": 0, "J4": 0, "J5": 0, "J6": 0},
            "io_signals": {
                "START_BUTTON": False,
                "E_STOP": False,
                "READY_LAMP": True,
                "WORKING_LAMP": False
            },
            "safety_status": "OK",
            "current_operation": "Waiting for Visual Components connection"
        }
        
        self.teaching_points = []
        self.production_stats = {
            "total_parts": 0,
            "completed_parts": 0,
            "cycle_time": 8.5,
            "efficiency": 95.2
        }
        
        self.vc_messages = []  # Visual Componentsとのメッセージ履歴
        
        # TCP通信ハンドラー設定
        self._setup_tcp_handlers()

    def _setup_tcp_handlers(self):
        """TCP通信ハンドラーを設定"""
        
        def handle_robot_status(message_data):
            """ロボット状態更新処理"""
            try:
                params = message_data.get("parameters", {})
                
                # ジョイント位置更新
                if "joint_positions" in params:
                    joints = params["joint_positions"]
                    if len(joints) >= 6:
                        self.robot_status["joints"] = {
                            "J1": joints[0], "J2": joints[1], "J3": joints[2],
                            "J4": joints[3], "J5": joints[4], "J6": joints[5]
                        }
                
                # TCP位置更新 (簡単な変換)
                if "tcp_position" in params:
                    tcp = params["tcp_position"]
                    if len(tcp) >= 6:
                        self.robot_status["position"] = {
                            "x": tcp[0], "y": tcp[1], "z": tcp[2],
                            "rx": tcp[3], "ry": tcp[4], "rz": tcp[5]
                        }
                
                self.robot_status["vc_connected"] = True
                self.robot_status["current_operation"] = params.get("operation", "Connected")
                
                self._log_vc_message(f"Robot status updated: {params}")
                
                return MessageBuilder.create_response(message_data, "success", {
                    "status": "updated",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self._log_vc_message(f"Error handling robot status: {e}")
                return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
        
        def handle_robot_move(message_data):
            """ロボット移動処理"""
            try:
                params = message_data.get("parameters", {})
                
                if "joint_positions" in params:
                    joints = params["joint_positions"]
                    self.robot_status["joints"] = {
                        "J1": joints[0] if len(joints) > 0 else 0,
                        "J2": joints[1] if len(joints) > 1 else 0,
                        "J3": joints[2] if len(joints) > 2 else 0,
                        "J4": joints[3] if len(joints) > 3 else 0,
                        "J5": joints[4] if len(joints) > 4 else 0,
                        "J6": joints[5] if len(joints) > 5 else 0
                    }
                    
                    self.robot_status["current_operation"] = "Moving"
                    self._log_vc_message(f"Robot moving to joints: {joints}")
                    
                    # 1秒後にIdle状態に
                    threading.Timer(1.0, lambda: self._set_operation("Idle")).start()
                
                return MessageBuilder.create_response(message_data, "success", {
                    "move_completed": True
                })
                
            except Exception as e:
                return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
        
        def handle_io_control(message_data):
            """I/O制御処理"""
            try:
                params = message_data.get("parameters", {})
                io_data = params.get("io_data", {})
                
                # I/O信号更新
                for signal, value in io_data.items():
                    if signal in self.robot_status["io_signals"]:
                        self.robot_status["io_signals"][signal] = value
                
                self._log_vc_message(f"I/O signals updated: {io_data}")
                
                return MessageBuilder.create_response(message_data, "success", {
                    "io_updated": True
                })
                
            except Exception as e:
                return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
        
        def handle_operation_step(message_data):
            """作業ステップ処理"""
            try:
                params = message_data.get("parameters", {})
                step_name = params.get("step_name", "Unknown Step")
                
                self.robot_status["current_operation"] = f"Executing: {step_name}"
                self._log_vc_message(f"Operation step: {step_name}")
                
                # 生産統計更新
                if "completed" in step_name.lower():
                    self.production_stats["completed_parts"] += 1
                
                return MessageBuilder.create_response(message_data, "success", {
                    "step_completed": True,
                    "step_name": step_name
                })
                
            except Exception as e:
                return MessageBuilder.create_response(message_data, "error", {"error": str(e)})
        
        # ハンドラー登録
        self.tcp_server.register_handler("robot_status", handle_robot_status)
        self.tcp_server.register_handler("robot_move", handle_robot_move)
        self.tcp_server.register_handler("io_control", handle_io_control)
        self.tcp_server.register_handler("operation_step", handle_operation_step)
    
    def _log_vc_message(self, message):
        """Visual Componentsメッセージログ"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.vc_messages.append(f"[{timestamp}] {message}")
        
        # 最新100件のみ保持
        if len(self.vc_messages) > 100:
            self.vc_messages = self.vc_messages[-100:]
    
    def _set_operation(self, operation):
        """動作状態設定"""
        self.robot_status["current_operation"] = operation

    def start_tcp_server(self):
        """TCPサーバー開始"""
        try:
            tcp_thread = threading.Thread(target=self.tcp_server.start_server, daemon=True)
            tcp_thread.start()
            
            self.robot_status["connected"] = True
            self._log_vc_message(f"TCP Server started on port {self.tcp_port}")
            return True
            
        except Exception as e:
            print(f"Failed to start TCP server: {e}")
            return False

    def start_web_server(self):
        """Webサーバー開始"""
        try:
            handler = self._create_web_handler()
            self.web_server = socketserver.TCPServer(("", self.web_port), handler)
            self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
            self.web_thread.start()
            
            print(f"Integrated Web Server started: http://localhost:{self.web_port}")
            return True
            
        except Exception as e:
            print(f"Failed to start web server: {e}")
            return False

    def stop_servers(self):
        """全サーバー停止"""
        if self.web_server:
            self.web_server.shutdown()
            self.web_server.server_close()
        
        if self.tcp_server:
            self.tcp_server.stop_server()
        
        print("All servers stopped")

    def _create_web_handler(self):
        """Webリクエストハンドラー作成"""
        system = self
        
        class IntegratedRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self._serve_main_page()
                elif self.path == '/api/robot_status':
                    self._serve_api_response(system.robot_status)
                elif self.path == '/api/teaching_points':
                    self._serve_api_response(system.teaching_points)
                elif self.path == '/api/production_stats':
                    self._serve_api_response(system.production_stats)
                elif self.path == '/api/vc_messages':
                    self._serve_api_response({"messages": system.vc_messages[-20:]})  # 最新20件
                else:
                    self.send_error(404)
                    
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                if self.path == '/api/move_robot':
                    self._handle_move_robot(post_data)
                elif self.path == '/api/teach_point':
                    self._handle_teach_point(post_data)
                elif self.path == '/api/send_to_vc':
                    self._handle_send_to_vc(post_data)
                else:
                    self.send_error(404)
            
            def _serve_main_page(self):
                """統合メインページ配信"""
                html_content = system._generate_integrated_html()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Content-length', str(len(html_content.encode())))
                self.end_headers()
                self.wfile.write(html_content.encode())
            
            def _serve_api_response(self, data):
                """API応答配信"""
                json_data = json.dumps(data, ensure_ascii=False)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-length', str(len(json_data.encode())))
                self.end_headers()
                self.wfile.write(json_data.encode())
            
            def _handle_move_robot(self, post_data):
                """ロボット移動処理"""
                try:
                    data = json.loads(post_data.decode())
                    
                    # WebUIからの移動指令をVisual Componentsに送信
                    if system.robot_status["vc_connected"]:
                        move_command = MessageBuilder.create_robot_move_command(
                            "robot_1",
                            [data["joints"]["J1"], data["joints"]["J2"], data["joints"]["J3"],
                             data["joints"]["J4"], data["joints"]["J5"], data["joints"]["J6"]]
                        )
                        
                        # TCPで送信（実際のVC接続時）
                        system._log_vc_message(f"Sending move command to VC: {data['joints']}")
                    
                    system.robot_status["joints"] = data["joints"]
                    system.robot_status["current_operation"] = "Moving"
                    
                    # 1秒後にIdle状態に
                    threading.Timer(1.0, lambda: system._set_operation("Idle")).start()
                    
                    self._serve_api_response({"status": "success"})
                except Exception as e:
                    self._serve_api_response({"status": "error", "message": str(e)})
            
            def _handle_teach_point(self, post_data):
                """教示ポイント追加処理"""
                try:
                    data = json.loads(post_data.decode())
                    point = {
                        "name": data["name"],
                        "joints": system.robot_status["joints"].copy(),
                        "timestamp": time.time()
                    }
                    system.teaching_points.append(point)
                    
                    system._log_vc_message(f"Teaching point recorded: {data['name']}")
                    
                    self._serve_api_response({"status": "success", "points_count": len(system.teaching_points)})
                except Exception as e:
                    self._serve_api_response({"status": "error", "message": str(e)})
            
            def _handle_send_to_vc(self, post_data):
                """Visual Componentsへのコマンド送信"""
                try:
                    data = json.loads(post_data.decode())
                    command = data.get("command")
                    
                    system._log_vc_message(f"Command sent to VC: {command}")
                    
                    # 実際のコマンド実行はここに実装
                    if command == "start_demo":
                        system._set_operation("Running Demo")
                    elif command == "stop_demo":
                        system._set_operation("Demo Stopped")
                    
                    self._serve_api_response({"status": "success", "command": command})
                except Exception as e:
                    self._serve_api_response({"status": "error", "message": str(e)})
        
        return IntegratedRequestHandler

    def _generate_integrated_html(self):
        """統合HTML生成"""
        return '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Teaching System - Visual Components Integration</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .panel { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .status-item { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .status-value { font-size: 1.5em; font-weight: bold; color: #27ae60; }
        .connected { color: #27ae60; }
        .disconnected { color: #e74c3c; }
        .control-buttons { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
        button { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        button:hover { opacity: 0.8; }
        .joints-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .joint-control { display: flex; align-items: center; gap: 10px; }
        .joint-input { padding: 8px; border: 1px solid #ddd; border-radius: 3px; width: 80px; }
        .vc-messages { background: #1e1e1e; color: #00ff00; font-family: monospace; padding: 15px; border-radius: 5px; height: 200px; overflow-y: auto; font-size: 12px; }
        .teaching-points { max-height: 300px; overflow-y: auto; }
        .point-item { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .connection-status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .vc-panel { border: 2px solid #3498db; background: #ebf3ff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Robot Teaching System - Visual Components Integration</h1>
            <p>Real-time bidirectional communication between Web UI and Visual Components</p>
        </div>
        
        <div class="panel vc-panel">
            <h3>Visual Components Connection Status</h3>
            <div class="connection-status">
                <div>TCP Server: <span id="tcp-status" class="status-value">Starting...</span></div>
                <div>VC Connection: <span id="vc-status" class="status-value disconnected">Waiting for Visual Components</span></div>
                <div>Port: <strong>8888</strong> | Web UI: <strong>8080</strong></div>
            </div>
            <div class="control-buttons">
                <button class="btn-primary" onclick="sendToVC('start_demo')">Start VC Demo</button>
                <button class="btn-warning" onclick="sendToVC('stop_demo')">Stop VC Demo</button>
                <button class="btn-success" onclick="sendToVC('home_position')">Home Position</button>
            </div>
        </div>
        
        <div class="status-grid">
            <div class="panel">
                <h3>Robot Status</h3>
                <div class="status-item">
                    <div>Operation: <span id="current-operation" class="status-value">Idle</span></div>
                </div>
                <div class="status-item">
                    <div>Safety: <span id="safety-status" class="status-value">OK</span></div>
                </div>
                
                <h4>Joint Positions (from VC)</h4>
                <div class="joints-grid">
                    <div class="joint-control">
                        <label>J1:</label>
                        <input type="number" id="joint-j1" class="joint-input" step="0.1" onchange="updateJoint()">
                    </div>
                    <div class="joint-control">
                        <label>J2:</label>
                        <input type="number" id="joint-j2" class="joint-input" step="0.1" onchange="updateJoint()">
                    </div>
                    <div class="joint-control">
                        <label>J3:</label>
                        <input type="number" id="joint-j3" class="joint-input" step="0.1" onchange="updateJoint()">
                    </div>
                    <div class="joint-control">
                        <label>J4:</label>
                        <input type="number" id="joint-j4" class="joint-input" step="0.1" onchange="updateJoint()">
                    </div>
                    <div class="joint-control">
                        <label>J5:</label>
                        <input type="number" id="joint-j5" class="joint-input" step="0.1" onchange="updateJoint()">
                    </div>
                    <div class="joint-control">
                        <label>J6:</label>
                        <input type="number" id="joint-j6" class="joint-input" step="0.1" onchange="updateJoint()">
                    </div>
                </div>
                <div class="control-buttons">
                    <button class="btn-primary" onclick="moveRobot()">Move Robot</button>
                    <button class="btn-success" onclick="teachPoint()">Teach Point</button>
                </div>
            </div>
            
            <div class="panel">
                <h3>Production Stats</h3>
                <div class="status-item">
                    <div>Completed Parts: <span id="completed-parts" class="status-value">0</span></div>
                </div>
                <div class="status-item">
                    <div>Cycle Time: <span id="cycle-time" class="status-value">8.5s</span></div>
                </div>
                <div class="status-item">
                    <div>Efficiency: <span id="efficiency" class="status-value">95.2%</span></div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h3>Teaching Points</h3>
            <div id="teaching-points" class="teaching-points">
                <p>No teaching points recorded yet.</p>
            </div>
        </div>
        
        <div class="panel">
            <h3>Visual Components Communication Log</h3>
            <div id="vc-messages" class="vc-messages">
                Waiting for Visual Components connection...<br>
                TCP Server listening on port 8888<br>
                Connect from Visual Components using TCP client<br>
            </div>
        </div>
    </div>
    
    <script>
        let teachPointCounter = 0;
        
        // データ更新
        function updateStatus() {
            fetch('/api/robot_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('tcp-status').textContent = data.connected ? 'Active' : 'Inactive';
                    document.getElementById('vc-status').textContent = data.vc_connected ? 'Connected' : 'Waiting for VC';
                    document.getElementById('vc-status').className = 'status-value ' + (data.vc_connected ? 'connected' : 'disconnected');
                    
                    document.getElementById('current-operation').textContent = data.current_operation;
                    document.getElementById('safety-status').textContent = data.safety_status;
                    
                    // ジョイント位置更新
                    document.getElementById('joint-j1').value = data.joints.J1;
                    document.getElementById('joint-j2').value = data.joints.J2;
                    document.getElementById('joint-j3').value = data.joints.J3;
                    document.getElementById('joint-j4').value = data.joints.J4;
                    document.getElementById('joint-j5').value = data.joints.J5;
                    document.getElementById('joint-j6').value = data.joints.J6;
                });
            
            fetch('/api/production_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('completed-parts').textContent = data.completed_parts;
                    document.getElementById('cycle-time').textContent = data.cycle_time + 's';
                    document.getElementById('efficiency').textContent = data.efficiency + '%';
                });
            
            fetch('/api/teaching_points')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('teaching-points');
                    if (data.length === 0) {
                        container.innerHTML = '<p>No teaching points recorded yet.</p>';
                    } else {
                        container.innerHTML = data.map((point, idx) => 
                            `<div class="point-item">
                                <strong>${point.name}</strong><br>
                                Joints: J1:${point.joints.J1}, J2:${point.joints.J2}, J3:${point.joints.J3}
                            </div>`
                        ).join('');
                    }
                });
            
            // Visual Componentsメッセージ更新
            fetch('/api/vc_messages')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('vc-messages');
                    if (data.messages.length > 0) {
                        container.innerHTML = data.messages.join('<br>');
                        container.scrollTop = container.scrollHeight;
                    }
                });
        }
        
        // ロボット移動
        function moveRobot() {
            const joints = {
                J1: parseFloat(document.getElementById('joint-j1').value),
                J2: parseFloat(document.getElementById('joint-j2').value),
                J3: parseFloat(document.getElementById('joint-j3').value),
                J4: parseFloat(document.getElementById('joint-j4').value),
                J5: parseFloat(document.getElementById('joint-j5').value),
                J6: parseFloat(document.getElementById('joint-j6').value)
            };
            
            fetch('/api/move_robot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({joints})
            });
        }
        
        // 教示ポイント記録
        function teachPoint() {
            teachPointCounter++;
            const pointName = `Point_${teachPointCounter}`;
            
            fetch('/api/teach_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: pointName})
            });
        }
        
        // Visual Componentsへのコマンド送信
        function sendToVC(command) {
            fetch('/api/send_to_vc', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command})
            });
        }
        
        // ジョイント値更新
        function updateJoint() {
            // リアルタイム更新は moveRobot() を呼ぶか、自動更新設定
        }
        
        // 定期更新
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>'''

def main():
    """メイン実行関数"""
    print("Integrated Robot Teaching System")
    print("Web UI with Visual Components Integration")
    print("=" * 50)
    
    system = IntegratedRobotSystem(web_port=8080, tcp_port=8888)
    
    # TCPサーバー開始
    if not system.start_tcp_server():
        print("Failed to start TCP server")
        return
    
    # Webサーバー開始
    if not system.start_web_server():
        print("Failed to start web server")
        return
    
    print("\n=== System Ready ===")
    print("Web UI: http://localhost:8080")
    print("TCP Server: localhost:8888 (for Visual Components)")
    print("\nNext steps:")
    print("1. Open http://localhost:8080 in your browser")
    print("2. Start Visual Components")
    print("3. Run VC connection script to connect to localhost:8888")
    print("4. Watch real-time bidirectional communication!")
    
    try:
        print("\nPress Ctrl+C to stop all servers")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop_servers()

if __name__ == "__main__":
    main()