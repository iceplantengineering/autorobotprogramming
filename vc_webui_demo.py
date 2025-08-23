# Visual Components Script for Web UI Integration Demo
# Visual ComponentsでこのスクリプトをPythonコードとして実行してください

import socket
import json
import time
import threading

# Visual Components用 - エラーハンドリング強化版
def safe_connect_to_webui():
    """WebUI統合システムへの安全な接続"""
    HOST = "localhost"
    PORT = 8888
    
    print("=" * 50)
    print("Visual Components - Web UI Integration Demo")
    print("=" * 50)
    
    try:
        # TCP接続
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10.0)
        client.connect((HOST, PORT))
        
        print("Connected to Web UI Integration System")
        print("Server: " + HOST + ":" + str(PORT))
        
        # Visual Componentsロボット取得
        try:
            robot = getComponent("Robot")
            print("Robot component found: " + str(robot))
        except:
            print("Warning: Robot component not found - using simulation mode")
            robot = None
        
        # 接続確認メッセージ送信
        connection_msg = {
            "command_type": "robot_status",
            "target_component": "robot_1",
            "parameters": {
                "operation": "Visual Components Connected",
                "vc_version": "4.0+",
                "joint_positions": [0, 0, 0, 0, 0, 0],
                "tcp_position": [0, 0, 200, 0, 0, 0]
            }
        }
        
        client.send((json.dumps(connection_msg) + '\n').encode())
        print("Connection status sent to Web UI")
        
        # リアルタイム位置送信スレッド開始
        def position_sender():
            """位置情報をWebUIにリアルタイム送信"""
            while True:
                try:
                    if robot:
                        # 実際のロボット位置取得
                        current_joints = [robot.J1, robot.J2, robot.J3, robot.J4, robot.J5, robot.J6]
                    else:
                        # シミュレーション用ランダム位置
                        import random
                        current_joints = [
                            random.uniform(-45, 45),  # J1
                            random.uniform(-30, 30),  # J2  
                            random.uniform(-20, 40),  # J3
                            random.uniform(-45, 45),  # J4
                            random.uniform(-30, 30),  # J5
                            random.uniform(-45, 45)   # J6
                        ]
                    
                    # WebUIに位置情報送信
                    status_msg = {
                        "command_type": "robot_status",
                        "target_component": "robot_1", 
                        "parameters": {
                            "joint_positions": current_joints,
                            "operation": "Active",
                            "timestamp": time.time()
                        }
                    }
                    
                    client.send((json.dumps(status_msg) + '\n').encode())
                    time.sleep(2)  # 2秒間隔で送信
                    
                except Exception as e:
                    print("Position sender error: " + str(e))
                    break
        
        # 位置送信スレッド開始
        pos_thread = threading.Thread(target=position_sender, daemon=True)
        pos_thread.start()
        
        print("\n=== Real-time Communication Started ===")
        print("Web UI URL: http://localhost:8080")
        print("Position data being sent every 2 seconds...")
        print("\nDemonstration scenarios:")
        
        # デモシナリオ1: 基本位置移動
        print("\n1. Basic Position Movement Demo")
        demo_positions = [
            [0, 0, 0, 0, 0, 0],       # Home
            [30, -20, 40, 0, 0, 0],   # Position 1
            [-20, -30, 50, 0, 15, 0], # Position 2
            [15, -25, 35, 0, 10, 0],  # Position 3
            [0, 0, 0, 0, 0, 0]        # Home
        ]
        
        for i, pos in enumerate(demo_positions):
            print("Moving to demo position " + str(i+1) + ": " + str(pos))
            
            if robot:
                robot.J1, robot.J2, robot.J3 = pos[0], pos[1], pos[2]
                robot.J4, robot.J5, robot.J6 = pos[3], pos[4], pos[5]
            
            # WebUIに移動開始通知
            move_msg = {
                "command_type": "robot_move",
                "target_component": "robot_1",
                "parameters": {
                    "joint_positions": pos,
                    "operation": "Moving to Position " + str(i+1)
                }
            }
            client.send((json.dumps(move_msg) + '\n').encode())
            
            time.sleep(3)  # 移動時間待機
        
        # デモシナリオ2: ピック&プレース作業
        print("\n2. Pick and Place Operation Demo")
        pick_place_sequence = [
            {"step": "Approach Pick", "joints": [25, -35, 50, 0, 20, 0]},
            {"step": "Pick Position", "joints": [25, -40, 55, 0, 25, 0]},
            {"step": "Pick Complete", "joints": [25, -35, 50, 0, 20, 0]},
            {"step": "Transfer", "joints": [0, -20, 30, 0, 0, 45]},
            {"step": "Approach Place", "joints": [-25, -30, 45, 0, 15, 45]},
            {"step": "Place Position", "joints": [-25, -35, 50, 0, 20, 45]},
            {"step": "Place Complete", "joints": [-25, -30, 45, 0, 15, 45]},
            {"step": "Return Home", "joints": [0, 0, 0, 0, 0, 0]}
        ]
        
        for step in pick_place_sequence:
            print("Executing: " + step["step"])
            
            if robot:
                joints = step["joints"]
                robot.J1, robot.J2, robot.J3 = joints[0], joints[1], joints[2]
                robot.J4, robot.J5, robot.J6 = joints[3], joints[4], joints[5]
            
            # WebUIに作業ステップ送信
            step_msg = {
                "command_type": "operation_step",
                "target_component": "robot_1",
                "parameters": {
                    "step_name": step["step"],
                    "joint_positions": step["joints"]
                }
            }
            client.send((json.dumps(step_msg) + '\n').encode())
            
            # I/O制御シミュレーション
            if step["step"] == "Pick Complete":
                io_msg = {
                    "command_type": "io_control",
                    "target_component": "robot_1",
                    "parameters": {"io_data": {"gripper": True, "WORKING_LAMP": True}}
                }
                client.send((json.dumps(io_msg) + '\n').encode())
                print("  -> Gripper activated")
                
            elif step["step"] == "Place Complete":
                io_msg = {
                    "command_type": "io_control", 
                    "target_component": "robot_1",
                    "parameters": {"io_data": {"gripper": False, "WORKING_LAMP": False}}
                }
                client.send((json.dumps(io_msg) + '\n').encode())
                print("  -> Gripper released")
            
            time.sleep(2.5)  # 作業時間待機
        
        print("\n=== Demo Completed Successfully ===")
        print("Check the Web UI at http://localhost:8080 for real-time updates!")
        print("- Robot positions synchronized")
        print("- Operation status updated")
        print("- I/O signals controlled")
        print("- Production statistics tracked")
        
        # 継続通信
        print("\nContinuous communication active...")
        print("Web UI and Visual Components are now synchronized!")
        
        # 無限ループ（実際の運用では必要に応じて制御）
        try:
            while True:
                time.sleep(10)
                # 定期的なヘルスチェックメッセージ
                health_msg = {
                    "command_type": "robot_status",
                    "target_component": "robot_1",
                    "parameters": {"operation": "System Running", "health": "OK"}
                }
                client.send((json.dumps(health_msg) + '\n').encode())
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        
    except ConnectionRefusedError:
        print("Error: Could not connect to Web UI server")
        print("Make sure the integrated web server is running:")
        print("python integrated_web_server.py")
        
    except Exception as e:
        print("Connection error: " + str(e))
        
    finally:
        if 'client' in locals():
            client.close()
        print("Connection closed")

# デモ実行
safe_connect_to_webui()