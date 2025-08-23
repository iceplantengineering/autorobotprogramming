# Visual Components Compatible Client
# Visual Componentsの古いPython環境対応版

import socket
import json
import time
import threading

def vc_compatible_connection():
    """Visual Components互換接続"""
    HOST = "localhost"
    PORT = 8888
    
    print("=" * 50)
    print("Visual Components Web UI Integration")
    print("=" * 50)
    
    client = None
    
    try:
        # TCP接続
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect((HOST, PORT))
        
        print("Connected to Web UI System")
        print("Server: " + HOST + ":" + str(PORT))
        
        # Visual Componentsロボット取得 (エラーハンドリング改善)
        robot = None
        try:
            robot = getComponent("Robot")
            print("Robot component found")
        except:
            print("Robot component not found - using simulation mode")
            robot = None
        
        # 初期接続メッセージ送信
        connection_msg = {
            "command_type": "robot_status",
            "target_component": "robot_1",
            "parameters": {
                "operation": "Visual Components Connected",
                "joint_positions": [0, 0, 0, 0, 0, 0],
                "tcp_position": [0, 0, 200, 0, 0, 0]
            }
        }
        
        client.send((json.dumps(connection_msg) + '\n').encode())
        print("Initial connection message sent")
        
        # 基本位置移動デモ
        print("\n=== Basic Movement Demo ===")
        demo_positions = [
            {"name": "Home", "joints": [0, 0, 0, 0, 0, 0]},
            {"name": "Position 1", "joints": [30, -20, 40, 0, 0, 0]},
            {"name": "Pick Approach", "joints": [25, -35, 50, 0, 20, 0]},
            {"name": "Pick Position", "joints": [25, -40, 55, 0, 25, 0]},
            {"name": "Transfer", "joints": [0, -20, 30, 0, 0, 45]},
            {"name": "Place Approach", "joints": [-25, -30, 45, 0, 15, 45]},
            {"name": "Place Position", "joints": [-25, -35, 50, 0, 20, 45]},
            {"name": "Home", "joints": [0, 0, 0, 0, 0, 0]}
        ]
        
        for i, pos in enumerate(demo_positions):
            print("Step " + str(i+1) + ": " + pos["name"])
            
            # Visual Componentsロボット制御
            if robot:
                try:
                    joints = pos["joints"]
                    robot.J1, robot.J2, robot.J3 = joints[0], joints[1], joints[2]
                    robot.J4, robot.J5, robot.J6 = joints[3], joints[4], joints[5]
                    print("  Robot moved to: " + str(joints))
                except Exception as e:
                    print("  Robot move error: " + str(e))
            
            # WebUIに移動コマンド送信
            move_msg = {
                "command_type": "robot_move",
                "target_component": "robot_1",
                "parameters": {
                    "joint_positions": pos["joints"],
                    "operation": "Moving to " + pos["name"]
                }
            }
            
            client.send((json.dumps(move_msg) + '\n').encode())
            
            # I/O制御
            if pos["name"] == "Pick Position":
                io_msg = {
                    "command_type": "io_control",
                    "target_component": "robot_1",
                    "parameters": {
                        "io_data": {"gripper": True, "WORKING_LAMP": True}
                    }
                }
                client.send((json.dumps(io_msg) + '\n').encode())
                print("  -> Gripper activated")
                
            elif pos["name"] == "Place Position":
                io_msg = {
                    "command_type": "io_control",
                    "target_component": "robot_1",
                    "parameters": {
                        "io_data": {"gripper": False, "WORKING_LAMP": False}
                    }
                }
                client.send((json.dumps(io_msg) + '\n').encode())
                print("  -> Gripper released")
            
            # 作業ステップ送信
            step_msg = {
                "command_type": "operation_step",
                "target_component": "robot_1",
                "parameters": {
                    "step_name": pos["name"] + " completed",
                    "joint_positions": pos["joints"]
                }
            }
            client.send((json.dumps(step_msg) + '\n').encode())
            
            time.sleep(3)  # 動作時間
        
        print("\n=== Starting Continuous Communication ===")
        print("Web UI: http://localhost:8080")
        print("Real-time position updates every 2 seconds...")
        
        # 継続的な位置更新
        counter = 0
        while True:
            try:
                # 現在のロボット位置取得
                if robot:
                    try:
                        current_joints = [robot.J1, robot.J2, robot.J3, robot.J4, robot.J5, robot.J6]
                    except:
                        current_joints = [0, 0, 0, 0, 0, 0]
                else:
                    # シミュレーション用ランダム位置
                    import random
                    current_joints = [
                        round(random.uniform(-45, 45), 1),
                        round(random.uniform(-30, 30), 1),
                        round(random.uniform(-20, 40), 1),
                        round(random.uniform(-45, 45), 1),
                        round(random.uniform(-30, 30), 1),
                        round(random.uniform(-45, 45), 1)
                    ]
                
                # WebUIに状態送信
                status_msg = {
                    "command_type": "robot_status",
                    "target_component": "robot_1",
                    "parameters": {
                        "joint_positions": current_joints,
                        "operation": "Live communication #" + str(counter),
                        "timestamp": time.time()
                    }
                }
                
                client.send((json.dumps(status_msg) + '\n').encode())
                
                # 定期的な作業ステップ報告
                if counter % 5 == 0 and counter > 0:
                    step_msg = {
                        "command_type": "operation_step",
                        "target_component": "robot_1",
                        "parameters": {
                            "step_name": "Cycle " + str(counter // 5) + " completed",
                            "joint_positions": current_joints
                        }
                    }
                    client.send((json.dumps(step_msg) + '\n').encode())
                    print("Cycle " + str(counter // 5) + " completed")
                
                counter += 1
                time.sleep(2)  # 2秒間隔
                
            except Exception as e:
                print("Communication error: " + str(e))
                break
        
    except Exception as e:
        # 古いPython環境用のエラーハンドリング
        error_msg = str(e)
        if "Connection refused" in error_msg or "10061" in error_msg:
            print("Connection refused - Web server not running")
            print("Please start: python integrated_web_server.py")
        elif "timed out" in error_msg:
            print("Connection timeout")
        else:
            print("Connection error: " + error_msg)
    
    finally:
        if client:
            try:
                client.close()
                print("Connection closed")
            except:
                pass

# メイン実行
vc_compatible_connection()