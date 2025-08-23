#!/usr/bin/env python3
"""
Simple Visual Components Connection Test
Visual Componentsなしでも動作する簡易接続テスト
"""

import socket
import json
import time
import threading
import random

def simple_vc_connection_test():
    """簡易VC接続テスト"""
    HOST = "localhost"
    PORT = 8888
    
    print("=" * 50)
    print("Simple VC Connection Test")
    print("Connecting to Web UI Integration System")
    print("=" * 50)
    
    try:
        # TCP接続
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect((HOST, PORT))
        
        print(f"Connected to {HOST}:{PORT}")
        print("TCP connection established")
        
        # 初期接続メッセージ送信
        connection_msg = {
            "command_type": "robot_status",
            "target_component": "robot_1",
            "parameters": {
                "operation": "Simple Test Client Connected",
                "joint_positions": [0, 0, 0, 0, 0, 0],
                "tcp_position": [0, 0, 200, 0, 0, 0]
            }
        }
        
        client.send((json.dumps(connection_msg) + '\n').encode())
        print("Initial connection message sent")
        
        def simulate_robot_positions():
            """ロボット位置をシミュレーション"""
            positions = [
                [0, 0, 0, 0, 0, 0],          # Home
                [30, -20, 40, 0, 0, 0],      # Position 1
                [25, -35, 50, 0, 20, 0],     # Pick approach
                [25, -40, 55, 0, 25, 0],     # Pick position
                [0, -20, 30, 0, 0, 45],      # Transfer
                [-25, -30, 45, 0, 15, 45],   # Place approach
                [-25, -35, 50, 0, 20, 45],   # Place position
                [0, 0, 0, 0, 0, 0]           # Home
            ]
            
            for i, pos in enumerate(positions):
                try:
                    # 位置更新メッセージ
                    move_msg = {
                        "command_type": "robot_move",
                        "target_component": "robot_1",
                        "parameters": {
                            "joint_positions": pos,
                            "operation": f"Moving to test position {i+1}"
                        }
                    }
                    
                    client.send((json.dumps(move_msg) + '\n').encode())
                    print(f"→ Position {i+1}: {pos}")
                    
                    # I/O制御シミュレーション
                    if i == 3:  # Pick position
                        io_msg = {
                            "command_type": "io_control",
                            "target_component": "robot_1",
                            "parameters": {
                                "io_data": {"gripper": True, "WORKING_LAMP": True}
                            }
                        }
                        client.send((json.dumps(io_msg) + '\n').encode())
                        print("  → Gripper ON")
                        
                    elif i == 6:  # Place position
                        io_msg = {
                            "command_type": "io_control",
                            "target_component": "robot_1",
                            "parameters": {
                                "io_data": {"gripper": False, "WORKING_LAMP": False}
                            }
                        }
                        client.send((json.dumps(io_msg) + '\n').encode())
                        print("  → Gripper OFF")
                    
                    time.sleep(2)  # 動作時間シミュレーション
                    
                except Exception as e:
                    print(f"Error sending position {i+1}: {e}")
                    break
        
        def continuous_status_updates():
            """継続的な状態更新"""
            counter = 0
            while True:
                try:
                    # ランダム位置生成
                    random_joints = [
                        round(random.uniform(-45, 45), 1),   # J1
                        round(random.uniform(-30, 30), 1),   # J2
                        round(random.uniform(-20, 40), 1),   # J3
                        round(random.uniform(-45, 45), 1),   # J4
                        round(random.uniform(-30, 30), 1),   # J5
                        round(random.uniform(-45, 45), 1)    # J6
                    ]
                    
                    status_msg = {
                        "command_type": "robot_status",
                        "target_component": "robot_1",
                        "parameters": {
                            "joint_positions": random_joints,
                            "operation": f"Live simulation #{counter}",
                            "timestamp": time.time()
                        }
                    }
                    
                    client.send((json.dumps(status_msg) + '\n').encode())
                    counter += 1
                    
                    # 10秒ごとに作業ステップ送信
                    if counter % 5 == 0:
                        step_msg = {
                            "command_type": "operation_step",
                            "target_component": "robot_1",
                            "parameters": {
                                "step_name": f"Operation Step {counter // 5}",
                                "joint_positions": random_joints
                            }
                        }
                        client.send((json.dumps(step_msg) + '\n').encode())
                        print(f"Operation step {counter // 5} completed")
                    
                    time.sleep(2)  # 2秒間隔で更新
                    
                except Exception as e:
                    print(f"Status update error: {e}")
                    break
        
        # 基本動作シーケンス実行
        print("\n=== Basic Movement Sequence ===")
        simulate_robot_positions()
        
        print("\n=== Starting Continuous Updates ===")
        print("WebUI shows real-time communication!")
        print("Check http://localhost:8080 for live updates")
        print("Press Ctrl+C to stop...")
        
        # 継続更新開始
        continuous_status_updates()
        
    except ConnectionRefusedError:
        print("Connection refused - Web server not running")
        print("Please start: python integrated_web_server.py")
        
    except socket.timeout:
        print("Connection timeout")
        
    except Exception as e:
        print(f"Connection error: {e}")
        
    finally:
        if 'client' in locals():
            try:
                client.close()
                print("Connection closed")
            except:
                pass

if __name__ == "__main__":
    simple_vc_connection_test()