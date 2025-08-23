# Visual Components用の最もシンプルなスクリプト
# このコードをVisual ComponentsのPythonスクリプトにコピーして実行

import socket
import json
import time

# 接続設定
HOST = "localhost"
PORT = 8888

print("Connecting to Web UI...")

try:
    # 接続
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    print("Connected!")
    
    # ロボット取得
    try:
        robot = getComponent("Robot")
        print("Robot found")
    except:
        robot = None
        print("No robot - simulation mode")
    
    # 接続通知
    msg = {
        "command_type": "robot_status",
        "target_component": "robot_1", 
        "parameters": {
            "operation": "Visual Components Connected!",
            "joint_positions": [0, 0, 0, 0, 0, 0]
        }
    }
    client.send((json.dumps(msg) + '\n').encode())
    print("Status sent to Web UI")
    
    # 基本動作
    positions = [
        [0, 0, 0, 0, 0, 0],
        [30, -20, 40, 0, 0, 0], 
        [0, -30, 50, 0, 0, 45],
        [0, 0, 0, 0, 0, 0]
    ]
    
    for i, pos in enumerate(positions):
        print("Moving to position " + str(i+1))
        
        # ロボット移動
        if robot:
            robot.J1, robot.J2, robot.J3 = pos[0], pos[1], pos[2]
            robot.J4, robot.J5, robot.J6 = pos[3], pos[4], pos[5]
        
        # WebUIに送信
        move_msg = {
            "command_type": "robot_move",
            "target_component": "robot_1",
            "parameters": {
                "joint_positions": pos,
                "operation": "Position " + str(i+1)
            }
        }
        client.send((json.dumps(move_msg) + '\n').encode())
        time.sleep(2)
    
    print("Demo completed - check Web UI!")
    print("http://localhost:8080")
    
except Exception as e:
    print("Error: " + str(e))
finally:
    try:
        client.close()
    except:
        pass