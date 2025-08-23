import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config_manager import config_manager
from basic_handling_workflow import Position, WorkPiece

logger = logging.getLogger(__name__)

class TrajectoryType(Enum):
    LINEAR = "linear"
    CIRCULAR = "circular"
    SPLINE = "spline"
    JOINT = "joint"

class InterpolationMethod(Enum):
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    POLYNOMIAL = "polynomial"
    TRAPEZOIDAL = "trapezoidal"

@dataclass
class TrajectoryPoint:
    position: Position
    speed: float
    acceleration: Optional[float] = None
    timestamp: Optional[float] = None
    description: str = ""

@dataclass
class TrajectorySegment:
    start_point: TrajectoryPoint
    end_point: TrajectoryPoint
    trajectory_type: TrajectoryType
    interpolation_method: InterpolationMethod
    duration: float
    intermediate_points: List[TrajectoryPoint] = None

class CollisionChecker:
    """衝突検査システム"""
    
    def __init__(self):
        self.obstacles: List[Dict[str, Any]] = []
        self.robot_envelope = self._load_robot_envelope()
        self.safety_margin = 10.0  # mm
    
    def _load_robot_envelope(self) -> Dict[str, Any]:
        """ロボット動作範囲読み込み"""
        robot_limits = config_manager.get_robot_limits()
        workspace_limits = robot_limits.get("workspace_limits", {})
        
        return {
            "x_min": workspace_limits.get("x_min", -1000.0),
            "x_max": workspace_limits.get("x_max", 1000.0),
            "y_min": workspace_limits.get("y_min", -1000.0),
            "y_max": workspace_limits.get("y_max", 1000.0),
            "z_min": workspace_limits.get("z_min", 0.0),
            "z_max": workspace_limits.get("z_max", 1000.0),
            "reach_radius": 1200.0  # ロボットリーチ半径
        }
    
    def add_obstacle(self, obstacle: Dict[str, Any]):
        """障害物追加"""
        self.obstacles.append(obstacle)
        logger.debug(f"Added obstacle: {obstacle.get('name', 'unnamed')}")
    
    def clear_obstacles(self):
        """障害物クリア"""
        self.obstacles.clear()
    
    def check_point_collision(self, point: Position) -> bool:
        """点の衝突チェック"""
        # ワークスペース制限チェック
        if not self._is_within_workspace(point):
            return True
        
        # 障害物との衝突チェック
        for obstacle in self.obstacles:
            if self._point_in_obstacle(point, obstacle):
                return True
        
        return False
    
    def check_trajectory_collision(self, trajectory: List[TrajectoryPoint]) -> List[int]:
        """軌道の衝突チェック"""
        collision_points = []
        
        for i, traj_point in enumerate(trajectory):
            if self.check_point_collision(traj_point.position):
                collision_points.append(i)
        
        return collision_points
    
    def _is_within_workspace(self, point: Position) -> bool:
        """ワークスペース内チェック"""
        envelope = self.robot_envelope
        
        return (
            envelope["x_min"] <= point.x <= envelope["x_max"] and
            envelope["y_min"] <= point.y <= envelope["y_max"] and
            envelope["z_min"] <= point.z <= envelope["z_max"]
        )
    
    def _point_in_obstacle(self, point: Position, obstacle: Dict[str, Any]) -> bool:
        """点が障害物内にあるかチェック"""
        obs_type = obstacle.get("type", "box")
        
        if obs_type == "box":
            return self._point_in_box(point, obstacle)
        elif obs_type == "cylinder":
            return self._point_in_cylinder(point, obstacle)
        elif obs_type == "sphere":
            return self._point_in_sphere(point, obstacle)
        
        return False
    
    def _point_in_box(self, point: Position, box: Dict[str, Any]) -> bool:
        """箱型障害物との衝突チェック"""
        center = box.get("center", [0, 0, 0])
        dimensions = box.get("dimensions", [100, 100, 100])
        
        return (
            abs(point.x - center[0]) <= dimensions[0]/2 + self.safety_margin and
            abs(point.y - center[1]) <= dimensions[1]/2 + self.safety_margin and
            abs(point.z - center[2]) <= dimensions[2]/2 + self.safety_margin
        )
    
    def _point_in_cylinder(self, point: Position, cylinder: Dict[str, Any]) -> bool:
        """円柱型障害物との衝突チェック"""
        center = cylinder.get("center", [0, 0, 0])
        radius = cylinder.get("radius", 50.0)
        height = cylinder.get("height", 100.0)
        
        # 水平距離チェック
        horizontal_dist = math.sqrt((point.x - center[0])**2 + (point.y - center[1])**2)
        if horizontal_dist > radius + self.safety_margin:
            return False
        
        # 高さチェック
        return abs(point.z - center[2]) <= height/2 + self.safety_margin
    
    def _point_in_sphere(self, point: Position, sphere: Dict[str, Any]) -> bool:
        """球型障害物との衝突チェック"""
        center = sphere.get("center", [0, 0, 0])
        radius = sphere.get("radius", 50.0)
        
        distance = math.sqrt(
            (point.x - center[0])**2 +
            (point.y - center[1])**2 +
            (point.z - center[2])**2
        )
        
        return distance <= radius + self.safety_margin

class AdvancedTrajectoryGenerator:
    """高度な軌道生成システム"""
    
    def __init__(self):
        self.collision_checker = CollisionChecker()
        self.robot_config = config_manager.get_config_value("APPLICATION", "robot.default_settings", {})
        self.robot_limits = config_manager.get_robot_limits()
        
        # デフォルト設定
        self.default_speed = self.robot_config.get("move_speed", 100)
        self.default_acceleration = self.robot_limits.get("max_acceleration", 100)
        self.safety_height = self.robot_config.get("safety_height", 50.0)
        
    def generate_pick_place_trajectory(self, 
                                     pick_position: Position, 
                                     place_position: Position,
                                     workpiece: WorkPiece,
                                     parameters: Dict[str, Any] = None) -> List[TrajectoryPoint]:
        """ピック&プレース軌道生成"""
        params = parameters or {}
        
        approach_speed = params.get("approach_speed", 80)
        work_speed = params.get("work_speed", 30)
        safety_height = params.get("safety_height", self.safety_height)
        
        trajectory = []
        
        # 1. 現在位置（仮想的なホーム位置）
        home_position = Position(0, 0, 300, 0, 0, 0)
        trajectory.append(TrajectoryPoint(
            position=home_position,
            speed=approach_speed,
            description="Start from home position"
        ))
        
        # 2. ピック接近位置
        pick_approach = pick_position.add_offset(Position(0, 0, safety_height, 0, 0, 0))
        trajectory.append(TrajectoryPoint(
            position=pick_approach,
            speed=approach_speed,
            description="Move to pick approach position"
        ))
        
        # 3. ピック位置
        trajectory.append(TrajectoryPoint(
            position=pick_position,
            speed=work_speed,
            description="Move to pick position"
        ))
        
        # 4. ピック接近位置に戻る
        trajectory.append(TrajectoryPoint(
            position=pick_approach,
            speed=work_speed,
            description="Lift workpiece"
        ))
        
        # 5. 中間安全位置（必要に応じて）
        if self._requires_intermediate_position(pick_position, place_position):
            intermediate_pos = self._calculate_intermediate_position(pick_position, place_position, safety_height)
            trajectory.append(TrajectoryPoint(
                position=intermediate_pos,
                speed=approach_speed,
                description="Move through intermediate safe position"
            ))
        
        # 6. プレース接近位置
        place_approach = place_position.add_offset(Position(0, 0, safety_height, 0, 0, 0))
        trajectory.append(TrajectoryPoint(
            position=place_approach,
            speed=approach_speed,
            description="Move to place approach position"
        ))
        
        # 7. プレース位置
        trajectory.append(TrajectoryPoint(
            position=place_position,
            speed=work_speed,
            description="Move to place position"
        ))
        
        # 8. プレース接近位置に戻る
        trajectory.append(TrajectoryPoint(
            position=place_approach,
            speed=work_speed,
            description="Retract from place position"
        ))
        
        # 9. ホーム位置に戻る（オプション）
        if params.get("return_home", False):
            trajectory.append(TrajectoryPoint(
                position=home_position,
                speed=approach_speed,
                description="Return to home position"
            ))
        
        # 衝突チェックと軌道最適化
        trajectory = self._optimize_trajectory(trajectory)
        
        return trajectory
    
    def generate_multi_point_trajectory(self,
                                      pick_points: List[Position],
                                      place_points: List[Position],
                                      workpiece: WorkPiece,
                                      parameters: Dict[str, Any] = None) -> List[TrajectoryPoint]:
        """マルチポイント軌道生成"""
        params = parameters or {}
        cycle_mode = params.get("cycle_mode", "sequential")
        
        if len(pick_points) != len(place_points):
            raise ValueError("Number of pick and place points must match")
        
        trajectory = []
        
        if cycle_mode == "sequential":
            # 順次処理
            for i, (pick_pos, place_pos) in enumerate(zip(pick_points, place_points)):
                single_trajectory = self.generate_pick_place_trajectory(
                    pick_pos, place_pos, workpiece, parameters
                )
                
                # 最初以外はホーム位置から開始しない
                if i > 0:
                    single_trajectory = single_trajectory[1:]
                
                trajectory.extend(single_trajectory)
        
        elif cycle_mode == "optimized":
            # 移動距離最小化
            optimized_order = self._optimize_multi_point_order(pick_points, place_points)
            for i in optimized_order:
                single_trajectory = self.generate_pick_place_trajectory(
                    pick_points[i], place_points[i], workpiece, parameters
                )
                
                if len(trajectory) > 0:
                    single_trajectory = single_trajectory[1:]
                
                trajectory.extend(single_trajectory)
        
        return trajectory
    
    def generate_conveyor_tracking_trajectory(self,
                                            conveyor_config: Dict[str, Any],
                                            place_position: Position,
                                            workpiece: WorkPiece,
                                            parameters: Dict[str, Any] = None) -> List[TrajectoryPoint]:
        """コンベア追従軌道生成"""
        params = parameters or {}
        
        conveyor_speed = conveyor_config.get("speed", 50.0)  # mm/s
        conveyor_direction = conveyor_config.get("direction", [1, 0, 0])
        pick_position = Position.from_list(conveyor_config.get("pick_position", [0, 0, 0, 0, 0, 0]))
        
        # コンベア追従のための追加ポイント計算
        tracking_distance = params.get("tracking_distance", 200.0)
        tracking_speed = params.get("tracking_speed", conveyor_speed * 1.1)
        
        trajectory = []
        
        # 1. コンベア上部待機位置
        wait_position = pick_position.add_offset(Position(0, 0, 100, 0, 0, 0))
        trajectory.append(TrajectoryPoint(
            position=wait_position,
            speed=80,
            description="Wait above conveyor"
        ))
        
        # 2. 追従開始位置
        track_start = pick_position.add_offset(
            Position(-tracking_distance/2 * conveyor_direction[0],
                    -tracking_distance/2 * conveyor_direction[1],
                    50, 0, 0, 0)
        )
        trajectory.append(TrajectoryPoint(
            position=track_start,
            speed=int(tracking_speed),
            description="Start tracking position"
        ))
        
        # 3. ピック位置（追従中）
        trajectory.append(TrajectoryPoint(
            position=pick_position,
            speed=int(tracking_speed),
            description="Pick while tracking"
        ))
        
        # 4. 追従終了位置
        track_end = pick_position.add_offset(
            Position(tracking_distance/2 * conveyor_direction[0],
                    tracking_distance/2 * conveyor_direction[1],
                    50, 0, 0, 0)
        )
        trajectory.append(TrajectoryPoint(
            position=track_end,
            speed=int(tracking_speed),
            description="End tracking position"
        ))
        
        # 5. 通常のプレース軌道
        place_trajectory = self.generate_pick_place_trajectory(
            track_end, place_position, workpiece, parameters
        )[4:]  # ピック部分を除外
        
        trajectory.extend(place_trajectory)
        
        return trajectory
    
    def generate_circular_trajectory(self,
                                   center: Position,
                                   radius: float,
                                   start_angle: float,
                                   end_angle: float,
                                   parameters: Dict[str, Any] = None) -> List[TrajectoryPoint]:
        """円弧軌道生成"""
        params = parameters or {}
        
        resolution = params.get("resolution", 10)  # 度あたりのポイント数
        speed = params.get("speed", 50)
        
        trajectory = []
        
        # 角度範囲計算
        angle_range = end_angle - start_angle
        if angle_range < 0:
            angle_range += 360
        
        num_points = max(int(angle_range * resolution / 10), 2)
        
        for i in range(num_points + 1):
            angle = start_angle + (angle_range * i / num_points)
            angle_rad = math.radians(angle)
            
            x = center.x + radius * math.cos(angle_rad)
            y = center.y + radius * math.sin(angle_rad)
            z = center.z
            
            position = Position(x, y, z, center.rx, center.ry, center.rz)
            
            trajectory.append(TrajectoryPoint(
                position=position,
                speed=speed,
                description=f"Circular path point {i+1}"
            ))
        
        return trajectory
    
    def _requires_intermediate_position(self, start: Position, end: Position) -> bool:
        """中間位置が必要かチェック"""
        # 距離による判定
        distance = math.sqrt(
            (end.x - start.x)**2 + 
            (end.y - start.y)**2 + 
            (end.z - start.z)**2
        )
        
        # 500mm以上の移動では中間位置を推奨
        return distance > 500.0
    
    def _calculate_intermediate_position(self, start: Position, end: Position, safety_height: float) -> Position:
        """中間位置計算"""
        # 中点の上空に中間位置を設定
        mid_x = (start.x + end.x) / 2
        mid_y = (start.y + end.y) / 2
        mid_z = max(start.z, end.z) + safety_height
        
        return Position(mid_x, mid_y, mid_z, 0, 0, 0)
    
    def _optimize_multi_point_order(self, pick_points: List[Position], place_points: List[Position]) -> List[int]:
        """マルチポイント順序最適化（TSP的アプローチ）"""
        n = len(pick_points)
        if n <= 2:
            return list(range(n))
        
        # 簡単な貪欲法による最適化
        visited = [False] * n
        order = []
        current = 0  # 最初のポイントから開始
        
        order.append(current)
        visited[current] = True
        
        for _ in range(n - 1):
            min_distance = float('inf')
            next_point = -1
            
            for i in range(n):
                if not visited[i]:
                    # 現在位置からピック位置、ピック位置からプレース位置の総距離
                    current_pos = place_points[current] if order else pick_points[current]
                    distance = (
                        self._calculate_distance(current_pos, pick_points[i]) +
                        self._calculate_distance(pick_points[i], place_points[i])
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        next_point = i
            
            if next_point != -1:
                order.append(next_point)
                visited[next_point] = True
                current = next_point
        
        return order
    
    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """2点間距離計算"""
        return math.sqrt(
            (pos2.x - pos1.x)**2 +
            (pos2.y - pos1.y)**2 +
            (pos2.z - pos1.z)**2
        )
    
    def _optimize_trajectory(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """軌道最適化"""
        # 衝突チェック
        collision_points = self.collision_checker.check_trajectory_collision(trajectory)
        
        if collision_points:
            logger.warning(f"Collision detected at trajectory points: {collision_points}")
            # 衝突点での高度調整
            for point_idx in collision_points:
                if point_idx < len(trajectory):
                    trajectory[point_idx].position.z += self.safety_height
        
        # 速度プロファイル最適化
        trajectory = self._optimize_speed_profile(trajectory)
        
        return trajectory
    
    def _optimize_speed_profile(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """速度プロファイル最適化"""
        if len(trajectory) < 2:
            return trajectory
        
        max_speed = self.robot_limits.get("max_speed", 100)
        max_acceleration = self.robot_limits.get("max_acceleration", 100)
        
        for i in range(1, len(trajectory) - 1):
            current = trajectory[i]
            next_point = trajectory[i + 1]
            
            # 距離に基づく速度調整
            distance = self._calculate_distance(current.position, next_point.position)
            
            if distance < 50:  # 近距離移動
                current.speed = min(current.speed, 20)
            elif distance > 500:  # 長距離移動
                current.speed = min(current.speed, max_speed)
            
            # 速度制限適用
            current.speed = max(1, min(current.speed, max_speed))
        
        return trajectory
    
    def add_obstacle(self, obstacle: Dict[str, Any]):
        """障害物追加"""
        self.collision_checker.add_obstacle(obstacle)
    
    def clear_obstacles(self):
        """障害物クリア"""
        self.collision_checker.clear_obstacles()
    
    def validate_trajectory(self, trajectory: List[TrajectoryPoint]) -> Dict[str, Any]:
        """軌道検証"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "collision_points": []
        }
        
        # 基本検証
        if not trajectory:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Empty trajectory")
            return validation_result
        
        # 衝突検証
        collision_points = self.collision_checker.check_trajectory_collision(trajectory)
        if collision_points:
            validation_result["collision_points"] = collision_points
            validation_result["warnings"].append(f"Collision detected at {len(collision_points)} points")
        
        # 速度制限検証
        max_speed = self.robot_limits.get("max_speed", 100)
        for i, point in enumerate(trajectory):
            if point.speed > max_speed:
                validation_result["errors"].append(f"Speed limit exceeded at point {i}: {point.speed}")
                validation_result["is_valid"] = False
        
        # ワークスペース検証
        for i, point in enumerate(trajectory):
            if not self.collision_checker._is_within_workspace(point.position):
                validation_result["errors"].append(f"Point {i} outside workspace")
                validation_result["is_valid"] = False
        
        return validation_result

# グローバルインスタンス
trajectory_generator = AdvancedTrajectoryGenerator()

def generate_handling_trajectory(operation_config: Dict[str, Any]) -> List[TrajectoryPoint]:
    """ハンドリング軌道生成（統一インターフェース）"""
    operation_type = operation_config.get("operation_type", "basic_pick_place")
    
    try:
        if operation_type == "basic_pick_place":
            return trajectory_generator.generate_pick_place_trajectory(
                Position.from_list(operation_config["pick_position"]),
                Position.from_list(operation_config["place_position"]),
                WorkPiece(**operation_config.get("workpiece", {})),
                operation_config.get("parameters", {})
            )
        
        elif operation_type == "multi_point_handling":
            pick_positions = [Position.from_list(pos) for pos in operation_config["pick_points"]]
            place_positions = [Position.from_list(pos) for pos in operation_config["place_points"]]
            
            return trajectory_generator.generate_multi_point_trajectory(
                pick_positions,
                place_positions,
                WorkPiece(**operation_config.get("workpiece", {})),
                operation_config.get("parameters", {})
            )
        
        elif operation_type == "conveyor_handling":
            return trajectory_generator.generate_conveyor_tracking_trajectory(
                operation_config["conveyor_config"],
                Position.from_list(operation_config["place_position"]),
                WorkPiece(**operation_config.get("workpiece", {})),
                operation_config.get("parameters", {})
            )
        
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    except Exception as e:
        logger.error(f"Trajectory generation failed: {e}")
        raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    test_config = {
        "operation_type": "basic_pick_place",
        "pick_position": [100, -200, 150, 0, 0, 0],
        "place_position": [300, 100, 150, 0, 0, 0],
        "workpiece": {
            "name": "test_part",
            "part_type": "component",
            "weight": 1.0,
            "dimensions": [50, 50, 25],
            "material": "plastic"
        },
        "parameters": {
            "approach_speed": 70,
            "work_speed": 25,
            "safety_height": 60.0,
            "return_home": True
        }
    }
    
    try:
        trajectory = generate_handling_trajectory(test_config)
        logger.info(f"Generated trajectory with {len(trajectory)} points")
        
        # 軌道検証
        validation = trajectory_generator.validate_trajectory(trajectory)
        logger.info(f"Trajectory validation: {validation}")
        
        # 詳細表示
        for i, point in enumerate(trajectory):
            logger.info(f"Point {i}: {point.position.to_list()} @ {point.speed}% - {point.description}")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")