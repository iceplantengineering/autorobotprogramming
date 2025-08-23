"""
視覚認識統合システム (Phase 3)
OpenCVを使用したワークピース位置検出・認識システム
リアルタイム画像処理とロボット制御の統合
"""

import cv2
import numpy as np
import time
import threading
import logging
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

from basic_handling_workflow import Position, WorkPiece
from integrated_safety_system import safety_system
from config_manager import config_manager

logger = logging.getLogger(__name__)

class VisionState(Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    DETECTING = "detecting"
    TRACKING = "tracking"
    ERROR = "error"

class DetectionMethod(Enum):
    CONTOUR_BASED = "contour"
    TEMPLATE_MATCHING = "template"
    FEATURE_MATCHING = "feature"
    COLOR_BASED = "color"
    ML_BASED = "ml"

@dataclass
class CameraConfig:
    """カメラ設定"""
    camera_id: int
    resolution: Tuple[int, int]  # (width, height)
    fps: int
    exposure: Optional[int] = None
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None

@dataclass
class CalibrationData:
    """カメラキャリブレーションデータ"""
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    pixel_to_mm_ratio: float
    origin_offset: Tuple[float, float, float]

@dataclass
class DetectionResult:
    """検出結果"""
    object_id: str
    position: Position  # ロボット座標系での位置
    pixel_position: Tuple[int, int]  # 画像座標
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    orientation: float  # 角度（ラジアン）
    timestamp: float
    object_type: str = "unknown"
    additional_data: Dict[str, Any] = field(default_factory=dict)

class VisionProcessor(ABC):
    """視覚処理基底クラス"""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """フレーム処理"""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]):
        """設定"""
        pass

class ContourBasedProcessor(VisionProcessor):
    """輪郭ベース検出プロセッサー"""
    
    def __init__(self):
        self.min_area = 100
        self.max_area = 10000
        self.min_contour_length = 50
        self.approx_epsilon = 0.02
        
    def configure(self, config: Dict[str, Any]):
        self.min_area = config.get("min_area", self.min_area)
        self.max_area = config.get("max_area", self.max_area)
        self.min_contour_length = config.get("min_contour_length", self.min_contour_length)
        self.approx_epsilon = config.get("approx_epsilon", self.approx_epsilon)
    
    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        detections = []
        
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンフィルタでノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 適応的閾値処理
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # サイズフィルタ
            if self.min_area <= area <= self.max_area:
                # 輪郭近似
                epsilon = self.approx_epsilon * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # バウンディングボックス
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # モーメント計算（向き検出用）
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # 主軸方向計算
                    mu20 = moments["mu20"] / moments["m00"]
                    mu02 = moments["mu02"] / moments["m00"]
                    mu11 = moments["mu11"] / moments["m00"]
                    
                    # 向き角度計算
                    if mu20 != mu02:
                        orientation = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
                    else:
                        orientation = 0.0
                    
                    # 信頼度計算（輪郭の滑らかさに基づく）
                    perimeter = cv2.arcLength(contour, True)
                    confidence = min(1.0, area / (perimeter * perimeter / (4 * math.pi)))
                    
                    detection = DetectionResult(
                        object_id=f"contour_{i:03d}",
                        position=Position(0, 0, 0, 0, 0, 0),  # 後でキャリブレーション適用
                        pixel_position=(center_x, center_y),
                        confidence=confidence,
                        bounding_box=(x, y, w, h),
                        orientation=orientation,
                        timestamp=time.time(),
                        object_type="contour_object",
                        additional_data={"area": area, "vertices": len(approx)}
                    )
                    
                    detections.append(detection)
        
        return detections

class TemplateMatchingProcessor(VisionProcessor):
    """テンプレートマッチングプロセッサー"""
    
    def __init__(self):
        self.templates: Dict[str, np.ndarray] = {}
        self.match_threshold = 0.7
        self.matching_method = cv2.TM_CCOEFF_NORMED
        
    def configure(self, config: Dict[str, Any]):
        self.match_threshold = config.get("threshold", self.match_threshold)
        
        # テンプレート読み込み
        templates_config = config.get("templates", {})
        for name, template_path in templates_config.items():
            try:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates[name] = template
                    logger.info(f"Loaded template: {name}")
            except Exception as e:
                logger.error(f"Failed to load template {name}: {e}")
    
    def add_template(self, name: str, template_image: np.ndarray):
        """テンプレート追加"""
        self.templates[name] = template_image.copy()
        logger.info(f"Added template: {name}")
    
    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        detections = []
        
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for template_name, template in self.templates.items():
            # テンプレートマッチング実行
            result = cv2.matchTemplate(gray, template, self.matching_method)
            
            # 閾値以上のマッチを検出
            locations = np.where(result >= self.match_threshold)
            
            template_h, template_w = template.shape
            
            for pt in zip(*locations[::-1]):  # Switch columns and rows
                x, y = pt
                center_x = x + template_w // 2
                center_y = y + template_h // 2
                
                # マッチスコア取得
                confidence = result[y + template_h//2, x + template_w//2]
                
                detection = DetectionResult(
                    object_id=f"{template_name}_{x}_{y}",
                    position=Position(0, 0, 0, 0, 0, 0),  # 後でキャリブレーション適用
                    pixel_position=(center_x, center_y),
                    confidence=float(confidence),
                    bounding_box=(x, y, template_w, template_h),
                    orientation=0.0,  # テンプレートマッチングでは回転未考慮
                    timestamp=time.time(),
                    object_type=template_name,
                    additional_data={"match_score": float(confidence)}
                )
                
                detections.append(detection)
        
        return detections

class ColorBasedProcessor(VisionProcessor):
    """色ベース検出プロセッサー"""
    
    def __init__(self):
        self.color_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.min_area = 500
        
    def configure(self, config: Dict[str, Any]):
        self.min_area = config.get("min_area", self.min_area)
        
        # 色範囲設定
        color_config = config.get("color_ranges", {})
        for color_name, range_data in color_config.items():
            lower = np.array(range_data["lower"])
            upper = np.array(range_data["upper"])
            self.color_ranges[color_name] = (lower, upper)
    
    def add_color_range(self, color_name: str, lower_hsv: List[int], upper_hsv: List[int]):
        """色範囲追加"""
        self.color_ranges[color_name] = (np.array(lower_hsv), np.array(upper_hsv))
    
    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        detections = []
        
        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # 色範囲でマスク作成
            mask = cv2.inRange(hsv, lower, upper)
            
            # ノイズ除去
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if area > self.min_area:
                    # バウンディングボックス
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # フィット楕円で向き検出
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        orientation = math.radians(ellipse[2])  # 角度をラジアンに変換
                    else:
                        orientation = 0.0
                    
                    # 信頼度：面積とコンパクトさから計算
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        compactness = 4 * math.pi * area / (perimeter * perimeter)
                        confidence = min(1.0, compactness)
                    else:
                        confidence = 0.5
                    
                    detection = DetectionResult(
                        object_id=f"{color_name}_{i:03d}",
                        position=Position(0, 0, 0, 0, 0, 0),  # 後でキャリブレーション適用
                        pixel_position=(center_x, center_y),
                        confidence=confidence,
                        bounding_box=(x, y, w, h),
                        orientation=orientation,
                        timestamp=time.time(),
                        object_type=color_name,
                        additional_data={"area": area, "compactness": compactness}
                    )
                    
                    detections.append(detection)
        
        return detections

class VisionSystem:
    """視覚システムメインクラス"""
    
    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config
        self.calibration_data: Optional[CalibrationData] = None
        self.processors: Dict[str, VisionProcessor] = {}
        
        # カメラ初期化
        self.camera: Optional[cv2.VideoCapture] = None
        self.frame_buffer: Optional[np.ndarray] = None
        
        # 制御フラグ
        self.state = VisionState.IDLE
        self.capturing = False
        self.processing_enabled = True
        
        # スレッド制御
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        
        # 結果管理
        self.latest_detections: List[DetectionResult] = []
        self.detection_history: List[List[DetectionResult]] = []
        self.max_history_size = 100
        
        # コールバック
        self.on_detection: Optional[Callable[[List[DetectionResult]], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # パフォーマンス監視
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0
    
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            # カメラ初期化
            self.camera = cv2.VideoCapture(self.camera_config.camera_id)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_config.camera_id}")
            
            # カメラ設定
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
            
            if self.camera_config.exposure is not None:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.camera_config.exposure)
            if self.camera_config.brightness is not None:
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_config.brightness)
            if self.camera_config.contrast is not None:
                self.camera.set(cv2.CAP_PROP_CONTRAST, self.camera_config.contrast)
            
            logger.info(f"Camera {self.camera_config.camera_id} initialized successfully")
            
            # デフォルトプロセッサー追加
            self.add_processor("contour", ContourBasedProcessor())
            self.add_processor("color", ColorBasedProcessor())
            self.add_processor("template", TemplateMatchingProcessor())
            
            self.state = VisionState.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Vision system initialization failed: {e}")
            self.state = VisionState.ERROR
            if self.on_error:
                self.on_error(f"Initialization failed: {e}")
            return False
    
    def start_capture(self) -> bool:
        """キャプチャ開始"""
        if self.state == VisionState.ERROR or not self.camera:
            return False
        
        if not self.capturing:
            self.capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            if self.processing_enabled:
                self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
                self.process_thread.start()
            
            logger.info("Vision capture started")
            return True
        
        return False
    
    def stop_capture(self):
        """キャプチャ停止"""
        self.capturing = False
        self.processing_enabled = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        logger.info("Vision capture stopped")
    
    def add_processor(self, name: str, processor: VisionProcessor):
        """プロセッサー追加"""
        self.processors[name] = processor
        logger.info(f"Added vision processor: {name}")
    
    def configure_processor(self, name: str, config: Dict[str, Any]) -> bool:
        """プロセッサー設定"""
        if name in self.processors:
            try:
                self.processors[name].configure(config)
                logger.info(f"Configured processor: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to configure processor {name}: {e}")
                return False
        return False
    
    def calibrate_camera(self, calibration_points: List[Tuple[Tuple[int, int], Position]]) -> bool:
        """カメラキャリブレーション"""
        try:
            self.state = VisionState.CALIBRATING
            
            # キャリブレーションポイントから変換行列を計算
            image_points = np.array([point[0] for point in calibration_points], dtype=np.float32)
            world_points = np.array([[point[1].x, point[1].y, point[1].z] for point in calibration_points], dtype=np.float32)
            
            # 簡易な平面キャリブレーション（実際の実装ではより高度な手法を使用）
            if len(calibration_points) >= 4:
                # ホモグラフィ計算
                world_points_2d = world_points[:, :2]  # Z座標を無視（平面として扱う）
                
                homography, _ = cv2.findHomography(image_points, world_points_2d, cv2.RANSAC)
                
                # キャリブレーションデータ作成
                self.calibration_data = CalibrationData(
                    camera_matrix=np.eye(3),  # 簡略化
                    distortion_coeffs=np.zeros(4),
                    rotation_matrix=np.eye(3),
                    translation_vector=np.zeros(3),
                    pixel_to_mm_ratio=1.0,  # ホモグラフィから計算
                    origin_offset=(0.0, 0.0, world_points[0, 2])  # 平面のZ座標
                )
                
                # ホモグラフィを保存
                self.calibration_data.additional_data = {"homography": homography}
                
                logger.info("Camera calibration completed")
                self.state = VisionState.IDLE
                return True
            else:
                raise ValueError("At least 4 calibration points required")
                
        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            self.state = VisionState.ERROR
            if self.on_error:
                self.on_error(f"Calibration failed: {e}")
            return False
    
    def start_detection(self, processor_names: List[str] = None) -> bool:
        """検出開始"""
        if self.state != VisionState.IDLE:
            return False
        
        if processor_names is None:
            processor_names = list(self.processors.keys())
        
        # 指定されたプロセッサーが存在するかチェック
        for name in processor_names:
            if name not in self.processors:
                logger.error(f"Processor {name} not found")
                return False
        
        self.active_processors = processor_names
        self.state = VisionState.DETECTING
        
        logger.info(f"Detection started with processors: {processor_names}")
        return True
    
    def stop_detection(self):
        """検出停止"""
        self.state = VisionState.IDLE
        self.latest_detections.clear()
        logger.info("Detection stopped")
    
    def get_latest_detections(self) -> List[DetectionResult]:
        """最新検出結果取得"""
        return self.latest_detections.copy()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """現在フレーム取得"""
        with self.frame_lock:
            return self.frame_buffer.copy() if self.frame_buffer is not None else None
    
    def pixel_to_world_position(self, pixel_pos: Tuple[int, int], z_height: float = 0.0) -> Optional[Position]:
        """画素座標を実世界座標に変換"""
        if not self.calibration_data:
            logger.warning("Camera not calibrated")
            return None
        
        try:
            homography = self.calibration_data.additional_data.get("homography")
            if homography is None:
                return None
            
            # ホモグラフィ変換
            pixel_point = np.array([[pixel_pos]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(pixel_point, homography)
            
            world_x = float(world_point[0, 0, 0])
            world_y = float(world_point[0, 0, 1])
            world_z = z_height + self.calibration_data.origin_offset[2]
            
            return Position(world_x, world_y, world_z, 0, 0, 0)
            
        except Exception as e:
            logger.error(f"Pixel to world conversion failed: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            "state": self.state.value,
            "capturing": self.capturing,
            "fps": self.current_fps,
            "processors": list(self.processors.keys()),
            "latest_detections": len(self.latest_detections),
            "calibrated": self.calibration_data is not None,
            "camera_connected": self.camera is not None and self.camera.isOpened()
        }
    
    def shutdown(self):
        """システム終了"""
        self.stop_capture()
        self.stop_detection()
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        cv2.destroyAllWindows()
        logger.info("Vision system shutdown")
    
    def _capture_loop(self):
        """キャプチャループ"""
        logger.info("Capture loop started")
        
        while self.capturing and self.camera:
            try:
                ret, frame = self.camera.read()
                
                if ret:
                    with self.frame_lock:
                        self.frame_buffer = frame.copy()
                    
                    # FPS計算
                    self.fps_counter += 1
                    if time.time() - self.fps_timer >= 1.0:
                        self.current_fps = self.fps_counter
                        self.fps_counter = 0
                        self.fps_timer = time.time()
                
                else:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                if self.on_error:
                    self.on_error(f"Capture error: {e}")
                break
        
        logger.info("Capture loop ended")
    
    def _process_loop(self):
        """処理ループ"""
        logger.info("Process loop started")
        
        while self.processing_enabled:
            try:
                if self.state == VisionState.DETECTING:
                    current_frame = self.get_current_frame()
                    
                    if current_frame is not None:
                        # 全プロセッサーで検出実行
                        all_detections = []
                        
                        for processor_name in getattr(self, 'active_processors', []):
                            if processor_name in self.processors:
                                try:
                                    detections = self.processors[processor_name].process_frame(current_frame)
                                    
                                    # 座標変換
                                    for detection in detections:
                                        world_pos = self.pixel_to_world_position(detection.pixel_position)
                                        if world_pos:
                                            detection.position = world_pos
                                    
                                    all_detections.extend(detections)
                                    
                                except Exception as e:
                                    logger.error(f"Processor {processor_name} error: {e}")
                        
                        # 結果更新
                        self.latest_detections = all_detections
                        
                        # 履歴更新
                        self.detection_history.append(all_detections.copy())
                        if len(self.detection_history) > self.max_history_size:
                            self.detection_history.pop(0)
                        
                        # コールバック実行
                        if self.on_detection and all_detections:
                            self.on_detection(all_detections)
                
                time.sleep(0.05)  # 20Hz処理
                
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                if self.on_error:
                    self.on_error(f"Process error: {e}")
                break
        
        logger.info("Process loop ended")

# グローバルインスタンス
_vision_systems: Dict[str, VisionSystem] = {}

def create_vision_system(name: str, camera_config: CameraConfig) -> VisionSystem:
    """視覚システム作成"""
    system = VisionSystem(camera_config)
    _vision_systems[name] = system
    return system

def get_vision_system(name: str) -> Optional[VisionSystem]:
    """視覚システム取得"""
    return _vision_systems.get(name)

def shutdown_all_vision_systems():
    """全視覚システム終了"""
    for system in _vision_systems.values():
        system.shutdown()
    _vision_systems.clear()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    camera_config = CameraConfig(
        camera_id=0,
        resolution=(640, 480),
        fps=30,
        brightness=128,
        contrast=128
    )
    
    # 視覚システム作成・初期化
    vision_system = create_vision_system("test_camera", camera_config)
    
    if vision_system.initialize():
        logger.info("Vision system initialized successfully")
        
        # テスト用コールバック設定
        def on_detection_callback(detections: List[DetectionResult]):
            logger.info(f"Detected {len(detections)} objects")
            for detection in detections:
                logger.info(f"  {detection.object_id}: {detection.position.to_list()} (conf: {detection.confidence:.2f})")
        
        vision_system.on_detection = on_detection_callback
        
        # カラー検出設定
        color_config = {
            "color_ranges": {
                "red": {
                    "lower": [0, 50, 50],
                    "upper": [10, 255, 255]
                },
                "blue": {
                    "lower": [100, 50, 50],
                    "upper": [130, 255, 255]
                }
            },
            "min_area": 500
        }
        
        vision_system.configure_processor("color", color_config)
        
        # テストキャリブレーション
        calibration_points = [
            ((100, 100), Position(0, 0, 0, 0, 0, 0)),
            ((500, 100), Position(200, 0, 0, 0, 0, 0)),
            ((500, 400), Position(200, 150, 0, 0, 0, 0)),
            ((100, 400), Position(0, 150, 0, 0, 0, 0))
        ]
        
        if vision_system.calibrate_camera(calibration_points):
            logger.info("Camera calibration completed")
        
        try:
            # キャプチャ開始
            vision_system.start_capture()
            
            # 検出開始
            vision_system.start_detection(["color", "contour"])
            
            # テスト実行
            time.sleep(5.0)
            
            # ステータス確認
            status = vision_system.get_system_status()
            logger.info(f"System status: {status}")
            
        finally:
            vision_system.shutdown()
            shutdown_all_vision_systems()
    
    else:
        logger.error("Failed to initialize vision system")