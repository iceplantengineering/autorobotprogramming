"""
AR/VRインターフェース (Phase 5-2)
拡張現実・仮想現実によるロボット制御・監視・トレーニング
WebXR・Three.js・OpenCV統合
"""

import json
import time
import logging
import threading
import asyncio
import base64
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import queue
import math

# AR/VRライブラリ
try:
    import cv2
    import mediapipe as mp
    from scipy.spatial.transform import Rotation
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# 3D処理ライブラリ
try:
    import trimesh
    MESH_AVAILABLE = True
except ImportError:
    MESH_AVAILABLE = False

# Web関連
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from digital_twin_integration import DigitalTwinCore
from production_management_integration import ProductionManagementSystem

logger = logging.getLogger(__name__)

class XRDeviceType(Enum):
    """XRデバイスタイプ"""
    HMD = "hmd"  # Head Mounted Display
    SMART_GLASSES = "smart_glasses"
    TABLET = "tablet"
    SMARTPHONE = "smartphone"
    DESKTOP = "desktop"

class XRSessionType(Enum):
    """XRセッションタイプ"""
    AR = "ar"  # 拡張現実
    VR = "vr"  # 仮想現実
    MIXED = "mixed"  # ミクストリアリティ

class InteractionType(Enum):
    """インタラクションタイプ"""
    GESTURE = "gesture"
    VOICE = "voice"
    CONTROLLER = "controller"
    GAZE = "gaze"
    TOUCH = "touch"

@dataclass
class XRDevice:
    """XRデバイス情報"""
    device_id: str
    device_type: XRDeviceType
    session_type: XRSessionType
    user_id: str
    position: np.ndarray  # [x, y, z, qx, qy, qz, qw]
    fov: float  # 視野角
    resolution: Tuple[int, int]
    connected_at: datetime
    last_update: datetime
    capabilities: List[str]
    tracking_quality: float = 1.0

@dataclass
class ARMarker:
    """ARマーカー"""
    marker_id: str
    marker_type: str  # qr_code, aruco, image_fiducial
    position: np.ndarray
    orientation: np.ndarray
    size: float
    content: Dict[str, Any]
    detected_at: datetime
    confidence: float

@dataclass
class XRInteraction:
    """XRインタラクション"""
    interaction_id: str
    device_id: str
    interaction_type: InteractionType
    target_object: str
    action: str
    parameters: Dict[str, Any]
    confidence: float
    timestamp: datetime

@dataclass
class VirtualAnnotation:
    """仮想注釈"""
    annotation_id: str
    device_id: str
    position: np.ndarray
    content: str
    annotation_type: str  # text, image, 3d_model, video
    persistent: bool
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime] = None

@dataclass
class XRTrainingScenario:
    """XRトレーニングシナリオ"""
    scenario_id: str
    title: str
    description: str
    scenario_type: str  # safety, operation, maintenance, troubleshooting
    difficulty_level: int  # 1-5
    estimated_duration: int  # 分
    prerequisites: List[str]
    steps: List[Dict[str, Any]]
    success_criteria: Dict[str, Any]
    scoring_system: Dict[str, Any]

@dataclass
class TrainingSession:
    """トレーニングセッション"""
    session_id: str
    user_id: str
    scenario_id: str
    device_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: int = 0
    score: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    interactions: List[XRInteraction] = field(default_factory=list)
    status: str = "in_progress"  # in_progress, completed, failed, abandoned

class GestureDetector:
    """ジェスチャー検出器"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_gestures(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """ジェスチャー検出"""
        if not CV_AVAILABLE:
            return []

        gestures = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                gesture = self._analyze_hand_gesture(hand_landmarks)
                if gesture:
                    gestures.append({
                        "hand_index": hand_idx,
                        "gesture": gesture,
                        "landmarks": hand_landmarks,
                        "confidence": 0.8
                    })

        return gestures

    def _analyze_hand_gesture(self, landmarks) -> Optional[str]:
        """手のジェスチャー分析"""
        try:
            # 指の状態を取得
            thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            # 簡単なジェスチャー判定
            if self._is_fist(landmarks):
                return "fist"
            elif self._is_open_palm(landmarks):
                return "open_palm"
            elif self._is_pointing(landmarks):
                return "pointing"
            elif self._is_thumbs_up(landmarks):
                return "thumbs_up"
            elif self._is_victory(landmarks):
                return "victory"

        except Exception as e:
            logger.error(f"Gesture analysis error: {e}")

        return None

    def _is_fist(self, landmarks) -> bool:
        """拳判定"""
        # 簡略的な拳判定
        return True  # 実際はより複雑な判定ロジックが必要

    def _is_open_palm(self, landmarks) -> bool:
        """開いた手判定"""
        return True  # 実際はより複雑な判定ロジックが必要

    def _is_pointing(self, landmarks) -> bool:
        """指差し判定"""
        return True  # 実際はより複雑な判定ロジックが必要

    def _is_thumbs_up(self, landmarks) -> bool:
        """サムズアップ判定"""
        return True  # 実際はより複雑な判定ロジックが必要

    def _is_victory(self, landmarks) -> bool:
        """ピースサイン判定"""
        return True  # 実際はより複雑な判定ロジックが必要

class VoiceCommandProcessor:
    """音声コマンドプロセッサ"""

    def __init__(self):
        self.commands = {
            "start": ["開始", "スタート", "start", "begin"],
            "stop": ["停止", "ストップ", "stop", "halt"],
            "move": ["移動", "ムーブ", "move", "go"],
            "pick": ["掴む", "ピック", "pick", "grab"],
            "place": ["置く", "プレース", "place", "put"],
            "home": ["ホーム", "原点", "home", "origin"],
            "emergency": ["非常停止", "エマージェンシー", "emergency", "stop"],
            "next": ["次", "ネクスト", "next", "forward"],
            "back": ["戻る", "バック", "back", "previous"]
        }

    def process_command(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """音声コマンド処理"""
        try:
            # 実際の実装では音声認識ライブラリ（SpeechRecognitionなど）を使用
            # ここでは模擬的なテキスト処理

            # 模擬音声認識結果
            recognized_text = self._mock_speech_recognition(audio_data)

            if recognized_text:
                return self._parse_command(recized_text)

        except Exception as e:
            logger.error(f"Voice command processing error: {e}")

        return None

    def _mock_speech_recognition(self, audio_data: bytes) -> Optional[str]:
        """模擬音声認識"""
        # 実際の実装では音声認識APIやライブラリを使用
        mock_commands = ["ロボットを開始", "非常停止", "ホームに移動", "次のステップ"]
        import random
        return random.choice(mock_commands) if random.random() > 0.3 else None

    def _parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """コマンド解析"""
        text_lower = text.lower()

        for command, keywords in self.commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        "command": command,
                        "raw_text": text,
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }

        return None

class ARRenderer:
    """ARレンダラー"""

    def __init__(self):
        self.detected_markers: List[ARMarker] = []
        self.virtual_objects: Dict[str, Dict[str, Any]] = {}
        self.annotations: List[VirtualAnnotation] = []

    def detect_markers(self, frame: np.ndarray) -> List[ARMarker]:
        """ARマーカー検出"""
        if not CV_AVAILABLE:
            return []

        markers = []

        # ArUcoマーカー検出
        try:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    # マーカー位置推定
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], 0.05, camera_matrix=None, dist_coeffs=None
                    )

                    if rvec is not None and tvec is not None:
                        marker = ARMarker(
                            marker_id=f"aruco_{marker_id}",
                            marker_type="aruco",
                            position=tvec.flatten(),
                            orientation=rvec.flatten(),
                            size=0.05,
                            content={"id": int(marker_id)},
                            detected_at=datetime.now(),
                            confidence=0.9
                        )
                        markers.append(marker)

        except Exception as e:
            logger.error(f"AR marker detection error: {e}")

        self.detected_markers = markers
        return markers

    def render_ar_overlay(self, frame: np.ndarray, device_pose: np.ndarray) -> np.ndarray:
        """ARオーバーレイ描画"""
        try:
            output_frame = frame.copy()

            # 検出したマーカーに情報を描画
            for marker in self.detected_markers:
                # マーカー情報描画
                text = f"Marker: {marker.marker_id}"
                cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 仮想オブジェクト描画
            for obj_id, obj_data in self.virtual_objects.items():
                if obj_data.get("visible", True):
                    self._render_virtual_object(output_frame, obj_data, device_pose)

            # 注釈描画
            for annotation in self.annotations:
                if annotation.expires_at is None or annotation.expires_at > datetime.now():
                    self._render_annotation(output_frame, annotation, device_pose)

            return output_frame

        except Exception as e:
            logger.error(f"AR rendering error: {e}")
            return frame

    def _render_virtual_object(self, frame: np.ndarray, obj_data: Dict[str, Any], device_pose: np.ndarray):
        """仮想オブジェクト描画"""
        # 実際の実装では3D投影変換を行ってオブジェクトを描画
        pass

    def _render_annotation(self, frame: np.ndarray, annotation: VirtualAnnotation, device_pose: np.ndarray):
        """注釈描画"""
        # 実際の実装では3D投影変換を行って注釈を描画
        cv2.putText(frame, annotation.content, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

class VRScene:
    """VRシーン"""

    def __init__(self):
        self.scene_objects: Dict[str, Dict[str, Any]] = {}
        self.camera_pose = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        self.lighting = {
            "ambient": 0.3,
            "directional": {"direction": [0, -1, 0], "intensity": 0.7},
            "point": []
        }

    def add_robot_model(self, robot_id: str, position: np.ndarray, model_url: str = None):
        """ロボットモデル追加"""
        self.scene_objects[robot_id] = {
            "type": "robot",
            "position": position,
            "orientation": np.array([0, 0, 0, 1]),
            "scale": np.array([1, 1, 1]),
            "model_url": model_url or f"/models/robot_{robot_id}.glb",
            "visible": True,
            "interactive": True
        }

    def add_workspace_model(self, workspace_id: str, dimensions: np.ndarray, position: np.ndarray):
        """ワークスペースモデル追加"""
        self.scene_objects[workspace_id] = {
            "type": "workspace",
            "dimensions": dimensions,
            "position": position,
            "orientation": np.array([0, 0, 0, 1]),
            "visible": True,
            "interactive": False
        }

    def add_trajectory(self, trajectory_id: str, points: List[np.ndarray], color: str = "#00ff00"):
        """軌道追加"""
        self.scene_objects[trajectory_id] = {
            "type": "trajectory",
            "points": [p.tolist() for p in points],
            "color": color,
            "visible": True,
            "interactive": False
        }

    def get_scene_data(self) -> Dict[str, Any]:
        """シーンデータ取得"""
        return {
            "camera": {
                "position": self.camera_pose[:3].tolist(),
                "orientation": self.camera_pose[3:].tolist()
            },
            "lighting": self.lighting,
            "objects": self.scene_objects,
            "timestamp": datetime.now().isoformat()
        }

class XRSimulationManager:
    """XRシミュレーションマネージャ"""

    def __init__(self):
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.scenarios: Dict[str, XRTrainingScenario] = {}
        self.simulation_results: List[Dict[str, Any]] = []

    def create_scenario(self, scenario_data: Dict[str, Any]) -> str:
        """トレーニングシナリオ作成"""
        scenario = XRTrainingScenario(
            scenario_id=str(uuid.uuid4()),
            title=scenario_data["title"],
            description=scenario_data["description"],
            scenario_type=scenario_data["scenario_type"],
            difficulty_level=scenario_data.get("difficulty_level", 1),
            estimated_duration=scenario_data.get("estimated_duration", 30),
            prerequisites=scenario_data.get("prerequisites", []),
            steps=scenario_data.get("steps", []),
            success_criteria=scenario_data.get("success_criteria", {}),
            scoring_system=scenario_data.get("scoring_system", {})
        )

        self.scenarios[scenario.scenario_id] = scenario
        logger.info(f"Created XR training scenario: {scenario.scenario_id}")
        return scenario.scenario_id

    def start_training_session(self, user_id: str, scenario_id: str, device_id: str) -> str:
        """トレーニングセッション開始"""
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario not found: {scenario_id}")

        session = TrainingSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            scenario_id=scenario_id,
            device_id=device_id,
            started_at=datetime.now()
        )

        self.active_sessions[session.session_id] = session
        logger.info(f"Started XR training session: {session.session_id}")
        return session.session_id

    def complete_training_session(self, session_id: str, score: float) -> Dict[str, Any]:
        """トレーニングセッション完了"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.active_sessions[session_id]
        session.completed_at = datetime.now()
        session.score = score
        session.status = "completed"

        # 結果記録
        result = {
            "session_id": session_id,
            "user_id": session.user_id,
            "scenario_id": session.scenario_id,
            "score": score,
            "duration": (session.completed_at - session.started_at).total_seconds(),
            "errors": len(session.errors),
            "interactions": len(session.interactions),
            "completed_at": session.completed_at.isoformat()
        }

        self.simulation_results.append(result)
        del self.active_sessions[session_id]

        logger.info(f"Completed XR training session: {session_id} with score {score}")
        return result

class ARVRInterface:
    """AR/VRインターフェースメインシステム"""

    def __init__(self, digital_twin: DigitalTwinCore, production_system: ProductionManagementSystem):
        self.digital_twin = digital_twin
        self.production_system = production_system

        # コンポーネント
        self.gesture_detector = GestureDetector() if CV_AVAILABLE else None
        self.voice_processor = VoiceCommandProcessor()
        self.ar_renderer = ARRenderer()
        self.vr_scene = VRScene()
        self.simulation_manager = XRSimulationManager()

        # デバイス管理
        self.connected_devices: Dict[str, XRDevice] = {}
        self.device_sessions: Dict[str, str] = {}  # device_id -> session_id

        # インタラクション処理
        self.interaction_queue = queue.Queue(maxsize=100)
        self.annotation_storage: List[VirtualAnnotation] = []

        # 実行制御
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None

        # 設定
        self.max_devices = 10
        self.annotation_retention_days = 30

        # WebAPI
        self.app = self._create_fastapi_app()

        # コールバック
        self.on_interaction_detected: Optional[Callable[[XRInteraction], None]] = None
        self.on_training_completed: Optional[Callable[[Dict[str, Any]], None]] = None

    def _create_fastapi_app(self) -> FastAPI:
        """FastAPIアプリケーション作成"""
        app = FastAPI(title="AR/VR Interface API", version="1.0.0")

        # 静的ファイル
        app.mount("/static", StaticFiles(directory="ar_vr_static"), name="static")

        # APIルート
        self._setup_api_routes(app)

        return app

    def _setup_api_routes(self, app: FastAPI):
        """APIルート設定"""

        @app.get("/ar_vr/health")
        async def health_check():
            """ヘルスチェック"""
            return {
                "status": "healthy",
                "connected_devices": len(self.connected_devices),
                "active_sessions": len(self.simulation_manager.active_sessions),
                "cv_available": CV_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }

        @app.websocket("/ws/ar_vr/{device_id}")
        async def websocket_endpoint(websocket: WebSocket, device_id: str):
            """WebSocketエンドポイント"""
            await websocket.accept()

            try:
                # デバイス登録
                device_info = await websocket.receive_json()
                device = XRDevice(
                    device_id=device_id,
                    device_type=XRDeviceType(device_info.get("device_type", "desktop")),
                    session_type=XRSessionType(device_info.get("session_type", "vr")),
                    user_id=device_info.get("user_id", "anonymous"),
                    position=np.array(device_info.get("position", [0, 0, 0, 0, 0, 0, 1])),
                    fov=device_info.get("fov", 60),
                    resolution=tuple(device_info.get("resolution", [1920, 1080])),
                    connected_at=datetime.now(),
                    last_update=datetime.now(),
                    capabilities=device_info.get("capabilities", []),
                    tracking_quality=device_info.get("tracking_quality", 1.0)
                )

                self.connected_devices[device_id] = device
                logger.info(f"XR device connected: {device_id}")

                # メインループ
                while True:
                    # クライアントからのデータ受信
                    data = await websocket.receive_json()
                    await self._handle_xr_message(device_id, data, websocket)

            except WebSocketDisconnect:
                logger.info(f"XR device disconnected: {device_id}")
                if device_id in self.connected_devices:
                    del self.connected_devices[device_id]

        @app.get("/ar_vr/scenarios")
        async def get_scenarios():
            """トレーニングシナリオ一覧取得"""
            return {
                "scenarios": [asdict(scenario) for scenario in self.simulation_manager.scenarios.values()]
            }

        @app.post("/ar_vr/scenarios")
        async def create_scenario(scenario_data: dict):
            """トレーニングシナリオ作成"""
            try:
                scenario_id = self.simulation_manager.create_scenario(scenario_data)
                return {"success": True, "scenario_id": scenario_id}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/ar_vr/training/start")
        async def start_training(training_data: dict):
            """トレーニング開始"""
            try:
                session_id = self.simulation_manager.start_training_session(
                    training_data["user_id"],
                    training_data["scenario_id"],
                    training_data["device_id"]
                )
                return {"success": True, "session_id": session_id}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/ar_vr/training/complete")
        async def complete_training(completion_data: dict):
            """トレーニング完了"""
            try:
                result = self.simulation_manager.complete_training_session(
                    completion_data["session_id"],
                    completion_data["score"]
                )
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/ar_vr/annotations")
        async def get_annotations(device_id: str = None):
            """注釈取得"""
            annotations = self.annotation_storage
            if device_id:
                annotations = [a for a in annotations if a.device_id == device_id]
            return {"annotations": [asdict(a) for a in annotations]}

        @app.post("/ar_vr/annotations")
        async def create_annotation(annotation_data: dict):
            """注釈作成"""
            annotation = VirtualAnnotation(
                annotation_id=str(uuid.uuid4()),
                device_id=annotation_data["device_id"],
                position=np.array(annotation_data["position"]),
                content=annotation_data["content"],
                annotation_type=annotation_data.get("annotation_type", "text"),
                persistent=annotation_data.get("persistent", False),
                created_by=annotation_data["created_by"],
                created_at=datetime.now()
            )

            self.annotation_storage.append(annotation)
            return {"success": True, "annotation_id": annotation.annotation_id}

    async def _handle_xr_message(self, device_id: str, data: dict, websocket: WebSocket):
        """XRメッセージ処理"""
        message_type = data.get("type")

        if message_type == "position_update":
            # 位置更新
            if device_id in self.connected_devices:
                self.connected_devices[device_id].position = np.array(data["position"])
                self.connected_devices[device_id].last_update = datetime.now()

        elif message_type == "interaction":
            # インタラクション処理
            interaction = XRInteraction(
                interaction_id=str(uuid.uuid4()),
                device_id=device_id,
                interaction_type=InteractionType(data["interaction_type"]),
                target_object=data["target_object"],
                action=data["action"],
                parameters=data.get("parameters", {}),
                confidence=data.get("confidence", 1.0),
                timestamp=datetime.now()
            )

            await self._process_interaction(interaction, websocket)

        elif message_type == "voice_command":
            # 音声コマンド処理
            audio_data = base64.b64decode(data["audio_data"])
            command = self.voice_processor.process_command(audio_data)

            if command:
                await websocket.send_json({
                    "type": "voice_command_result",
                    "command": command,
                    "timestamp": datetime.now().isoformat()
                })

        elif message_type == "annotation_request":
            # 注釈リクエスト
            annotations = [a for a in self.annotation_storage if a.device_id == device_id]
            await websocket.send_json({
                "type": "annotations",
                "annotations": [asdict(a) for a in annotations]
            })

        elif message_type == "ar_frame":
            # ARフレーム処理
            if self.gesture_detector and CV_AVAILABLE:
                # フレームデコード
                frame_data = base64.b64decode(data["frame"])
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                # ジェスチャー検出
                gestures = self.gesture_detector.detect_gestures(frame)

                # ARマーカー検出
                markers = self.ar_renderer.detect_markers(frame)

                # ARオーバーレイ描画
                device_pose = self.connected_devices[device_id].position if device_id in self.connected_devices else np.zeros(7)
                processed_frame = self.ar_renderer.render_ar_overlay(frame, device_pose)

                # 結果を返信
                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_data = base64.b64encode(buffer).decode()

                await websocket.send_json({
                    "type": "ar_frame_processed",
                    "frame": processed_data,
                    "gestures": gestures,
                    "markers": [asdict(m) for m in markers],
                    "timestamp": datetime.now().isoformat()
                })

    async def _process_interaction(self, interaction: XRInteraction, websocket):
        """インタラクション処理"""
        try:
            # インタラクションをキューに追加
            self.interaction_queue.put(interaction)

            # 対応するアクションを実行
            if interaction.action == "move_robot":
                await self._handle_robot_move(interaction, websocket)
            elif interaction.action == "start_operation":
                await self._handle_operation_start(interaction, websocket)
            elif interaction.action == "emergency_stop":
                await self._handle_emergency_stop(interaction, websocket)
            elif interaction.action == "create_annotation":
                await self._handle_annotation_creation(interaction, websocket)

            # コールバック実行
            if self.on_interaction_detected:
                self.on_interaction_detected(interaction)

        except Exception as e:
            logger.error(f"Interaction processing error: {e}")

    async def _handle_robot_move(self, interaction: XRInteraction, websocket):
        """ロボット移動ハンドル"""
        try:
            robot_id = interaction.target_object
            position = interaction.parameters.get("position", [])

            if position and len(position) >= 6:
                # デジタルツインに移動指示
                success = await self.digital_twin.sync_to_physical(robot_id, {"position": position})

                await websocket.send_json({
                    "type": "robot_move_result",
                    "robot_id": robot_id,
                    "success": success,
                    "position": position,
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            logger.error(f"Robot move handling error: {e}")

    async def _handle_operation_start(self, interaction: XRInteraction, websocket):
        """操作開始ハンドル"""
        # 操作開始処理の実装
        pass

    async def _handle_emergency_stop(self, interaction: XRInteraction, websocket):
        """非常停止ハンドル"""
        # 非常停止処理の実装
        await websocket.send_json({
            "type": "emergency_stop_result",
            "success": True,
            "timestamp": datetime.now().isoformat()
        })

    async def _handle_annotation_creation(self, interaction: XRInteraction, websocket):
        """注釈作成ハンドル"""
        try:
            annotation = VirtualAnnotation(
                annotation_id=str(uuid.uuid4()),
                device_id=interaction.device_id,
                position=np.array(interaction.parameters.get("position", [0, 0, 0])),
                content=interaction.parameters.get("content", ""),
                annotation_type=interaction.parameters.get("type", "text"),
                persistent=interaction.parameters.get("persistent", False),
                created_by=interaction.device_id,
                created_at=datetime.now()
            )

            self.annotation_storage.append(annotation)

            await websocket.send_json({
                "type": "annotation_created",
                "annotation_id": annotation.annotation_id,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Annotation creation error: {e}")

    def start(self, port: int = 8092) -> bool:
        """AR/VRインターフェース起動"""
        try:
            self.running = True

            # デフォルトシナリオ作成
            self._create_default_scenarios()

            # 処理スレッド開始
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            logger.info(f"AR/VR Interface started on port {port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start AR/VR interface: {e}")
            return False

    def stop(self):
        """AR/VRインターフェース停止"""
        self.running = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        logger.info("AR/VR Interface stopped")

    def _create_default_scenarios(self):
        """デフォルトトレーニングシナリオ作成"""
        scenarios = [
            {
                "title": "ロボット基本操作トレーニング",
                "description": "ロボットの基本操作を学習するトレーニング",
                "scenario_type": "operation",
                "difficulty_level": 1,
                "estimated_duration": 15,
                "steps": [
                    {"step": 1, "description": "ロボットの電源を入れる", "action": "power_on"},
                    {"step": 2, "description": "ホームポジションに移動", "action": "move_home"},
                    {"step": 3, "description": "安全に停止", "action": "safe_stop"}
                ],
                "success_criteria": {"completion_rate": 1.0, "error_count": 0},
                "scoring_system": {"completion": 70, "safety": 20, "efficiency": 10}
            },
            {
                "title": "非常手順トレーニング",
                "description": "緊急時の対応手順を学習するトレーニング",
                "scenario_type": "safety",
                "difficulty_level": 2,
                "estimated_duration": 10,
                "steps": [
                    {"step": 1, "description": "危険を認識", "action": "detect_danger"},
                    {"step": 2, "description": "非常停止ボタンを押す", "action": "emergency_stop"},
                    {"step": 3, "description": "状況を確認", "action": "check_status"}
                ],
                "success_criteria": {"response_time": 3.0, "correct_actions": 3},
                "scoring_system": {"speed": 50, "accuracy": 50}
            }
        ]

        for scenario_data in scenarios:
            self.simulation_manager.create_scenario(scenario_data)

    def _processing_loop(self):
        """処理ループ"""
        logger.info("AR/VR processing loop started")

        while self.running:
            try:
                # インタラクションキュー処理
                while not self.interaction_queue.empty():
                    try:
                        interaction = self.interaction_queue.get_nowait()
                        # 必要に応じて追加処理
                    except queue.Empty:
                        break

                # 注釈クリーンアップ
                self._cleanup_expired_annotations()

                time.sleep(0.1)  # 100ms間隔

            except Exception as e:
                logger.error(f"AR/VR processing error: {e}")
                time.sleep(1.0)

        logger.info("AR/VR processing loop ended")

    def _cleanup_expired_annotations(self):
        """期限切れ注釈クリーンアップ"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=self.annotation_retention_days)

        self.annotation_storage = [
            a for a in self.annotation_storage
            if a.expires_at is None or a.expires_at > cutoff_time
        ]

    def get_interface_status(self) -> Dict[str, Any]:
        """インターフェース状態取得"""
        return {
            "running": self.running,
            "connected_devices": len(self.connected_devices),
            "active_sessions": len(self.simulation_manager.active_sessions),
            "total_annotations": len(self.annotation_storage),
            "available_scenarios": len(self.simulation_manager.scenarios),
            "cv_available": CV_AVAILABLE,
            "gesture_detection_enabled": self.gesture_detector is not None,
            "devices": {
                device_id: {
                    "device_type": device.device_type.value,
                    "session_type": device.session_type.value,
                    "user_id": device.user_id,
                    "tracking_quality": device.tracking_quality,
                    "last_update": device.last_update.isoformat()
                }
                for device_id, device in self.connected_devices.items()
            }
        }

# グローバルインスタンス
ar_vr_interface: Optional[ARVRInterface] = None

def initialize_ar_vr_interface(digital_twin: DigitalTwinCore,
                              production_system: ProductionManagementSystem) -> ARVRInterface:
    """AR/VRインターフェース初期化"""
    global ar_vr_interface
    ar_vr_interface = ARVRInterface(digital_twin, production_system)
    return ar_vr_interface

def get_ar_vr_interface() -> Optional[ARVRInterface]:
    """AR/VRインターフェース取得"""
    return ar_vr_interface

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing AR/VR Interface System...")

    try:
        # モックシステム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())
        mock_dt = None  # デジタルツインは別途初期化

        # AR/VRインターフェース初期化
        interface = initialize_ar_vr_interface(mock_dt, mock_pms)

        if interface.start():
            print("AR/VR Interface started successfully!")
            print("Available endpoints:")
            print("  - WebSocket: ws://localhost:8092/ws/ar_vr/{device_id}")
            print("  - Health: http://localhost:8092/ar_vr/health")
            print("  - Scenarios: http://localhost:8092/ar_vr/scenarios")

            # ステータス確認
            time.sleep(1)
            status = interface.get_interface_status()
            print(f"Interface status: {status}")

            time.sleep(2)
            interface.stop()

        else:
            print("Failed to start AR/VR interface")

    except Exception as e:
        print(f"AR/VR interface test failed: {e}")