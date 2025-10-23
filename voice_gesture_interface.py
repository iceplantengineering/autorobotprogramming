"""
音声制御・ジェスチャー制御インターフェース (Phase 5-6)
自然言語処理・コンピュータビジョン・マルチモーダルインタラクション
リアルタイム認識・コマンド実行・学習機能
"""

import json
import time
import logging
import threading
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import queue
import os
import tempfile
from collections import defaultdict, deque

# 音声処理ライブラリ
try:
    import speech_recognition as sr
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# コンピュータビジョンライブラリ
try:
    import cv2
    import mediapipe as mp
    from scipy.spatial.transform import Rotation
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# 自然言語処理ライブラリ
try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# 機械学習ライブラリ
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from production_management_integration import ProductionManagementSystem
from multi_robot_coordination import RobotInfo, RobotState
from vc_robot_controller import VCRobotController

logger = logging.getLogger(__name__)

class InteractionMode(Enum):
    """インタラクションモード"""
    VOICE_ONLY = "voice_only"
    GESTURE_ONLY = "gesture_only"
    MULTIMODAL = "multimodal"  # 音声+ジェスチャー
    ADAPTIVE = "adaptive"  # 適応的

class CommandType(Enum):
    """コマンドタイプ"""
    MOVEMENT = "movement"
    GRIPPER = "gripper"
    TOOL = "tool"
    PROGRAM = "program"
    SYSTEM = "system"
    QUERY = "query"
    EMERGENCY = "emergency"

class VoiceCommand:
    """音声コマンド"""
    def __init__(self, command_text: str, confidence: float, timestamp: datetime):
        self.command_text = command_text
        self.confidence = confidence
        self.timestamp = timestamp
        self.intent = None
        self.entities = {}
        self.parameters = {}

class Gesture:
    """ジェスチャー"""
    def __init__(self, gesture_type: str, confidence: float, landmarks: Any, timestamp: datetime):
        self.gesture_type = gesture_type
        self.confidence = confidence
        self.landmarks = landmarks
        self.timestamp = timestamp
        self.parameters = {}

class MultimodalCommand:
    """マルチモーダルコマンド"""
    def __init__(self, voice: VoiceCommand = None, gesture: Gesture = None):
        self.voice = voice
        self.gesture = gesture
        self.timestamp = datetime.now()
        self.fused_modalities = []
        self.confidence = 0.0
        self.intent = None
        self.parameters = {}

        if voice:
            self.used_modalities.append("voice")
            self.confidence += voice.confidence * 0.6

        if gesture:
            self.used_modalities.append("gesture")
            self.confidence += gesture.confidence * 0.4

        # 両方がある場合は重みを調整
        if voice and gesture:
            self.confidence = min(1.0, self.confidence)

class VoiceRecognitionEngine:
    """音声認識エンジン"""

    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.is_listening = False

        # 言定
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        # 言語モデル
        self.command_patterns = self._initialize_command_patterns()
        self.intent_classifier = None

        # 統計
        self.recognition_stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "confidence_threshold": 0.5
        }

        self._initialize_recognizer()

    def _initialize_recognizer(self):
        """認識エンジン初期化"""
        try:
            if not AUDIO_AVAILABLE:
                logger.error("Audio libraries not available")
                return

            self.recognizer = sr.Recognizer()
            logger.info("Voice recognition engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize voice recognition: {e}")

    def _initialize_command_patterns(self) -> Dict[str, Any]:
        """コマンドパターン初期化"""
        return {
            # 移動コマンド
            "move": {
                "keywords": ["移動", "動かして", "ムーブ", "move", "go to", "移動して"],
                "patterns": [
                    r"移動して(.*?)に",
                    r"(.*?)に移動",
                    r"go to (.*)",
                    r"move to (.*)"
                ],
                "parameters": ["destination", "speed", "path"]
            },
            # 把持コマンド
            "grip": {
                "keywords": ["掴んで", "握って", "グリップ", "grip", "pick up", "掴む"],
                "patterns": [
                    r"掴んで",
                    r"握って",
                    r"grip",
                    r"pick up"
                ],
                "parameters": ["object", "force", "position"]
            },
            # 放置コマンド
            "release": {
                "keywords": ["置いて", "離して", "リリース", "release", "put down", "放す"],
                "patterns": [
                    r"置いて",
                    r"離して",
                    r"release",
                    r"put down"
                ],
                "parameters": ["position", "force"]
            },
            # ツ�ールコマンド
            "tool": {
                "keywords": ["ツール", "交換", "tool", "change", "スワップ"],
                "patterns": [
                    r"ツールを(.*?)に交換",
                    r"(.*?)ツール",
                    r"change tool to (.*)"
                ],
                "parameters": ["tool_type"]
            },
            # 非常停止コマンド
            "emergency": {
                "keywords": ["停止", "止まれ", "非常停止", "stop", "emergency", "ストップ"],
                "patterns": [
                    r"停止",
                    r"止まれ",
                    r"emergency stop",
                    r"stop"
                ],
                "parameters": []
            },
            # クエリコマンド
            "query": {
                "keywords": ["状態", "状況", "クエリ", "status", "どう", "何"],
                "patterns": [
                    r"状態は",
                    r"状況は",
                    r"どう",
                    r"何を"
                ],
                "parameters": ["query_type"]
            }
        }

    def start_listening(self) -> bool:
        """リスニング開始"""
        try:
            if not AUDIO_AVAILABLE or not self.recognizer:
                logger.error("Voice recognition not available")
                return False

            if self.is_listening:
                return True

            self.microphone = pyaudio.PyAudio()
            self.is_listening = True

            # リスニングスレッド開始
            listening_thread = threading.Thread(target=self._listening_loop, daemon=True)
            listening_thread.start()

            logger.info("Started voice listening")
            return True

        except Exception as e:
            logger.error(f"Failed to start voice listening: {e}")
            return False

    def stop_listening(self):
        """リスニング停止"""
        self.is_listening = False

        if self.microphone:
            self.microphone.terminate()
            self.microphone = None

        logger.info("Stopped voice listening")

    def _listening_loop(self):
        """リスニングループ"""
        logger.info("Voice listening loop started")

        while self.is_listening:
            try:
                # 音声検出
                voice_detected = self._detect_voice_activity()

                if voice_detected:
                    # 音声認識実行
                    command = self._recognize_speech()

                    if command:
                        # コマンド解析
                        self._parse_command(command)
                        self.recognition_stats["total_commands"] += 1
                        if command.confidence >= self.recognition_stats["confidence_threshold"]:
                            self.recognition_stats["successful_commands"] += 1

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Voice listening error: {e}")
                time.sleep(1.0)

        logger.info("Voice listening loop ended")

    def _detect_voice_activity(self) -> bool:
        """音声活動検出"""
        try:
            # 簡易的な音声検出（実際はより高度なVADを使用）
            return True  # 常に検出
        except Exception as e:
            logger.error(f"Voice detection error: {e}")
            return False

    def _recognize_speech(self) -> Optional[VoiceCommand]:
        """音声認識実行"""
        try:
            # マイクから音声録音（模擬）
            with sr.Microphone() as source:
                logger.info("Listening for command...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)

                if audio:
                    logger.info("Recognizing speech...")
                    try:
                        # Google Speech Recognition APIを使用
                        text = self.recognizer.recognize_google(audio, language="ja-JP")
                        confidence = 0.8  # 実際は信頼度を取得

                        return VoiceCommand(text, confidence, datetime.now())

                    except sr.UnknownValueError:
                        logger.warning("Could not understand audio")
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition error: {e}")

        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")

        return None

    def _parse_command(self, command: VoiceCommand):
        """コマンド解析"""
        text = command.command_text.lower()

        for intent_type, pattern_info in self.command_patterns.items():
            # キーワードマッチング
            for keyword in pattern_info["keywords"]:
                if keyword in text:
                    command.intent = intent_type

                    # パターンマッチングでパラメータ抽出
                    for pattern in pattern_info["patterns"]:
                        import re
                        match = re.search(pattern, text)
                        if match:
                            self._extract_parameters(command, match, pattern_info["parameters"])
                            break

                    break

    def _extract_parameters(self, command: VoiceCommand, match, param_types: List[str]):
        """パラメータ抽出"""
        try:
            for i, param_type in enumerate(param_types):
                if i < len(match.groups()):
                    value = match.group(i+1)
                    command.parameters[param_type] = value.strip()

        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")

class GestureRecognitionEngine:
    """ジェスチャー認識エンジン"""

    def __init__(self):
        self.hands_detector = None
        self.pose_detector = None
        self.is_detecting = False

        # ジェスチャーモデル
        self.gesture_classifier = None
        self.gesture_patterns = self._initialize_gesture_patterns()

        # 統計
        self.gesture_stats = {
            "total_gestures": 0,
            "recognized_gestures": 0,
            "confidence_threshold": 0.7
        }

        self._initialize_detectors()

    def _initialize_detectors(self):
        """検出器初期化"""
        try:
            if not CV_AVAILABLE:
                logger.error("Computer vision libraries not available")
                return

            self.hands_detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.pose_detector = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            logger.info("Gesture recognition detectors initialized")

        except Exception as e:
            logger.error(f"Failed to initialize detectors: {e}")

    def _initialize_gesture_patterns(self) -> Dict[str, Any]:
        """ジェスチャーパターン初期化"""
        return {
            # 手のジェスチャー
            "fist": {
                "name": "拳",
                "keywords": ["拳", "握りしめ"],
                "description": "拳を握るジェスチャー",
                "conditions": self._fist_conditions
            },
            "open_palm": {
                "name": "開いた手",
                "keywords": ["開いた手", "パーム"],
                "description": "手を開くジェスチャー",
                "conditions": self._open_palm_conditions
            },
            "pointing": {
                "name": "指差し",
                "keywords": ["指差し", "ポイント"],
                "description": "人差し指を立てるジェスチャー",
                "conditions": self._pointing_conditions
            },
            "thumbs_up": {
                "name": "サムズアップ",
                "keywords": ["サムズアップ", "いいね"],
                "description": "親指を立てるジェスチャー",
                "conditions": self._thumbs_up_conditions
            },
            "ok": {
                "name": "OK",
                "keywords": ["OK", "オーケー"],
                "description": "OKサインジェスチャー",
                "conditions": self._ok_conditions
            },
            # ポ体のジェスチャー
            "wave": {
                "name": "手招き",
                "keywords": ["手招き", "ウェーブ"],
                "description": "手を振るジェスチャー",
                "conditions": self._wave_conditions
            }
        }

    def start_detection(self, camera_index: int = 0) -> bool:
        """検出開始"""
        try:
            if not CV_AVAILABLE or not self.hands_detector:
                logger.error("Gesture detection not available")
                return False

            if self.is_detecting:
                return True

            self.is_detecting = True

            # 検出スレッド開始
            detection_thread = threading.Thread(
                target=self._detection_loop,
                args=(camera_index,),
                daemon=True
            )
            detection_thread.start()

            logger.info(f"Started gesture detection with camera {camera_index}")
            return True

        except Exception as e:
            logger.error(f"Failed to start gesture detection: {e}")
            return False

    def stop_detection(self):
        """検出停止"""
        self.is_detecting = False
        logger.info("Stopped gesture detection")

    def _detection_loop(self, camera_index: int):
        """検出ループ"""
        logger.info("Gesture detection loop started")

        try:
            cap = cv2.VideoCapture(camera_index)

            while self.is_detecting:
                ret, frame = cap.read()
                if not ret:
                    continue

                # ジェスチャー検出
                gestures = self._detect_gestures(frame)

                for gesture in gestures:
                    self.gesture_stats["total_gestures"] += 1
                    if gesture.confidence >= self.gesture_stats["confidence_threshold"]:
                        self.gesture_stats["recognized_gestures"] += 1

                time.sleep(0.1)  # 10fps

            cap.release()

        except Exception as e:
            logger.error(f"Gesture detection error: {e}")
        finally:
            self.is_detecting = False

        logger.info("Gesture detection loop ended")

    def _detect_gestures(self, frame: np.ndarray) -> List[Gesture]:
        """ジェスチャー検出"""
        gestures = []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 手の検出
            hand_results = self.hands_detector.process(rgb_frame)

            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # 各ジェスチャーパターンをチェック
                    for gesture_type, pattern in self.gesture_patterns.items():
                        confidence = pattern["conditions"](hand_landmarks)
                        if confidence > 0.7:
                            gesture = Gesture(
                                gesture_type=gesture_type,
                                confidence=confidence,
                                landmarks=hand_landmarks,
                                timestamp=datetime.now()
                            )

                            # パラメータ抽出
                            self._extract_gesture_parameters(gesture, hand_landmarks)
                            gestures.append(gesture)

        except Exception as e:
            logger.error(f"Gesture detection failed: {e}")

        return gestures

    def _extract_gesture_parameters(self, gesture: Gesture, landmarks):
        """ジェスチャーパラメータ抽出"""
        try:
            # 手の中心位置計算
            x_coords = [landmark.x for landmark in landmarks.landmark]
            y_coords = [landmark.y for landmark in landmarks.landmark]

            hand_center_x = sum(x_coords) / len(x_coords)
            hand_center_y = sum(y_coords) / len(y_coords)

            gesture.parameters = {
                "hand_center": (hand_center_x, hand_center_y),
                "hand_size": self._calculate_hand_size(landmarks),
                "handedness": "right" if hand_center_x > 0.5 else "left"
            }

        except Exception as e:
            logger.error(f"Gesture parameter extraction failed: {e}")

    def _calculate_hand_size(self, landmarks) -> float:
        """手のサイズ計算"""
        try:
            # 親指先と手首の距離でサイズを計算
            wrist = landmarks.landmark[self.hands_detector.HandLandmark.WRIST]
            thumb_tip = landmarks.landmark[self.hands_detector.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[self.hands_detector.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = landmarks.landmark[self.hands_detector.HandLandmark.PINKY_TIP]

            distances = [
                np.sqrt((wrist.x - thumb_tip.x)**2 + (wrist.y - thumb_tip.y)**2),
                np.sqrt((wrist.x - index_tip.x)**2 + (wrist.y - index_tip.y)**2),
                np.sqrt((wrist.x - pinky_tip.x)**2 + (wrist.y - pinky_tip.y)**2)
            ]

            return sum(distances) / len(distances)

        except Exception as e:
            logger.error(f"Hand size calculation failed: {e}")
            return 0.0

    # ジェスチャー条件関数
    def _fist_conditions(self, landmarks) -> float:
        """拳判定条件"""
        try:
            # 指が曲がっているかチェック
            finger_tips = [
                self.hands_detector.HandLandmark.THUMB_TIP,
                self.hands_detector.HandLandmark.INDEX_FINGER_TIP,
                self.hands_detector.HandLandmark.MIDDLE_FINGER_TIP,
                self.hands_detector.HandLandmark.RING_FINGER_TIP,
                self.hands_detector.HandLandmark.PINKY_TIP
            ]

            finger_mcp = [
                self.hands_detector.HandLandmark.INDEX_FINGER_MCP,
                self.hands_detector.HandLandmark.MIDDLE_FINGER_MCP,
                self.hands_detector.HandLandmark.RING_FINGER_MCP,
                self.hands_detector.HandLandmark.PINKY_MCP
            ]

            folded_fingers = 0
            for i, tip in enumerate(finger_tips[1:], 1):  # 親指を除く
                mcp = finger_mcp[i-1]
                if tip.y > mcp.y:  # MCPよりも指尖が下なら曲がっている
                    folded_fingers += 1

            return folded_fingers / 4  # 4指のうち何本曲がっているか

        except Exception as e:
            return 0.0

    def _open_palm_conditions(self, landmarks) -> float:
        """開いた手判定条件"""
        try:
            # 指が伸びているかチェック
            finger_tips = [
                self.hands_detector.HandLandmark.THUMB_TIP,
                self.hands_detector.HandLandmark.INDEX_FINGER_TIP,
                self.hands_detector.HandLandmark.MIDDLE_FINGER_TIP,
                self.hands_detector.HandLandmark.RING_FINGER_TIP,
                self.hands_detector.HandLandmark.PINKY_TIP
            ]

            finger_mcp = [
                self.hands_detector.HandLandmark.INDEX_FINGER_MCP,
                self.hands_detector.HandLandmark.MIDDLE_FINGER_MCP,
                self.hands_detector.HandLandmark.RING_FINGER_MCP,
                self.hands_detector.HandLandmark.PINKY_MCP
            ]

            extended_fingers = 0
            for i, tip in enumerate(finger_tips[1:], 1):  # 親指を除く
                mcp = finger_mcp[i-1]
                if tip.y < mcp.y:  # MCPよりも指尖が上なら伸びている
                    extended_fingers += 1

            return extended_fingers / 4  # 4指のうち何本伸びているか

        except Exception as e:
            return 0.0

    def _pointing_conditions(self, landmarks) -> float:
        """指差し判定条件"""
        try:
            # 人差し指が伸びていて、他の指は曲がっているかチェック
            index_tip = landmarks.landmark[self.hands_detector.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = landmarks.landmark[self.hands_detector.HandLandmark.INDEX_FINGER_MCP]
            middle_tip = landmarks.landmark[self.hands_detector.HandLandmark.MIDDLE_FINGER_TIP]
            middle_mcp = landmarks.landmark[self.hands_detector.HandLandmark.MIDDLE_FINGER_MCP]

            # 人差し指が伸びている
            index_extended = index_tip.y < index_mcp.y
            # 中指が曲がっている
            middle_folded = middle_tip.y > middle_mcp.y

            return 1.0 if index_extended and middle_folded else 0.0

        except Exception as e:
            return 0.0

    def _thumbs_up_conditions(self, landmarks) -> float:
        """サムズアップ判定条件"""
        try:
            thumb_tip = landmarks.landmark[self.hands_detector.HandLandmark.THUMB_TIP]
            thumb_mcp = landmarks.landmark[self.hands_detector.HandLandmark.THUMB_MCP]
            index_mcp = landmarks.landmark[self.hands_detector.HandLandmark.INDEX_FINGER_MCP]

            # 親指が上を向いて、他の指が曲がっている
            thumb_up = thumb_tip.y < thumb_mcp.y
            others_folded = self._fist_conditions(landmarks) > 0.5

            return 1.0 if thumb_up and others_folded else 0.0

        except Exception as e:
            return 0.0

    def _ok_conditions(self, landmarks) -> float:
        """OK判定条件"""
        try:
            # 親指と人差し指で輪を作っているかチェック
            thumb_tip = landmarks.landmark[self.hands_detector.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[self.hands_detector.HandLandmark.INDEX_FINGER_TIP]

            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            # 適切な距離ならOKサイン
            if 0.02 < distance < 0.1:
                return 1.0

            return 0.0

        except Exception as e:
            return 0.0

    def _wave_conditions(self, landmarks) -> float:
        """手招き判定条件"""
        try:
            # 手首のY座標より指が上にあるかチェック
            wrist = landmarks.landmark[self.hands_detector.HandLandmark.WRIST]
            fingers = [
                landmarks.landmark[self.hands_detector.HandLandmark.INDEX_FINGER_PIP],
                landmarks.landmark[self.hands_detector.HandLandmark.MIDDLE_FINGER_PIP],
                landmarks.landmark[self.hands_detector.HandLandmark.RING_FINGER_PIP],
                landmarks.landmark[self.hands_detector.HandLandmark.PINKY_PIP]
            ]

            fingers_up = sum(1 for finger in fingers if finger.y < wrist.y)
            return fingers_up / len(fingers)

        except Exception as e:
            return 0.0

class MultimodalCommandProcessor:
    """マルチモーダルコマンドプロセッサ"""

    def __init__(self):
        self.voice_engine = VoiceRecognitionEngine()
        self.gesture_engine = GestureRecognitionEngine()

        # コマンド履歴
        self.command_history: deque = deque(maxlen=1000)
        self.learning_model = None

        # マッピングルール
        self.modality_mappings = self._initialize_modality_mappings()

        # 統計
        self.processing_stats = {
            "total_commands": 0,
            "voice_only": 0,
            "gesture_only": 0,
            "multimodal": 0,
            "successful_executions": 0
        }

        self._initialize_learning_model()

    def _initialize_modality_mappings(self) -> Dict[str, Any]:
        """モダリティマッピング初期化"""
        return {
            # 移動コマンドのマッピング
            "move_to_gesture": {
                "gesture_type": "pointing",
                "voice_keywords": ["移動", "動かして"],
                "parameter_extraction": {
                    "destination": "gesture.hand_center",
                    "speed": "voice.parameters.get('speed', 'normal')"
                }
            },
            "grip_with_gesture": {
                "gesture_type": "fist",
                "voice_keywords": ["掴んで", "握って"],
                "parameter_extraction": {
                    "force": "gesture.hand_size",  # 握りの強さで調整
                    "position": "gesture.hand_center"
                }
            },
            # 非常停止のマッピング
            "emergency_stop_multimodal": {
                "gesture_type": ["open_palm", "fist"],
                "voice_keywords": ["停止", "ストップ"],
                "immediate_execution": True
            }
        }

    def _initialize_learning_model(self):
        """学習モデル初期化"""
        try:
            if not ML_AVAILABLE:
                logger.warning("Machine learning libraries not available")
                return

            # 簡単な分類モデル
            self.learning_model = RandomForestClassifier(
                n_estimators=10,
                random_state=42
            )

            # 特徴スケーラー
            self.feature_scaler = StandardScaler()

        except Exception as e:
            logger.error(f"Failed to initialize learning model: {e}")

    def start_interaction(self, camera_index: int = 0) -> bool:
        """インタラクション開始"""
        try:
            # 音声認識開始
            voice_success = self.voice_engine.start_listening()

            # ジェスチャー認識開始
            gesture_success = self.gesture_engine.start_detection(camera_index)

            success = voice_success or gesture_success
            if success:
                logger.info("Multimodal command processor started")
            else:
                logger.error("Failed to start any recognition engine")

            return success

        except Exception as e:
            logger.error(f"Failed to start multimodal interaction: {e}")
            return False

    def stop_interaction(self):
        """インタラクション停止"""
        self.voice_engine.stop_listening()
        self.gesture_engine.stop_detection()
        logger.info("Multimodal command processor stopped")

    def process_multimodal_command(self, voice: VoiceCommand = None,
                                   gesture: Gesture = None) -> Optional[MultimodalCommand]:
        """マルチモーダルコマンド処理"""
        try:
            # マルチモーダルコマンド作成
            command = MultimodalCommand(voice, gesture)

            # モ向性を統合
            self._combine_intents(command)

            # 履歴に追加
            self.command_history.append(command)
            self.processing_stats["total_commands"] += 1

            # モ向性と信頼度の確認
            if command.confidence >= 0.7 and command.intent:
                # 実行統計更新
                if voice and gesture:
                    self.processing_stats["multimodal"] += 1
                elif voice:
                    self.processing_stats["voice_only"] += 1
                else:
                    self.processing_stats["gesture_only"] += 1

                return command

            return None

        except Exception as e:
            logger.error(f"Multimodal command processing failed: {e}")
            return None

    def _combine_intents(self, command: MultimodalCommand):
        """意図を統合"""
        try:
            intents = []
            confidences = []

            if command.voice and command.voice.intent:
                intents.append(command.voice.intent)
                confidences.append(command.voice.confidence)

            if command.gesture and command.gesture.gesture_type:
                # ジェスチャータイプを意図に変換
                gesture_intent = self._map_gesture_to_intent(command.gesture.gesture_type)
                if gesture_intent:
                    intents.append(gesture_intent)
                    confidences.append(command.gesture.confidence)

            if intents:
                # 最も信頼度の高い意図を選択
                max_index = np.argmax(confidences)
                command.intent = intents[max_index]
                command.confidence = confidences[max_index]

                # パラメータをマージ
                command.parameters = self._merge_parameters(command)

            elif command.voice:
                # 音声のみの場合
                command.intent = command.voice.intent
                command.parameters = command.voice.parameters
                command.confidence = command.voice.confidence

            elif command.gesture:
                # ジェスチャーのみの場合
                command.intent = self._map_gesture_to_intent(command.gesture.gesture_type)
                command.parameters = command.gesture.parameters
                command.confidence = command.gesture.confidence

        except Exception as e:
            logger.error(f"Intent combination failed: {e}")

    def _map_gesture_to_intent(self, gesture_type: str) -> Optional[str]:
        """ジェスチャータイプを意図にマッピング"""
        gesture_intent_map = {
            "fist": "grip",
            "open_palm": "release",
            "pointing": "move",
            "thumbs_up": "accept",
            "ok": "confirm",
            "wave": "greet"
        }

        return gesture_intent_map.get(gesture_type)

    def _merge_parameters(self, command: MultimodalCommand):
        """パラメータをマージ"""
        try:
            merged_params = {}

            if command.voice:
                merged_params.update(command.voice.parameters)

            if command.gesture:
                merged_params.update(command.gesture.parameters)

            # モ向性に応じたパラメータ調整
            if command.intent in self.modality_mappings:
                mapping = self.modality_mappings[command.intent]
                if "parameter_extraction" in mapping:
                    for param_name, param_source in mapping["parameter_extraction"].items():
                        if param_source in merged_params:
                            merged_params[param_name] = merged_params[param_source]

            command.parameters = merged_params

        except Exception as e:
            logger.error(f"Parameter merging failed: {e}")

    def learn_from_history(self) -> bool:
        """履歴から学習"""
        try:
            if not self.learning_model or len(self.command_history) < 10:
                return False

            # 特徴抽出と学習
            features, labels = self._extract_features_from_history()

            if len(features) < 5:
                return False

            # 特徴スケーリング
            X_scaled = self.feature_scaler.fit_transform(features)

            # モ向性のエンコーディング
            intent_map = {
                "move": 0, "grip": 1, "release": 2, "stop": 3, "tool": 4,
                "query": 5, "emergency": 6, "accept": 7, "confirm": 8, "greet": 9
            }

            y_encoded = [intent_map.get(intent, 0) for intent in labels]

            # モデル学習
            self.learning_model.fit(X_scaled, y_encoded)

            logger.info(f"Learned from {len(features)} command examples")
            return True

        except Exception as e:
            logger.error(f"Learning from history failed: {e}")
            return False

    def _extract_features_from_history(self) -> Tuple[List[List[float]], List[str]]:
        """履歴から特徴抽出"""
        features = []
        labels = []

        for command in self.command_history:
            feature_vector = []

            # 音声特徴
            if command.voice:
                # 音声テキストの特徴（テキスト長、単語数など）
                text = command.voice.command_text
                feature_vector.extend([
                    len(text),
                    len(text.split()),
                    command.voice.confidence
                ])

                # 音声の意図をone-hotエンコーディング
                intent_features = [0] * 10
                if command.voice.intent:
                    intent_map = {
                        "move": 0, "grip": 1, "release": 2, "stop": 3, "tool": 4,
                        "query": 5, "emergency": 6, "accept": 7, "confirm": 8, "greet": 9
                    }
                    intent_features[intent_map.get(command.voice.intent, 0))] = 1
                feature_vector.extend(intent_features)

            else:
                feature_vector.extend([0] * 13)  # 音声なしの場合

            # ジェスチャー特徴
            if command.gesture:
                feature_vector.extend([
                    command.gesture.confidence,
                    len(command.gesture.parameters)
                ])

                # ジェスチャータイプのone-hotエンコーディング
                gesture_features = [0] * 6
                gesture_map = {
                    "fist": 0, "open_palm": 1, "pointing": 2,
                    "thumbs_up": 3, "ok": 4, "wave": 5
                }
                gesture_features[gesture_map.get(command.gesture.gesture_type, 0)] = 1
                feature_vector.extend(gesture_features)

            else:
                feature_vector.extend([0.0] * 7])  # ジェスチャーなしの場合

            features.append(feature_vector)
            labels.append(command.intent or "unknown")

        return features, labels

    def get_interaction_statistics(self) -> Dict[str, Any]:
        """インタラクション統計取得"""
        return {
            "processing_stats": self.processing_stats.copy(),
            "voice_stats": self.voice_engine.recognition_stats,
            "gesture_stats": self.gesture_engine.gesture_stats,
            "command_history_size": len(self.command_history),
            "model_trained": self.learning_model is not None
        }

class VoiceGestureController:
    """音声・ジェスチャーコントローラ"""

    def __init__(self, production_system: ProductionManagementSystem):
        self.production_system = production_system
        self.command_processor = MultimodalCommandProcessor()

        # コマンド実行マッピング
        self.command_handlers = {
            "move": self._handle_move_command,
            "grip": self._handle_grip_command,
            "release": self._handle_release_command,
            "stop": self._handle_stop_command,
            "tool": self._handle_tool_command,
            "query": self._handle_query_command,
            "emergency": self._handle_emergency_command
        }

        # コールバック
        self.on_command_executed: Optional[Callable[[MultimodalCommand, Any], None]] = None
        self.on_interaction_detected: Optional[Callable[[MultimodalCommand], None]] = None

        # 実行制御
        self.execution_queue = queue.Queue(maxsize=100)
        self.execution_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 設定
        self.command_timeout = 30.0  # 秒
        self.max_concurrent_commands = 3

    def start_controller(self, camera_index: int = 0) -> bool:
        """コントローラ起動"""
        try:
            self.is_running = True

            # マルチモーダルインタラクション開始
            if not self.command_processor.start_interaction(camera_index):
                logger.error("Failed to start multimodal interaction")
                return False

            # 実行スレッド開始
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()

            logger.info("Voice/Gesture controller started")
            return True

        except Exception as e:
            logger.error(f"Failed to start controller: {e}")
            return False

    def stop_controller(self):
        """コントローラ停止"""
        self.is_running = False

        self.command_processor.stop_interaction()

        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5.0)

        logger.info("Voice/Gesture controller stopped")

    def _execution_loop(self):
        """実行ループ"""
        logger.info("Command execution loop started")

        while self.is_running:
            try:
                # コマンドキューから取得
                try:
                    command = self.execution_queue.get(timeout=1.0)
                    self._execute_command(command)
                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                time.sleep(1.0)

        logger.info("Command execution loop ended")

    def _execute_command(self, command: MultimodalCommand):
        """コマンド実行"""
        try:
            start_time = time.time()

            if command.intent in self.command_handlers:
                # ハンドラ実行
                result = self.command_handlers[command.intent](command)

                # 実行時間記録
                execution_time = time.time() - start_time

                # コールバック実行
                if self.on_command_executed:
                    self.on_command_executed(command, result)

                # インタラクションコールバック
                if self.on_interaction_detected:
                    self.on_interaction_detected(command)

                logger.info(f"Executed command {command.intent} in {execution_time:.2f}s")

            else:
                logger.warning(f"Unknown command intent: {command.intent}")

        except Exception as e:
            logger.error(f"Command execution failed: {e}")

    # コマンドハンドラ実装
    def _handle_move_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """移動コマンド処理"""
        try:
            destination = command.parameters.get("destination")
            speed = command.parameters.get("speed", "normal")

            # 速度レベルを数値に変換
            speed_map = {"slow": 0.3, "normal": 0.5, "fast": 0.8}
            actual_speed = speed_map.get(speed, 0.5)

            result = {
                "command": "move",
                "destination": destination,
                "speed": actual_speed,
                "executed": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Move command: destination={destination}, speed={actual_speed}")
            return result

        except Exception as e:
            logger.error(f"Move command handling failed: {e}")
            return {"command": "move", "executed": False, "error": str(e)}

    def _handle_grip_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """把持コマンド処理"""
        try:
            force = command.parameters.get("force", "normal")
            position = command.parameters.get("position")

            # 力レベルを数値に変換
            force_map = {"light": 30.0, "normal": 50.0, "strong": 80.0}
            actual_force = force_map.get(force, 50.0)

            result = {
                "command": "grip",
                "force": actual_force,
                "position": position,
                "executed": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Grip command: force={actual_force}N, position={position}")
            return result

        except Exception as e:
            logger.error(f"Grip command handling failed: {e}")
            return {"command": "grip", "executed": False, "error": str(e)}

    def _handle_release_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """離脱コマンド処理"""
        try:
            result = {
                "command": "release",
                "executed": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Release command executed")
            return result

        except Exception as e:
            logger.error(f"Release command handling failed: {e}")
            return {"command": "release", "executed": False, "error": str(e)}

    def _handle_stop_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """停止コマンド処理"""
        try:
            result = {
                "command": "stop",
                "executed": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.warning("Stop command executed")
            return result

        except Exception as e:
            logger.error(f"Stop command handling failed: {e}")
            return {"command": "stop", "executed": False, "error": str(e)}

    def _handle_tool_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """ツールコマンド処理"""
        try:
            tool_type = command.parameters.get("tool_type")

            result = {
                "command": "tool",
                "tool_type": tool_type,
                "executed": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Tool command: tool_type={tool_type}")
            return result

        except Exception as e:
            logger.error(f"Tool command handling failed: {e}")
            return {"command": "tool", "executed": False, "error": str(e)}

    def _handle_query_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """クエリコマンド処理"""
        try:
            query_type = command.parameters.get("query_type", "status")

            # ステータス情報取得
            status_info = {
                "robot_status": "operational",
                "system_status": "running",
                "last_command": command.command_text if command.voice else "gesture",
                "timestamp": datetime.now().isoformat()
            }

            result = {
                "command": "query",
                "query_type": query_type,
                "result": status_info,
                "executed": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Query command: {query_type}")
            return result

        except Exception as e:
            logger.error(f"Query command handling failed: {e}")
            return {"command": "query", "executed": False, "error": str(e)}

    def _handle_emergency_command(self, command: MultimodalCommand) -> Dict[str, Any]:
        """緊急コマンド処理"""
        try:
            # 即時実行
            result = {
                "command": "emergency",
                "executed": True,
                "timestamp": datetime.now().isoformat(),
                "priority": "critical"
            }

            logger.warning("Emergency command executed immediately")
            return result

        except Exception as e:
            logger.error(f"Emergency command handling failed: {e}")
            return {"command": "emergency", "executed": False, "error": str(e)}

    def submit_command(self, voice: VoiceCommand = None, gesture: Gesture = None) -> Optional[str]:
        """コマンド送信"""
        try:
            command = self.command_processor.process_multimodal_command(voice, gesture)

            if command:
                # 実行キューに追加
                self.execution_queue.put(command)

                # 統計更新
                stats = self.command_processor.get_interaction_statistics()
                logger.info(f"Submitted command: {command.intent} (confidence: {command.confidence:.2f})")

                return command.timestamp.isoformat()

            return None

        except Exception as e:
            logger.error(f"Command submission failed: {e}")
            return None

    def get_controller_status(self) -> Dict[str, Any]:
        """コントローラ状態取得"""
        return {
            "running": self.is_running,
            "execution_queue_size": self.execution_queue.qsize(),
            "interaction_stats": self.command_processor.get_interaction_statistics(),
            "available_handlers": list(self.command_handlers.keys())
        }

# グローバルインスタンス
voice_gesture_controller: Optional[VoiceGestureController] = None

def initialize_voice_gesture_controller(production_system: ProductionManagementSystem) -> VoiceGestureController:
    """音声・ジェスチャーコントローラ初期化"""
    global voice_gesture_controller
    voice_gesture_controller = VoiceGestureController(production_system)
    return voice_gesture_controller

def get_voice_gesture_controller() -> Optional[VoiceGestureController]:
    """音声・ジェスチャーコントローラ取得"""
    return voice_gesture_controller

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Voice and Gesture Interface System...")

    try:
        # モック生産管理システム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())

        # 音声・ジェスチャーコントローラ初期化
        controller = initialize_voice_gesture_controller(mock_pms)

        if controller.start_controller():
            print("Voice and gesture controller started successfully!")

            # テスト実行
            print("Testing voice commands...")
            time.sleep(2)

            # コントローラ状態確認
            status = controller.get_controller_status()
            print(f"Controller status: {status}")

            time.sleep(2)
            controller.stop_controller()

        else:
            print("Failed to start voice and gesture controller")

    except Exception as e:
        print(f"Voice and gesture controller test failed: {e}")