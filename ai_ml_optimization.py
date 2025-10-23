"""
AI/MLプロセス最適化機能 (Phase 4-5)
機械学習による自動最適化・予測・異常検知
強化学習・深層学習・統計的モデリング
"""

import json
import time
import logging
import threading
import pickle
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid
from abc import ABC, abstractmethod

# 機械学習ライブラリ
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, classification_report
    from sklearn.cluster import KMeans, DBSCAN
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 深層学習ライブラリ（オプション）
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from production_management_integration import ProductionManagementSystem, ProductionMetrics
from multi_robot_coordination import RobotInfo, RobotState

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """最適化タイプ"""
    PARAMETER_TUNING = "parameter_tuning"
    SCHEDULE_OPTIMIZATION = "schedule_optimization"
    QUALITY_PREDICTION = "quality_prediction"
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    ENERGY_OPTIMIZATION = "energy_optimization"

class ModelType(Enum):
    """モデルタイプ"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    ISOLATION_FOREST = "isolation_forest"
    KMEANS = "kmeans"
    CUSTOM = "custom"

class PredictionResult:
    """予測結果"""
    def __init__(self, predicted_value: float, confidence: float,
                 feature_importance: Dict[str, float] = None):
        self.predicted_value = predicted_value
        self.confidence = confidence
        self.feature_importance = feature_importance or {}

@dataclass
class OptimizationRecommendation:
    """最適化推奨"""
    recommendation_id: str
    optimization_type: OptimizationType
    target_robot: str
    current_parameters: Dict[str, Any]
    recommended_parameters: Dict[str, Any]
    expected_improvement: float  # 期待改善率
    confidence: float  # 信頼度
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    estimated_savings: float  # 見込みコスト削減
    created_at: datetime
    applied: bool = False
    applied_at: Optional[datetime] = None

@dataclass
class ModelPerformance:
    """モデル性能"""
    model_id: str
    model_type: ModelType
    target_variable: str
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    training_date: datetime
    validation_samples: int = 0

class MLModel(ABC):
    """機械学習モデル基底クラス"""

    def __init__(self, model_id: str, model_type: ModelType):
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.target_column = ""
        self.is_trained = False
        self.performance = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """モデル学習"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測実行"""
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """モデル保存"""
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """モデル読み込み"""
        pass

class LinearRegressionModel(MLModel):
    """線形回帰モデル"""

    def __init__(self, model_id: str):
        super().__init__(model_id, ModelType.LINEAR_REGRESSION)

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            if not ML_AVAILABLE:
                logger.error("scikit-learn not available")
                return False

            # データ前処理
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # モデル学習
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)

            self.is_trained = True

            # 性能評価
            y_pred = self.model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            self.performance = ModelPerformance(
                model_id=self.model_id,
                model_type=self.model_type,
                target_variable=self.target_column,
                accuracy=r2,
                mse=mse,
                r2_score=r2,
                training_date=datetime.now(),
                validation_samples=len(X)
            )

            logger.info(f"Linear regression model trained: {self.model_id}, R²={r2:.3f}")
            return True

        except Exception as e:
            logger.error(f"Linear regression training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, filepath: str) -> bool:
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'is_trained': self.is_trained,
                'performance': self.performance
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save linear regression model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.is_trained = model_data['is_trained']
            self.performance = model_data['performance']
            return True
        except Exception as e:
            logger.error(f"Failed to load linear regression model: {e}")
            return False

class RandomForestModel(MLModel):
    """ランダムフォレストモデル"""

    def __init__(self, model_id: str, n_estimators: int = 100):
        super().__init__(model_id, ModelType.RANDOM_FOREST)
        self.n_estimators = n_estimators

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            if not ML_AVAILABLE:
                logger.error("scikit-learn not available")
                return False

            # モデル学習
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)

            self.is_trained = True

            # 性能評価
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            self.performance = ModelPerformance(
                model_id=self.model_id,
                model_type=self.model_type,
                target_variable=self.target_column,
                accuracy=r2,
                mse=mse,
                r2_score=r2,
                training_date=datetime.now(),
                validation_samples=len(X)
            )

            logger.info(f"Random forest model trained: {self.model_id}, R²={r2:.3f}")
            return True

        except Exception as e:
            logger.error(f"Random forest training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度取得"""
        if not self.is_trained:
            return {}

        importance_dict = {}
        for i, importance in enumerate(self.model.feature_importances_):
            if i < len(self.feature_columns):
                importance_dict[self.feature_columns[i]] = importance

        return importance_dict

    def save_model(self, filepath: str) -> bool:
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'is_trained': self.is_trained,
                'performance': self.performance
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save random forest model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.is_trained = model_data['is_trained']
            self.performance = model_data['performance']
            return True
        except Exception as e:
            logger.error(f"Failed to load random forest model: {e}")
            return False

class NeuralNetworkModel(MLModel):
    """ニューラルネットワークモデル"""

    def __init__(self, model_id: str, hidden_layers: List[int] = None):
        super().__init__(model_id, ModelType.NEURAL_NETWORK)
        self.hidden_layers = hidden_layers or [64, 32]

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            if not TF_AVAILABLE:
                logger.error("TensorFlow not available")
                return False

            # データ前処理
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # モデル構築
            self.model = keras.Sequential()
            self.model.add(layers.Input(shape=(X.shape[1],)))

            # 隠れ層
            for units in self.hidden_layers:
                self.model.add(layers.Dense(units, activation='relu'))
                self.model.add(layers.Dropout(0.2))

            # 出力層
            self.model.add(layers.Dense(1))

            # コンパイル
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

            # 学習
            history = self.model.fit(
                X_scaled, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )

            self.is_trained = True

            # 性能評価
            y_pred = self.model.predict(X_scaled).flatten()
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            self.performance = ModelPerformance(
                model_id=self.model_id,
                model_type=self.model_type,
                target_variable=self.target_column,
                accuracy=r2,
                mse=mse,
                r2_score=r2,
                training_date=datetime.now(),
                validation_samples=len(X)
            )

            logger.info(f"Neural network model trained: {self.model_id}, R²={r2:.3f}")
            return True

        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).flatten()

    def save_model(self, filepath: str) -> bool:
        try:
            # Kerasモデルとスケーラーを別々に保存
            model_path = filepath.replace('.pkl', '_model.h5')
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')

            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)

            # メタデータ保存
            metadata = {
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'is_trained': self.is_trained,
                'performance': self.performance,
                'model_path': model_path,
                'scaler_path': scaler_path
            }
            joblib.dump(metadata, filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save neural network model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            metadata = joblib.load(filepath)
            self.feature_columns = metadata['feature_columns']
            self.target_column = metadata['target_column']
            self.is_trained = metadata['is_trained']
            self.performance = metadata['performance']

            # Kerasモデルとスケーラーを読み込み
            self.model = keras.models.load_model(metadata['model_path'])
            self.scaler = joblib.load(metadata['scaler_path'])
            return True
        except Exception as e:
            logger.error(f"Failed to load neural network model: {e}")
            return False

class AnomalyDetectionModel(MLModel):
    """異常検知モデル"""

    def __init__(self, model_id: str, contamination: float = 0.1):
        super().__init__(model_id, ModelType.ISOLATION_FOREST)
        self.contamination = contamination

    def train(self, X: np.ndarray, y: np.ndarray = None) -> bool:
        try:
            if not ML_AVAILABLE:
                logger.error("scikit-learn not available")
                return False

            # 異常検知モデル学習
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(X)

            self.is_trained = True

            self.performance = ModelPerformance(
                model_id=self.model_id,
                model_type=self.model_type,
                target_variable="anomaly_score",
                accuracy=0.95,  # 推定値
                training_date=datetime.now(),
                validation_samples=len(X)
            )

            logger.info(f"Anomaly detection model trained: {self.model_id}")
            return True

        except Exception as e:
            logger.error(f"Anomaly detection training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        # 異常スコアを返す（-1が異常、1が正常）
        return self.model.decision_function(X)

    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """異常検知実行"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)  # -1: 異常, 1: 正常

    def save_model(self, filepath: str) -> bool:
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'performance': self.performance
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save anomaly detection model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            self.performance = model_data['performance']
            return True
        except Exception as e:
            logger.error(f"Failed to load anomaly detection model: {e}")
            return False

class ProcessOptimizer:
    """プロセス最適化エンジン"""

    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.recommendations: List[OptimizationRecommendation] = []
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        # 設定
        self.min_training_samples = 50
        self.model_retrain_interval = 7  # 日
        self.recommendation_confidence_threshold = 0.7

    def add_training_data(self, data_type: str, data: pd.DataFrame):
        """学習データ追加"""
        if data_type not in self.training_data:
            self.training_data[data_type] = pd.DataFrame()

        self.training_data[data_type] = pd.concat([self.training_data[data_type], data], ignore_index=True)

        # データ量を制限
        max_samples = 10000
        if len(self.training_data[data_type]) > max_samples:
            self.training_data[data_type] = self.training_data[data_type].tail(max_samples)

    def create_model(self, model_id: str, model_type: ModelType, **kwargs) -> bool:
        """モデル作成"""
        try:
            if model_type == ModelType.LINEAR_REGRESSION:
                model = LinearRegressionModel(model_id)
            elif model_type == ModelType.RANDOM_FOREST:
                n_estimators = kwargs.get('n_estimators', 100)
                model = RandomForestModel(model_id, n_estimators)
            elif model_type == ModelType.NEURAL_NETWORK:
                hidden_layers = kwargs.get('hidden_layers', [64, 32])
                model = NeuralNetworkModel(model_id, hidden_layers)
            elif model_type == ModelType.ISOLATION_FOREST:
                contamination = kwargs.get('contamination', 0.1)
                model = AnomalyDetectionModel(model_id, contamination)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False

            self.models[model_id] = model
            logger.info(f"Model created: {model_id} ({model_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to create model {model_id}: {e}")
            return False

    def train_model(self, model_id: str, data_type: str,
                   feature_columns: List[str], target_column: str) -> bool:
        """モデル学習"""
        try:
            if model_id not in self.models:
                logger.error(f"Model not found: {model_id}")
                return False

            if data_type not in self.training_data:
                logger.error(f"Training data not found: {data_type}")
                return False

            data = self.training_data[data_type]

            # データチェック
            if len(data) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(data)} < {self.min_training_samples}")
                return False

            # 欠損値処理
            data_clean = data.dropna()

            if len(data_clean) < self.min_training_samples:
                logger.warning(f"Insufficient clean data: {len(data_clean)} < {self.min_training_samples}")
                return False

            # 特徴量とターゲット準備
            X = data_clean[feature_columns].values
            y = data_clean[target_column].values

            # モデル設定
            model = self.models[model_id]
            model.feature_columns = feature_columns
            model.target_column = target_column

            # 学習実行
            success = model.train(X, y)

            if success:
                # モデル保存
                model_path = f"models/{model_id}.pkl"
                self._ensure_models_directory()
                model.save_model(model_path)

                # 学習記録
                self.optimization_history.append({
                    "action": "model_trained",
                    "model_id": model_id,
                    "data_type": data_type,
                    "samples": len(X),
                    "performance": asdict(model.performance) if model.performance else None,
                    "timestamp": datetime.now().isoformat()
                })

            return success

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def predict_quality(self, robot_id: str, process_parameters: Dict[str, float]) -> Optional[PredictionResult]:
        """品質予測"""
        try:
            model_id = f"quality_predictor_{robot_id}"

            if model_id not in self.models or not self.models[model_id].is_trained:
                logger.warning(f"Quality prediction model not available for {robot_id}")
                return None

            model = self.models[model_id]

            # 特徴量準備
            X = np.array([[process_parameters.get(col, 0) for col in model.feature_columns]])

            # 予測実行
            predicted_quality = model.predict(X)[0]

            # 信頼度計算（モデル性能に基づく）
            confidence = 0.8
            if model.performance and model.performance.accuracy:
                confidence = min(model.performance.accuracy, 0.95)

            # 特徴量重要度
            feature_importance = {}
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()

            return PredictionResult(
                predicted_value=predicted_quality,
                confidence=confidence,
                feature_importance=feature_importance
            )

        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            return None

    def predict_maintenance_need(self, robot_id: str, operational_data: Dict[str, float]) -> Optional[PredictionResult]:
        """メンテナンス必要度予測"""
        try:
            model_id = f"maintenance_predictor_{robot_id}"

            if model_id not in self.models or not self.models[model_id].is_trained:
                logger.warning(f"Maintenance prediction model not available for {robot_id}")
                return None

            model = self.models[model_id]

            # 特徴量準備
            X = np.array([[operational_data.get(col, 0) for col in model.feature_columns]])

            # 予測実行（次回メンテナンスまでの時間）
            predicted_time = model.predict(X)[0]

            # 信頼度計算
            confidence = 0.7
            if model.performance and model.performance.accuracy:
                confidence = min(model.performance.accuracy, 0.9)

            return PredictionResult(
                predicted_value=predicted_time,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Maintenance prediction failed: {e}")
            return None

    def detect_anomalies(self, robot_id: str, current_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """異常検知"""
        try:
            model_id = f"anomaly_detector_{robot_id}"

            if model_id not in self.models or not self.models[model_id].is_trained:
                logger.warning(f"Anomaly detection model not available for {robot_id}")
                return None

            model = self.models[model_id]

            # 特徴量準備
            X = np.array([[current_data.get(col, 0) for col in model.feature_columns]])

            # 異常スコア計算
            anomaly_score = model.predict(X)[0]
            is_anomaly = model.detect_anomalies(X)[0] == -1

            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "threshold": -0.1,  # 異常判定閾値
                "confidence": abs(anomaly_score)
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return None

    def generate_optimization_recommendations(self, robot_id: str,
                                            current_state: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """最適化推奨生成"""
        recommendations = []

        try:
            # 品質改善推奨
            quality_rec = self._generate_quality_optimization(robot_id, current_state)
            if quality_rec:
                recommendations.append(quality_rec)

            # エネルギー効率化推奨
            energy_rec = self._generate_energy_optimization(robot_id, current_state)
            if energy_rec:
                recommendations.append(energy_rec)

            # サイクルタイム最適化推奨
            cycle_rec = self._generate_cycle_time_optimization(robot_id, current_state)
            if cycle_rec:
                recommendations.append(cycle_rec)

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")

        return recommendations

    def _generate_quality_optimization(self, robot_id: str, current_state: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """品質最適化推奨生成"""
        try:
            current_quality = current_state.get('quality_score', 0.95)

            if current_quality >= 0.98:
                return None  # すでに高品質

            # パラメータ最適化シミュレーション
            best_params = current_state.get('parameters', {}).copy()
            expected_improvement = 0.0

            # 各パラメータの影響を評価
            param_effects = {
                'speed': -0.02,    # 速度を下げると品質向上
                'pressure': 0.03,   # 圧力を上げると品質向上
                'temperature': 0.01 # 温度を上げると品質向上
            }

            for param, effect in param_effects.items():
                if param in best_params:
                    # パラメータ調整（10%変更）
                    old_value = best_params[param]
                    best_params[param] *= 1.1
                    expected_improvement += abs(effect * 0.1)

            if expected_improvement < 0.01:
                return None

            return OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.PARAMETER_TUNING,
                target_robot=robot_id,
                current_parameters=current_state.get('parameters', {}),
                recommended_parameters=best_params,
                expected_improvement=expected_improvement * 100,  # パーセントで返す
                confidence=0.75,
                implementation_effort="low",
                risk_level="low",
                estimated_savings=expected_improvement * 1000,  # 推定コスト削減
                created_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Quality optimization generation failed: {e}")
            return None

    def _generate_energy_optimization(self, robot_id: str, current_state: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """エネルギー最適化推奨生成"""
        try:
            current_energy = current_state.get('energy_consumption', 5.0)

            # アイドル時間削減によるエネルギー節約
            idle_reduction = current_state.get('idle_time_percentage', 10) * 0.1
            energy_saving = current_energy * idle_reduction

            if energy_saving < 0.1:
                return None

            # 推奨パラメータ
            recommended_params = current_state.get('parameters', {}).copy()
            recommended_params['idle_speed_reduction'] = 0.3  # アイドル時30%速度低下

            return OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.ENERGY_OPTIMIZATION,
                target_robot=robot_id,
                current_parameters=current_state.get('parameters', {}),
                recommended_parameters=recommended_params,
                expected_improvement=(energy_saving / current_energy) * 100,
                confidence=0.8,
                implementation_effort="low",
                risk_level="low",
                estimated_savings=energy_saving * 0.15 * 24 * 365,  # 年間コスト削減（$0.15/kWh）
                created_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Energy optimization generation failed: {e}")
            return None

    def _generate_cycle_time_optimization(self, robot_id: str, current_state: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """サイクルタイム最適化推奨生成"""
        try:
            current_cycle_time = current_state.get('cycle_time', 45.0)

            if current_cycle_time <= 35.0:
                return None  # すでに最適

            # 軌道最適化による短縮
            trajectory_improvement = current_cycle_time * 0.05  # 5%改善

            recommended_params = current_state.get('parameters', {}).copy()
            recommended_params['trajectory_optimization'] = True
            recommended_params['motion_smoothing'] = 0.8

            return OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.PARAMETER_TUNING,
                target_robot=robot_id,
                current_parameters=current_state.get('parameters', {}),
                recommended_parameters=recommended_params,
                expected_improvement=5.0,  # 5%改善
                confidence=0.7,
                implementation_effort="medium",
                risk_level="low",
                estimated_savings=trajectory_improvement / current_cycle_time * 500,  # 時間価値
                created_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Cycle time optimization generation failed: {e}")
            return None

    def apply_recommendation(self, recommendation_id: str) -> bool:
        """推奨適用"""
        try:
            for rec in self.recommendations:
                if rec.recommendation_id == recommendation_id:
                    rec.applied = True
                    rec.applied_at = datetime.now()

                    # 適用記録
                    self.optimization_history.append({
                        "action": "recommendation_applied",
                        "recommendation_id": recommendation_id,
                        "robot_id": rec.target_robot,
                        "improvement": rec.expected_improvement,
                        "timestamp": datetime.now().isoformat()
                    })

                    logger.info(f"Applied recommendation: {recommendation_id}")
                    return True

            logger.warning(f"Recommendation not found: {recommendation_id}")
            return False

        except Exception as e:
            logger.error(f"Failed to apply recommendation: {e}")
            return False

    def _ensure_models_directory(self):
        """モデルディレクトリ作成"""
        import os
        os.makedirs("models", exist_ok=True)

    def get_optimization_status(self) -> Dict[str, Any]:
        """最適化ステータス取得"""
        return {
            "total_models": len(self.models),
            "trained_models": len([m for m in self.models.values() if m.is_trained]),
            "total_recommendations": len(self.recommendations),
            "applied_recommendations": len([r for r in self.recommendations if r.applied]),
            "pending_recommendations": len([r for r in self.recommendations if not r.applied]),
            "training_data_types": list(self.training_data.keys()),
            "optimization_history_count": len(self.optimization_history),
            "models": {
                model_id: {
                    "type": model.model_type.value,
                    "trained": model.is_trained,
                    "performance": asdict(model.performance) if model.performance else None
                }
                for model_id, model in self.models.items()
            }
        }

class AIOptimizationService:
    """AI最適化サービス"""

    def __init__(self, production_system: ProductionManagementSystem):
        self.production_system = production_system
        self.optimizer = ProcessOptimizer()

        # 実行制御
        self.running = False
        self.optimization_thread: Optional[threading.Thread] = None

        # 設定
        self.optimization_interval = 300  # 5分間隔
        self.auto_apply_threshold = 0.9  # 自動適用閾値

        # コールバック
        self.on_optimization_completed: Optional[Callable[[List[OptimizationRecommendation]], None]] = None
        self.on_anomaly_detected: Optional[Callable[[Dict[str, Any]], None]] = None

    def start(self) -> bool:
        """AI最適化サービス開始"""
        try:
            if not ML_AVAILABLE:
                logger.error("scikit-learn not available. AI optimization service disabled.")
                return False

            if self.running:
                logger.warning("AI optimization service already running")
                return False

            self.running = True

            # モデル初期化
            self._initialize_models()

            # 最適化スレッド開始
            self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
            self.optimization_thread.start()

            logger.info("AI optimization service started")
            return True

        except Exception as e:
            logger.error(f"Failed to start AI optimization service: {e}")
            return False

    def stop(self):
        """AI最適化サービス停止"""
        self.running = False

        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5.0)

        logger.info("AI optimization service stopped")

    def _initialize_models(self):
        """モデル初期化"""
        # 品質予測モデル
        for robot_id in ['robot_001', 'robot_002', 'robot_003']:
            self.optimizer.create_model(
                f"quality_predictor_{robot_id}",
                ModelType.RANDOM_FOREST,
                n_estimators=50
            )

            # メンテナンス予測モデル
            self.optimizer.create_model(
                f"maintenance_predictor_{robot_id}",
                ModelType.LINEAR_REGRESSION
            )

            # 異常検知モデル
            self.optimizer.create_model(
                f"anomaly_detector_{robot_id}",
                ModelType.ISOLATION_FOREST,
                contamination=0.05
            )

    def _optimization_loop(self):
        """最適化ループ"""
        logger.info("AI optimization loop started")

        while self.running:
            try:
                # データ収集
                self._collect_training_data()

                # 異常検知
                self._perform_anomaly_detection()

                # 最適化推奨生成
                self._generate_optimization_recommendations()

                time.sleep(self.optimization_interval)

            except Exception as e:
                logger.error(f"AI optimization error: {e}")
                time.sleep(60.0)  # エラー時は1分後に再試行

        logger.info("AI optimization loop ended")

    def _collect_training_data(self):
        """学習データ収集"""
        try:
            # 生産データ取得
            dashboard_data = self.production_system.get_production_dashboard()

            # 模擬学習データ生成
            current_time = datetime.now()
            training_samples = []

            for robot_id in ['robot_001', 'robot_002', 'robot_003']:
                # ランダムなパラメータと結果を生成
                sample = {
                    'timestamp': current_time,
                    'robot_id': robot_id,
                    'speed': 50 + np.random.normal(0, 5),
                    'pressure': 100 + np.random.normal(0, 10),
                    'temperature': 25 + np.random.normal(0, 2),
                    'cycle_time': 45 + np.random.normal(0, 5),
                    'quality_score': 0.95 + np.random.normal(0, 0.02),
                    'energy_consumption': 5.0 + np.random.normal(0, 0.5),
                    'vibration': 0.5 + np.random.normal(0, 0.1),
                    'temperature_rise': 10 + np.random.normal(0, 2)
                }

                training_samples.append(sample)

            # データフレームに変換して追加
            if training_samples:
                df = pd.DataFrame(training_samples)
                self.optimizer.add_training_data("production_data", df)

        except Exception as e:
            logger.error(f"Training data collection failed: {e}")

    def _perform_anomaly_detection(self):
        """異常検知実行"""
        try:
            for robot_id in ['robot_001', 'robot_002', 'robot_003']:
                # 現在データ（模擬）
                current_data = {
                    'speed': 50 + np.random.normal(0, 5),
                    'pressure': 100 + np.random.normal(0, 10),
                    'temperature': 25 + np.random.normal(0, 2),
                    'cycle_time': 45 + np.random.normal(0, 5),
                    'energy_consumption': 5.0 + np.random.normal(0, 0.5),
                    'vibration': 0.5 + np.random.normal(0, 0.1)
                }

                # 異常検知
                anomaly_result = self.optimizer.detect_anomalies(robot_id, current_data)

                if anomaly_result and anomaly_result['is_anomaly']:
                    logger.warning(f"Anomaly detected for {robot_id}: score={anomaly_result['anomaly_score']:.3f}")

                    if self.on_anomaly_detected:
                        self.on_anomaly_detected({
                            "robot_id": robot_id,
                            "anomaly_result": anomaly_result,
                            "timestamp": datetime.now().isoformat()
                        })

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

    def _generate_optimization_recommendations(self):
        """最適化推奨生成"""
        try:
            for robot_id in ['robot_001', 'robot_002', 'robot_003']:
                # 現在状態（模擬）
                current_state = {
                    'quality_score': 0.94 + np.random.normal(0, 0.02),
                    'cycle_time': 47 + np.random.normal(0, 3),
                    'energy_consumption': 5.2 + np.random.normal(0, 0.3),
                    'idle_time_percentage': 8 + np.random.normal(0, 2),
                    'parameters': {
                        'speed': 50,
                        'pressure': 100,
                        'temperature': 25
                    }
                }

                # 推奨生成
                recommendations = self.optimizer.generate_optimization_recommendations(robot_id, current_state)

                for rec in recommendations:
                    self.optimizer.recommendations.append(rec)

                    # 高信頼度推奨は自動適用
                    if rec.confidence >= self.auto_apply_threshold:
                        self.optimizer.apply_recommendation(rec.recommendation_id)
                        logger.info(f"Auto-applied recommendation for {robot_id}: {rec.expected_improvement:.1f}% improvement")

            # コールバック実行
            if self.on_optimization_completed and self.optimizer.recommendations:
                recent_recommendations = self.optimizer.recommendations[-5:]  # 最新5件
                self.on_optimization_completed(recent_recommendations)

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")

    def predict_quality(self, robot_id: str, parameters: Dict[str, float]) -> Optional[PredictionResult]:
        """品質予測API"""
        return self.optimizer.predict_quality(robot_id, parameters)

    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """最適化ダッシュボードデータ取得"""
        return {
            "timestamp": datetime.now().isoformat(),
            "status": self.optimizer.get_optimization_status(),
            "recent_recommendations": [asdict(rec) for rec in self.optimizer.recommendations[-10:]],
            "optimization_history": self.optimizer.optimization_history[-20:],
            "service_running": self.running,
            "ml_available": ML_AVAILABLE,
            "tf_available": TF_AVAILABLE
        }

# グローバルインスタンス
ai_optimization_service: Optional[AIOptimizationService] = None

def initialize_ai_optimization(production_system: ProductionManagementSystem) -> AIOptimizationService:
    """AI最適化サービス初期化"""
    global ai_optimization_service
    ai_optimization_service = AIOptimizationService(production_system)
    return ai_optimization_service

def get_ai_optimization_service() -> Optional[AIOptimizationService]:
    """AI最適化サービス取得"""
    return ai_optimization_service

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing AI/ML Optimization Service...")

    try:
        # モック生産管理システム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())

        # AI最適化サービス初期化
        service = initialize_ai_optimization(mock_pms)

        if service.start():
            print("AI/ML optimization service started successfully!")

            # テスト予測実行
            time.sleep(2)

            # 品質予測テスト
            test_params = {'speed': 50, 'pressure': 100, 'temperature': 25}
            prediction = service.predict_quality('robot_001', test_params)

            if prediction:
                print(f"Quality prediction: {prediction.predicted_value:.3f} (confidence: {prediction.confidence:.2f})")

            # 最適化ダッシュボード取得
            dashboard = service.get_optimization_dashboard()
            print(f"Dashboard status:")
            print(f"  Models: {dashboard['status']['total_models']}")
            print(f"  Trained: {dashboard['status']['trained_models']}")
            print(f"  Recommendations: {dashboard['status']['total_recommendations']}")

            time.sleep(2)
            service.stop()

        else:
            print("Failed to start AI/ML optimization service")

    except Exception as e:
        print(f"AI/ML optimization service test failed: {e}")