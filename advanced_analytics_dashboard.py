"""
高度な分析ダッシュボード (Phase 4-4)
リアルタイムKPI計算・統計分析・予測分析機能
産業レベルの生産分析・OEE改善提案システム
"""

import json
import time
import logging
import threading
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import uuid
from abc import ABC, abstractmethod

# データ分析ライブラリ
try:
    import scipy.stats as stats
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

from production_management_integration import ProductionManagementSystem, ProductionMetrics, QualityRecord
from multi_robot_coordination import RobotState, RobotInfo

logger = logging.getLogger(__name__)

class KPICategory(Enum):
    """KPIカテゴリ"""
    PRODUCTIVITY = "productivity"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    MAINTENANCE = "maintenance"
    COST = "cost"
    SAFETY = "safety"

class TrendDirection(Enum):
    """トレンド方向"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """アラート重大度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class KPIDefinition:
    """KPI定義"""
    kpi_id: str
    name: str
    category: KPICategory
    description: str
    unit: str
    target_value: float
    minimum_acceptable: float
    calculation_method: str
    data_sources: List[str]
    update_frequency: int  # 分
    aggregation_period: str  # hour, day, week, month

@dataclass
class KPIValue:
    """KPI値"""
    kpi_id: str
    timestamp: datetime
    value: float
    target: float
    performance_percentage: float
    trend_direction: TrendDirection
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class TrendAnalysis:
    """トレンド分析"""
    kpi_id: str
    period_start: datetime
    period_end: datetime
    trend_direction: TrendDirection
    trend_strength: float  # 0-1
    slope: float
    r_squared: float
    prediction_next_period: float
    prediction_confidence: float

@dataclass
class AnomalyDetection:
    """異常検知"""
    detection_id: str
    kpi_id: str
    timestamp: datetime
    value: float
    expected_range: Tuple[float, float]
    anomaly_score: float  # 0-1
    severity: AlertSeverity
    description: str

@dataclass
class OEEAnalysis:
    """OEE分析"""
    timestamp: datetime
    overall_oee: float
    availability: float
    performance: float
    quality_rate: float
    availability_losses: Dict[str, float]
    performance_losses: Dict[str, float]
    quality_losses: Dict[str, float]
    improvement_potential: float

@dataclass
class ProductionInsight:
    """生産インサイト"""
    insight_id: str
    category: str
    title: str
    description: str
    impact_level: str  # low, medium, high
    actionable: bool
    recommendations: List[str]
    data_evidence: Dict[str, Any]
    created_at: datetime

class KPIEngine:
    """KPI計算エンジン"""

    def __init__(self):
        self.kpi_definitions: Dict[str, KPIDefinition] = {}
        self.kpi_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_values: Dict[str, KPIValue] = {}

        # データバッファ
        self.metrics_buffer: deque = deque(maxlen=1000)
        self.quality_buffer: deque = deque(maxlen=500)

        # 計算統計
        self.calculation_stats = {
            'total_calculations': 0,
            'calculation_errors': 0,
            'last_calculation': None
        }

        self._initialize_kpi_definitions()

    def _initialize_kpi_definitions(self):
        """KPI定義初期化"""
        kpis = [
            KPIDefinition(
                kpi_id="oee",
                name="Overall Equipment Effectiveness",
                category=KPICategory.EFFICIENCY,
                description="Overall equipment effectiveness combining availability, performance, and quality",
                unit="%",
                target_value=85.0,
                minimum_acceptable=75.0,
                calculation_method="oee_formula",
                data_sources=["production_metrics", "quality_records"],
                update_frequency=5,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="availability",
                name="Equipment Availability",
                category=KPICategory.EFFICIENCY,
                description="Percentage of scheduled time that equipment is available for production",
                unit="%",
                target_value=95.0,
                minimum_acceptable=90.0,
                calculation_method="availability_formula",
                data_sources=["production_metrics", "maintenance_logs"],
                update_frequency=5,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="performance",
                name="Equipment Performance",
                category=KPICategory.EFFICIENCY,
                description="Speed at which the equipment runs as a percentage of its designed speed",
                unit="%",
                target_value=95.0,
                minimum_acceptable=85.0,
                calculation_method="performance_formula",
                data_sources=["production_metrics"],
                update_frequency=5,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="quality_rate",
                name="Quality Rate",
                category=KPICategory.QUALITY,
                description="Percentage of good parts produced",
                unit="%",
                target_value=99.0,
                minimum_acceptable=95.0,
                calculation_method="quality_formula",
                data_sources=["quality_records"],
                update_frequency=10,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="cycle_time",
                name="Average Cycle Time",
                category=KPICategory.PRODUCTIVITY,
                description="Average time to complete one production cycle",
                unit="seconds",
                target_value=45.0,
                minimum_acceptable=60.0,
                calculation_method="average_cycle_time",
                data_sources=["production_metrics"],
                update_frequency=5,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="throughput",
                name="Production Throughput",
                category=KPICategory.PRODUCTIVITY,
                description="Number of units produced per hour",
                unit="units/hour",
                target_value=80.0,
                minimum_acceptable=60.0,
                calculation_method="throughput_formula",
                data_sources=["production_metrics"],
                update_frequency=5,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="first_pass_yield",
                name="First Pass Yield",
                category=KPICategory.QUALITY,
                description="Percentage of products that pass quality inspection without rework",
                unit="%",
                target_value=98.0,
                minimum_acceptable=95.0,
                calculation_method="fpy_formula",
                data_sources=["quality_records"],
                update_frequency=10,
                aggregation_period="hour"
            ),
            KPIDefinition(
                kpi_id="mtbf",
                name="Mean Time Between Failures",
                category=KPICategory.MAINTENANCE,
                description="Average time between equipment failures",
                unit="hours",
                target_value=168.0,  # 1 week
                minimum_acceptable=72.0,  # 3 days
                calculation_method="mtbf_formula",
                data_sources=["maintenance_logs", "error_logs"],
                update_frequency=60,
                aggregation_period="day"
            ),
            KPIDefinition(
                kpi_id="cost_per_unit",
                name="Cost Per Unit",
                category=KPICategory.COST,
                description="Production cost per unit produced",
                unit="currency",
                target_value=10.0,
                minimum_acceptable=15.0,
                calculation_method="cost_per_unit_formula",
                data_sources=["production_metrics", "cost_data"],
                update_frequency=60,
                aggregation_period="day"
            ),
            KPIDefinition(
                kpi_id="energy_efficiency",
                name="Energy Efficiency",
                category=KPICategory.COST,
                description="Energy consumption per unit produced",
                unit="kWh/unit",
                target_value=0.5,
                minimum_acceptable=0.8,
                calculation_method="energy_efficiency_formula",
                data_sources=["production_metrics", "energy_data"],
                update_frequency=15,
                aggregation_period="hour"
            )
        ]

        for kpi in kpis:
            self.kpi_definitions[kpi.kpi_id] = kpi

    def add_metrics_data(self, metrics: ProductionMetrics):
        """メトリクスデータ追加"""
        self.metrics_buffer.append(metrics)

    def add_quality_data(self, quality_record: QualityRecord):
        """品質データ追加"""
        self.quality_buffer.append(quality_record)

    def calculate_all_kpis(self) -> Dict[str, KPIValue]:
        """すべてのKPIを計算"""
        current_time = datetime.now()
        results = {}

        for kpi_id, definition in self.kpi_definitions.items():
            try:
                value = self._calculate_kpi(kpi_id, definition)

                if value is not None:
                    # トレンド分析
                    trend = self._analyze_trend(kpi_id)

                    # 信頼区間計算
                    confidence_interval = self._calculate_confidence_interval(kpi_id, value)

                    # KPI値作成
                    kpi_value = KPIValue(
                        kpi_id=kpi_id,
                        timestamp=current_time,
                        value=value,
                        target=definition.target_value,
                        performance_percentage=(value / definition.target_value) * 100 if definition.target_value > 0 else 0,
                        trend_direction=trend,
                        confidence_interval=confidence_interval
                    )

                    results[kpi_id] = kpi_value
                    self.current_values[kpi_id] = kpi_value
                    self.kpi_history[kpi_id].append(kpi_value)

                else:
                    logger.warning(f"Could not calculate KPI: {kpi_id}")

            except Exception as e:
                logger.error(f"Error calculating KPI {kpi_id}: {e}")
                self.calculation_stats['calculation_errors'] += 1

        self.calculation_stats['total_calculations'] += 1
        self.calculation_stats['last_calculation'] = current_time

        return results

    def _calculate_kpi(self, kpi_id: str, definition: KPIDefinition) -> Optional[float]:
        """個別KPI計算"""
        if kpi_id == "oee":
            return self._calculate_oee()
        elif kpi_id == "availability":
            return self._calculate_availability()
        elif kpi_id == "performance":
            return self._calculate_performance()
        elif kpi_id == "quality_rate":
            return self._calculate_quality_rate()
        elif kpi_id == "cycle_time":
            return self._calculate_cycle_time()
        elif kpi_id == "throughput":
            return self._calculate_throughput()
        elif kpi_id == "first_pass_yield":
            return self._calculate_first_pass_yield()
        elif kpi_id == "mtbf":
            return self._calculate_mtbf()
        elif kpi_id == "cost_per_unit":
            return self._calculate_cost_per_unit()
        elif kpi_id == "energy_efficiency":
            return self._calculate_energy_efficiency()
        else:
            logger.warning(f"Unknown KPI calculation method: {kpi_id}")
            return None

    def _calculate_oee(self) -> Optional[float]:
        """OEE計算"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-10:]  # 最新10件

        if not recent_metrics:
            return None

        # OEE = Availability × Performance × Quality Rate
        availabilities = [m.availability for m in recent_metrics if m.availability > 0]
        performances = [m.performance for m in recent_metrics if m.performance > 0]
        quality_rates = [m.quality_rate for m in recent_metrics if m.quality_rate > 0]

        if not availabilities or not performances or not quality_rates:
            return None

        avg_availability = statistics.mean(availabilities)
        avg_performance = statistics.mean(performances)
        avg_quality = statistics.mean(quality_rates)

        oee = avg_availability * avg_performance * avg_quality
        return oee * 100  # パーセントで返す

    def _calculate_availability(self) -> Optional[float]:
        """稼働率計算"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-20:]
        availabilities = [m.availability for m in recent_metrics if m.availability > 0]

        if not availabilities:
            return None

        return statistics.mean(availabilities) * 100

    def _calculate_performance(self) -> Optional[float]:
        """性能率計算"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-20:]
        performances = [m.performance for m in recent_metrics if m.performance > 0]

        if not performances:
            return None

        return statistics.mean(performances) * 100

    def _calculate_quality_rate(self) -> Optional[float]:
        """品質率計算"""
        if not self.metrics_buffer and not self.quality_buffer:
            return None

        quality_scores = []

        # メトリクスからの品質スコア
        if self.metrics_buffer:
            recent_metrics = list(self.metrics_buffer)[-20:]
            quality_scores.extend([m.quality_score for m in recent_metrics if m.quality_score > 0])

        # 品質記録からの合格率
        if self.quality_buffer:
            recent_quality = list(self.quality_buffer)[-50:]
            passed = sum(1 for q in recent_quality if q.status.value == 'pass')
            total = len(recent_quality)
            if total > 0:
                quality_scores.append(passed / total)

        if not quality_scores:
            return None

        return statistics.mean(quality_scores) * 100

    def _calculate_cycle_time(self) -> Optional[float]:
        """サイクルタイム計算"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-20:]
        cycle_times = [m.cycle_time for m in recent_metrics if m.cycle_time > 0]

        if not cycle_times:
            return None

        return statistics.mean(cycle_times)

    def _calculate_throughput(self) -> Optional[float]:
        """スループット計算"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-20:]
        throughputs = [m.throughput for m in recent_metrics if m.throughput > 0]

        if not throughputs:
            return None

        return statistics.mean(throughputs)

    def _calculate_first_pass_yield(self) -> Optional[float]:
        """初回合格率計算"""
        if not self.quality_buffer:
            return None

        recent_quality = list(self.quality_buffer)[-100:]

        # 再作業なしの合格品をカウント
        first_pass = sum(1 for q in recent_quality
                        if q.status.value == 'pass' and not q.defects)
        total = len(recent_quality)

        if total == 0:
            return None

        return (first_pass / total) * 100

    def _calculate_mtbf(self) -> Optional[float]:
        """平均故障間隔計算（模擬）"""
        # 実際の実装ではメンテナンスログから計算
        return 168.0 + (hash(str(datetime.now())) % 48) - 24  # 144-192時間

    def _calculate_cost_per_unit(self) -> Optional[float]:
        """単位あたりコスト計算（模擬）"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-20:]
        total_energy = sum(m.energy_consumption for m in recent_metrics)
        total_units = len(recent_metrics)

        if total_units == 0:
            return None

        # エネルギーコスト + 人件費 + 設備コスト（模擬）
        energy_cost = total_energy * 0.15  # $0.15/kWh
        labor_cost = total_units * 2.0  # $2/unit
        equipment_cost = total_units * 3.0  # $3/unit

        total_cost = energy_cost + labor_cost + equipment_cost
        return total_cost / total_units

    def _calculate_energy_efficiency(self) -> Optional[float]:
    """エネルギー効率計算"""
        if not self.metrics_buffer:
            return None

        recent_metrics = list(self.metrics_buffer)[-20:]

        total_energy = sum(m.energy_consumption for m in recent_metrics)
        total_units = len(recent_metrics)

        if total_units == 0 or total_energy == 0:
            return None

        return total_energy / total_units

    def _analyze_trend(self, kpi_id: str) -> TrendDirection:
        """トレンド分析"""
        history = list(self.kpi_history[kpi_id])

        if len(history) < 5:
            return TrendDirection.UNKNOWN

        # 最新の値と前の値を比較
        recent_values = [h.value for h in history[-5:]]
        earlier_values = [h.value for h in history[-10:-5]] if len(history) >= 10 else recent_values[:len(recent_values)]

        if not earlier_values:
            return TrendDirection.UNKNOWN

        recent_avg = statistics.mean(recent_values)
        earlier_avg = statistics.mean(earlier_values)

        change_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100 if earlier_avg > 0 else 0

        if change_percent > 5:
            return TrendDirection.IMPROVING
        elif change_percent < -5:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE

    def _calculate_confidence_interval(self, kpi_id: str, current_value: float) -> Optional[Tuple[float, float]]:
        """信頼区間計算"""
        history = list(self.kpi_history[kpi_id])

        if len(history) < 10:
            return None

        values = [h.value for h in history[-30:]]  # 最新30件

        if len(values) < 3:
            return None

        try:
            # 95%信頼区間
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            margin = 1.96 * (std_val / len(values) ** 0.5)  # 95% CI

            return (mean_val - margin, mean_val + margin)
        except statistics.StatisticsError:
            return None

    def get_kpi_summary(self) -> Dict[str, Any]:
        """KPIサマリー取得"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_kpis": len(self.kpi_definitions),
            "calculated_kpis": len(self.current_values),
            "categories": {},
            "overall_health": "unknown"
        }

        # カテゴリ別集計
        category_stats = defaultdict(lambda: {"count": 0, "meeting_target": 0, "critical_issues": 0})

        for kpi_id, kpi_value in self.current_values.items():
            if kpi_id in self.kpi_definitions:
                definition = self.kpi_definitions[kpi_id]
                category = definition.category.value

                category_stats[category]["count"] += 1

                # 目標達成チェック
                if kpi_value.performance_percentage >= 100:
                    category_stats[category]["meeting_target"] += 1

                # 重大問題チェック
                if kpi_value.performance_percentage < 50:
                    category_stats[category]["critical_issues"] += 1

        summary["categories"] = dict(category_stats)

        # 全体健全性評価
        total_critical = sum(cat["critical_issues"] for cat in category_stats.values())
        total_meeting = sum(cat["meeting_target"] for cat in category_stats.values())

        if total_critical > 0:
            summary["overall_health"] = "critical"
        elif total_meeting / len(self.current_values) > 0.8:
            summary["overall_health"] = "excellent"
        elif total_meeting / len(self.current_values) > 0.6:
            summary["overall_health"] = "good"
        else:
            summary["overall_health"] = "poor"

        return summary

class TrendAnalyzer:
    """トレンド分析エンジン"""

    def __init__(self):
        self.trend_cache: Dict[str, TrendAnalysis] = {}
        self.anomaly_history: List[AnomalyDetection] = []

    def analyze_trends(self, kpi_history: Dict[str, deque]) -> Dict[str, TrendAnalysis]:
        """トレンド分析実行"""
        results = {}

        for kpi_id, history_deque in kpi_history.items():
            if len(history_deque) < 10:
                continue

            history = list(history_deque)
            trend_analysis = self._perform_trend_analysis(kpi_id, history)
            results[kpi_id] = trend_analysis
            self.trend_cache[kpi_id] = trend_analysis

        return results

    def _perform_trend_analysis(self, kpi_id: str, history: List[KPIValue]) -> TrendAnalysis:
        """トレンド分析実行"""
        if not ANALYTICS_AVAILABLE:
            # 簡易分析（scipy/sklearnなし）
            return self._simple_trend_analysis(kpi_id, history)

        try:
            # 数値データと時間データを準備
            values = [h.value for h in history]
            timestamps = [h.timestamp.timestamp() for h in history]

            if len(values) < 2:
                return self._create_default_trend(kpi_id, history)

            # 線形回帰
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(values)

            model = LinearRegression()
            model.fit(X, y)

            # トレンド特性
            slope = model.coef_[0]
            r_squared = model.score(X, y)

            # トレンド方向判定
            if slope > 0.001:
                direction = TrendDirection.IMPROVING
            elif slope < -0.001:
                direction = TrendDirection.DECLINING
            else:
                direction = TrendDirection.STABLE

            # トレンド強度（R²で評価）
            strength = abs(r_squared)

            # 次期間予測
            next_timestamp = timestamps[-1] + (timestamps[-1] - timestamps[0]) / len(values)
            next_value = model.predict([[next_timestamp]])[0]

            # 予測信頼度
            prediction_confidence = r_squared if r_squared > 0 else 0.1

            return TrendAnalysis(
                kpi_id=kpi_id,
                period_start=history[0].timestamp,
                period_end=history[-1].timestamp,
                trend_direction=direction,
                trend_strength=strength,
                slope=slope,
                r_squared=r_squared,
                prediction_next_period=next_value,
                prediction_confidence=prediction_confidence
            )

        except Exception as e:
            logger.error(f"Trend analysis failed for {kpi_id}: {e}")
            return self._create_default_trend(kpi_id, history)

    def _simple_trend_analysis(self, kpi_id: str, history: List[KPIValue]) -> TrendAnalysis:
        """簡易トレンド分析（ライブラリなし）"""
        values = [h.value for h in history]

        if len(values) < 2:
            return self._create_default_trend(kpi_id, history)

        # 前半と後半の平均を比較
        mid_point = len(values) // 2
        first_half_avg = statistics.mean(values[:mid_point])
        second_half_avg = statistics.mean(values[mid_point:])

        change = second_half_avg - first_half_avg
        change_percent = (change / first_half_avg) * 100 if first_half_avg > 0 else 0

        # トレンド方向
        if change_percent > 5:
            direction = TrendDirection.IMPROVING
        elif change_percent < -5:
            direction = TrendDirection.DECLINING
        else:
            direction = TrendDirection.STABLE

        return TrendAnalysis(
            kpi_id=kpi_id,
            period_start=history[0].timestamp,
            period_end=history[-1].timestamp,
            trend_direction=direction,
            trend_strength=abs(change_percent) / 100,
            slope=change,
            r_squared=0.5,  # 推定値
            prediction_next_period=second_half_avg + change,
            prediction_confidence=0.6
        )

    def _create_default_trend(self, kpi_id: str, history: List[KPIValue]) -> TrendAnalysis:
        """デフォルトトレンド分析"""
        current_time = datetime.now()

        return TrendAnalysis(
            kpi_id=kpi_id,
            period_start=current_time,
            period_end=current_time,
            trend_direction=TrendDirection.UNKNOWN,
            trend_strength=0.0,
            slope=0.0,
            r_squared=0.0,
            prediction_next_period=0.0,
            prediction_confidence=0.0
        )

class AnomalyDetector:
    """異常検知エンジン"""

    def __init__(self):
        self.detection_history: List[AnomalyDetection] = []
        self.baseline_cache: Dict[str, Dict[str, float]] = {}

    def detect_anomalies(self, current_kpis: Dict[str, KPIValue]) -> List[AnomalyDetection]:
        """異常検知実行"""
        anomalies = []

        for kpi_id, kpi_value in current_kpis.items():
            anomaly = self._check_for_anomaly(kpi_id, kpi_value)
            if anomaly:
                anomalies.append(anomaly)
                self.detection_history.append(anomaly)

        return anomalies

    def _check_for_anomaly(self, kpi_id: str, kpi_value: KPIValue) -> Optional[AnomalyDetection]:
        """異常チェック"""
        # ベースライン統計を取得または計算
        baseline = self._get_baseline_stats(kpi_id)
        if not baseline:
            return None

        value = kpi_value.value
        mean = baseline['mean']
        std = baseline['std']

        if std == 0:
            return None

        # Z-score計算
        z_score = abs(value - mean) / std

        # 異常判定（3σルール）
        if z_score > 3:
            severity = AlertSeverity.CRITICAL
        elif z_score > 2.5:
            severity = AlertSeverity.HIGH
        elif z_score > 2:
            severity = AlertSeverity.MEDIUM
        else:
            return None

        # 期待範囲計算
        expected_range = (mean - 2 * std, mean + 2 * std)
        anomaly_score = min(z_score / 4, 1.0)  # 0-1に正規化

        return AnomalyDetection(
            detection_id=str(uuid.uuid4()),
            kpi_id=kpi_id,
            timestamp=datetime.now(),
            value=value,
            expected_range=expected_range,
            anomaly_score=anomaly_score,
            severity=severity,
            description=f"Anomaly detected in {kpi_id}: value {value:.2f} is {z_score:.1f} standard deviations from normal"
        )

    def _get_baseline_stats(self, kpi_id: str) -> Optional[Dict[str, float]]:
        """ベースライン統計取得"""
        # 実際の実装では履歴データから計算
        # ここでは模擬データを返す
        baseline_data = {
            "oee": {"mean": 0.80, "std": 0.05},
            "availability": {"mean": 0.90, "std": 0.03},
            "performance": {"mean": 0.92, "std": 0.04},
            "quality_rate": {"mean": 0.95, "std": 0.02},
            "cycle_time": {"mean": 45.0, "std": 5.0},
            "throughput": {"mean": 60.0, "std": 8.0}
        }

        return baseline_data.get(kpi_id)

class InsightGenerator:
    """インサイト生成エンジン"""

    def __init__(self):
        self.insights: List[ProductionInsight] = []
        self.insight_templates = self._initialize_insight_templates()

    def _initialize_insight_templates(self) -> List[Dict[str, Any]]:
        """インサイトテンプレート初期化"""
        return [
            {
                "category": "oee_improvement",
                "title_template": "OEE Improvement Opportunity Detected",
                "description_template": "OEE can be improved by focusing on {bottleneck_factor}. Current OEE is {current_oee}% with potential to reach {potential_oee}%",
                "impact_level": "high",
                "actionable": True,
                "recommendations": [
                    "Analyze and address {bottleneck_factor} issues",
                    "Implement predictive maintenance for critical equipment",
                    "Optimize production scheduling to minimize downtime"
                ]
            },
            {
                "category": "quality_alert",
                "title_template": "Quality Rate Below Target",
                "description_template": "Quality rate of {current_quality}% is below target of {target_quality}%. Main issue: {main_issue}",
                "impact_level": "medium",
                "actionable": True,
                "recommendations": [
                    "Review quality control procedures",
                    "Provide additional training for operators",
                    "Investigate raw material quality"
                ]
            },
            {
                "category": "performance_optimization",
                "title_template": "Performance Optimization Available",
                "description_template": "Equipment performance can be improved from {current_performance}% to {potential_performance}%",
                "impact_level": "medium",
                "actionable": True,
                "recommendations": [
                    "Optimize robot programming and motion paths",
                    "Review tooling and fixture design",
                    "Implement real-time performance monitoring"
                ]
            }
        ]

    def generate_insights(self, kpi_values: Dict[str, KPIValue],
                         trend_analyses: Dict[str, TrendAnalysis],
                         anomalies: List[AnomalyDetection]) -> List[ProductionInsight]:
        """インサイト生成"""
        insights = []

        # OEE改善インサイト
        oee_insight = self._generate_oee_insight(kpi_values, trend_analyses)
        if oee_insight:
            insights.append(oee_insight)

        # 品質インサイト
        quality_insight = self._generate_quality_insight(kpi_values, anomalies)
        if quality_insight:
            insights.append(quality_insight)

        # 性能インサイト
        performance_insight = self._generate_performance_insight(kpi_values, trend_analyses)
        if performance_insight:
            insights.append(performance_insight)

        # 異常関連インサイト
        for anomaly in anomalies:
            if anomaly.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                insight = self._generate_anomaly_insight(anomaly)
                insights.append(insight)

        self.insights.extend(insights)

        # 最新100件を保持
        if len(self.insights) > 100:
            self.insights = self.insights[-100:]

        return insights

    def _generate_oee_insight(self, kpi_values: Dict[str, KPIValue],
                             trend_analyses: Dict[str, TrendAnalysis]) -> Optional[ProductionInsight]:
        """OEE改善インサイト生成"""
        if "oee" not in kpi_values:
            return None

        oee_value = kpi_values["oee"]

        if oee_value.performance_percentage >= 95:
            return None  # OEEが良好なのでインサイト不要

        # ボトルネック因子を特定
        bottleneck = "performance"
        bottleneck_value = 0

        if "availability" in kpi_values:
            avail_value = kpi_values["availability"].performance_percentage
            if avail_value < bottleneck_value:
                bottleneck = "availability"
                bottleneck_value = avail_value

        if "performance" in kpi_values:
            perf_value = kpi_values["performance"].performance_percentage
            if perf_value < bottleneck_value:
                bottleneck = "performance"
                bottleneck_value = perf_value

        if "quality_rate" in kpi_values:
            quality_value = kpi_values["quality_rate"].performance_percentage
            if quality_value < bottleneck_value:
                bottleneck = "quality"
                bottleneck_value = quality_value

        # 潜在OEE計算
        potential_oee = min(95, oee_value.performance_percentage + 10)

        template = self.insight_templates[0]  # OEE改善テンプレート

        return ProductionInsight(
            insight_id=str(uuid.uuid4()),
            category=template["category"],
            title=template["title_template"],
            description=template["description_template"].format(
                bottleneck_factor=bottleneck,
                current_oee=oee_value.value,
                potential_oee=potential_oee
            ),
            impact_level=template["impact_level"],
            actionable=template["actionable"],
            recommendations=[rec.format(bottleneck_factor=bottleneck) for rec in template["recommendations"]],
            data_evidence={
                "current_oee": oee_value.value,
                "target_oee": oee_value.target,
                "bottleneck_factor": bottleneck,
                "trend_direction": oee_value.trend_direction.value
            },
            created_at=datetime.now()
        )

    def _generate_quality_insight(self, kpi_values: Dict[str, KPIValue],
                                anomalies: List[AnomalyDetection]) -> Optional[ProductionInsight]:
        """品質インサイト生成"""
        if "quality_rate" not in kpi_values:
            return None

        quality_value = kpi_values["quality_rate"]

        if quality_value.performance_percentage >= 98:
            return None

        # 品質異常をチェック
        quality_anomalies = [a for a in anomalies if a.kpi_id == "quality_rate"]
        main_issue = "statistical variation"
        if quality_anomalies:
            main_issue = "process instability detected"

        template = self.insight_templates[1]  # 品質アラートテンプレート

        return ProductionInsight(
            insight_id=str(uuid.uuid4()),
            category=template["category"],
            title=template["title_template"],
            description=template["description_template"].format(
                current_quality=quality_value.value,
                target_quality=quality_value.target,
                main_issue=main_issue
            ),
            impact_level=template["impact_level"],
            actionable=template["actionable"],
            recommendations=template["recommendations"],
            data_evidence={
                "current_quality": quality_value.value,
                "target_quality": quality_value.target,
                "trend_direction": quality_value.trend_direction.value,
                "anomalies_count": len(quality_anomalies)
            },
            created_at=datetime.now()
        )

    def _generate_performance_insight(self, kpi_values: Dict[str, KPIValue],
                                    trend_analyses: Dict[str, TrendAnalysis]) -> Optional[ProductionInsight]:
        """性能インサイト生成"""
        if "performance" not in kpi_values:
            return None

        performance_value = kpi_values["performance"]

        if performance_value.performance_percentage >= 95:
            return None

        # トレンドを考慮して潜在性能を計算
        potential_performance = min(98, performance_value.performance_percentage + 8)

        template = self.insight_templates[2]  # 性能最適化テンプレート

        return ProductionInsight(
            insight_id=str(uuid.uuid4()),
            category=template["category"],
            title=template["title_template"],
            description=template["description_template"].format(
                current_performance=performance_value.value,
                potential_performance=potential_performance
            ),
            impact_level=template["impact_level"],
            actionable=template["actionable"],
            recommendations=template["recommendations"],
            data_evidence={
                "current_performance": performance_value.value,
                "target_performance": performance_value.target,
                "trend_direction": performance_value.trend_direction.value
            },
            created_at=datetime.now()
        )

    def _generate_anomaly_insight(self, anomaly: AnomalyDetection) -> ProductionInsight:
        """異常インサイト生成"""
        return ProductionInsight(
            insight_id=str(uuid.uuid4()),
            category="anomaly_alert",
            title=f"Critical Anomaly: {anomaly.kpi_id}",
            description=anomaly.description,
            impact_level="high" if anomaly.severity == AlertSeverity.CRITICAL else "medium",
            actionable=True,
            recommendations=[
                "Immediately investigate the cause of this anomaly",
                "Check equipment status and sensor calibration",
                "Review recent process changes or maintenance activities"
            ],
            data_evidence={
                "anomaly_score": anomaly.anomaly_score,
                "value": anomaly.value,
                "expected_range": anomaly.expected_range,
                "severity": anomaly.severity.value
            },
            created_at=datetime.now()
        )

class AdvancedAnalyticsDashboard:
    """高度な分析ダッシュボード"""

    def __init__(self, production_system: ProductionManagementSystem):
        self.production_system = production_system

        # 分析コンポーネント
        self.kpi_engine = KPIEngine()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.insight_generator = InsightGenerator()

        # 実行制御
        self.running = False
        self.analysis_thread: Optional[threading.Thread] = None

        # 分析結果キャッシュ
        self.current_kpis: Dict[str, KPIValue] = {}
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        self.recent_anomalies: List[AnomalyDetection] = []
        self.recent_insights: List[ProductionInsight] = []

        # 設定
        self.analysis_interval = 60  # 60秒間隔
        self.history_retention_days = 30

        # コールバック
        self.on_analysis_completed: Optional[Callable[[Dict[str, Any]], None]] = None

    def start(self) -> bool:
        """分析開始"""
        try:
            if self.running:
                logger.warning("Advanced analytics dashboard already running")
                return False

            self.running = True

            # 分析スレッド開始
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()

            logger.info("Advanced analytics dashboard started")
            return True

        except Exception as e:
            logger.error(f"Failed to start advanced analytics dashboard: {e}")
            return False

    def stop(self):
        """分析停止"""
        self.running = False

        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5.0)

        logger.info("Advanced analytics dashboard stopped")

    def _analysis_loop(self):
        """分析ループ"""
        logger.info("Advanced analysis loop started")

        while self.running:
            try:
                # データ収集
                self._collect_data()

                # KPI計算
                self.current_kpis = self.kpi_engine.calculate_all_kpis()

                # トレンド分析
                self.trend_analyses = self.trend_analyzer.analyze_trends(self.kpi_engine.kpi_history)

                # 異常検知
                self.recent_anomalies = self.anomaly_detector.detect_anomalies(self.current_kpis)

                # インサイト生成
                self.recent_insights = self.insight_generator.generate_insights(
                    self.current_kpis, self.trend_analyses, self.recent_anomalies
                )

                # 分析完了コールバック
                if self.on_analysis_completed:
                    analysis_results = {
                        "timestamp": datetime.now().isoformat(),
                        "kpis": {k: asdict(v) for k, v in self.current_kpis.items()},
                        "trends": {k: asdict(v) for k, v in self.trend_analyses.items()},
                        "anomalies": [asdict(a) for a in self.recent_anomalies],
                        "insights": [asdict(i) for i in self.recent_insights]
                    }
                    self.on_analysis_completed(analysis_results)

                time.sleep(self.analysis_interval)

            except Exception as e:
                logger.error(f"Advanced analysis error: {e}")
                time.sleep(30.0)  # エラー時は短い間隔で再試行

        logger.info("Advanced analysis loop ended")

    def _collect_data(self):
        """データ収集"""
        # 生産システムからデータ収集
        dashboard_data = self.production_system.get_production_dashboard()

        # 模擬メトリクスデータ生成
        current_time = datetime.now()
        for robot_id in ['robot_001', 'robot_002', 'robot_003']:
            metrics = ProductionMetrics(
                timestamp=current_time,
                robot_id=robot_id,
                production_order_id="PO_CURRENT",
                work_order_id=f"WO_{robot_id}",
                cycle_time=45.0 + (hash(robot_id) % 20) - 10,
                throughput=60.0 + (hash(robot_id) % 10) - 5,
                quality_score=0.95 - (hash(robot_id) % 8) * 0.01,
                oee=0.80 + (hash(robot_id) % 15) * 0.01,
                availability=0.90 + (hash(robot_id) % 10) * 0.01,
                performance=0.92 + (hash(robot_id) % 8) * 0.01,
                quality_rate=0.95 - (hash(robot_id) % 6) * 0.01,
                energy_consumption=5.5 + (hash(robot_id) % 3) - 1,
                maintenance_alerts=0
            )

            self.kpi_engine.add_metrics_data(metrics)

    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """分析ダッシュボードデータ取得"""
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": self.kpi_engine.get_kpi_summary(),
            "kpis": {k: asdict(v) for k, v in self.current_kpis.items()},
            "trends": {k: asdict(v) for k, v in self.trend_analyses.items()},
            "anomalies": [asdict(a) for a in self.recent_anomalies[-10:]],  # 最新10件
            "insights": [asdict(i) for i in self.recent_insights[-20:]],  # 最新20件
            "statistics": {
                "total_kpis": len(self.kpi_engine.kpi_definitions),
                "active_anomalies": len(self.recent_anomalies),
                "unread_insights": len(self.recent_insights),
                "analysis_running": self.running,
                "last_analysis": datetime.now().isoformat()
            }
        }

    def get_kpi_trends(self, kpi_id: str, period_days: int = 7) -> Dict[str, Any]:
        """KPIトレンド詳細取得"""
        if kpi_id not in self.kpi_engine.kpi_history:
            return {"error": "KPI not found"}

        history = list(self.kpi_engine.kpi_history[kpi_id])

        # 期間でフィルタリング
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_history = [h for h in history if h.timestamp >= cutoff_date]

        if not recent_history:
            return {"error": "No data available for specified period"}

        trend_analysis = self.trend_analyses.get(kpi_id)

        return {
            "kpi_id": kpi_id,
            "period_days": period_days,
            "data_points": len(recent_history),
            "current_value": recent_history[-1].value if recent_history else None,
            "period_start": recent_history[0].timestamp.isoformat() if recent_history else None,
            "period_end": recent_history[-1].timestamp.isoformat() if recent_history else None,
            "trend_analysis": asdict(trend_analysis) if trend_analysis else None,
            "values": [{"timestamp": h.timestamp.isoformat(), "value": h.value} for h in recent_history]
        }

# グローバルインスタンス
analytics_dashboard: Optional[AdvancedAnalyticsDashboard] = None

def initialize_analytics_dashboard(production_system: ProductionManagementSystem) -> AdvancedAnalyticsDashboard:
    """高度な分析ダッシュボード初期化"""
    global analytics_dashboard
    analytics_dashboard = AdvancedAnalyticsDashboard(production_system)
    return analytics_dashboard

def get_analytics_dashboard() -> Optional[AdvancedAnalyticsDashboard]:
    """高度な分析ダッシュボード取得"""
    return analytics_dashboard

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Advanced Analytics Dashboard...")

    try:
        # モック生産管理システム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())

        # 高度な分析ダッシュボード初期化
        dashboard = initialize_analytics_dashboard(mock_pms)

        if dashboard.start():
            print("Advanced analytics dashboard started successfully!")

            # テスト分析実行
            time.sleep(5)

            # 分析結果取得
            results = dashboard.get_analytics_dashboard()
            print(f"Analysis results:")
            print(f"  Active KPIs: {len(results['kpis'])}")
            print(f"  Recent anomalies: {len(results['anomalies'])}")
            print(f"  Generated insights: {len(results['insights'])}")
            print(f"  Overall health: {results['summary']['overall_health']}")

            # OEEトレンド詳細取得
            oee_trends = dashboard.get_kpi_trends("oee", period_days=7)
            if "error" not in oee_trends:
                print(f"OEE trends: {oee_trends['trend_analysis']['trend_direction']}")

            time.sleep(2)
            dashboard.stop()

        else:
            print("Failed to start advanced analytics dashboard")

    except Exception as e:
        print(f"Advanced analytics dashboard test failed: {e}")