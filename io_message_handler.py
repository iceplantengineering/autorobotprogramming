import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
from error_recovery import (
    error_recovery_manager,
    ErrorType,
    ErrorSeverity,
    with_retry_and_circuit_breaker
)

logger = logging.getLogger(__name__)

class IOType(Enum):
    DIGITAL_INPUT = "digital_input"
    DIGITAL_OUTPUT = "digital_output"
    ANALOG_INPUT = "analog_input"
    ANALOG_OUTPUT = "analog_output"

class SignalState(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class IOSignal:
    name: str
    io_type: IOType
    address: str
    value: Any = None
    state: SignalState = SignalState.UNKNOWN
    timestamp: datetime = None
    description: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class IOController:
    """基本I/O制御クラス"""
    
    def __init__(self):
        self.signals: Dict[str, IOSignal] = {}
        self.signal_history: Dict[str, List[Dict[str, Any]]] = {}
        self.signal_callbacks: Dict[str, List[Callable]] = {}
        self.monitoring_thread = None
        self.is_monitoring = False
        self._lock = threading.Lock()
        
        # 標準的な産業用ロボットI/Oシグナルの定義
        self._initialize_standard_signals()
    
    def _initialize_standard_signals(self):
        """標準I/Oシグナル初期化"""
        standard_inputs = [
            ("START_BUTTON", "I0.0", "サイクル開始ボタン"),
            ("E_STOP", "I0.1", "非常停止"),
            ("PAUSE_BUTTON", "I0.2", "一時停止ボタン"),
            ("RESUME_BUTTON", "I0.3", "再開ボタン"),
            ("MODE_SELECT", "I0.4", "モード選択（AUTO/MANUAL）"),
            ("PART_PRESENT", "I0.5", "ワーク検出センサー"),
            ("JIG_CLAMPED", "I0.6", "治具クランプ確認"),
            ("DOOR_CLOSED", "I0.7", "安全扉閉状態"),
            ("AIR_PRESSURE_OK", "I1.0", "エア圧力正常"),
            ("WELD_ENABLE", "I1.1", "溶接許可"),
            ("WELDER_READY", "I1.2", "溶接機準備完了"),
            ("SUPPLY_PART_OK", "I1.3", "供給部品正常"),
            ("DISCHARGE_READY", "I1.4", "排出準備完了"),
            ("GRIPPER_OPEN_CONFIRM", "I1.5", "グリッパー開確認"),
            ("GRIPPER_CLOSE_CONFIRM", "I1.6", "グリッパー閉確認")
        ]
        
        standard_outputs = [
            ("WORK_COMPLETE", "Q0.0", "作業完了"),
            ("ERROR_OCCURRED", "Q0.1", "エラー発生"),
            ("READY_LAMP", "Q0.2", "準備完了ランプ"),
            ("WORKING_LAMP", "Q0.3", "作業中ランプ"),
            ("STEP_NUMBER", "Q0.4", "ステップ番号表示"),
            ("ERROR_CODE", "Q0.5", "エラーコード表示"),
            ("WELD_EXECUTE", "Q0.6", "溶接実行"),
            ("WELD_COMPLETE", "Q0.7", "溶接完了"),
            ("PICK_COMPLETE", "Q1.0", "ピック完了"),
            ("PLACE_COMPLETE", "Q1.1", "プレース完了"),
            ("GRIPPER_OPEN", "Q1.2", "グリッパー開"),
            ("GRIPPER_CLOSE", "Q1.3", "グリッパー閉"),
            ("CONVEYOR_FORWARD", "Q1.4", "コンベア正転"),
            ("CONVEYOR_REVERSE", "Q1.5", "コンベア逆転"),
            ("ALARM_HORN", "Q1.6", "警報ブザー")
        ]
        
        # デジタル入力信号登録
        for name, address, description in standard_inputs:
            signal = IOSignal(
                name=name,
                io_type=IOType.DIGITAL_INPUT,
                address=address,
                value=False,
                description=description
            )
            self.add_signal(signal)
        
        # デジタル出力信号登録
        for name, address, description in standard_outputs:
            signal = IOSignal(
                name=name,
                io_type=IOType.DIGITAL_OUTPUT,
                address=address,
                value=False,
                description=description
            )
            self.add_signal(signal)
        
        logger.info(f"Initialized {len(standard_inputs)} input and {len(standard_outputs)} output signals")
    
    def add_signal(self, signal: IOSignal):
        """信号追加"""
        with self._lock:
            self.signals[signal.name] = signal
            self.signal_history[signal.name] = []
            logger.debug(f"Added signal: {signal.name} ({signal.address})")
    
    def set_signal_value(self, signal_name: str, value: Any) -> bool:
        """信号値設定"""
        try:
            with self._lock:
                if signal_name not in self.signals:
                    logger.warning(f"Unknown signal: {signal_name}")
                    return False
                
                signal = self.signals[signal_name]
                old_value = signal.value
                signal.value = value
                signal.timestamp = datetime.now()
                signal.state = SignalState.ACTIVE if value else SignalState.INACTIVE
                
                # 履歴記録
                self.signal_history[signal_name].append({
                    "timestamp": signal.timestamp.isoformat(),
                    "old_value": old_value,
                    "new_value": value,
                    "state": signal.state.value
                })
                
                # 履歴を最新100件に制限
                if len(self.signal_history[signal_name]) > 100:
                    self.signal_history[signal_name] = self.signal_history[signal_name][-100:]
                
                # 値変化時のコールバック実行
                if old_value != value:
                    self._trigger_callbacks(signal_name, signal)
                
                logger.debug(f"Signal {signal_name} set to {value}")
                return True
                
        except Exception as e:
            error_recovery_manager.record_error(
                ErrorType.UNKNOWN_ERROR,
                f"Signal set error: {signal_name} = {value}, {str(e)}",
                ErrorSeverity.MEDIUM,
                {"signal_name": signal_name, "value": value}
            )
            return False
    
    def get_signal_value(self, signal_name: str) -> Any:
        """信号値取得"""
        with self._lock:
            if signal_name in self.signals:
                return self.signals[signal_name].value
            return None
    
    def get_signal_info(self, signal_name: str) -> Optional[Dict[str, Any]]:
        """信号情報取得"""
        with self._lock:
            if signal_name in self.signals:
                signal = self.signals[signal_name]
                return {
                    "name": signal.name,
                    "type": signal.io_type.value,
                    "address": signal.address,
                    "value": signal.value,
                    "state": signal.state.value,
                    "timestamp": signal.timestamp.isoformat(),
                    "description": signal.description
                }
            return None
    
    def register_callback(self, signal_name: str, callback: Callable[[IOSignal], None]):
        """信号変化コールバック登録"""
        if signal_name not in self.signal_callbacks:
            self.signal_callbacks[signal_name] = []
        self.signal_callbacks[signal_name].append(callback)
    
    def _trigger_callbacks(self, signal_name: str, signal: IOSignal):
        """コールバック実行"""
        if signal_name in self.signal_callbacks:
            for callback in self.signal_callbacks[signal_name]:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Callback error for signal {signal_name}: {e}")
    
    def get_all_signals_status(self) -> Dict[str, Any]:
        """全信号状態取得"""
        with self._lock:
            status = {}
            for name, signal in self.signals.items():
                status[name] = {
                    "value": signal.value,
                    "state": signal.state.value,
                    "timestamp": signal.timestamp.isoformat(),
                    "type": signal.io_type.value,
                    "address": signal.address
                }
            return status
    
    def start_monitoring(self, interval: float = 0.1):
        """監視開始"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("I/O monitoring started")
    
    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("I/O monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """監視ループ"""
        while self.is_monitoring:
            try:
                # 実際のVC環境では、ここでPLCやI/Oデバイスから信号を読み取る
                self._simulate_io_updates()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)
    
    def _simulate_io_updates(self):
        """I/O更新シミュレーション（実際の実装ではPLCから読み取り）"""
        # 実際の実装では vcScript でI/Oを読み取る
        pass

class MessageProcessor:
    """メッセージ処理クラス"""
    
    def __init__(self, io_controller: IOController):
        self.io_controller = io_controller
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = []
        self.processing_thread = None
        self.is_processing = False
        self._queue_lock = threading.Lock()
        
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """デフォルトメッセージハンドラー設定"""
        self.message_handlers.update({
            "io_read": self._handle_io_read,
            "io_write": self._handle_io_write,
            "io_status": self._handle_io_status,
            "io_monitor_start": self._handle_monitor_start,
            "io_monitor_stop": self._handle_monitor_stop,
            "io_batch_update": self._handle_batch_update,
            "io_signal_info": self._handle_signal_info
        })
    
    def register_handler(self, message_type: str, handler: Callable):
        """メッセージハンドラー登録"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    def add_message(self, message: Dict[str, Any]):
        """メッセージ追加"""
        with self._queue_lock:
            self.message_queue.append(message)
    
    def start_processing(self):
        """メッセージ処理開始"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Message processing started")
    
    def stop_processing(self):
        """メッセージ処理停止"""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("Message processing stopped")
    
    def _processing_loop(self):
        """メッセージ処理ループ"""
        while self.is_processing:
            try:
                message = self._get_next_message()
                if message:
                    self._process_message(message)
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                time.sleep(0.1)
    
    def _get_next_message(self) -> Optional[Dict[str, Any]]:
        """次のメッセージ取得"""
        with self._queue_lock:
            if self.message_queue:
                return self.message_queue.pop(0)
            return None
    
    def _process_message(self, message: Dict[str, Any]):
        """メッセージ処理"""
        try:
            command_type = message.get("command_type")
            if command_type in self.message_handlers:
                response = self.message_handlers[command_type](message)
                return response
            else:
                logger.warning(f"Unknown message type: {command_type}")
                return self._create_error_response(message, f"Unknown command: {command_type}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return self._create_error_response(message, str(e))
    
    def _handle_io_read(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """I/O読み取り処理"""
        try:
            parameters = message.get("parameters", {})
            signal_name = parameters.get("signal_name")
            
            if signal_name:
                value = self.io_controller.get_signal_value(signal_name)
                signal_info = self.io_controller.get_signal_info(signal_name)
                
                return self._create_success_response(message, {
                    "signal_name": signal_name,
                    "value": value,
                    "signal_info": signal_info
                })
            else:
                all_status = self.io_controller.get_all_signals_status()
                return self._create_success_response(message, {"all_signals": all_status})
                
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _handle_io_write(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """I/O書き込み処理"""
        try:
            parameters = message.get("parameters", {})
            signal_name = parameters.get("signal_name")
            value = parameters.get("value")
            
            if signal_name is None:
                return self._create_error_response(message, "signal_name is required")
            
            if value is None:
                return self._create_error_response(message, "value is required")
            
            success = self.io_controller.set_signal_value(signal_name, value)
            
            if success:
                return self._create_success_response(message, {
                    "signal_name": signal_name,
                    "value": value,
                    "message": "Signal updated successfully"
                })
            else:
                return self._create_error_response(message, "Failed to update signal")
                
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _handle_io_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """I/Oステータス処理"""
        try:
            all_status = self.io_controller.get_all_signals_status()
            return self._create_success_response(message, {
                "io_status": all_status,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _handle_monitor_start(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """監視開始処理"""
        try:
            parameters = message.get("parameters", {})
            interval = parameters.get("interval", 0.1)
            
            self.io_controller.start_monitoring(interval)
            return self._create_success_response(message, {
                "message": f"I/O monitoring started with interval {interval}s"
            })
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _handle_monitor_stop(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """監視停止処理"""
        try:
            self.io_controller.stop_monitoring()
            return self._create_success_response(message, {
                "message": "I/O monitoring stopped"
            })
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _handle_batch_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """バッチ更新処理"""
        try:
            parameters = message.get("parameters", {})
            updates = parameters.get("updates", [])
            results = []
            
            for update in updates:
                signal_name = update.get("signal_name")
                value = update.get("value")
                
                if signal_name and value is not None:
                    success = self.io_controller.set_signal_value(signal_name, value)
                    results.append({
                        "signal_name": signal_name,
                        "value": value,
                        "success": success
                    })
            
            return self._create_success_response(message, {
                "batch_results": results,
                "total_updates": len(results)
            })
            
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _handle_signal_info(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """信号情報処理"""
        try:
            parameters = message.get("parameters", {})
            signal_name = parameters.get("signal_name")
            
            if signal_name:
                signal_info = self.io_controller.get_signal_info(signal_name)
                if signal_info:
                    return self._create_success_response(message, signal_info)
                else:
                    return self._create_error_response(message, f"Signal not found: {signal_name}")
            else:
                # 全信号情報取得
                all_signals = {}
                for name, signal in self.io_controller.signals.items():
                    all_signals[name] = self.io_controller.get_signal_info(name)
                
                return self._create_success_response(message, {"all_signals_info": all_signals})
                
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    def _create_success_response(self, original_message: Dict[str, Any], 
                               data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """成功レスポンス作成"""
        return {
            "message_id": f"response_{original_message.get('message_id')}",
            "timestamp": datetime.now().isoformat(),
            "response_to": original_message.get("message_id"),
            "status": "success",
            "data": data or {}
        }
    
    def _create_error_response(self, original_message: Dict[str, Any], 
                              error_message: str) -> Dict[str, Any]:
        """エラーレスポンス作成"""
        return {
            "message_id": f"response_{original_message.get('message_id')}",
            "timestamp": datetime.now().isoformat(),
            "response_to": original_message.get("message_id"),
            "status": "error",
            "error": error_message
        }

# グローバルインスタンス
io_controller = IOController()
message_processor = MessageProcessor(io_controller)

def initialize_io_system() -> Dict[str, Any]:
    """I/Oシステム初期化"""
    try:
        io_controller.start_monitoring()
        message_processor.start_processing()
        
        logger.info("I/O system initialized successfully")
        return {
            "status": "success",
            "message": "I/O system initialized",
            "signal_count": len(io_controller.signals)
        }
    except Exception as e:
        logger.error(f"I/O system initialization failed: {e}")
        return {
            "status": "error", 
            "error": str(e)
        }

def shutdown_io_system():
    """I/Oシステム終了"""
    try:
        io_controller.stop_monitoring()
        message_processor.stop_processing()
        logger.info("I/O system shutdown completed")
    except Exception as e:
        logger.error(f"I/O system shutdown error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    result = initialize_io_system()
    logger.info(f"Initialization result: {result}")
    
    try:
        # テスト信号設定
        io_controller.set_signal_value("START_BUTTON", True)
        io_controller.set_signal_value("PART_PRESENT", True)
        io_controller.set_signal_value("DOOR_CLOSED", True)
        
        # ステータス確認
        status = io_controller.get_all_signals_status()
        logger.info(f"Current status sample: START_BUTTON={status.get('START_BUTTON', {}).get('value')}")
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        logger.info("Stopping I/O system...")
    finally:
        shutdown_io_system()