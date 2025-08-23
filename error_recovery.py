import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ErrorType(Enum):
    CONNECTION_TIMEOUT = "connection_timeout"
    API_TIMEOUT = "api_timeout"
    NETWORK_ERROR = "network_error"
    PROTOCOL_ERROR = "protocol_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorEvent:
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    retry_count: int = 0
    context: Dict[str, Any] = None
    resolved: bool = False

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker reset to CLOSED state")
        self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class ErrorRecoveryManager:
    def __init__(self):
        self.error_history: List[ErrorEvent] = []
        self.error_handlers: Dict[ErrorType, Callable] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[ErrorType, Dict[str, Any]] = {}
        self._setup_default_strategies()
        self._error_lock = threading.Lock()
    
    def _setup_default_strategies(self):
        self.recovery_strategies = {
            ErrorType.CONNECTION_TIMEOUT: {
                "max_retries": 5,
                "base_delay": 2.0,
                "backoff_multiplier": 2.0,
                "max_delay": 30.0,
                "jitter": True
            },
            ErrorType.API_TIMEOUT: {
                "max_retries": 3,
                "base_delay": 1.0,
                "backoff_multiplier": 1.5,
                "max_delay": 10.0,
                "jitter": True
            },
            ErrorType.NETWORK_ERROR: {
                "max_retries": 10,
                "base_delay": 1.0,
                "backoff_multiplier": 2.0,
                "max_delay": 60.0,
                "jitter": True
            },
            ErrorType.RATE_LIMIT_ERROR: {
                "max_retries": 5,
                "base_delay": 30.0,
                "backoff_multiplier": 1.2,
                "max_delay": 300.0,
                "jitter": False
            },
            ErrorType.SERVICE_UNAVAILABLE: {
                "max_retries": 8,
                "base_delay": 5.0,
                "backoff_multiplier": 1.8,
                "max_delay": 120.0,
                "jitter": True
            }
        }
    
    def register_error_handler(self, error_type: ErrorType, handler: Callable):
        self.error_handlers[error_type] = handler
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def record_error(self, error_type: ErrorType, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        with self._error_lock:
            error_event = ErrorEvent(
                error_type=error_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            self.error_history.append(error_event)
            logger.error(f"Error recorded: {error_type.value} - {message}")
            
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
            
            if error_type in self.error_handlers:
                try:
                    self.error_handlers[error_type](error_event)
                except Exception as e:
                    logger.error(f"Error handler failed: {e}")
    
    def should_retry(self, error_type: ErrorType, current_retry_count: int) -> bool:
        strategy = self.recovery_strategies.get(error_type)
        if not strategy:
            return current_retry_count < 3
        
        return current_retry_count < strategy["max_retries"]
    
    def calculate_retry_delay(self, error_type: ErrorType, retry_count: int) -> float:
        strategy = self.recovery_strategies.get(error_type)
        if not strategy:
            return 1.0
        
        base_delay = strategy["base_delay"]
        multiplier = strategy["backoff_multiplier"]
        max_delay = strategy["max_delay"]
        use_jitter = strategy["jitter"]
        
        delay = base_delay * (multiplier ** retry_count)
        delay = min(delay, max_delay)
        
        if use_jitter:
            jitter_factor = 0.1 + (0.2 * (retry_count / 10))
            jitter = delay * jitter_factor * (2 * (time.time() % 1) - 1)
            delay += jitter
        
        return max(delay, 0.1)
    
    def get_recent_errors(self, minutes: int = 10) -> List[ErrorEvent]:
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [error for error in self.error_history if error.timestamp >= cutoff_time]
    
    def get_error_statistics(self, hours: int = 1) -> Dict[str, Any]:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [error for error in self.error_history if error.timestamp >= cutoff_time]
        
        error_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_type = error.error_type.value
            severity = error.severity.value
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": error_counts,
            "severity_distribution": severity_counts,
            "time_range_hours": hours
        }
    
    def is_service_healthy(self, service_name: str) -> bool:
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
            return False
        
        recent_errors = self.get_recent_errors(5)
        critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
        
        return len(critical_errors) == 0

class HealthChecker:
    def __init__(self, error_manager: ErrorRecoveryManager):
        self.error_manager = error_manager
        self.health_checks: Dict[str, Callable] = {}
        self.check_intervals: Dict[str, int] = {}
        self.is_running = False
        self.check_thread = None
    
    def register_health_check(self, service_name: str, check_func: Callable, interval_seconds: int = 30):
        self.health_checks[service_name] = check_func
        self.check_intervals[service_name] = interval_seconds
    
    def start_monitoring(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.check_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        self.is_running = False
        if self.check_thread:
            self.check_thread.join()
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        last_check_times = {service: 0 for service in self.health_checks}
        
        while self.is_running:
            current_time = time.time()
            
            for service_name, check_func in self.health_checks.items():
                interval = self.check_intervals[service_name]
                last_check = last_check_times[service_name]
                
                if current_time - last_check >= interval:
                    try:
                        if not check_func():
                            self.error_manager.record_error(
                                ErrorType.SERVICE_UNAVAILABLE,
                                f"Health check failed for {service_name}",
                                ErrorSeverity.HIGH,
                                {"service": service_name}
                            )
                    except Exception as e:
                        self.error_manager.record_error(
                            ErrorType.UNKNOWN_ERROR,
                            f"Health check error for {service_name}: {e}",
                            ErrorSeverity.MEDIUM,
                            {"service": service_name, "error": str(e)}
                        )
                    
                    last_check_times[service_name] = current_time
            
            time.sleep(1)

error_recovery_manager = ErrorRecoveryManager()

def with_retry_and_circuit_breaker(service_name: str, error_type: ErrorType = ErrorType.UNKNOWN_ERROR):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            circuit_breaker = error_recovery_manager.get_circuit_breaker(service_name)
            retry_count = 0
            
            while True:
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                    
                except Exception as e:
                    error_recovery_manager.record_error(
                        error_type,
                        f"{service_name} error: {str(e)}",
                        ErrorSeverity.MEDIUM,
                        {"service": service_name, "function": func.__name__}
                    )
                    
                    if not error_recovery_manager.should_retry(error_type, retry_count):
                        raise e
                    
                    delay = error_recovery_manager.calculate_retry_delay(error_type, retry_count)
                    logger.info(f"Retrying {service_name} after {delay:.1f}s (attempt {retry_count + 1})")
                    time.sleep(delay)
                    retry_count += 1
        
        return wrapper
    return decorator