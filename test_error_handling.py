#!/usr/bin/env python3
import time
import threading
import logging
from tcp_communication import TCPClient, TCPServer, MessageBuilder
from error_recovery import error_recovery_manager, ErrorType, ErrorSeverity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorHandlingTestSuite:
    def __init__(self):
        self.test_results = {}
        self.server = None
        self.client = None
    
    def setup_test_server(self, port=9999):
        """テスト用のTCPサーバーを起動"""
        def test_handler(message_data):
            command = message_data.get("parameters", {}).get("command", "")
            
            if command == "timeout_test":
                time.sleep(15)
                return MessageBuilder.create_response(message_data, "success", {"message": "timeout test"})
            elif command == "error_test":
                raise Exception("Simulated server error")
            else:
                return MessageBuilder.create_response(message_data, "success", {"message": "test response"})
        
        self.server = TCPServer(port=port)
        self.server.register_handler("test_command", test_handler)
        
        server_thread = threading.Thread(target=self.server.start_server, daemon=True)
        server_thread.start()
        
        time.sleep(1)
        logger.info(f"Test server started on port {port}")
    
    def test_connection_retry(self):
        """接続再試行のテスト"""
        logger.info("Testing connection retry logic...")
        
        client = TCPClient(host="localhost", port=9998)
        start_time = time.time()
        
        success = client.connect()
        end_time = time.time()
        
        self.test_results["connection_retry"] = {
            "success": not success,
            "duration": end_time - start_time,
            "expected_behavior": "Should fail after retries with exponential backoff"
        }
        
        client.disconnect()
        logger.info(f"Connection retry test completed in {end_time - start_time:.2f}s")
    
    def test_timeout_handling(self):
        """タイムアウトハンドリングのテスト"""
        logger.info("Testing timeout handling...")
        
        self.setup_test_server(9999)
        client = TCPClient(host="localhost", port=9999)
        client.socket_timeout = 5.0
        
        if client.connect():
            message = MessageBuilder.create_message(
                "test_command", 
                "test_target",
                {"command": "timeout_test"}
            )
            
            start_time = time.time()
            result = client.send_message(message)
            end_time = time.time()
            
            self.test_results["timeout_handling"] = {
                "message_sent": result,
                "duration": end_time - start_time,
                "expected_behavior": "Should handle timeout gracefully"
            }
            
            client.disconnect()
        
        self.server.stop_server()
        logger.info("Timeout handling test completed")
    
    def test_error_recovery(self):
        """エラーリカバリのテスト"""
        logger.info("Testing error recovery...")
        
        error_recovery_manager.record_error(
            ErrorType.API_TIMEOUT,
            "Test API timeout",
            ErrorSeverity.MEDIUM,
            {"test": True}
        )
        
        error_recovery_manager.record_error(
            ErrorType.CONNECTION_TIMEOUT,
            "Test connection timeout", 
            ErrorSeverity.HIGH,
            {"test": True}
        )
        
        stats = error_recovery_manager.get_error_statistics(1)
        recent_errors = error_recovery_manager.get_recent_errors(5)
        
        self.test_results["error_recovery"] = {
            "total_errors": stats["total_errors"],
            "recent_errors_count": len(recent_errors),
            "error_types": stats["error_types"],
            "expected_behavior": "Should track and categorize errors properly"
        }
        
        logger.info("Error recovery test completed")
    
    def test_circuit_breaker(self):
        """サーキットブレーカーのテスト"""
        logger.info("Testing circuit breaker...")
        
        circuit_breaker = error_recovery_manager.get_circuit_breaker("test_service")
        
        def failing_function():
            raise Exception("Test failure")
        
        failure_count = 0
        for i in range(7):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                failure_count += 1
        
        is_service_healthy = error_recovery_manager.is_service_healthy("test_service")
        
        self.test_results["circuit_breaker"] = {
            "failure_count": failure_count,
            "circuit_breaker_state": circuit_breaker.state.value,
            "service_healthy": is_service_healthy,
            "expected_behavior": "Should open circuit after threshold failures"
        }
        
        logger.info("Circuit breaker test completed")
    
    def test_message_retry(self):
        """メッセージ送信再試行のテスト"""
        logger.info("Testing message retry logic...")
        
        self.setup_test_server(9997)
        client = TCPClient(host="localhost", port=9997)
        
        if client.connect():
            message = MessageBuilder.create_message(
                "test_command",
                "test_target", 
                {"command": "normal_test"}
            )
            
            start_time = time.time()
            success_count = 0
            
            for i in range(5):
                if client.send_message(message):
                    success_count += 1
                time.sleep(0.1)
            
            end_time = time.time()
            
            self.test_results["message_retry"] = {
                "success_count": success_count,
                "total_attempts": 5,
                "duration": end_time - start_time,
                "expected_behavior": "Should successfully send messages with retry"
            }
            
            client.disconnect()
        
        self.server.stop_server()
        logger.info("Message retry test completed")
    
    def run_all_tests(self):
        """全てのテストを実行"""
        logger.info("Starting comprehensive error handling tests...")
        
        tests = [
            self.test_connection_retry,
            self.test_error_recovery,
            self.test_circuit_breaker,
            self.test_message_retry,
            self.test_timeout_handling
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                self.test_results[test.__name__] = {
                    "error": str(e),
                    "status": "failed"
                }
    
    def print_results(self):
        """テスト結果を出力"""
        logger.info("\n" + "="*60)
        logger.info("ERROR HANDLING TEST RESULTS")
        logger.info("="*60)
        
        for test_name, results in self.test_results.items():
            logger.info(f"\nTest: {test_name}")
            logger.info("-" * 40)
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
        
        error_stats = error_recovery_manager.get_error_statistics(1)
        logger.info(f"\nError Statistics (last hour):")
        logger.info(f"  Total errors: {error_stats['total_errors']}")
        logger.info(f"  Error types: {error_stats['error_types']}")
        logger.info(f"  Severity distribution: {error_stats['severity_distribution']}")

def main():
    logger.info("Initializing error handling test suite...")
    
    test_suite = ErrorHandlingTestSuite()
    test_suite.run_all_tests()
    test_suite.print_results()
    
    logger.info("Error handling tests completed!")

if __name__ == "__main__":
    main()