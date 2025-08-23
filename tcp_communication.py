import socket
import json
import threading
import time
import random
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import logging
from error_recovery import (
    error_recovery_manager, 
    with_retry_and_circuit_breaker,
    ErrorType,
    ErrorSeverity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCPServer:
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.is_running = False
        self.message_handlers: Dict[str, Callable] = {}
        self.connection_status = "DISCONNECTED"
        
    def register_handler(self, command_type: str, handler: Callable):
        self.message_handlers[command_type] = handler
        
    def start_server(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.is_running = True
            
            logger.info(f"TCP Server started on {self.host}:{self.port}")
            
            while self.is_running:
                try:
                    self.client_socket, client_address = self.socket.accept()
                    self.connection_status = "CONNECTED"
                    logger.info(f"Client connected from {client_address}")
                    
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        args=(self.client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.is_running:
                        logger.error(f"Socket error: {e}")
                        
        except Exception as e:
            logger.error(f"Server startup error: {e}")
            
    def _handle_client(self, client_socket):
        buffer = ""
        try:
            while self.is_running:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_message(line.strip(), client_socket)
                        
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()
            self.connection_status = "DISCONNECTED"
            logger.info("Client disconnected")
            
    def _process_message(self, message: str, client_socket):
        try:
            msg_data = json.loads(message)
            command_type = msg_data.get("command_type")
            
            if command_type in self.message_handlers:
                response = self.message_handlers[command_type](msg_data)
                if msg_data.get("response_required", False):
                    self.send_response(response, client_socket)
            else:
                logger.warning(f"Unknown command type: {command_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            
    def send_response(self, response_data: Dict[str, Any], client_socket):
        try:
            response_json = json.dumps(response_data) + '\n'
            client_socket.send(response_json.encode('utf-8'))
        except Exception as e:
            logger.error(f"Response send error: {e}")
            
    def send_message(self, message_data: Dict[str, Any]) -> bool:
        if self.client_socket and self.connection_status == "CONNECTED":
            try:
                message_json = json.dumps(message_data) + '\n'
                self.client_socket.send(message_json.encode('utf-8'))
                return True
            except Exception as e:
                logger.error(f"Message send error: {e}")
                return False
        return False
        
    def stop_server(self):
        self.is_running = False
        if self.socket:
            self.socket.close()

class TCPClient:
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        self.message_handlers: Dict[str, Callable] = {}
        self.auto_reconnect = True
        self.connect_timeout = 10.0
        self.socket_timeout = 30.0
        self.max_retries = 10
        self.base_retry_delay = 1.0
        self.max_retry_delay = 60.0
        self.retry_count = 0
        
    def register_handler(self, command_type: str, handler: Callable):
        self.message_handlers[command_type] = handler
        
    def connect(self) -> bool:
        return self._connect_with_retry()
    
    def _connect_with_retry(self) -> bool:
        for attempt in range(self.max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.connect_timeout)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(self.socket_timeout)
                
                self.is_connected = True
                self.retry_count = 0
                logger.info(f"Connected to server {self.host}:{self.port} (attempt {attempt + 1})")
                
                receive_thread = threading.Thread(target=self._receive_messages)
                receive_thread.daemon = True
                receive_thread.start()
                
                return True
                
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                self.is_connected = False
                retry_delay = self._calculate_retry_delay(attempt)
                
                error_type = ErrorType.CONNECTION_TIMEOUT if isinstance(e, socket.timeout) else ErrorType.NETWORK_ERROR
                error_recovery_manager.record_error(
                    error_type,
                    f"Connection failed: {str(e)}",
                    ErrorSeverity.MEDIUM,
                    {"host": self.host, "port": self.port, "attempt": attempt + 1}
                )
                
                logger.warning(f"Connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect after {self.max_retries} attempts")
                    
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}")
                self.is_connected = False
                return False
                
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        delay = self.base_retry_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3) * delay
        return min(delay + jitter, self.max_retry_delay)
            
    def _receive_messages(self):
        buffer = ""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while self.is_connected:
                try:
                    self.socket.settimeout(self.socket_timeout)
                    data = self.socket.recv(4096).decode('utf-8')
                    
                    if not data:
                        logger.warning("Server closed connection")
                        break
                        
                    consecutive_errors = 0
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            self._process_message(line.strip())
                            
                except socket.timeout:
                    logger.debug("Socket timeout during receive")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive timeout errors ({consecutive_errors})")
                        break
                    continue
                    
                except (ConnectionResetError, ConnectionAbortedError) as e:
                    logger.warning(f"Connection reset by peer: {e}")
                    break
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Receive error ({consecutive_errors}): {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors})")
                        break
                        
        except Exception as e:
            logger.error(f"Critical receive error: {e}")
        finally:
            self.is_connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            if self.auto_reconnect:
                self._attempt_reconnect()
                
    def _process_message(self, message: str):
        try:
            msg_data = json.loads(message)
            command_type = msg_data.get("command_type")
            
            if command_type in self.message_handlers:
                self.message_handlers[command_type](msg_data)
            else:
                logger.info(f"Received message: {msg_data}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            
    def send_message(self, message_data: Dict[str, Any]) -> bool:
        return self._send_message_with_retry(message_data)
    
    def _send_message_with_retry(self, message_data: Dict[str, Any], max_send_retries: int = 3) -> bool:
        if not self.is_connected or not self.socket:
            logger.warning("Not connected, attempting to reconnect before sending")
            if not self.connect():
                return False
        
        for attempt in range(max_send_retries):
            try:
                message_json = json.dumps(message_data) + '\n'
                self.socket.settimeout(10.0)
                self.socket.send(message_json.encode('utf-8'))
                return True
                
            except socket.timeout:
                logger.warning(f"Send timeout (attempt {attempt + 1}/{max_send_retries})")
                if attempt < max_send_retries - 1:
                    time.sleep(0.5)
                    
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                logger.warning(f"Connection lost during send: {e}")
                self.is_connected = False
                if attempt < max_send_retries - 1:
                    if self.connect():
                        continue
                break
                
            except Exception as e:
                logger.error(f"Send error (attempt {attempt + 1}): {e}")
                if attempt < max_send_retries - 1:
                    time.sleep(0.2)
                    
        logger.error(f"Failed to send message after {max_send_retries} attempts")
        return False
        
    def _attempt_reconnect(self):
        if not self.auto_reconnect:
            return
            
        logger.info("Attempting to reconnect...")
        threading.Thread(target=self._background_reconnect, daemon=True).start()
    
    def _background_reconnect(self):
        reconnect_attempts = 0
        max_reconnect_attempts = 20
        
        while reconnect_attempts < max_reconnect_attempts and not self.is_connected and self.auto_reconnect:
            reconnect_delay = self._calculate_retry_delay(reconnect_attempts)
            logger.info(f"Reconnect attempt {reconnect_attempts + 1}/{max_reconnect_attempts} in {reconnect_delay:.1f}s")
            time.sleep(reconnect_delay)
            
            if self._connect_with_retry():
                logger.info("Reconnected successfully")
                return
                
            reconnect_attempts += 1
            
        if not self.is_connected:
            logger.error(f"Failed to reconnect after {max_reconnect_attempts} attempts")
            
    def disconnect(self):
        self.is_connected = False
        self.auto_reconnect = False
        if self.socket:
            self.socket.close()

class MessageBuilder:
    @staticmethod
    def create_message(command_type: str, target_component: str, parameters: Dict[str, Any], 
                      response_required: bool = True) -> Dict[str, Any]:
        return {
            "message_id": f"{int(time.time() * 1000)}_{command_type}",
            "timestamp": datetime.now().isoformat() + "Z",
            "command_type": command_type,
            "target_component": target_component,
            "parameters": parameters,
            "response_required": response_required
        }
    
    @staticmethod
    def create_response(original_message: Dict[str, Any], status: str, 
                       data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "message_id": f"response_{original_message.get('message_id')}",
            "timestamp": datetime.now().isoformat() + "Z",
            "response_to": original_message.get("message_id"),
            "status": status,
            "data": data or {}
        }
    
    @staticmethod
    def create_robot_move_command(target_component: str, position: list, speed: int = 100) -> Dict[str, Any]:
        return MessageBuilder.create_message(
            "robot_move",
            target_component,
            {"position": position, "speed": speed}
        )
    
    @staticmethod
    def create_io_control_command(target_component: str, io_data: Dict[str, bool]) -> Dict[str, Any]:
        return MessageBuilder.create_message(
            "io_control",
            target_component,
            {"io_data": io_data}
        )
    
    @staticmethod
    def create_tool_control_command(target_component: str, tool_command: str) -> Dict[str, Any]:
        return MessageBuilder.create_message(
            "tool_control",
            target_component,
            {"tool_command": tool_command}
        )
    
    @staticmethod
    def create_status_request(target_component: str) -> Dict[str, Any]:
        return MessageBuilder.create_message(
            "status_request",
            target_component,
            {},
            response_required=True
        )

if __name__ == "__main__":
    def test_ping_handler(message_data):
        logger.info("Received ping, sending pong")
        return MessageBuilder.create_response(message_data, "success", {"message": "pong"})
    
    server = TCPServer()
    server.register_handler("ping", test_ping_handler)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop_server()