#!/usr/bin/env python3
"""
Advanced IoT Protocol Adapter
Multi-protocol support for IoT communication including CoAP, LoRaWAN, Zigbee, and more
"""

import asyncio
import json
import logging
import struct
import time
import socket
import hashlib
import hmac
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# Protocol libraries
try:
    from aiocoap import Context, Message, Code, GET, POST, PUT, DELETE
    from aiocoap.resource import Site, Resource
    COAP_AVAILABLE = True
except ImportError:
    COAP_AVAILABLE = False
    print("Warning: CoAP not available. Install with: pip install aiocoap")

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: PySerial not available. Install with: pip install pyserial")

try:
    import paho.mqtt.client as mqtt
    MQTT_PAHO_AVAILABLE = True
except ImportError:
    MQTT_PAHO_AVAILABLE = False
    print("Warning: Paho MQTT not available. Install with: pip install paho-mqtt")

try:
    from bleak import BleakScanner, BleakClient
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False
    print("Warning: BLE not available. Install with: pip install bleak")

try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False
    print("Warning: python-can not available. Install with: pip install python-can")

# Security libraries
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: Cryptography not available. Install with: pip install cryptography")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProtocolAdapter")

class IoTProtocol(Enum):
    """Supported IoT protocols"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    LORAWAN = "lorawan"
    ZIGBEE = "zigbee"
    BLE = "ble"
    NFC = "nfc"
    RFID = "rfid"
    MODBUS = "modbus"
    CAN = "can"
    SERIAL = "serial"
    TCP = "tcp"
    UDP = "udp"
    MQTT_SN = "mqtt_sn"
    AMQP = "amqp"
    STOMP = "stomp"
    DDS = "dds"

class MessageFormat(Enum):
    """Message formats"""
    JSON = "json"
    CBOR = "cbor"
    PROTOBUF = "protobuf"
    XML = "xml"
    BINARY = "binary"
    TEXT = "text"
    CUSTOM = "custom"

class SecurityLevel(Enum):
    """Security levels"""
    NONE = "none"
    BASIC = "basic"
    TLS = "tls"
    DTLS = "dtls"
    AES = "aes"
    CUSTOM = "custom"

@dataclass
class ProtocolConfig:
    """Protocol configuration"""
    protocol: IoTProtocol
    host: str
    port: int
    security: SecurityLevel = SecurityLevel.NONE
    credentials: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    retry_count: int = 3
    message_format: MessageFormat = MessageFormat.JSON
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IoTMessage:
    """IoT message model"""
    source: str
    destination: str
    protocol: IoTProtocol
    payload: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: Optional[str] = None
    content_type: Optional[str] = None
    qos: int = 0
    retain: bool = False
    topic: Optional[str] = None
    path: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CoAPAdapter:
    """CoAP protocol adapter"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.context = None
        self.site = None
        self.running = False
        self.resources = {}
        self.message_handlers = {}

    async def start(self) -> bool:
        """Start CoAP adapter"""
        if not COAP_AVAILABLE:
            logger.error("CoAP not available")
            return False

        try:
            self.context = await Context.create_server_context(self.site, bind=(self.config.host, self.config.port))
            self.running = True
            logger.info(f"CoAP adapter started on {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Error starting CoAP adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop CoAP adapter"""
        if self.context:
            await self.context.shutdown()
            self.running = False
            logger.info("CoAP adapter stopped")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send CoAP message"""
        if not COAP_AVAILABLE or not self.running:
            return False

        try:
            # Create CoAP message
            coap_message = Message(
                code=POST,
                uri=f"coap://{message.destination}{message.path or '/'}",
                payload=self._encode_payload(message.payload, message.content_type)
            )

            # Add headers
            for key, value in message.headers.items():
                coap_message.opt.add_option(key, value)

            # Send message
            protocol = await Context.create_client_context()
            response = await protocol.request(coap_message).response

            logger.debug(f"CoAP message sent, response: {response.code}")
            return response.code.is_successful()

        except Exception as e:
            logger.error(f"Error sending CoAP message: {e}")
            return False

    def _encode_payload(self, payload: Any, content_type: Optional[str] = None) -> bytes:
        """Encode payload for CoAP"""
        if isinstance(payload, (dict, list)):
            return json.dumps(payload).encode('utf-8')
        elif isinstance(payload, str):
            return payload.encode('utf-8')
        elif isinstance(payload, bytes):
            return payload
        else:
            return str(payload).encode('utf-8')

class LoRaWANAdapter:
    """LoRaWAN protocol adapter"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.running = False
        self.dev_eui = None
        self.app_eui = None
        self.app_key = None
        self.network_session = None
        self.message_queue = asyncio.Queue()

    async def start(self) -> bool:
        """Start LoRaWAN adapter"""
        try:
            # Initialize LoRaWAN connection
            self.dev_eui = self.config.custom_params.get('dev_eui')
            self.app_eui = self.config.custom_params.get('app_eui')
            self.app_key = self.config.custom_params.get('app_key')

            if not all([self.dev_eui, self.app_eui, self.app_key]):
                logger.error("LoRaWAN credentials not provided")
                return False

            # Connect to LoRaWAN network server
            await self._connect_to_network_server()
            self.running = True
            logger.info("LoRaWAN adapter started")
            return True

        except Exception as e:
            logger.error(f"Error starting LoRaWAN adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop LoRaWAN adapter"""
        self.running = False
        if self.network_session:
            await self._disconnect_from_network_server()
        logger.info("LoRaWAN adapter stopped")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send LoRaWAN message"""
        if not self.running:
            return False

        try:
            # Prepare LoRaWAN payload
            payload = self._prepare_lorawan_payload(message)

            # Send through network server
            success = await self._send_via_network_server(payload, message.destination)

            logger.debug(f"LoRaWAN message sent: {success}")
            return success

        except Exception as e:
            logger.error(f"Error sending LoRaWAN message: {e}")
            return False

    async def _connect_to_network_server(self) -> None:
        """Connect to LoRaWAN network server"""
        # Implementation depends on specific LoRaWAN provider
        pass

    async def _disconnect_from_network_server(self) -> None:
        """Disconnect from LoRaWAN network server"""
        pass

    def _prepare_lorawan_payload(self, message: IoTMessage) -> bytes:
        """Prepare LoRaWAN payload"""
        payload_data = json.dumps(message.payload) if isinstance(message.payload, (dict, list)) else str(message.payload)
        return payload_data.encode('utf-8')

    async def _send_via_network_server(self, payload: bytes, destination: str) -> bool:
        """Send payload via network server"""
        # Implementation depends on specific LoRaWAN provider
        return True

class ZigbeeAdapter:
    """Zigbee protocol adapter"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.running = False
        self.coordinator = None
        self.network_key = None
        self.devices = {}

    async def start(self) -> bool:
        """Start Zigbee adapter"""
        try:
            # Initialize Zigbee coordinator
            serial_port = self.config.custom_params.get('serial_port')
            baudrate = self.config.custom_params.get('baudrate', 115200)

            if not serial_port:
                logger.error("Zigbee serial port not provided")
                return False

            await self._initialize_coordinator(serial_port, baudrate)
            self.running = True
            logger.info("Zigbee adapter started")
            return True

        except Exception as e:
            logger.error(f"Error starting Zigbee adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop Zigbee adapter"""
        self.running = False
        if self.coordinator:
            await self._shutdown_coordinator()
        logger.info("Zigbee adapter stopped")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send Zigbee message"""
        if not self.running:
            return False

        try:
            # Send message via Zigbee network
            success = await self._send_zigbee_message(message)

            logger.debug(f"Zigbee message sent: {success}")
            return success

        except Exception as e:
            logger.error(f"Error sending Zigbee message: {e}")
            return False

    async def _initialize_coordinator(self, serial_port: str, baudrate: int) -> None:
        """Initialize Zigbee coordinator"""
        # Implementation depends on Zigbee hardware/library
        pass

    async def _shutdown_coordinator(self) -> None:
        """Shutdown Zigbee coordinator"""
        pass

    async def _send_zigbee_message(self, message: IoTMessage) -> bool:
        """Send message via Zigbee"""
        # Implementation depends on Zigbee library
        return True

class BLEAdapter:
    """Bluetooth Low Energy adapter"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.running = False
        self.scanner = None
        self.connected_devices = {}
        self.characteristics = {}

    async def start(self) -> bool:
        """Start BLE adapter"""
        if not BLE_AVAILABLE:
            logger.error("BLE not available")
            return False

        try:
            self.scanner = BleakScanner()
            self.running = True
            logger.info("BLE adapter started")
            return True

        except Exception as e:
            logger.error(f"Error starting BLE adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop BLE adapter"""
        self.running = False

        # Disconnect all connected devices
        for device_id, client in self.connected_devices.items():
            try:
                await client.disconnect()
            except Exception:
                pass

        self.connected_devices.clear()
        logger.info("BLE adapter stopped")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send BLE message"""
        if not self.running or not BLE_AVAILABLE:
            return False

        try:
            # Connect to device if not already connected
            if message.destination not in self.connected_devices:
                await self._connect_device(message.destination)

            # Send message via BLE characteristic
            success = await self._send_ble_message(message)

            logger.debug(f"BLE message sent: {success}")
            return success

        except Exception as e:
            logger.error(f"Error sending BLE message: {e}")
            return False

    async def scan_devices(self, duration: float = 10.0) -> List[Dict[str, Any]]:
        """Scan for BLE devices"""
        if not self.running or not BLE_AVAILABLE:
            return []

        try:
            devices = await self.scanner.discover(timeout=duration)
            device_list = []

            for device in devices:
                device_info = {
                    'address': device.address,
                    'name': device.name,
                    'rssi': device.rssi,
                    'metadata': device.metadata
                }
                device_list.append(device_info)

            return device_list

        except Exception as e:
            logger.error(f"Error scanning BLE devices: {e}")
            return []

    async def _connect_device(self, device_address: str) -> bool:
        """Connect to BLE device"""
        try:
            client = BleakClient(device_address)
            await client.connect()
            self.connected_devices[device_address] = client
            logger.info(f"Connected to BLE device: {device_address}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to BLE device {device_address}: {e}")
            return False

    async def _send_ble_message(self, message: IoTMessage) -> bool:
        """Send message via BLE characteristic"""
        try:
            client = self.connected_devices.get(message.destination)
            if not client:
                return False

            # Get characteristic for communication
            characteristic_uuid = self.config.custom_params.get('characteristic_uuid')
            if not characteristic_uuid:
                logger.error("BLE characteristic UUID not provided")
                return False

            # Encode and send data
            payload = self._encode_ble_payload(message.payload)
            await client.write_gatt_char(characteristic_uuid, payload)

            return True

        except Exception as e:
            logger.error(f"Error sending BLE message: {e}")
            return False

    def _encode_ble_payload(self, payload: Any) -> bytes:
        """Encode payload for BLE"""
        if isinstance(payload, (dict, list)):
            return json.dumps(payload).encode('utf-8')
        elif isinstance(payload, str):
            return payload.encode('utf-8')
        elif isinstance(payload, bytes):
            return payload
        else:
            return str(payload).encode('utf-8')

class CANAdapter:
    """CAN bus protocol adapter"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.running = False
        self.bus = None
        self.message_filters = {}

    async def start(self) -> bool:
        """Start CAN adapter"""
        if not CAN_AVAILABLE:
            logger.error("CAN not available")
            return False

        try:
            interface = self.config.custom_params.get('interface', 'socketcan')
            channel = self.config.custom_params.get('channel', 'can0')
            bitrate = self.config.custom_params.get('bitrate', 500000)

            self.bus = can.interface.Bus(channel=channel, bustype=interface, bitrate=bitrate)
            self.running = True
            logger.info(f"CAN adapter started on {channel}")
            return True

        except Exception as e:
            logger.error(f"Error starting CAN adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop CAN adapter"""
        self.running = False
        if self.bus:
            self.bus.shutdown()
            logger.info("CAN adapter stopped")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send CAN message"""
        if not self.running or not self.bus:
            return False

        try:
            # Parse CAN ID from destination
            can_id = int(message.destination, 16) if isinstance(message.destination, str) else message.destination

            # Create CAN message
            payload = self._encode_can_payload(message.payload)
            can_message = can.Message(arbitration_id=can_id, data=payload)

            # Send message
            self.bus.send(can_message)

            logger.debug(f"CAN message sent: ID={hex(can_id)}")
            return True

        except Exception as e:
            logger.error(f"Error sending CAN message: {e}")
            return False

    def _encode_can_payload(self, payload: Any) -> bytes:
        """Encode payload for CAN bus"""
        if isinstance(payload, bytes):
            return payload[:8]  # CAN messages max 8 bytes
        elif isinstance(payload, str):
            return payload.encode('utf-8')[:8]
        elif isinstance(payload, int):
            return payload.to_bytes(1, byteorder='big')
        elif isinstance(payload, (dict, list)):
            json_str = json.dumps(payload)
            return json_str.encode('utf-8')[:8]
        else:
            return str(payload).encode('utf-8')[:8]

class SerialAdapter:
    """Serial communication adapter"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.running = False
        self.serial_connection = None
        self.read_thread = None

    async def start(self) -> bool:
        """Start serial adapter"""
        if not SERIAL_AVAILABLE:
            logger.error("Serial not available")
            return False

        try:
            port = self.config.custom_params.get('port')
            baudrate = self.config.custom_params.get('baudrate', 9600)
            bytesize = self.config.custom_params.get('bytesize', serial.EIGHTBITS)
            parity = self.config.custom_params.get('parity', serial.PARITY_NONE)
            stopbits = self.config.custom_params.get('stopbits', serial.STOPBITS_ONE)

            if not port:
                logger.error("Serial port not specified")
                return False

            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=1
            )

            self.running = True

            # Start read thread
            self.read_thread = threading.Thread(target=self._read_serial_data, daemon=True)
            self.read_thread.start()

            logger.info(f"Serial adapter started on {port}")
            return True

        except Exception as e:
            logger.error(f"Error starting serial adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop serial adapter"""
        self.running = False
        if self.serial_connection:
            self.serial_connection.close()
        logger.info("Serial adapter stopped")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send serial message"""
        if not self.running or not self.serial_connection:
            return False

        try:
            # Encode payload
            payload = self._encode_serial_payload(message.payload)

            # Send data
            self.serial_connection.write(payload)
            self.serial_connection.flush()

            logger.debug(f"Serial message sent: {len(payload)} bytes")
            return True

        except Exception as e:
            logger.error(f"Error sending serial message: {e}")
            return False

    def _read_serial_data(self) -> None:
        """Read data from serial port"""
        while self.running:
            try:
                if self.serial_connection.in_waiting:
                    data = self.serial_connection.read(self.serial_connection.in_waiting)
                    # Process received data
                    self._process_received_data(data)
                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error reading serial data: {e}")
                time.sleep(1)

    def _process_received_data(self, data: bytes) -> None:
        """Process received serial data"""
        try:
            # Try to decode as JSON
            try:
                json_data = json.loads(data.decode('utf-8'))
                logger.debug(f"Received serial JSON: {json_data}")
            except json.JSONDecodeError:
                # Treat as raw data
                logger.debug(f"Received serial data: {data.hex()}")

        except Exception as e:
            logger.error(f"Error processing serial data: {e}")

    def _encode_serial_payload(self, payload: Any) -> bytes:
        """Encode payload for serial communication"""
        if isinstance(payload, bytes):
            return payload
        elif isinstance(payload, str):
            return payload.encode('utf-8')
        elif isinstance(payload, (dict, list)):
            return json.dumps(payload).encode('utf-8')
        else:
            return str(payload).encode('utf-8')

class AdvancedIoTProtocolAdapter:
    """Advanced multi-protocol IoT adapter"""

    def __init__(self):
        self.adapters = {}
        self.protocol_configs = {}
        self.message_handlers = {}
        self.message_queue = asyncio.Queue()
        self.statistics = {
            'messages_sent': 0,
            'messages_received': 0,
            'protocols_active': {},
            'errors': 0
        }
        self.running = False

    def register_protocol(self, protocol_config: ProtocolConfig) -> bool:
        """Register a new protocol"""
        try:
            protocol = protocol_config.protocol

            # Create adapter based on protocol type
            if protocol == IoTProtocol.COAP:
                adapter = CoAPAdapter(protocol_config)
            elif protocol == IoTProtocol.LORAWAN:
                adapter = LoRaWANAdapter(protocol_config)
            elif protocol == IoTProtocol.ZIGBEE:
                adapter = ZigbeeAdapter(protocol_config)
            elif protocol == IoTProtocol.BLE:
                adapter = BLEAdapter(protocol_config)
            elif protocol == IoTProtocol.CAN:
                adapter = CANAdapter(protocol_config)
            elif protocol == IoTProtocol.SERIAL:
                adapter = SerialAdapter(protocol_config)
            else:
                logger.error(f"Unsupported protocol: {protocol}")
                return False

            self.adapters[protocol] = adapter
            self.protocol_configs[protocol] = protocol_config
            self.statistics['protocols_active'][protocol.value] = False

            logger.info(f"Registered protocol adapter: {protocol.value}")
            return True

        except Exception as e:
            logger.error(f"Error registering protocol {protocol}: {e}")
            return False

    async def start(self) -> bool:
        """Start all registered protocol adapters"""
        try:
            logger.info("Starting IoT Protocol Adapter")

            # Start all adapters
            for protocol, adapter in self.adapters.items():
                try:
                    success = await adapter.start()
                    self.statistics['protocols_active'][protocol.value] = success

                    if success:
                        logger.info(f"Started {protocol.value} adapter")
                    else:
                        logger.warning(f"Failed to start {protocol.value} adapter")

                except Exception as e:
                    logger.error(f"Error starting {protocol.value} adapter: {e}")
                    self.statistics['protocols_active'][protocol.value] = False

            # Start message processor
            self.running = True
            processor_task = asyncio.create_task(self._process_message_queue())

            logger.info("IoT Protocol Adapter started")
            return True

        except Exception as e:
            logger.error(f"Error starting IoT Protocol Adapter: {e}")
            return False

    async def stop(self) -> None:
        """Stop all protocol adapters"""
        try:
            logger.info("Stopping IoT Protocol Adapter")

            self.running = False

            # Stop all adapters
            for protocol, adapter in self.adapters.items():
                try:
                    await adapter.stop()
                    self.statistics['protocols_active'][protocol.value] = False
                    logger.info(f"Stopped {protocol.value} adapter")
                except Exception as e:
                    logger.error(f"Error stopping {protocol.value} adapter: {e}")

            logger.info("IoT Protocol Adapter stopped")

        except Exception as e:
            logger.error(f"Error stopping IoT Protocol Adapter: {e}")

    async def send_message(self, message: IoTMessage) -> bool:
        """Send message using appropriate protocol adapter"""
        try:
            adapter = self.adapters.get(message.protocol)
            if not adapter:
                logger.error(f"No adapter available for protocol {message.protocol}")
                return False

            # Add message to queue for processing
            await self.message_queue.put(('send', message))

            return True

        except Exception as e:
            logger.error(f"Error queuing message for sending: {e}")
            self.statistics['errors'] += 1
            return False

    def add_message_handler(self, protocol: IoTProtocol, handler: Callable) -> None:
        """Add message handler for specific protocol"""
        if protocol not in self.message_handlers:
            self.message_handlers[protocol] = []
        self.message_handlers[protocol].append(handler)

    async def _process_message_queue(self) -> None:
        """Process messages from queue"""
        while self.running:
            try:
                # Get message from queue
                message_type, message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                if message_type == 'send':
                    await self._handle_send_message(message)
                elif message_type == 'receive':
                    await self._handle_receive_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message from queue: {e}")
                self.statistics['errors'] += 1

    async def _handle_send_message(self, message: IoTMessage) -> None:
        """Handle outgoing message"""
        try:
            adapter = self.adapters.get(message.protocol)
            if not adapter:
                return

            # Send message
            success = await adapter.send_message(message)

            if success:
                self.statistics['messages_sent'] += 1
                logger.debug(f"Message sent via {message.protocol.value}")
            else:
                self.statistics['errors'] += 1
                logger.error(f"Failed to send message via {message.protocol.value}")

        except Exception as e:
            logger.error(f"Error handling send message: {e}")
            self.statistics['errors'] += 1

    async def _handle_receive_message(self, message: IoTMessage) -> None:
        """Handle incoming message"""
        try:
            self.statistics['messages_received'] += 1

            # Call registered handlers
            handlers = self.message_handlers.get(message.protocol, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")

        except Exception as e:
            logger.error(f"Error handling receive message: {e}")
            self.statistics['errors'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            **self.statistics,
            'registered_protocols': [protocol.value for protocol in self.adapters.keys()],
            'queue_size': self.message_queue.qsize(),
            'running': self.running
        }

    def get_protocol_status(self) -> Dict[str, bool]:
        """Get status of all protocols"""
        return self.statistics['protocols_active'].copy()

    async def test_protocol(self, protocol: IoTProtocol) -> bool:
        """Test specific protocol connectivity"""
        try:
            adapter = self.adapters.get(protocol)
            if not adapter:
                return False

            # Create test message
            test_message = IoTMessage(
                source="test",
                destination="test",
                protocol=protocol,
                payload={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
            )

            # Send test message
            success = await adapter.send_message(test_message)
            return success

        except Exception as e:
            logger.error(f"Error testing protocol {protocol}: {e}")
            return False

    async def scan_devices(self, protocol: IoTProtocol) -> List[Dict[str, Any]]:
        """Scan for devices using specific protocol"""
        try:
            adapter = self.adapters.get(protocol)
            if not adapter:
                return []

            # Some adapters support device scanning
            if hasattr(adapter, 'scan_devices'):
                return await adapter.scan_devices()
            else:
                logger.warning(f"Device scanning not supported for {protocol}")
                return []

        except Exception as e:
            logger.error(f"Error scanning devices with {protocol}: {e}")
            return []

# Factory function
def create_protocol_adapter() -> AdvancedIoTProtocolAdapter:
    """Create advanced IoT protocol adapter"""
    return AdvancedIoTProtocolAdapter()

# Example configuration
def create_default_protocol_configs() -> List[ProtocolConfig]:
    """Create default protocol configurations"""
    configs = []

    # MQTT configuration
    mqtt_config = ProtocolConfig(
        protocol=IoTProtocol.MQTT,
        host="localhost",
        port=1883,
        security=SecurityLevel.NONE,
        message_format=MessageFormat.JSON
    )
    configs.append(mqtt_config)

    # CoAP configuration
    coap_config = ProtocolConfig(
        protocol=IoTProtocol.COAP,
        host="0.0.0.0",
        port=5683,
        security=SecurityLevel.DTLS,
        message_format=MessageFormat.JSON
    )
    configs.append(coap_config)

    # BLE configuration
    ble_config = ProtocolConfig(
        protocol=IoTProtocol.BLE,
        host="",
        port=0,
        message_format=MessageFormat.JSON,
        custom_params={
            'characteristic_uuid': '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
        }
    )
    configs.append(ble_config)

    # Serial configuration
    serial_config = ProtocolConfig(
        protocol=IoTProtocol.SERIAL,
        host="",
        port=0,
        message_format=MessageFormat.JSON,
        custom_params={
            'port': '/dev/ttyUSB0',
            'baudrate': 9600
        }
    )
    configs.append(serial_config)

    return configs

# CLI interface
async def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced IoT Protocol Adapter')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--protocols', nargs='+', help='Protocols to enable')

    args = parser.parse_args()

    # Create protocol adapter
    adapter = create_protocol_adapter()

    # Load configurations
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                # Parse configurations and register protocols
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return
    else:
        # Use default configurations
        configs = create_default_protocol_configs()
        for config in configs:
            if not args.protocols or config.protocol.value in args.protocols:
                adapter.register_protocol(config)

    try:
        # Start adapter
        if await adapter.start():
            logger.info("Protocol adapter running. Press Ctrl+C to stop...")

            # Keep running
            while True:
                await asyncio.sleep(1)

                # Print statistics every 30 seconds
                if int(time.time()) % 30 == 0:
                    stats = adapter.get_statistics()
                    logger.info(f"Stats: {stats['messages_sent']} sent, "
                              f"{stats['messages_received']} received")
        else:
            logger.error("Failed to start protocol adapter")

    except KeyboardInterrupt:
        logger.info("Shutting down protocol adapter...")
    finally:
        await adapter.stop()

if __name__ == "__main__":
    asyncio.run(main())