#!/usr/bin/env python3
"""
Comprehensive IoT Gateway with Advanced MQTT Broker
IoT Hub implementation for device management, data collection, and protocol translation
"""

import asyncio
import json
import logging
import ssl
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor

# MQTT Broker
try:
    from hbmqtt.broker import Broker
    from hbmqtt.client import MQTTClient
    from hbmqtt.mqtt.constants import QOS_0, QOS_1, QOS_2
    from hbmqqt.server import get_broker
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: MQTT not available. Install with: pip install hbmqtt")

# CoAP Protocol
try:
    from aiocoap import Context, Message, Code
    from aiocoap.resource import Site, Resource
    COAP_AVAILABLE = True
except ImportError:
    COAP_AVAILABLE = False
    print("Warning: CoAP not available. Install with: pip install aiocoap")

# LoRaWAN
try:
    from lorawan_stack import LoRaWANStack
    LORAWAN_AVAILABLE = True
except ImportError:
    LORAWAN_AVAILABLE = False
    print("Warning: LoRaWAN not available")

# WebSocket
try:
    import websockets
    import asyncio
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: WebSockets not available")

# Database
try:
    import sqlite3
    import aiosqlite
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: Database not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IoTGateway")

class DeviceType(Enum):
    """IoT Device types"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CONTROLLER = "controller"
    CAMERA = "camera"
    RFID_READER = "rfid_reader"
    TEMPERATURE_SENSOR = "temperature_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    MOTION_DETECTOR = "motion_detector"
    SMART_METER = "smart_meter"
    INDUSTRIAL_ROBOT = "industrial_robot"
    PLC = "plc"

class ProtocolType(Enum):
    """Supported IoT protocols"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    LORAWAN = "lorawan"
    MODBUS = "modbus"
    OPC_UA = "opc_ua"

class DeviceStatus(Enum):
    """Device status"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    CONNECTING = "connecting"
    UNKNOWN = "unknown"

@dataclass
class IoTDevice:
    """IoT Device model"""
    device_id: str
    name: str
    device_type: DeviceType
    protocol: ProtocolType
    connection_string: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    hardware_version: Optional[str] = None
    status: DeviceStatus = DeviceStatus.UNKNOWN
    last_seen: Optional[datetime] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    data_schema: Optional[Dict[str, Any]] = None
    security_credentials: Optional[Dict[str, str]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class TelemetryData:
    """Telemetry data model"""
    device_id: str
    timestamp: datetime
    data_type: str
    value: Any
    unit: Optional[str] = None
    quality: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceCommand:
    """Device command model"""
    command_id: str
    device_id: str
    command_type: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"
    response: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

class DeviceRegistry:
    """Registry for managing IoT devices"""

    def __init__(self, db_path: str = "iot_devices.db"):
        self.db_path = db_path
        self.devices: Dict[str, IoTDevice] = {}
        self.devices_by_type: Dict[DeviceType, List[str]] = {}
        self.devices_by_protocol: Dict[ProtocolType, List[str]] = {}
        self._lock = threading.RLock()

        if DB_AVAILABLE:
            self._init_database()

    def _init_database(self):
        """Initialize SQLite database for device registry"""
        async def init_db():
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS devices (
                        device_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        device_type TEXT NOT NULL,
                        protocol TEXT NOT NULL,
                        connection_string TEXT NOT NULL,
                        manufacturer TEXT,
                        model TEXT,
                        firmware_version TEXT,
                        hardware_version TEXT,
                        status TEXT NOT NULL,
                        last_seen TIMESTAMP,
                        location TEXT,
                        metadata TEXT,
                        configuration TEXT,
                        capabilities TEXT,
                        topics TEXT,
                        data_schema TEXT,
                        security_credentials TEXT,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        data_type TEXT NOT NULL,
                        value TEXT NOT NULL,
                        unit TEXT,
                        quality TEXT,
                        metadata TEXT,
                        FOREIGN KEY (device_id) REFERENCES devices (device_id)
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS commands (
                        command_id TEXT PRIMARY KEY,
                        device_id TEXT NOT NULL,
                        command_type TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        response TEXT,
                        execution_time REAL,
                        FOREIGN KEY (device_id) REFERENCES devices (device_id)
                    )
                """)

                await db.commit()
                logger.info("Database initialized successfully")

        asyncio.run(init_db())

    def register_device(self, device: IoTDevice) -> bool:
        """Register a new IoT device"""
        with self._lock:
            try:
                if device.device_id in self.devices:
                    logger.warning(f"Device {device.device_id} already registered")
                    return False

                self.devices[device.device_id] = device

                # Update type mapping
                if device.device_type not in self.devices_by_type:
                    self.devices_by_type[device.device_type] = []
                self.devices_by_type[device.device_type].append(device.device_id)

                # Update protocol mapping
                if device.protocol not in self.devices_by_protocol:
                    self.devices_by_protocol[device.protocol] = []
                self.devices_by_protocol[device.protocol].append(device.device_id)

                logger.info(f"Device registered: {device.name} ({device.device_id})")
                return True

            except Exception as e:
                logger.error(f"Error registering device {device.device_id}: {e}")
                return False

    def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device"""
        with self._lock:
            try:
                if device_id not in self.devices:
                    logger.warning(f"Device {device_id} not found")
                    return False

                device = self.devices[device_id]

                # Remove from type mapping
                if device.device_type in self.devices_by_type:
                    self.devices_by_type[device.device_type].remove(device_id)

                # Remove from protocol mapping
                if device.protocol in self.devices_by_protocol:
                    self.devices_by_protocol[device.protocol].remove(device_id)

                del self.devices[device_id]

                logger.info(f"Device unregistered: {device_id}")
                return True

            except Exception as e:
                logger.error(f"Error unregistering device {device_id}: {e}")
                return False

    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Get device by ID"""
        with self._lock:
            return self.devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[IoTDevice]:
        """Get devices by type"""
        with self._lock:
            device_ids = self.devices_by_type.get(device_type, [])
            return [self.devices[device_id] for device_id in device_ids if device_id in self.devices]

    def get_devices_by_protocol(self, protocol: ProtocolType) -> List[IoTDevice]:
        """Get devices by protocol"""
        with self._lock:
            device_ids = self.devices_by_protocol.get(protocol, [])
            return [self.devices[device_id] for device_id in device_ids if device_id in self.devices]

    def get_all_devices(self) -> List[IoTDevice]:
        """Get all registered devices"""
        with self._lock:
            return list(self.devices.values())

    def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """Update device status"""
        with self._lock:
            try:
                if device_id not in self.devices:
                    return False

                self.devices[device_id].status = status
                self.devices[device_id].last_seen = datetime.now(timezone.utc)
                self.devices[device_id].updated_at = datetime.now(timezone.utc)

                logger.info(f"Device {device_id} status updated to {status.value}")
                return True

            except Exception as e:
                logger.error(f"Error updating device status {device_id}: {e}")
                return False

class TelemetryStorage:
    """Telemetry data storage"""

    def __init__(self, db_path: str = "iot_devices.db"):
        self.db_path = db_path
        self._lock = threading.RLock()

    async def store_telemetry(self, telemetry: TelemetryData) -> bool:
        """Store telemetry data"""
        if not DB_AVAILABLE:
            logger.warning("Database not available, telemetry not stored")
            return False

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO telemetry (device_id, timestamp, data_type, value, unit, quality, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    telemetry.device_id,
                    telemetry.timestamp.isoformat(),
                    telemetry.data_type,
                    json.dumps(telemetry.value),
                    telemetry.unit,
                    telemetry.quality,
                    json.dumps(telemetry.metadata)
                ))
                await db.commit()

            logger.debug(f"Telemetry stored for device {telemetry.device_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing telemetry: {e}")
            return False

    async def get_telemetry(self, device_id: str, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None, limit: int = 1000) -> List[TelemetryData]:
        """Get telemetry data for a device"""
        if not DB_AVAILABLE:
            return []

        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = "SELECT * FROM telemetry WHERE device_id = ?"
                params = [device_id]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                telemetry_list = []
                for row in rows:
                    telemetry = TelemetryData(
                        device_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        data_type=row[3],
                        value=json.loads(row[4]),
                        unit=row[5],
                        quality=row[6],
                        metadata=json.loads(row[7]) if row[7] else {}
                    )
                    telemetry_list.append(telemetry)

                return telemetry_list

        except Exception as e:
            logger.error(f"Error getting telemetry: {e}")
            return []

class AdvancedMQTTBroker:
    """Advanced MQTT Broker with device management"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.broker = None
        self.running = False
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.device_registry = DeviceRegistry()
        self.telemetry_storage = TelemetryStorage()

        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'clients_connected': 0,
            'clients_disconnected': 0,
            'bytes_received': 0,
            'bytes_sent': 0
        }

        self._setup_broker_config()

    def _setup_broker_config(self):
        """Setup broker configuration"""
        broker_config = {
            'listeners': {
                'default': {
                    'type': 'tcp',
                    'bind': f"{self.config.get('host', '0.0.0.0')}:{self.config.get('port', 1883)}"
                },
                'ws-mqtt': {
                    'type': 'ws',
                    'bind': f"{self.config.get('ws_host', '0.0.0.0')}:{self.config.get('ws_port', 9001)}"
                }
            },
            'sys_interval': 10,
            'auth': {
                'allow-anonymous': self.config.get('allow_anonymous', True),
                'password-file': self.config.get('password_file', None)
            },
            'plugins': {
                'auth_file': None,
                'auth_mongo': None,
                'auth_anonymous': self.config.get('allow_anonymous', True)
            }
        }

        # Add SSL configuration if enabled
        if self.config.get('ssl_enabled', False):
            broker_config['listeners']['tls-default'] = {
                'type': 'ssl',
                'bind': f"{self.config.get('ssl_host', '0.0.0.0')}:{self.config.get('ssl_port', 8883)}",
                'ssl': 'default'
            }
            broker_config['tls'] = {
                'default': {
                    'certfile': self.config.get('ssl_certfile'),
                    'keyfile': self.config.get('ssl_keyfile')
                }
            }

        self.broker_config = broker_config

    async def start(self) -> bool:
        """Start the MQTT broker"""
        if not MQTT_AVAILABLE:
            logger.error("MQTT not available")
            return False

        try:
            if self.running:
                logger.warning("Broker already running")
                return True

            self.broker = Broker(self.broker_config)

            await self.broker.start()
            self.running = True

            logger.info(f"MQTT Broker started on {self.config.get('host', '0.0.0.0')}:{self.config.get('port', 1883)}")
            return True

        except Exception as e:
            logger.error(f"Error starting MQTT broker: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the MQTT broker"""
        if not MQTT_AVAILABLE or not self.broker:
            return False

        try:
            if not self.running:
                return True

            await self.broker.shutdown()
            self.running = False

            logger.info("MQTT Broker stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping MQTT broker: {e}")
            return False

    def add_message_handler(self, topic_pattern: str, handler: Callable):
        """Add message handler for topic pattern"""
        if topic_pattern not in self.message_handlers:
            self.message_handlers[topic_pattern] = []
        self.message_handlers[topic_pattern].append(handler)
        logger.info(f"Added handler for topic pattern: {topic_pattern}")

    async def publish_message(self, topic: str, payload: Any, qos: int = 0, retain: bool = False) -> bool:
        """Publish message to topic"""
        if not MQTT_AVAILABLE or not self.running:
            return False

        try:
            if isinstance(payload, dict):
                payload = json.dumps(payload)

            client = MQTTClient()
            await client.connect(f"mqtt://{self.config.get('host', 'localhost')}:{self.config.get('port', 1883)}")
            await client.publish(topic, payload.encode(), qos, retain)
            await client.disconnect()

            self.stats['messages_sent'] += 1
            logger.debug(f"Published message to {topic}")
            return True

        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False

class DeviceDiscoveryService:
    """Service for automatic IoT device discovery"""

    def __init__(self, device_registry: DeviceRegistry):
        self.device_registry = device_registry
        self.discovery_methods = []
        self.running = False

        # Setup discovery methods
        self._setup_discovery_methods()

    def _setup_discovery_methods(self):
        """Setup available discovery methods"""
        # MQTT auto-discovery
        if MQTT_AVAILABLE:
            self.discovery_methods.append(self._mqtt_discovery)

        # mDNS/Bonjour discovery
        self.discovery_methods.append(self._mdns_discovery)

        # UPnP discovery
        self.discovery_methods.append(self._upnp_discovery)

        # Network scanning
        self.discovery_methods.append(self._network_scan)

    async def start_discovery(self, interval: int = 300) -> None:
        """Start automatic device discovery"""
        self.running = True

        while self.running:
            logger.info("Starting device discovery scan")

            for discovery_method in self.discovery_methods:
                try:
                    await discovery_method()
                except Exception as e:
                    logger.error(f"Error in discovery method {discovery_method.__name__}: {e}")

            await asyncio.sleep(interval)

    def stop_discovery(self) -> None:
        """Stop device discovery"""
        self.running = False
        logger.info("Device discovery stopped")

    async def _mqtt_discovery(self) -> None:
        """MQTT-based device discovery"""
        # Implementation for MQTT auto-discovery
        # Listen to discovery topics and auto-register devices
        pass

    async def _mdns_discovery(self) -> None:
        """mDNS/Bonjour device discovery"""
        # Implementation for mDNS discovery
        pass

    async def _upnp_discovery(self) -> None:
        """UPnP device discovery"""
        # Implementation for UPnP discovery
        pass

    async def _network_scan(self) -> None:
        """Network scanning for IoT devices"""
        # Implementation for network scanning
        pass

class ProtocolAdapter:
    """Adapter for different IoT protocols"""

    def __init__(self, device_registry: DeviceRegistry, mqtt_broker: AdvancedMQTTBroker):
        self.device_registry = device_registry
        self.mqtt_broker = mqtt_broker
        self.adapters = {}
        self._setup_adapters()

    def _setup_adapters(self):
        """Setup protocol adapters"""
        # MQTT adapter
        self.adapters[ProtocolType.MQTT] = self._mqtt_adapter

        # CoAP adapter
        if COAP_AVAILABLE:
            self.adapters[ProtocolType.COAP] = self._coap_adapter

        # HTTP adapter
        self.adapters[ProtocolType.HTTP] = self._http_adapter

        # WebSocket adapter
        if WEBSOCKET_AVAILABLE:
            self.adapters[ProtocolType.WEBSOCKET] = self._websocket_adapter

    async def translate_message(self, device: IoTDevice, message: Any,
                              target_protocol: ProtocolType) -> Optional[Any]:
        """Translate message from device protocol to target protocol"""
        adapter = self.adapters.get(target_protocol)
        if not adapter:
            logger.error(f"No adapter available for protocol {target_protocol}")
            return None

        try:
            return await adapter(device, message)
        except Exception as e:
            logger.error(f"Error translating message: {e}")
            return None

    async def _mqtt_adapter(self, device: IoTDevice, message: Any) -> Any:
        """MQTT protocol adapter"""
        # Convert message to MQTT format
        if isinstance(message, dict):
            return json.dumps(message)
        return str(message)

    async def _coap_adapter(self, device: IoTDevice, message: Any) -> Any:
        """CoAP protocol adapter"""
        # Convert message to CoAP format
        return message

    async def _http_adapter(self, device: IoTDevice, message: Any) -> Any:
        """HTTP protocol adapter"""
        # Convert message to HTTP format
        return message

    async def _websocket_adapter(self, device: IoTDevice, message: Any) -> Any:
        """WebSocket protocol adapter"""
        # Convert message to WebSocket format
        return json.dumps(message) if isinstance(message, dict) else str(message)

class ComprehensiveIoTGateway:
    """Comprehensive IoT Gateway with all features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False

        # Core components
        self.device_registry = DeviceRegistry()
        self.mqtt_broker = AdvancedMQTTBroker(config.get('mqtt', {}))
        self.telemetry_storage = TelemetryStorage()
        self.device_discovery = DeviceDiscoveryService(self.device_registry)
        self.protocol_adapter = ProtocolAdapter(self.device_registry, self.mqtt_broker)

        # Command queue
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.telemetry_queue: asyncio.Queue = asyncio.Queue()

        # Statistics
        self.stats = {
            'devices_registered': 0,
            'messages_processed': 0,
            'commands_executed': 0,
            'uptime': 0,
            'start_time': None
        }

        # Task executor
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def start(self) -> bool:
        """Start the IoT Gateway"""
        try:
            logger.info("Starting Comprehensive IoT Gateway")

            # Start MQTT broker
            if not await self.mqtt_broker.start():
                logger.error("Failed to start MQTT broker")
                return False

            # Start device discovery
            discovery_task = asyncio.create_task(
                self.device_discovery.start_discovery(
                    interval=self.config.get('discovery_interval', 300)
                )
            )

            # Start telemetry processor
            telemetry_task = asyncio.create_task(self._process_telemetry())

            # Start command processor
            command_task = asyncio.create_task(self._process_commands())

            # Setup default message handlers
            self._setup_message_handlers()

            self.running = True
            self.stats['start_time'] = time.time()

            logger.info("IoT Gateway started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting IoT Gateway: {e}")
            return False

    async def stop(self) -> None:
        """Stop the IoT Gateway"""
        try:
            logger.info("Stopping IoT Gateway")

            self.running = False

            # Stop device discovery
            self.device_discovery.stop_discovery()

            # Stop MQTT broker
            await self.mqtt_broker.stop()

            # Shutdown executor
            self.executor.shutdown(wait=True)

            logger.info("IoT Gateway stopped")

        except Exception as e:
            logger.error(f"Error stopping IoT Gateway: {e}")

    def _setup_message_handlers(self):
        """Setup default message handlers"""
        # Telemetry handler
        self.mqtt_broker.add_message_handler(
            "devices/+/telemetry",
            self._handle_telemetry_message
        )

        # Device status handler
        self.mqtt_broker.add_message_handler(
            "devices/+/status",
            self._handle_status_message
        )

        # Device registration handler
        self.mqtt_broker.add_message_handler(
            "devices/+/register",
            self._handle_registration_message
        )

    async def _handle_telemetry_message(self, topic: str, payload: Any) -> None:
        """Handle incoming telemetry messages"""
        try:
            # Parse topic to get device_id
            topic_parts = topic.split('/')
            if len(topic_parts) < 3:
                return

            device_id = topic_parts[1]

            # Parse telemetry data
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')

            try:
                data = json.loads(payload) if isinstance(payload, str) else payload
            except json.JSONDecodeError:
                data = {'value': payload}

            # Create telemetry object
            telemetry = TelemetryData(
                device_id=device_id,
                timestamp=datetime.now(timezone.utc),
                data_type=data.get('type', 'unknown'),
                value=data.get('value', data),
                unit=data.get('unit'),
                quality=data.get('quality'),
                metadata=data.get('metadata', {})
            )

            # Add to telemetry queue
            await self.telemetry_queue.put(telemetry)

            # Update device status
            await self.device_registry.update_device_status(device_id, DeviceStatus.ONLINE)

            # Update stats
            self.stats['messages_processed'] += 1

            logger.debug(f"Received telemetry from device {device_id}")

        except Exception as e:
            logger.error(f"Error handling telemetry message: {e}")

    async def _handle_status_message(self, topic: str, payload: Any) -> None:
        """Handle device status messages"""
        try:
            topic_parts = topic.split('/')
            if len(topic_parts) < 3:
                return

            device_id = topic_parts[1]

            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')

            try:
                data = json.loads(payload) if isinstance(payload, str) else payload
                status = data.get('status', 'unknown')
            except json.JSONDecodeError:
                status = str(payload)

            # Convert to DeviceStatus enum
            device_status = DeviceStatus.UNKNOWN
            for status_enum in DeviceStatus:
                if status_enum.value == status:
                    device_status = status_enum
                    break

            await self.device_registry.update_device_status(device_id, device_status)

        except Exception as e:
            logger.error(f"Error handling status message: {e}")

    async def _handle_registration_message(self, topic: str, payload: Any) -> None:
        """Handle device registration messages"""
        try:
            topic_parts = topic.split('/')
            if len(topic_parts) < 3:
                return

            device_id = topic_parts[1]

            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')

            try:
                data = json.loads(payload) if isinstance(payload, str) else payload
            except json.JSONDecodeError:
                logger.error(f"Invalid registration data from device {device_id}")
                return

            # Create device object
            device = IoTDevice(
                device_id=device_id,
                name=data.get('name', f"Device_{device_id}"),
                device_type=DeviceType(data.get('type', 'sensor')),
                protocol=ProtocolType(data.get('protocol', 'mqtt')),
                connection_string=data.get('connection_string', ''),
                manufacturer=data.get('manufacturer'),
                model=data.get('model'),
                firmware_version=data.get('firmware_version'),
                hardware_version=data.get('hardware_version'),
                location=data.get('location'),
                metadata=data.get('metadata', {}),
                configuration=data.get('configuration', {}),
                capabilities=data.get('capabilities', []),
                topics=data.get('topics', []),
                data_schema=data.get('data_schema')
            )

            # Register device
            if self.device_registry.register_device(device):
                self.stats['devices_registered'] += 1
                logger.info(f"Device auto-registered: {device.name}")

                # Send acknowledgment
                await self.mqtt_broker.publish_message(
                    f"devices/{device_id}/registered",
                    {"status": "success", "device_id": device_id}
                )

        except Exception as e:
            logger.error(f"Error handling registration message: {e}")

    async def _process_telemetry(self) -> None:
        """Process telemetry data from queue"""
        while self.running:
            try:
                telemetry = await self.telemetry_queue.get()

                # Store telemetry
                await self.telemetry_storage.store_telemetry(telemetry)

                # Process telemetry for rules, alerts, etc.
                await self._process_telemetry_rules(telemetry)

                # Forward to other systems if needed
                await self._forward_telemetry(telemetry)

            except Exception as e:
                logger.error(f"Error processing telemetry: {e}")
                await asyncio.sleep(1)

    async def _process_commands(self) -> None:
        """Process commands from queue"""
        while self.running:
            try:
                command = await self.command_queue.get()

                # Execute command
                start_time = time.time()
                result = await self._execute_command(command)
                execution_time = time.time() - start_time

                # Update command status
                command.status = "completed" if result else "failed"
                command.execution_time = execution_time
                command.response = result

                self.stats['commands_executed'] += 1

            except Exception as e:
                logger.error(f"Error processing command: {e}")
                await asyncio.sleep(1)

    async def _process_telemetry_rules(self, telemetry: TelemetryData) -> None:
        """Process telemetry against rules and generate alerts"""
        # Implementation for rule-based processing
        pass

    async def _forward_telemetry(self, telemetry: TelemetryData) -> None:
        """Forward telemetry to other systems"""
        # Implementation for forwarding to cloud, analytics, etc.
        pass

    async def _execute_command(self, command: DeviceCommand) -> Optional[Dict[str, Any]]:
        """Execute device command"""
        try:
            device = self.device_registry.get_device(command.device_id)
            if not device:
                logger.error(f"Device {command.device_id} not found")
                return None

            # Execute command based on device protocol
            if device.protocol == ProtocolType.MQTT:
                return await self._execute_mqtt_command(device, command)
            elif device.protocol == ProtocolType.HTTP:
                return await self._execute_http_command(device, command)
            else:
                logger.error(f"Protocol {device.protocol} not supported for commands")
                return None

        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return None

    async def _execute_mqtt_command(self, device: IoTDevice, command: DeviceCommand) -> Optional[Dict[str, Any]]:
        """Execute command via MQTT"""
        topic = f"devices/{device.device_id}/commands/{command.command_type}"

        success = await self.mqtt_broker.publish_message(
            topic,
            command.parameters,
            qos=1
        )

        return {"success": success, "topic": topic}

    async def _execute_http_command(self, device: IoTDevice, command: DeviceCommand) -> Optional[Dict[str, Any]]:
        """Execute command via HTTP"""
        # Implementation for HTTP command execution
        return None

    async def send_command(self, device_id: str, command_type: str,
                          parameters: Dict[str, Any]) -> Optional[str]:
        """Send command to device"""
        try:
            command = DeviceCommand(
                command_id=str(uuid.uuid4()),
                device_id=device_id,
                command_type=command_type,
                parameters=parameters
            )

            await self.command_queue.put(command)
            return command.command_id

        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0

        return {
            **self.stats,
            'uptime': uptime,
            'devices_online': len([
                d for d in self.device_registry.get_all_devices()
                if d.status == DeviceStatus.ONLINE
            ]),
            'devices_total': len(self.device_registry.get_all_devices()),
            'mqtt_stats': self.mqtt_broker.stats,
            'running': self.running
        }

    def get_device_summary(self) -> Dict[str, Any]:
        """Get device summary information"""
        devices = self.device_registry.get_all_devices()

        summary = {
            'total_devices': len(devices),
            'devices_by_type': {},
            'devices_by_status': {},
            'devices_by_protocol': {},
            'recently_active': []
        }

        # Count by type
        for device_type in DeviceType:
            count = len([d for d in devices if d.device_type == device_type])
            if count > 0:
                summary['devices_by_type'][device_type.value] = count

        # Count by status
        for status in DeviceStatus:
            count = len([d for d in devices if d.status == status])
            if count > 0:
                summary['devices_by_status'][status.value] = count

        # Count by protocol
        for protocol in ProtocolType:
            count = len([d for d in devices if d.protocol == protocol])
            if count > 0:
                summary['devices_by_protocol'][protocol.value] = count

        # Recently active devices
        recent_devices = sorted(
            [d for d in devices if d.last_seen],
            key=lambda d: d.last_seen,
            reverse=True
        )[:10]

        summary['recently_active'] = [
            {
                'device_id': d.device_id,
                'name': d.name,
                'type': d.device_type.value,
                'status': d.status.value,
                'last_seen': d.last_seen.isoformat() if d.last_seen else None
            }
            for d in recent_devices
        ]

        return summary

# Factory function for creating IoT Gateway
def create_iot_gateway(config: Optional[Dict[str, Any]] = None) -> ComprehensiveIoTGateway:
    """Create IoT Gateway with default configuration"""
    if config is None:
        config = {
            'mqtt': {
                'host': '0.0.0.0',
                'port': 1883,
                'ws_host': '0.0.0.0',
                'ws_port': 9001,
                'allow_anonymous': True,
                'ssl_enabled': False
            },
            'discovery_interval': 300,
            'database_path': 'iot_devices.db'
        }

    return ComprehensiveIoTGateway(config)

# CLI interface
async def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive IoT Gateway')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--host', default='0.0.0.0', help='MQTT broker host')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--ws-port', type=int, default=9001, help='WebSocket port')
    parser.add_argument('--discovery-interval', type=int, default=300,
                       help='Device discovery interval (seconds)')

    args = parser.parse_args()

    # Create configuration
    config = {
        'mqtt': {
            'host': args.host,
            'port': args.port,
            'ws_host': args.host,
            'ws_port': args.ws_port,
            'allow_anonymous': True
        },
        'discovery_interval': args.discovery_interval
    }

    # Load additional config from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return

    # Create and start gateway
    gateway = create_iot_gateway(config)

    try:
        if await gateway.start():
            logger.info("IoT Gateway running. Press Ctrl+C to stop...")

            # Keep running
            while True:
                await asyncio.sleep(1)

                # Print statistics every 60 seconds
                if int(time.time()) % 60 == 0:
                    stats = gateway.get_statistics()
                    logger.info(f"Stats: {stats['devices_total']} devices, "
                              f"{stats['messages_processed']} messages processed")
        else:
            logger.error("Failed to start IoT Gateway")

    except KeyboardInterrupt:
        logger.info("Shutting down IoT Gateway...")
    finally:
        await gateway.stop()

if __name__ == "__main__":
    asyncio.run(main())