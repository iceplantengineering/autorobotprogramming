#!/usr/bin/env python3
"""
Advanced IoT Device Manager
Device lifecycle management, discovery, and orchestration system
"""

import asyncio
import json
import logging
import socket
import struct
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import uuid

# Network discovery libraries
try:
    import netifaces
    NETIFACES_AVAILABLE = True
except ImportError:
    NETIFACES_AVAILABLE = False
    print("Warning: netifaces not available. Install with: pip install netifaces")

try:
    import zeroconf
    from zeroconf import ServiceBrowser, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    print("Warning: zeroconf not available. Install with: pip install zeroconf")

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False
    print("Warning: python-nmap not available. Install with: pip install python-nmap")

try:
    import upnpclient
    UPNP_AVAILABLE = True
except ImportError:
    UPNP_AVAILABLE = False
    print("Warning: upnpclient not available. Install with: pip install upnpclient")

try:
    from scapy.all import ARP, Ether, srp, IP, TCP, UDP, ICMP, sr1
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: scapy not available. Install with: pip install scapy")

# Local definitions to avoid circular imports
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeviceManager")

class DiscoveryMethod(Enum):
    """Device discovery methods"""
    PASSIVE_LISTENING = "passive_listening"
    ACTIVE_SCANNING = "active_scanning"
    MDNS_BONJOUR = "mdns_bonjour"
    UPNP_DISCOVERY = "upnp_discovery"
    NETWORK_SCAN = "network_scan"
    ARP_SCAN = "arp_scan"
    PORT_SCAN = "port_scan"
    PROBE_REQUESTS = "probe_requests"
    DEVICE_FINGERPRINTING = "device_fingerprinting"

class DeviceLifecycleState(Enum):
    """Device lifecycle states"""
    DISCOVERED = "discovered"
    IDENTIFIED = "identified"
    PROVISIONED = "provisioned"
    CONFIGURED = "configured"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"
    ERROR = "error"

class DeviceCapability(Enum):
    """Device capabilities"""
    TELEMETRY = "telemetry"
    COMMANDS = "commands"
    CONFIGURATION = "configuration"
    FIRMWARE_UPDATE = "firmware_update"
    LOGGING = "logging"
    TIME_SYNC = "time_sync"
    SECURITY = "security"
    BATCH_OPERATIONS = "batch_operations"
    STREAMING = "streaming"
    EDGE_PROCESSING = "edge_processing"

@dataclass
class DeviceFingerprint:
    """Device fingerprint for identification"""
    mac_address: Optional[str] = None
    ip_addresses: List[str] = field(default_factory=list)
    vendor: Optional[str] = None
    device_type: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    open_ports: List[int] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    http_headers: Dict[str, str] = field(default_factory=dict)
    response_patterns: List[str] = field(default_factory=list)
    mqtt_topics: List[str] = field(default_factory=list)
    unique_identifiers: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 0.0

@dataclass
class DeviceProvisioningConfig:
    """Device provisioning configuration"""
    device_id: str
    name: str
    location: Optional[str] = None
    security_policy: Optional[str] = None
    network_config: Optional[Dict[str, Any]] = None
    telemetry_config: Optional[Dict[str, Any]] = None
    command_config: Optional[Dict[str, Any]] = None
    firmware_config: Optional[Dict[str, Any]] = None
    certificates: Optional[Dict[str, str]] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceHealthMetrics:
    """Device health and performance metrics"""
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    storage_usage: Optional[float] = None
    network_latency: Optional[float] = None
    packet_loss: Optional[float] = None
    signal_strength: Optional[float] = None
    battery_level: Optional[float] = None
    temperature: Optional[float] = None
    uptime: Optional[float] = None
    error_count: int = 0
    warning_count: int = 0
    last_restart: Optional[datetime] = None

class DeviceFingerprinter:
    """Advanced device fingerprinting system"""

    def __init__(self):
        self.fingerprint_rules = {}
        self.vendor_oui_db = self._load_oui_database()
        self.port_service_mapping = self._load_port_services()

    def _load_oui_database(self) -> Dict[str, str]:
        """Load OUI database for vendor identification"""
        # Simplified OUI database - in production, load from file
        return {
            "00:1B:44": "Siemens",
            "00:00:DE": "Siemens",
            "08:00:27": "Oracle",
            "00:0C:29": "VMware",
            "00:50:56": "VMware",
            "00:05:69": "VMware",
            "B8:27:EB": "Raspberry Pi",
            "DC:A6:32": "Raspberry Pi",
            "E4:5F:01": "Raspberry Pi",
            "28:CD:C1": "Raspberry Pi",
            "B8:AE:ED": "Raspberry Pi",
            "00:1C:23": "Beckhoff",
            "00:30:DE": "Beckhoff"
        }

    def _load_port_services(self) -> Dict[int, str]:
        """Load port to service mapping"""
        return {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S",
            1883: "MQTT",
            5683: "CoAP",
            8080: "HTTP-Alt",
            8443: "HTTPS-Alt",
            8883: "MQTT-SSL",
            9001: "MQTT-WS",
            502: "Modbus",
            4840: "OPC-UA",
            2404: "IEC 61850"
        }

    async def fingerprint_device(self, ip_address: str, ports: List[int] = None) -> DeviceFingerprint:
        """Comprehensive device fingerprinting"""
        if ports is None:
            ports = [21, 22, 23, 53, 80, 135, 139, 443, 445, 993, 995, 1723, 3306, 3389, 5432, 5900, 8080, 8443, 1883, 8883, 9001]

        fingerprint = DeviceFingerprint(ip_addresses=[ip_address])

        try:
            # MAC address discovery via ARP
            fingerprint.mac_address = await self._discover_mac_address(ip_address)

            # Vendor identification from MAC
            if fingerprint.mac_address:
                fingerprint.vendor = self._identify_vendor(fingerprint.mac_address)

            # Port scanning
            fingerprint.open_ports = await self._scan_ports(ip_address, ports)

            # Service identification
            fingerprint.services = await self._identify_services(ip_address, fingerprint.open_ports)

            # Protocol detection
            fingerprint.protocols = await self._detect_protocols(ip_address)

            # HTTP fingerprinting
            if 80 in fingerprint.open_ports or 443 in fingerprint.open_ports or 8080 in fingerprint.open_ports:
                fingerprint.http_headers = await self._get_http_fingerprint(ip_address)

            # MQTT fingerprinting
            if 1883 in fingerprint.open_ports or 8883 in fingerprint.open_ports or 9001 in fingerprint.open_ports:
                fingerprint.mqtt_topics = await self._get_mqtt_fingerprint(ip_address)

            # Calculate confidence score
            fingerprint.confidence_score = self._calculate_confidence(fingerprint)

        except Exception as e:
            logger.error(f"Error fingerprinting device {ip_address}: {e}")

        return fingerprint

    async def _discover_mac_address(self, ip_address: str) -> Optional[str]:
        """Discover MAC address for IP"""
        if not SCAPY_AVAILABLE:
            return None

        try:
            # Create ARP request
            arp_request = ARP(pdst=ip_address)
            broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
            arp_request_broadcast = broadcast / arp_request

            # Send and receive packets
            answered_list = srp(arp_request_broadcast, timeout=2, verbose=False)[0]

            if answered_list:
                return answered_list[0][1].hwsrc

        except Exception as e:
            logger.debug(f"Error discovering MAC for {ip_address}: {e}")

        return None

    def _identify_vendor(self, mac_address: str) -> Optional[str]:
        """Identify vendor from MAC address OUI"""
        try:
            # Extract OUI (first 3 octets)
            oui = mac_address.upper().replace(':', '')[:6]
            oui_formatted = ':'.join([oui[i:i+2] for i in range(0, 6, 2)])

            return self.vendor_oui_db.get(oui_formatted)

        except Exception:
            return None

    async def _scan_ports(self, ip_address: str, ports: List[int]) -> List[int]:
        """Scan for open ports"""
        open_ports = []

        for port in ports:
            if await self._is_port_open(ip_address, port):
                open_ports.append(port)

        return open_ports

    async def _is_port_open(self, ip_address: str, port: int, timeout: float = 1.0) -> bool:
        """Check if port is open"""
        try:
            future = asyncio.open_connection(ip_address, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _identify_services(self, ip_address: str, open_ports: List[int]) -> List[str]:
        """Identify services running on open ports"""
        services = []

        for port in open_ports:
            service = self.port_service_mapping.get(port, f"Unknown-{port}")
            services.append(service)

        return services

    async def _detect_protocols(self, ip_address: str) -> List[str]:
        """Detect supported protocols"""
        protocols = []

        # Test for common protocols
        protocol_tests = [
            ("HTTP", self._test_http, [80, 443, 8080, 8443]),
            ("MQTT", self._test_mqtt, [1883, 8883, 9001]),
            ("CoAP", self._test_coap, [5683]),
            ("SSH", self._test_ssh, [22]),
            ("FTP", self._test_ftp, [21]),
            ("Modbus", self._test_modbus, [502]),
            ("OPC-UA", self._test_opcua, [4840])
        ]

        for protocol_name, test_func, ports in protocol_tests:
            for port in ports:
                try:
                    if await test_func(ip_address, port):
                        protocols.append(protocol_name)
                        break
                except Exception:
                    continue

        return protocols

    async def _test_http(self, ip_address: str, port: int) -> bool:
        """Test HTTP protocol"""
        try:
            reader, writer = await asyncio.open_connection(ip_address, port, timeout=2)

            if port == 443 or port == 8443:
                # HTTPS would require SSL context
                writer.close()
                await writer.wait_closed()
                return True
            else:
                # HTTP
                request = f"GET / HTTP/1.1\r\nHost: {ip_address}\r\n\r\n"
                writer.write(request.encode())
                await writer.drain()

                response = await reader.read(1024)
                writer.close()
                await writer.wait_closed()

                return b"HTTP" in response

        except Exception:
            return False

    async def _test_mqtt(self, ip_address: str, port: int) -> bool:
        """Test MQTT protocol"""
        if not MQTT_AVAILABLE:
            return False

        try:
            client = MQTTClient()
            await client.connect(f"mqtt://{ip_address}:{port}", timeout=2)
            await client.disconnect()
            return True

        except Exception:
            return False

    async def _test_coap(self, ip_address: str, port: int) -> bool:
        """Test CoAP protocol"""
        # Simplified CoAP test
        return await self._is_port_open(ip_address, port, timeout=1)

    async def _test_ssh(self, ip_address: str, port: int) -> bool:
        """Test SSH protocol"""
        return await self._is_port_open(ip_address, port, timeout=2)

    async def _test_ftp(self, ip_address: str, port: int) -> bool:
        """Test FTP protocol"""
        return await self._is_port_open(ip_address, port, timeout=2)

    async def _test_modbus(self, ip_address: str, port: int) -> bool:
        """Test Modbus protocol"""
        return await self._is_port_open(ip_address, port, timeout=2)

    async def _test_opcua(self, ip_address: str, port: int) -> bool:
        """Test OPC-UA protocol"""
        return await self._is_port_open(ip_address, port, timeout=2)

    async def _get_http_fingerprint(self, ip_address: str) -> Dict[str, str]:
        """Get HTTP server fingerprint"""
        headers = {}

        try:
            # Try common HTTP ports
            for port in [80, 443, 8080, 8443]:
                try:
                    reader, writer = await asyncio.open_connection(ip_address, port, timeout=3)

                    if port in [443, 8443]:
                        # Would need SSL context for HTTPS
                        headers[f"Server-{port}"] = "HTTPS (SSL/TLS)"
                    else:
                        request = f"GET / HTTP/1.1\r\nHost: {ip_address}\r\nUser-Agent: IoT-Device-Manager\r\n\r\n"
                        writer.write(request.encode())
                        await writer.drain()

                        response = await reader.read(2048)
                        response_text = response.decode('utf-8', errors='ignore')

                        # Extract headers
                        lines = response_text.split('\n')
                        for line in lines:
                            if ':' in line and not line.startswith('HTTP'):
                                key, value = line.split(':', 1)
                                headers[key.strip()] = value.strip()

                    writer.close()
                    await writer.wait_closed()
                    break

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"Error getting HTTP fingerprint for {ip_address}: {e}")

        return headers

    async def _get_mqtt_fingerprint(self, ip_address: str) -> List[str]:
        """Get MQTT broker fingerprint"""
        topics = []

        if not MQTT_AVAILABLE:
            return topics

        try:
            client = MQTTClient()
            await client.connect(f"mqtt://{ip_address}:1883", timeout=2)

            # Try to discover common topics
            common_topics = [
                "devices/+/telemetry",
                "devices/+/status",
                "sensors/+/data",
                "actuators/+/command",
                "$SYS/broker/version",
                "$SYS/broker/uptime"
            ]

            for topic in common_topics:
                try:
                    await client.subscribe([(topic, QOS_0)])
                    await client.unsubscribe([topic])
                    topics.append(topic)
                except Exception:
                    continue

            await client.disconnect()

        except Exception:
            pass

        return topics

    def _calculate_confidence(self, fingerprint: DeviceFingerprint) -> float:
        """Calculate fingerprint confidence score"""
        score = 0.0

        # MAC address and vendor identification
        if fingerprint.mac_address:
            score += 0.2
        if fingerprint.vendor:
            score += 0.1

        # Open ports
        if fingerprint.open_ports:
            score += min(len(fingerprint.open_ports) * 0.05, 0.2)

        # Services identified
        if fingerprint.services:
            score += min(len(fingerprint.services) * 0.05, 0.15)

        # Protocols detected
        if fingerprint.protocols:
            score += min(len(fingerprint.protocols) * 0.1, 0.2)

        # HTTP fingerprinting
        if fingerprint.http_headers:
            score += 0.1

        # MQTT fingerprinting
        if fingerprint.mqtt_topics:
            score += 0.05

        return min(score, 1.0)

class NetworkScanner:
    """Advanced network scanning capabilities"""

    def __init__(self):
        self.interfaces = self._get_network_interfaces()
        self.scan_ranges = self._calculate_scan_ranges()

    def _get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interfaces and their configurations"""
        interfaces = []

        if not NETIFACES_AVAILABLE:
            # Fallback method
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                interfaces.append({
                    'name': 'default',
                    'ip': local_ip,
                    'netmask': '255.255.255.0'
                })
            except Exception:
                pass
        else:
            try:
                for interface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addrs:
                        for addr_info in addrs[netifaces.AF_INET]:
                            interfaces.append({
                                'name': interface,
                                'ip': addr_info['addr'],
                                'netmask': addr_info['netmask']
                            })
            except Exception as e:
                logger.error(f"Error getting network interfaces: {e}")

        return interfaces

    def _calculate_scan_ranges(self) -> List[str]:
        """Calculate network ranges to scan"""
        ranges = []

        for interface in self.interfaces:
            try:
                ip = ipaddress.IPv4Address(interface['ip'])
                netmask = ipaddress.IPv4Address(interface['netmask'])

                # Calculate network
                network = ipaddress.IPv4Network(f"{interface['ip']}/{interface['netmask']}", strict=False)
                ranges.append(str(network))

            except Exception as e:
                logger.error(f"Error calculating network range for {interface}: {e}")

        return ranges

    async def scan_network(self, network_range: str, ports: List[int] = None,
                          max_concurrent: int = 50) -> List[Dict[str, Any]]:
        """Scan network range for active devices"""
        if ports is None:
            ports = [22, 80, 443, 1883, 5683, 8080, 8443]

        try:
            network = ipaddress.IPv4Network(network_range)
            active_hosts = []

            # Create semaphore to limit concurrent scans
            semaphore = asyncio.Semaphore(max_concurrent)

            async def scan_host(ip_address: str):
                async with semaphore:
                    if await self._is_host_alive(ip_address):
                        open_ports = await self._scan_host_ports(ip_address, ports)
                        return {
                            'ip': ip_address,
                            'open_ports': open_ports,
                            'alive': True
                        }
                    return None

            # Scan all hosts in network
            tasks = [scan_host(str(ip)) for ip in network.hosts()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, dict) and result:
                    active_hosts.append(result)

            logger.info(f"Network scan completed: {len(active_hosts)} active hosts found")
            return active_hosts

        except Exception as e:
            logger.error(f"Error scanning network {network_range}: {e}")
            return []

    async def _is_host_alive(self, ip_address: str) -> bool:
        """Check if host is alive using ping"""
        try:
            # Simple TCP connection test as ping alternative
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip_address, 80),
                timeout=2
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _scan_host_ports(self, ip_address: str, ports: List[int]) -> List[int]:
        """Scan specific ports on host"""
        open_ports = []

        for port in ports:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip_address, port),
                    timeout=1
                )
                writer.close()
                await writer.wait_closed()
                open_ports.append(port)
            except Exception:
                continue

        return open_ports

class DeviceLifecycleManager:
    """Device lifecycle management system"""

    def __init__(self, device_registry, mqtt_broker):
        self.device_registry = device_registry
        self.mqtt_broker = mqtt_broker
        self.lifecycle_rules = self._load_lifecycle_rules()
        self.provisioning_queue = asyncio.Queue()
        self.decommission_queue = asyncio.Queue()

    def _load_lifecycle_rules(self) -> Dict[str, Any]:
        """Load device lifecycle rules"""
        return {
            'auto_provisioning': {
                'enabled': True,
                'trusted_vendors': ['Siemens', 'Beckhoff', 'Raspberry Pi'],
                'allowed_protocols': ['mqtt', 'http', 'coap'],
                'required_capabilities': ['telemetry'],
                'security_policy': 'default'
            },
            'health_monitoring': {
                'enabled': True,
                'check_interval': 300,  # 5 minutes
                'offline_threshold': 600,  # 10 minutes
                'error_threshold': 5,
                'auto_recovery': True
            },
            'firmware_management': {
                'auto_update': False,
                'update_window': '02:00-04:00',
                'rollback_enabled': True,
                'max_retries': 3
            }
        }

    async def process_lifecycle_event(self, device_id: str, event: str, data: Dict[str, Any]) -> None:
        """Process device lifecycle event"""
        try:
            device = self.device_registry.get_device(device_id)
            if not device:
                logger.error(f"Device {device_id} not found for lifecycle event")
                return

            logger.info(f"Processing lifecycle event {event} for device {device_id}")

            # Handle different lifecycle events
            if event == "discovered":
                await self._handle_discovered(device, data)
            elif event == "connected":
                await self._handle_connected(device, data)
            elif event == "disconnected":
                await self._handle_disconnected(device, data)
            elif event == "error":
                await self._handle_error(device, data)
            elif event == "maintenance_required":
                await self._handle_maintenance_required(device, data)
            elif event == "firmware_update_available":
                await self._handle_firmware_update(device, data)

        except Exception as e:
            logger.error(f"Error processing lifecycle event: {e}")

    async def _handle_discovered(self, device, data: Dict[str, Any]) -> None:
        """Handle device discovered event"""
        fingerprint = data.get('fingerprint')
        if fingerprint:
            # Auto-provisioning check
            if await self._should_auto_provision(device, fingerprint):
                provisioning_config = self._create_provisioning_config(device, fingerprint)
                await self.provisioning_queue.put((device.device_id, provisioning_config))

    async def _handle_connected(self, device, data: Dict[str, Any]) -> None:
        """Handle device connected event"""
        await self.device_registry.update_device_status(device.device_id, DeviceStatus.ONLINE)

        # Send welcome message and configuration
        await self.mqtt_broker.publish_message(
            f"devices/{device.device_id}/welcome",
            {"message": "Welcome to IoT Gateway", "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    async def _handle_disconnected(self, device, data: Dict[str, Any]) -> None:
        """Handle device disconnected event"""
        await self.device_registry.update_device_status(device.device_id, DeviceStatus.OFFLINE)

    async def _handle_error(self, device, data: Dict[str, Any]) -> None:
        """Handle device error event"""
        await self.device_registry.update_device_status(device.device_id, DeviceStatus.ERROR)

        # Log error and potentially trigger recovery
        error_info = data.get('error', {})
        logger.error(f"Device {device.device_id} error: {error_info}")

        # Auto-recovery if enabled
        if self.lifecycle_rules['health_monitoring']['auto_recovery']:
            await self._attempt_device_recovery(device, error_info)

    async def _handle_maintenance_required(self, device, data: Dict[str, Any]) -> None:
        """Handle maintenance required event"""
        await self.device_registry.update_device_status(device.device_id, DeviceStatus.MAINTENANCE)

    async def _handle_firmware_update(self, device, data: Dict[str, Any]) -> None:
        """Handle firmware update event"""
        if self.lifecycle_rules['firmware_management']['auto_update']:
            await self._schedule_firmware_update(device, data)

    async def _should_auto_provision(self, device, fingerprint: DeviceFingerprint) -> bool:
        """Check if device should be auto-provisioned"""
        rules = self.lifecycle_rules['auto_provisioning']

        if not rules['enabled']:
            return False

        # Check trusted vendor
        if fingerprint.vendor not in rules['trusted_vendors']:
            return False

        # Check allowed protocols
        if not any(proto in rules['allowed_protocols'] for proto in fingerprint.protocols):
            return False

        return True

    def _create_provisioning_config(self, device, fingerprint: DeviceFingerprint) -> DeviceProvisioningConfig:
        """Create provisioning configuration for device"""
        return DeviceProvisioningConfig(
            device_id=device.device_id,
            name=f"Auto-{fingerprint.vendor or 'Unknown'}-{device.device_id[:8]}",
            location=f"Auto-detected",
            security_policy=self.lifecycle_rules['auto_provisioning']['security_policy'],
            telemetry_config={
                'interval': 60,
                'enabled': True,
                'topics': ['telemetry', 'status', 'health']
            },
            command_config={
                'enabled': True,
                'timeout': 30
            }
        )

    async def _attempt_device_recovery(self, device, error_info: Dict[str, Any]) -> None:
        """Attempt automatic device recovery"""
        try:
            recovery_commands = [
                {'command': 'restart', 'delay': 10},
                {'command': 'reset_connection', 'delay': 5},
                {'command': 'clear_errors', 'delay': 2}
            ]

            for command in recovery_commands:
                # Send recovery command
                await self.mqtt_broker.publish_message(
                    f"devices/{device.device_id}/commands/{command['command']}",
                    {'timestamp': datetime.now(timezone.utc).isoformat()}
                )

                # Wait for command to execute
                await asyncio.sleep(command['delay'])

                # Check if device recovered
                updated_device = self.device_registry.get_device(device.device_id)
                if updated_device and updated_device.status == DeviceStatus.ONLINE:
                    logger.info(f"Device {device.device_id} recovered with command {command['command']}")
                    return

            logger.warning(f"Automatic recovery failed for device {device.device_id}")

        except Exception as e:
            logger.error(f"Error during device recovery: {e}")

    async def _schedule_firmware_update(self, device, data: Dict[str, Any]) -> None:
        """Schedule firmware update for device"""
        # Implementation for firmware update scheduling
        pass

class AdvancedDeviceDiscovery:
    """Advanced device discovery system"""

    def __init__(self, device_registry, mqtt_broker):
        self.device_registry = device_registry
        self.mqtt_broker = mqtt_broker
        self.fingerprinter = DeviceFingerprinter()
        self.network_scanner = NetworkScanner()
        self.lifecycle_manager = DeviceLifecycleManager(device_registry, mqtt_broker)

        self.discovery_methods = {
            DiscoveryMethod.MDNS_BONJOUR: self._mdns_discovery,
            DiscoveryMethod.UPNP_DISCOVERY: self._upnp_discovery,
            DiscoveryMethod.NETWORK_SCAN: self._network_scan_discovery,
            DiscoveryMethod.PASSIVE_LISTENING: self._passive_listening,
            DiscoveryMethod.DEVICE_FINGERPRINTING: self._device_fingerprinting
        }

        self.discovered_devices = {}
        self.discovery_stats = {
            'total_discovered': 0,
            'successfully_registered': 0,
            'failed_registrations': 0,
            'last_scan_time': None
        }

    async def start_discovery(self, config: Dict[str, Any] = None) -> None:
        """Start comprehensive device discovery"""
        if config is None:
            config = {
                'enabled_methods': [
                    DiscoveryMethod.NETWORK_SCAN,
                    DiscoveryMethod.DEVICE_FINGERPRINTING,
                    DiscoveryMethod.PASSIVE_LISTENING
                ],
                'scan_interval': 300,  # 5 minutes
                'fingerprint_enabled': True,
                'auto_register': True
            }

        logger.info("Starting advanced device discovery")

        # Start background tasks for each discovery method
        tasks = []
        for method in config['enabled_methods']:
            if method in self.discovery_methods:
                task = asyncio.create_task(
                    self._run_discovery_method(method, config)
                )
                tasks.append(task)

        # Start lifecycle manager
        lifecycle_task = asyncio.create_task(
            self._process_lifecycle_events()
        )
        tasks.append(lifecycle_task)

        # Wait for all tasks (they should run indefinitely)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_discovery_method(self, method: DiscoveryMethod, config: Dict[str, Any]) -> None:
        """Run specific discovery method continuously"""
        discovery_func = self.discovery_methods[method]

        while True:
            try:
                logger.info(f"Running {method.value} discovery")
                discovered = await discovery_func(config)

                if discovered:
                    await self._process_discovered_devices(discovered, config)

                self.discovery_stats['last_scan_time'] = datetime.now(timezone.utc)

                # Wait before next scan
                await asyncio.sleep(config.get('scan_interval', 300))

            except Exception as e:
                logger.error(f"Error in {method.value} discovery: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _mdns_discovery(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """mDNS/Bonjour device discovery"""
        discovered = []

        if not ZEROCONF_AVAILABLE:
            return discovered

        try:
            zeroconf = Zeroconf()

            class DeviceListener:
                def __init__(self):
                    self.devices = []

                def add_service(self, zeroconf, type, name):
                    info = zeroconf.get_service_info(type, name)
                    if info:
                        device_info = {
                            'name': name,
                            'type': type,
                            'addresses': [socket.inet_ntoa(addr) for addr in info.addresses],
                            'port': info.port,
                            'properties': {k.decode(): v.decode() for k, v in info.properties.items()},
                            'discovery_method': DiscoveryMethod.MDNS_BONJOUR
                        }
                        self.devices.append(device_info)

            listener = DeviceListener()

            # Browse for common service types
            service_types = [
                "_http._tcp.local.",
                "_https._tcp.local.",
                "_mqtt._tcp.local.",
                "_coap._udp.local.",
                "_iot._tcp.local.",
                "_homekit._tcp.local."
            ]

            browsers = []
            for service_type in service_types:
                browser = ServiceBrowser(zeroconf, service_type, listener)
                browsers.append(browser)

            # Wait for discovery
            await asyncio.sleep(30)

            # Cleanup
            for browser in browsers:
                browser.cancel()
            zeroconf.close()

            discovered = listener.devices
            logger.info(f"mDNS discovery found {len(discovered)} devices")

        except Exception as e:
            logger.error(f"Error in mDNS discovery: {e}")

        return discovered

    async def _upnp_discovery(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """UPnP device discovery"""
        discovered = []

        if not UPNP_AVAILABLE:
            return discovered

        try:
            devices = upnpclient.discover()

            for device in devices:
                device_info = {
                    'name': device.friendly_name,
                    'manufacturer': device.manufacturer,
                    'model_name': device.model_name,
                    'location': device.location,
                    'udn': device.udn,
                    'device_type': device.device_type,
                    'services': [service.serviceType for service in device.services],
                    'discovery_method': DiscoveryMethod.UPNP_DISCOVERY
                }
                discovered.append(device_info)

            logger.info(f"UPnP discovery found {len(discovered)} devices")

        except Exception as e:
            logger.error(f"Error in UPnP discovery: {e}")

        return discovered

    async def _network_scan_discovery(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Network scanning discovery"""
        discovered = []

        try:
            # Scan each network range
            for network_range in self.network_scanner.scan_ranges:
                logger.info(f"Scanning network range: {network_range}")

                active_hosts = await self.network_scanner.scan_network(
                    network_range,
                    ports=config.get('scan_ports', [22, 80, 443, 1883, 5683, 8080]),
                    max_concurrent=config.get('max_concurrent', 50)
                )

                for host in active_hosts:
                    host['discovery_method'] = DiscoveryMethod.NETWORK_SCAN
                    discovered.append(host)

            logger.info(f"Network scan discovery found {len(discovered)} devices")

        except Exception as e:
            logger.error(f"Error in network scan discovery: {e}")

        return discovered

    async def _passive_listening(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Passive listening for device announcements"""
        discovered = []

        try:
            # Listen for MQTT device announcements
            if MQTT_AVAILABLE:
                client = MQTTClient()
                await client.connect(f"mqtt://{config.get('mqtt_host', 'localhost')}:1883")

                # Subscribe to device announcement topics
                topics = [
                    ("devices/+/announce", QOS_0),
                    ("iot/+/hello", QOS_0),
                    ("hass/status", QOS_0)
                ]

                await client.subscribe(topics)

                # Listen for announcements
                for _ in range(60):  # Listen for 1 minute
                    try:
                        message = await client.deliver_message(timeout=1)
                        topic = message.topic
                        payload = json.loads(message.data.decode())

                        device_info = {
                            'topic': topic,
                            'payload': payload,
                            'discovery_method': DiscoveryMethod.PASSIVE_LISTENING
                        }
                        discovered.append(device_info)

                    except asyncio.TimeoutError:
                        continue

                await client.disconnect()

            logger.info(f"Passive listening found {len(discovered)} device announcements")

        except Exception as e:
            logger.error(f"Error in passive listening: {e}")

        return discovered

    async def _device_fingerprinting(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Device fingerprinting for known devices"""
        discovered = []

        try:
            # Get existing devices that need fingerprinting
            existing_devices = self.device_registry.get_all_devices()

            for device in existing_devices:
                if not device.metadata.get('fingerprinted', False):
                    # Get device IP addresses
                    ips = device.metadata.get('ip_addresses', [])

                    for ip in ips:
                        fingerprint = await self.fingerprinter.fingerprint_device(ip)

                        device_info = {
                            'device_id': device.device_id,
                            'ip_address': ip,
                            'fingerprint': fingerprint,
                            'discovery_method': DiscoveryMethod.DEVICE_FINGERPRINTING
                        }
                        discovered.append(device_info)

            logger.info(f"Device fingerprinting processed {len(discovered)} devices")

        except Exception as e:
            logger.error(f"Error in device fingerprinting: {e}")

        return discovered

    async def _process_discovered_devices(self, discovered: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Process discovered devices"""
        for device_info in discovered:
            try:
                device_id = await self._create_or_update_device(device_info)

                if device_id and config.get('auto_register', True):
                    # Trigger lifecycle event
                    await self.lifecycle_manager.process_lifecycle_event(
                        device_id, 'discovered', device_info
                    )

                    self.discovery_stats['successfully_registered'] += 1
                else:
                    self.discovery_stats['failed_registrations'] += 1

                self.discovery_stats['total_discovered'] += 1

            except Exception as e:
                logger.error(f"Error processing discovered device: {e}")
                self.discovery_stats['failed_registrations'] += 1

    async def _create_or_update_device(self, device_info: Dict[str, Any]) -> Optional[str]:
        """Create or update device from discovery info"""
        try:
            # Generate device ID
            device_id = self._generate_device_id(device_info)

            # Check if device already exists
            existing_device = self.device_registry.get_device(device_id)

            if existing_device:
                # Update existing device
                await self._update_device_from_discovery(existing_device, device_info)
                return device_id
            else:
                # Create new device
                new_device = await self._create_device_from_discovery(device_id, device_info)

                if self.device_registry.register_device(new_device):
                    return device_id

            return None

        except Exception as e:
            logger.error(f"Error creating/updating device: {e}")
            return None

    def _generate_device_id(self, device_info: Dict[str, Any]) -> str:
        """Generate unique device ID from discovery info"""
        # Use MAC address if available
        if 'fingerprint' in device_info and device_info['fingerprint'].mac_address:
            return device_info['fingerprint'].mac_address.replace(':', '-')

        # Use IP address if available
        if 'ip_address' in device_info:
            return f"ip-{device_info['ip_address'].replace('.', '-')}"

        # Use name if available
        if 'name' in device_info:
            return device_info['name'].replace(' ', '-').lower()

        # Generate random ID
        return str(uuid.uuid4())[:8]

    async def _update_device_from_discovery(self, device, device_info: Dict[str, Any]) -> None:
        """Update existing device from discovery info"""
        # Update last seen
        device.last_seen = datetime.now(timezone.utc)
        device.updated_at = datetime.now(timezone.utc)

        # Update status
        device.status = DeviceStatus.ONLINE

        # Add discovery metadata
        device.metadata.update({
            'discovery_method': device_info.get('discovery_method').value,
            'last_discovery': datetime.now(timezone.utc).isoformat()
        })

        # Add fingerprint if available
        if 'fingerprint' in device_info:
            device.metadata['fingerprint'] = device_info['fingerprint'].__dict__
            device.metadata['fingerprinted'] = True

    async def _create_device_from_discovery(self, device_id: str, device_info: Dict[str, Any]) -> Any:
        """Create new device from discovery info"""
        from comprehensive_iot_gateway import IoTDevice, DeviceType, ProtocolType, DeviceStatus

        # Determine device type and protocol from discovery info
        device_type = self._determine_device_type(device_info)
        protocol = self._determine_protocol(device_info)

        # Create device
        device = IoTDevice(
            device_id=device_id,
            name=device_info.get('name', f"Device-{device_id[:8]}"),
            device_type=device_type,
            protocol=protocol,
            connection_string=self._create_connection_string(device_info),
            manufacturer=device_info.get('manufacturer'),
            model_name=device_info.get('model_name'),
            status=DeviceStatus.ONLINE,
            location=device_info.get('location'),
            metadata={
                'discovery_method': device_info.get('discovery_method').value,
                'discovery_time': datetime.now(timezone.utc).isoformat(),
                'auto_discovered': True
            }
        )

        # Add fingerprint if available
        if 'fingerprint' in device_info:
            device.metadata['fingerprint'] = device_info['fingerprint'].__dict__
            device.metadata['fingerprinted'] = True

        return device

    def _determine_device_type(self, device_info: Dict[str, Any]) -> DeviceType:
        """Determine device type from discovery info"""
        # Use fingerprint to determine type
        if 'fingerprint' in device_info:
            fingerprint = device_info['fingerprint']

            # Check for specific services/ports
            if 1883 in fingerprint.open_ports or 8883 in fingerprint.open_ports:
                return DeviceType.GATEWAY
            elif 502 in fingerprint.open_ports:
                return DeviceType.PLC
            elif 4840 in fingerprint.open_ports:
                return DeviceType.CONTROLLER
            elif any('camera' in service.lower() for service in fingerprint.services):
                return DeviceType.CAMERA

        # Use service type information
        services = device_info.get('services', [])
        if any('mqtt' in service.lower() for service in services):
            return DeviceType.GATEWAY
        elif any('plc' in service.lower() for service in services):
            return DeviceType.PLC

        # Default to sensor
        return DeviceType.SENSOR

    def _determine_protocol(self, device_info: Dict[str, Any]) -> ProtocolType:
        """Determine device protocol from discovery info"""
        # Use fingerprint to determine protocol
        if 'fingerprint' in device_info:
            fingerprint = device_info['fingerprint']

            if 'MQTT' in fingerprint.protocols:
                return ProtocolType.MQTT
            elif 'CoAP' in fingerprint.protocols:
                return ProtocolType.COAP
            elif 'HTTP' in fingerprint.protocols:
                return ProtocolType.HTTP
            elif 'OPC-UA' in fingerprint.protocols:
                return ProtocolType.OPC_UA

        # Use port information
        open_ports = device_info.get('open_ports', [])
        if 1883 in open_ports or 8883 in open_ports or 9001 in open_ports:
            return ProtocolType.MQTT
        elif 5683 in open_ports:
            return ProtocolType.COAP
        elif 80 in open_ports or 443 in open_ports or 8080 in open_ports:
            return ProtocolType.HTTP
        elif 502 in open_ports:
            return ProtocolType.MODBUS
        elif 4840 in open_ports:
            return ProtocolType.OPC_UA

        # Default to MQTT
        return ProtocolType.MQTT

    def _create_connection_string(self, device_info: Dict[str, Any]) -> str:
        """Create connection string from discovery info"""
        if 'ip_address' in device_info:
            return f"{device_info['ip_address']}"
        elif 'location' in device_info:
            return device_info['location']
        else:
            return "auto-discovered"

    async def _process_lifecycle_events(self) -> None:
        """Process device lifecycle events"""
        while True:
            try:
                # Process provisioning queue
                while not self.lifecycle_manager.provisioning_queue.empty():
                    device_id, config = await self.lifecycle_manager.provisioning_queue.get()
                    await self._provision_device(device_id, config)

                # Process decommission queue
                while not self.lifecycle_manager.decommission_queue.empty():
                    device_id = await self.lifecycle_manager.decommission_queue.get()
                    await self._decommission_device(device_id)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error processing lifecycle events: {e}")
                await asyncio.sleep(30)

    async def _provision_device(self, device_id: str, config: DeviceProvisioningConfig) -> None:
        """Provision device with configuration"""
        try:
            logger.info(f"Provisioning device {device_id}")

            # Update device configuration
            device = self.device_registry.get_device(device_id)
            if device:
                device.name = config.name
                device.location = config.location
                device.configuration = {
                    'telemetry': config.telemetry_config,
                    'commands': config.command_config,
                    'network': config.network_config,
                    'security': config.security_policy
                }

                # Send provisioning configuration to device
                await self.mqtt_broker.publish_message(
                    f"devices/{device_id}/provision",
                    {
                        'config': config.__dict__,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )

                logger.info(f"Device {device_id} provisioned successfully")

        except Exception as e:
            logger.error(f"Error provisioning device {device_id}: {e}")

    async def _decommission_device(self, device_id: str) -> None:
        """Decommission device"""
        try:
            logger.info(f"Decommissioning device {device_id}")

            # Send decommission command
            await self.mqtt_broker.publish_message(
                f"devices/{device_id}/decommission",
                {'timestamp': datetime.now(timezone.utc).isoformat()}
            )

            # Update device status
            device = self.device_registry.get_device(device_id)
            if device:
                device.status = DeviceStatus.OFFLINE

            logger.info(f"Device {device_id} decommissioned")

        except Exception as e:
            logger.error(f"Error decommissioning device {device_id}: {e}")

    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        return {
            **self.discovery_stats,
            'active_methods': [method.value for method in self.discovery_methods.keys()],
            'total_registered_devices': len(self.device_registry.get_all_devices()),
            'devices_by_type': {
                device_type.value: len(self.device_registry.get_devices_by_type(device_type))
                for device_type in DeviceType
            }
        }

# Factory function
def create_device_discovery(device_registry, mqtt_broker) -> AdvancedDeviceDiscovery:
    """Create advanced device discovery system"""
    return AdvancedDeviceDiscovery(device_registry, mqtt_broker)