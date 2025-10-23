#!/usr/bin/env python3
"""
IoT Integration Demonstration
Complete demonstration of enhanced IoT/MQTT capabilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any

# Import our IoT components
from comprehensive_iot_gateway import create_iot_gateway, IoTDevice, DeviceType, ProtocolType, DeviceStatus
from iot_device_manager import create_device_discovery, DiscoveryMethod
from iot_protocol_adapter import create_protocol_adapter, create_default_protocol_configs, IoTMessage, IoTProtocol

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IoTDemo")

class IoTIntegrationDemo:
    """Complete IoT integration demonstration"""

    def __init__(self):
        self.iot_gateway = None
        self.device_discovery = None
        self.protocol_adapter = None
        self.demo_devices = []
        self.running = False

    async def start_demo(self) -> None:
        """Start the complete IoT demo"""
        logger.info("Starting Comprehensive IoT Integration Demo")

        try:
            # 1. Start IoT Gateway with enhanced MQTT broker
            await self._start_iot_gateway()

            # 2. Initialize Protocol Adapter with multiple protocols
            await self._start_protocol_adapter()

            # 3. Start Advanced Device Discovery
            await self._start_device_discovery()

            # 4. Create demo devices
            await self._create_demo_devices()

            # 5. Demonstrate multi-protocol communication
            await self._demonstrate_protocols()

            # 6. Show device lifecycle management
            await self._demonstrate_lifecycle()

            # 7. Display comprehensive statistics
            await self._show_statistics()

            logger.info("IoT Integration Demo completed successfully!")

        except Exception as e:
            logger.error(f"Error in demo: {e}")
        finally:
            await self._cleanup()

    async def _start_iot_gateway(self) -> None:
        """Start IoT Gateway with enhanced MQTT broker"""
        logger.info("Starting IoT Gateway with enhanced MQTT broker...")

        config = {
            'mqtt': {
                'host': '0.0.0.0',
                'port': 1883,
                'ws_host': '0.0.0.0',
                'ws_port': 9001,
                'allow_anonymous': True,
                'ssl_enabled': False
            },
            'discovery_interval': 60,
            'database_path': 'iot_demo.db'
        }

        self.iot_gateway = create_iot_gateway(config)

        if await self.iot_gateway.start():
            logger.info("IoT Gateway started successfully")
            logger.info(f"   - MQTT Broker: {config['mqtt']['host']}:{config['mqtt']['port']}")
            logger.info(f"   - WebSocket: {config['mqtt']['ws_host']}:{config['mqtt']['ws_port']}")
        else:
            raise Exception("Failed to start IoT Gateway")

    async def _start_protocol_adapter(self) -> None:
        """Start multi-protocol adapter"""
        logger.info("🔌 Starting Multi-Protocol Adapter...")

        self.protocol_adapter = create_protocol_adapter()

        # Register default protocols
        configs = create_default_protocol_configs()
        for config in configs:
            if self.protocol_adapter.register_protocol(config):
                logger.info(f"   ✅ Registered: {config.protocol.value}")
            else:
                logger.warning(f"   ❌ Failed to register: {config.protocol.value}")

        # Start the adapter
        if await self.protocol_adapter.start():
            logger.info("✅ Protocol Adapter started successfully")
        else:
            raise Exception("Failed to start Protocol Adapter")

    async def _start_device_discovery(self) -> None:
        """Start advanced device discovery"""
        logger.info("🔍 Starting Advanced Device Discovery...")

        self.device_discovery = create_device_discovery(
            self.iot_gateway.device_registry,
            self.iot_gateway.mqtt_broker
        )

        # Start discovery in background
        discovery_task = asyncio.create_task(
            self.device_discovery.start_discovery({
                'enabled_methods': [
                    DiscoveryMethod.NETWORK_SCAN,
                    DiscoveryMethod.DEVICE_FINGERPRINTING,
                    DiscoveryMethod.PASSIVE_LISTENING
                ],
                'scan_interval': 120,  # 2 minutes for demo
                'fingerprint_enabled': True,
                'auto_register': True
            })
        )

        logger.info("✅ Device Discovery started")

    async def _create_demo_devices(self) -> None:
        """Create demo IoT devices"""
        logger.info("🏭 Creating Demo IoT Devices...")

        demo_device_configs = [
            {
                'device_id': 'robot-arm-001',
                'name': 'Industrial Robot Arm 1',
                'device_type': DeviceType.INDUSTRIAL_ROBOT,
                'protocol': ProtocolType.MQTT,
                'connection_string': '192.168.1.100',
                'manufacturer': 'Siemens',
                'model': 'SR-210',
                'capabilities': ['telemetry', 'commands', 'configuration'],
                'topics': ['robot/status', 'robot/telemetry', 'robot/commands']
            },
            {
                'device_id': 'temp-sensor-001',
                'name': 'Temperature Sensor 1',
                'device_type': DeviceType.TEMPERATURE_SENSOR,
                'protocol': ProtocolType.COAP,
                'connection_string': '192.168.1.101',
                'manufacturer': 'Sensirion',
                'model': 'STC31',
                'capabilities': ['telemetry'],
                'topics': ['sensor/temperature']
            },
            {
                'device_id': 'plc-controller-001',
                'name': 'PLC Controller 1',
                'device_type': DeviceType.PLC,
                'protocol': ProtocolType.MODBUS,
                'connection_string': '192.168.1.102',
                'manufacturer': 'Beckhoff',
                'model': 'CX2040',
                'capabilities': ['telemetry', 'commands'],
                'topics': ['plc/status', 'plc/data']
            },
            {
                'device_id': 'camera-001',
                'name': 'Vision Camera 1',
                'device_type': DeviceType.CAMERA,
                'protocol': ProtocolType.HTTP,
                'connection_string': '192.168.1.103',
                'manufacturer': 'Basler',
                'model': 'ace-2',
                'capabilities': ['telemetry', 'streaming'],
                'topics': ['camera/status', 'camera/image']
            },
            {
                'device_id': 'gateway-001',
                'name': 'IoT Gateway 1',
                'device_type': DeviceType.GATEWAY,
                'protocol': ProtocolType.MQTT,
                'connection_string': '192.168.1.104',
                'manufacturer': 'Raspberry Pi',
                'model': '4B',
                'capabilities': ['telemetry', 'commands', 'edge_processing'],
                'topics': ['gateway/status', 'gateway/telemetry']
            }
        ]

        for device_config in demo_device_configs:
            device = IoTDevice(
                device_id=device_config['device_id'],
                name=device_config['name'],
                device_type=device_config['device_type'],
                protocol=device_config['protocol'],
                connection_string=device_config['connection_string'],
                manufacturer=device_config['manufacturer'],
                model=device_config['model'],
                status=DeviceStatus.ONLINE,
                capabilities=device_config['capabilities'],
                topics=device_config['topics']
            )

            if self.iot_gateway.device_registry.register_device(device):
                self.demo_devices.append(device)
                logger.info(f"   ✅ Registered: {device.name} ({device.device_id})")
            else:
                logger.error(f"   ❌ Failed to register: {device.name}")

        logger.info(f"✅ Created {len(self.demo_devices)} demo devices")

    async def _demonstrate_protocols(self) -> None:
        """Demonstrate multi-protocol communication"""
        logger.info("📡 Demonstrating Multi-Protocol Communication...")

        # Wait a moment for devices to be ready
        await asyncio.sleep(2)

        for device in self.demo_devices[:3]:  # Test first 3 devices
            logger.info(f"   🔄 Testing {device.name} with {device.protocol.value}")

            # Create test message
            message = IoTMessage(
                source="demo-system",
                destination=device.connection_string,
                protocol=IoTProtocol[device.protocol.value.upper()],
                payload={
                    'command': 'status_check',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'request_id': f"demo-{int(time.time())}"
                }
            )

            # Send message via protocol adapter
            success = await self.protocol_adapter.send_message(message)

            if success:
                logger.info(f"      ✅ Message sent via {device.protocol.value}")
            else:
                logger.warning(f"      ⚠️  Failed to send via {device.protocol.value}")

            # Small delay between devices
            await asyncio.sleep(1)

        logger.info("✅ Multi-protocol demonstration completed")

    async def _demonstrate_lifecycle(self) -> None:
        """Demonstrate device lifecycle management"""
        logger.info("🔄 Demonstrating Device Lifecycle Management...")

        # Simulate device events
        device = self.demo_devices[0]  # Use first device

        # Device maintenance event
        logger.info(f"   🔧 Simulating maintenance for {device.name}")
        await self.device_discovery.lifecycle_manager.process_lifecycle_event(
            device.device_id,
            'maintenance_required',
            {'reason': 'Scheduled maintenance', 'duration': 3600}
        )

        await asyncio.sleep(1)

        # Device error event
        logger.info(f"   ❌ Simulating error for {device.name}")
        await self.device_discovery.lifecycle_manager.process_lifecycle_event(
            device.device_id,
            'error',
            {'error_code': 'COMM_LOSS', 'message': 'Communication lost'}
        )

        await asyncio.sleep(1)

        # Device recovery
        logger.info(f"   ✅ Simulating recovery for {device.name}")
        await self.device_discovery.lifecycle_manager.process_lifecycle_event(
            device.device_id,
            'connected',
            {'reconnection_time': datetime.now(timezone.utc).isoformat()}
        )

        logger.info("✅ Device lifecycle demonstration completed")

    async def _show_statistics(self) -> None:
        """Show comprehensive statistics"""
        logger.info("📊 Comprehensive System Statistics:")
        logger.info("=" * 50)

        # IoT Gateway Statistics
        gateway_stats = self.iot_gateway.get_statistics()
        logger.info("🏭 IoT Gateway:")
        logger.info(f"   Devices Registered: {gateway_stats['devices_registered']}")
        logger.info(f"   Messages Processed: {gateway_stats['messages_processed']}")
        logger.info(f"   Commands Executed: {gateway_stats['commands_executed']}")
        logger.info(f"   Uptime: {gateway_stats['uptime']:.1f}s")

        # Device Summary
        device_summary = self.iot_gateway.get_device_summary()
        logger.info("\n📱 Device Summary:")
        logger.info(f"   Total Devices: {device_summary['total_devices']}")
        logger.info(f"   Devices Online: {device_summary['devices_by_status'].get('online', 0)}")
        logger.info(f"   By Type: {device_summary['devices_by_type']}")
        logger.info(f"   By Protocol: {device_summary['devices_by_protocol']}")

        # Protocol Adapter Statistics
        adapter_stats = self.protocol_adapter.get_statistics()
        logger.info("\n🔌 Protocol Adapter:")
        logger.info(f"   Messages Sent: {adapter_stats['messages_sent']}")
        logger.info(f"   Messages Received: {adapter_stats['messages_received']}")
        logger.info(f"   Active Protocols: {list(adapter_stats['protocols_active'].keys())}")

        # Discovery Statistics
        discovery_stats = self.device_discovery.get_discovery_statistics()
        logger.info("\n🔍 Device Discovery:")
        logger.info(f"   Total Discovered: {discovery_stats['total_discovered']}")
        logger.info(f"   Successfully Registered: {discovery_stats['successfully_registered']}")
        logger.info(f"   Active Methods: {discovery_stats['active_methods']}")

        logger.info("=" * 50)

    async def _cleanup(self) -> None:
        """Clean up resources"""
        logger.info("🧹 Cleaning up resources...")

        try:
            if self.iot_gateway:
                await self.iot_gateway.stop()
                logger.info("   ✅ IoT Gateway stopped")

            if self.protocol_adapter:
                await self.protocol_adapter.stop()
                logger.info("   ✅ Protocol Adapter stopped")

            if self.device_discovery:
                self.device_discovery.device_discovery.stop_discovery()
                logger.info("   ✅ Device Discovery stopped")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main demo function"""
    print("Comprehensive IoT Integration Demonstration")
    print("=" * 50)
    print("This demo showcases the enhanced IoT and MQTT capabilities:")
    print("  - Advanced MQTT broker with device management")
    print("  - Multi-protocol support (MQTT, CoAP, HTTP, etc.)")
    print("  - Automatic device discovery and fingerprinting")
    print("  - Device lifecycle management")
    print("  - Real-time telemetry and command processing")
    print("=" * 50)

    demo = IoTIntegrationDemo()
    await demo.start_demo()

if __name__ == "__main__":
    asyncio.run(main())