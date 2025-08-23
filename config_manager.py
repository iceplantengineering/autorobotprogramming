import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigType(Enum):
    APPLICATION = "application"
    OPERATIONS = "operations"  
    TEMPLATES = "templates"

@dataclass
class ConfigValidationError(Exception):
    config_type: str
    field: str
    message: str

class ConfigManager:
    """設定ファイル管理クラス"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[ConfigType, Dict[str, Any]] = {}
        self.config_files = {
            ConfigType.APPLICATION: "application_config.json",
            ConfigType.OPERATIONS: "handling_operations.json", 
            ConfigType.TEMPLATES: "work_templates.json"
        }
        self._ensure_config_directory()
    
    def _ensure_config_directory(self):
        """設定ディレクトリの確認・作成"""
        self.config_dir.mkdir(exist_ok=True)
        logger.info(f"Config directory: {self.config_dir.absolute()}")
    
    def load_all_configs(self) -> bool:
        """全設定ファイル読み込み"""
        try:
            success_count = 0
            for config_type, filename in self.config_files.items():
                if self.load_config(config_type):
                    success_count += 1
            
            logger.info(f"Loaded {success_count}/{len(self.config_files)} configuration files")
            return success_count == len(self.config_files)
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            return False
    
    def load_config(self, config_type: ConfigType) -> bool:
        """指定設定ファイル読み込み"""
        try:
            filename = self.config_files.get(config_type)
            if not filename:
                logger.error(f"Unknown config type: {config_type}")
                return False
            
            file_path = self.config_dir / filename
            
            if not file_path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 設定検証
            if self._validate_config(config_type, config_data):
                self.configs[config_type] = config_data
                logger.info(f"Loaded config: {filename}")
                return True
            else:
                logger.error(f"Config validation failed: {filename}")
                return False
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filename}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            return False
    
    def save_config(self, config_type: ConfigType) -> bool:
        """設定ファイル保存"""
        try:
            if config_type not in self.configs:
                logger.error(f"Config not loaded: {config_type}")
                return False
            
            filename = self.config_files.get(config_type)
            file_path = self.config_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.configs[config_type], f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved config: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config {config_type}: {e}")
            return False
    
    def get_config(self, config_type: ConfigType) -> Optional[Dict[str, Any]]:
        """設定取得"""
        return self.configs.get(config_type)
    
    def get_config_value(self, config_type: ConfigType, key_path: str, default: Any = None) -> Any:
        """設定値取得（ドット記法対応）"""
        try:
            config = self.configs.get(config_type)
            if not config:
                return default
            
            keys = key_path.split('.')
            value = config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value {key_path}: {e}")
            return default
    
    def set_config_value(self, config_type: ConfigType, key_path: str, value: Any) -> bool:
        """設定値更新（ドット記法対応）"""
        try:
            if config_type not in self.configs:
                logger.error(f"Config not loaded: {config_type}")
                return False
            
            keys = key_path.split('.')
            config = self.configs[config_type]
            
            # 最後のキー以外まで辿る
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # 最後のキーに値を設定
            config[keys[-1]] = value
            
            logger.debug(f"Updated config value: {key_path} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {e}")
            return False
    
    def _validate_config(self, config_type: ConfigType, config_data: Dict[str, Any]) -> bool:
        """設定データ検証"""
        try:
            if config_type == ConfigType.APPLICATION:
                return self._validate_application_config(config_data)
            elif config_type == ConfigType.OPERATIONS:
                return self._validate_operations_config(config_data)
            elif config_type == ConfigType.TEMPLATES:
                return self._validate_templates_config(config_data)
            else:
                logger.warning(f"No validation rules for config type: {config_type}")
                return True
                
        except ConfigValidationError as e:
            logger.error(f"Config validation error: {e.field} - {e.message}")
            return False
        except Exception as e:
            logger.error(f"Config validation exception: {e}")
            return False
    
    def _validate_application_config(self, config: Dict[str, Any]) -> bool:
        """アプリケーション設定検証"""
        required_sections = ['application', 'communication', 'robot', 'tools', 'safety', 'io', 'logging']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError("application", section, f"Required section missing: {section}")
        
        # 通信設定検証
        comm = config['communication']
        if 'tcp_server' not in comm:
            raise ConfigValidationError("application", "communication.tcp_server", "TCP server config missing")
        
        tcp_server = comm['tcp_server']
        if not isinstance(tcp_server.get('port'), int) or not (1 <= tcp_server['port'] <= 65535):
            raise ConfigValidationError("application", "communication.tcp_server.port", "Invalid port number")
        
        return True
    
    def _validate_operations_config(self, config: Dict[str, Any]) -> bool:
        """操作設定検証"""
        required_sections = ['handling_operations', 'welding_operations', 'trajectory_templates']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError("operations", section, f"Required section missing: {section}")
        
        # ハンドリング操作検証
        for op_name, op_config in config['handling_operations'].items():
            if 'parameters' not in op_config:
                raise ConfigValidationError("operations", f"handling_operations.{op_name}.parameters", "Parameters section missing")
        
        return True
    
    def _validate_templates_config(self, config: Dict[str, Any]) -> bool:
        """テンプレート設定検証"""
        required_sections = ['work_templates', 'station_types', 'material_properties']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError("templates", section, f"Required section missing: {section}")
        
        # ワークテンプレート検証
        for template_name, template_config in config['work_templates'].items():
            required_fields = ['name', 'application_type', 'cycle_parameters']
            for field in required_fields:
                if field not in template_config:
                    raise ConfigValidationError("templates", f"work_templates.{template_name}.{field}", f"Required field missing: {field}")
        
        return True
    
    def get_operation_config(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """操作設定取得"""
        operations = self.get_config(ConfigType.OPERATIONS)
        if not operations:
            return None
        
        # ハンドリング操作から検索
        handling_ops = operations.get('handling_operations', {})
        if operation_name in handling_ops:
            return handling_ops[operation_name]
        
        # 溶接操作から検索
        welding_ops = operations.get('welding_operations', {})
        if operation_name in welding_ops:
            return welding_ops[operation_name]
        
        return None
    
    def get_work_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """ワークテンプレート取得"""
        templates = self.get_config(ConfigType.TEMPLATES)
        if not templates:
            return None
        
        work_templates = templates.get('work_templates', {})
        return work_templates.get(template_name)
    
    def get_trajectory_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """軌道テンプレート取得"""
        operations = self.get_config(ConfigType.OPERATIONS)
        if not operations:
            return None
        
        trajectory_templates = operations.get('trajectory_templates', {})
        return trajectory_templates.get(template_name)
    
    def list_available_operations(self) -> List[str]:
        """利用可能操作一覧"""
        operations = self.get_config(ConfigType.OPERATIONS)
        if not operations:
            return []
        
        op_list = []
        op_list.extend(operations.get('handling_operations', {}).keys())
        op_list.extend(operations.get('welding_operations', {}).keys())
        
        return sorted(op_list)
    
    def list_available_templates(self) -> List[str]:
        """利用可能テンプレート一覧"""
        templates = self.get_config(ConfigType.TEMPLATES)
        if not templates:
            return []
        
        work_templates = templates.get('work_templates', {})
        return sorted(work_templates.keys())
    
    def get_robot_limits(self) -> Dict[str, Any]:
        """ロボット制限値取得"""
        return self.get_config_value(ConfigType.APPLICATION, 'robot.limits', {})
    
    def get_tool_config(self, tool_type: str) -> Dict[str, Any]:
        """ツール設定取得"""
        return self.get_config_value(ConfigType.APPLICATION, f'tools.{tool_type}', {})
    
    def get_safety_config(self) -> Dict[str, Any]:
        """安全設定取得"""
        return self.get_config_value(ConfigType.APPLICATION, 'safety', {})
    
    def get_io_signal_mapping(self) -> Dict[str, Dict[str, str]]:
        """I/O信号マッピング取得"""
        return self.get_config_value(ConfigType.APPLICATION, 'io.signal_mapping', {})

# グローバルインスタンス
config_manager = ConfigManager()

def initialize_config_system() -> bool:
    """設定システム初期化"""
    try:
        success = config_manager.load_all_configs()
        if success:
            logger.info("Configuration system initialized successfully")
        else:
            logger.warning("Configuration system initialized with warnings")
        return success
    except Exception as e:
        logger.error(f"Failed to initialize configuration system: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    if initialize_config_system():
        logger.info("Available operations:")
        for op in config_manager.list_available_operations():
            logger.info(f"  - {op}")
        
        logger.info("Available templates:")
        for template in config_manager.list_available_templates():
            logger.info(f"  - {template}")
        
        # 設定値取得例
        tcp_port = config_manager.get_config_value(ConfigType.APPLICATION, 'communication.tcp_server.port')
        logger.info(f"TCP Server Port: {tcp_port}")
        
        gripper_config = config_manager.get_tool_config('gripper')
        logger.info(f"Gripper TCP Offset: {gripper_config.get('tcp_offset', 'Not set')}")
    else:
        logger.error("Configuration system initialization failed")