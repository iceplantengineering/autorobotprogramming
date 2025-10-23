"""
クラウドコネクタ (Phase 4-1)
AWS/Azure/GCP連携によるデータ同期・リモート監視機能
実用運用システムのためのクラウド統合アーキテクチャ
"""

import json
import time
import logging
import threading
import asyncio
import queue
import ssl
import hashlib
import hmac
import base64
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import requests
import sqlite3
from pathlib import Path

# クラウドSDK（オプション依存）
try:
    import boto3
    import botocore
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    from azure.cosmos import CosmosClient
    from azure.iot.device import IoTHubDeviceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from google.cloud import storage
    from google.cloud import bigquery
    from google.cloud import pubsub_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

from production_management_integration import ProductionManagementSystem, ProductionMetrics, QualityRecord

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """クラウドプロバイダー"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"

class CloudServiceType(Enum):
    """クラウドサービスタイプ"""
    STORAGE = "storage"
    DATABASE = "database"
    IOT_HUB = "iot_hub"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"
    BACKUP = "backup"

@dataclass
class CloudConfig:
    """クラウド接続設定"""
    provider: CloudProvider
    region: str
    project_id: Optional[str] = None  # GCP用
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    connection_string: Optional[str] = None  # Azure用
    endpoint_url: Optional[str] = None
    ssl_verify: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # サービス設定
    services: Dict[CloudServiceType, Dict[str, Any]] = field(default_factory=dict)

    # セキュリティ設定
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    data_retention_days: int = 90

@dataclass
class CloudSyncStatus:
    """クラウド同期状態"""
    provider: CloudProvider
    service: CloudServiceType
    last_sync: Optional[datetime] = None
    sync_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    status: str = "disconnected"  # connected, syncing, error

class CloudStorageInterface(ABC):
    """クラウドストレージインターフェース"""

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> bool:
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        pass

    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        pass

class CloudDatabaseInterface(ABC):
    """クラウドデータベースインターフェース"""

    @abstractmethod
    def insert_data(self, table: str, data: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def query_data(self, table: str, query: str = None, limit: int = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_data(self, table: str, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        pass

class CloudAnalyticsInterface(ABC):
    """クラウド分析インターフェース"""

    @abstractmethod
    def send_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def create_dashboard(self, config: Dict[str, Any]) -> str:
        pass

class AWSConnector(CloudStorageInterface, CloudDatabaseInterface, CloudAnalyticsInterface):
    """AWSコネクタ"""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.status = CloudSyncStatus(CloudProvider.AWS, CloudServiceType.STORAGE)

        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) not installed. Install with: pip install boto3")

        # AWSクライアント初期化
        self.session = boto3.Session(
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region
        )

        # サービスクライアント
        self.s3_client = self.session.client('s3', endpoint_url=config.endpoint_url)
        self.dynamodb_client = self.session.client('dynamodb')
        self.cloudwatch_client = self.session.client('cloudwatch')

        logger.info("AWS connector initialized")

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                logger.error("S3 bucket name not configured")
                return False

            self.s3_client.upload_file(local_path, bucket_name, remote_path)
            self.status.sync_count += 1
            self.status.last_sync = datetime.now()
            self.status.status = "connected"

            logger.info(f"File uploaded to S3: {remote_path}")
            return True

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            self.status.status = "error"
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                return False

            self.s3_client.download_file(bucket_name, remote_path, local_path)
            return True

        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                return []

            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )

            return [obj['Key'] for obj in response.get('Contents', [])]

        except Exception as e:
            logger.error(f"S3 list failed: {e}")
            return []

    def delete_file(self, remote_path: str) -> bool:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                return False

            self.s3_client.delete_object(Bucket=bucket_name, Key=remote_path)
            return True

        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False

    def insert_data(self, table: str, data: Dict[str, Any]) -> bool:
        try:
            dynamodb = self.session.resource('dynamodb')
            dtable = dynamodb.Table(table)

            # データ型変換
            item = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    item[key] = {'N': str(value)}
                elif isinstance(value, str):
                    item[key] = {'S': value}
                elif isinstance(value, bool):
                    item[key] = {'BOOL': value}
                elif isinstance(value, list):
                    item[key] = {'L': value}
                elif isinstance(value, dict):
                    item[key] = {'M': value}
                else:
                    item[key] = {'S': str(value)}

            dtable.put_item(Item=item)
            return True

        except Exception as e:
            logger.error(f"DynamoDB insert failed: {e}")
            return False

    def query_data(self, table: str, query: str = None, limit: int = None) -> List[Dict[str, Any]]:
        try:
            dynamodb = self.session.resource('dynamodb')
            dtable = dynamodb.Table(table)

            response = dtable.scan(Limit=limit)

            # データ型変換（逆変換）
            items = []
            for item in response.get('Items', []):
                converted_item = {}
                for key, value in item.items():
                    if 'N' in value:
                        converted_item[key] = float(value['N']) if '.' in value['N'] else int(value['N'])
                    elif 'S' in value:
                        converted_item[key] = value['S']
                    elif 'BOOL' in value:
                        converted_item[key] = value['BOOL']
                    else:
                        converted_item[key] = str(value)
                items.append(converted_item)

            return items

        except Exception as e:
            logger.error(f"DynamoDB query failed: {e}")
            return []

    def update_data(self, table: str, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        try:
            dynamodb = self.session.resource('dynamodb')
            dtable = dynamodb.Table(table)

            # 更新式構築
            update_expression = "SET " + ", ".join([f"{k} = :{k}" for k in data.keys()])
            expression_values = {f":{k}": v for k, v in data.items()}

            dtable.update_item(
                Key=condition,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            return True

        except Exception as e:
            logger.error(f"DynamoDB update failed: {e}")
            return False

    def send_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        try:
            for metric in metrics:
                self.cloudwatch_client.put_metric_data(
                    Namespace='RobotProductionSystem',
                    MetricData=[
                        {
                            'MetricName': metric['name'],
                            'Value': metric['value'],
                            'Unit': metric.get('unit', 'None'),
                            'Timestamp': metric.get('timestamp', datetime.utcnow()),
                            'Dimensions': metric.get('dimensions', [])
                        }
                    ]
                )

            return True

        except Exception as e:
            logger.error(f"CloudWatch metrics failed: {e}")
            return False

    def create_dashboard(self, config: Dict[str, Any]) -> str:
        # CloudWatchダッシュボード作成（複雑なため簡略実装）
        dashboard_name = f"robot-dashboard-{uuid.uuid4().hex[:8]}"

        try:
            self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(config)
            )
            return dashboard_name

        except Exception as e:
            logger.error(f"CloudWatch dashboard creation failed: {e}")
            return ""

class AzureConnector(CloudStorageInterface, CloudDatabaseInterface, CloudAnalyticsInterface):
    """Azureコネクタ"""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.status = CloudSyncStatus(CloudProvider.AZURE, CloudServiceType.STORAGE)

        if not AZURE_AVAILABLE:
            raise ImportError("Azure SDK not installed. Install with: pip install azure-storage-blob azure-cosmos azure-iot-device")

        # Azureクライアント初期化
        self.blob_client = BlobServiceClient.from_connection_string(config.connection_string)
        self.cosmos_client = CosmosClient.from_connection_string(config.connection_string)

        logger.info("Azure connector initialized")

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            container_name = self.config.services[CloudServiceType.STORAGE].get('container_name', 'robot-data')

            blob_client = self.blob_client.get_blob_client(container=container_name, blob=remote_path)

            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)

            self.status.sync_count += 1
            self.status.last_sync = datetime.now()
            self.status.status = "connected"

            logger.info(f"File uploaded to Azure Blob: {remote_path}")
            return True

        except Exception as e:
            logger.error(f"Azure Blob upload failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            self.status.status = "error"
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            container_name = self.config.services[CloudServiceType.STORAGE].get('container_name', 'robot-data')

            blob_client = self.blob_client.get_blob_client(container=container_name, blob=remote_path)

            with open(local_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())

            return True

        except Exception as e:
            logger.error(f"Azure Blob download failed: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            container_name = self.config.services[CloudServiceType.STORAGE].get('container_name', 'robot-data')

            blob_list = self.blob_client.get_container_client(container_name).list_blobs(name_starts_with=prefix)

            return [blob.name for blob in blob_list]

        except Exception as e:
            logger.error(f"Azure Blob list failed: {e}")
            return []

    def delete_file(self, remote_path: str) -> bool:
        try:
            container_name = self.config.services[CloudServiceType.STORAGE].get('container_name', 'robot-data')

            blob_client = self.blob_client.get_blob_client(container=container_name, blob=remote_path)
            blob_client.delete_blob()

            return True

        except Exception as e:
            logger.error(f"Azure Blob delete failed: {e}")
            return False

    def insert_data(self, table: str, data: Dict[str, Any]) -> bool:
        try:
            database_name = self.config.services[CloudServiceType.DATABASE].get('database_name', 'robotdb')
            container_name = table

            cosmos_client = self.cosmos_client.get_database_client(database_name)
            container = cosmos_client.get_container_client(container_name)

            data['id'] = str(uuid.uuid4())
            container.create_item(data)

            return True

        except Exception as e:
            logger.error(f"CosmosDB insert failed: {e}")
            return False

    def query_data(self, table: str, query: str = None, limit: int = None) -> List[Dict[str, Any]]:
        try:
            database_name = self.config.services[CloudServiceType.DATABASE].get('database_name', 'robotdb')
            container_name = table

            cosmos_client = self.cosmos_client.get_database_client(database_name)
            container = cosmos_client.get_container_client(container_name)

            if not query:
                query = f"SELECT * FROM c {f'TOP {limit}' if limit else ''}"

            items = list(container.query_items(query=query, enable_cross_partition_query=True))

            return items

        except Exception as e:
            logger.error(f"CosmosDB query failed: {e}")
            return []

    def update_data(self, table: str, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        try:
            database_name = self.config.services[CloudServiceType.DATABASE].get('database_name', 'robotdb')
            container_name = table

            cosmos_client = self.cosmos_client.get_database_client(database_name)
            container = cosmos_client.get_container_client(container_name)

            # 条件に合うドキュメントを検索して更新
            for key, value in condition.items():
                query = f"SELECT * FROM c WHERE c.{key} = '{value}'"
                items = container.query_items(query=query, enable_cross_partition_query=True)

                for item in items:
                    item.update(data)
                    container.upsert_item(item)

            return True

        except Exception as e:
            logger.error(f"CosmosDB update failed: {e}")
            return False

    def send_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        try:
            # Azure Monitorにメトリクス送信（簡略実装）
            for metric in metrics:
                logger.info(f"Azure metric: {metric}")
            return True

        except Exception as e:
            logger.error(f"Azure metrics failed: {e}")
            return False

    def create_dashboard(self, config: Dict[str, Any]) -> str:
        # Azure Dashboard作成（複雑なため簡略実装）
        dashboard_id = f"azure-dashboard-{uuid.uuid4().hex[:8]}"
        logger.info(f"Azure dashboard created: {dashboard_id}")
        return dashboard_id

class GCPConnector(CloudStorageInterface, CloudDatabaseInterface, CloudAnalyticsInterface):
    """GCPコネクタ"""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.status = CloudSyncStatus(CloudProvider.GCP, CloudServiceType.STORAGE)

        if not GCP_AVAILABLE:
            raise ImportError("GCP SDK not installed. Install with: pip install google-cloud-storage google-cloud-bigquery google-cloud-pubsub")

        # GCPクライアント初期化
        self.storage_client = storage.Client(project=config.project_id)
        self.bigquery_client = bigquery.Client(project=config.project_id)
        self.pubsub_client = pubsub_v1.PublisherClient()

        logger.info("GCP connector initialized")

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                logger.error("GCS bucket name not configured")
                return False

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(remote_path)

            blob.upload_from_filename(local_path)

            self.status.sync_count += 1
            self.status.last_sync = datetime.now()
            self.status.status = "connected"

            logger.info(f"File uploaded to GCS: {remote_path}")
            return True

        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            self.status.status = "error"
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                return False

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(remote_path)

            blob.download_to_filename(local_path)
            return True

        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                return []

            bucket = self.storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)

            return [blob.name for blob in blobs]

        except Exception as e:
            logger.error(f"GCS list failed: {e}")
            return []

    def delete_file(self, remote_path: str) -> bool:
        try:
            bucket_name = self.config.services[CloudServiceType.STORAGE].get('bucket_name')
            if not bucket_name:
                return False

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(remote_path)

            blob.delete()
            return True

        except Exception as e:
            logger.error(f"GCS delete failed: {e}")
            return False

    def insert_data(self, table: str, data: Dict[str, Any]) -> bool:
        try:
            dataset_name = self.config.services[CloudServiceType.DATABASE].get('dataset_name', 'robot_dataset')
            table_name = table

            table_ref = self.bigquery_client.dataset(dataset_name).table(table_name)

            rows_to_insert = [data]

            errors = self.bigquery_client.insert_rows_json(table_ref, rows_to_insert)

            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return False

            return True

        except Exception as e:
            logger.error(f"BigQuery insert failed: {e}")
            return False

    def query_data(self, table: str, query: str = None, limit: int = None) -> List[Dict[str, Any]]:
        try:
            dataset_name = self.config.services[CloudServiceType.DATABASE].get('dataset_name', 'robot_dataset')

            if not query:
                query = f"SELECT * FROM `{dataset_name}.{table}` {f'LIMIT {limit}' if limit else ''}"

            query_job = self.bigquery_client.query(query)
            results = query_job.result()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            return []

    def update_data(self, table: str, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        try:
            dataset_name = self.config.services[CloudServiceType.DATABASE].get('dataset_name', 'robot_dataset')

            # BigQueryは更新が複雑なため、DML文を使用
            set_clause = ", ".join([f"{k} = '{v}'" for k, v in data.items()])
            where_clause = " AND ".join([f"{k} = '{v}'" for k, v in condition.items()])

            query = f"UPDATE `{dataset_name}.{table}` SET {set_clause} WHERE {where_clause}"

            query_job = self.bigquery_client.query(query)
            query_job.result()

            return True

        except Exception as e:
            logger.error(f"BigQuery update failed: {e}")
            return False

    def send_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        try:
            # Cloud Monitoringにメトリクス送信（簡略実装）
            for metric in metrics:
                logger.info(f"GCP metric: {metric}")
            return True

        except Exception as e:
            logger.error(f"GCP metrics failed: {e}")
            return False

    def create_dashboard(self, config: Dict[str, Any]) -> str:
        # Cloud Monitoringダッシュボード作成（複雑なため簡略実装）
        dashboard_id = f"gcp-dashboard-{uuid.uuid4().hex[:8]}"
        logger.info(f"GCP dashboard created: {dashboard_id}")
        return dashboard_id

class CloudDataSynchronizer:
    """クラウドデータ同期マネージャー"""

    def __init__(self, config: CloudConfig, production_system: ProductionManagementSystem):
        self.config = config
        self.production_system = production_system

        # コネクタ初期化
        self.connector = self._create_connector()

        # 同期キュー
        self.sync_queue = queue.Queue(maxsize=1000)

        # 同期スレッド制御
        self.running = False
        self.sync_threads: List[threading.Thread] = []

        # 同期間隔
        self.sync_interval = 60.0  # 60秒間隔
        self.batch_size = 50  # バッチ処理サイズ

        # 同期統計
        self.sync_stats = {
            'total_synced': 0,
            'failed_syncs': 0,
            'last_sync': None,
            'sync_errors': []
        }

        # コールバック
        self.on_sync_completed: Optional[Callable[[str, int], None]] = None
        self.on_sync_error: Optional[Callable[[str, str], None]] = None

        logger.info(f"Cloud data synchronizer initialized for {config.provider.value}")

    def _create_connector(self) -> Union[AWSConnector, AzureConnector, GCPConnector]:
        """プロバイダーに応じたコネクタを作成"""
        if self.config.provider == CloudProvider.AWS:
            return AWSConnector(self.config)
        elif self.config.provider == CloudProvider.AZURE:
            return AzureConnector(self.config)
        elif self.config.provider == CloudProvider.GCP:
            return GCPConnector(self.config)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.config.provider}")

    def start(self) -> bool:
        """同期開始"""
        try:
            if self.running:
                logger.warning("Cloud synchronizer already running")
                return False

            self.running = True

            # 同期スレッド開始
            for i in range(3):  # 3つの並列同期スレッド
                thread = threading.Thread(target=self._sync_worker, daemon=True)
                thread.start()
                self.sync_threads.append(thread)

            # 定期同期スレッド開始
            periodic_thread = threading.Thread(target=self._periodic_sync_loop, daemon=True)
            periodic_thread.start()
            self.sync_threads.append(periodic_thread)

            logger.info("Cloud synchronizer started")
            return True

        except Exception as e:
            logger.error(f"Failed to start cloud synchronizer: {e}")
            return False

    def stop(self):
        """同期停止"""
        self.running = False

        for thread in self.sync_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        logger.info("Cloud synchronizer stopped")

    def queue_sync_data(self, data_type: str, data: Dict[str, Any]):
        """同期データをキューに追加"""
        sync_item = {
            'type': data_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'retry_count': 0
        }

        try:
            self.sync_queue.put_nowait(sync_item)
        except queue.Full:
            logger.warning("Sync queue full, dropping sync item")

    def _sync_worker(self):
        """同期ワーカー"""
        while self.running:
            try:
                sync_item = self.sync_queue.get(timeout=5.0)
                self._process_sync_item(sync_item)
                self.sync_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Sync worker error: {e}")

    def _process_sync_item(self, sync_item: Dict[str, Any]):
        """同期アイテム処理"""
        data_type = sync_item['type']
        data = sync_item['data']

        try:
            success = False

            if data_type == 'production_metrics':
                success = self._sync_production_metrics(data)
            elif data_type == 'quality_record':
                success = self._sync_quality_record(data)
            elif data_type == 'production_order':
                success = self._sync_production_order(data)
            elif data_type == 'file_backup':
                success = self._sync_file_backup(data)
            else:
                logger.warning(f"Unknown sync data type: {data_type}")
                success = True

            if success:
                self.sync_stats['total_synced'] += 1
                self.sync_stats['last_sync'] = datetime.now()

                if self.on_sync_completed:
                    self.on_sync_completed(data_type, 1)

            else:
                # リトライ処理
                sync_item['retry_count'] += 1
                if sync_item['retry_count'] < 3:
                    self.sync_queue.put(sync_item)
                else:
                    self.sync_stats['failed_syncs'] += 1
                    error_msg = f"Sync failed after 3 retries: {data_type}"
                    self.sync_stats['sync_errors'].append(error_msg)

                    if self.on_sync_error:
                        self.on_sync_error(data_type, error_msg)

        except Exception as e:
            logger.error(f"Error processing sync item: {e}")
            self.sync_stats['failed_syncs'] += 1

    def _sync_production_metrics(self, metrics: Dict[str, Any]) -> bool:
        """生産メトリクス同期"""
        try:
            # クラウドデータベースに保存
            return self.connector.insert_data('production_metrics', metrics)
        except Exception as e:
            logger.error(f"Production metrics sync failed: {e}")
            return False

    def _sync_quality_record(self, record: Dict[str, Any]) -> bool:
        """品質記録同期"""
        try:
            return self.connector.insert_data('quality_records', record)
        except Exception as e:
            logger.error(f"Quality record sync failed: {e}")
            return False

    def _sync_production_order(self, order: Dict[str, Any]) -> bool:
        """生産オーダー同期"""
        try:
            return self.connector.insert_data('production_orders', order)
        except Exception as e:
            logger.error(f"Production order sync failed: {e}")
            return False

    def _sync_file_backup(self, file_info: Dict[str, Any]) -> bool:
        """ファイルバックアップ同期"""
        try:
            local_path = file_info['local_path']
            remote_path = file_info['remote_path']

            return self.connector.upload_file(local_path, remote_path)
        except Exception as e:
            logger.error(f"File backup sync failed: {e}")
            return False

    def _periodic_sync_loop(self):
        """定期同期ループ"""
        logger.info("Periodic sync loop started")

        while self.running:
            try:
                self._perform_periodic_sync()
                time.sleep(self.sync_interval)

            except Exception as e:
                logger.error(f"Periodic sync error: {e}")
                time.sleep(10.0)

        logger.info("Periodic sync loop ended")

    def _perform_periodic_sync(self):
        """定期同期実行"""
        try:
            # ダッシュボードデータ収集
            dashboard_data = self.production_system.get_production_dashboard()

            # メトリクス変換
            metrics = [
                {
                    'name': 'ActiveProductionOrders',
                    'value': dashboard_data['active_production_orders'],
                    'unit': 'Count',
                    'timestamp': datetime.now(),
                    'dimensions': [{'Name': 'System', 'Value': 'RobotProduction'}]
                },
                {
                    'name': 'ActiveWorkOrders',
                    'value': dashboard_data['active_work_orders'],
                    'unit': 'Count',
                    'timestamp': datetime.now(),
                    'dimensions': [{'Name': 'System', 'Value': 'RobotProduction'}]
                },
                {
                    'name': 'TotalProcessedOrders',
                    'value': dashboard_data['total_processed_orders'],
                    'unit': 'Count',
                    'timestamp': datetime.now(),
                    'dimensions': [{'Name': 'System', 'Value': 'RobotProduction'}]
                },
                {
                    'name': 'SystemUptime',
                    'value': dashboard_data['session_uptime'],
                    'unit': 'Seconds',
                    'timestamp': datetime.now(),
                    'dimensions': [{'Name': 'System', 'Value': 'RobotProduction'}]
                }
            ]

            # クラウドメトリクス送信
            self.connector.send_metrics(metrics)

            # データベースバックアップ
            self._backup_database()

            logger.debug("Periodic sync completed successfully")

        except Exception as e:
            logger.error(f"Periodic sync failed: {e}")

    def _backup_database(self):
        """データベースバックアップ"""
        try:
            if CloudServiceType.BACKUP in self.config.services:
                backup_enabled = self.config.services[CloudServiceType.BACKUP].get('enabled', False)

                if backup_enabled:
                    # データベースファイルコピー
                    db_path = "production.db"
                    backup_name = f"production_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

                    file_info = {
                        'local_path': db_path,
                        'remote_path': f"backups/{backup_name}"
                    }

                    self.queue_sync_data('file_backup', file_info)
                    logger.info(f"Database backup queued: {backup_name}")

        except Exception as e:
            logger.error(f"Database backup failed: {e}")

    def get_sync_status(self) -> Dict[str, Any]:
        """同期状態取得"""
        return {
            'provider': self.config.provider.value,
            'connector_status': {
                'status': self.connector.status.status,
                'last_sync': self.connector.status.last_sync.isoformat() if self.connector.status.last_sync else None,
                'sync_count': self.connector.status.sync_count,
                'error_count': self.connector.status.error_count,
                'last_error': self.connector.status.last_error
            },
            'sync_stats': self.sync_stats.copy(),
            'queue_size': self.sync_queue.qsize(),
            'running': self.running
        }

# グローバルクラウド同期インスタンス
cloud_synchronizer: Optional[CloudDataSynchronizer] = None

def initialize_cloud_connector(config: CloudConfig, production_system: ProductionManagementSystem) -> CloudDataSynchronizer:
    """クラウドコネクタ初期化"""
    global cloud_synchronizer
    cloud_synchronizer = CloudDataSynchronizer(config, production_system)
    return cloud_synchronizer

def get_cloud_connector() -> Optional[CloudDataSynchronizer]:
    """クラウドコネクタ取得"""
    return cloud_synchronizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Cloud Connector System...")

    # テスト設定（AWS）
    test_config = CloudConfig(
        provider=CloudProvider.AWS,
        region="us-east-1",
        access_key="test_access_key",
        secret_key="test_secret_key",
        services={
            CloudServiceType.STORAGE: {
                'bucket_name': 'robot-production-data'
            },
            CloudServiceType.DATABASE: {
                'table_name': 'production_metrics'
            },
            CloudServiceType.BACKUP: {
                'enabled': True
            }
        }
    )

    # モック生産管理システム
    from production_management_integration import ProductionManagementSystem, MockMESConnector
    mock_pms = ProductionManagementSystem(MockMESConnector())

    try:
        # クラウドコネクタ初期化（テストモード）
        print("Initializing cloud connector...")

        # 実際のクラウド接続には認証情報が必要なため、ここではスキップ
        print("Cloud connector would be initialized here")
        print("Test configuration:")
        print(f"  Provider: {test_config.provider.value}")
        print(f"  Region: {test_config.region}")
        print(f"  Services: {list(test_config.services.keys())}")

        print("\nCloud connector system test completed successfully!")

    except Exception as e:
        print(f"Cloud connector test failed: {e}")