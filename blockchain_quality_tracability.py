"""
ブロックチェーン品質トレーサビリティ (Phase 5-5)
分散型台帳による品質記録・改ざん防止・信頼性確保
スマートコントラクト・品質証明書・サプライチェーン追跡
"""

import json
import time
import logging
import threading
import hashlib
import ecdsa
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import sqlite3
import os
from pathlib import Path

# 暗号通貨ライブラリ
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Web3ライブラリ（オプション）
try:
    from web3 import Web3
    from web3.contract import Contract
    from web3.providers.ethereum import EthereumProvider
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

from production_management_integration import ProductionManagementSystem, QualityRecord

logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """ブロックチェーンタイプ"""
    PRIVATE = "private"  # プライベートブロックチェーン
    ETHEREUM = "ethereum"  # イーサリアム
    HYPERLEDGER = "hyperledger"  # ハイパーレッジャー
    CONSENSUS = "consensus"  # コンセンサス

class QualityStatus(Enum):
    """品質ステータス"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REWORK = "rework"
    CERTIFIED = "certified"

class CertificateType(Enum):
    """証明書タイプ"""
    QUALITY_CERTIFICATE = "quality_certificate"
    INSPECTION_REPORT = "inspection_report"
    MATERIAL_CERTIFICATE = "material_certificate"
    PROCESS_CERTIFICATE = "process_certificate"
    COMPLIANCE_CERTIFICATE = "compliance_certificate"

@dataclass
class QualityTransaction:
    """品質トランザクション"""
    transaction_id: str
    block_number: int
    timestamp: datetime
    product_id: str
    batch_id: str
    quality_data: Dict[str, Any]
    measurements: Dict[str, float]
    inspector_id: str
    inspector_signature: str
    status: QualityStatus
    hash_previous: str
    merkle_root: str
    nonce: int
    verified: bool = False

@dataclass
class QualityBlock:
    """品質ブロック"""
    block_number: int
    timestamp: datetime
    previous_hash: str
    merkle_root: str
    transactions: List[QualityTransaction]
    nonce: int
    difficulty: int
    miner_id: str
    block_hash: str

@dataclass
class QualityCertificate:
    """品質証明書"""
    certificate_id: str
    certificate_type: CertificateType
    product_id: str
    batch_id: str
    issuance_date: datetime
    expiry_date: Optional[datetime]
    issuer_id: str
    issuer_signature: str
    certificate_data: Dict[str, Any]
    blockchain_hash: str
    verified: bool = False
    revoked: bool = False

@dataclass
class SupplyChainNode:
    """サプライチェーンノード"""
    node_id: str
    node_type: str  # supplier, manufacturer, distributor, retailer
    node_name: str
    location: Dict[str, str]
    contact_info: Dict[str, str]
    certifications: List[str]
    public_key: str
    trust_score: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)

class CryptoUtils:
    """暗号化ユーティリティ"""

    @staticmethod
    def generate_key_pair() -> Tuple[bytes, str]:
        """鍵ペア生成"""
        try:
            if CRYPTO_AVAILABLE:
                # ECDSA鍵ペア生成
                private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
                public_key = private_key.public_key()

                private_bytes = private_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=None
                )
                public_bytes = public_key.public_bytes(
                    Encoding.PEM,
                    format=PublicFormat.SubjectPublicKeyInfo
                )

                return private_bytes, public_bytes.decode('utf-8')
            else:
                # 模擬鍵ペア生成
                private_key = os.urandom(32)
                public_key = hashlib.sha256(private_key).hexdigest()
                return private_key, public_key

        except Exception as e:
            logger.error(f"Key pair generation failed: {e}")
            raise

    @staticmethod
    def sign_data(private_key_bytes: bytes, data: str) -> str:
        """データ署名"""
        try:
            if CRYPTO_AVAILABLE:
                private_key = ec.load_pem_private_key(
                    private_key_bytes,
                    password=None,
                    backend=default_backend()
                )

                message = data.encode('utf-8')
                signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
                return base64.b64encode(signature).decode('utf-8')
            else:
                # 模擬署名
                combined = private_key_bytes + data.encode('utf-8')
                return base64.b64encode(hashlib.sha256(combined).digest()).decode('utf-8')

        except Exception as e:
            logger.error(f"Data signing failed: {e}")
            raise

    @staticmethod
    def verify_signature(public_key_str: str, data: str, signature: str) -> bool:
        """署名検証"""
        try:
            if CRYPTO_AVAILABLE:
                public_key_bytes = public_key_str.encode('utf-8')
                public_key = ec.load_pem_public_key(
                    public_key_bytes,
                    backend=default_backend()
                )

                message = data.encode('utf-8')
                signature_bytes = base64.b64decode(signature.encode('utf-8'))

                try:
                    public_key.verify(signature_bytes, message, ec.ECDSA(hashes.SHA256()))
                    return True
                except ecdsa.BadSignatureError:
                    return False
            else:
                # 模擬検証
                combined = (public_key_str + data).encode('utf-8')
                expected_hash = base64.b64encode(hashlib.sha256(combined).digest()).decode('utf-8')
                return signature == expected_hash

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    @staticmethod
    def calculate_hash(data: Any) -> str:
        """ハッシュ計算"""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return ""

class MerkleTree:
    """マークルツリー"""

    def __init__(self):
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []

    def add_leaf(self, data: str) -> str:
        """葉ノード追加"""
        leaf_hash = CryptoUtils.calculate_hash(data)
        self.leaves.append(leaf_hash)
        return leaf_hash

    def build_tree(self) -> str:
        """ツリー構築"""
        if not self.leaves:
            return ""

        # レベル0（葉ノード）
        self.tree = [self.leaves.copy()]

        # 上位レベルを構築
        current_level = self.leaves.copy()
        while len(current_level) > 1:
            next_level = []

            # ペアが奇数の場合は最後の要素を複製
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            # ペアのハッシュを計算
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                parent_hash = CryptoUtils.calculate_hash(combined)
                next_level.append(parent_hash)

            self.tree.append(next_level)
            current_level = next_level

        return current_level[0] if current_level else ""

    def get_proof(self, leaf_index: int) -> List[Tuple[str, str]]:
        """マークルプルーフ取得"""
        if not self.tree or leaf_index >= len(self.leaves):
            return []

        proof = []
        current_index = leaf_index

        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]
            is_right = current_index % 2 == 1

            if is_right:
                sibling_index = current_index - 1
            else:
                sibling_index = current_index + 1
                if sibling_index >= len(current_level):
                    sibling_index = current_index  # 奇数の場合は自分自身

            sibling_hash = current_level[sibling_index]
            direction = "left" if is_right else "right"
            proof.append((sibling_hash, direction))

            current_index = current_index // 2

        return proof

    def verify_proof(self, leaf_hash: str, proof: List[Tuple[str, str]], root_hash: str) -> bool:
        """マークルプルーフ検証"""
        current_hash = leaf_hash

        for sibling_hash, direction in proof:
            if direction == "left":
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash

            current_hash = CryptoUtils.calculate_hash(combined)

        return current_hash == root_hash

class PrivateBlockchain:
    """プライベートブロックチェーン"""

    def __init__(self, node_id: str, difficulty: int = 2):
        self.node_id = node_id
        self.difficulty = difficulty
        self.chain: List[QualityBlock] = []
        self.pending_transactions: List[QualityTransaction] = []
        self.merkle_tree = MerkleTree()

        # マイニング設定
        self.mining_reward = 1.0
        self.block_time = 10.0  # 秒
        self.max_transactions_per_block = 100

        # ノード鍵ペア
        self.private_key, self.public_key = CryptoUtils.generate_key_pair()

        # データベース
        self.db_path = Path(f"blockchain_{node_id}.db")
        self._initialize_database()

        # ジェネシスブロック作成
        self._create_genesis_block()

        # マイニングスレッド
        self.mining_thread: Optional[threading.Thread] = None
        self.is_mining = False

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS blocks (
                    block_number INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    merkle_root TEXT NOT NULL,
                    transactions TEXT NOT NULL,
                    nonce INTEGER NOT NULL,
                    difficulty INTEGER NOT NULL,
                    miner_id TEXT NOT NULL,
                    block_hash TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    block_number INTEGER,
                    timestamp TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    quality_data TEXT NOT NULL,
                    measurements TEXT NOT NULL,
                    inspector_id TEXT NOT NULL,
                    inspector_signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    hash_previous TEXT NOT NULL,
                    merkle_root TEXT NOT NULL,
                    nonce INTEGER NOT NULL,
                    verified BOOLEAN DEFAULT FALSE
                );

                CREATE TABLE IF NOT EXISTS certificates (
                    certificate_id TEXT PRIMARY KEY,
                    certificate_type TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    issuance_date TEXT NOT NULL,
                    expiry_date TEXT,
                    issuer_id TEXT NOT NULL,
                    issuer_signature TEXT NOT NULL,
                    certificate_data TEXT NOT NULL,
                    blockchain_hash TEXT NOT NULL,
                    verified BOOLEAN DEFAULT FALSE,
                    revoked BOOLEAN DEFAULT FALSE
                );

                CREATE TABLE IF NOT EXISTS supply_chain_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    contact_info TEXT NOT NULL,
                    certifications TEXT NOT NULL,
                    public_key TEXT NOT NULL,
                    trust_score REAL DEFAULT 0.0,
                    last_activity TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(product_id);
                CREATE INDEX IF NOT EXISTS idx_transactions_batch ON transactions(batch_id);
                CREATE INDEX IF NOT EXISTS idx_certificates_product ON certificates(product_id);
            """)

    def _create_genesis_block(self):
        """ジェネシスブロック作成"""
        if len(self.chain) == 0:
            genesis_block = QualityBlock(
                block_number=0,
                timestamp=datetime.now(),
                previous_hash="0" * 64,
                merkle_root=CryptoUtils.calculate_hash("genesis"),
                transactions=[],
                nonce=0,
                difficulty=self.difficulty,
                miner_id=self.node_id,
                block_hash=""
            )

            genesis_block.block_hash = self._calculate_block_hash(genesis_block)
            self.chain.append(genesis_block)
            self._save_block_to_db(genesis_block)

            logger.info("Created genesis block")

    def add_transaction(self, transaction: QualityTransaction) -> bool:
        """トランザクション追加"""
        try:
            # トランザクション検証
            if not self._validate_transaction(transaction):
                return False

            # 重複チェック
            if self._transaction_exists(transaction.transaction_id):
                return False

            self.pending_transactions.append(transaction)
            logger.info(f"Added transaction to pending pool: {transaction.transaction_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add transaction: {e}")
            return False

    def _validate_transaction(self, transaction: QualityTransaction) -> bool:
        """トランザクション検証"""
        try:
            # 署名検証
            if not CryptoUtils.verify_signature(
                self.public_key,  # 実際は検査者の公開鍵を使用
                json.dumps(transaction.quality_data, sort_keys=True, default=str),
                transaction.inspector_signature
            ):
                return False

            # タイムスタンプ検証
            if transaction.timestamp > datetime.now() + timedelta(minutes=5):
                return False

            return True

        except Exception as e:
            logger.error(f"Transaction validation failed: {e}")
            return False

    def _transaction_exists(self, transaction_id: str) -> bool:
        """トランザクション存在チェック"""
        # ペンディングトランザクションをチェック
        for tx in self.pending_transactions:
            if tx.transaction_id == transaction_id:
                return True

        # チェーン内のトランザクションをチェック
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM transactions WHERE transaction_id = ?",
                (transaction_id,)
            )
            return cursor.fetchone() is not None

    def start_mining(self) -> bool:
        """マイニング開始"""
        try:
            if self.is_mining:
                logger.warning("Mining already running")
                return False

            self.is_mining = True
            self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
            self.mining_thread.start()

            logger.info("Started blockchain mining")
            return True

        except Exception as e:
            logger.error(f"Failed to start mining: {e}")
            return False

    def stop_mining(self):
        """マイニング停止"""
        self.is_mining = False

        if self.mining_thread and self.mining_thread.is_alive():
            self.mining_thread.join(timeout=5.0)

        logger.info("Stopped blockchain mining")

    def _mining_loop(self):
        """マイニングループ"""
        logger.info("Mining loop started")

        while self.is_mining:
            try:
                # ブロック作成条件チェック
                if len(self.pending_transactions) == 0:
                    time.sleep(1.0)
                    continue

                # ブロック作成
                block = self._create_block()
                if block:
                    # マイニング（プルーフ・オブ・ワーク）
                    mined_block = self._mine_block(block)
                    if mined_block:
                        self._add_block_to_chain(mined_block)
                        logger.info(f"Mined block {mined_block.block_number}")

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Mining error: {e}")
                time.sleep(5.0)

        logger.info("Mining loop ended")

    def _create_block(self) -> Optional[QualityBlock]:
        """ブロック作成"""
        try:
            # トランザクション選択
            transactions = self.pending_transactions[:self.max_transactions_per_block]

            if not transactions:
                return None

            # マークルツリー構築
            merkle_tree = MerkleTree()
            for tx in transactions:
                merkle_tree.add_leaf(json.dumps(asdict(tx), sort_keys=True, default=str))

            merkle_root = merkle_tree.build_tree()

            # 前のブロックハッシュ取得
            previous_block = self.chain[-1]
            previous_hash = previous_block.block_hash

            block = QualityBlock(
                block_number=len(self.chain),
                timestamp=datetime.now(),
                previous_hash=previous_hash,
                merkle_root=merkle_root,
                transactions=transactions,
                nonce=0,
                difficulty=self.difficulty,
                miner_id=self.node_id,
                block_hash=""
            )

            return block

        except Exception as e:
            logger.error(f"Block creation failed: {e}")
            return None

    def _mine_block(self, block: QualityBlock) -> Optional[QualityBlock]:
        """ブロックマイニング"""
        try:
            target = "0" * self.difficulty

            # 簡易的なプルーフ・オブ・ワーク
            for nonce in range(1000000):  # 最大100万回試行
                block.nonce = nonce

                # ブロックハッシュ計算
                block.block_hash = self._calculate_block_hash(block)

                if block.block_hash.startswith(target):
                    logger.info(f"Block mined: nonce={nonce}, hash={block.block_hash[:16]}...")
                    return block

                # 難囲チェック
                if nonce % 10000 == 0 and not self.is_mining:
                    break

            return None

        except Exception as e:
            logger.error(f"Block mining failed: {e}")
            return None

    def _calculate_block_hash(self, block: QualityBlock) -> str:
        """ブロックハッシュ計算"""
        block_data = {
            "block_number": block.block_number,
            "timestamp": block.timestamp.isoformat(),
            "previous_hash": block.previous_hash,
            "merkle_root": block.merkle_root,
            "transactions": [asdict(tx) for tx in block.transactions],
            "nonce": block.nonce,
            "difficulty": block.difficulty,
            "miner_id": block.miner_id
        }

        return CryptoUtils.calculate_hash(block_data)

    def _add_block_to_chain(self, block: QualityBlock):
        """チェーンにブロック追加"""
        self.chain.append(block)

        # トランザクションをペンディングから削除
        for tx in block.transactions:
            if tx in self.pending_transactions:
                self.pending_transactions.remove(tx)

        # データベースに保存
        self._save_block_to_db(block)
        self._save_transactions_to_db(block)

    def _save_block_to_db(self, block: QualityBlock):
        """ブロックをデータベースに保存"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO blocks
                (block_number, timestamp, previous_hash, merkle_root, transactions,
                 nonce, difficulty, miner_id, block_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block.block_number,
                block.timestamp.isoformat(),
                block.previous_hash,
                block.merkle_root,
                json.dumps([asdict(tx) for tx in block.transactions], default=str),
                block.nonce,
                block.difficulty,
                block.miner_id,
                block.block_hash
            ))

    def _save_transactions_to_db(self, block: QualityBlock):
        """トランザクションをデータベースに保存"""
        with sqlite3.connect(str(self.db_path)) as conn:
            for tx in block.transactions:
                conn.execute("""
                    INSERT OR REPLACE INTO transactions
                    (transaction_id, block_number, timestamp, product_id, batch_id,
                     quality_data, measurements, inspector_id, inspector_signature,
                     status, hash_previous, merkle_root, nonce, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx.transaction_id,
                    tx.block_number,
                    tx.timestamp.isoformat(),
                    tx.product_id,
                    tx.batch_id,
                    json.dumps(tx.quality_data, default=str),
                    json.dumps(tx.measurements, default=str),
                    tx.inspector_id,
                    tx.inspector_signature,
                    tx.status.value,
                    tx.hash_previous,
                    tx.merkle_root,
                    tx.nonce,
                    tx.verified
                ))

    def get_product_history(self, product_id: str) -> List[Dict[str, Any]]:
        """製品履歴取得"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT t.*, b.block_hash, b.timestamp as block_timestamp
                FROM transactions t
                JOIN blocks b ON t.block_number = b.block_number
                WHERE t.product_id = ?
                ORDER BY t.timestamp ASC
            """, (product_id,))

            rows = cursor.fetchall()
            history = []

            for row in rows:
                history.append({
                    "transaction_id": row[0],
                    "block_number": row[1],
                    "timestamp": row[2],
                    "product_id": row[3],
                    "batch_id": row[4],
                    "quality_data": json.loads(row[5]),
                    "measurements": json.loads(row[6]),
                    "inspector_id": row[7],
                    "inspector_signature": row[8],
                    "status": row[9],
                    "block_hash": row[14],
                    "block_timestamp": row[15]
                })

            return history

    def create_certificate(self, certificate: QualityCertificate) -> bool:
        """証明書作成"""
        try:
            # ブロックチェーンハッシュ計算
            certificate_data = {
                "certificate_id": certificate.certificate_id,
                "certificate_type": certificate.certificate_type.value,
                "product_id": certificate.product_id,
                "batch_id": certificate.batch_id,
                "issuance_date": certificate.issuance_date.isoformat(),
                "expiry_date": certificate.expiry_date.isoformat() if certificate.expiry_date else None,
                "issuer_id": certificate.issuer_id,
                "certificate_data": certificate.certificate_data
            }

            certificate.blockchain_hash = CryptoUtils.calculate_hash(certificate_data)

            # データベースに保存
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO certificates
                    (certificate_id, certificate_type, product_id, batch_id,
                     issuance_date, expiry_date, issuer_id, issuer_signature,
                     certificate_data, blockchain_hash, verified, revoked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    certificate.certificate_id,
                    certificate.certificate_type.value,
                    certificate.product_id,
                    certificate.batch_id,
                    certificate.issuance_date.isoformat(),
                    certificate.expiry_date.isoformat() if certificate.expiry_date else None,
                    certificate.issuer_id,
                    certificate.issuer_signature,
                    json.dumps(certificate.certificate_data, default=str),
                    certificate.blockchain_hash,
                    certificate.verified,
                    certificate.revoked
                ))

            logger.info(f"Created certificate: {certificate.certificate_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create certificate: {e}")
            return False

    def verify_certificate(self, certificate_id: str) -> bool:
        """証明書検証"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT * FROM certificates WHERE certificate_id = ?
                """, (certificate_id,))

                row = cursor.fetchone()
                if not row:
                    return False

                certificate_data = {
                    "certificate_id": row[0],
                    "certificate_type": row[1],
                    "product_id": row[2],
                    "batch_id": row[3],
                    "issuance_date": row[4],
                    "expiry_date": row[5],
                    "issuer_id": row[6],
                    "certificate_data": json.loads(row[8])
                }

                # ブロックチェーンハッシュ検証
                calculated_hash = CryptoUtils.calculate_hash(certificate_data)
                stored_hash = row[9]

                if calculated_hash != stored_hash:
                    return False

                # 発行者署名検証（実際は発行者の公開鍵で検証）
                # ここでは簡略化

                # 有効期限検証
                if row[5]:  # expiry_date
                    expiry_date = datetime.fromisoformat(row[5])
                    if datetime.now() > expiry_date:
                        return False

                return True

        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False

    def get_chain_statistics(self) -> Dict[str, Any]:
        """チェーン統計取得"""
        return {
            "total_blocks": len(self.chain),
            "total_transactions": sum(len(block.transactions) for block in self.chain),
            "pending_transactions": len(self.pending_transactions),
            "difficulty": self.difficulty,
            "node_id": self.node_id,
            "is_mining": self.is_mining,
            "last_block": self.chain[-1].block_number if self.chain else 0,
            "last_block_hash": self.chain[-1].block_hash if self.chain else None
        }

class QualityTracabilitySystem:
    """品質トレーサビリティシステム"""

    def __init__(self, production_system: ProductionManagementSystem):
        self.production_system = production_system

        # ブロックチェーンノード
        self.blockchain_node = PrivateBlockchain("quality_node")
        self.supply_chain_nodes: Dict[str, SupplyChainNode] = {}

        # 証定
        self.auto_mining = True
        self.certificate_expiry_days = 365

        # 実行制御
        self.running = False
        self.verification_thread: Optional[threading.Thread] = None

        # スマートコントラクト機能
        self.smart_contracts: Dict[str, Any] = {}

    def start_system(self) -> bool:
        """システム起動"""
        try:
            self.running = True

            # ブロックチェーンマイニング開始
            if self.auto_mining:
                self.blockchain_node.start_mining()

            # 品質記録自動検証スレッド開始
            self.verification_thread = threading.Thread(target=self._verification_loop, daemon=True)
            self.verification_thread.start()

            logger.info("Quality tracability system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start quality tracability system: {e}")
            return False

    def stop_system(self):
        """システム停止"""
        self.running = False

        self.blockchain_node.stop_mining()

        if self.verification_thread and self.verification_thread.is_alive():
            self.verification_thread.join(timeout=5.0)

        logger.info("Quality tracability system stopped")

    def record_quality_inspection(self, product_id: str, batch_id: str,
                                quality_data: Dict[str, Any],
                                measurements: Dict[str, float],
                                inspector_id: str) -> str:
        """品質検査記録"""
        try:
            transaction = QualityTransaction(
                transaction_id=str(uuid.uuid4()),
                block_number=0,  # マイニング時に設定
                timestamp=datetime.now(),
                product_id=product_id,
                batch_id=batch_id,
                quality_data=quality_data,
                measurements=measurements,
                inspector_id=inspector_id,
                inspector_signature="",  # 実際は署名が必要
                status=QualityStatus.PENDING,
                hash_previous="",  # マイニング時に設定
                merkle_root="",  # マイニング時に設定
                nonce=0  # マイニング時に設定
            )

            # ブロックチェーンに追加
            success = self.blockchain_node.add_transaction(transaction)
            if success:
                logger.info(f"Recorded quality inspection: {transaction.transaction_id}")
                return transaction.transaction_id
            else:
                return ""

        except Exception as e:
            logger.error(f"Failed to record quality inspection: {e}")
            return ""

    def issue_quality_certificate(self, product_id: str, batch_id: str,
                                certificate_type: CertificateType,
                                certificate_data: Dict[str, Any],
                                issuer_id: str) -> str:
        """品質証明書発行"""
        try:
            certificate = QualityCertificate(
                certificate_id=str(uuid.uuid4()),
                certificate_type=certificate_type,
                product_id=product_id,
                batch_id=batch_id,
                issuance_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=self.certificate_expiry_days),
                issuer_id=issuer_id,
                issuer_signature="",  # 実際は署名が必要
                certificate_data=certificate_data,
                blockchain_hash="",  # 作成時に設定
                verified=False
            )

            success = self.blockchain_node.create_certificate(certificate)
            if success:
                logger.info(f"Issued quality certificate: {certificate.certificate_id}")
                return certificate.certificate_id
            else:
                return ""

        except Exception as e:
            logger.error(f"Failed to issue certificate: {e}")
            return ""

    def get_product_tracability(self, product_id: str) -> Dict[str, Any]:
        """製品トレーサビリティ取得"""
        try:
            # ブロックチェーン履歴取得
            blockchain_history = self.blockchain_node.get_product_history(product_id)

            # 証定書情報取得
            certificates = []
            with sqlite3.connect(str(self.blockchain_node.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT * FROM certificates WHERE product_id = ?
                    ORDER BY issuance_date DESC
                """, (product_id,))

                for row in cursor.fetchall():
                    certificates.append({
                        "certificate_id": row[0],
                        "certificate_type": row[1],
                        "issuance_date": row[4],
                        "expiry_date": row[5],
                        "issuer_id": row[6],
                        "blockchain_hash": row[9],
                        "verified": bool(row[11]),
                        "revoked": bool(row[12])
                    })

            return {
                "product_id": product_id,
                "blockchain_history": blockchain_history,
                "certificates": certificates,
                "total_transactions": len(blockchain_history),
                "verification_status": "verified" if self._verify_product_integrity(product_id) else "unverified"
            }

        except Exception as e:
            logger.error(f"Failed to get product tracability: {e}")
            return {"error": str(e)}

    def _verify_product_integrity(self, product_id: str) -> bool:
        """製品完全性検証"""
        try:
            history = self.blockchain_node.get_product_history(product_id)

            if len(history) < 2:
                return True  # 1つしかない場合は検証不要

            # ハッシュ連鎖を検証
            for i in range(1, len(history)):
                current_tx = history[i]
                previous_tx = history[i-1]

                if current_tx["hash_previous"] != previous_tx["block_hash"]:
                    logger.warning(f"Hash chain broken for product {product_id}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Product integrity verification failed: {e}")
            return False

    def _verification_loop(self):
        """検証ループ"""
        logger.info("Quality verification loop started")

        while self.running:
            try:
                # 期限切れ証明書チェック
                self._check_expired_certificates()

                # 定期的整合性チェック
                if int(time.time()) % 3600 == 0:  # 1時間ごと
                    self._perform_integrity_check()

                time.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"Verification loop error: {e}")
                time.sleep(300)  # エラー時は5分後に再試行

        logger.info("Quality verification loop ended")

    def _check_expired_certificates(self):
        """期限切れ証明書チェック"""
        try:
            with sqlite3.connect(str(self.blockchain_node.db_path)) as conn:
                cursor = conn.execute("""
                    UPDATE certificates
                    SET revoked = TRUE
                    WHERE expiry_date < ? AND revoked = FALSE
                """, (datetime.now().isoformat(),))

                revoked_count = cursor.rowcount
                if revoked_count > 0:
                    logger.info(f"Revoked {revoked_count} expired certificates")

        except Exception as e:
            logger.error(f"Certificate expiry check failed: {e}")

    def _perform_integrity_check(self):
        """整合性チェック実行"""
        try:
            # ブロックチェーン整合性チェック
            chain = self.blockchain_node.chain

            for i in range(1, len(chain)):
                current_block = chain[i]
                previous_block = chain[i-1]

                if current_block.previous_hash != previous_block.block_hash:
                    logger.error(f"Block chain integrity broken at block {current_block.block_number}")
                    break

            logger.info("Blockchain integrity check completed")

        except Exception as e:
            logger.error(f"Integrity check failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            "running": self.running,
            "auto_mining": self.auto_mining,
            "blockchain": self.blockchain_node.get_chain_statistics(),
            "supply_chain_nodes": len(self.supply_chain_nodes),
            "smart_contracts": len(self.smart_contracts),
            "verification_thread_active": self.verification_thread.is_alive() if self.verification_thread else False
        }

# グローバルインスタンス
quality_tracability_system: Optional[QualityTracabilitySystem] = None

def initialize_quality_tracability_system(production_system: ProductionManagementSystem) -> QualityTracabilitySystem:
    """品質トレーサビリティシステム初期化"""
    global quality_tracability_system
    quality_tracability_system = QualityTracabilitySystem(production_system)
    return quality_tracability_system

def get_quality_tracability_system() -> Optional[QualityTracabilitySystem]:
    """品質トレーサビリティシステム取得"""
    return quality_tracability_system

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    print("Testing Blockchain Quality Tracability System...")

    try:
        # モック生産管理システム
        from production_management_integration import ProductionManagementSystem, MockMESConnector
        mock_pms = ProductionManagementSystem(MockMESConnector())

        # 品質トレーサビリティシステム初期化
        qt_system = initialize_quality_tracability_system(mock_pms)

        if qt_system.start_system():
            print("Quality tracability system started successfully!")

            # 品質検査記録
            inspection_id = qt_system.record_quality_inspection(
                product_id="PROD_001",
                batch_id="BATCH_001",
                quality_data={
                    "inspection_type": "final_quality",
                    "standards": ["ISO_9001", "ISO_14001"],
                    "passed": True
                },
                measurements={
                    "dimension_x": 100.1,
                    "dimension_y": 50.2,
                    "weight": 250.5
                },
                inspector_id="inspector_001"
            )

            print(f"Recorded quality inspection: {inspection_id}")

            # 品質証明書発行
            certificate_id = qt_system.issue_quality_certificate(
                product_id="PROD_001",
                batch_id="BATCH_001",
                certificate_type=CertificateType.QUALITY_CERTIFICATE,
                certificate_data={
                    "grade": "A",
                    "compliance": "ISO_9001",
                    "test_results": "PASSED"
                },
                issuer_id="qa_department"
            )

            print(f"Issued quality certificate: {certificate_id}")

            # トレーサビリティ確認
            time.sleep(3)

            tracability = qt_system.get_product_tracability("PROD_001")
            print(f"Product tracability retrieved: {len(tracability['blockchain_history'])} transactions")

            # システム状態確認
            status = qt_system.get_system_status()
            print(f"System status: {status}")

            time.sleep(2)
            qt_system.stop_system()

        else:
            print("Failed to start quality tracability system")

    except Exception as e:
        print(f"Quality tracability system test failed: {e}")