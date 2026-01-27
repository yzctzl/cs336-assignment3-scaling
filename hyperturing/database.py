import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "hyperturing.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for API keys and their total FLOPs usage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key TEXT PRIMARY KEY,
                    total_flops_used REAL DEFAULT 0.0
                )
            """)
            # Table for recording runs with status tracking
            # Status: PENDING, RUNNING, SUCCESS, FAILED
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT,
                    d_model INTEGER,
                    num_layers INTEGER,
                    num_heads INTEGER,
                    batch_size INTEGER,
                    learning_rate REAL,
                    train_flops REAL,
                    vocab_size INTEGER,
                    context_length INTEGER,
                    status TEXT DEFAULT 'PENDING',
                    loss REAL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (api_key) REFERENCES api_keys(api_key)
                )
            """)
            conn.commit()

    def get_total_flops(self, api_key: str) -> Optional[float]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT total_flops_used FROM api_keys WHERE api_key = ?", (api_key,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def add_api_key(self, api_key: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO api_keys (api_key) VALUES (?)", (api_key,)
            )
            conn.commit()

    def update_total_flops(self, api_key: str, flops: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET total_flops_used = total_flops_used + ? WHERE api_key = ?",
                (flops, api_key),
            )
            conn.commit()

    def get_previous_runs(self, api_key: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT d_model, num_layers, num_heads, batch_size, learning_rate, train_flops, vocab_size, context_length, status, loss, error_message
                FROM runs WHERE api_key = ? ORDER BY created_at DESC
            """,
                (api_key,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "d_model": r[0],
                    "num_layers": r[1],
                    "num_heads": r[2],
                    "batch_size": r[3],
                    "learning_rate": r[4],
                    "train_flops": r[5],
                    "vocab_size": r[6],
                    "context_length": r[7],
                    "status": r[8],
                    "loss": r[9],
                    "error_message": r[10],
                }
                for r in rows
            ]

    def get_existing_run(self, api_key: str, config: Dict[str, Any]) -> Optional[float]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT loss FROM runs 
                WHERE api_key = ? AND d_model = ? AND num_layers = ? AND num_heads = ? 
                AND batch_size = ? AND learning_rate = ? AND train_flops = ? 
                AND vocab_size = ? AND context_length = ? AND status = 'SUCCESS'
            """,
                (
                    api_key,
                    config["d_model"],
                    config["num_layers"],
                    config["num_heads"],
                    config["batch_size"],
                    config["learning_rate"],
                    config["train_flops"],
                    config.get("vocab_size", 32000),
                    config.get("context_length", 512),
                ),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def initialize_run(self, api_key: str, config: Dict[str, Any]) -> int:
        """Create a placeholder for a new run and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (api_key, d_model, num_layers, num_heads, batch_size, learning_rate, train_flops, vocab_size, context_length, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
                (
                    api_key,
                    config["d_model"],
                    config["num_layers"],
                    config["num_heads"],
                    config["batch_size"],
                    config["learning_rate"],
                    config["train_flops"],
                    config.get("vocab_size", 32000),
                    config.get("context_length", 512),
                ),
            )
            conn.commit()
            return cursor.lastrowid  # pyright: ignore[reportReturnType]

    def update_run_status(
        self,
        run_id: int,
        status: str,
        loss: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status == "SUCCESS" and loss is not None:
                cursor.execute(
                    "UPDATE runs SET status = ?, loss = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, loss, run_id),
                )
            elif status == "FAILED":
                cursor.execute(
                    "UPDATE runs SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, error_message, run_id),
                )
            else:
                cursor.execute(
                    "UPDATE runs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, run_id),
                )
            conn.commit()

    def record_run(self, api_key: str, config: Dict[str, Any], loss: float):
        """Legacy support for recording a completed run directly."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (api_key, d_model, num_layers, num_heads, batch_size, learning_rate, train_flops, vocab_size, context_length, status, loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUCCESS', ?)
            """,
                (
                    api_key,
                    config["d_model"],
                    config["num_layers"],
                    config["num_heads"],
                    config["batch_size"],
                    config["learning_rate"],
                    config["train_flops"],
                    config.get("vocab_size", 32000),
                    config.get("context_length", 512),
                    loss,
                ),
            )
            conn.commit()
