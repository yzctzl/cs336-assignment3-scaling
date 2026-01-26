import sqlite3
from typing import Any, Dict, List, Optional


class Database:
    def __init__(self, db_path: str = "hyperturing.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for API keys and their total FLOPs usage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key TEXT PRIMARY KEY,
                    total_flops_used REAL DEFAULT 0.0
                )
            """)
            # Table for recording runs
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
                    loss REAL,
                    FOREIGN KEY (api_key) REFERENCES api_keys(api_key)
                )
            """)
            conn.commit()

    def get_total_flops(self, api_key: str) -> Optional[float]:
        # ... (same)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT total_flops_used FROM api_keys WHERE api_key = ?", (api_key,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def add_api_key(self, api_key: str):
        # ... (same)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO api_keys (api_key) VALUES (?)", (api_key,)
            )
            conn.commit()

    def update_total_flops(self, api_key: str, flops: float):
        # ... (same)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET total_flops_used = total_flops_used + ? WHERE api_key = ?",
                (flops, api_key),
            )
            conn.commit()

    def get_previous_runs(self, api_key: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT d_model, num_layers, num_heads, batch_size, learning_rate, train_flops, vocab_size, context_length, loss 
                FROM runs WHERE api_key = ?
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
                    "loss": r[8],
                }
                for r in rows
            ]

    def get_existing_run(self, api_key: str, config: Dict[str, Any]) -> Optional[float]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT loss FROM runs 
                WHERE api_key = ? AND d_model = ? AND num_layers = ? AND num_heads = ? 
                AND batch_size = ? AND learning_rate = ? AND train_flops = ? 
                AND vocab_size = ? AND context_length = ?
            """,
                (
                    api_key,
                    config["d_model"],
                    config["num_layers"],
                    config["num_heads"],
                    config["batch_size"],
                    config["learning_rate"],
                    config["train_flops"],
                    config.get("vocab_size", 32000),  # Handle legacy
                    config.get("context_length", 512),
                ),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def record_run(self, api_key: str, config: Dict[str, Any], loss: float):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (api_key, d_model, num_layers, num_heads, batch_size, learning_rate, train_flops, vocab_size, context_length, loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
