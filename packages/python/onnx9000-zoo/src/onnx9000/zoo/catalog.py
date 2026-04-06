import hashlib
import os
import sqlite3
from typing import Any, Optional


class ZooCatalog:
    """Distributed SQLite indexer for tracking exact Git SHAs and tensor hash digests."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize the ZooCatalog.

        Args:
            db_path: Path to the SQLite database. Defaults to in-memory.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the necessary tables if they do not exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                hub TEXT NOT NULL,
                git_sha TEXT NOT NULL,
                hyperparameters TEXT,
                tensor_hash TEXT
            )
        """)
        self.conn.commit()

    def add_model(
        self,
        model_id: str,
        hub: str,
        git_sha: str,
        hyperparameters: str = "{}",
        tensor_hash: str = "",
    ) -> None:
        """Add or update a model in the catalog.

        Args:
            model_id: Unique identifier for the model.
            hub: The hub it belongs to (e.g., 'huggingface', 'timm', 'bonsai').
            git_sha: The git commit SHA for reproducible tracking.
            hyperparameters: JSON string of hyperparameters.
            tensor_hash: Hash digest of the weights/tensors.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO models (id, hub, git_sha, hyperparameters, tensor_hash)
            VALUES (?, ?, ?, ?, ?)
        """,
            (model_id, hub, git_sha, hyperparameters, tensor_hash),
        )
        self.conn.commit()

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a model from the catalog.

        Args:
            model_id: The ID of the model to retrieve.

        Returns:
            Dictionary containing model metadata, or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, hub, git_sha, hyperparameters, tensor_hash FROM models WHERE id = ?",
            (model_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "hub": row[1],
                "git_sha": row[2],
                "hyperparameters": row[3],
                "tensor_hash": row[4],
            }
        return None

    def list_models(self, hub: Optional[str] = None) -> list[dict[str, Any]]:
        """List all models in the catalog, optionally filtered by hub.

        Args:
            hub: Optional hub to filter by.

        Returns:
            List of dictionaries containing model metadata.
        """
        cursor = self.conn.cursor()
        if hub:
            cursor.execute(
                "SELECT id, hub, git_sha, hyperparameters, tensor_hash FROM models WHERE hub = ?",
                (hub,),
            )
        else:
            cursor.execute("SELECT id, hub, git_sha, hyperparameters, tensor_hash FROM models")

        rows = cursor.fetchall()
        return [
            {"id": r[0], "hub": r[1], "git_sha": r[2], "hyperparameters": r[3], "tensor_hash": r[4]}
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
