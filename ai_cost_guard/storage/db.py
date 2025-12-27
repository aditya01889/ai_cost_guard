"""
Database connection management.

Provides SQLite connection for data persistence.
"""

import sqlite3
from pathlib import Path


def get_connection(db_path: str = "ai_cost_guard.db") -> sqlite3.Connection:
    """Create and return a SQLite database connection with foreign keys enabled.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite connection with foreign key constraints enabled
    """
    path = Path(db_path)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
