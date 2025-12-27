"""
Repository pattern for data access.

Handles database operations and data persistence logic.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .db import get_connection
from .models import LLMUsageEvent


class UsageRepository:
    """Repository for accessing and managing LLM usage data.
    
    This class provides a higher-level interface to the database operations,
    making it easier to work with usage data in a type-safe manner.
    """
    
    def __init__(self, db_path: str = "ai_cost_guard.db"):
        """Initialize the repository with a database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
    
    def get_recent_events(
        self,
        feature: Optional[str] = None,
        model: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 1000
    ) -> List[LLMUsageEvent]:
        """Get recent usage events with optional filtering.
        
        Args:
            feature: Optional filter for specific feature
            model: Optional filter for specific model
            days: Optional number of days to look back
            limit: Maximum number of events to return
            
        Returns:
            List of usage events ordered by timestamp (newest first)
        """
        conn = get_connection(self.db_path)
        try:
            query = """
                SELECT timestamp, feature, model, prompt_tokens, 
                       completion_tokens, total_tokens, estimated_cost, 
                       retry_count, request_id 
                FROM llm_usage_event
            """
            params = []
            conditions = []
            
            if feature:
                conditions.append("feature = ?")
                params.append(feature)
            if model:
                conditions.append("model = ?")
                params.append(model)
            if days is not None:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("timestamp >= ?")
                params.append(cutoff)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            events = []
            for row in cursor.fetchall():
                events.append(LLMUsageEvent(
                    timestamp=datetime.fromisoformat(row[0]),
                    feature=row[1],
                    model=row[2],
                    prompt_tokens=row[3],
                    completion_tokens=row[4],
                    total_tokens=row[5],
                    estimated_cost=row[6],
                    retry_count=row[7],
                    request_id=row[8]
                ))
            return events
        finally:
            conn.close()
    
    def get_usage_stats(
        self,
        feature: Optional[str] = None,
        model: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, float]:
        """Get usage statistics for the specified time period.
        
        Args:
            feature: Optional filter for specific feature
            model: Optional filter for specific model
            days: Number of days to include in the statistics
            
        Returns:
            Dictionary containing usage statistics
        """
        conn = get_connection(self.db_path)
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            query = """
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(estimated_cost) as total_cost,
                    AVG(estimated_cost) as avg_cost,
                    SUM(total_tokens) as total_tokens
                FROM llm_usage_event
                WHERE timestamp >= ?
            """
            params = [cutoff]
            
            if feature:
                query += " AND feature = ?"
                params.append(feature)
            if model:
                query += " AND model = ?"
                params.append(model)
            
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            return {
                "total_requests": row[0] or 0,
                "total_cost": float(row[1] or 0),
                "avg_cost": float(row[2] or 0),
                "total_tokens": row[3] or 0
            }
        finally:
            conn.close()


# Global repository instance
_default_repository: Optional[UsageRepository] = None


def get_repository(db_path: str = "ai_cost_guard.db") -> UsageRepository:
    """Get a repository instance.
    
    This function provides a singleton instance of the UsageRepository.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        An instance of UsageRepository
    """
    global _default_repository
    if _default_repository is None:
        _default_repository = UsageRepository(db_path)
    return _default_repository


# Keep the existing functions for backward compatibility
def initialize_schema(db_path: str = "ai_cost_guard.db") -> None:
    """Create the llm_usage_event table if it doesn't exist.
    
    This creates an append-only ledger for immutable usage events.
    No UPDATE or DELETE operations should ever be performed on this table.
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                feature TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost REAL NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                request_id TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_usage_event(event: LLMUsageEvent, db_path: str = "ai_cost_guard.db") -> None:
    """Insert a single usage event into the append-only ledger.
    
    This operation is append-only - events cannot be modified after insertion.
    Transaction ensures atomicity of the write operation.
    
    Args:
        event: The usage event to record
        db_path: Path to SQLite database file
    """
    conn = get_connection(db_path)
    try:
        conn.execute("""
            INSERT INTO llm_usage_event 
            (timestamp, feature, model, prompt_tokens, completion_tokens, 
             total_tokens, estimated_cost, retry_count, request_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp.isoformat(),
            event.feature,
            event.model,
            event.prompt_tokens,
            event.completion_tokens,
            event.total_tokens,
            event.estimated_cost,
            event.retry_count,
            event.request_id
        ))
        conn.commit()
    finally:
        conn.close()


def insert_usage_events(events: List[LLMUsageEvent], db_path: str = "ai_cost_guard.db") -> None:
    """Insert multiple usage events atomically into the append-only ledger.
    
    All events are inserted in a single transaction to ensure consistency.
    This operation is append-only - events cannot be modified after insertion.
    
    Args:
        events: List of usage events to record
        db_path: Path to SQLite database file
    """
    if not events:
        return
    
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN TRANSACTION")
        for event in events:
            conn.execute("""
                INSERT INTO llm_usage_event 
                (timestamp, feature, model, prompt_tokens, completion_tokens, 
                 total_tokens, estimated_cost, retry_count, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp.isoformat(),
                event.feature,
                event.model,
                event.prompt_tokens,
                event.completion_tokens,
                event.total_tokens,
                event.estimated_cost,
                event.retry_count,
                event.request_id
            ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_recent_usage_events(
    feature: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 100,
    db_path: str = "ai_cost_guard.db"
) -> List[LLMUsageEvent]:
    """Fetch recent usage events, optionally filtered by feature and model.
    
    Returns events in reverse chronological order (newest first).
    This is a read-only operation that preserves the append-only nature.
    
    Args:
        feature: Optional filter for specific feature
        model: Optional filter for specific model
        limit: Maximum number of events to return
        db_path: Path to SQLite database file
        
    Returns:
        List of usage events ordered by timestamp (newest first)
    """
    conn = get_connection(db_path)
    try:
        query = "SELECT timestamp, feature, model, prompt_tokens, completion_tokens, total_tokens, estimated_cost, retry_count, request_id FROM llm_usage_event"
        params = []
        
        if feature or model:
            conditions = []
            if feature:
                conditions.append("feature = ?")
                params.append(feature)
            if model:
                conditions.append("model = ?")
                params.append(model)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        events = []
        for row in cursor.fetchall():
            events.append(LLMUsageEvent(
                timestamp=datetime.fromisoformat(row[0]),
                feature=row[1],
                model=row[2],
                prompt_tokens=row[3],
                completion_tokens=row[4],
                total_tokens=row[5],
                estimated_cost=row[6],
                retry_count=row[7],
                request_id=row[8]
            ))
        return events
    finally:
        conn.close()