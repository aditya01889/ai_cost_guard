"""
Unit tests for storage layer.

Tests schema creation, event insertion, and retrieval operations.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ai_cost_guard.storage.models import LLMUsageEvent
from ai_cost_guard.storage.repository import (
    initialize_schema,
    insert_usage_event,
    insert_usage_events,
    fetch_recent_usage_events
)


class TestStorageSchema:
    """Test database schema creation and structure."""
    
    def test_schema_creation(self):
        """Verify table is created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Verify table exists
            from ai_cost_guard.storage.db import get_connection
            conn = get_connection(db_path)
            try:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='llm_usage_event'
                """)
                tables = cursor.fetchall()
                assert len(tables) == 1
                assert tables[0][0] == "llm_usage_event"
                
                # Verify column structure
                cursor = conn.execute("PRAGMA table_info(llm_usage_event)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                expected_columns = [
                    'id', 'timestamp', 'feature', 'model', 
                    'prompt_tokens', 'completion_tokens', 'total_tokens',
                    'estimated_cost', 'retry_count', 'request_id'
                ]
                assert column_names == expected_columns
            finally:
                conn.close()


class TestEventInsertion:
    """Test usage event insertion operations."""
    
    def test_insert_single_event(self):
        """Test inserting a single usage event."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            event = LLMUsageEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                feature="chat_completion",
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                estimated_cost=4.50,
                retry_count=0,
                request_id="req_123"
            )
            
            insert_usage_event(event, db_path)
            
            # Verify event was inserted
            events = fetch_recent_usage_events(db_path=db_path)
            assert len(events) == 1
            assert events[0].feature == "chat_completion"
            assert events[0].model == "gpt-4"
            assert events[0].prompt_tokens == 100
            assert events[0].completion_tokens == 50
            assert events[0].total_tokens == 150
            assert events[0].estimated_cost == 4.50
            assert events[0].retry_count == 0
            assert events[0].request_id == "req_123"
    
    def test_insert_multiple_events(self):
        """Test inserting multiple usage events in a transaction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            events = [
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    feature="chat_completion",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    estimated_cost=4.50
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 1, 0),
                    feature="embedding",
                    model="text-embedding-ada-002",
                    prompt_tokens=200,
                    completion_tokens=0,
                    total_tokens=200,
                    estimated_cost=0.10
                )
            ]
            
            insert_usage_events(events, db_path)
            
            # Verify all events were inserted
            all_events = fetch_recent_usage_events(db_path=db_path)
            assert len(all_events) == 2
            assert all_events[0].feature == "embedding"  # Most recent first
            assert all_events[1].feature == "chat_completion"
    
    def test_insert_empty_event_list(self):
        """Test inserting empty list of events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            insert_usage_events([], db_path)
            
            # Verify no events were inserted
            events = fetch_recent_usage_events(db_path=db_path)
            assert len(events) == 0
    
    def test_event_persistence_across_connections(self):
        """Test that data persists across different database connections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Insert event using first connection
            event = LLMUsageEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                feature="chat_completion",
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                estimated_cost=4.50
            )
            insert_usage_event(event, db_path)
            
            # Verify event persists across new connection
            events = fetch_recent_usage_events(db_path=db_path)
            assert len(events) == 1
            assert events[0].feature == "chat_completion"
    
    def test_retry_count_default(self):
        """Test that retry_count defaults to 0 when not specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            event = LLMUsageEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                feature="chat_completion",
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                estimated_cost=4.50
                # retry_count not specified, should default to 0
            )
            
            insert_usage_event(event, db_path)
            
            # Verify retry_count is 0
            events = fetch_recent_usage_events(db_path=db_path)
            assert len(events) == 1
            assert events[0].retry_count == 0


class TestEventRetrieval:
    """Test usage event retrieval operations."""
    
    def test_fetch_all_events(self):
        """Test fetching all recent events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Insert test events
            events = [
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    feature="chat_completion",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    estimated_cost=4.50
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 1, 0),
                    feature="embedding",
                    model="text-embedding-ada-002",
                    prompt_tokens=200,
                    completion_tokens=0,
                    total_tokens=200,
                    estimated_cost=0.10
                )
            ]
            insert_usage_events(events, db_path)
            
            # Fetch all events
            fetched_events = fetch_recent_usage_events(db_path=db_path)
            assert len(fetched_events) == 2
            assert fetched_events[0].feature == "embedding"  # Most recent first
            assert fetched_events[1].feature == "chat_completion"
    
    def test_fetch_events_by_feature(self):
        """Test fetching events filtered by feature."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Insert test events
            events = [
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    feature="chat_completion",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    estimated_cost=4.50
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 1, 0),
                    feature="embedding",
                    model="text-embedding-ada-002",
                    prompt_tokens=200,
                    completion_tokens=0,
                    total_tokens=200,
                    estimated_cost=0.10
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 2, 0),
                    feature="chat_completion",
                    model="gpt-3.5-turbo",
                    prompt_tokens=50,
                    completion_tokens=25,
                    total_tokens=75,
                    estimated_cost=0.15
                )
            ]
            insert_usage_events(events, db_path)
            
            # Fetch events by feature
            chat_events = fetch_recent_usage_events(feature="chat_completion", db_path=db_path)
            assert len(chat_events) == 2
            assert all(event.feature == "chat_completion" for event in chat_events)
            
            embedding_events = fetch_recent_usage_events(feature="embedding", db_path=db_path)
            assert len(embedding_events) == 1
            assert embedding_events[0].feature == "embedding"
    
    def test_fetch_events_by_model(self):
        """Test fetching events filtered by model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Insert test events
            events = [
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    feature="chat_completion",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    estimated_cost=4.50
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 1, 0),
                    feature="chat_completion",
                    model="gpt-3.5-turbo",
                    prompt_tokens=50,
                    completion_tokens=25,
                    total_tokens=75,
                    estimated_cost=0.15
                )
            ]
            insert_usage_events(events, db_path)
            
            # Fetch events by model
            gpt4_events = fetch_recent_usage_events(model="gpt-4", db_path=db_path)
            assert len(gpt4_events) == 1
            assert gpt4_events[0].model == "gpt-4"
            
            gpt35_events = fetch_recent_usage_events(model="gpt-3.5-turbo", db_path=db_path)
            assert len(gpt35_events) == 1
            assert gpt35_events[0].model == "gpt-3.5-turbo"
    
    def test_fetch_events_by_feature_and_model(self):
        """Test fetching events filtered by both feature and model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Insert test events
            events = [
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    feature="chat_completion",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    estimated_cost=4.50
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 1, 0),
                    feature="chat_completion",
                    model="gpt-3.5-turbo",
                    prompt_tokens=50,
                    completion_tokens=25,
                    total_tokens=75,
                    estimated_cost=0.15
                ),
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, 2, 0),
                    feature="embedding",
                    model="gpt-4",
                    prompt_tokens=200,
                    completion_tokens=0,
                    total_tokens=200,
                    estimated_cost=0.10
                )
            ]
            insert_usage_events(events, db_path)
            
            # Fetch events by both feature and model
            filtered_events = fetch_recent_usage_events(
                feature="chat_completion", 
                model="gpt-4", 
                db_path=db_path
            )
            assert len(filtered_events) == 1
            assert filtered_events[0].feature == "chat_completion"
            assert filtered_events[0].model == "gpt-4"
    
    def test_fetch_events_with_limit(self):
        """Test fetching events with limit applied."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Insert test events
            events = [
                LLMUsageEvent(
                    timestamp=datetime(2024, 1, 1, 12, i, 0),
                    feature="chat_completion",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    estimated_cost=4.50
                )
                for i in range(5)  # Create 5 events
            ]
            insert_usage_events(events, db_path)
            
            # Fetch with limit
            limited_events = fetch_recent_usage_events(limit=3, db_path=db_path)
            assert len(limited_events) == 3
    
    def test_fetch_events_from_empty_database(self):
        """Test fetching events from empty database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            initialize_schema(db_path)
            
            # Fetch from empty database
            events = fetch_recent_usage_events(db_path=db_path)
            assert len(events) == 0


class TestAppendOnlyNature:
    """Test that storage maintains append-only behavior."""
    
    def test_no_update_methods_exist(self):
        """Verify that no UPDATE methods are exposed in the repository."""
        # Check that repository module doesn't contain update/delete functions
        import ai_cost_guard.storage.repository as repo_module

        # Get all function attributes from the module (exclude imports and classes)
        functions = [
            name for name in dir(repo_module)
            if callable(getattr(repo_module, name))
            and not name.startswith('_')
            and not name[0].isupper()  # Exclude class names like LLMUsageEvent
            and name not in ['List', 'Optional', 'datetime', 'get_connection', 'dataclass', 'timedelta']  # Exclude imports and imported functions
        ]

        # Verify only expected functions exist
        expected_functions = {
            'initialize_schema',
            'insert_usage_event',
            'insert_usage_events',
            'fetch_recent_usage_events',
            'get_repository'
        }

        actual_functions = set(functions)
        assert actual_functions == expected_functions

        # Verify no update/delete operations exist
        for func_name in functions:
            assert 'update' not in func_name.lower()
            assert 'delete' not in func_name.lower()
            assert 'remove' not in func_name.lower()
            assert 'modify' not in func_name.lower()
