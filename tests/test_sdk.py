"""
Unit tests for SDK layer.

Tests OpenAI client wrapper behavior and usage recording.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from ai_cost_guard.sdk.openai_client import GuardedOpenAI
from ai_cost_guard.storage.models import LLMUsageEvent
from ai_cost_guard.storage.repository import fetch_recent_usage_events, initialize_schema


class TestGuardedOpenAI:
    """Test GuardedOpenAI client wrapper."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        initialize_schema(self.db_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_init_success(self, mock_openai_class):
        """Test successful initialization."""
        mock_openai_class.return_value = Mock()
        
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        assert client.model == "gpt-4"
        assert client.feature == "chat_completion"
        assert client.db_path == self.db_path
        assert client.client is not None
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_init_default_db_path(self, mock_openai_class):
        """Test initialization with default database path."""
        mock_openai_class.return_value = Mock()
        
        client = GuardedOpenAI(model="gpt-4", feature="chat_completion")
        
        assert client.db_path == ".ai-cost-guard.db"
    
    def test_init_missing_model(self):
        """Test initialization fails with missing model."""
        with pytest.raises(ValueError, match="model is required"):
            GuardedOpenAI(model="", feature="chat_completion")
        
        with pytest.raises(ValueError, match="model is required"):
            GuardedOpenAI(model=None, feature="chat_completion")
    
    def test_init_missing_feature(self):
        """Test initialization fails with missing feature."""
        with pytest.raises(ValueError, match="feature is required"):
            GuardedOpenAI(model="gpt-4", feature="")
        
        with pytest.raises(ValueError, match="feature is required"):
            GuardedOpenAI(model="gpt-4", feature=None)
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_chat_success_records_event(self, mock_openai_class):
        """Test successful chat call records usage event."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.id = "chat_123"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create client and make call
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages=messages)
        
        # Verify OpenAI was called correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=messages,
            temperature=None,
            max_tokens=None
        )
        
        # Verify response is returned unchanged
        assert response == mock_response
        
        # Verify event was recorded
        events = fetch_recent_usage_events(db_path=self.db_path)
        assert len(events) == 1
        
        event = events[0]
        assert event.feature == "chat_completion"
        assert event.model == "gpt-4"
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50
        assert event.total_tokens == 150
        assert event.request_id == "chat_123"
        assert event.estimated_cost == 6.00  # GPT-4: 100/1000*30 + 50/1000*60 = 3.0 + 3.0 = 6.0
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_chat_with_optional_parameters(self, mock_openai_class):
        """Test chat call with optional parameters."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.id = "chat_456"
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 300
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create client and make call with parameters
        client = GuardedOpenAI(
            model="gpt-3.5-turbo",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Verify OpenAI was called with all parameters
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Verify event was recorded with correct cost
        events = fetch_recent_usage_events(db_path=self.db_path)
        assert len(events) == 1
        assert events[0].estimated_cost == 0.50  # GPT-3.5: 200/1000*1.5 + 100/1000*2.0 = 0.3 + 0.2 = 0.5
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_chat_openai_failure_no_event_recorded(self, mock_openai_class):
        """Test OpenAI API failure does not record event."""
        # Mock OpenAI to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        # Create client and make call
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Verify exception is raised
        with pytest.raises(Exception, match="API Error"):
            client.chat(messages=messages)
        
        # Verify no event was recorded
        events = fetch_recent_usage_events(db_path=self.db_path)
        assert len(events) == 0
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    @patch('ai_cost_guard.sdk.openai_client.insert_usage_event')
    def test_chat_db_failure_raises_error(self, mock_insert, mock_openai_class):
        """Test database failure raises error without swallowing."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.id = "chat_789"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Mock database insert to fail
        mock_insert.side_effect = Exception("DB Error")
        
        # Create client and make call
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Verify database error is raised (not swallowed)
        with pytest.raises(Exception, match="DB Error"):
            client.chat(messages=messages)
        
        # Verify insert was attempted
        mock_insert.assert_called_once()
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_chat_missing_usage_raises_error(self, mock_openai_class):
        """Test response without usage information raises error."""
        # Mock OpenAI response without usage
        mock_response = Mock()
        mock_response.usage = None
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create client and make call
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Verify error is raised
        with pytest.raises(ValueError, match="usage information"):
            client.chat(messages=messages)
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_chat_empty_messages_raises_error(self, mock_openai_class):
        """Test empty messages raises error."""
        mock_openai_class.return_value = Mock()
        
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        # Verify error is raised for empty messages
        with pytest.raises(ValueError, match="messages is required"):
            client.chat(messages=[])
        
        with pytest.raises(ValueError, match="messages is required"):
            client.chat(messages=None)
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_exactly_one_event_per_successful_call(self, mock_openai_class):
        """Test exactly one event is recorded per successful call."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.id = "chat_single"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create client
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Make multiple calls
        for i in range(3):
            mock_response.id = f"chat_{i}"
            client.chat(messages=messages)
        
        # Verify exactly 3 events recorded
        events = fetch_recent_usage_events(db_path=self.db_path)
        assert len(events) == 3
        
        # Verify all events have unique request IDs
        request_ids = [event.request_id for event in events]
        assert len(set(request_ids)) == 3
        assert "chat_0" in request_ids
        assert "chat_1" in request_ids
        assert "chat_2" in request_ids
    
    @patch('ai_cost_guard.sdk.openai_client.OpenAI')
    def test_chat_with_additional_kwargs(self, mock_openai_class):
        """Test chat call passes through additional kwargs."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.id = "chat_kwargs"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create client and make call with additional kwargs
        client = GuardedOpenAI(
            model="gpt-4",
            feature="chat_completion",
            db_path=self.db_path
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(
            messages=messages,
            temperature=0.5,
            stream=True,
            stop=["\n"]
        )
        
        # Verify all kwargs were passed through
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
            max_tokens=None,
            stream=True,
            stop=["\n"]
        )
        
        # Verify event was still recorded
        events = fetch_recent_usage_events(db_path=self.db_path)
        assert len(events) == 1
