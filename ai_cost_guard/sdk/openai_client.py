"""
Guarded OpenAI client wrapper.

Records usage events for cost tracking without modifying behavior.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..core.pricing import calculate_cost
from ..core.token_counter import TokenUsage
from ..storage.models import LLMUsageEvent
from ..storage.repository import insert_usage_event


class GuardedOpenAI:
    """OpenAI client wrapper that records usage events.
    
    Wraps OpenAI chat completions to create an immutable audit trail.
    All failures are loud to ensure no silent data loss.
    """
    
    def __init__(self, model: str, feature: str, db_path: Optional[str] = None):
        """Initialize guarded OpenAI client.
        
        Args:
            model: OpenAI model name (required)
            feature: Feature identifier for tracking (required)
            db_path: Database file path (defaults to ".ai-cost-guard.db")
            
        Raises:
            ValueError: If model or feature is missing/empty
        """
        if not model or not model.strip():
            raise ValueError("model is required and cannot be empty")
        if not feature or not feature.strip():
            raise ValueError("feature is required and cannot be empty")
        
        self.model = model
        self.feature = feature
        self.db_path = db_path or ".ai-cost-guard.db"
        self.client = OpenAI()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Create chat completion with usage recording.
        
        Makes OpenAI API call and records usage event on success.
        Failures are loud to prevent silent data loss.
        
        Args:
            messages: List of message dictionaries (required)
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional OpenAI parameters
            
        Returns:
            OpenAI chat completion response
            
        Raises:
            ValueError: If messages is empty
            OpenAI API errors: Propagated without modification
            Database errors: Propagated without modification
        """
        if not messages:
            raise ValueError("messages is required and cannot be empty")
        
        # Make OpenAI API call - any failure here stops execution
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Extract usage information from response
        usage = response.usage
        if not usage:
            raise ValueError("OpenAI response missing usage information")
        
        # Calculate cost using our pricing module
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens
        )
        estimated_cost = calculate_cost(self.model, token_usage)
        
        # Create and store usage event - any failure here is loud
        event = LLMUsageEvent(
            timestamp=datetime.now(),
            feature=self.feature,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=estimated_cost,
            request_id=response.id
        )
        
        insert_usage_event(event, self.db_path)
        
        # Return original OpenAI response unchanged
        return response
