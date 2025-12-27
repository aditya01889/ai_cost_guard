"""
Token counting and usage tracking.

Manages token calculations for different AI model formats.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """Token usage data for cost calculation.
    
    Contains exact token counts without estimation or model-specific logic.
    """
    prompt_tokens: int
    completion_tokens: int
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens
