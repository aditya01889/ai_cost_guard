"""
Data models for storage layer.

Defines database entities and data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class LLMUsageEvent:
    """Immutable record of LLM usage for financial tracking.
    
    Append-only events that create an auditable ledger of AI costs.
    Once written, these records must never be modified.
    """
    timestamp: datetime
    feature: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    retry_count: int = 0
    request_id: Optional[str] = None
