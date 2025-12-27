"""
Baseline cost analysis and comparison.

Establishes normal usage patterns for anomaly detection.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List

from ai_cost_guard.storage.models import LLMUsageEvent


class BaselineState(Enum):
    """State of baseline computation based on data availability."""
    COLD = "cold"  # Insufficient data for reliable baseline
    WARM = "warm"  # Sufficient data for reliable baseline


@dataclass(frozen=True)
class BaselineMetrics:
    """Statistical metrics computed from usage events."""
    median_cost: float
    p90_cost: float
    median_tokens: int
    sample_count: int
    
    def __post_init__(self):
        """Validate metrics are reasonable."""
        if self.median_cost < 0:
            raise ValueError("median_cost cannot be negative")
        if self.p90_cost < 0:
            raise ValueError("p90_cost cannot be negative")
        if self.median_tokens < 0:
            raise ValueError("median_tokens cannot be negative")
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")


@dataclass(frozen=True)
class BaselineResult:
    """Complete baseline computation result."""
    metrics: BaselineMetrics
    state: BaselineState
    window_start: datetime
    window_end: datetime
    
    def __post_init__(self):
        """Validate time window is logical."""
        if self.window_start > self.window_end:
            raise ValueError("window_start must be before window_end")


def compute_baseline(events: List[LLMUsageEvent]) -> BaselineResult:
    """Compute baseline metrics from usage events.
    
    Uses median and P90 statistics because they are robust to outliers
    and provide stable baselines for anomaly detection. Medians represent
    typical behavior without being skewed by expensive edge cases.
    
    Args:
        events: List of usage events (assumed filtered by feature + model)
        
    Returns:
        BaselineResult with computed metrics and state
        
    Raises:
        ValueError: If events list is empty or contains invalid data
    """
    if not events:
        raise ValueError("Events list cannot be empty")
    
    # Validate all events have required cost data
    for i, event in enumerate(events):
        if not hasattr(event, 'estimated_cost') or event.estimated_cost is None:
            raise ValueError(f"Event at index {i} missing estimated_cost")
    
    # Sort events by timestamp (newest first) for deterministic processing
    sorted_events = sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    # Apply time window (last 7 days) and event limit (200 events)
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    
    filtered_events = []
    for event in sorted_events:
        if event.timestamp < seven_days_ago:
            break  # Stop when we reach events older than 7 days
        filtered_events.append(event)
        if len(filtered_events) >= 200:
            break  # Stop at 200 events
    
    if not filtered_events:
        raise ValueError("No events found within 7-day window")
    
    # Determine baseline state based on sample size
    state = BaselineState.WARM if len(filtered_events) >= 20 else BaselineState.COLD
    
    # Extract costs and tokens for computation
    costs = [event.estimated_cost for event in filtered_events]
    tokens = [event.total_tokens for event in filtered_events]
    
    # Compute exact percentiles (no approximation)
    median_cost = _compute_exact_percentile(costs, 50)
    p90_cost = _compute_exact_percentile(costs, 90)
    median_tokens = _compute_exact_percentile(tokens, 50)
    
    # Determine time window
    window_end = filtered_events[0].timestamp  # Most recent event
    window_start = filtered_events[-1].timestamp  # Oldest event in sample
    
    metrics = BaselineMetrics(
        median_cost=median_cost,
        p90_cost=p90_cost,
        median_tokens=int(median_tokens),
        sample_count=len(filtered_events)
    )
    
    return BaselineResult(
        metrics=metrics,
        state=state,
        window_start=window_start,
        window_end=window_end
    )


def _compute_exact_percentile(values: List[float], percentile: int) -> float:
    """Compute exact percentile using linear interpolation.
    
    Uses the same method as numpy.percentile with interpolation='linear'
    for deterministic and mathematically correct results.
    
    Args:
        values: List of numeric values
        percentile: Percentile to compute (0-100)
        
    Returns:
        Exact percentile value
    """
    if not values:
        raise ValueError("Values list cannot be empty")
    
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    # Sort values for percentile computation
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Convert percentile to position (0-indexed)
    position = (percentile / 100.0) * (n - 1)
    
    # Linear interpolation between adjacent values
    lower_index = int(position)
    upper_index = min(lower_index + 1, n - 1)
    fraction = position - lower_index
    
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    
    return lower_value + fraction * (upper_value - lower_value)
