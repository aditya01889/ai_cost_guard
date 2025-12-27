"""
Anomaly detection for cost patterns.

Identifies unusual spending behavior and potential issues.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from .baseline import BaselineResult, BaselineState
from ai_cost_guard.storage.models import LLMUsageEvent


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AnomalyEvent:
    """Detected anomaly with details and explanation."""
    feature: str
    model: str
    rule: str  # "A", "B", or "C"
    severity: AnomalySeverity
    observed_value: float
    baseline_value: float
    threshold: float
    message: str


def detect_anomalies(
    feature: str,
    model: str,
    baseline: BaselineResult,
    current_event: LLMUsageEvent,
) -> List[AnomalyEvent]:
    """Detect anomalies based on current event vs baseline.
    
    Rules:
    - Rule A (CRITICAL): Cost > 1.5 * P90 cost
    - Rule B (WARNING): Tokens > 1.7 * median tokens
    - Rule C (WARNING): Retry amplification (retries > 1 and total_cost > 1.3 * P90)
    
    Args:
        feature: Name of the feature being checked
        model: Model being used
        baseline: Baseline metrics for comparison
        current_event: Current usage event to check
        
    Returns:
        List of detected anomalies (empty if none)
        
    Raises:
        ValueError: If required metrics are missing
    """
    if baseline.state == BaselineState.COLD:
        return []

    # Validate required fields
    if not hasattr(current_event, 'estimated_cost') or current_event.estimated_cost is None:
        raise ValueError("Current event missing estimated_cost")
    if not hasattr(current_event, 'total_tokens') or current_event.total_tokens is None:
        raise ValueError("Current event missing total_tokens")
    if not hasattr(current_event, 'retry_count') or current_event.retry_count is None:
        raise ValueError("Current event missing retry_count")

    anomalies = []
    
    # Rule A: Cost Spike (CRITICAL)
    if baseline.metrics.p90_cost is not None:
        threshold = baseline.metrics.p90_cost * 1.5
        if current_event.estimated_cost > threshold:
            anomalies.append(AnomalyEvent(
                feature=feature,
                model=model,
                rule="A",
                severity=AnomalySeverity.CRITICAL,
                observed_value=current_event.estimated_cost,
                baseline_value=baseline.metrics.p90_cost,
                threshold=threshold,
                message=f"Cost spike detected: ${current_event.estimated_cost:.2f} (P90: ${baseline.metrics.p90_cost:.2f} * 1.5 = ${threshold:.2f})"
            ))
    
    # Rule B: Token Explosion (WARNING)
    if baseline.metrics.median_tokens is not None:
        threshold = baseline.metrics.median_tokens * 1.7
        if current_event.total_tokens > threshold:
            anomalies.append(AnomalyEvent(
                feature=feature,
                model=model,
                rule="B",
                severity=AnomalySeverity.WARNING,
                observed_value=current_event.total_tokens,
                baseline_value=baseline.metrics.median_tokens,
                threshold=threshold,
                message=f"High token usage: {current_event.total_tokens:,} (Median: {baseline.metrics.median_tokens:,} * 1.7 = {threshold:,.0f})"
            ))
    
    # Rule C: Retry Amplification (WARNING)
    if (baseline.metrics.p90_cost is not None and 
        current_event.retry_count > 1):
        threshold = baseline.metrics.p90_cost * 1.3
        total_cost = current_event.estimated_cost * current_event.retry_count
        if total_cost > threshold:
            anomalies.append(AnomalyEvent(
                feature=feature,
                model=model,
                rule="C",
                severity=AnomalySeverity.WARNING,
                observed_value=total_cost,
                baseline_value=baseline.metrics.p90_cost,
                threshold=threshold,
                message=f"Retry amplification: ${total_cost:.2f} (${current_event.estimated_cost:.2f} * {current_event.retry_count} retries) > ${threshold:.2f} (P90: ${baseline.metrics.p90_cost:.2f} * 1.3)"
            ))
    
    return anomalies
