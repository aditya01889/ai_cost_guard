"""
Cost guardrails and limits enforcement.

Implements spending limits and cost controls with strict enforcement policies.

Enforcement Order:
1. Per-request max cost - Prevents catastrophic single-request costs
2. Budget limits - Ensures overall spending stays within budget
3. Anomaly detection - Handles unexpected usage patterns
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from .anomaly import AnomalyEvent, AnomalySeverity
from .baseline import BaselineResult, BaselineState
from ai_cost_guard.storage.models import LLMUsageEvent


class EnforcementAction(Enum):
    """Available enforcement actions in order of severity."""
    ALLOW = auto()    # Allow the request (no action)
    WARN = auto()      # Log warning but allow request
    DOWNGRADE = auto() # Suggest using a cheaper model
    THROTTLE = auto()  # Delay or rate-limit the request
    BLOCK = auto()     # Reject the request entirely


class GuardrailViolation(Exception):
    """Raised when a guardrail is enforced with BLOCK or THROTTLE action."""
    def __init__(self, message: str, action: EnforcementAction):
        super().__init__(message)
        self.action = action


@dataclass
class BudgetState:
    """Current budget state for a feature or model."""
    amount_used: float
    amount_remaining: float
    budget_period_days: int


@dataclass
class GuardrailConfig:
    """Configuration for guardrail enforcement."""
    max_cost_per_request: Optional[float] = None
    budget_limit: Optional[float] = None
    on_budget_breach: EnforcementAction = EnforcementAction.BLOCK
    on_critical_anomaly: EnforcementAction = EnforcementAction.BLOCK
    on_warning_anomaly: EnforcementAction = EnforcementAction.WARN


def enforce_guardrails(
    feature: str,
    model: str,
    config: GuardrailConfig,
    baseline: BaselineResult,
    current_event: LLMUsageEvent,
    anomalies: List[AnomalyEvent],
    budget_state: BudgetState
) -> EnforcementAction:
    """
    Enforce guardrails in a specific order of precedence.

    Enforcement Order:
    1. Per-request max cost - Prevents catastrophic single-request costs
    2. Budget limits - Ensures overall spending stays within budget
    3. Anomaly detection - Handles unexpected usage patterns

    Args:
        feature: Name of the feature making the request
        model: Model being used
        config: Guardrail configuration
        baseline: Baseline metrics for the feature/model
        current_event: Current usage event
        anomalies: List of detected anomalies
        budget_state: Current budget state

    Returns:
        EnforcementAction: The most severe action that was triggered

    Raises:
        GuardrailViolation: If BLOCK or THROTTLE action is required
    """
    action_taken = EnforcementAction.ALLOW  # Default to no action
    message = ""

    def _update_action(new_action: EnforcementAction, new_message: str) -> None:
        nonlocal action_taken, message
        if new_action.value > action_taken.value:
            action_taken = new_action
            message = new_message

    # 1. Check per-request max cost
    if (config.max_cost_per_request is not None and 
            current_event.estimated_cost > config.max_cost_per_request):
        action = EnforcementAction.BLOCK
        msg = (
            f"Request cost ${current_event.estimated_cost:.4f} exceeds "
            f"maximum allowed ${config.max_cost_per_request:.4f} for {feature}/{model}"
        )
        _update_action(action, msg)

    # 2. Check budget limits
    if (config.budget_limit is not None and 
            budget_state.amount_remaining <= 0):
        action = config.on_budget_breach
        msg = (
            f"Budget limit of ${config.budget_limit:.2f} reached for {feature}. "
            f"Current spend: ${budget_state.amount_used:.2f}"
        )
        _update_action(action, msg)

    # 3. Check for critical anomalies (only if baseline is WARM)
    if baseline.state == BaselineState.WARM:
        for anomaly in anomalies:
            if anomaly.severity == AnomalySeverity.CRITICAL:
                action = config.on_critical_anomaly
                msg = f"Critical anomaly detected in {feature}/{model}: {anomaly.message}"
                _update_action(action, msg)
            elif anomaly.severity == AnomalySeverity.WARNING:
                action = config.on_warning_anomaly
                msg = f"Warning anomaly in {feature}/{model}: {anomaly.message}"
                _update_action(action, msg)

    # If we have a blocking action, raise an exception
    if action_taken in (EnforcementAction.BLOCK, EnforcementAction.THROTTLE):
        raise GuardrailViolation(message, action_taken)

    return action_taken
