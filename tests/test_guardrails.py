"""
Tests for guardrail enforcement logic.
"""
import pytest
from datetime import datetime, timedelta

from ai_cost_guard.core.guardrails import (
    enforce_guardrails,
    EnforcementAction,
    GuardrailConfig,
    GuardrailViolation,
    BudgetState
)
from ai_cost_guard.core.anomaly import AnomalyEvent, AnomalySeverity
from ai_cost_guard.core.baseline import BaselineResult, BaselineMetrics, BaselineState
from ai_cost_guard.storage.models import LLMUsageEvent


class TestGuardrailEnforcement:
    """Test guardrail enforcement logic."""

    def create_baseline(self, state=BaselineState.WARM):
        """Create a test baseline."""
        return BaselineResult(
            metrics=BaselineMetrics(
                median_cost=0.1,
                p90_cost=0.2,
                median_tokens=1000,
                sample_count=100
            ),
            state=state,
            window_start=datetime.now() - timedelta(days=1),
            window_end=datetime.now()
        )

    def create_event(self, cost=0.1, tokens=100, retries=1):
        """Create a test event."""
        return LLMUsageEvent(
            timestamp=datetime.now(),
            feature="test_feature",
            model="test_model",
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
            estimated_cost=cost,
            retry_count=retries
        )

    def test_max_cost_per_request_block(self):
        """Test blocking when request exceeds max cost."""
        config = GuardrailConfig(max_cost_per_request=1.0)
        baseline = self.create_baseline()
        event = self.create_event(cost=2.0)
        
        with pytest.raises(GuardrailViolation) as excinfo:
            enforce_guardrails(
                "test_feature", "test_model", 
                config, baseline, event, [], BudgetState(0, 100, 30)
            )
        
        assert excinfo.value.action == EnforcementAction.BLOCK
        assert "exceeds maximum allowed" in str(excinfo.value)

    def test_budget_breach_action(self):
        """Test budget breach triggers configured action."""
        config = GuardrailConfig(
            budget_limit=100,
            on_budget_breach=EnforcementAction.THROTTLE
        )
        baseline = self.create_baseline()
        event = self.create_event()
        
        with pytest.raises(GuardrailViolation) as excinfo:
            enforce_guardrails(
                "test_feature", "test_model",
                config, baseline, event, [],
                BudgetState(amount_used=100, amount_remaining=0, budget_period_days=30)
            )
        
        assert excinfo.value.action == EnforcementAction.THROTTLE

    def test_critical_anomaly_enforcement(self):
        """Test critical anomaly triggers enforcement."""
        config = GuardrailConfig(on_critical_anomaly=EnforcementAction.BLOCK)
        baseline = self.create_baseline()
        event = self.create_event()
        anomalies = [
            AnomalyEvent(
                feature="test_feature",
                model="test_model",
                rule="A",
                severity=AnomalySeverity.CRITICAL,
                observed_value=100,
                baseline_value=10,
                threshold=50,
                message="Critical cost anomaly"
            )
        ]
        
        with pytest.raises(GuardrailViolation) as excinfo:
            enforce_guardrails(
                "test_feature", "test_model",
                config, baseline, event, anomalies,
                BudgetState(0, 100, 30)
            )
        
        assert excinfo.value.action == EnforcementAction.BLOCK

    def test_warning_anomaly_no_enforcement(self):
        """Test warning anomaly doesn't enforce by default."""
        config = GuardrailConfig()
        baseline = self.create_baseline()
        event = self.create_event()
        anomalies = [
            AnomalyEvent(
                feature="test_feature",
                model="test_model",
                rule="B",
                severity=AnomalySeverity.WARNING,
                observed_value=100,
                baseline_value=50,
                threshold=85,
                message="Warning token anomaly"
            )
        ]
        
        # Should not raise
        action = enforce_guardrails(
            "test_feature", "test_model",
            config, baseline, event, anomalies,
            BudgetState(0, 100, 30)
        )
        
        assert action == EnforcementAction.WARN

    def test_cold_baseline_suppresses_anomaly_enforcement(self):
        """Test that cold baselines suppress anomaly enforcement."""
        config = GuardrailConfig(on_critical_anomaly=EnforcementAction.BLOCK)
        baseline = self.create_baseline(state=BaselineState.COLD)
        event = self.create_event()
        anomalies = [
            AnomalyEvent(
                feature="test_feature",
                model="test_model",
                rule="A",
                severity=AnomalySeverity.CRITICAL,
                observed_value=100,
                baseline_value=10,
                threshold=50,
                message="Critical anomaly"
            )
        ]
        
        # Should not raise despite critical anomaly
        action = enforce_guardrails(
            "test_feature", "test_model",
            config, baseline, event, anomalies,
            BudgetState(0, 100, 30)
        )
        
        assert action == EnforcementAction.ALLOW

    def test_enforcement_order_max_cost_first(self):
        """Test that max cost is checked before budget and anomalies."""
        config = GuardrailConfig(
            max_cost_per_request=1.0,
            budget_limit=100,
            on_budget_breach=EnforcementAction.BLOCK,
            on_critical_anomaly=EnforcementAction.BLOCK
        )
        baseline = self.create_baseline()
        
        # This would trigger all checks, but max cost should be first
        event = self.create_event(cost=2.0)
        
        with pytest.raises(GuardrailViolation) as excinfo:
            enforce_guardrails(
                "test_feature", "test_model",
                config, baseline, event, 
                [AnomalyEvent("test", "test", "A", AnomalySeverity.CRITICAL, 100, 10, 50, "test")],
                BudgetState(amount_used=100, amount_remaining=0, budget_period_days=30)
            )
        
        # Should be BLOCK due to max cost, not budget or anomaly
        assert "exceeds maximum allowed" in str(excinfo.value)
        assert excinfo.value.action == EnforcementAction.BLOCK

    def test_downgrade_action_returns_without_raising(self):
        """Test DOWNGRADE action returns without raising exception."""
        config = GuardrailConfig(
            on_critical_anomaly=EnforcementAction.DOWNGRADE
        )
        baseline = self.create_baseline()
        event = self.create_event()
        anomalies = [
            AnomalyEvent(
                feature="test_feature",
                model="test_model",
                rule="A",
                severity=AnomalySeverity.CRITICAL,
                observed_value=100,
                baseline_value=10,
                threshold=50,
                message="Critical anomaly"
            )
        ]
        
        # Should not raise, should return DOWNGRADE
        action = enforce_guardrails(
            "test_feature", "test_model",
            config, baseline, event, anomalies,
            BudgetState(0, 100, 30)
        )
        
        assert action == EnforcementAction.DOWNGRADE
