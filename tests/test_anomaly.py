"""
Unit tests for anomaly detection.

Tests conservative anomaly detection rules and validation.
"""

from datetime import datetime, timedelta

import pytest

from ai_cost_guard.core.anomaly import (
    detect_anomalies,
    AnomalySeverity,
    AnomalyEvent
)
from ai_cost_guard.core.baseline import (
    BaselineResult,
    BaselineMetrics,
    BaselineState
)
from ai_cost_guard.storage.models import LLMUsageEvent


class TestAnomalyDetection:
    """Test anomaly detection rules and validation."""
    
    def create_baseline(self, p90_cost: float = 10.0, median_tokens: int = 1000, state: BaselineState = BaselineState.WARM) -> BaselineResult:
        """Create a baseline result for testing.
        
        Args:
            p90_cost: P90 cost for the baseline (default: 10.0)
            median_tokens: Median tokens for the baseline (default: 1000)
            state: Baseline state (default: WARM)
        """
        metrics = BaselineMetrics(
            median_cost=8.0,  # Provide a default median_cost that's less than p90_cost
            p90_cost=p90_cost,
            median_tokens=median_tokens,
            sample_count=100
        )
        
        return BaselineResult(
            metrics=metrics,
            state=state,
            window_start=datetime.now() - timedelta(days=1),
            window_end=datetime.now()
        )
    
    def create_current_event(self, cost: float = 10.0, tokens: int = 1000, retries: int = 0) -> LLMUsageEvent:
        """Create a current usage event for testing."""
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
    
    def test_no_anomalies_for_normal_behavior(self):
        """Test that normal behavior produces no anomalies."""
        baseline = self.create_baseline(
            p90_cost=15.0,
            median_tokens=1000
        )
        
        # Below all thresholds
        current_event = self.create_current_event(cost=10.0, tokens=1000, retries=1)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        
        assert len(anomalies) == 0
    
    def test_rule_a_cost_spike(self):
        """Test Rule A (CRITICAL) triggers when cost > 1.5 * P90."""
        baseline = self.create_baseline(p90_cost=10.0)
        
        # Cost above 1.5 * P90 (15.0) should trigger Rule A
        current_event = self.create_current_event(cost=16.0)  # 16 > 15 (10 * 1.5)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_a = [a for a in anomalies if a.rule == "A"]
        
        assert len(rule_a) == 1, "Should have exactly one Rule A anomaly"
        assert rule_a[0].severity == AnomalySeverity.CRITICAL
        assert rule_a[0].observed_value == 16.0
        assert rule_a[0].baseline_value == 10.0
        assert rule_a[0].threshold == 15.0
        assert "Cost spike" in rule_a[0].message
    
    def test_rule_a_no_trigger_below_threshold(self):
        """Test Rule A does not trigger at or below 1.5 * P90."""
        baseline = self.create_baseline(p90_cost=10.0)
        
        # Cost exactly at 1.5 * P90 should not trigger Rule A
        current_event = self.create_current_event(cost=15.0)  # 15 == 10 * 1.5
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_a = [a for a in anomalies if a.rule == "A"]
        
        assert len(rule_a) == 0, "Rule A should not trigger at 1.5 * P90"
    
    def test_rule_b_token_explosion(self):
        """Test Rule B (WARNING) triggers when tokens > 1.7 * median."""
        baseline = self.create_baseline(median_tokens=1000)
        
        # Tokens above 1.7 * median (1700) should trigger Rule B
        current_event = self.create_current_event(tokens=1800)  # 1800 > 1700 (1000 * 1.7)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_b = [a for a in anomalies if a.rule == "B"]
        
        assert len(rule_b) == 1, "Should have exactly one Rule B anomaly"
        assert rule_b[0].severity == AnomalySeverity.WARNING
        assert rule_b[0].observed_value == 1800
        assert rule_b[0].baseline_value == 1000
        assert rule_b[0].threshold == 1700
        assert "token usage" in rule_b[0].message.lower()
    
    def test_rule_b_no_trigger_below_threshold(self):
        """Test Rule B does not trigger at or below 1.7 * median tokens."""
        baseline = self.create_baseline(median_tokens=1000)
        
        # Tokens exactly at 1.7 * median should not trigger Rule B
        current_event = self.create_current_event(tokens=1700)  # 1700 == 1000 * 1.7
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_b = [a for a in anomalies if a.rule == "B"]
        
        assert len(rule_b) == 0, "Rule B should not trigger at 1.7 * median tokens"
    
    def test_rule_c_retry_amplification(self):
        """Test Rule C (WARNING) triggers for retry amplification."""
        baseline = self.create_baseline(p90_cost=10.0)
        
        # 2 retries * 8.0 = 16.0 > 13.0 (10 * 1.3) should trigger Rule C
        current_event = self.create_current_event(cost=8.0, retries=2)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_c = [a for a in anomalies if a.rule == "C"]
        
        assert len(rule_c) == 1, "Should have exactly one Rule C anomaly"
        assert rule_c[0].severity == AnomalySeverity.WARNING
        assert rule_c[0].observed_value == 16.0  # 8 * 2 retries
        assert rule_c[0].baseline_value == 10.0
        assert rule_c[0].threshold == 13.0  # 10 * 1.3
        assert "retry" in rule_c[0].message.lower()
    
    def test_rule_c_no_retry_no_trigger(self):
        """Test Rule C does not trigger without retries."""
        baseline = self.create_baseline(p90_cost=10.0)
        
        # No retries, so Rule C should not trigger even if cost is high
        current_event = self.create_current_event(cost=20.0, retries=1)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_c = [a for a in anomalies if a.rule == "C"]
        
        assert len(rule_c) == 0, "Rule C should not trigger without retries > 1"
    
    def test_multiple_rules_can_trigger_independently(self):
        """Test that multiple rules can trigger for the same event."""
        baseline = self.create_baseline(
            p90_cost=10.0,
            median_tokens=1000
        )
        
        # This should trigger:
        # - Rule A: 16.0 > 15.0 (10 * 1.5)
        # - Rule B: 2000 > 1700 (1000 * 1.7)
        current_event = self.create_current_event(cost=16.0, tokens=2000, retries=1)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        
        # Should trigger both Rule A and B
        rules = {a.rule for a in anomalies}
        assert rules == {"A", "B"}, "Should trigger both Rule A and B"
        
        # Verify both anomalies have correct details
        for anomaly in anomalies:
            assert anomaly.feature == "test_feature"
            assert anomaly.model == "test_model"
    
    def test_all_three_rules_can_trigger(self):
        """Test that all three rules can trigger simultaneously."""
        baseline = self.create_baseline(
            p90_cost=10.0,
            median_tokens=1000
        )
        
        # This should trigger:
        # - Rule A: 16.0 > 15.0 (10 * 1.5)
        # - Rule B: 2000 > 1700 (1000 * 1.7)
        # - Rule C: 2 retries * 8.0 = 16.0 > 13.0 (10 * 1.3)
        current_event = self.create_current_event(cost=16.0, tokens=2000, retries=2)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        
        # Should trigger all three rules
        rules = {a.rule for a in anomalies}
        assert len(rules) == 3, f"Expected 3 rules to trigger, got {rules}"
        assert "A" in rules
        assert "B" in rules
        assert "C" in rules
        
        # Verify severities
        rule_severities = {a.rule: a.severity for a in anomalies}
        assert rule_severities["A"] == AnomalySeverity.CRITICAL
        assert rule_severities["B"] == AnomalySeverity.WARNING
        assert rule_severities["C"] == AnomalySeverity.WARNING
    
    def test_cold_baseline_produces_no_anomalies(self):
        """Test that cold baseline produces no anomalies."""
        baseline = self.create_baseline(
            p90_cost=15.0,
            median_tokens=1000,
            state=BaselineState.COLD
        )
        
        # Even with high cost and retries, cold baseline should produce no anomalies
        current_event = self.create_current_event(cost=100.0, tokens=2000, retries=3)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        
        assert len(anomalies) == 0, "Cold baseline should not produce any anomalies"
    
    def test_missing_baseline_metrics_raises_error(self):
        """Test that missing baseline metrics raises error."""
        baseline = BaselineResult(
            metrics=None,  # Missing metrics
            state=BaselineState.WARM,
            window_start=datetime.now() - timedelta(days=1),
            window_end=datetime.now()
        )
        
        current_event = self.create_current_event(cost=10.0, tokens=1000)
        
        with pytest.raises(AttributeError):
            detect_anomalies("test_feature", "test_model", baseline, current_event)
    
    def test_missing_current_event_cost_raises_error(self):
        """Test that missing current event cost raises error."""
        baseline = self.create_baseline(p90_cost=15.0, median_tokens=1000)
        
        # Create event without cost
        current_event = LLMUsageEvent(
            timestamp=datetime.now(),
            feature="test_feature",
            model="test_model",
            prompt_tokens=500,
            completion_tokens=500,
            total_tokens=1000,
            estimated_cost=None,  # Missing cost
            retry_count=1
        )
        
        with pytest.raises(ValueError, match="Current event missing estimated_cost"):
            detect_anomalies("test_feature", "test_model", baseline, current_event)
    
    def test_missing_current_event_tokens_raises_error(self):
        """Test that missing current event tokens raises error."""
        baseline = self.create_baseline(p90_cost=15.0, median_tokens=1000)
        
        # Create event without tokens
        current_event = LLMUsageEvent(
            timestamp=datetime.now(),
            feature="test_feature",
            model="test_model",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=None,  # Missing tokens
            estimated_cost=10.0,
            retry_count=1
        )
        
        with pytest.raises(ValueError, match="Current event missing total_tokens"):
            detect_anomalies("test_feature", "test_model", baseline, current_event)
    
    def test_missing_retry_count_raises_error(self):
        """Test that missing retry_count raises error."""
        baseline = self.create_baseline(p90_cost=15.0, median_tokens=1000)
        
        # Create event without retry_count
        current_event = LLMUsageEvent(
            timestamp=datetime.now(),
            feature="test_feature",
            model="test_model",
            prompt_tokens=500,
            completion_tokens=500,
            total_tokens=1000,
            estimated_cost=10.0,
            retry_count=None  # Missing retry_count
        )
        
        with pytest.raises(ValueError, match="Current event missing retry_count"):
            detect_anomalies("test_feature", "test_model", baseline, current_event)
    
    def test_anomaly_event_dataclass_immutability(self):
        """Test that AnomalyEvent is immutable."""
        event = AnomalyEvent(
            feature="test",
            model="test_model",
            rule="A",
            severity=AnomalySeverity.CRITICAL,
            observed_value=16.0,
            baseline_value=10.0,
            threshold=15.0,
            message="Test message"
        )
        
        # Should not be able to modify attributes
        with pytest.raises(Exception):
            event.rule = "B"
    
    def test_anomaly_messages_are_human_readable(self):
        """Test that anomaly messages are human-readable and accurate."""
        baseline = self.create_baseline(p90_cost=10.0, median_tokens=1000)
        
        # Test Rule A message
        current_event = self.create_current_event(cost=16.0)  # 16 > 15 (10 * 1.5)
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_a = next((a for a in anomalies if a.rule == "A"), None)
        assert rule_a is not None
        assert "Cost spike" in rule_a.message
        assert "16.00" in rule_a.message  # Check cost is formatted with 2 decimal places
        assert "10.00" in rule_a.message  # Check baseline is included
        
        # Test Rule B message
        current_event = self.create_current_event(tokens=1800)  # 1800 > 1700 (1000 * 1.7)
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_b = next((a for a in anomalies if a.rule == "B"), None)
        
    def test_rule_c_with_high_retry_count(self):
        """Test Rule C triggers with high retry count."""
        baseline = self.create_baseline(p90_cost=10.0)
        
        # High retry count should trigger Rule C
        current_event = self.create_current_event(cost=5.0, retries=3)  # 5 * 3 = 15 > 13 (10 * 1.3)
        
        anomalies = detect_anomalies("test_feature", "test_model", baseline, current_event)
        rule_c = [a for a in anomalies if a.rule == "C"]
        
        assert len(rule_c) == 1, "Rule C should trigger with high retry count"
        assert rule_c[0].observed_value == 15.0  # 5 * 3 retries
