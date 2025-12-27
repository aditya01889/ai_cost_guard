"""
Tests for the cost simulation engine.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ai_cost_guard.core.simulation import (
    SimulationVerdict,
    simulate_cost_impact,
    FeatureSimulationResult,
    SimulationResult
)
from ai_cost_guard.core.anomaly import AnomalyEvent, AnomalySeverity
from ai_cost_guard.core.baseline import BaselineResult, BaselineState, BaselineMetrics
from ai_cost_guard.core.guardrails import (
    GuardrailConfig,
    EnforcementAction,
    BudgetState
)
from ai_cost_guard.storage.models import LLMUsageEvent


class TestSimulation:
    """Test simulation functionality."""

    def create_test_event(
        self, 
        feature="test_feature", 
        model="test_model",
        cost=1.0,
        tokens=100,
        retries=0
    ) -> LLMUsageEvent:
        """Create a test LLM usage event."""
        return LLMUsageEvent(
            timestamp=datetime.now(),
            feature=feature,
            model=model,
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
            estimated_cost=cost,
            retry_count=retries
        )
    
    def create_mock_repository(self, events=None):
        """Create a mock repository with test data."""
        if events is None:
            events = [self.create_test_event()]
            
        mock_repo = MagicMock()
        mock_repo.get_recent_events.return_value = events
        return mock_repo
    
    def test_simulate_cost_impact_no_events(self):
        """Test simulation with no historical events."""
        mock_repo = self.create_mock_repository(events=[])
        config = GuardrailConfig()
        
        result = simulate_cost_impact(
            feature=None,
            config=config,
            repository=mock_repo
        )
        
        assert result.overall_verdict == SimulationVerdict.PASS
        assert result.estimated_monthly_impact == 0.0
        assert not result.per_feature_results
        
    def test_simulate_cost_impact_pass_verdict(self):
        """Test simulation with passing verdict."""
        # Create test events with normal usage
        events = [
            self.create_test_event(cost=1.0, tokens=100),
            self.create_test_event(cost=1.1, tokens=110),
            self.create_test_event(cost=0.9, tokens=90),
        ]
        
        mock_repo = self.create_mock_repository(events=events)
        config = GuardrailConfig(
            max_cost_per_request=2.0,
            budget_limit=100.0
        )
        
        result = simulate_cost_impact(
            feature=None,
            config=config,
            repository=mock_repo
        )
        
        assert result.overall_verdict == SimulationVerdict.PASS
        assert result.estimated_monthly_impact > 0
        assert len(result.per_feature_results) == 1
        
        feature_result = result.per_feature_results[0]
        assert not feature_result.violations
        
    def test_simulate_cost_impact_warn_verdict(self):
        """Test simulation with warning verdict."""
        # Create test events with warning-level anomalies
        events = [
            self.create_test_event(cost=1.0, tokens=1000),  # High token usage
            self.create_test_event(cost=1.0, tokens=1000),
            self.create_test_event(cost=1.0, tokens=1000),
        ]
        
        mock_repo = self.create_mock_repository(events=events)
        config = GuardrailConfig(
            max_cost_per_request=2.0,
            budget_limit=100.0,
            on_warning_anomaly=EnforcementAction.WARN
        )
        
        with patch('ai_cost_guard.core.simulation.detect_anomalies') as mock_detect:
            # Simulate warning anomaly
            mock_detect.return_value = [
                AnomalyEvent(
                    feature="test_feature",
                    model="test_model",
                    rule="B",
                    severity=AnomalySeverity.WARNING,
                    observed_value=1000,
                    baseline_value=500,
                    threshold=850,
                    message="High token usage"
                )
            ]
            
            result = simulate_cost_impact(
                feature=None,
                config=config,
                repository=mock_repo
            )
            
            assert result.overall_verdict == SimulationVerdict.WARN
            assert any(
                r.violations for r in result.per_feature_results
            )
    
    def test_simulate_cost_impact_fail_verdict(self):
        """Test simulation with fail verdict due to blocking violation."""
        # Create test events that would exceed max cost
        events = [
            self.create_test_event(cost=3.0, tokens=100),  # Over max cost
            self.create_test_event(cost=3.0, tokens=100),
            self.create_test_event(cost=3.0, tokens=100),
        ]
        
        mock_repo = self.create_mock_repository(events=events)
        config = GuardrailConfig(
            max_cost_per_request=2.0,  # Will trigger BLOCK
            budget_limit=100.0
        )
        
        result = simulate_cost_impact(
            feature=None,
            config=config,
            repository=mock_repo
        )
        
        assert result.overall_verdict == SimulationVerdict.FAIL
        assert any(
            any(a in (EnforcementAction.BLOCK, EnforcementAction.THROTTLE) 
                for a, _ in r.violations)
            for r in result.per_feature_results
        )
    
    def test_simulate_cost_impact_cold_baseline(self):
        """Test simulation with cold baseline suppresses anomalies."""
        # Create test events with warning-level anomalies
        events = [
            self.create_test_event(cost=1.0, tokens=1000),
        ]
        
        mock_repo = self.create_mock_repository(events=events)
        config = GuardrailConfig()
        
        with patch('ai_cost_guard.core.simulation.detect_anomalies') as mock_detect:
            # Simulate that detect_anomalies would return a critical anomaly
            mock_detect.return_value = [
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
            
            # But with a cold baseline, it should be suppressed
            with patch('ai_cost_guard.core.simulation._create_baseline_from_events') as mock_baseline:
                mock_baseline.return_value = BaselineResult(
                    metrics=None,
                    state=BaselineState.COLD,
                    window_start=datetime.now() - timedelta(days=1),
                    window_end=datetime.now()
                )
                
                result = simulate_cost_impact(
                    feature=None,
                    config=config,
                    repository=mock_repo
                )
                
                # Should pass despite the anomaly due to cold baseline
                assert result.overall_verdict == SimulationVerdict.PASS
    
    def test_determine_overall_verdict(self):
        """Test verdict determination logic."""
        from ai_cost_guard.core.simulation import _determine_overall_verdict, FeatureSimulationResult
        
        # Test PASS with no violations
        results = [
            FeatureSimulationResult(
                feature="test1",
                model="model1",
                estimated_monthly_cost=10.0,
                violations=[],
                anomalies=[]
            )
        ]
        assert _determine_overall_verdict(results) == SimulationVerdict.PASS
        
        # Test WARN with only warnings
        results[0].violations = [(EnforcementAction.WARN, "Warning")]
        assert _determine_overall_verdict(results) == SimulationVerdict.WARN
        
        # Test FAIL with blocking violation
        results[0].violations = [(EnforcementAction.BLOCK, "Blocked")]
        assert _determine_overall_verdict(results) == SimulationVerdict.FAIL
        
        # Test FAIL with throttling violation
        results[0].violations = [(EnforcementAction.THROTTLE, "Throttled")]
        assert _determine_overall_verdict(results) == SimulationVerdict.FAIL
        
        # Test WARN with mixed violations
        results[0].violations = [
            (EnforcementAction.WARN, "Warning"),
            (EnforcementAction.THROTTLE, "Throttled")
        ]
        assert _determine_overall_verdict(results) == SimulationVerdict.FAIL
        
        # Test multiple features - most severe wins
        results.append(
            FeatureSimulationResult(
                feature="test2",
                model="model2",
                estimated_monthly_cost=20.0,
                violations=[(EnforcementAction.WARN, "Warning")],
                anomalies=[]
            )
        )
        assert _determine_overall_verdict(results) == SimulationVerdict.FAIL
