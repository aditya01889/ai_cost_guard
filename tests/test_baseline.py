"""
Unit tests for baseline computation.

Tests deterministic baseline calculation from usage events.
"""

from datetime import datetime, timedelta

import pytest

from ai_cost_guard.core.baseline import (
    compute_baseline,
    BaselineState,
    BaselineMetrics,
    BaselineResult,
    _compute_exact_percentile
)
from ai_cost_guard.storage.models import LLMUsageEvent


class TestExactPercentile:
    """Test exact percentile computation."""
    
    def test_median_even_count(self):
        """Test median computation with even number of values."""
        values = [1.0, 2.0, 3.0, 4.0]
        median = _compute_exact_percentile(values, 50)
        assert median == 2.5  # (2.0 + 3.0) / 2
    
    def test_median_odd_count(self):
        """Test median computation with odd number of values."""
        values = [1.0, 2.0, 3.0]
        median = _compute_exact_percentile(values, 50)
        assert median == 2.0
    
    def test_p90_exact(self):
        """Test P90 computation with exact position."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p90 = _compute_exact_percentile(values, 90)
        assert p90 == 9.1  # Position 8.1: 9.0 + 0.1 * (10.0 - 9.0)
    
    def test_p90_interpolation(self):
        """Test P90 computation with interpolation."""
        values = [10.0, 20.0, 30.0, 40.0]
        p90 = _compute_exact_percentile(values, 90)
        assert p90 == 37.0  # Position 2.7: 30.0 + 0.7 * (40.0 - 30.0)
    
    def test_percentile_zero(self):
        """Test 0th percentile (minimum)."""
        values = [5.0, 10.0, 15.0]
        result = _compute_exact_percentile(values, 0)
        assert result == 5.0
    
    def test_percentile_hundred(self):
        """Test 100th percentile (maximum)."""
        values = [5.0, 10.0, 15.0]
        result = _compute_exact_percentile(values, 100)
        assert result == 15.0
    
    def test_empty_values_raises_error(self):
        """Test that empty values list raises error."""
        with pytest.raises(ValueError, match="Values list cannot be empty"):
            _compute_exact_percentile([], 50)
    
    def test_invalid_percentile_raises_error(self):
        """Test that invalid percentile raises error."""
        values = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            _compute_exact_percentile(values, -1)
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            _compute_exact_percentile(values, 101)


class TestBaselineComputation:
    """Test baseline computation from usage events."""
    
    def create_test_event(self, cost: float, tokens: int, timestamp: datetime) -> LLMUsageEvent:
        """Create a test usage event."""
        return LLMUsageEvent(
            timestamp=timestamp,
            feature="test_feature",
            model="test_model",
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
            estimated_cost=cost
        )
    
    def test_cold_baseline_less_than_20_events(self):
        """Test cold baseline with less than 20 events."""
        now = datetime.now()
        events = [
            self.create_test_event(
                cost=float(i) + 1.0,
                tokens=(i + 1) * 10,
                timestamp=now - timedelta(hours=i)
            )
            for i in range(10)  # 10 events < 20 threshold
        ]
        
        result = compute_baseline(events)
        
        assert result.state == BaselineState.COLD
        assert result.metrics.sample_count == 10
        assert result.metrics.median_cost == 5.5  # Median of 1.0 to 10.0
        assert result.metrics.p90_cost == 9.1  # P90 of 1.0 to 10.0
        assert result.metrics.median_tokens == 55  # Median of 10, 20, ..., 100
    
    def test_warm_baseline_20_or_more_events(self):
        """Test warm baseline with 20 or more events."""
        now = datetime.now()
        events = [
            self.create_test_event(
                cost=float(i) + 1.0,
                tokens=(i + 1) * 10,
                timestamp=now - timedelta(hours=i)
            )
            for i in range(25)  # 25 events >= 20 threshold
        ]
        
        result = compute_baseline(events)
        
        assert result.state == BaselineState.WARM
        assert result.metrics.sample_count == 25
        assert result.metrics.median_cost == 13.0  # Median of 1.0 to 25.0
        assert result.metrics.p90_cost == 22.6  # P90 of 1.0 to 25.0
        assert result.metrics.median_tokens == 130  # Median of 10, 20, ..., 250
    
    def test_empty_events_raises_error(self):
        """Test that empty events list raises error."""
        with pytest.raises(ValueError, match="Events list cannot be empty"):
            compute_baseline([])
    
    def test_missing_cost_raises_error(self):
        """Test that events missing estimated_cost raise error."""
        # Create event without cost
        event = LLMUsageEvent(
            timestamp=datetime.now(),
            feature="test_feature",
            model="test_model",
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
            estimated_cost=None  # Missing cost
        )
        
        with pytest.raises(ValueError, match="Event at index 0 missing estimated_cost"):
            compute_baseline([event])
    
    def test_time_window_enforcement_7_days(self):
        """Test that only events within 7 days are used."""
        now = datetime.now()
        
        # Create events: 5 recent, 5 old (>7 days)
        recent_events = [
            self.create_test_event(
                cost=float(i) + 1.0,  # Different costs: 1.0, 2.0, 3.0, 4.0, 5.0
                tokens=100,
                timestamp=now - timedelta(days=i)
            )
            for i in range(5)  # 0 to 4 days ago
        ]
        
        old_events = [
            self.create_test_event(
                cost=10.0,
                tokens=200,
                timestamp=now - timedelta(days=8 + i)
            )
            for i in range(5)  # 8 to 12 days ago
        ]
        
        all_events = recent_events + old_events
        result = compute_baseline(all_events)
        
        # Should only use recent events
        assert result.metrics.sample_count == 5
        assert result.metrics.median_cost == 3.0  # Median of 1.0 to 5.0
    
    def test_event_limit_enforcement_200_events(self):
        """Test that only last 200 events are used."""
        now = datetime.now()
        
        # Create 250 events
        events = [
            self.create_test_event(
                cost=float(i) + 1.0,
                tokens=(i + 1) * 10,
                timestamp=now - timedelta(minutes=i)
            )
            for i in range(250)  # 250 events > 200 limit
        ]
        
        result = compute_baseline(events)
        
        # Should only use last 200 events
        assert result.metrics.sample_count == 200
        # Median should be of events 0-199 (newest 200)
        assert result.metrics.median_cost == 100.5  # Median of 1.0 to 200.0
    
    def test_deterministic_ordering(self):
        """Test that same input produces same output."""
        now = datetime.now()
        events = [
            self.create_test_event(
                cost=float(i) + 1.0,
                tokens=(i + 1) * 10,
                timestamp=now - timedelta(hours=i)
            )
            for i in range(30)
        ]
        
        # Compute baseline twice
        result1 = compute_baseline(events)
        result2 = compute_baseline(events)
        
        # Results should be identical
        assert result1.metrics.median_cost == result2.metrics.median_cost
        assert result1.metrics.p90_cost == result2.metrics.p90_cost
        assert result1.metrics.median_tokens == result2.metrics.median_tokens
        assert result1.state == result2.state
        assert result1.window_start == result2.window_start
        assert result1.window_end == result2.window_end
    
    def test_time_window_correctness(self):
        """Test that time window is computed correctly."""
        now = datetime.now()
        events = [
            self.create_test_event(
                cost=1.0,
                tokens=100,
                timestamp=now - timedelta(hours=3)
            ),
            self.create_test_event(
                cost=2.0,
                tokens=200,
                timestamp=now - timedelta(hours=2)
            ),
            self.create_test_event(
                cost=3.0,
                tokens=300,
                timestamp=now - timedelta(hours=1)
            )
        ]
        
        result = compute_baseline(events)
        
        # Window should span from oldest to newest event
        # After sorting by timestamp descending: [1h ago, 2h ago, 3h ago]
        assert result.window_end == events[2].timestamp  # Most recent (1 hour ago)
        assert result.window_start == events[0].timestamp  # Oldest (3 hours ago)
        assert result.window_start < result.window_end
    
    def test_median_correctness_with_duplicates(self):
        now = datetime.now()
        events = [
            self.create_test_event(
                cost=5.0,
                tokens=100,
                timestamp=now - timedelta(hours=i)
            )
            for i in range(5)  # 5 events all with cost 5.0
        ]
        
        result = compute_baseline(events)
        
        # Median should be 5.0 (all values are the same)
        assert result.metrics.median_cost == 5.0
        assert result.metrics.p90_cost == 5.0
    
    def test_p90_correctness_with_outliers(self):
        """Test P90 computation with outlier values."""
        now = datetime.now()
        
        # Create events with costs that will produce a P90 > 1.0
        events = [
            self.create_test_event(
                cost=float(i + 1),  # Costs: 1.0, 2.0, 3.0, ..., 20.0
                tokens=100,
                timestamp=now - timedelta(hours=i)
            )
            for i in range(20)  # 20 events with increasing costs
        ]
        
        result = compute_baseline(events)
        
        # P90 should be greater than median and less than max
        assert result.metrics.p90_cost > result.metrics.median_cost
        assert result.metrics.p90_cost < 20.0  # Less than maximum
        assert result.metrics.median_cost == 10.5  # Median of 1.0 to 20.0
    
    def test_token_median_correctness(self):
        """Test token median computation."""
        now = datetime.now()
        events = [
            self.create_test_event(
                cost=1.0,
                tokens=50 + i * 10,  # 50, 60, 70, ..., 140
                timestamp=now - timedelta(hours=i)
            )
            for i in range(10)
        ]
        
        result = compute_baseline(events)
        
        # Median tokens should be 95 (median of 50, 60, ..., 140)
        assert result.metrics.median_tokens == 95
    
    def test_baseline_metrics_validation(self):
        """Test that baseline metrics are validated."""
        # This test ensures the dataclass validation works
        with pytest.raises(ValueError, match="median_cost cannot be negative"):
            BaselineMetrics(
                median_cost=-1.0,
                p90_cost=1.0,
                median_tokens=100,
                sample_count=10
            )
    
    def test_baseline_result_validation(self):
        """Test that baseline result time window is validated."""
        now = datetime.now()
        past = now - timedelta(hours=1)
        
        with pytest.raises(ValueError, match="window_start must be before window_end"):
            BaselineResult(
                metrics=BaselineMetrics(1.0, 2.0, 100, 10),
                state=BaselineState.WARM,
                window_start=now,  # Start after end
                window_end=past
            )
    
    def test_events_outside_window_ignored(self):
        """Test that events outside 7-day window are completely ignored."""
        now = datetime.now()
        
        # Mix of events within and outside window
        events = []
        
        # Events within window (should be used)
        for i in range(5):
            events.append(
                self.create_test_event(
                    cost=float(i) + 1.0,  # Different costs: 1.0, 2.0, 3.0, 4.0, 5.0
                    tokens=100,
                    timestamp=now - timedelta(days=i)
                )
            )
        
        # Events outside window (should be ignored)
        for i in range(5):
            events.append(
                self.create_test_event(
                    cost=100.0,
                    tokens=1000,
                    timestamp=now - timedelta(days=8 + i)
                )
            )
        
        result = compute_baseline(events)
        
        # Should only count events within window
        assert result.metrics.sample_count == 5
        assert result.metrics.median_cost == 3.0  # Median of 1.0, 2.0, 3.0, 4.0, 5.0
