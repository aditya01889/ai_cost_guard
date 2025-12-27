"""
Unit tests for pricing calculations.

Tests cost accuracy, rounding behavior, and error handling.
"""

import pytest
from decimal import Decimal

from ai_cost_guard.core.pricing import calculate_cost, PRICING_TABLE
from ai_cost_guard.core.token_counter import TokenUsage


class TestTokenUsage:
    """Test TokenUsage dataclass."""
    
    def test_total_tokens_calculation(self):
        """Verify total_tokens is computed correctly."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150
    
    def test_zero_tokens(self):
        """Verify zero token handling."""
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        assert usage.total_tokens == 0


class TestPricingTable:
    """Test pricing table functionality."""
    
    def test_get_supported_model(self):
        """Verify pricing retrieval for supported models."""
        gpt4_pricing = PRICING_TABLE.get_pricing("gpt-4")
        assert gpt4_pricing.prompt_cost_per_1k == Decimal("30.00")
        assert gpt4_pricing.completion_cost_per_1k == Decimal("60.00")
    
    def test_unsupported_model_raises_error(self):
        """Verify error for unknown models."""
        with pytest.raises(ValueError, match="Unsupported model: unknown-model"):
            PRICING_TABLE.get_pricing("unknown-model")


class TestCostCalculation:
    """Test cost calculation accuracy and rounding."""
    
    def test_exact_cost_gpt4(self):
        """Verify exact cost calculation for GPT-4."""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost = calculate_cost("gpt-4", usage)
        # Prompt: 1000/1000 * $30.00 = $30.00
        # Completion: 500/1000 * $60.00 = $30.00
        # Total: $60.00
        assert cost == 60.00
    
    def test_exact_cost_gpt35_turbo(self):
        """Verify exact cost calculation for GPT-3.5-Turbo."""
        usage = TokenUsage(prompt_tokens=2000, completion_tokens=1000)
        cost = calculate_cost("gpt-3.5-turbo", usage)
        # Prompt: 2000/1000 * $1.50 = $3.00
        # Completion: 1000/1000 * $2.00 = $2.00
        # Total: $5.00
        assert cost == 5.00
    
    def test_exact_cost_claude3_opus(self):
        """Verify exact cost calculation for Claude-3-Opus."""
        usage = TokenUsage(prompt_tokens=1500, completion_tokens=300)
        cost = calculate_cost("claude-3-opus", usage)
        # Prompt: 1500/1000 * $15.00 = $22.50
        # Completion: 300/1000 * $75.00 = $22.50
        # Total: $45.00
        assert cost == 45.00
    
    def test_rounding_up_behavior(self):
        """Verify costs round UP (conservative bias)."""
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
        cost = calculate_cost("gpt-3.5-turbo", usage)
        # Prompt: 1/1000 * $1.50 = $0.0015
        # Completion: 1/1000 * $2.00 = $0.0020
        # Total: $0.0035 -> should round UP to $0.01
        assert cost == 0.01
    
    def test_rounding_up_edge_case(self):
        """Verify rounding up on exact cent boundary."""
        usage = TokenUsage(prompt_tokens=667, completion_tokens=0)
        cost = calculate_cost("gpt-3.5-turbo", usage)
        # Prompt: 667/1000 * $1.50 = $1.0005 -> should round UP to $1.01
        assert cost == 1.01
    
    def test_large_token_counts(self):
        """Verify calculation with very large token counts."""
        usage = TokenUsage(prompt_tokens=1000000, completion_tokens=500000)
        cost = calculate_cost("gpt-4", usage)
        # Prompt: 1000000/1000 * $30.00 = $30,000.00
        # Completion: 500000/1000 * $60.00 = $30,000.00
        # Total: $60,000.00
        assert cost == 60000.00
    
    def test_zero_tokens_cost(self):
        """Verify cost calculation with zero tokens."""
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        cost = calculate_cost("gpt-4", usage)
        assert cost == 0.00
    
    def test_prompt_only_cost(self):
        """Verify cost calculation with only prompt tokens."""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=0)
        cost = calculate_cost("gpt-4", usage)
        # Prompt: 1000/1000 * $30.00 = $30.00
        # Completion: 0/1000 * $60.00 = $0.00
        # Total: $30.00
        assert cost == 30.00
    
    def test_completion_only_cost(self):
        """Verify cost calculation with only completion tokens."""
        usage = TokenUsage(prompt_tokens=0, completion_tokens=1000)
        cost = calculate_cost("gpt-4", usage)
        # Prompt: 0/1000 * $30.00 = $0.00
        # Completion: 1000/1000 * $60.00 = $60.00
        # Total: $60.00
        assert cost == 60.00
    
    def test_unknown_model_error(self):
        """Verify error handling for unknown models."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        with pytest.raises(ValueError, match="Unsupported model: unknown-model"):
            calculate_cost("unknown-model", usage)
    
    def test_fractional_token_calculation(self):
        """Verify precise calculation with fractional tokens."""
        usage = TokenUsage(prompt_tokens=333, completion_tokens=667)
        cost = calculate_cost("gpt-3.5-turbo", usage)
        # Prompt: 333/1000 * $1.50 = $0.4995
        # Completion: 667/1000 * $2.00 = $1.334
        # Total: $1.8335 -> should round UP to $1.84
        assert cost == 1.84
