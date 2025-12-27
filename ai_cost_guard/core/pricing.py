"""
Pricing calculations and rate management.

Handles cost computations for various AI models and services.
"""

from dataclasses import dataclass
from typing import Dict
from decimal import Decimal, ROUND_UP

from .token_counter import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """Per-token pricing for a specific model."""
    prompt_cost_per_1k: Decimal  # Cost per 1K prompt tokens
    completion_cost_per_1k: Decimal  # Cost per 1K completion tokens


@dataclass(frozen=True)
class PricingTable:
    """Fixed pricing table for supported models."""
    prices: Dict[str, ModelPricing]

    def get_pricing(self, model: str) -> ModelPricing:
        """Get pricing for a specific model.
        
        Args:
            model: Model identifier
            
        Returns:
            ModelPricing for the model
            
        Raises:
            ValueError: If model is not supported
        """
        if model not in self.prices:
            raise ValueError(f"Unsupported model: {model}")
        return self.prices[model]


# Fixed pricing table - no dynamic fetching, no defaults
PRICING_TABLE = PricingTable({
    "gpt-4": ModelPricing(
        prompt_cost_per_1k=Decimal("30.00"),
        completion_cost_per_1k=Decimal("60.00")
    ),
    "gpt-3.5-turbo": ModelPricing(
        prompt_cost_per_1k=Decimal("1.50"),
        completion_cost_per_1k=Decimal("2.00")
    ),
    "claude-3-opus": ModelPricing(
        prompt_cost_per_1k=Decimal("15.00"),
        completion_cost_per_1k=Decimal("75.00")
    )
})


def calculate_cost(model: str, usage: TokenUsage) -> float:
    """Calculate total cost for model usage with conservative rounding.
    
    Args:
        model: Model identifier
        usage: Token usage data
        
    Returns:
        Total cost rounded UP to 2 decimal places
        
    Raises:
        ValueError: If model is not supported
    """
    pricing = PRICING_TABLE.get_pricing(model)
    
    # Calculate prompt cost: (tokens / 1000) * cost_per_1k
    prompt_cost = (Decimal(usage.prompt_tokens) / Decimal("1000")) * pricing.prompt_cost_per_1k
    
    # Calculate completion cost: (tokens / 1000) * cost_per_1k
    completion_cost = (Decimal(usage.completion_tokens) / Decimal("1000")) * pricing.completion_cost_per_1k
    
    # Total cost with conservative rounding (always round UP)
    total_cost = prompt_cost + completion_cost
    rounded_cost = total_cost.quantize(Decimal("0.01"), rounding=ROUND_UP)
    
    return float(rounded_cost)
