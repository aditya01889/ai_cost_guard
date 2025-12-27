"""
Configuration management and loading.

Handles application settings and environment variables.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import yaml


class BreachAction(Enum):
    """Actions to take when guardrails are breached."""
    WARN = "warn"
    BLOCK = "block"
    THROTTLE = "throttle"
    DOWNGRADE = "downgrade"


@dataclass(frozen=True)
class BudgetConfig:
    """Budget limits for cost control."""
    daily: float
    monthly: float
    
    def __post_init__(self):
        """Validate budget values are positive."""
        if self.daily <= 0:
            raise ValueError("daily budget must be > 0")
        if self.monthly <= 0:
            raise ValueError("monthly budget must be > 0")


@dataclass(frozen=True)
class FeatureGuardrailConfig:
    """Guardrail configuration for a specific feature."""
    max_cost_per_request: float
    action_on_breach: BreachAction
    
    def __post_init__(self):
        """Validate feature guardrail values."""
        if self.max_cost_per_request <= 0:
            raise ValueError("max_cost_per_request must be > 0")


@dataclass(frozen=True)
class GuardrailConfig:
    """Complete guardrail configuration."""
    budget: BudgetConfig
    features: Dict[str, FeatureGuardrailConfig]
    defaults: FeatureGuardrailConfig
    
    def get_feature_config(self, feature: str) -> FeatureGuardrailConfig:
        """Get configuration for a specific feature, using defaults if not specified."""
        return self.features.get(feature, self.defaults)


def load_guardrail_config(path: str) -> GuardrailConfig:
    """Load and validate guardrail configuration from YAML file.
    
    Strict validation ensures no silent misconfigurations that could
    lead to unexpected cost overruns or security issues.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Validated GuardrailConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If configuration is invalid
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Guardrail config file not found: {path}")
    
    # Load YAML content
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {path}: {e}")
    
    if not raw_config:
        raise ValueError("Configuration file is empty")
    
    # Validate top-level structure
    allowed_top_keys = {'budget', 'features', 'defaults'}
    unknown_keys = set(raw_config.keys()) - allowed_top_keys
    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {unknown_keys}")
    
    # Parse and validate budget
    if 'budget' not in raw_config:
        raise ValueError("Missing required 'budget' section")
    
    budget_data = raw_config['budget']
    if not isinstance(budget_data, dict):
        raise ValueError("'budget' must be a dictionary")
    
    allowed_budget_keys = {'daily', 'monthly'}
    unknown_budget_keys = set(budget_data.keys()) - allowed_budget_keys
    if unknown_budget_keys:
        raise ValueError(f"Unknown budget keys: {unknown_budget_keys}")
    
    if 'daily' not in budget_data:
        raise ValueError("Missing required 'daily' budget")
    if 'monthly' not in budget_data:
        raise ValueError("Missing required 'monthly' budget")
    
    budget = BudgetConfig(
        daily=float(budget_data['daily']),
        monthly=float(budget_data['monthly'])
    )
    
    # Parse and validate defaults
    if 'defaults' not in raw_config:
        raise ValueError("Missing required 'defaults' section")
    
    defaults_data = raw_config['defaults']
    if not isinstance(defaults_data, dict):
        raise ValueError("'defaults' must be a dictionary")
    
    defaults = _parse_feature_config(defaults_data, "defaults")
    
    # Parse and validate features
    features_data = raw_config.get('features', {})
    if not isinstance(features_data, dict):
        raise ValueError("'features' must be a dictionary")
    
    features = {}
    for feature_name, feature_data in features_data.items():
        if not isinstance(feature_data, dict):
            raise ValueError(f"Feature '{feature_name}' must be a dictionary")
        features[feature_name] = _parse_feature_config(feature_data, f"features.{feature_name}")
    
    return GuardrailConfig(
        budget=budget,
        features=features,
        defaults=defaults
    )


def _parse_feature_config(data: Dict, path: str) -> FeatureGuardrailConfig:
    """Parse and validate feature guardrail configuration.
    
    Args:
        data: Feature configuration data
        path: Path for error messages
        
    Returns:
        Validated FeatureGuardrailConfig
        
    Raises:
        ValueError: If configuration is invalid
    """
    allowed_keys = {'max_cost_per_request', 'action_on_breach'}
    unknown_keys = set(data.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown keys in {path}: {unknown_keys}")
    
    # Validate max_cost_per_request
    if 'max_cost_per_request' not in data:
        raise ValueError(f"Missing required 'max_cost_per_request' in {path}")
    
    max_cost = data['max_cost_per_request']
    if not isinstance(max_cost, (int, float)) or max_cost <= 0:
        raise ValueError(f"'max_cost_per_request' in {path} must be > 0")
    
    # Validate action_on_breach
    if 'action_on_breach' not in data:
        raise ValueError(f"Missing required 'action_on_breach' in {path}")
    
    action_str = data['action_on_breach']
    if not isinstance(action_str, str):
        raise ValueError(f"'action_on_breach' in {path} must be a string")
    
    try:
        action = BreachAction(action_str.lower())
    except ValueError:
        valid_actions = [action.value for action in BreachAction]
        raise ValueError(f"'action_on_breach' in {path} must be one of: {valid_actions}")
    
    return FeatureGuardrailConfig(
        max_cost_per_request=float(max_cost),
        action_on_breach=action
    )
