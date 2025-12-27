"""
Unit tests for configuration loading and validation.

Tests strict validation and error handling for guardrail configs.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ai_cost_guard.config.loader import (
    load_guardrail_config,
    BudgetConfig,
    FeatureGuardrailConfig,
    GuardrailConfig,
    BreachAction
)


class TestConfigLoading:
    """Test configuration loading and validation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _write_config(self, config_data: dict, filename: str = "config.yaml") -> str:
        """Write configuration data to temporary file."""
        config_path = os.path.join(self.temp_dir, filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        return config_path
    
    def test_valid_config_loads_correctly(self):
        """Test that a valid configuration loads correctly."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "features": {
                "chat_completion": {
                    "max_cost_per_request": 10.0,
                    "action_on_breach": "block"
                }
            }
        }
        
        config_path = self._write_config(config_data)
        config = load_guardrail_config(config_path)
        
        # Verify budget
        assert config.budget.daily == 100.0
        assert config.budget.monthly == 3000.0
        
        # Verify defaults
        assert config.defaults.max_cost_per_request == 5.0
        assert config.defaults.action_on_breach == BreachAction.WARN
        
        # Verify features
        assert len(config.features) == 1
        chat_config = config.features["chat_completion"]
        assert chat_config.max_cost_per_request == 10.0
        assert chat_config.action_on_breach == BreachAction.BLOCK
    
    def test_config_without_features_loads_correctly(self):
        """Test that config without features section loads correctly."""
        config_data = {
            "budget": {
                "daily": 50.0,
                "monthly": 1500.0
            },
            "defaults": {
                "max_cost_per_request": 2.0,
                "action_on_breach": "throttle"
            }
        }
        
        config_path = self._write_config(config_data)
        config = load_guardrail_config(config_path)
        
        assert len(config.features) == 0
        assert config.defaults.action_on_breach == BreachAction.THROTTLE
    
    def test_get_feature_config_with_defaults(self):
        """Test feature config lookup with defaults fallback."""
        config_data = {
            "budget": {"daily": 100.0, "monthly": 3000.0},
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "features": {
                "chat_completion": {
                    "max_cost_per_request": 10.0,
                    "action_on_breach": "block"
                }
            }
        }
        
        config_path = self._write_config(config_data)
        config = load_guardrail_config(config_path)
        
        # Test configured feature
        chat_config = config.get_feature_config("chat_completion")
        assert chat_config.max_cost_per_request == 10.0
        assert chat_config.action_on_breach == BreachAction.BLOCK
        
        # Test unconfigured feature (should use defaults)
        embedding_config = config.get_feature_config("embedding")
        assert embedding_config.max_cost_per_request == 5.0
        assert embedding_config.action_on_breach == BreachAction.WARN
    
    def test_missing_file_raises_error(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError, match="Guardrail config file not found"):
            load_guardrail_config("nonexistent.yaml")
    
    def test_empty_config_raises_error(self):
        """Test that empty config file raises error."""
        config_path = self._write_config({})
        
        with pytest.raises(ValueError, match="Configuration file is empty"):
            load_guardrail_config(config_path)
    
    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises error."""
        config_path = os.path.join(self.temp_dir, "invalid.yaml")
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            load_guardrail_config(config_path)
    
    def test_missing_budget_raises_error(self):
        """Test that missing budget section raises error."""
        config_data = {
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Missing required 'budget' section"):
            load_guardrail_config(config_path)
    
    def test_missing_defaults_raises_error(self):
        """Test that missing defaults section raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Missing required 'defaults' section"):
            load_guardrail_config(config_path)
    
    def test_missing_daily_budget_raises_error(self):
        """Test that missing daily budget raises error."""
        config_data = {
            "budget": {
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Missing required 'daily' budget"):
            load_guardrail_config(config_path)
    
    def test_missing_monthly_budget_raises_error(self):
        """Test that missing monthly budget raises error."""
        config_data = {
            "budget": {
                "daily": 100.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Missing required 'monthly' budget"):
            load_guardrail_config(config_path)
    
    def test_negative_daily_budget_raises_error(self):
        """Test that negative daily budget raises error."""
        config_data = {
            "budget": {
                "daily": -10.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="daily budget must be > 0"):
            load_guardrail_config(config_path)
    
    def test_negative_monthly_budget_raises_error(self):
        """Test that negative monthly budget raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": -100.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="monthly budget must be > 0"):
            load_guardrail_config(config_path)
    
    def test_zero_daily_budget_raises_error(self):
        """Test that zero daily budget raises error."""
        config_data = {
            "budget": {
                "daily": 0.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="daily budget must be > 0"):
            load_guardrail_config(config_path)
    
    def test_invalid_action_on_breach_raises_error(self):
        """Test that invalid action_on_breach raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "invalid_action"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="must be one of"):
            load_guardrail_config(config_path)
    
    def test_negative_max_cost_per_request_raises_error(self):
        """Test that negative max_cost_per_request raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": -5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="must be > 0"):
            load_guardrail_config(config_path)
    
    def test_zero_max_cost_per_request_raises_error(self):
        """Test that zero max_cost_per_request raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 0.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="must be > 0"):
            load_guardrail_config(config_path)
    
    def test_unknown_top_level_keys_raise_error(self):
        """Test that unknown top-level keys raise error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "unknown_key": "value"
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Unknown configuration keys"):
            load_guardrail_config(config_path)
    
    def test_unknown_budget_keys_raise_error(self):
        """Test that unknown budget keys raise error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0,
                "weekly": 700.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Unknown budget keys"):
            load_guardrail_config(config_path)
    
    def test_unknown_feature_keys_raise_error(self):
        """Test that unknown feature keys raise error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "features": {
                "chat_completion": {
                    "max_cost_per_request": 10.0,
                    "action_on_breach": "block",
                    "unknown_setting": "value"
                }
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Unknown keys in features.chat_completion"):
            load_guardrail_config(config_path)
    
    def test_partial_feature_config_raises_error(self):
        """Test that partial feature config raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "features": {
                "chat_completion": {
                    "max_cost_per_request": 10.0
                    # Missing action_on_breach
                }
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Missing required 'action_on_breach' in features.chat_completion"):
            load_guardrail_config(config_path)
    
    def test_partial_defaults_config_raises_error(self):
        """Test that partial defaults config raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0
                # Missing action_on_breach
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Missing required 'action_on_breach' in defaults"):
            load_guardrail_config(config_path)
    
    def test_invalid_budget_type_raises_error(self):
        """Test that invalid budget type raises error."""
        config_data = {
            "budget": "not_a_dict",
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="'budget' must be a dictionary"):
            load_guardrail_config(config_path)
    
    def test_invalid_features_type_raises_error(self):
        """Test that invalid features type raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "features": "not_a_dict"
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="'features' must be a dictionary"):
            load_guardrail_config(config_path)
    
    def test_invalid_defaults_type_raises_error(self):
        """Test that invalid defaults type raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": "not_a_dict"
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="'defaults' must be a dictionary"):
            load_guardrail_config(config_path)
    
    def test_invalid_feature_type_raises_error(self):
        """Test that invalid feature type raises error."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "warn"
            },
            "features": {
                "chat_completion": "not_a_dict"
            }
        }
        
        config_path = self._write_config(config_data)
        with pytest.raises(ValueError, match="Feature 'chat_completion' must be a dictionary"):
            load_guardrail_config(config_path)
    
    def test_case_insensitive_action_on_breach(self):
        """Test that action_on_breach is case insensitive."""
        config_data = {
            "budget": {
                "daily": 100.0,
                "monthly": 3000.0
            },
            "defaults": {
                "max_cost_per_request": 5.0,
                "action_on_breach": "BLOCK"
            }
        }
        
        config_path = self._write_config(config_data)
        config = load_guardrail_config(config_path)
        
        assert config.defaults.action_on_breach == BreachAction.BLOCK
    
    def test_all_valid_actions(self):
        """Test all valid breach actions."""
        valid_actions = ["warn", "block", "throttle", "downgrade"]
        
        for action in valid_actions:
            config_data = {
                "budget": {
                    "daily": 100.0,
                    "monthly": 3000.0
                },
                "defaults": {
                    "max_cost_per_request": 5.0,
                    "action_on_breach": action
                }
            }
            
            config_path = self._write_config(config_data)
            config = load_guardrail_config(config_path)
            
            expected_action = BreachAction(action)
            assert config.defaults.action_on_breach == expected_action
