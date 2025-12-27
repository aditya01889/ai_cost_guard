"""
Tests for the CLI interface.
"""
import sys
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from ai_cost_guard.cli.main import app, EXIT_CODE_PASS, EXIT_CODE_FAIL
from ai_cost_guard.core.simulation import SimulationVerdict, SimulationResult, FeatureSimulationResult
from ai_cost_guard.core.guardrails import EnforcementAction

runner = CliRunner()

@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
    with patch('ai_cost_guard.cli.main.get_repository') as mock_repo:
        yield mock_repo

@pytest.fixture
def mock_simulate():
    """Mock the simulate_cost_impact function."""
    with patch('ai_cost_guard.cli.main.simulate_cost_impact') as mock:
        yield mock

class TestCLI:
    """Test CLI commands."""

    def test_simulate_command_basic(self, mock_repository, mock_simulate):
        """Test basic simulate command."""
        # Setup mock
        mock_result = SimulationResult(
            per_feature_results=[],
            overall_verdict=SimulationVerdict.PASS,
            estimated_monthly_impact=100.0
        )
        mock_simulate.return_value = mock_result

        # Run CLI
        result = runner.invoke(app, ["simulate"])
        
        # Verify
        assert result.exit_code == EXIT_CODE_PASS
        assert "AI Cost Simulation Result" in result.output

    def test_simulate_with_feature_flag(self, mock_repository, mock_simulate):
        """Test simulate with feature filter."""
        # Setup mock
        mock_result = SimulationResult(
            per_feature_results=[],
            overall_verdict=SimulationVerdict.PASS,
            estimated_monthly_impact=100.0
        )
        mock_simulate.return_value = mock_result

        # Run CLI with feature filter
        result = runner.invoke(app, ["simulate", "--feature", "test_feature"])
        
        # Verify
        assert result.exit_code == EXIT_CODE_PASS
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["feature"] == "test_feature"

    def test_simulate_enforced_flag_failure(self, mock_repository, mock_simulate):
        """Test enforced flag with failure."""
        # Setup mock with FAIL verdict
        mock_result = SimulationResult(
            per_feature_results=[
                FeatureSimulationResult(
                    feature="test_feature",
                    model="test_model",
                    estimated_monthly_cost=1000.0,
                    violations=[(EnforcementAction.BLOCK, "Cost exceeded")]
                )
            ],
            overall_verdict=SimulationVerdict.FAIL,
            estimated_monthly_impact=1000.0
        )
        mock_simulate.return_value = mock_result

        # Run with enforced flag
        result = runner.invoke(app, ["simulate", "--enforced"])
        
        # Should exit with failure code
        assert result.exit_code == EXIT_CODE_FAIL

    def test_simulate_warning_exits_zero(self, mock_repository, mock_simulate):
        """Test that WARN verdict exits with 0 (success) code."""
        # Setup mock with WARN verdict
        mock_result = SimulationResult(
            per_feature_results=[
                FeatureSimulationResult(
                    feature="test_feature",
                    model="test_model",
                    estimated_monthly_cost=500.0,
                    violations=[(EnforcementAction.WARN, "Approaching limit")]
                )
            ],
            overall_verdict=SimulationVerdict.WARN,
            estimated_monthly_impact=500.0
        )
        mock_simulate.return_value = mock_result

        # Run with enforced flag - should still exit 0 for WARN
        result = runner.invoke(app, ["simulate", "--enforced"])
        
        # Should exit with success code (0)
        assert result.exit_code == EXIT_CODE_PASS
        assert "Verdict: WARN" in result.output

    def test_output_contains_financial_info(self, mock_repository, mock_simulate):
        """Test that output contains financial information."""
        # Setup mock with test data
        mock_result = SimulationResult(
            per_feature_results=[
                FeatureSimulationResult(
                    feature="test_feature",
                    model="test_model",
                    estimated_monthly_cost=1000.0,
                    violations=[]
                )
            ],
            overall_verdict=SimulationVerdict.PASS,
            estimated_monthly_impact=1000.0
        )
        mock_simulate.return_value = mock_result

        # Run CLI
        result = runner.invoke(app, ["simulate"])
        
        # Check for financial information in output
        assert "$" in result.output  # Currency symbol
        assert "cost/request" in result.output
        assert "monthly impact" in result.output.lower()
        assert "1,000.00" in result.output  # Formatted number with thousands separator
