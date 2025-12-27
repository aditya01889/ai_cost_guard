"""
CLI interface for AI Cost Guard.

Provides command-line access to all tool functionality.
"""

import os
import sys
from typing import Optional
import sqlite3

import typer
from rich.console import Console
from rich.table import Table

from ai_cost_guard.core.guardrails import GuardrailConfig
from ai_cost_guard.core.simulation import (
    SimulationVerdict,
    simulate_cost_impact
)
from ai_cost_guard.storage.repository import get_repository, initialize_schema

app = typer.Typer()
console = Console()

# Exit codes - WARN is non-failing (0) as per requirements
EXIT_CODE_PASS = 0
EXIT_CODE_WARN = 0  # Non-failing warning
EXIT_CODE_FAIL = 1  # Failing error


def _verdict_to_exit_code(verdict: SimulationVerdict) -> int:
    """Convert simulation verdict to CLI exit code."""
    return {
        SimulationVerdict.PASS: EXIT_CODE_PASS,
        SimulationVerdict.WARN: EXIT_CODE_WARN,
        SimulationVerdict.FAIL: EXIT_CODE_FAIL,
    }[verdict]


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """AI Cost Guard CLI."""
    if ctx.invoked_subcommand is None:
        console.print("AI Cost Guard - Use --help to see available commands")


@app.command()
def status():
    """Check initialization status of AI Cost Guard."""
    console.print("[green]✓[/] AI Cost Guard is initialized")

@app.command()
def init():
    """Initialize the AI Cost Guard database."""
    try:
        initialize_schema()
        console.print("[green]✓[/] Database initialized successfully")
        sys.exit(EXIT_CODE_PASS)
    except Exception as e:
        console.print(f"[red]Error initializing database:[/] {str(e)}")
        sys.exit(EXIT_CODE_FAIL)
        
@app.command()
def simulate(
    feature: Optional[str] = typer.Option(
        None, 
        "--feature", 
        "-f", 
        help="Filter simulation to a specific feature"
    ),
    max_cost: Optional[float] = typer.Option(
        None,
        "--max-cost",
        "-m",
        help="Set maximum allowed cost per request"
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        "-b",
        help="Set monthly budget limit"
    ),
    enforced: bool = typer.Option(
        False,
        "--enforced",
        "-e",
        help="Exit with error code if simulation fails"
    )
):
    """
    Simulate cost impact of current usage patterns.
    
    This is a read-only operation that predicts potential guardrail violations
    based on historical data and current configuration.
    
    Note: Simulation requires historical usage data. To generate data:
    1. Enable the SDK in your application
    2. Make some API calls through the SDK
    3. Run this command again to see the simulation results
    """
    try:
        # Initialize repository and config
        repository = get_repository()
        
        # Create guardrail config from CLI args
        config = GuardrailConfig(
            max_cost_per_request=max_cost,
            budget_limit=budget
        )
        
        # Run simulation
        result = simulate_cost_impact(
            feature=feature,
            config=config,
            repository=repository
        )
        
        # If no data was found, show a helpful message
        if not result.per_feature_results and result.estimated_monthly_impact == 0:
            console.print("\n[bold yellow]No historical AI usage data found[/]")
            console.print("\nTo get started with AI Cost Guard:")
            console.print("1. Enable the SDK in your application")
            console.print("2. Make some API calls through the SDK")
            console.print("3. Run this command again to see the simulation results\n")
            sys.exit(EXIT_CODE_PASS)
        
        # Display results
        _display_simulation_result(result)
        
        # Handle exit code - WARN is non-failing (0) as per requirements
        if enforced and result.overall_verdict == SimulationVerdict.FAIL:
            sys.exit(EXIT_CODE_FAIL)
        else:
            sys.exit(EXIT_CODE_PASS)  # Both PASS and WARN exit with 0
            
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            console.print("\n[bold yellow]No historical AI usage data found[/]")
            console.print("\nTo get started with AI Cost Guard:")
            console.print("1. Run `ai-cost-guard init` to initialize the database")
            console.print("2. Enable the SDK in your application")
            console.print("3. Make some API calls through the SDK")
            console.print("4. Run this command again to see the simulation results\n")
            sys.exit(EXIT_CODE_PASS)
        raise
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")
        sys.exit(EXIT_CODE_FAIL)


def _format_currency(amount: float) -> str:
    """Format currency with proper symbols and formatting."""
    return f"${abs(amount):,.2f}"

def _format_percent_change(before: float, after: float) -> str:
    """Format percentage change with sign."""
    if before == 0:
        return "N/A"
    percent = ((after - before) / before) * 100
    return f"{'+' if percent >= 0 else ''}{percent:,.1f}%"

def _display_simulation_result(result):
    """Display simulation results in a clean, financial format."""
    console.print("\n[bold]AI Cost Simulation Result[/bold]")
    console.print("-" * 40)
    
    if not result.per_feature_results:
        console.print("\n[dim]No usage data found for simulation.[/]")
        return
    
    for feature in result.per_feature_results:
        # Simplified metrics - in a real app, these would come from baseline
        baseline_cost = feature.estimated_monthly_cost / 2  # Mock baseline
        
        console.print(f"\n[bold]Feature:[/bold] {feature.feature}")
        console.print(f"Baseline cost/request: {_format_currency(baseline_cost)}")
        console.print(f"Simulated cost/request: {_format_currency(feature.estimated_monthly_cost)}")
        console.print(f"Change: {_format_percent_change(baseline_cost, feature.estimated_monthly_cost)}")
        console.print(f"Estimated monthly impact: {_format_currency(result.estimated_monthly_impact)}")
        
        # Show violations if any
        if feature.violations:
            for action, message in feature.violations:
                console.print(f"\n[bold]Verdict:[/bold] {action.name} ({message})")
        
        print()  # Add spacing between features


if __name__ == "__main__":
    app()
