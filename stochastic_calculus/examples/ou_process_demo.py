"""Comprehensive demo for Ornstein-Uhlenbeck processes.

This demo showcases:
- Single and multi-process OU simulation
- Parameter impact analysis
- Mean reversion characteristics
- Generic visualization techniques adaptable to any process
- Parameter estimation and statistical analysis
"""

import sys
import os

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import numpy as np
from stochastic_calculus.processes.mean_reverting import (
    OrnsteinUhlenbeckProcess,
    OUParameters,
)
from stochastic_calculus.visualization import ProcessPlotter, StatisticalPlotter


def demo_single_ou_process():
    """Demonstrate single OU process with parameter analysis."""
    print("\n" + "=" * 60)
    print("SINGLE ORNSTEIN-UHLENBECK PROCESS DEMONSTRATION")
    print("=" * 60)

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Single OU process with typical parameters
    print("\n1. Single OU Process")
    print("-" * 25)

    params = OUParameters(alpha=2.0, gamma=0.0, beta=0.2)
    ou_single = OrnsteinUhlenbeckProcess(params, n_processes=1)
    result_single = ou_single.simulate(n_steps=2000, dt=1 / 252, random_state=42)

    time_years = np.arange(result_single.paths.shape[0]) / 252

    # Generic path visualization
    plotter.plot_paths(
        data=result_single.paths,
        title="Single Ornstein-Uhlenbeck Process",
        xlabel="Time (years)",
        ylabel="X(t)",
        time_axis=time_years,
        process_labels=["OU Process"],
        figsize=(12, 6),
    )

    # Statistical analysis using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_single.paths,
        title="OU Process - Statistical Properties",
        process_labels=["X(t)"],
    )

    print(f"   Process parameters: α={params.alpha}, γ={params.gamma}, β={params.beta}")


def demo_parameter_impact():
    """Demonstrate impact of different OU parameters."""
    print("\n" + "=" * 60)
    print("OU PARAMETER IMPACT ANALYSIS")
    print("=" * 60)

    plotter = ProcessPlotter()

    # Mean reversion speed impact
    print("\n1. Mean Reversion Speed Impact")
    print("-" * 35)

    alpha_values = [0.5, 2.0, 5.0]
    alpha_results = []
    alpha_labels = []

    for alpha in alpha_values:
        print(f"   Simulating with α = {alpha}")
        params = OUParameters(alpha=alpha, gamma=0.0, beta=0.2)
        ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
        result = ou.simulate(n_steps=1000, dt=1 / 252, random_state=42)
        alpha_results.append(result.paths[:, 0])
        alpha_labels.append(f"α = {alpha}")

    # Generic visualization for parameter comparison
    alpha_matrix = np.column_stack(alpha_results)
    time_years = np.arange(alpha_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=alpha_matrix,
        title="OU Process - Mean Reversion Speed Impact",
        xlabel="Time (years)",
        ylabel="X(t)",
        time_axis=time_years,
        process_labels=alpha_labels,
        figsize=(12, 6),
    )

    # Long-term mean impact
    print("\n2. Long-term Mean Impact")
    print("-" * 30)

    gamma_values = [-0.5, 0.0, 0.5]
    gamma_results = []
    gamma_labels = []

    for gamma in gamma_values:
        print(f"   Simulating with γ = {gamma}")
        params = OUParameters(alpha=2.0, gamma=gamma, beta=0.2)
        ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
        result = ou.simulate(n_steps=1000, dt=1 / 252, random_state=42)
        gamma_results.append(result.paths[:, 0])
        gamma_labels.append(f"γ = {gamma}")

    gamma_matrix = np.column_stack(gamma_results)
    plotter.plot_paths(
        data=gamma_matrix,
        title="OU Process - Long-term Mean Impact",
        xlabel="Time (years)",
        ylabel="X(t)",
        time_axis=time_years,
        process_labels=gamma_labels,
        figsize=(12, 6),
    )

    # Volatility impact
    print("\n3. Volatility Impact")
    print("-" * 20)

    beta_values = [0.1, 0.2, 0.4]
    beta_results = []
    beta_labels = []

    for beta in beta_values:
        print(f"   Simulating with β = {beta}")
        params = OUParameters(alpha=2.0, gamma=0.0, beta=beta)
        ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
        result = ou.simulate(n_steps=1000, dt=1 / 252, random_state=42)
        beta_results.append(result.paths[:, 0])
        beta_labels.append(f"β = {beta}")

    beta_matrix = np.column_stack(beta_results)
    plotter.plot_paths(
        data=beta_matrix,
        title="OU Process - Volatility Impact",
        xlabel="Time (years)",
        ylabel="X(t)",
        time_axis=time_years,
        process_labels=beta_labels,
        figsize=(12, 6),
    )


def demo_multi_process_ou():
    """Demonstrate multiple correlated OU processes."""
    print("\n" + "=" * 60)
    print("MULTI-PROCESS OU DEMONSTRATION")
    print("=" * 60)

    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Create different OU processes with varying characteristics
    print("\n1. Multiple OU Processes with Different Parameters")
    print("-" * 50)

    params = (
        OUParameters(alpha=1.0, gamma=0.0, beta=0.15),  # Slow reversion
        OUParameters(alpha=3.0, gamma=0.1, beta=0.2),  # Fast reversion, positive mean
        OUParameters(
            alpha=2.0, gamma=-0.1, beta=0.25
        ),  # Medium reversion, negative mean
    )

    # Test different correlation levels
    correlations = [0.0, 0.5, 0.9]

    for rho in correlations:
        print(f"   Simulating with correlation ρ = {rho}")

        ou_multi = OrnsteinUhlenbeckProcess(params, correlation=rho)
        result_multi = ou_multi.simulate(n_steps=1000, dt=1 / 252, random_state=42)

        time_years = np.arange(result_multi.paths.shape[0]) / 252

        # Generic visualization for multiple processes
        plotter.plot_paths(
            data=result_multi.paths,
            title=f"Multi-Process OU (ρ = {rho})",
            xlabel="Time (years)",
            ylabel="X(t)",
            time_axis=time_years,
            process_labels=["Slow Reversion", "Fast + Mean", "Medium - Mean"],
            figsize=(12, 6),
        )

        # Calculate increments for correlation analysis
        increments = np.diff(result_multi.paths, axis=0)

        # Correlation analysis using generic heatmap
        plotter.plot_correlation_heatmap(
            data=increments,
            title=f"OU Correlation Matrix (Target ρ = {rho})",
            process_labels=["Process 1", "Process 2", "Process 3"],
        )

        # Calculate and display actual correlation
        actual_corr = np.corrcoef(increments.T)
        print(f"   Target correlation: {rho:.3f}")
        print(f"   Actual correlation: {actual_corr[0,1]:.3f}")


def demo_mean_reversion_analysis():
    """Analyze mean reversion characteristics."""
    print("\n" + "=" * 60)
    print("MEAN REVERSION ANALYSIS")
    print("=" * 60)

    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Generate OU process starting from extreme value
    print("\n1. Mean Reversion from Extreme Starting Point")
    print("-" * 45)

    params = OUParameters(alpha=3.0, gamma=0.0, beta=0.2)
    ou = OrnsteinUhlenbeckProcess(params, n_processes=1)

    # Start from different initial values
    initial_values = [-2.0, 0.0, 2.0]
    reversion_results = []
    reversion_labels = []

    for x0 in initial_values:
        result = ou.simulate_with_initial_value(n_steps=1000, dt=1 / 252, X_0=x0, random_state=42)
        reversion_results.append(result.paths[:, 0])
        reversion_labels.append(f"X = {x0}")

    reversion_matrix = np.column_stack(reversion_results)
    time_years = np.arange(reversion_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=reversion_matrix,
        title="OU Mean Reversion from Different Starting Points",
        xlabel="Time (years)",
        ylabel="X(t)",
        time_axis=time_years,
        process_labels=reversion_labels,
        figsize=(12, 6),
    )

    # Half-life analysis
    print("\n2. Half-life Analysis")
    print("-" * 20)

    alpha_values = [0.5, 1.0, 2.0, 4.0]
    half_lives = []

    for alpha in alpha_values:
        half_life = np.log(2) / alpha * 252  # Convert to days
        half_lives.append(half_life)
        print(f"   α = {alpha:.1f} → Half-life = {half_life:.0f} days")

    # Empirical half-life verification
    print("\n3. Empirical Half-life Verification")
    print("-" * 35)

    params = OUParameters(alpha=2.0, gamma=0.0, beta=0.2)
    ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
    result = ou.simulate_with_initial_value(n_steps=2000, dt=1 / 252, X_0=1.0, random_state=42)

    theoretical_half_life = np.log(2) / params.alpha * 252
    print(f"   Theoretical half-life: {theoretical_half_life:.0f} days")

    # Find empirical half-life (time to reach 0.5 from 1.0)
    path = result.paths[:, 0]
    half_point_indices = np.where(path <= 0.5)[0]
    if len(half_point_indices) > 0:
        empirical_half_life = half_point_indices[0]
        print(f"   Empirical half-life: {empirical_half_life:.0f} days")

    print("   Mean reversion analysis complete")


def demo_parameter_estimation():
    """Demonstrate OU parameter estimation."""
    print("\n" + "=" * 60)
    print("OU PARAMETER ESTIMATION")
    print("=" * 60)

    stat_plotter = StatisticalPlotter()

    # Generate data with known parameters
    true_params = OUParameters(alpha=2.5, gamma=0.1, beta=0.3)
    ou_estimation = OrnsteinUhlenbeckProcess(true_params, n_processes=1)
    result_estimation = ou_estimation.simulate(
        n_steps=2000, dt=1 / 252, random_state=42
    )

    # Estimate parameters from the generated data
    from stochastic_calculus.processes.mean_reverting.ou_process import (
        estimate_ou_parameters,
    )

    estimated_params = estimate_ou_parameters(result_estimation.paths[:, 0], dt=1 / 252)

    print(
        f"   True parameters:      α = {true_params.alpha:.3f}, γ = {true_params.gamma:.3f}, β = {true_params.beta:.3f}"
    )
    print(
        f"   Estimated parameters: α = {estimated_params.alpha:.3f}, γ = {estimated_params.gamma:.3f}, β = {estimated_params.beta:.3f}"
    )

    # Visualize estimation quality using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_estimation.paths,
        title="OU Parameter Estimation - Data Quality Check",
        process_labels=["X(t)"],
    )


def main():

    try:
        # Run all demonstrations
        demo_single_ou_process()
        demo_parameter_impact()
        demo_multi_process_ou()
        demo_mean_reversion_analysis()
        demo_parameter_estimation()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
