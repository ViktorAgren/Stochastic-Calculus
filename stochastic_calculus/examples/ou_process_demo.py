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
    print("\nSingle Ornstein-Uhlenbeck Process")

    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    print("1. Single OU Process")

    params = OUParameters(alpha=2.0, gamma=0.0, beta=0.2)
    ou_single = OrnsteinUhlenbeckProcess(params, n_processes=1)
    result_single = ou_single.simulate(n_steps=2000, dt=1 / 252, random_state=42)

    time_years = np.arange(result_single.paths.shape[0]) / 252

    plotter.plot_paths(
        data=result_single.paths,
        title="Single Ornstein-Uhlenbeck Process",
        xlabel="Time $t$ (years)",
        ylabel="$X(t)$",
        time_axis=time_years,
        process_labels=["$X(t)$"],
        figsize=(12, 6),
    )

    stat_plotter.plot_distribution_analysis(
        data=result_single.paths,
        title="OU Process - Statistical Properties",
        process_labels=["$X(t)$"],
    )

    print(f"   Process parameters: $\\alpha = {params.alpha}$, $\\gamma = {params.gamma}$, $\\beta = {params.beta}$")


def demo_parameter_impact():
    """Demonstrate impact of different OU parameters."""
    print("\nOU Parameter Impact Analysis")

    plotter = ProcessPlotter()

    print("\n1. Mean Reversion Speed Impact")

    alpha_values = [0.5, 2.0, 5.0]
    alpha_results = []
    alpha_labels = []

    for alpha in alpha_values:
        params = OUParameters(alpha=alpha, gamma=0.0, beta=0.2)
        ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
        result = ou.simulate(n_steps=1000, dt=1 / 252, random_state=42)
        alpha_results.append(result.paths[:, 0])
        alpha_labels.append(f"$\\alpha = {alpha}$")

    alpha_matrix = np.column_stack(alpha_results)
    time_years = np.arange(alpha_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=alpha_matrix,
        title="OU Process - Mean Reversion Speed Impact",
        xlabel="Time $t$ (years)",
        ylabel="$X(t)$",
        time_axis=time_years,
        process_labels=alpha_labels,
        figsize=(12, 6),
    )

    print("\n2. Long-term Mean Impact")

    gamma_values = [-0.5, 0.0, 0.5]
    gamma_results = []
    gamma_labels = []

    for gamma in gamma_values:
        params = OUParameters(alpha=2.0, gamma=gamma, beta=0.2)
        ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
        result = ou.simulate(n_steps=1000, dt=1 / 252, random_state=42)
        gamma_results.append(result.paths[:, 0])
        gamma_labels.append(f"$\\gamma = {gamma}$")

    gamma_matrix = np.column_stack(gamma_results)
    plotter.plot_paths(
        data=gamma_matrix,
        title="OU Process - Long-term Mean Impact",
        xlabel="Time $t$ (years)",
        ylabel="$X(t)$",
        time_axis=time_years,
        process_labels=gamma_labels,
        figsize=(12, 6),
    )

    print("\n3. Volatility Impact")

    beta_values = [0.1, 0.2, 0.4]
    beta_results = []
    beta_labels = []

    for beta in beta_values:
        params = OUParameters(alpha=2.0, gamma=0.0, beta=beta)
        ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
        result = ou.simulate(n_steps=1000, dt=1 / 252, random_state=42)
        beta_results.append(result.paths[:, 0])
        beta_labels.append(f"$\\beta = {beta}$")

    beta_matrix = np.column_stack(beta_results)
    plotter.plot_paths(
        data=beta_matrix,
        title="OU Process - Volatility Impact",
        xlabel="Time $t$ (years)",
        ylabel="$X(t)$",
        time_axis=time_years,
        process_labels=beta_labels,
        figsize=(12, 6),
    )


def demo_multi_process_ou():
    """Demonstrate multiple correlated OU processes."""
    print("\nMulti-Process OU Demonstration")

    plotter = ProcessPlotter()

    print("\n1. Multiple OU Processes with Different Parameters")

    params = (
        OUParameters(alpha=1.0, gamma=0.0, beta=0.15),
        OUParameters(alpha=3.0, gamma=0.1, beta=0.2),
        OUParameters(alpha=2.0, gamma=-0.1, beta=0.25),
    )

    correlations = [0.0, 0.5, 0.9]

    for rho in correlations:
        print(f"Simulating with correlation $\\rho = {rho}$")

        ou_multi = OrnsteinUhlenbeckProcess(params, correlation=rho)
        result_multi = ou_multi.simulate(n_steps=1000, dt=1 / 252, random_state=42)

        time_years = np.arange(result_multi.paths.shape[0]) / 252

        plotter.plot_paths(
            data=result_multi.paths,
            title=f"Multi-Process OU ($\\rho = {rho}$)",
            xlabel="Time $t$ (years)",
            ylabel="$X(t)$",
            time_axis=time_years,
            process_labels=["Slow Reversion", "Fast + Mean", "Medium - Mean"],
            figsize=(12, 6),
        )

        increments = np.diff(result_multi.paths, axis=0)

        plotter.plot_correlation_heatmap(
            data=increments,
            title=f"OU Correlation Matrix (Target $\\rho = {rho}$)",
            process_labels=["Process 1", "Process 2", "Process 3"],
        )

        actual_corr = np.corrcoef(increments.T)
        print(f"   Target: {rho:.3f}, Actual: {actual_corr[0,1]:.3f}")


def demo_mean_reversion_analysis():
    """Analyze mean reversion characteristics."""
    print("\nMean Reversion Analysis")

    plotter = ProcessPlotter()

    print("\n1. Mean Reversion from Different Starting Points")

    params = OUParameters(alpha=3.0, gamma=0.0, beta=0.2)
    ou = OrnsteinUhlenbeckProcess(params, n_processes=1)

    initial_values = [-2.0, 0.0, 2.0]
    reversion_results = []
    reversion_labels = []

    for x0 in initial_values:
        result = ou.simulate_with_initial_value(n_steps=1000, dt=1 / 252, X_0=x0, random_state=42)
        reversion_results.append(result.paths[:, 0])
        reversion_labels.append(f"$X_0 = {x0}$")

    reversion_matrix = np.column_stack(reversion_results)
    time_years = np.arange(reversion_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=reversion_matrix,
        title="OU Mean Reversion from Different Starting Points",
        xlabel="Time $t$ (years)",
        ylabel="$X(t)$",
        time_axis=time_years,
        process_labels=reversion_labels,
        figsize=(12, 6),
    )

    print("\n2. Half-life Analysis")

    alpha_values = [0.5, 1.0, 2.0, 4.0]

    for alpha in alpha_values:
        half_life = np.log(2) / alpha * 252
        print(f"   $\\alpha = {alpha:.1f}$ → Half-life = {half_life:.0f} days")

    print("\n3. Empirical Half-life Verification")

    params = OUParameters(alpha=2.0, gamma=0.0, beta=0.2)
    ou = OrnsteinUhlenbeckProcess(params, n_processes=1)
    result = ou.simulate_with_initial_value(n_steps=2000, dt=1 / 252, X_0=1.0, random_state=42)

    theoretical_half_life = np.log(2) / params.alpha * 252
    print(f"   Theoretical half-life: {theoretical_half_life:.0f} days")

    path = result.paths[:, 0]
    half_point_indices = np.where(path <= 0.5)[0]
    if len(half_point_indices) > 0:
        empirical_half_life = half_point_indices[0]
        print(f"   Empirical half-life: {empirical_half_life:.0f} days")


def demo_parameter_estimation():
    """Demonstrate OU parameter estimation."""
    print("\nOU Parameter Estimation")

    stat_plotter = StatisticalPlotter()

    true_params = OUParameters(alpha=2.5, gamma=0.1, beta=0.3)
    ou_estimation = OrnsteinUhlenbeckProcess(true_params, n_processes=1)
    result_estimation = ou_estimation.simulate(
        n_steps=2000, dt=1 / 252, random_state=42
    )

    from stochastic_calculus.processes.mean_reverting.ou_process import (
        estimate_ou_parameters,
    )

    estimated_params = estimate_ou_parameters(result_estimation.paths[:, 0], dt=1 / 252)

    print(
        f"   True: $\\alpha = {true_params.alpha:.3f}$, $\\gamma = {true_params.gamma:.3f}$, $\\beta = {true_params.beta:.3f}$"
    )
    print(
        f"   Estimated: $\\alpha = {estimated_params.alpha:.3f}$, $\\gamma = {estimated_params.gamma:.3f}$, $\\beta = {estimated_params.beta:.3f}$"
    )

    stat_plotter.plot_distribution_analysis(
        data=result_estimation.paths,
        title="OU Parameter Estimation - Data Quality",
        process_labels=["$X(t)$"],
    )


def main():
    try:
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
