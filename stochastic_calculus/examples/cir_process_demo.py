"""Comprehensive demo for Cox-Ingersoll-Ross (CIR) processes.

This demo showcases:
- Single and multi-process CIR simulation
- Parameter impact analysis
- Non-negative constraint behavior
- Interest rate modeling applications
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
from stochastic_calculus.processes.mean_reverting import CIRProcess, CIRParameters
from stochastic_calculus.visualization import ProcessPlotter, StatisticalPlotter


def demo_single_cir_process():
    """Demonstrate single CIR process with realistic interest rate parameters."""
    print("\nSingle Cox-Ingersoll-Ross Process")
    
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    print("1. Interest Rate Model")

    params = CIRParameters(a=3.0, b=0.04, c=0.2)

    cir_single = CIRProcess(params, n_processes=1)
    result_single = cir_single.simulate_with_initial_value(
        n_steps=2000, dt=1 / 252, sigma_0=0.03, random_state=42
    )

    time_years = np.arange(result_single.paths.shape[0]) / 252
    rates_pct = result_single.paths * 100

    plotter.plot_paths(
        data=rates_pct,
        title="Cox-Ingersoll-Ross Process - Interest Rate Model",
        xlabel="Time $t$ (years)",
        ylabel="Interest Rate $r(t)$ (%)",
        time_axis=time_years,
        process_labels=["$r(t)$"],
        figsize=(12, 6),
    )

    stat_plotter.plot_distribution_analysis(
        data=result_single.paths,
        title="CIR Process - Statistical Analysis",
        process_labels=["$r(t)$"],
    )

    print(f"   Process parameters: a={params.a}, b={params.b}, c={params.c}")
    print(
        f"   Feller condition: 2ab = {2*params.a*params.b:.3f}, c² = {params.c**2:.3f}"
    )
    feller_satisfied = 2 * params.a * params.b >= params.c**2
    print(f"   Feller condition satisfied: {feller_satisfied}")


def demo_parameter_impact():
    """Demonstrate impact of different CIR parameters."""
    print("\nCIR Parameter Impact Analysis")

    plotter = ProcessPlotter()

    print("\n1. Mean Reversion Speed Impact")

    a_values = [1.0, 3.0, 6.0]
    a_results = []
    a_labels = []

    for a in a_values:
        params = CIRParameters(a=a, b=0.04, c=0.2)
        cir = CIRProcess(params, n_processes=1)
        result = cir.simulate(n_steps=1000, dt=1 / 252, sigma_0=0.08, random_state=42)
        a_results.append(result.paths[:, 0] * 100)  # Convert to %
        a_labels.append(f"$a = {a}$")

    a_matrix = np.column_stack(a_results)
    time_years = np.arange(a_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=a_matrix,
        title="CIR Process - Mean Reversion Speed Impact",
        xlabel="Time $t$ (years)",
        ylabel="Interest Rate $r(t)$ (%)",
        time_axis=time_years,
        process_labels=a_labels,
        figsize=(12, 6),
    )

    print("\n2. Long-term Mean Impact")

    b_values = [0.02, 0.04, 0.08]
    b_results = []
    b_labels = []

    for b in b_values:
        params = CIRParameters(a=3.0, b=b, c=0.2)
        cir = CIRProcess(params, n_processes=1)
        result = cir.simulate(n_steps=1000, dt=1 / 252, sigma_0=0.03, random_state=42)
        b_results.append(result.paths[:, 0] * 100)
        b_labels.append(f"$b = {b:.3f}$")

    b_matrix = np.column_stack(b_results)
    plotter.plot_paths(
        data=b_matrix,
        title="CIR Process - Long-term Mean Impact",
        xlabel="Time $t$ (years)",
        ylabel="Interest Rate $r(t)$ (%)",
        time_axis=time_years,
        process_labels=b_labels,
        figsize=(12, 6),
    )

    print("\n3. Volatility Impact and Feller Condition")

    c_values = [0.1, 0.3, 0.5]
    c_results = []
    c_labels = []

    for c in c_values:
        try:
            params = CIRParameters(a=3.0, b=0.04, c=c)
            feller_condition = 2 * params.a * params.b >= params.c**2

            cir = CIRProcess(params, n_processes=1)
            result = cir.simulate(
                n_steps=1000, dt=1 / 252, sigma_0=0.04, random_state=42
            )
            c_results.append(result.paths[:, 0] * 100)
            feller_check = "+" if feller_condition else "-"
            c_labels.append(f"$c = {c}$ ({feller_check})")
        except ValueError:
            continue

    if c_results:
        c_matrix = np.column_stack(c_results)
        plotter.plot_paths(
            data=c_matrix,
            title="CIR Process - Volatility Impact (Feller Condition)",
            xlabel="Time $t$ (years)",
            ylabel="Interest Rate $r(t)$ (%)",
            time_axis=time_years,
            process_labels=c_labels,
            figsize=(12, 6),
        )


def demo_multi_process_cir():
    """Demonstrate multiple correlated CIR processes for yield curve modeling."""
    print("\nMulti-Process CIR (Yield Curve)")

    plotter = ProcessPlotter()

    print("\n1. Yield Curve with Multiple CIR Processes")

    params = (
        CIRParameters(a=5.0, b=0.03, c=0.15),
        CIRParameters(a=2.0, b=0.04, c=0.18),
        CIRParameters(a=1.0, b=0.05, c=0.20),
    )

    correlations = [0.3, 0.7]

    for rho in correlations:
        print(f"Simulating yield curve with correlation ρ = {rho}")

        cir_multi = CIRProcess(params, correlation=rho)
        result_multi = cir_multi.simulate(
            n_steps=1000,
            dt=1 / 252,
            sigma_0=np.array([0.02, 0.02, 0.02]),
            random_state=42,
        )

        time_years = np.arange(result_multi.paths.shape[0]) / 252
        rates_pct = result_multi.paths * 100

        plotter.plot_paths(
            data=rates_pct,
            title=f"Multi-Factor Yield Curve Model ($\\rho = {rho}$)",
            xlabel="Time $t$ (years)",
            ylabel="Interest Rate $r_i(t)$ (%)",
            time_axis=time_years,
            process_labels=["$r_s(t)$ (Short)", "$r_m(t)$ (Medium)", "$r_l(t)$ (Long)"],
            figsize=(12, 6),
        )

        # Calculate increments for correlation analysis
        increments = np.diff(result_multi.paths, axis=0)

        # Correlation analysis using generic heatmap
        plotter.plot_correlation_heatmap(
            data=increments,
            title=f"Yield Curve Correlations (Target ρ = {rho})",
            process_labels=["Short", "Medium", "Long"],
        )

        # Calculate and display actual correlation
        actual_corr = np.corrcoef(increments.T)
        print(f"   Target correlation: {rho:.3f}")
        print(f"   Actual correlation: {actual_corr[0,1]:.3f}")

    print("   Multi-process CIR demonstration complete")


def demo_boundary_behavior():
    """Analyze CIR boundary behavior and non-negativity constraint."""
    print("\nCIR Boundary Behavior Analysis")

    plotter = ProcessPlotter()

    print("1. Near-Zero Boundary Behavior")

    params = CIRParameters(a=2.0, b=0.04, c=0.3)
    cir = CIRProcess(params, n_processes=1)

    initial_values = [0.001, 0.01, 0.05]
    boundary_results = []
    boundary_labels = []

    for sigma0 in initial_values:
        result = cir.simulate(n_steps=1000, dt=1 / 252, sigma_0=sigma0, random_state=42)
        boundary_results.append(result.paths[:, 0] * 100)
        boundary_labels.append(f"$\\sigma_0 = {sigma0:.3f}$")

    boundary_matrix = np.column_stack(boundary_results)
    time_years = np.arange(boundary_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=boundary_matrix,
        title="CIR Near-Zero Boundary Behavior",
        xlabel="Time $t$ (years)",
        ylabel="Interest Rate $r(t)$ (%)",
        time_axis=time_years,
        process_labels=boundary_labels,
        figsize=(12, 6),
    )

    min_values = [np.min(result) for result in boundary_results]
    print(f"   Minimum values: {[f'{val:.4f}%' for val in min_values]}")
    print(f"   Non-negativity maintained: {all(val >= 0 for val in min_values)}")


def demo_parameter_estimation():
    """Demonstrate CIR parameter estimation."""
    print("\nCIR Parameter Estimation")

    stat_plotter = StatisticalPlotter()

    true_params = CIRParameters(a=2.5, b=0.04, c=0.25)
    cir_estimation = CIRProcess(true_params, n_processes=1)
    result_estimation = cir_estimation.simulate(
        n_steps=2000, dt=1 / 252, sigma_0=0.03, random_state=42
    )

    from stochastic_calculus.processes.mean_reverting.cir_process import (
        estimate_cir_parameters,
    )

    estimated_params = estimate_cir_parameters(
        result_estimation.paths[:, 0], dt=1 / 252
    )

    print(
        f"   True: $a = {true_params.a:.3f}$, $b = {true_params.b:.3f}$, $c = {true_params.c:.3f}$"
    )
    print(
        f"   Estimated: $a = {estimated_params.a:.3f}$, $b = {estimated_params.b:.3f}$, $c = {estimated_params.c:.3f}$"
    )

    true_feller = 2 * true_params.a * true_params.b >= true_params.c**2
    est_feller = 2 * estimated_params.a * estimated_params.b >= estimated_params.c**2
    print(f"   True Feller condition: {true_feller}")
    print(f"   Estimated Feller condition: {est_feller}")

    stat_plotter.plot_distribution_analysis(
        data=result_estimation.paths,
        title="CIR Parameter Estimation - Data Quality",
        process_labels=["$r(t)$"],
    )


def main():
    try:
        demo_single_cir_process()
        demo_parameter_impact()
        demo_multi_process_cir()
        demo_boundary_behavior()
        demo_parameter_estimation()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
