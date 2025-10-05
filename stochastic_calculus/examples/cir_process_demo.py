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
    print("\n" + "=" * 60)
    print("SINGLE COX-INGERSOLL-ROSS PROCESS DEMONSTRATION")
    print("=" * 60)

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Single CIR process with typical interest rate parameters
    print("\n1. Single CIR Process (Interest Rate Model)")
    print("-" * 45)

    # Typical parameters for short-term interest rate modeling
    params = CIRParameters(
        a=3.0,  # Mean reversion speed
        b=0.04,  # Long-term mean (4%)
        c=0.2,  # Volatility parameter
    )

    cir_single = CIRProcess(params, n_processes=1)
    result_single = cir_single.simulate(
        n_steps=2000, dt=1 / 252, sigma_0=0.03, random_state=42  # Start at 3%
    )

    time_years = np.arange(result_single.paths.shape[0]) / 252

    # Convert to percentage for visualization
    rates_pct = result_single.paths * 100

    # Generic path visualization
    plotter.plot_paths(
        data=rates_pct,
        title="Single CIR Process - Interest Rate Model",
        xlabel="Time (years)",
        ylabel="Interest Rate (%)",
        time_axis=time_years,
        process_labels=["Short Rate"],
        figsize=(12, 6),
    )

    # Statistical analysis using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_single.paths,
        title="CIR Process - Statistical Properties",
        process_labels=["r(t)"],
    )

    print(f"   Process parameters: a={params.a}, b={params.b}, c={params.c}")
    print(
        f"   Feller condition: 2ab = {2*params.a*params.b:.3f}, c² = {params.c**2:.3f}"
    )
    feller_satisfied = 2 * params.a * params.b >= params.c**2
    print(f"   Feller condition satisfied: {feller_satisfied}")


def demo_parameter_impact():
    """Demonstrate impact of different CIR parameters."""
    print("\n" + "=" * 60)
    print("CIR PARAMETER IMPACT ANALYSIS")
    print("=" * 60)

    plotter = ProcessPlotter()

    # Mean reversion speed impact
    print("\n1. Mean Reversion Speed Impact")
    print("-" * 35)

    a_values = [1.0, 3.0, 6.0]
    a_results = []
    a_labels = []

    for a in a_values:
        print(f"   Simulating with a = {a}")
        params = CIRParameters(a=a, b=0.04, c=0.2)
        cir = CIRProcess(params, n_processes=1)
        result = cir.simulate(n_steps=1000, dt=1 / 252, sigma_0=0.08, random_state=42)
        a_results.append(result.paths[:, 0] * 100)  # Convert to %
        a_labels.append(f"a = {a}")

    # Generic visualization for parameter comparison
    a_matrix = np.column_stack(a_results)
    time_years = np.arange(a_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=a_matrix,
        title="CIR Process - Mean Reversion Speed Impact",
        xlabel="Time (years)",
        ylabel="Interest Rate (%)",
        time_axis=time_years,
        process_labels=a_labels,
        figsize=(12, 6),
    )

    # Long-term mean impact
    print("\n2. Long-term Mean Impact")
    print("-" * 25)

    b_values = [0.02, 0.04, 0.08]
    b_results = []
    b_labels = []

    for b in b_values:
        print(f"   Simulating with b = {b:.2%}")
        params = CIRParameters(a=3.0, b=b, c=0.2)
        cir = CIRProcess(params, n_processes=1)
        result = cir.simulate(n_steps=1000, dt=1 / 252, sigma_0=0.03, random_state=42)
        b_results.append(result.paths[:, 0] * 100)
        b_labels.append(f"b = {b:.1%}")

    b_matrix = np.column_stack(b_results)
    plotter.plot_paths(
        data=b_matrix,
        title="CIR Process - Long-term Mean Impact",
        xlabel="Time (years)",
        ylabel="Interest Rate (%)",
        time_axis=time_years,
        process_labels=b_labels,
        figsize=(12, 6),
    )

    # Volatility impact and Feller condition
    print("\n3. Volatility Impact and Feller Condition")
    print("-" * 40)

    c_values = [0.1, 0.3, 0.5]
    c_results = []
    c_labels = []

    for c in c_values:
        print(f"   Simulating with c = {c}")
        try:
            params = CIRParameters(a=3.0, b=0.04, c=c)
            feller_condition = 2 * params.a * params.b >= params.c**2
            print(f"   Feller condition satisfied: {feller_condition}")

            cir = CIRProcess(params, n_processes=1)
            result = cir.simulate(
                n_steps=1000, dt=1 / 252, sigma_0=0.04, random_state=42
            )
            c_results.append(result.paths[:, 0] * 100)
            c_labels.append(f"c = {c}")
        except ValueError as e:
            print(f"   Parameter rejected: {e}")
            continue

    if c_results:
        c_matrix = np.column_stack(c_results)
        plotter.plot_paths(
            data=c_matrix,
            title="CIR Process - Volatility Impact (Feller Condition Test)",
            xlabel="Time (years)",
            ylabel="Interest Rate (%)",
            time_axis=time_years,
            process_labels=c_labels,
            figsize=(12, 6),
        )


def demo_multi_process_cir():
    """Demonstrate multiple correlated CIR processes for yield curve modeling."""
    print("\n" + "=" * 60)
    print("MULTI-PROCESS CIR DEMONSTRATION (YIELD CURVE MODELING)")
    print("=" * 60)

    plotter = ProcessPlotter()

    # Create different CIR processes representing different maturities
    print("\n1. Yield Curve with Multiple CIR Processes")
    print("-" * 45)

    params = (
        CIRParameters(a=5.0, b=0.03, c=0.15),  # Short-term (fast reversion)
        CIRParameters(a=2.0, b=0.04, c=0.18),  # Medium-term
        CIRParameters(a=1.0, b=0.05, c=0.20),  # Long-term (slow reversion)
    )

    # Test different correlation levels
    correlations = [0.3, 0.7]

    for rho in correlations:
        print(f"   Simulating yield curve with correlation ρ = {rho}")

        cir_multi = CIRProcess(params, correlation=rho)
        result_multi = cir_multi.simulate(
            n_steps=1000,
            dt=1 / 252,
            sigma_0=np.array([0.02, 0.035, 0.045]),  # Initial yield curve
            random_state=42,
        )

        time_years = np.arange(result_multi.paths.shape[0]) / 252
        rates_pct = result_multi.paths * 100

        # Generic visualization for multiple processes
        plotter.plot_paths(
            data=rates_pct,
            title=f"Multi-Factor Yield Curve Model (ρ = {rho})",
            xlabel="Time (years)",
            ylabel="Interest Rate (%)",
            time_axis=time_years,
            process_labels=["Short Rate", "Medium Rate", "Long Rate"],
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
    print("\n" + "=" * 60)
    print("CIR BOUNDARY BEHAVIOR ANALYSIS")
    print("=" * 60)

    plotter = ProcessPlotter()

    # Test behavior near zero boundary
    print("\n1. Near-Zero Boundary Behavior")
    print("-" * 30)

    params = CIRParameters(a=2.0, b=0.04, c=0.3)
    cir = CIRProcess(params, n_processes=1)

    # Start from very low initial values
    initial_values = [0.001, 0.01, 0.05]
    boundary_results = []
    boundary_labels = []

    for sigma0 in initial_values:
        result = cir.simulate(n_steps=1000, dt=1 / 252, sigma_0=sigma0, random_state=42)
        boundary_results.append(result.paths[:, 0] * 100)
        boundary_labels.append(f"σ= {sigma0:.1%}")

    boundary_matrix = np.column_stack(boundary_results)
    time_years = np.arange(boundary_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=boundary_matrix,
        title="CIR Near-Zero Boundary Behavior",
        xlabel="Time (years)",
        ylabel="Interest Rate (%)",
        time_axis=time_years,
        process_labels=boundary_labels,
        figsize=(12, 6),
    )

    # Verify non-negativity
    min_values = [np.min(result) for result in boundary_results]
    print(f"   Minimum values achieved: {[f'{val:.4f}%' for val in min_values]}")
    print(f"   Non-negativity maintained: {all(val >= 0 for val in min_values)}")


def demo_parameter_estimation():
    """Demonstrate CIR parameter estimation."""
    print("\n" + "=" * 60)
    print("CIR PARAMETER ESTIMATION")
    print("=" * 60)

    stat_plotter = StatisticalPlotter()

    # Generate data with known parameters
    true_params = CIRParameters(a=2.5, b=0.04, c=0.25)
    cir_estimation = CIRProcess(true_params, n_processes=1)
    result_estimation = cir_estimation.simulate(
        n_steps=2000, dt=1 / 252, sigma_0=0.03, random_state=42
    )

    # Estimate parameters from the generated data
    from stochastic_calculus.processes.mean_reverting.cir_process import (
        estimate_cir_parameters,
    )

    estimated_params = estimate_cir_parameters(
        result_estimation.paths[:, 0], dt=1 / 252
    )

    print(
        f"   True parameters:      a = {true_params.a:.3f}, b = {true_params.b:.3f}, c = {true_params.c:.3f}"
    )
    print(
        f"   Estimated parameters: a = {estimated_params.a:.3f}, b = {estimated_params.b:.3f}, c = {estimated_params.c:.3f}"
    )

    # Check Feller condition for both
    true_feller = 2 * true_params.a * true_params.b >= true_params.c**2
    est_feller = 2 * estimated_params.a * estimated_params.b >= estimated_params.c**2
    print(f"   True Feller condition: {true_feller}")
    print(f"   Estimated Feller condition: {est_feller}")

    # Visualize estimation quality using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_estimation.paths,
        title="CIR Parameter Estimation - Data Quality Check",
        process_labels=["r(t)"],
    )


def main():

    try:
        # Run all demonstrations
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
