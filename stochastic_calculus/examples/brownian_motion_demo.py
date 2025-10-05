"""Comprehensive demo for Brownian motion processes - both standard and geometric.

This demo showcases:
- Standard Brownian motion (single and correlated)
- Geometric Brownian motion (single and multi-asset)
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
from stochastic_calculus.processes.brownian import (
    BrownianMotion, 
    GeometricBrownianMotion,
    ConstantDrift,
    ConstantVolatility, 
    FixedInitialPrices
)
from stochastic_calculus.visualization import ProcessPlotter, StatisticalPlotter

RANDOM_STATE = 1


def demo_standard_brownian_motion():
    """Demonstrate standard Brownian motion with generic visualizations."""
    print("\nStandard Brownian Motion Demo")

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    print("1. Single Brownian Motion")

    # Simulate single process
    bm_single = BrownianMotion(n_processes=1)
    result_single = bm_single.simulate(
        n_steps=2000, dt=1 / 252, random_state=RANDOM_STATE
    )

    # Time axis in years
    time_years = np.arange(result_single.paths.shape[0]) / 252

    # path visualization
    plotter.plot_paths(
        data=result_single.paths,
        title="Standard Brownian Motion",
        xlabel="$t$ (years)",
        ylabel="$W(t)$",
        time_axis=time_years,
        process_labels=["Brownian Motion"],
        figsize=(12, 6),
    )

    # Statistical analysis
    stat_plotter.plot_distribution_analysis(
        data=result_single.paths,
        title="Standard Brownian Motion - Statistical Properties",
        process_labels=["W(t)"],
    )

    print("\n2. Correlated Brownian Motions")

    # Test different correlation levels
    correlations = [0.0, 0.5, 0.9]

    for i, rho in enumerate(correlations):
        print(f"Simulating correlation ρ = {rho}")

        # Simulate correlated processes
        bm_corr = BrownianMotion(n_processes=3, correlation=rho)
        result_corr = bm_corr.simulate(
            n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE
        )

        # Generic visualization for multiple processes
        plotter.plot_paths(
            data=result_corr.paths,
            title=f"Correlated Brownian Motion (ρ = {rho})",
            xlabel="Time (years)",
            ylabel="W(t)",
            time_axis=np.arange(result_corr.paths.shape[0]) / 252,
            process_labels=[f"Process {j+1}" for j in range(3)],
            figsize=(12, 6),
        )

        # Correlation analysis
        plotter.plot_correlation_heatmap(
            data=result_corr.increments,
            title=f"Correlation Matrix (Target ρ = {rho})",
            process_labels=[f"BM {j+1}" for j in range(3)],
        )

        # Calculate and display actual correlation
        actual_corr = np.corrcoef(result_corr.increments.T)
        print(f"Target: {rho:.3f}, Actual: {actual_corr[0,1]:.3f}")

    print("\n3. Properties Analysis")

    # Generate longer series for analysis
    bm_analysis = BrownianMotion(n_processes=1)
    result_analysis = bm_analysis.simulate(
        n_steps=5000, dt=1 / 252, random_state=RANDOM_STATE
    )

    stat_plotter.plot_distribution_analysis(
        data=result_analysis.paths,
        title="Brownian Motion - Statistical Analysis",
        process_labels=["$W(t)$"],
    )

    # Analyze scaling properties
    time_points = [252, 504, 1260, 2520]  # 1, 2, 5, 10 years
    scaling_data = []
    scaling_labels = []

    for i, t in enumerate(time_points):
        # Generate independent paths for each time horizon to show different realizations
        bm_scaling = BrownianMotion(n_processes=1)
        result_scaling = bm_scaling.simulate(n_steps=t, dt=1/252, random_state=RANDOM_STATE + i + 10)
        scaling_data.append(result_scaling.paths[:, 0])
        scaling_labels.append(f"{t/252:.0f} year(s)")

    # Plot scaling
    max_len = max(len(data) for data in scaling_data)
    scaling_matrix = np.full((max_len, len(scaling_data)), np.nan)

    for i, data in enumerate(scaling_data):
        scaling_matrix[: len(data), i] = data

    plotter.plot_paths(
        data=scaling_matrix,
        title="Brownian Motion - Time Scaling Properties",
        xlabel="Time Steps",
        ylabel="$W(t)$",
        process_labels=scaling_labels,
        figsize=(12, 6),
    )


def demo_geometric_brownian_motion():
    """Demonstrate geometric Brownian motion with generic visualizations."""
    print("\nGeometric Brownian Motion Demo")

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    print("1. Geometric Brownian Motion.")

    # Create GBM with typical stock parameters using proper components
    drift = ConstantDrift(n_steps=1000, mu=0.08)  # 8% expected return
    volatility = ConstantVolatility(n_steps=1000, sigma=0.2)  # 20% volatility
    initial = FixedInitialPrices(S_0=100.0)  # $100 initial price
    gbm_single = GeometricBrownianMotion(drift, volatility, initial)

    result_single = gbm_single.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)
    time_years = np.arange(result_single.prices.shape[0]) / 252

    # Generic visualization for prices
    plotter.plot_paths(
        data=result_single.prices,
        title="Geometric Brownian Motion - Single Asset",
        xlabel="Time $t$ (years)",
        ylabel="Asset Price $S(t)$",
        time_axis=time_years,
        process_labels=["$S(t)$"],
        figsize=(12, 6),
    )

    # Statistical analysis of log returns (should be normally distributed)
    single_log_returns = np.diff(result_single.log_prices, axis=0)
    stat_plotter.plot_distribution_analysis(
        data=single_log_returns,
        title="GBM - Log Return Statistical Analysis",
        process_labels=["$d\\log S(t)$"],
    )

    print("\n2. Multi-Asset Portfolio")

    # Create portfolio with different risk/return profiles
    asset_configs = [
        {"name": "Tech", "mu": 0.12, "sigma": 0.30, "S_0": 100.0},
        {"name": "Utility", "mu": 0.06, "sigma": 0.15, "S_0": 100.0},
        {"name": "Bond", "mu": 0.02, "sigma": 0.03, "S_0": 100.0},
    ]

    # Create multi-asset GBM using proper components
    multi_drift = ConstantDrift(
        n_steps=1000, 
        mu=tuple(config["mu"] for config in asset_configs)
    )
    multi_volatility = ConstantVolatility(
        n_steps=1000,
        sigma=tuple(config["sigma"] for config in asset_configs)
    )
    multi_initial = FixedInitialPrices(
        S_0=tuple(config["S_0"] for config in asset_configs)
    )
    gbm_multi = GeometricBrownianMotion(multi_drift, multi_volatility, multi_initial)

    result_multi = gbm_multi.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)

    # Generic visualization for multiple assets  
    plotter.plot_paths(
        data=result_multi.prices,
        title="Geometric Brownian Motion - Multi-Asset Portfolio",
        xlabel="Time $t$ (years)",
        ylabel="Asset Price $S(t)$",
        time_axis=time_years,
        process_labels=[config["name"] for config in asset_configs],
        figsize=(12, 6),
    )

    # Correlation analysis
    log_returns = np.diff(result_multi.log_prices, axis=0)
    plotter.plot_correlation_heatmap(
        data=log_returns,
        title="Asset Return Correlations",
        process_labels=[config["name"] for config in asset_configs],
    )

    print("\n3. Parameter Impact Analysis")

    # Demonstrate impact of different parameters

    # A) Volatility impact
    print("Analyzing volatility impact")
    volatilities = [0.1, 0.2, 0.4]
    vol_results = []
    vol_labels = []

    for vol in volatilities:
        # Create GBM with different volatilities using proper components
        vol_drift = ConstantDrift(n_steps=1000, mu=0.08)
        vol_volatility = ConstantVolatility(n_steps=1000, sigma=vol)
        vol_initial = FixedInitialPrices(S_0=100.0)
        gbm_vol = GeometricBrownianMotion(vol_drift, vol_volatility, vol_initial)
        result_vol = gbm_vol.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)
        vol_results.append(result_vol.prices[:, 0])
        vol_labels.append(f"$\\sigma = {vol:.1f}$")

    # visualization for parameter comparison
    vol_matrix = np.column_stack(vol_results)
    plotter.plot_paths(
        data=vol_matrix,
        title="Geometric Brownian Motion - Volatility Parameter Impact",
        xlabel="Time $t$ (years)",
        ylabel="Asset Price $S(t)$",
        time_axis=time_years,
        process_labels=vol_labels,
        figsize=(12, 6),
    )

    # B) Drift impact
    print("Analyzing drift impact")
    drifts = [0.02, 0.08, 0.15]
    drift_results = []
    drift_labels = []

    for mu in drifts:
        # Create GBM with different drifts
        drift_component = ConstantDrift(n_steps=1000, mu=mu)
        drift_volatility = ConstantVolatility(n_steps=1000, sigma=0.2)
        drift_initial = FixedInitialPrices(S_0=100.0)
        gbm_drift = GeometricBrownianMotion(drift_component, drift_volatility, drift_initial)
        result_drift = gbm_drift.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)
        drift_results.append(result_drift.prices[:, 0])
        drift_labels.append(f"$\\mu = {mu:.2f}$")

    # Generic visualization for parameter comparison
    drift_matrix = np.column_stack(drift_results)
    plotter.plot_paths(
        data=drift_matrix,
        title="Geometric Brownian Motion - Drift Parameter Impact",
        xlabel="Time $t$ (years)",
        ylabel="Asset Price $S(t)$",
        time_axis=time_years,
        process_labels=drift_labels,
        figsize=(12, 6),
    )

    print("\n4. Parameter Estimation")

    # Generate data with known parameters
    true_mu, true_sigma = 0.10, 0.25
    est_drift = ConstantDrift(n_steps=10000, mu=true_mu)
    est_volatility = ConstantVolatility(n_steps=10000, sigma=true_sigma)
    est_initial = FixedInitialPrices(S_0=100.0)
    gbm_estimation = GeometricBrownianMotion(est_drift, est_volatility, est_initial)
    result_estimation = gbm_estimation.simulate(n_steps=10000, dt=1 / 252, random_state=RANDOM_STATE)

    # Estimate parameters from the generated data
    from stochastic_calculus.processes.brownian.geometric import estimate_gbm_parameters

    estimated_params = estimate_gbm_parameters(
        result_estimation.prices[:, 0], dt=1 / 252
    )

    print(f"True: μ={true_mu:.3f}, σ={true_sigma:.3f}")
    print(
        f"Estimated: μ={estimated_params['mu']:.3f}, σ={estimated_params['sigma']:.3f}"
    )

    # Visualize estimation quality using log returns (which should be normally distributed)
    log_returns = np.diff(result_estimation.log_prices, axis=0)
    stat_plotter.plot_distribution_analysis(
        data=log_returns,
        title="GBM Parameter Estimation - Log Return Analysis",
        process_labels=["$d\\log S(t)$"],
    )


def demo_comparison_analysis():
    """Compare standard and geometric Brownian motion."""
    print("\nComparison Analysis")

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Generate comparable processes
    n_steps = 1000
    dt = 1 / 252
    time_years = np.arange(n_steps + 1) / 252

    # Standard Brownian motion
    bm = BrownianMotion(n_processes=1)
    bm_result = bm.simulate(n_steps=n_steps, dt=dt, random_state=RANDOM_STATE)

    # Geometric Brownian motion (scaled for comparison) using proper components
    comp_drift = ConstantDrift(n_steps=n_steps, mu=0.0)
    comp_volatility = ConstantVolatility(n_steps=n_steps, sigma=0.3)
    comp_initial = FixedInitialPrices(S_0=1.0)
    gbm = GeometricBrownianMotion(comp_drift, comp_volatility, comp_initial)
    gbm_result = gbm.simulate(n_steps=n_steps, dt=dt, random_state=RANDOM_STATE)

    # Combine for comparison
    comparison_data = np.column_stack(
        [
            bm_result.paths[:, 0],
            gbm_result.prices[:, 0] - 1.0,
            np.log(gbm_result.prices[:, 0]),
        ]
    )

    plotter.plot_paths(
        data=comparison_data,
        title="Brownian Motion Variants - Comparison",
        xlabel="$t$ (years)",
        ylabel="Value",
        time_axis=time_years,
        process_labels=["Standard BM", "GBM - 1", "Log(GBM)"],
        figsize=(12, 6),
    )

    # Statistical comparison
    stat_plotter.plot_distribution_analysis(
        data=comparison_data,
        title="Brownian Motion Variants - Statistical Comparison",
        process_labels=["Standard BM", "GBM - 1", "Log(GBM)"],
    )

    # Correlation analysis
    plotter.plot_correlation_heatmap(
        data=comparison_data,
        title="Brownian Motion Variants - Correlations",
        process_labels=["Standard BM", "GBM - 1", "Log(GBM)"],
    )


def main():
    try:
        # Run all demonstrations
        demo_standard_brownian_motion()
        demo_geometric_brownian_motion()
        demo_comparison_analysis()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
