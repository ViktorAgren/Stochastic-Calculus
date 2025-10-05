"""Comprehensive demo for Heston stochastic volatility processes.

This demo showcases:
- Single and multi-asset Heston simulation
- Parameter impact analysis (kappa, theta, sigma, rho, mu)
- Leverage effect demonstration
- Volatility clustering analysis
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
from stochastic_calculus.processes.stochastic_vol import HestonProcess, HestonParameters
from stochastic_calculus.visualization import ProcessPlotter, StatisticalPlotter


def demo_single_heston_process():
    """Demonstrate single Heston process with realistic stock parameters."""
    print("\nSingle Heston Process Demo")

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Single Heston process with typical stock parameters
    print("1. Single Asset Model")

    # Realistic parameters for equity modeling
    params = HestonParameters(
        kappa=3.0,  # Fast mean reversion of volatility
        theta=0.04,  # Long-term variance (20% volatility)
        sigma=0.3,  # Vol-of-vol
        rho=-0.7,  # Strong negative correlation (leverage effect)
        mu=0.08,  # Expected return
    )

    heston_single = HestonProcess(params, n_processes=1)
    result_single = heston_single.simulate(
        n_steps=2000, dt=1 / 252, S_0=100.0, random_state=42
    )

    time_years = np.arange(result_single.prices.shape[0]) / 252

    # Generic price and volatility visualization
    plotter.plot_price_and_volatility(
        result_single.prices,
        np.sqrt(result_single.volatilities) * 100,  # Convert to %
        time_axis=time_years,
        asset_names=["Heston Asset"],
    )

    # Statistical analysis using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_single.prices,
        title="Heston Process - Price Statistical Properties",
        process_labels=["S(t)"],
    )

    # Leverage effect analysis
    stat_plotter.plot_returns_vs_volatility(
        result_single.prices,
        np.sqrt(result_single.volatilities),
        title="Heston Model - Leverage Effect Analysis",
    )

    print(
        f"   Process parameters: κ={params.kappa}, θ={params.theta}, σ={params.sigma}"
    )
    print(f"   Leverage: ρ={params.rho}, μ={params.mu}")


def demo_parameter_impact():
    """Demonstrate impact of different Heston parameters."""
    print("\n" + "=" * 60)
    print("HESTON PARAMETER IMPACT ANALYSIS")
    print("=" * 60)

    plotter = ProcessPlotter()

    # Mean reversion speed impact (kappa)
    print("\n1. Volatility Mean Reversion Speed Impact")
    print("-" * 40)

    kappa_values = [2.0, 3.0, 6.0]  # Increased min kappa to satisfy Feller condition
    kappa_results_prices = []
    kappa_results_vols = []
    kappa_labels = []

    for kappa in kappa_values:
        print(f"   Simulating with κ = {kappa}")
        # Ensure Feller condition: 2κθ ≥ σ²
        # With θ=0.04, σ=0.25: need κ ≥ σ²/(2θ) = 0.0625/0.08 = 0.78
        params = HestonParameters(
            kappa=kappa, theta=0.04, sigma=0.25, rho=-0.7, mu=0.08
        )
        heston = HestonProcess(params, n_processes=1)
        result = heston.simulate(n_steps=1000, dt=1 / 252, S_0=100.0, random_state=42)
        kappa_results_prices.append(result.prices[:, 0])
        kappa_results_vols.append(np.sqrt(result.volatilities[:, 0]) * 100)
        kappa_labels.append(f"κ = {kappa}")

    # Generic visualization for volatility comparison
    kappa_vol_matrix = np.column_stack(kappa_results_vols)
    time_years = np.arange(kappa_vol_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=kappa_vol_matrix,
        title="Heston Process - Volatility Mean Reversion Speed Impact",
        xlabel="Time (years)",
        ylabel="Volatility (%)",
        time_axis=time_years,
        process_labels=kappa_labels,
        figsize=(12, 6),
    )

    # Leverage effect impact (rho)
    print("\n2. Leverage Effect Impact")

    rho_values = [-0.9, -0.5, 0.0]
    rho_results_prices = []
    rho_results_vols = []
    rho_labels = []

    for rho in rho_values:
        print(f"Simulating ρ = {rho}")
        params = HestonParameters(
            kappa=3.0, theta=0.04, sigma=0.2, rho=rho, mu=0.08
        )  # Ensure Feller: 2*3*0.04=0.24 > 0.04
        heston = HestonProcess(params, n_processes=1)
        result = heston.simulate(n_steps=1000, dt=1 / 252, S_0=100.0, random_state=42)
        rho_results_prices.append(result.prices[:, 0])
        rho_results_vols.append(np.sqrt(result.volatilities[:, 0]) * 100)
        rho_labels.append(f"ρ = {rho}")

    # Show price paths for different leverage effects
    rho_price_matrix = np.column_stack(rho_results_prices)
    plotter.plot_paths(
        data=rho_price_matrix,
        title="Heston Process - Leverage Effect Impact on Prices",
        xlabel="Time (years)",
        ylabel="Stock Price ($)",
        time_axis=time_years,
        process_labels=rho_labels,
        figsize=(12, 6),
    )

    # Vol-of-vol impact (sigma)
    print("\n3. Vol-of-Vol Impact")

    sigma_values = [
        0.1,
        0.3,
        0.45,
    ]  # Max 0.45 to satisfy Feller: 2*3*0.04=0.24 > 0.45²=0.2025
    sigma_results_vols = []
    sigma_labels = []

    for sigma in sigma_values:
        print(f"Simulating σ = {sigma}")
        params = HestonParameters(
            kappa=3.0, theta=0.04, sigma=sigma, rho=-0.7, mu=0.08
        )  # theta high enough for Feller
        heston = HestonProcess(params, n_processes=1)
        result = heston.simulate(n_steps=1000, dt=1 / 252, S_0=100.0, random_state=42)
        sigma_results_vols.append(np.sqrt(result.volatilities[:, 0]) * 100)
        sigma_labels.append(f"σ = {sigma}")

    sigma_vol_matrix = np.column_stack(sigma_results_vols)
    plotter.plot_paths(
        data=sigma_vol_matrix,
        title="Heston Process - Vol-of-Vol Impact",
        xlabel="Time (years)",
        ylabel="Volatility (%)",
        time_axis=time_years,
        process_labels=sigma_labels,
        figsize=(12, 6),
    )


def demo_multi_asset_heston():
    """Demonstrate multiple correlated Heston processes."""
    print("\nMulti-Asset Heston Demo")

    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Create different assets with varying characteristics
    print("\n1. Multi-Asset Portfolio with Different Risk Profiles")
    print("-" * 55)

    params = (
        HestonParameters(
            kappa=2.5, theta=0.04, sigma=0.3, rho=-0.6, mu=0.08
        ),  # Tech stock
        HestonParameters(
            kappa=3.5, theta=0.02, sigma=0.2, rho=-0.4, mu=0.06
        ),  # Utility
        HestonParameters(
            kappa=2.0, theta=0.06, sigma=0.4, rho=-0.8, mu=0.10
        ),  # Emerging market
    )

    asset_names = ["Tech Stock", "Utility Stock", "Emerging Market"]
    initial_prices = np.array([100.0, 150.0, 200.0])

    # Test different correlation levels
    correlations = [0.3, 0.7]

    for rho in correlations:
        print(f"   Simulating portfolio with correlation ρ = {rho}")

        heston_multi = HestonProcess(params, inter_asset_correlation=rho)
        result_multi = heston_multi.simulate(
            n_steps=1000, dt=1 / 252, S_0=initial_prices, random_state=42
        )

        time_years = np.arange(result_multi.prices.shape[0]) / 252

        # Generic visualization for multiple assets
        plotter.plot_price_and_volatility(
            result_multi.prices,
            np.sqrt(result_multi.volatilities) * 100,
            time_axis=time_years,
            asset_names=asset_names,
        )

        # Calculate log returns for correlation analysis
        log_returns = np.diff(np.log(result_multi.prices), axis=0)

        # Correlation analysis using generic heatmap
        plotter.plot_correlation_heatmap(
            data=log_returns,
            title=f"Multi-Asset Return Correlations (Target ρ = {rho})",
            process_labels=asset_names,
        )

        # Calculate and display actual correlation
        actual_corr = np.corrcoef(log_returns.T)
        print(f"   Target correlation: {rho:.3f}")
        print(f"   Actual correlations:")
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):
                print(
                    f"     {asset_names[i]} vs {asset_names[j]}: {actual_corr[i,j]:.3f}"
                )


def demo_volatility_clustering():
    """Analyze volatility clustering characteristics."""
    print("\n" + "=" * 60)
    print("VOLATILITY CLUSTERING ANALYSIS")
    print("=" * 60)

    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    # Generate Heston process with pronounced volatility clustering
    print("\n1. Volatility Clustering Demonstration")
    print("-" * 40)

    params = HestonParameters(
        kappa=1.5, theta=0.06, sigma=0.4, rho=-0.8, mu=0.08
    )  # Ensure Feller: 2*1.5*0.06=0.18 > 0.16
    heston = HestonProcess(params, n_processes=1)
    result = heston.simulate(n_steps=2000, dt=1 / 252, S_0=100.0, random_state=42)

    time_years = np.arange(result.prices.shape[0]) / 252

    # Show price and volatility evolution
    plotter.plot_price_and_volatility(
        result.prices,
        np.sqrt(result.volatilities) * 100,
        time_axis=time_years,
        asset_names=["Asset with Vol Clustering"],
    )

    # Calculate returns for clustering analysis
    returns = np.diff(np.log(result.prices[:, 0])) * 100  # Daily returns in %
    volatilities = np.sqrt(result.volatilities[1:, 0]) * 100  # Matching volatilities

    # Show returns vs volatility relationship
    stat_plotter.plot_returns_vs_volatility(
        result.prices,
        np.sqrt(result.volatilities),
        title="Heston Model - Returns vs Volatility Clustering",
    )

    # Volatility persistence analysis
    print("\n2. Volatility Persistence Analysis")
    print("-" * 35)

    # Compare different mean reversion speeds
    kappa_values = [1.5, 3.0, 6.0]  # All satisfy Feller condition
    persistence_results = []
    persistence_labels = []

    for kappa in kappa_values:
        # Adjust theta to ensure Feller condition: 2κθ ≥ σ²
        # For σ=0.3: need 2κθ ≥ 0.09, so θ ≥ 0.09/(2κ)
        theta = max(0.04, 0.09 / (2 * kappa) + 0.01)  # Add small buffer
        params = HestonParameters(
            kappa=kappa, theta=theta, sigma=0.3, rho=-0.7, mu=0.08
        )
        heston = HestonProcess(params, n_processes=1)
        result = heston.simulate(n_steps=1000, dt=1 / 252, S_0=100.0, random_state=42)
        persistence_results.append(np.sqrt(result.volatilities[:, 0]) * 100)
        persistence_labels.append(
            f"κ = {kappa} ({'Fast' if kappa > 3 else 'Slow' if kappa < 1.5 else 'Medium'} reversion)"
        )

    persistence_matrix = np.column_stack(persistence_results)
    time_years = np.arange(persistence_matrix.shape[0]) / 252

    plotter.plot_paths(
        data=persistence_matrix,
        title="Heston Volatility Persistence - Mean Reversion Speed Impact",
        xlabel="Time (years)",
        ylabel="Volatility (%)",
        time_axis=time_years,
        process_labels=persistence_labels,
        figsize=(12, 6),
    )


def demo_parameter_estimation():
    """Demonstrate Heston parameter estimation."""
    print("\n" + "=" * 60)
    print("HESTON PARAMETER ESTIMATION")
    print("=" * 60)

    stat_plotter = StatisticalPlotter()

    # Generate data with known parameters
    true_params = HestonParameters(kappa=2.0, theta=0.04, sigma=0.25, rho=-0.6, mu=0.07)
    heston_estimation = HestonProcess(true_params, n_processes=1)
    result_estimation = heston_estimation.simulate(
        n_steps=5000, dt=1 / 252, S_0=100.0, random_state=42
    )

    # Estimate parameters from the generated data
    from stochastic_calculus.processes.stochastic_vol.heston import (
        estimate_heston_parameters,
    )

    estimated_params = estimate_heston_parameters(
        result_estimation.prices.flatten(),
        np.sqrt(result_estimation.volatilities.flatten()),
        dt=1 / 252,
    )

    print(f"   True parameters:")
    print(f"     κ = {true_params.kappa:.3f}, θ = {true_params.theta:.3f}")
    print(
        f"     σ = {true_params.sigma:.3f}, ρ = {true_params.rho:.3f}, μ = {true_params.mu:.3f}"
    )

    print(f"\n   Estimated parameters:")
    print(f"     κ = {estimated_params.kappa:.3f}, θ = {estimated_params.theta:.3f}")
    print(
        f"     σ = {estimated_params.sigma:.3f}, ρ = {estimated_params.rho:.3f}, μ = {estimated_params.mu:.3f}"
    )

    # Calculate estimation errors
    errors = {
        "kappa": abs(estimated_params.kappa - true_params.kappa) / true_params.kappa,
        "theta": abs(estimated_params.theta - true_params.theta) / true_params.theta,
        "sigma": abs(estimated_params.sigma - true_params.sigma) / true_params.sigma,
        "rho": abs(estimated_params.rho - true_params.rho) / abs(true_params.rho),
        "mu": abs(estimated_params.mu - true_params.mu) / true_params.mu,
    }

    print(f"\n   Estimation errors:")
    for param, error in errors.items():
        print(f"     {param}: {error:.1%}")

    # Visualize estimation quality using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_estimation.prices,
        title="Heston Parameter Estimation - Price Data Quality Check",
        process_labels=["S(t)"],
    )

    print("   Parameter estimation demonstration complete")


def main():
    try:
        # Run all demonstrations
        demo_single_heston_process()
        demo_parameter_impact()
        demo_multi_asset_heston()
        demo_volatility_clustering()
        demo_parameter_estimation()

    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
