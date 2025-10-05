"""Comprehensive demo for jump diffusion processes.

This demo showcases:
- Merton jump diffusion model (single and multi-path)
- Kou double exponential jump diffusion model
- Bates model (Heston with jumps)
- Parameter impact analysis (jump intensity, mean, volatility)
- Jump-specific statistical analysis and timing
- Model comparison and correlation analysis
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
from stochastic_calculus.processes.jump_diffusion import (
    create_simple_merton,
    create_simple_kou,
    create_simple_bates,
)
from stochastic_calculus.visualization import ProcessPlotter, StatisticalPlotter

RANDOM_STATE = 42


def demo_jump_diffusion_processes():
    """Demonstrate jump diffusion processes with generic visualizations."""
    print("\nJump Diffusion Processes Demo")

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    print("1. Single Merton Jump Diffusion Model")

    # Create Merton model with realistic parameters
    merton_single = create_simple_merton(
        mu=0.08,  # 8% expected return
        sigma=0.20,  # 20% diffusion volatility
        jump_intensity=3.0,  # 3 jumps per year on average
        jump_mean=-0.02,  # Slightly negative mean jumps
        jump_volatility=0.15,  # Jump volatility
    )

    # Simulate single path
    result_single = merton_single.simulate_with_initial_value(
        n_steps=1000, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE
    )

    time_years = np.arange(result_single.prices.shape[0]) / 252

    # Generic path visualization for prices
    plotter.plot_paths(
        data=result_single.prices,
        title="Single Merton Jump Diffusion - Price Path",
        xlabel="Time (years)",
        ylabel="Stock Price ($)",
        time_axis=time_years,
        process_labels=["Merton Model"],
        figsize=(12, 6),
    )

    # Generic path visualization for log prices (shows jumps clearly)
    plotter.plot_paths(
        data=result_single.log_prices,
        title="Single Merton Jump Diffusion - Log Price Path",
        xlabel="Time (years)",
        ylabel="Log Price",
        time_axis=time_years,
        process_labels=["Log Price"],
        figsize=(12, 6),
    )

    # Statistical analysis using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_single.prices,
        title="Merton Jump Diffusion - Price Statistical Properties",
        process_labels=["S(t)"],
    )

    # Display jump information
    print(f"Number of jumps: {len(result_single.jump_times[0])}")
    if result_single.jump_sizes[0]:
        print(f"Average jump size: {np.mean(result_single.jump_sizes[0]):.4f}")
        print(f"Jump times: {[f'{t:.3f}' for t in result_single.jump_times[0][:5]]}")
    print(f"Final price: {result_single.prices[-1, 0]:.2f}")

    print("\n2. Parameter Impact Analysis")

    # Test different jump intensities
    jump_intensities = [1.0, 3.0, 8.0]
    intensity_results = []
    intensity_labels = []

    for intensity in jump_intensities:
        print(f"Simulating jump intensity λ = {intensity}")
        merton_param = create_simple_merton(
            mu=0.08,
            sigma=0.20,
            jump_intensity=intensity,
            jump_mean=-0.02,
            jump_volatility=0.15,
        )
        result = merton_param.simulate_with_initial_value(
            n_steps=1000, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE
        )
        intensity_results.append(result.prices[:, 0])
        intensity_labels.append(f"λ = {intensity}")

    # Generic visualization for parameter comparison
    intensity_matrix = np.column_stack(intensity_results)
    plotter.plot_paths(
        data=intensity_matrix,
        title="Merton Model - Jump Intensity Impact",
        xlabel="Time (years)",
        ylabel="Stock Price ($)",
        time_axis=time_years,
        process_labels=intensity_labels,
        figsize=(12, 6),
    )

    # Test different jump volatilities
    jump_vols = [0.05, 0.15, 0.30]
    vol_results = []
    vol_labels = []

    for vol in jump_vols:
        print(f"Simulating jump volatility σ_J = {vol}")
        merton_vol = create_simple_merton(
            mu=0.08,
            sigma=0.20,
            jump_intensity=3.0,
            jump_mean=-0.02,
            jump_volatility=vol,
        )
        result = merton_vol.simulate_with_initial_value(
            n_steps=1000, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE
        )
        vol_results.append(result.prices[:, 0])
        vol_labels.append(f"σ_J = {vol}")

    vol_matrix = np.column_stack(vol_results)
    plotter.plot_paths(
        data=vol_matrix,
        title="Merton Model - Jump Volatility Impact",
        xlabel="Time (years)",
        ylabel="Stock Price ($)",
        time_axis=time_years,
        process_labels=vol_labels,
        figsize=(12, 6),
    )

    print("\n3. Multi-Model Comparison")

    # Create different jump diffusion models with comparable parameters
    merton_comp = create_simple_merton(mu=0.08, sigma=0.20, jump_intensity=3.0)
    kou_comp = create_simple_kou(
        mu=0.08, sigma=0.20, jump_intensity=3.0, p_up=0.4, eta_up=8.0, eta_down=12.0
    )

    # Simulate multiple paths for comparison
    n_paths = 5
    merton_paths = []
    kou_paths = []

    for i in range(n_paths):
        merton_result = merton_comp.simulate_with_initial_value(
            n_steps=1000, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE + i
        )
        kou_result = kou_comp.simulate(
            n_steps=1000, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE + i
        )
        merton_paths.append(merton_result.prices[:, 0])
        kou_paths.append(kou_result.prices[:, 0])

    # Combine for comparison visualization
    all_paths = []
    all_labels = []

    for i in range(min(3, n_paths)):  # Show first 3 paths of each
        all_paths.extend([merton_paths[i], kou_paths[i]])
        all_labels.extend([f"Merton {i+1}", f"Kou {i+1}"])

    comparison_matrix = np.column_stack(all_paths)
    plotter.plot_paths(
        data=comparison_matrix,
        title="Jump Diffusion Models Comparison",
        xlabel="Time (years)",
        ylabel="Stock Price ($)",
        time_axis=time_years,
        process_labels=all_labels,
        figsize=(12, 6),
    )

    # Statistical comparison using generic tools
    merton_final = np.array([path[-1] for path in merton_paths])
    kou_final = np.array([path[-1] for path in kou_paths])
    comparison_data = np.column_stack([merton_final, kou_final])

    stat_plotter.plot_distribution_analysis(
        data=comparison_data,
        title="Jump Models - Final Price Comparison",
        process_labels=["Merton", "Kou"],
    )

    print("\n4. Kou Model Asymmetric Jumps Analysis")

    # Demonstrate Kou model's asymmetric jump behavior
    kou_asym = create_simple_kou(
        mu=0.08,
        sigma=0.20,
        jump_intensity=5.0,
        p_up=0.3,
        eta_up=15.0,
        eta_down=8.0,  # More frequent, larger downward jumps
    )

    kou_result = kou_asym.simulate(
        n_steps=1000, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE
    )

    plotter.plot_paths(
        data=kou_result.prices,
        title="Kou Model - Asymmetric Jump Demonstration",
        xlabel="Time (years)",
        ylabel="Stock Price ($)",
        time_axis=time_years,
        process_labels=["Kou Asymmetric"],
        figsize=(12, 6),
    )

    # Analyze jump directions
    up_jumps = []
    down_jumps = []
    if kou_result.jump_directions[0]:
        for i, direction in enumerate(kou_result.jump_directions[0]):
            if i < len(kou_result.jump_sizes[0]):
                if direction == 1:
                    up_jumps.append(kou_result.jump_sizes[0][i])
                else:
                    down_jumps.append(kou_result.jump_sizes[0][i])

    print(f"Upward jumps: {len(up_jumps)}")
    print(f"Downward jumps: {len(down_jumps)}")
    if up_jumps:
        print(f"Average upward jump: {np.mean(up_jumps):.4f}")
    if down_jumps:
        print(f"Average downward jump: {np.mean(down_jumps):.4f}")

    print("\n5. Bates Model (Heston + Jumps)")

    # Create Bates model combining stochastic volatility with jumps
    bates = create_simple_bates(
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        mu=0.05,
        price_jump_intensity=2.0,
        price_jump_mean=-0.01,
        price_jump_volatility=0.12,
    )

    bates_result = bates.simulate(
        n_steps=1000, dt=1 / 252, S_0=100.0, v_0=0.04, random_state=RANDOM_STATE
    )

    # Use specialized price and volatility visualization
    plotter.plot_price_and_volatility(
        bates_result.prices,
        np.sqrt(bates_result.volatilities) * 100,  # Convert to %
        time_axis=time_years,
        asset_names=["Bates Model"],
    )

    # Leverage effect analysis with jumps
    stat_plotter.plot_returns_vs_volatility(
        bates_result.prices,
        np.sqrt(bates_result.volatilities),
        title="Bates Model - Returns vs Volatility (with Jumps)",
    )

    print(f"Price jumps: {len(bates_result.price_jump_times[0])}")
    print(f"Volatility jumps: {len(bates_result.vol_jump_times[0])}")

    print("\n6. Correlation Analysis")

    # Generate multiple paths from different models for correlation analysis
    n_corr_paths = 10
    model_names = ["Merton", "Kou", "Bates_Price"]

    merton_corr = create_simple_merton(mu=0.08, sigma=0.20, jump_intensity=3.0)
    kou_corr = create_simple_kou(
        mu=0.08, sigma=0.20, jump_intensity=3.0, p_up=0.5, eta_up=10.0, eta_down=10.0
    )

    # Simulate final prices for correlation
    merton_finals = []
    kou_finals = []
    bates_finals = []

    for i in range(n_corr_paths):
        m_res = merton_corr.simulate_with_initial_value(
            n_steps=252, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE + i * 10
        )
        k_res = kou_corr.simulate(
            n_steps=252, dt=1 / 252, S_0=100.0, random_state=RANDOM_STATE + i * 10
        )
        b_res = bates.simulate(
            n_steps=252,
            dt=1 / 252,
            S_0=100.0,
            v_0=0.04,
            random_state=RANDOM_STATE + i * 10,
        )

        merton_finals.append(m_res.prices[-1, 0])
        kou_finals.append(k_res.prices[-1, 0])
        bates_finals.append(b_res.prices[-1, 0])

    # Correlation analysis using generic heatmap
    corr_data = np.column_stack([merton_finals, kou_finals, bates_finals])
    plotter.plot_correlation_heatmap(
        data=corr_data.T,  # Transpose for correlation between models
        title="Jump Diffusion Models - Final Price Correlations",
        process_labels=model_names,
    )

    # Calculate and display correlations
    corr_matrix = np.corrcoef(corr_data.T)
    print("Model correlations:")
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                print(f"{name1} vs {name2}: {corr_matrix[i, j]:.3f}")


def main():
    """Run the comprehensive jump diffusion demonstration."""
    try:
        demo_jump_diffusion_processes()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
