"""Comprehensive demo for Lévy processes.

This demo showcases:
- Variance Gamma processes (single and multi-path)
- Normal Inverse Gaussian processes
- Parameter impact analysis (theta, sigma, nu, alpha, beta, delta)
- Characteristic function analysis
- Lévy measure visualization
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
from stochastic_calculus.processes.levy import create_simple_vg, create_simple_nig
from stochastic_calculus.visualization import ProcessPlotter, StatisticalPlotter

RANDOM_STATE = 123


def demo_levy_processes():
    """Demonstrate Lévy processes with generic visualizations."""
    print("\nLévy Processes Demo")

    # Create visualization objects
    plotter = ProcessPlotter()
    stat_plotter = StatisticalPlotter()

    print("1. Single Variance Gamma Process")

    # Create VG process with realistic parameters
    vg_single = create_simple_vg(
        theta=-0.1,  # Slight negative drift (market crashes)
        sigma=0.3,  # Volatility scaling
        nu=0.2,  # Jump activity (smaller = more jumps)
    )

    # Simulate single path
    result_vg = vg_single.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)

    time_years = np.arange(result_vg.paths.shape[0]) / 252

    # Generic path visualization
    plotter.plot_paths(
        data=result_vg.paths,
        title="Single Variance Gamma Process - Path",
        xlabel="Time (years)",
        ylabel="Process Value",
        time_axis=time_years,
        process_labels=["VG Process"],
        figsize=(12, 6),
    )

    # Statistical analysis using generic tools
    stat_plotter.plot_distribution_analysis(
        data=result_vg.paths,
        title="Variance Gamma - Statistical Properties",
        process_labels=["X(t)"],
    )

    print(
        f"VG Process range: [{np.min(result_vg.paths):.3f}, {np.max(result_vg.paths):.3f}]"
    )
    print(f"Average subordinator increment: {np.mean(result_vg.subordinator):.6f}")

    print("\n2. Parameter Impact Analysis - Variance Gamma")

    # Test different nu values (jump activity)
    nu_values = [0.1, 0.3, 0.8]
    nu_results = []
    nu_labels = []

    for nu in nu_values:
        print(f"Simulating VG with ν = {nu}")
        vg_param = create_simple_vg(theta=-0.1, sigma=0.3, nu=nu)
        result = vg_param.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)
        nu_results.append(result.paths[:, 0])
        nu_labels.append(f"ν = {nu}")

    # Generic visualization for parameter comparison
    nu_matrix = np.column_stack(nu_results)
    plotter.plot_paths(
        data=nu_matrix,
        title="Variance Gamma - Jump Activity Impact (ν parameter)",
        xlabel="Time (years)",
        ylabel="Process Value",
        time_axis=time_years,
        process_labels=nu_labels,
        figsize=(12, 6),
    )

    # Test different theta values (asymmetry)
    theta_values = [-0.3, 0.0, 0.2]
    theta_results = []
    theta_labels = []

    for theta in theta_values:
        print(f"Simulating VG with θ = {theta}")
        vg_theta = create_simple_vg(theta=theta, sigma=0.3, nu=0.2)
        result = vg_theta.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)
        theta_results.append(result.paths[:, 0])
        theta_labels.append(f"θ = {theta}")

    theta_matrix = np.column_stack(theta_results)
    plotter.plot_paths(
        data=theta_matrix,
        title="Variance Gamma - Asymmetry Impact (θ parameter)",
        xlabel="Time (years)",
        ylabel="Process Value",
        time_axis=time_years,
        process_labels=theta_labels,
        figsize=(12, 6),
    )

    print("\n3. Normal Inverse Gaussian Process")

    # Create NIG process with realistic parameters
    nig_single = create_simple_nig(
        alpha=15.0,  # Tail heaviness
        beta=-2.0,  # Negative asymmetry
        delta=0.5,  # Scale parameter
        mu=0.02,  # Location parameter
    )

    result_nig = nig_single.simulate(
        n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE
    )

    plotter.plot_paths(
        data=result_nig.paths,
        title="Single Normal Inverse Gaussian Process - Path",
        xlabel="Time (years)",
        ylabel="Process Value",
        time_axis=time_years,
        process_labels=["NIG Process"],
        figsize=(12, 6),
    )

    # Statistical analysis
    stat_plotter.plot_distribution_analysis(
        data=result_nig.paths,
        title="Normal Inverse Gaussian - Statistical Properties",
        process_labels=["X(t)"],
    )

    print(
        f"NIG Process range: [{np.min(result_nig.paths):.3f}, {np.max(result_nig.paths):.3f}]"
    )
    print(f"Average IG subordinator: {np.mean(result_nig.inverse_gaussian):.6f}")

    print("\n4. NIG Parameter Impact Analysis")

    # Test different alpha values (tail heaviness)
    alpha_values = [5.0, 15.0, 30.0]
    alpha_results = []
    alpha_labels = []

    for alpha in alpha_values:
        print(f"Simulating NIG with α = {alpha}")
        # Ensure constraint |β| < α
        beta_safe = min(-1.0, -alpha * 0.1)
        nig_alpha = create_simple_nig(alpha=alpha, beta=beta_safe, delta=0.5, mu=0.02)
        result = nig_alpha.simulate(n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE)
        alpha_results.append(result.paths[:, 0])
        alpha_labels.append(f"α = {alpha}")

    alpha_matrix = np.column_stack(alpha_results)
    plotter.plot_paths(
        data=alpha_matrix,
        title="NIG Process - Tail Heaviness Impact (α parameter)",
        xlabel="Time (years)",
        ylabel="Process Value",
        time_axis=time_years,
        process_labels=alpha_labels,
        figsize=(12, 6),
    )

    print("\n5. Lévy Process Comparison")

    # Create comparable VG and NIG processes
    vg_comp = create_simple_vg(theta=-0.1, sigma=0.3, nu=0.3)
    nig_comp = create_simple_nig(alpha=10.0, beta=-1.0, delta=0.3, mu=-0.05)

    # Simulate multiple paths for comparison
    n_paths = 5
    vg_paths = []
    nig_paths = []

    for i in range(n_paths):
        vg_result = vg_comp.simulate(
            n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE + i * 10
        )
        nig_result = nig_comp.simulate(
            n_steps=1000, dt=1 / 252, random_state=RANDOM_STATE + i * 10
        )
        vg_paths.append(vg_result.paths[:, 0])
        nig_paths.append(nig_result.paths[:, 0])

    # Combine for comparison visualization
    all_paths = []
    all_labels = []

    for i in range(min(3, n_paths)):  # Show first 3 paths of each
        all_paths.extend([vg_paths[i], nig_paths[i]])
        all_labels.extend([f"VG {i+1}", f"NIG {i+1}"])

    comparison_matrix = np.column_stack(all_paths)
    plotter.plot_paths(
        data=comparison_matrix,
        title="Lévy Processes Comparison - VG vs NIG",
        xlabel="Time (years)",
        ylabel="Process Value",
        time_axis=time_years,
        process_labels=all_labels,
        figsize=(12, 6),
    )

    # Statistical comparison using generic tools
    vg_final = np.array([path[-1] for path in vg_paths])
    nig_final = np.array([path[-1] for path in nig_paths])
    comparison_data = np.column_stack([vg_final, nig_final])

    stat_plotter.plot_distribution_analysis(
        data=comparison_data,
        title="Lévy Processes - Final Value Comparison",
        process_labels=["VG", "NIG"],
    )

    print("\n6. Correlation Analysis")

    # Generate multiple paths for correlation analysis
    n_corr_paths = 10
    model_names = ["VG", "NIG"]

    vg_corr = create_simple_vg(theta=-0.05, sigma=0.25, nu=0.25)
    nig_corr = create_simple_nig(alpha=12.0, beta=-1.5, delta=0.4, mu=0.0)

    # Simulate final values for correlation
    vg_finals = []
    nig_finals = []

    for i in range(n_corr_paths):
        vg_res = vg_corr.simulate(
            n_steps=252, dt=1 / 252, random_state=RANDOM_STATE + i * 5
        )
        nig_res = nig_corr.simulate(
            n_steps=252, dt=1 / 252, random_state=RANDOM_STATE + i * 5
        )

        vg_finals.append(vg_res.paths[-1, 0])
        nig_finals.append(nig_res.paths[-1, 0])

    # Correlation analysis using generic heatmap
    corr_data = np.column_stack([vg_finals, nig_finals])
    plotter.plot_correlation_heatmap(
        data=corr_data.T,
        title="Lévy Processes - Final Value Correlations",
        process_labels=model_names,
    )

    # Calculate and display correlations
    correlation = np.corrcoef(vg_finals, nig_finals)[0, 1]
    print(f"VG vs NIG correlation: {correlation:.3f}")

    print("\n7. Jump Activity Analysis")

    # Analyze jump characteristics of both processes
    vg_activity = create_simple_vg(theta=0.0, sigma=0.2, nu=0.1)  # High activity
    nig_activity = create_simple_nig(alpha=20.0, beta=0.0, delta=0.3, mu=0.0)

    vg_jumps = vg_activity.simulate(n_steps=2000, dt=1 / 252, random_state=RANDOM_STATE)
    nig_jumps = nig_activity.simulate(
        n_steps=2000, dt=1 / 252, random_state=RANDOM_STATE
    )

    # Analyze increment distributions (proxy for jump activity)
    vg_increments = np.diff(vg_jumps.paths[:, 0])
    nig_increments = np.diff(nig_jumps.paths[:, 0])

    increment_data = np.column_stack([vg_increments, nig_increments])
    stat_plotter.plot_distribution_analysis(
        data=increment_data,
        title="Lévy Processes - Increment Distributions (Jump Activity)",
        process_labels=["VG Increments", "NIG Increments"],
    )

    # Print statistics about jump activity
    print(f"VG increment std: {np.std(vg_increments):.4f}")
    print(f"NIG increment std: {np.std(nig_increments):.4f}")
    print(
        f"VG increment kurtosis: {np.mean(((vg_increments - np.mean(vg_increments)) / np.std(vg_increments))**4):.2f}"
    )
    print(
        f"NIG increment kurtosis: {np.mean(((nig_increments - np.mean(nig_increments)) / np.std(nig_increments))**4):.2f}"
    )


def main():
    """Run the comprehensive Lévy processes demonstration."""
    try:
        demo_levy_processes()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
