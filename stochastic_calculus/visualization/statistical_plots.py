"""Statistical analysis plotting utilities with mathematical formatting."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class StatisticalPlotter:
    """Statistical analysis plotting for stochastic processes with analytical formatting."""

    def __init__(self, mathematical_style: bool = True) -> None:
        """Initialize plotter with mathematical formatting."""
        if mathematical_style:
            # Configure matplotlib for classic mathematical appearance
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'text.usetex': False,  # Set to True if LaTeX is available
                'mathtext.fontset': 'cm',  # Computer Modern math fonts
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.axisbelow': True,
                'lines.linewidth': 1.5,
                'figure.dpi': 100,
                'savefig.dpi': 300,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.linewidth': 0.8
            })

    def plot_distribution_analysis(
        self,
        data: np.ndarray,
        title: str = "Distribution Analysis",
        process_labels: Optional[list[str]] = None,
        figsize: tuple[int, int] = (15, 5),
    ) -> None:
        """
        Plot distribution analysis.

        Args:
            data: Process data
            title: Plot title
            process_labels: Process labels
            figsize: Figure size
        """
        # Calculate increments
        if data.ndim == 1:
            increments = np.diff(data)
            n_processes = 1
        else:
            increments = np.diff(data, axis=0)
            n_processes = data.shape[1]
            # If single process but 2D, flatten for convenience
            if n_processes == 1:
                increments = increments.flatten()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Histogram with mathematical formatting
        if n_processes == 1:
            n, bins, patches = axes[0].hist(increments, bins=50, density=True, 
                                          alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
            # Add theoretical normal curve overlay
            x = np.linspace(increments.min(), increments.max(), 100)
            axes[0].plot(x, stats.norm.pdf(x, increments.mean(), increments.std()), 
                        'r-', linewidth=2, label='$\\mathcal{N}(\\mu, \\sigma^2)$')
            axes[0].legend()
        else:
            for i in range(min(n_processes, 5)):  # Limit to 5 processes for clarity
                label = process_labels[i] if process_labels else f"Process ${i+1}$"
                axes[0].hist(
                    increments[:, i], bins=50, alpha=0.6, density=True, label=label,
                    edgecolor='black', linewidth=0.3
                )
            axes[0].legend(frameon=True, fancybox=False, shadow=False, 
                          edgecolor='black', framealpha=0.9)

        axes[0].set_title("Distribution of Increments", fontweight='normal', pad=15)
        axes[0].set_xlabel("Increment Value $\\Delta X$")
        axes[0].set_ylabel("Probability Density $f(\\Delta X)$")
        axes[0].grid(True, alpha=0.2, linestyle='-')
        axes[0].minorticks_on()
        axes[0].grid(True, which='minor', alpha=0.1, linestyle=':')
        axes[0].tick_params(direction='in', which='both')

        # 2. Q-Q plot with mathematical formatting
        if n_processes == 1:
            stats.probplot(increments, dist="norm", plot=axes[1])
        else:
            # Use only the first process for Q-Q plot to avoid overlapping
            stats.probplot(increments[:, 0], dist="norm", plot=axes[1])
            if process_labels:
                axes[1].get_lines()[0].set_label(process_labels[0])
                axes[1].legend(frameon=True, fancybox=False, shadow=False, 
                              edgecolor='black', framealpha=0.9)

        # Improve Q-Q plot appearance
        line = axes[1].get_lines()
        if len(line) >= 2:
            line[0].set_markerfacecolor('steelblue')
            line[0].set_markeredgecolor('black')
            line[0].set_markeredgewidth(0.5)
            line[0].set_markersize(4)
            line[1].set_color('red')
            line[1].set_linewidth(2)

        axes[1].set_title("Quantile-Quantile Plot vs Normal", fontweight='normal', pad=15)
        axes[1].set_xlabel("Theoretical Quantiles")
        axes[1].set_ylabel("Ordered Values")
        axes[1].grid(True, alpha=0.2, linestyle='-')
        axes[1].minorticks_on()
        axes[1].grid(True, which='minor', alpha=0.1, linestyle=':')
        axes[1].tick_params(direction='in', which='both')

        # 3. Autocorrelation function
        max_lag = min(50, len(increments) // 4)

        if n_processes == 1:
            autocorr = self._calculate_autocorr(increments, max_lag)
            axes[2].plot(range(max_lag), autocorr, "steelblue", linewidth=2, marker='o', 
                        markersize=3, markerfacecolor='steelblue', markeredgecolor='black',
                        markeredgewidth=0.3)
        else:
            for i in range(min(n_processes, 3)):
                autocorr = self._calculate_autocorr(increments[:, i], max_lag)
                label = process_labels[i] if process_labels else f"Process ${i+1}$"
                axes[2].plot(range(max_lag), autocorr, linewidth=2, label=label, marker='o',
                           markersize=3, markeredgewidth=0.3)
            axes[2].legend(frameon=True, fancybox=False, shadow=False, 
                          edgecolor='black', framealpha=0.9)

        axes[2].set_title("Autocorrelation Function", fontweight='normal', pad=15)
        axes[2].set_xlabel("Lag $k$")
        axes[2].set_ylabel("Autocorrelation $\\rho(k)$")
        axes[2].axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1)
        axes[2].grid(True, alpha=0.2, linestyle='-')
        axes[2].minorticks_on()
        axes[2].grid(True, which='minor', alpha=0.1, linestyle=':')
        axes[2].tick_params(direction='in', which='both')

        plt.suptitle(title, fontweight='normal', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_returns_vs_volatility(
        self,
        prices: np.ndarray,
        volatilities: np.ndarray,
        title: str = "Returns vs Volatility",
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot returns against volatility (leverage effect analysis).

        Args:
            prices: Price data
            volatilities: Volatility data
            title: Plot title
            figsize: Figure size
        """
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            volatilities = volatilities.reshape(-1, 1)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        n_assets = prices.shape[1]

        for i in range(n_assets):
            # Calculate returns
            returns = np.diff(np.log(prices[:, i])) * 100  # Log returns in %
            vol_pct = (
                volatilities[1:, i] * 100
                if np.max(volatilities) <= 1
                else volatilities[1:, i]
            )

            # Scatter plot
            axes[0].scatter(vol_pct, returns, alpha=0.5, s=1, label=f"Asset {i+1}")

            # Rolling correlation
            window = min(50, len(returns) // 4)
            rolling_corr = []
            for j in range(window, len(returns)):
                corr = np.corrcoef(returns[j - window : j], vol_pct[j - window : j])[
                    0, 1
                ]
                rolling_corr.append(corr)

            axes[1].plot(rolling_corr, label=f"Asset {i+1}")

        axes[0].set_xlabel("Volatility (%)")
        axes[0].set_ylabel("Returns (%)")
        axes[0].set_title("Returns vs Volatility Scatter")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Rolling Correlation")
        axes[1].set_title("Rolling Returns-Volatility Correlation")
        axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1].grid(True, alpha=0.3)

        if n_assets > 1:
            axes[0].legend()
            axes[1].legend()

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _calculate_autocorr(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        autocorr = np.correlate(data, data, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        autocorr = autocorr[:max_lag] / autocorr[0]
        return autocorr

    def plot_parameter_evolution(
        self,
        data: np.ndarray,
        estimation_func,
        window_size: int = 252,
        title: str = "Parameter Evolution",
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot evolution of estimated parameters over time.

        Args:
            data: Time series data
            estimation_func: Function to estimate parameters
            window_size: Rolling window size
            title: Plot title
            figsize: Figure size
        """
        if len(data) < 2 * window_size:
            print(f"Data too short for window size {window_size}")
            return

        # Rolling parameter estimation
        param_history = []
        time_points = []

        for i in range(window_size, len(data)):
            window_data = data[i - window_size : i]
            try:
                params = estimation_func(window_data)
                param_history.append(params)
                time_points.append(i)
            except Exception:
                continue

        if not param_history:
            print("No valid parameter estimates")
            return

        # Extract parameter values
        param_dict = {}
        for key in param_history[0].__dict__.keys():
            param_dict[key] = [getattr(p, key) for p in param_history]

        # Plot
        n_params = len(param_dict)
        fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)

        if n_params == 1:
            axes = [axes]

        for i, (param_name, values) in enumerate(param_dict.items()):
            axes[i].plot(time_points, values, linewidth=2)
            axes[i].set_ylabel(param_name)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"{param_name} Evolution")

        axes[-1].set_xlabel("Time")
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
