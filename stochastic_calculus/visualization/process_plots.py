"""Process path plotting utilities with classic mathematical formatting."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class ProcessPlotter:
    """Plotting utilities for stochastic process paths with analytical formatting."""

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

    def plot_paths(
        self,
        data: np.ndarray,
        title: str = "Stochastic Process Paths",
        xlabel: str = "Time Steps",
        ylabel: str = "Value",
        time_axis: Optional[np.ndarray] = None,
        process_labels: Optional[list[str]] = None,
        figsize: tuple[int, int] = (12, 6),
        log_scale: bool = False,
    ) -> None:
        """
        Plot process paths.

        Args:
            data: Process data (1D or 2D array)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            time_axis: Custom time axis (optional)
            process_labels: Labels for processes
            figsize: Figure size
            log_scale: Use log scale for y-axis
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure 2D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_steps, n_processes = data.shape

        # Set time axis
        if time_axis is None:
            time_axis = np.arange(n_steps)

        # Plot each process
        for i in range(n_processes):
            label = process_labels[i] if process_labels else f"Process {i+1}"
            ax.plot(time_axis, data[:, i], linewidth=2, label=label)

        # Classic mathematical formatting
        ax.set_title(title, fontweight='normal', pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if n_processes > 1:
            ax.legend(frameon=True, fancybox=False, shadow=False, 
                     edgecolor='black', framealpha=0.9)

        if log_scale:
            ax.set_yscale("log")

        # Mathematical grid style
        ax.grid(True, linestyle='-', alpha=0.2, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', alpha=0.1, color='gray')
        
        # Clean mathematical appearance
        ax.tick_params(direction='in', which='both')
        
        plt.tight_layout()
        plt.show()

    def plot_price_and_volatility(
        self,
        prices: np.ndarray,
        volatilities: np.ndarray,
        time_axis: Optional[np.ndarray] = None,
        asset_names: Optional[list[str]] = None,
        figsize: tuple[int, int] = (12, 10),
    ) -> None:
        """
        Plot asset prices and their volatilities.

        Args:
            prices: Price data
            volatilities: Volatility data
            time_axis: Time axis
            asset_names: Asset names
            figsize: Figure size
        """
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            volatilities = volatilities.reshape(-1, 1)

        n_steps, n_assets = prices.shape

        if time_axis is None:
            time_axis = np.arange(n_steps)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot prices
        for i in range(n_assets):
            label = asset_names[i] if asset_names else f"Asset {i+1}"
            ax1.plot(time_axis, prices[:, i], linewidth=2, label=label)

        ax1.set_title("Asset Prices", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price", fontsize=12)
        if n_assets > 1:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot volatilities (convert to percentage if needed)
        vol_data = volatilities * 100 if np.max(volatilities) <= 1 else volatilities

        for i in range(n_assets):
            label = asset_names[i] if asset_names else f"Asset {i+1}"
            ax2.plot(time_axis, vol_data[:, i], linewidth=2, label=label)

        ax2.set_title("Volatilities", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Volatility (%)", fontsize=12)
        if n_assets > 1:
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(
        self,
        data: np.ndarray,
        title: str = "Correlation Matrix",
        process_labels: Optional[list[str]] = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            data: Process data
            title: Plot title
            process_labels: Process labels
            figsize: Figure size
        """
        if data.ndim == 1:
            print("Cannot plot correlation for 1D data")
            return

# Removed seaborn dependency for pure matplotlib mathematical formatting

        # Calculate correlation matrix
        if data.shape[0] > data.shape[1]:
            # Data is (time_steps, n_processes) - normal case
            corr_matrix = np.corrcoef(data.T)
        else:
            # Data is (n_processes, time_steps) - transposed case
            corr_matrix = np.corrcoef(data)

        fig, ax = plt.subplots(figsize=figsize)

        # Limit annotation for large matrices to avoid overcrowding
        show_annot = corr_matrix.shape[0] <= 10

        # Create mathematical heatmap without seaborn dependency
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect='equal')
        
        # Add colorbar with mathematical formatting
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient $\\rho$', rotation=270, labelpad=20)
        
        # Set ticks and labels
        n_vars = corr_matrix.shape[0]
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        
        if process_labels:
            ax.set_xticklabels(process_labels, rotation=45, ha='right')
            ax.set_yticklabels(process_labels, rotation=0)
        else:
            ax.set_xticklabels([f'${i+1}$' for i in range(n_vars)], rotation=45, ha='right')
            ax.set_yticklabels([f'${i+1}$' for i in range(n_vars)], rotation=0)
        
        # Add correlation values as text annotations
        if show_annot:
            for i in range(n_vars):
                for j in range(n_vars):
                    text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                           ha='center', va='center', color=text_color, fontweight='bold')

        ax.set_title(title, fontweight='normal', pad=20)
        
        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()
        plt.show()
