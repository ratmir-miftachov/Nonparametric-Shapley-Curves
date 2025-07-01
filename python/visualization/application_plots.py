"""
Application plots for real-world Shapley analysis.

This module provides comprehensive plotting capabilities for real-world applications,
matching the R application_plots.R functionality with enhanced Python capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler


class ApplicationPlotter:
    """
    Comprehensive plotting for real-world Shapley applications.
    
    This class implements the sophisticated plotting capabilities from R's
    application_plots.R with enhanced Python functionality.
    """
    
    def __init__(self, 
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 150,
                 color_palette: str = 'husl'):
        """Initialize application plotter."""
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = color_palette
        
        # Set plotting style
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('default')
            warnings.warn(f"Style '{self.style}' not found, using default")
        
        # Set color palette
        sns.set_palette(self.color_palette)
        
    def plot_variable_distributions(self, 
                                  data: pd.DataFrame,
                                  variables: List[str],
                                  outlier_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distributions of input variables with outlier filtering.
        
        Matches R boxplot functionality with enhanced statistical information.
        """
        n_vars = len(variables)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=self.dpi)
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            # Filter outliers if bounds provided
            if outlier_bounds and var in outlier_bounds:
                bounds = outlier_bounds[var]
                filtered_data = data[var][(data[var] >= bounds[0]) & (data[var] <= bounds[1])]
            else:
                filtered_data = data[var]
            
            # Create box plot
            box_parts = ax.boxplot(filtered_data, patch_artist=True, 
                                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2))
            
            # Add statistical information
            ax.text(0.02, 0.98, f'Mean: {filtered_data.mean():.2f}\n'
                               f'Std: {filtered_data.std():.2f}\n'
                               f'N: {len(filtered_data)}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{var} Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Variable Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_density_distributions(self, 
                                 data: pd.DataFrame,
                                 variables: List[str],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot kernel density estimates for variables.
        
        Matches R density() functionality.
        """
        n_vars = len(variables)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=self.dpi)
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        colors = sns.color_palette(self.color_palette, n_vars)
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            # Plot density
            data[var].plot(kind='density', ax=ax, color=colors[i], 
                          linewidth=2.5, alpha=0.8)
            
            # Add fill under curve
            x = np.linspace(data[var].min(), data[var].max(), 100)
            kde = stats.gaussian_kde(data[var].dropna())
            density = kde(x)
            ax.fill_between(x, density, alpha=0.3, color=colors[i])
            
            # Add statistics
            mean_val = data[var].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f}')
            
            ax.set_xlabel(f'{var}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{var} Density', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Variable Density Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_shapley_curves_with_ci(self, 
                                   curves: Dict[str, np.ndarray],
                                   evaluation_points: Dict[str, np.ndarray],
                                   confidence_intervals: Dict[str, Dict[str, np.ndarray]],
                                   variable_units: Optional[Dict[str, str]] = None,
                                   time_period: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Shapley curves with confidence intervals.
        
        Matches R slice plot functionality with confidence bands.
        """
        n_vars = len(curves)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), dpi=self.dpi)
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        colors = sns.color_palette(self.color_palette, n_vars)
        
        for i, (var, curve) in enumerate(curves.items()):
            ax = axes[i]
            points = evaluation_points[var]
            color = colors[i]
            
            # Plot main Shapley curve
            ax.plot(points, curve, color='blue', linewidth=2.5, 
                   label='Shapley Curve', zorder=3)
            
            # Plot confidence intervals if available
            if var in confidence_intervals:
                ci = confidence_intervals[var]
                if 'lower' in ci and 'upper' in ci:
                    # Bootstrap confidence intervals
                    ax.fill_between(points, ci['lower'], ci['upper'], 
                                  color='red', alpha=0.3, label='Bootstrap 95% CI')
                    ax.plot(points, ci['lower'], color='red', linewidth=1, alpha=0.7)
                    ax.plot(points, ci['upper'], color='red', linewidth=1, alpha=0.7)
                
                # Alternative confidence intervals if available
                if 'lower_alt' in ci and 'upper_alt' in ci:
                    ax.fill_between(points, ci['lower_alt'], ci['upper_alt'], 
                                  color='blue', alpha=0.2, label='Alternative 95% CI')
                    ax.plot(points, ci['lower_alt'], color='blue', linewidth=1, alpha=0.7)
                    ax.plot(points, ci['upper_alt'], color='blue', linewidth=1, alpha=0.7)
            
            # Add zero reference line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.6, linewidth=1)
            
            # Formatting
            unit_str = f" [{variable_units[var]}]" if variable_units and var in variable_units else ""
            time_str = f", {time_period}" if time_period else ""
            ax.set_xlabel(f'{var}{unit_str}{time_str}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Shapley Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Shapley Curve: {var}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits with some padding
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min * 1.1, y_max * 1.1)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        title = f'Shapley Curves with Confidence Intervals'
        if time_period:
            title += f' ({time_period})'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_combined_analysis_plot(self, 
                                    data: pd.DataFrame,
                                    shapley_results: Dict[str, Any],
                                    variables: List[str],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive analysis plot combining distributions and Shapley curves.
        
        This matches the comprehensive R application analysis.
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, len(variables), hspace=0.3, wspace=0.3)
        
        colors = sns.color_palette(self.color_palette, len(variables))
        
        # Top row: Variable distributions
        for i, var in enumerate(variables):
            ax = fig.add_subplot(gs[0, i])
            data[var].plot(kind='density', ax=ax, color=colors[i], linewidth=2)
            ax.set_title(f'{var} Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Middle row: Box plots
        for i, var in enumerate(variables):
            ax = fig.add_subplot(gs[1, i])
            ax.boxplot(data[var], patch_artist=True,
                      boxprops=dict(facecolor=colors[i], alpha=0.7))
            ax.set_title(f'{var} Box Plot', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Bottom row: Shapley curves
        for i, var in enumerate(variables):
            ax = fig.add_subplot(gs[2, i])
            if var in shapley_results['curves']:
                points = shapley_results['evaluation_points'][var]
                curve = shapley_results['curves'][var]
                ax.plot(points, curve, color=colors[i], linewidth=2.5)
                
                # Add confidence intervals if available
                if 'confidence_intervals' in shapley_results and var in shapley_results['confidence_intervals']:
                    ci = shapley_results['confidence_intervals'][var]
                    if 'lower' in ci and 'upper' in ci:
                        ax.fill_between(points, ci['lower'], ci['upper'], 
                                      color=colors[i], alpha=0.3)
                
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.6)
                ax.set_title(f'Shapley: {var}', fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Shapley Analysis', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_application_analysis(data: pd.DataFrame, 
                            shapley_results: Dict[str, Any],
                            variables: List[str],
                            save_path: Optional[str] = None) -> plt.Figure:
    """Quick function for comprehensive application analysis plotting."""
    plotter = ApplicationPlotter()
    return plotter.create_combined_analysis_plot(data, shapley_results, variables, save_path) 