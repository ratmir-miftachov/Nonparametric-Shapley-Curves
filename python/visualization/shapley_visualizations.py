import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

class ShapleyVisualizer:
    """Visualization tools for Shapley curves and analysis."""
    
    def __init__(self, 
                 style: str = 'seaborn-v0_8',
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 100,
                 color_palette: Optional[List[str]] = None):
        """Initialize visualizer with styling options."""
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = color_palette or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('default')
            warnings.warn(f"Style '{self.style}' not found, using default")

    def plot_shapley_curves(self, 
                           curves: Dict[str, np.ndarray],
                           evaluation_points: Dict[str, np.ndarray],
                           confidence_intervals: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                           title: str = "Shapley Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot Shapley curves for multiple variables."""
        n_vars = len(curves)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=self.dpi)
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_vars == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (var, curve) in enumerate(curves.items()):
            ax = axes[i] if n_vars > 1 else axes[0]
            points = evaluation_points[var]
            color = self.color_palette[i % len(self.color_palette)]
            
            # Plot main curve
            ax.plot(points, curve, color=color, linewidth=2.5, label=f'Shapley curve')
            
            # Plot confidence intervals if available
            if confidence_intervals and var in confidence_intervals:
                ci = confidence_intervals[var]
                if 'lower' in ci and 'upper' in ci:
                    ax.fill_between(points, ci['lower'], ci['upper'], 
                                  color=color, alpha=0.2, label='95% CI')
            
            # Styling
            ax.set_xlabel(f'{var}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Shapley Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Shapley Curve for {var}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def quick_shapley_plot(curves, evaluation_points, save_path=None):
    """Quick plotting function for Shapley curves."""
    viz = ShapleyVisualizer()
    return viz.plot_shapley_curves(curves, evaluation_points, save_path=save_path)
