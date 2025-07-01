"""
Sequential plots for Shapley curve analysis.

This module provides sequential plotting capabilities matching R's seq_plots.R
functionality with filled areas between curves for sequential analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings


class SequencePlotter:
    """
    Sequential plotting for Shapley curve analysis.
    
    This class implements the sophisticated sequential plotting from R's
    seq_plots.R with filled areas between curves.
    """
    
    def __init__(self, 
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 150):
        """Initialize sequence plotter."""
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set plotting style
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('default')
            warnings.warn(f"Style '{self.style}' not found, using default")
    
    def find_intersections(self, x1: np.ndarray, x2: np.ndarray, x_points: np.ndarray) -> pd.DataFrame:
        """
        Find intersection points between two curves.
        
        Matches R intersects() function exactly.
        """
        # Find segments where curves cross
        diff = x1 - x2
        sign_changes = np.diff(np.sign(diff))
        seg1 = np.where(sign_changes != 0)[0]  # First point in crossing segments
        
        if len(seg1) == 0:
            return pd.DataFrame(columns=['x', 'y', 'pindex', 'pabove'])
        
        # Determine which curve is above prior to crossing
        above = x2[seg1] > x1[seg1]
        
        # Calculate intersection points using linear interpolation
        slope1 = x1[seg1 + 1] - x1[seg1]
        slope2 = x2[seg1 + 1] - x2[seg1]
        
        # Avoid division by zero
        slope_diff = slope1 - slope2
        valid_mask = np.abs(slope_diff) > 1e-10
        
        if not np.any(valid_mask):
            return pd.DataFrame(columns=['x', 'y', 'pindex', 'pabove'])
        
        seg1 = seg1[valid_mask]
        above = above[valid_mask]
        slope1 = slope1[valid_mask]
        slope2 = slope2[valid_mask]
        slope_diff = slope_diff[valid_mask]
        
        # Calculate intersection coordinates
        x_intersect = seg1 + ((x2[seg1] - x1[seg1]) / slope_diff)
        y_intersect = x1[seg1] + slope1 * (x_intersect - seg1)
        
        # Convert to actual x-coordinates
        x_actual = np.interp(x_intersect, np.arange(len(x_points)), x_points)
        
        # pabove: which curve is above prior to crossing (1 or 2)
        pabove = np.where(above, 2, 1)
        
        return pd.DataFrame({
            'x': x_actual,
            'y': y_intersect,
            'pindex': x_intersect,
            'pabove': pabove
        })
    
    def create_filled_curve_plot(self, 
                               curve1: np.ndarray,
                               curve2: np.ndarray,
                               x_points: np.ndarray,
                               variable_name: str,
                               variable_units: str = '',
                               curve1_label: str = 'Shapley Curve',
                               curve2_label: str = 'Mean Prediction',
                               y_limits: Optional[Tuple[float, float]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create filled curve plot matching R fillColor() function.
        
        This creates the sophisticated filled area plots with intersection handling.
        """
        # Prepare data
        data = pd.DataFrame({
            'x': x_points,
            'curve1': curve1,
            'curve2': curve2
        })
        
        # Find intersections
        intersections = self.find_intersections(curve1, curve2, x_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Set y-limits
        if y_limits:
            ax.set_ylim(y_limits)
        else:
            y_min = min(np.min(curve1), np.min(curve2))
            y_max = max(np.max(curve1), np.max(curve2))
            margin = 0.1 * (y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
        
        # Fill areas between curves
        self._fill_between_curves(ax, data, intersections, x_points)
        
        # Plot the curves on top
        ax.plot(x_points, curve1, color='blue', linewidth=3, label=curve1_label, zorder=10)
        ax.plot(x_points, curve2, color='black', linewidth=3, label=curve2_label, zorder=10)
        
        # Formatting
        ax.set_xlabel(f'{variable_name} {variable_units}', fontsize=15, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        
        # Custom legend
        ax.legend(fontsize=12, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _fill_between_curves(self, ax: plt.Axes, data: pd.DataFrame, 
                           intersections: pd.DataFrame, x_points: np.ndarray):
        """Fill areas between curves handling intersections properly."""
        if len(intersections) == 0:
            # No intersections - simple fill
            mask = data['curve1'] >= data['curve2']
            ax.fill_between(x_points, data['curve1'], data['curve2'], 
                          where=mask, color='lightcoral', alpha=0.7, interpolate=True)
            ax.fill_between(x_points, data['curve1'], data['curve2'], 
                          where=~mask, color='lightgreen', alpha=0.7, interpolate=True)
            return
        
        # Handle intersections
        intersections = intersections.sort_values('pindex').reset_index(drop=True)
        
        # Create intervals based on intersections
        intervals = []
        start_idx = 0
        
        for _, intersection in intersections.iterrows():
            end_idx = int(np.round(intersection['pindex']))
            if end_idx > start_idx:
                intervals.append((start_idx, end_idx, intersection['pabove']))
            start_idx = end_idx
        
        # Add final interval
        if start_idx < len(data):
            last_above = data['curve2'].iloc[-1] > data['curve1'].iloc[-1]
            intervals.append((start_idx, len(data) - 1, 2 if last_above else 1))
        
        # Fill each interval
        for start_idx, end_idx, above_curve in intervals:
            x_segment = x_points[start_idx:end_idx + 1]
            y1_segment = data['curve1'].iloc[start_idx:end_idx + 1]
            y2_segment = data['curve2'].iloc[start_idx:end_idx + 1]
            
            color = 'lightcoral' if above_curve == 1 else 'lightgreen'
            ax.fill_between(x_segment, y1_segment, y2_segment, 
                          color=color, alpha=0.7)
    
    def create_sequential_analysis_plot(self, 
                                      shapley_curves: Dict[str, np.ndarray],
                                      evaluation_points: Dict[str, np.ndarray],
                                      mean_predictions: Dict[str, np.ndarray],
                                      variable_info: Dict[str, Dict[str, str]],
                                      save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create sequential analysis plots for all variables.
        
        This matches R's comprehensive sequential plotting.
        """
        figures = {}
        
        for var in shapley_curves.keys():
            if var in evaluation_points and var in mean_predictions:
                # Get variable information
                var_info = variable_info.get(var, {})
                units = var_info.get('units', '')
                
                # Adjust Shapley curve (add mean prediction to match R)
                adjusted_shapley = shapley_curves[var] + mean_predictions[var]
                
                # Create figure
                fig = self.create_filled_curve_plot(
                    curve1=adjusted_shapley,
                    curve2=mean_predictions[var],
                    x_points=evaluation_points[var],
                    variable_name=var,
                    variable_units=units,
                    curve1_label='Shapley + Mean',
                    curve2_label='Mean Prediction',
                    y_limits=var_info.get('y_limits'),
                    save_path=save_path.replace('.png', f'_{var}.png') if save_path else None
                )
                
                figures[var] = fig
        
        return figures
    
    def create_custom_axis_plot(self, 
                              curve1: np.ndarray,
                              curve2: np.ndarray,
                              x_values: np.ndarray,
                              custom_ticks: List[float],
                              custom_labels: List[str],
                              variable_name: str,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create plot with custom axis scaling matching R's custom axis functionality.
        
        This handles the complex axis transformations used in R seq_plots.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create the main plot without axis labels
        ax.plot(range(len(curve1)), curve1, color='blue', linewidth=3, label='Shapley + Mean')
        ax.plot(range(len(curve2)), curve2, color='black', linewidth=3, label='Mean Prediction')
        
        # Fill between curves
        data = pd.DataFrame({'curve1': curve1, 'curve2': curve2})
        intersections = self.find_intersections(curve1, curve2, np.arange(len(curve1)))
        self._fill_between_curves(ax, data, intersections, np.arange(len(curve1)))
        
        # Set custom ticks and labels
        tick_positions = []
        for tick_val in custom_ticks:
            # Find closest position in x_values
            closest_idx = np.argmin(np.abs(x_values - tick_val))
            tick_positions.append(closest_idx)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(custom_labels, fontsize=12)
        
        # Formatting
        ax.set_xlabel(f'{variable_name}', fontsize=15, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set y-limits with some padding
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, max(y_max, 80))  # Matching R ylim=c(0,80)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_sequence_plots(shapley_results: Dict[str, Any],
                         variable_info: Dict[str, Dict[str, str]],
                         save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
    """Quick function for creating sequential analysis plots."""
    plotter = SequencePlotter()
    return plotter.create_sequential_analysis_plot(
        shapley_results['curves'],
        shapley_results['evaluation_points'],
        shapley_results['mean_predictions'],
        variable_info,
        save_path
    ) 