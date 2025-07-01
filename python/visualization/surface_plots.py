"""
3D surface plots for Shapley curve visualization.

This module provides sophisticated 3D surface plotting capabilities matching
R's surface_plots.R functionality with enhanced interactivity using plotly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings


class SurfacePlotter:
    """
    3D surface plotting for Shapley curve analysis.
    
    This class implements sophisticated 3D surface visualization matching
    R's plotly surface plots with enhanced Python capabilities.
    """
    
    def __init__(self, 
                 backend: str = 'plotly',
                 figsize: Tuple[int, int] = (800, 600),
                 theme: str = 'plotly_white'):
        """
        Initialize surface plotter.
        
        Parameters
        ----------
        backend : str, default='plotly'
            Plotting backend ('plotly' or 'matplotlib')
        figsize : Tuple[int, int]
            Figure size (width, height)
        theme : str
            Plotly theme for styling
        """
        self.backend = backend
        self.figsize = figsize
        self.theme = theme
    
    def create_evaluation_grid(self, 
                             bounds: Dict[str, Tuple[float, float]],
                             resolution: int = 30) -> Dict[str, np.ndarray]:
        """
        Create evaluation grid for surface plotting.
        
        Parameters
        ----------
        bounds : Dict[str, Tuple[float, float]]
            Variable bounds for grid creation
        resolution : int
            Grid resolution (points per dimension)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Grid arrays for each variable
        """
        grids = {}
        for var, (low, high) in bounds.items():
            grids[var] = np.linspace(low, high, resolution)
        
        return grids
    
    def plot_surface_comparison(self, 
                              true_surface: np.ndarray,
                              estimated_surface: np.ndarray,
                              x_grid: np.ndarray,
                              y_grid: np.ndarray,
                              variable_names: Tuple[str, str] = ('x1', 'x2'),
                              shapley_variable: str = 'j',
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Plot comparison between true and estimated Shapley surfaces.
        
        Matches R's surface comparison plots with enhanced interactivity.
        """
        if self.backend == 'plotly':
            return self._plot_surface_comparison_plotly(
                true_surface, estimated_surface, x_grid, y_grid,
                variable_names, shapley_variable, save_path
            )
        else:
            return self._plot_surface_comparison_matplotlib(
                true_surface, estimated_surface, x_grid, y_grid,
                variable_names, shapley_variable, save_path
            )
    
    def _plot_surface_comparison_plotly(self, 
                                      true_surface: np.ndarray,
                                      estimated_surface: np.ndarray,
                                      x_grid: np.ndarray,
                                      y_grid: np.ndarray,
                                      variable_names: Tuple[str, str],
                                      shapley_variable: str,
                                      save_path: Optional[str]) -> go.Figure:
        """Create plotly surface comparison plot matching R implementation."""
        
        # Create figure with custom layout
        fig = go.Figure()
        
        # Add true surface (blue, matching R rgb(0,3,160))
        fig.add_trace(go.Surface(
            x=x_grid,
            y=y_grid,
            z=true_surface,
            name='True Shapley Surface',
            colorscale=[[0, 'rgb(0,3,160)'], [1, 'rgb(0,3,160)']],
            opacity=1.0,
            showscale=False
        ))
        
        # Add estimated surface (pink/red, matching R rgb(255,107,184) to rgb(128,0,64))
        fig.add_trace(go.Surface(
            x=x_grid,
            y=y_grid,
            z=estimated_surface,
            name='Estimated Shapley Surface',
            colorscale=[[0, 'rgb(255,107,184)'], [1, 'rgb(128,0,64)']],
            opacity=0.3,
            showscale=False
        ))
        
        # Update layout to match R camera settings
        fig.update_layout(
            title=f'Shapley Surface Comparison: Variable {shapley_variable}',
            scene=dict(
                xaxis_title=variable_names[0],
                yaxis_title=variable_names[1],
                zaxis_title=f'Shapley Value ({shapley_variable})',
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.75)  # Matching R camera settings
                )
            ),
            width=self.figsize[0],
            height=self.figsize[1],
            template=self.theme,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_squared_error_surface(self, 
                                 squared_error: np.ndarray,
                                 x_grid: np.ndarray,
                                 y_grid: np.ndarray,
                                 variable_names: Tuple[str, str] = ('x1', 'x2'),
                                 shapley_variable: str = 'j',
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Plot squared error surface.
        
        Matches R's SE surface plots.
        """
        fig = go.Figure()
        
        # Add squared error surface
        fig.add_trace(go.Surface(
            x=x_grid,
            y=y_grid,
            z=squared_error,
            name='Squared Error',
            colorscale=[[0, 'rgb(0,3,160)'], [1, 'rgb(0,3,160)']],
            opacity=1.0,
            showscale=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Squared Error Surface: Variable {shapley_variable}',
            scene=dict(
                xaxis_title=variable_names[0],
                yaxis_title=variable_names[1],
                zaxis_title='Squared Error',
                zaxis=dict(range=[0, 1]),  # Matching R range
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.75)
                )
            ),
            width=self.figsize[0],
            height=self.figsize[1],
            template=self.theme
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_comprehensive_surface_analysis(self, 
                                            shapley_results: Dict[str, Any],
                                            variable_bounds: Dict[str, Tuple[float, float]],
                                            resolution: int = 30,
                                            save_path: Optional[str] = None) -> Dict[str, go.Figure]:
        """
        Create comprehensive surface analysis with multiple variables.
        
        This creates the full set of surface plots matching R's comprehensive analysis.
        """
        # Create evaluation grids
        grids = self.create_evaluation_grid(variable_bounds, resolution)
        variable_names = list(variable_bounds.keys())
        
        if len(variable_names) < 2:
            raise ValueError("Need at least 2 variables for surface plotting")
        
        # Create meshgrids for the first two variables
        x_grid, y_grid = np.meshgrid(grids[variable_names[0]], grids[variable_names[1]])
        
        figures = {}
        
        # Create surface plots for each Shapley variable
        for shap_var in shapley_results.get('variables', []):
            if f'true_surface_{shap_var}' in shapley_results and f'estimated_surface_{shap_var}' in shapley_results:
                # Surface comparison
                fig_comp = self.plot_surface_comparison(
                    shapley_results[f'true_surface_{shap_var}'],
                    shapley_results[f'estimated_surface_{shap_var}'],
                    x_grid, y_grid,
                    (variable_names[0], variable_names[1]),
                    shap_var
                )
                figures[f'comparison_{shap_var}'] = fig_comp
                
                # Squared error surface if available
                if f'squared_error_{shap_var}' in shapley_results:
                    fig_se = self.plot_squared_error_surface(
                        shapley_results[f'squared_error_{shap_var}'],
                        x_grid, y_grid,
                        (variable_names[0], variable_names[1]),
                        shap_var
                    )
                    figures[f'squared_error_{shap_var}'] = fig_se
        
        # Save all figures if path provided
        if save_path:
            for name, fig in figures.items():
                file_path = save_path.replace('.html', f'_{name}.html')
                fig.write_html(file_path)
        
        return figures
    
    def _plot_surface_comparison_matplotlib(self, 
                                          true_surface: np.ndarray,
                                          estimated_surface: np.ndarray,
                                          x_grid: np.ndarray,
                                          y_grid: np.ndarray,
                                          variable_names: Tuple[str, str],
                                          shapley_variable: str,
                                          save_path: Optional[str]) -> plt.Figure:
        """Matplotlib fallback for surface plotting."""
        fig = plt.figure(figsize=(12, 5))
        
        # True surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(x_grid, y_grid, true_surface, 
                               cmap='Blues', alpha=0.8)
        ax1.set_title(f'True Shapley Surface ({shapley_variable})')
        ax1.set_xlabel(variable_names[0])
        ax1.set_ylabel(variable_names[1])
        ax1.set_zlabel('Shapley Value')
        
        # Estimated surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(x_grid, y_grid, estimated_surface, 
                               cmap='Reds', alpha=0.8)
        ax2.set_title(f'Estimated Shapley Surface ({shapley_variable})')
        ax2.set_xlabel(variable_names[0])
        ax2.set_ylabel(variable_names[1])
        ax2.set_zlabel('Shapley Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_shapley_surfaces(shapley_results: Dict[str, Any],
                         variable_bounds: Dict[str, Tuple[float, float]],
                         backend: str = 'plotly',
                         save_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick function for comprehensive surface plotting."""
    plotter = SurfacePlotter(backend=backend)
    return plotter.create_comprehensive_surface_analysis(
        shapley_results, variable_bounds, save_path=save_path
    )


class InteractiveSurfaceDashboard:
    """
    Interactive dashboard for Shapley surface exploration.
    
    This creates an interactive dashboard for exploring Shapley surfaces
    with parameter controls and real-time updates.
    """
    
    def __init__(self):
        """Initialize interactive dashboard."""
        pass
    
    def create_dashboard(self, 
                        shapley_results: Dict[str, Any],
                        variable_bounds: Dict[str, Tuple[float, float]]) -> go.Figure:
        """
        Create interactive dashboard with multiple surface plots.
        
        This creates a comprehensive dashboard matching R's interactive capabilities.
        """
        # Create subplots for multiple surfaces
        n_vars = len(shapley_results.get('variables', []))
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        subplot_titles = [f'Variable {var}' for var in shapley_results.get('variables', [])]
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'type': 'surface'} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
        
        # Add surfaces for each variable
        variable_names = list(variable_bounds.keys())
        if len(variable_names) >= 2:
            grids = {}
            for var, (low, high) in variable_bounds.items():
                grids[var] = np.linspace(low, high, 30)
            
            x_grid, y_grid = np.meshgrid(grids[variable_names[0]], grids[variable_names[1]])
            
            for i, var in enumerate(shapley_results.get('variables', [])):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                if f'true_surface_{var}' in shapley_results:
                    fig.add_trace(
                        go.Surface(
                            x=x_grid, y=y_grid,
                            z=shapley_results[f'true_surface_{var}'],
                            colorscale='Blues',
                            showscale=False,
                            name=f'True {var}'
                        ),
                        row=row, col=col
                    )
                
                if f'estimated_surface_{var}' in shapley_results:
                    fig.add_trace(
                        go.Surface(
                            x=x_grid, y=y_grid,
                            z=shapley_results[f'estimated_surface_{var}'],
                            colorscale='Reds',
                            opacity=0.7,
                            showscale=False,
                            name=f'Estimated {var}'
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title='Interactive Shapley Surface Dashboard',
            height=600 * n_rows,
            template='plotly_white'
        )
        
        return fig 