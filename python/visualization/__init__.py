"""
Comprehensive visualization package for Shapley curve analysis.

This package provides sophisticated visualization capabilities matching and
enhancing the R visualization functionality.
"""

from .shapley_visualizations import ShapleyVisualizer, quick_shapley_plot
from .application_plots import ApplicationPlotter, plot_application_analysis
from .surface_plots import SurfacePlotter, plot_shapley_surfaces, InteractiveSurfaceDashboard
from .sequence_plots import SequencePlotter, create_sequence_plots

__all__ = [
    'ShapleyVisualizer',
    'quick_shapley_plot',
    'ApplicationPlotter', 
    'plot_application_analysis',
    'SurfacePlotter',
    'plot_shapley_surfaces',
    'InteractiveSurfaceDashboard',
    'SequencePlotter',
    'create_sequence_plots'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Nonparametric Shapley Curves Team'
