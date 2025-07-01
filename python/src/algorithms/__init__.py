"""
Algorithms package for nonparametric Shapley value estimation.

This package contains the core algorithms for computing nonparametric Shapley curves,
including both basic estimation and advanced integration methods.
"""

# Core Shapley estimators
from .shapley_estimator import ShapleyEstimator
from .population_shapley import PopulationShapleyEstimator  

# Bootstrap and statistical procedures
from .bootstrap_procedures import AdvancedWildBootstrap

# Integration methods
from .integration_methods import ConditionalDensityIntegrator, CubatureStyleIntegrator, AdvancedShapleyIntegrator
from .conditional_densities import ConditionalDensity, setup_conditional_densities
from .advanced_integration import CubatureIntegrator, PopulationShapleyIntegrator, create_population_integrator

# Regression and analysis
from .nonparametric_regression import LocalLinearRegressor
from .squared_error_analysis import SquaredErrorAnalyzer
from .se_vector_functions import (
    SquaredErrorVectorCalculator, 
    SE_vec, 
    SE_vec_int, 
    compute_ise_results
)

# Model testing and validation
from .model_consistency_test import ModelConsistencyTester

__all__ = [
    # Core Shapley estimators
    'ShapleyEstimator',  # Uses statsmodels.KernelReg (R npreg equivalent)
    'PopulationShapleyEstimator', 
    
    # Bootstrap procedures
    'AdvancedWildBootstrap',
    
    # Integration methods
    'ConditionalDensityIntegrator',
    'CubatureStyleIntegrator', 
    'AdvancedShapleyIntegrator',
    'ConditionalDensity',
    'setup_conditional_densities',
    'CubatureIntegrator',
    'PopulationShapleyIntegrator',
    'create_population_integrator',
    
    # Regression and analysis
    'LocalLinearRegressor',
    'SquaredErrorAnalyzer',
    'SquaredErrorVectorCalculator',
    'SE_vec',
    'SE_vec_int', 
    'compute_ise_results',
    
    # Model testing
    'ModelConsistencyTester'
] 