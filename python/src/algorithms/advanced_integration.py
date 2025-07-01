"""
Advanced integration methods for Shapley value computation.

This module implements multi-dimensional integration methods that match the R cubature package,
specifically the methods used in integral_estimation.R and integral_population.R.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
from scipy import integrate
from scipy.stats import multivariate_normal
import warnings

from .conditional_densities import ConditionalDensity, setup_conditional_densities, create_default_covariance


class CubatureIntegrator:
    """
    Multi-dimensional integration using cubature-style methods.
    
    This class implements integration methods that match the R cubature package,
    specifically the 'cuhre' method used extensively in the R implementation.
    """
    
    def __init__(self, 
                 rel_tol: float = 3e-1,
                 abs_tol: float = 1e-6,
                 max_eval: int = 50000):
        """
        Initialize the integrator.
        
        Parameters
        ----------
        rel_tol : float, default=3e-1
            Relative tolerance (matches R relTol=3e-1)
        abs_tol : float, default=1e-6
            Absolute tolerance
        max_eval : int, default=50000
            Maximum number of function evaluations
        """
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.max_eval = max_eval
    
    def integrate_1d(self, func: Callable, lower: float, upper: float, **kwargs) -> Dict[str, Any]:
        """
        1D integration matching R cubintegrate for 1D case.
        
        Parameters
        ----------
        func : Callable
            Function to integrate
        lower : float
            Lower bound
        upper : float
            Upper bound
        **kwargs
            Additional arguments passed to func
            
        Returns
        -------
        Dict[str, Any]
            Integration result with 'integral' key
        """
        def wrapper(x):
            return func(x, **kwargs)
        
        try:
            result, error = integrate.quad(
                wrapper, lower, upper,
                epsabs=self.abs_tol,
                epsrel=self.rel_tol,
                limit=self.max_eval
            )
            return {'integral': result, 'error': error}
        except Exception as e:
            warnings.warn(f"Integration failed: {e}")
            return {'integral': 0.0, 'error': np.inf}
    
    def integrate_2d(self, func: Callable, lower: Tuple[float, float], 
                     upper: Tuple[float, float], **kwargs) -> Dict[str, Any]:
        """
        2D integration matching R cubintegrate for 2D case.
        
        Parameters
        ----------
        func : Callable
            Function to integrate, should accept x_out array of shape (2, n)
        lower : Tuple[float, float]
            Lower bounds
        upper : Tuple[float, float]
            Upper bounds
        **kwargs
            Additional arguments passed to func
            
        Returns
        -------
        Dict[str, Any]
            Integration result with 'integral' key
        """
        def wrapper(x1, x2):
            x_out = np.array([[x1], [x2]])
            result = func(x_out, **kwargs)
            if hasattr(result, '__len__') and len(result) == 1:
                return float(result[0])
            return float(result)
        
        try:
            result, error = integrate.dblquad(
                wrapper,
                lower[0], upper[0],  # x1 bounds
                lower[1], upper[1],  # x2 bounds
                epsabs=self.abs_tol,
                epsrel=self.rel_tol
            )
            return {'integral': result, 'error': error}
        except Exception as e:
            warnings.warn(f"2D integration failed: {e}")
            return {'integral': 0.0, 'error': np.inf}
    
    def integrate_3d(self, func: Callable, lower: Tuple[float, float, float],
                     upper: Tuple[float, float, float], **kwargs) -> Dict[str, Any]:
        """
        3D integration using scipy.integrate.tplquad.
        
        Parameters
        ----------
        func : Callable
            Function to integrate
        lower : Tuple[float, float, float]
            Lower bounds
        upper : Tuple[float, float, float]
            Upper bounds
        **kwargs
            Additional arguments passed to func
            
        Returns
        -------
        Dict[str, Any]
            Integration result with 'integral' key
        """
        def wrapper(x3, x2, x1):
            return func(x1, x2, x3, **kwargs)
        
        try:
            result, error = integrate.tplquad(
                wrapper,
                lower[0], upper[0],  # x1 bounds
                lower[1], upper[1],  # x2 bounds
                lower[2], upper[2],  # x3 bounds
                epsabs=self.abs_tol,
                epsrel=self.rel_tol
            )
            return {'integral': result, 'error': error}
        except Exception as e:
            warnings.warn(f"3D integration failed: {e}")
            return {'integral': 0.0, 'error': np.inf}


class PopulationShapleyIntegrator:
    """
    Population Shapley integration methods matching integral_population.R.
    
    This class implements the specific integration functions used for population
    Shapley value computation with conditional densities.
    """
    
    def __init__(self, sigma_sim: Optional[np.ndarray] = None, 
                 integration_method: str = 'cuhre',
                 rel_tol: float = 3e-1,
                 m_full_model: Optional[Callable] = None):
        """
        Initialize the population Shapley integrator.
        
        Parameters
        ----------
        sigma_sim : np.ndarray, optional
            Covariance matrix for conditional densities
        integration_method : str, default='cuhre'
            Numerical integration method to use
        rel_tol : float, default=3e-1
            Relative tolerance for integration
        m_full_model : callable, optional
            Fitted model for the full feature set (m_full_hat)
        """
        # Set up conditional density functions
        if sigma_sim is None:
            sigma_sim = create_default_covariance()
        
        # A single density object handles all cases
        self.cond_density = ConditionalDensity(sigma_sim)
        
        self.integration_method = integration_method
        self.rel_tol = rel_tol
        self.m_full_model = m_full_model
        
        # Integration bounds matching R code
        self.integration_bounds = (-5.0, 5.0)
    
    def m_full_function(self, x1: float, x2: float, x3: float) -> float:
        """
        True function for population Shapley (matches R m_full).
        
        This implements the function:
        m(x) = -sin(2*x1) + cos(3*x2) + 2*cos(x1)*sin(2*x2) + 0.5*x3
        """
        return (-np.sin(2*x1) + np.cos(3*x2) + 
                2*np.cos(x1)*np.sin(2*x2) + 0.5*x3)
    
    def m_full_hat(self, x1: float, x2: float, x3: float) -> float:
        """
        Prediction from the full fitted model. Matches R predict(model_list[[7]], ...).
        
        Parameters
        ----------
        x1, x2, x3 : float
            Evaluation point
            
        Returns
        -------
        float
            Model prediction
        """
        if self.m_full_model is None:
            raise ValueError("Full model (m_full_model) must be provided for this calculation.")
        
        x_eval = np.array([x1, x2, x3]).reshape(1, -1)
        return self.m_full_model.predict(x_eval)[0]
    
    def _m_x3_temp_int(self, x1: np.ndarray, x2: np.ndarray, x3: float) -> np.ndarray:
        """Integrand for m_x3_est, vectorized for scipy."""
        # This matches R: m_full_hat * norm2_vec(dep=x_out, cond=x3)
        m_full_predictions = self.m_full_hat(x1, x2, x3)
        x_out = np.vstack([x1, x2])
        norm_densities = self.cond_density.norm2_vec(x_out, x3)
        return m_full_predictions * norm_densities
    
    def m_x3_est(self, x3: float) -> float:
        """Matches R: m_x3_est function."""
        lower, upper = [-5, -5], [5, 5]
        
        # Use dblquad for 2D integration
        result, _ = integrate.dblquad(self._m_x3_temp_int, lower[0], upper[0], lambda x: lower[1], lambda x: upper[1],
                                      args=(x3,), epsrel=self.rel_tol)
        return result
    
    def _m_x2_temp_int(self, x1: np.ndarray, x3: np.ndarray, x2: float) -> np.ndarray:
        """Integrand for m_x2_est."""
        m_full_predictions = self.m_full_hat(x1, x2, x3)
        x_out = np.vstack([x1, x3])
        norm_densities = self.cond_density.norm2_vec(x_out, x2)
        return m_full_predictions * norm_densities
    
    def m_x2_est(self, x2: float) -> float:
        """Matches R: m_x2_est function."""
        lower, upper = [-5, -5], [5, 5]
        result, _ = integrate.dblquad(self._m_x2_temp_int, lower[0], upper[0], lambda x: lower[1], lambda x: upper[1],
                                      args=(x2,), epsrel=self.rel_tol)
        return result
    
    def _m_x1_temp_int(self, x2: np.ndarray, x3: np.ndarray, x1: float) -> np.ndarray:
        """Integrand for m_x1_est."""
        m_full_predictions = self.m_full_hat(x1, x2, x3)
        x_out = np.vstack([x2, x3])
        norm_densities = self.cond_density.norm2_vec(x_out, x1)
        return m_full_predictions * norm_densities
    
    def m_x1_est(self, x1: float) -> float:
        """Matches R: m_x1_est function."""
        lower, upper = [-5, -5], [5, 5]
        result, _ = integrate.dblquad(self._m_x1_temp_int, lower[0], upper[0], lambda x: lower[1], lambda x: upper[1],
                                      args=(x1,), epsrel=self.rel_tol)
        return result
    
    def _m_x1_x3_temp_int(self, x2: float, x1: float, x3: float) -> float:
        """Integrand for m_x1_x3_est."""
        m_full_prediction = self.m_full_hat(x1, x2, x3)
        norm_density = self.cond_density.norm1(x2, np.array([x1, x3]))
        return m_full_prediction * norm_density
    
    def m_x1_x3_est(self, x1: float, x3: float) -> float:
        """Matches R: m_x1_x3_est function."""
        lower, upper = -5, 5 # Simplified bounds
        result, _ = integrate.quad(self._m_x1_x3_temp_int, lower, upper,
                                   args=(x1, x3), epsrel=self.rel_tol)
        return result
    
    def _m_x2_x3_temp_int(self, x1: float, x2: float, x3: float) -> float:
        """Integrand for m_x2_x3_est."""
        m_full_prediction = self.m_full_hat(x1, x2, x3)
        norm_density = self.cond_density.norm1(x1, np.array([x2, x3]))
        return m_full_prediction * norm_density
    
    def m_x2_x3_est(self, x2: float, x3: float) -> float:
        """Matches R: m_x2_x3_est function."""
        lower, upper = -5, 5
        result, _ = integrate.quad(self._m_x2_x3_temp_int, lower, upper,
                                   args=(x2, x3), epsrel=self.rel_tol)
        return result
    
    def _m_x1_x2_temp_int(self, x3: float, x1: float, x2: float) -> float:
        """Integrand for m_x1_x2_est."""
        m_full_prediction = self.m_full_hat(x1, x2, x3)
        norm_density = self.cond_density.norm1(x3, np.array([x1, x2]))
        return m_full_prediction * norm_density
    
    def m_x1_x2_est(self, x1: float, x2: float) -> float:
        """Matches R: m_x1_x2_est function."""
        lower, upper = -5, 5
        result, _ = integrate.quad(self._m_x1_x2_temp_int, lower, upper,
                                   args=(x1, x2), epsrel=self.rel_tol)
        return result


# Convenience function for backward compatibility
def create_population_integrator(sigma_sim: Optional[np.ndarray] = None) -> PopulationShapleyIntegrator:
    """Create a population Shapley integrator with default settings."""
    return PopulationShapleyIntegrator(sigma_sim=sigma_sim) 