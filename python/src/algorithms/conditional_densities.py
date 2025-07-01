"""
Conditional density functions for Shapley integration.

This module implements the conditional normal density functions used in the R implementation
for population Shapley value calculations. These functions are critical for the integration
methods used in integral_population.R and integral_estimation.R.
"""

import numpy as np
from typing import Union, Tuple
from scipy.linalg import det


class ConditionalDensity:
    """
    Conditional density calculator for multivariate normal distributions.
    
    This class implements the conditional density functions from the R code:
    - norm1: 1D conditional density (e.g., x1 | x2,x3)
    - norm2: 2D conditional density (e.g., x1,x2 | x3)
    - norm1_vec: Vectorized version of norm1
    - norm2_vec: Vectorized version of norm2
    """
    
    def __init__(self, sigma_sim: np.ndarray):
        """
        Initialize conditional density calculator.
        
        Parameters
        ----------
        sigma_sim : np.ndarray
            3x3 covariance matrix for (X1, X2, X3)
        """
        self.sigma_sim = sigma_sim
        self._precompute_parameters()
    
    def _precompute_parameters(self):
        """Precompute parameters for conditional densities to match R implementation."""
        
        # For x1/x2,x3 and x2/x1,x3 and x3/x1,x2 (1D conditional on 2D)
        self.sigma_xy_1 = self.sigma_sim[0, 1:3].reshape(1, -1)  # t(sigma_sim[1, 2:3])
        self.sigma_yy_1 = self.sigma_sim[1:3, 1:3]  # sigma_sim[2:3, 2:3]
        self.sigma_yx_1 = self.sigma_xy_1.T  # t(sigma_xy_1)
        self.sigma_xx_1 = self.sigma_sim[0, 0]  # sigma_sim[1,1]
        
        # Conditional variance
        sigma_yy_1_inv = np.linalg.inv(self.sigma_yy_1)
        self.inv_yy_1 = self.sigma_xy_1 @ sigma_yy_1_inv
        self.c_var_1 = self.sigma_xx_1 - self.sigma_xy_1 @ sigma_yy_1_inv @ self.sigma_yx_1
        self.c_var_1 = float(self.c_var_1.item())  # Ensure scalar from array
        
        self.pre_mult = 1.0 / np.sqrt(self.c_var_1 * 2 * np.pi)
        self.sq = np.sqrt(self.c_var_1)
        
        # For x1,x2/x3 and x2,x3/x1 and x1,x3/x2 (2D conditional on 1D)
        self.sigma_xy_2 = self.sigma_sim[0:2, 2].reshape(-1, 1)  # matrix(sigma_sim[1:2, 3])
        self.sigma_yy_2 = self.sigma_sim[2, 2]  # sigma_sim[3, 3]
        self.sigma_yx_2 = self.sigma_xy_2.T  # t(sigma_xy_2)
        self.sigma_xx_2 = self.sigma_sim[0:2, 0:2]  # sigma_sim[1:2, 1:2]
        
        # Conditional variance  
        self.inv_yy_2 = self.sigma_xy_2 * (1.0 / self.sigma_yy_2)  # Element-wise multiplication
        self.c_var_2 = self.sigma_xx_2 - self.sigma_xy_2 * (1.0 / self.sigma_yy_2) @ self.sigma_yx_2
        
        self.pre_mult2 = (det(2 * np.pi * self.sigma_xx_2)) ** (-0.5)
        self.inv = np.linalg.inv(self.sigma_xx_2)
    
    def norm1(self, dep: float, cond: np.ndarray) -> float:
        """
        1D conditional normal density.
        
        Parameters
        ----------
        dep : float
            Dependent variable (scalar)
        cond : np.ndarray
            Conditioning variables (2D vector)
            
        Returns
        -------
        float
            Conditional density value
        """
        x = dep  # dependent variable, scalar
        Y_1 = cond  # what you condition on, 2D vector
        c_mu_1 = self.inv_yy_1 @ Y_1
        c_mu_1 = float(c_mu_1.item() if hasattr(c_mu_1, 'item') else c_mu_1)  # Ensure scalar
        
        # Conditional normal density
        return self.pre_mult * np.exp(-0.5 * ((x - c_mu_1) / self.sq) ** 2)
    
    def norm2(self, dep: np.ndarray, cond: float) -> float:
        """
        2D conditional normal density.
        
        Parameters
        ----------
        dep : np.ndarray
            Dependent variables (2D vector)
        cond : float
            Conditioning variable (scalar)
            
        Returns
        -------
        float
            Conditional density value
        """
        x = dep  # dependent variable, 2D vector
        Y_2 = cond  # what you condition on, scalar
        c_mu_2 = self.inv_yy_2 * Y_2  # Element-wise multiplication with scalar
        c_mu_2 = c_mu_2.flatten()  # Ensure 1D array
        
        # Conditional multivariate normal density
        diff = x - c_mu_2
        return self.pre_mult2 * np.exp(-0.5 * diff.T @ self.inv @ diff)
    
    def norm1_vec(self, dep: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """
        Vectorized 1D conditional normal density.
        
        Parameters
        ----------
        dep : np.ndarray
            Dependent variable values (1D array)
        cond : np.ndarray
            Conditioning variables (2D vector, same for all dep values)
            
        Returns
        -------
        np.ndarray
            Array of conditional density values
        """
        x = dep  # dependent variable, array
        Y_1 = cond  # what you condition on, 2D vector
        c_mu_1 = self.inv_yy_1 @ Y_1
        c_mu_1 = float(c_mu_1.item() if hasattr(c_mu_1, 'item') else c_mu_1)  # Ensure scalar
        
        # Vectorized conditional normal density
        return np.array([self.pre_mult * np.exp(-0.5 * ((x_val - c_mu_1) / self.sq) ** 2) 
                        for x_val in x])
    
    def norm2_vec(self, dep: np.ndarray, cond: float) -> np.ndarray:
        """
        Vectorized 2D conditional normal density.
        
        Parameters
        ----------
        dep : np.ndarray
            Dependent variables (2D array, shape (2, n))
        cond : float
            Conditioning variable (scalar)
            
        Returns
        -------
        np.ndarray
            Array of conditional density values
        """
        x = dep  # dependent variable, 2D array
        Y_2 = cond  # what you condition on, scalar
        c_mu_2 = self.inv_yy_2 * Y_2  # Element-wise multiplication with scalar
        c_mu_2 = c_mu_2.flatten()  # Ensure 1D array
        
        n_points = x.shape[1]
        results = np.zeros(n_points)
        
        for a in range(n_points):
            diff = x[:, a] - c_mu_2
            results[a] = self.pre_mult2 * np.exp(-0.5 * diff.T @ self.inv @ diff)
        
        return results


def create_default_covariance(d: int = 3, cova: float = 0.0) -> np.ndarray:
    """
    Create default covariance matrix matching R implementation.
    
    Parameters
    ----------
    d : int, default=3
        Dimension of covariance matrix
    cova : float, default=0.0
        Off-diagonal covariance value
        
    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    sigma = np.eye(d) * 4.0  # Diagonal elements = 4
    sigma[sigma == 0] = cova  # Off-diagonal elements = cova
    return sigma


# Convenience functions to match R interface
def setup_conditional_densities(sigma_sim: np.ndarray = None) -> ConditionalDensity:
    """
    Set up conditional density functions with given covariance matrix.
    
    Parameters
    ----------
    sigma_sim : np.ndarray, optional
        3x3 covariance matrix. If None, uses default from R code.
        
    Returns
    -------
    ConditionalDensity
        Configured conditional density calculator
    """
    if sigma_sim is None:
        sigma_sim = create_default_covariance(3, cova=0.0)
    
    return ConditionalDensity(sigma_sim) 