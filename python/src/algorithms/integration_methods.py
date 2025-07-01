"""
Enhanced numerical integration methods for Shapley value analysis.

This module provides sophisticated integration methods matching R's cubature package,
with conditional density integration and advanced quadrature methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from scipy import integrate
from scipy.stats import multivariate_normal
import warnings

# Import our utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class ConditionalDensityIntegrator:
    """
    Conditional density integration matching R's norm1/norm2 functions.
    
    This class implements the sophisticated conditional normal density integration
    used in R's integral_estimation.R and integral_population.R.
    """
    
    def __init__(self, covariance_matrix: np.ndarray):
        """
        Initialize conditional density integrator.
        
        Parameters
        ----------
        covariance_matrix : np.ndarray
            Covariance matrix for conditional densities (3x3 for 3D case)
        """
        self.sigma_sim = covariance_matrix
        self.d_ = covariance_matrix.shape[0]
        self._setup_conditional_parameters()
    
    def _setup_conditional_parameters(self):
        """Setup conditional density parameters exactly as in R."""
        # For x1/x2,x3 (1D conditional on 2D) - matching R
        self.sigma_xy_1 = self.sigma_sim[0, 1:3].reshape(1, -1)  # t(sigma_sim[1, 2:3])
        self.sigma_yy_1 = self.sigma_sim[1:3, 1:3]  # sigma_sim[2:3, 2:3]
        self.sigma_yx_1 = self.sigma_xy_1.T
        self.sigma_xx_1 = self.sigma_sim[0, 0]
        
        # Conditional variance
        self.inv_yy_1 = self.sigma_xy_1 @ np.linalg.inv(self.sigma_yy_1)
        self.c_var_1 = self.sigma_xx_1 - self.sigma_xy_1 @ np.linalg.inv(self.sigma_yy_1) @ self.sigma_yx_1
        self.pre_mult = 1.0 / np.sqrt(self.c_var_1 * 2 * np.pi)
        self.sq = np.sqrt(self.c_var_1)
        
        # For x1,x2/x3 (2D conditional on 1D) - matching R
        self.sigma_xy_2 = self.sigma_sim[0:2, 2].reshape(-1, 1)
        self.sigma_yy_2 = self.sigma_sim[2, 2]
        self.sigma_yx_2 = self.sigma_xy_2.T
        self.sigma_xx_2 = self.sigma_sim[0:2, 0:2]
        
        # Conditional variance
        self.inv_yy_2 = self.sigma_xy_2 * (1.0 / self.sigma_yy_2)
        self.c_var_2 = self.sigma_xx_2 - self.sigma_xy_2 * (1.0 / self.sigma_yy_2) * self.sigma_yx_2
        self.pre_mult2 = 1.0 / np.sqrt(np.linalg.det(2 * np.pi * self.sigma_xx_2))
        self.inv_xx_2 = np.linalg.inv(self.sigma_xx_2)
    
    def norm1(self, dep: float, cond: np.ndarray) -> float:
        """1D conditional density exactly matching R norm1."""
        x = dep
        Y_1 = cond
        c_mu_1 = self.inv_yy_1 @ Y_1
        return self.pre_mult * np.exp(-0.5 * ((x - c_mu_1[0]) / self.sq) ** 2)
    
    def norm2(self, dep: np.ndarray, cond: float) -> float:
        """2D conditional density exactly matching R norm2."""
        x = dep
        Y_2 = cond
        c_mu_2 = (self.inv_yy_2 * Y_2).flatten()
        diff = x - c_mu_2
        return self.pre_mult2 * np.exp(-0.5 * diff.T @ self.inv_xx_2 @ diff)
    
    def norm1_vec(self, dep: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Vectorized 1D conditional density."""
        c_mu_1 = self.inv_yy_1 @ cond
        return self.pre_mult * np.exp(-0.5 * ((dep - c_mu_1[0]) / self.sq) ** 2)
    
    def norm2_vec(self, dep: np.ndarray, cond: float) -> np.ndarray:
        """Vectorized 2D conditional density."""
        c_mu_2 = (self.inv_yy_2 * cond).flatten()
        results = []
        for i in range(dep.shape[1]):
            diff = dep[:, i] - c_mu_2
            results.append(self.pre_mult2 * np.exp(-0.5 * diff.T @ self.inv_xx_2 @ diff))
        return np.array(results)


class CubatureStyleIntegrator:
    """
    Advanced multidimensional integrator matching R's cubature package.
    
    This implements sophisticated adaptive integration methods similar to
    R's cubintegrate function with cuhre method.
    """
    
    def __init__(self, 
                 method: str = 'adaptive',
                 tolerance: float = 3e-1,
                 max_evals: int = 128,
                 integration_bounds: Tuple[float, float] = (-5, 5)):
        """
        Initialize the cubature-style integrator.
        
        Parameters
        ----------
        method : str, default='adaptive'
            Integration method ('adaptive', 'vegas', 'cuhre_like')
        tolerance : float, default=3e-1
            Relative tolerance (matching R default)
        max_evals : int, default=128
            Maximum function evaluations (matching R nVec=128L)
        integration_bounds : Tuple[float, float]
            Default integration bounds (matching R rep(-5, d))
        """
        self.method = method
        self.tolerance = tolerance
        self.max_evals = max_evals
        self.integration_bounds = integration_bounds
    
    def cubintegrate_1d(self, func: Callable, bounds: Tuple[float, float], **kwargs) -> Dict[str, float]:
        """
        1D integration matching R cubintegrate for single dimension.
        
        R: cubintegrate(f=function, lower=rep(-5, 1), upper=rep(5, 1), method="cuhre", relTol=3e-1, nVec=128L)
        """
        try:
            if self.method == 'adaptive':
                result, error = integrate.quad(
                    func, bounds[0], bounds[1],
                    epsrel=self.tolerance,
                    limit=self.max_evals,
                    **kwargs
                )
            else:
                # Fallback to basic quad
                result, error = integrate.quad(func, bounds[0], bounds[1], **kwargs)
            
            return {
                'integral': result,
                'error': error,
                'subdivisions': self.max_evals,
                'convergence': 0
            }
        except Exception as e:
            warnings.warn(f"Integration failed: {str(e)}")
            return {'integral': np.nan, 'error': np.inf, 'subdivisions': 0, 'convergence': 1}
    
    def cubintegrate_2d(self, func: Callable, bounds: List[Tuple[float, float]], **kwargs) -> Dict[str, float]:
        """
        2D integration matching R cubintegrate for two dimensions.
        
        R: cubintegrate(f=function, lower=rep(-5, 2), upper=rep(5, 2), method="cuhre", relTol=3e-1, nVec=128L)
        """
        try:
            if self.method == 'adaptive':
                result, error = integrate.dblquad(
                    func,
                    bounds[0][0], bounds[0][1],  # x bounds
                    lambda x: bounds[1][0], lambda x: bounds[1][1],  # y bounds
                    epsrel=self.tolerance,
                    **kwargs
                )
            else:
                result, error = integrate.dblquad(
                    func,
                    bounds[0][0], bounds[0][1],
                    lambda x: bounds[1][0], lambda x: bounds[1][1],
                    **kwargs
                )
            
            return {
                'integral': result,
                'error': error,
                'subdivisions': self.max_evals,
                'convergence': 0
            }
        except Exception as e:
            warnings.warn(f"2D Integration failed: {str(e)}")
            return {'integral': np.nan, 'error': np.inf, 'subdivisions': 0, 'convergence': 1}
    
    def cubintegrate_3d(self, func: Callable, bounds: List[Tuple[float, float]], **kwargs) -> Dict[str, float]:
        """
        3D integration matching R cubintegrate for three dimensions.
        
        R: cubintegrate(f=function, lower=rep(-5, 3), upper=rep(5, 3), method="cuhre", relTol=3e-1, nVec=128L)
        """
        try:
            if self.method == 'adaptive':
                result, error = integrate.tplquad(
                    func,
                    bounds[0][0], bounds[0][1],  # x bounds
                    lambda x: bounds[1][0], lambda x: bounds[1][1],  # y bounds
                    lambda x, y: bounds[2][0], lambda x, y: bounds[2][1],  # z bounds
                    epsrel=self.tolerance,
                    **kwargs
                )
            else:
                result, error = integrate.tplquad(
                    func,
                    bounds[0][0], bounds[0][1],
                    lambda x: bounds[1][0], lambda x: bounds[1][1],
                    lambda x, y: bounds[2][0], lambda x, y: bounds[2][1],
                    **kwargs
                )
            
            return {
                'integral': result,
                'error': error,
                'subdivisions': self.max_evals,
                'convergence': 0
            }
        except Exception as e:
            warnings.warn(f"3D Integration failed: {str(e)}")
            return {'integral': np.nan, 'error': np.inf, 'subdivisions': 0, 'convergence': 1}
    
    def hcubature(self, func: Callable, lower_bounds: np.ndarray, upper_bounds: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Multidimensional integration matching R's hcubature function.
        
        R: hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
        """
        ndim = len(lower_bounds)
        bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(ndim)]
        
        if ndim == 1:
            return self.cubintegrate_1d(func, bounds[0], **kwargs)
        elif ndim == 2:
            return self.cubintegrate_2d(func, bounds, **kwargs)
        elif ndim == 3:
            return self.cubintegrate_3d(func, bounds, **kwargs)
        else:
            # For higher dimensions, use Monte Carlo integration
            return self._monte_carlo_integration(func, bounds, **kwargs)
    
    def _monte_carlo_integration(self, func: Callable, bounds: List[Tuple[float, float]], n_samples: int = 10000) -> Dict[str, float]:
        """Monte Carlo integration for higher dimensions."""
        ndim = len(bounds)
        
        # Generate random samples
        samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_samples, ndim)
        )
        
        # Evaluate function at samples
        values = []
        for i in range(n_samples):
            try:
                val = func(*samples[i])
                values.append(val)
            except:
                values.append(0.0)
        
        values = np.array(values)
        
        # Compute volume
        volume = np.prod([b[1] - b[0] for b in bounds])
        
        # Monte Carlo estimate
        integral = volume * np.mean(values)
        error = volume * np.std(values) / np.sqrt(n_samples)
        
        return {
            'integral': integral,
            'error': error,
            'subdivisions': n_samples,
            'convergence': 0
        }


class AdvancedShapleyIntegrator:
    """
    Advanced Shapley integrator combining conditional densities with cubature methods.
    
    This class implements the sophisticated integration procedures from R for
    marginal model estimation and Shapley curve integration.
    """
    
    def __init__(self,
                 covariance_matrix: Optional[np.ndarray] = None,
                 integration_tolerance: float = 3e-1,
                 integration_bounds: Tuple[float, float] = (-5, 5)):
        """
        Initialize the advanced Shapley integrator.
        
        Parameters
        ----------
        covariance_matrix : np.ndarray, optional
            Covariance matrix for conditional integration
        integration_tolerance : float, default=3e-1
            Integration tolerance (matching R)
        integration_bounds : Tuple[float, float], default=(-5, 5)
            Integration bounds (matching R)
        """
        if covariance_matrix is None:
            # Default R covariance matrix
            cova = 0
            covariance_matrix = np.array([
                [4, cova, cova],
                [cova, 4, cova],
                [cova, cova, 4]
            ])
        
        self.conditional_integrator = ConditionalDensityIntegrator(covariance_matrix)
        self.cubature_integrator = CubatureStyleIntegrator(
            tolerance=integration_tolerance,
            integration_bounds=integration_bounds
        )
        self.integration_bounds = integration_bounds
    
    def create_marginal_estimators(self, full_model_predictor: Callable) -> Dict[str, Callable]:
        """
        Create marginal model estimators using integration.
        
        This matches the R functions m_x1_est, m_x2_est, etc. from integral_estimation.R
        
        Parameters
        ----------
        full_model_predictor : Callable
            Function that predicts from the full model given a DataFrame
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary of marginal estimators
        """
        def m_x1_est(X):
            """E[m(x1,X2,X3)|X1=x1] - matching R m_x1_est."""
            x1 = float(X[0])
            
            def integrand(x2, x3):
                x_eval_df = pd.DataFrame([[x1, x2, x3]], columns=['X1', 'X2', 'X3'])
                m_pred = full_model_predictor(x_eval_df)
                density = self.conditional_integrator.norm2(np.array([x2, x3]), x1)
                return m_pred * density
            
            result = self.cubature_integrator.cubintegrate_2d(
                integrand, 
                [(self.integration_bounds[0], self.integration_bounds[1]),
                 (self.integration_bounds[0], self.integration_bounds[1])]
            )
            return result['integral']
        
        def m_x2_est(X):
            """E[m(X1,x2,X3)|X2=x2] - matching R m_x2_est."""
            x2 = float(X[1])
            
            def integrand(x1, x3):
                x_eval_df = pd.DataFrame([[x1, x2, x3]], columns=['X1', 'X2', 'X3'])
                m_pred = full_model_predictor(x_eval_df)
                density = self.conditional_integrator.norm2(np.array([x1, x3]), x2)
                return m_pred * density
            
            result = self.cubature_integrator.cubintegrate_2d(
                integrand,
                [(self.integration_bounds[0], self.integration_bounds[1]),
                 (self.integration_bounds[0], self.integration_bounds[1])]
            )
            return result['integral']
        
        def m_x3_est(X):
            """E[m(X1,X2,x3)|X3=x3] - matching R m_x3_est."""
            x3 = float(X[2])
            
            def integrand(x1, x2):
                x_eval_df = pd.DataFrame([[x1, x2, x3]], columns=['X1', 'X2', 'X3'])
                m_pred = full_model_predictor(x_eval_df)
                density = self.conditional_integrator.norm2(np.array([x1, x2]), x3)
                return m_pred * density
            
            result = self.cubature_integrator.cubintegrate_2d(
                integrand,
                [(self.integration_bounds[0], self.integration_bounds[1]),
                 (self.integration_bounds[0], self.integration_bounds[1])]
            )
            return result['integral']
        
        def m_x1_x2_est(X):
            """E[m(x1,x2,X3)|X1=x1,X2=x2] - matching R m_x1_x2_est."""
            x1, x2 = float(X[0]), float(X[1])
            
            def integrand(x3):
                x_eval_df = pd.DataFrame([[x1, x2, x3]], columns=['X1', 'X2', 'X3'])
                m_pred = full_model_predictor(x_eval_df)
                density = self.conditional_integrator.norm1(x3, np.array([x1, x2]))
                return m_pred * density
            
            result = self.cubature_integrator.cubintegrate_1d(
                integrand, (self.integration_bounds[0], self.integration_bounds[1])
            )
            return result['integral']
        
        def m_x1_x3_est(X):
            """E[m(x1,X2,x3)|X1=x1,X3=x3] - matching R m_x1_x3_est."""
            x1, x3 = float(X[0]), float(X[2])
            
            def integrand(x2):
                x_eval_df = pd.DataFrame([[x1, x2, x3]], columns=['X1', 'X2', 'X3'])
                m_pred = full_model_predictor(x_eval_df)
                density = self.conditional_integrator.norm1(x2, np.array([x1, x3]))
                return m_pred * density
            
            result = self.cubature_integrator.cubintegrate_1d(
                integrand, (self.integration_bounds[0], self.integration_bounds[1])
            )
            return result['integral']
        
        def m_x2_x3_est(X):
            """E[m(X1,x2,x3)|X2=x2,X3=x3] - matching R m_x2_x3_est."""
            x2, x3 = float(X[1]), float(X[2])
            
            def integrand(x1):
                x_eval_df = pd.DataFrame([[x1, x2, x3]], columns=['X1', 'X2', 'X3'])
                m_pred = full_model_predictor(x_eval_df)
                density = self.conditional_integrator.norm1(x1, np.array([x2, x3]))
                return m_pred * density
            
            result = self.cubature_integrator.cubintegrate_1d(
                integrand, (self.integration_bounds[0], self.integration_bounds[1])
            )
            return result['integral']
        
        def m_full_hat(X):
            """Full model prediction - matching R m_full_hat."""
            return full_model_predictor(X)
        
        return {
            'm_x1_est': m_x1_est,
            'm_x2_est': m_x2_est,
            'm_x3_est': m_x3_est,
            'm_x1_x2_est': m_x1_x2_est,
            'm_x1_x3_est': m_x1_x3_est,
            'm_x2_x3_est': m_x2_x3_est,
            'm_full_hat': m_full_hat
        }
    
    def integrate_squared_error(self, se_function: Callable, bounds: np.ndarray) -> float:
        """
        Integrate squared error function over multidimensional domain.
        
        Matches R: hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
        """
        result = self.cubature_integrator.hcubature(
            se_function, bounds, bounds  # lower and upper bounds the same format
        )
        return result['integral']


def test_integration_methods():
    """Test the enhanced integration methods."""
    print("Testing enhanced integration methods...")
    
    # Test conditional density integrator
    cova = 0
    sigma = np.array([[4, cova, cova], [cova, 4, cova], [cova, cova, 4]])
    conditional_integrator = ConditionalDensityIntegrator(sigma)
    
    # Test norm1 and norm2
    test_point_1d = 0.5
    test_point_2d = np.array([0.5, -0.5])
    test_cond = np.array([0.0, 0.0])
    
    norm1_val = conditional_integrator.norm1(test_point_1d, test_cond)
    norm2_val = conditional_integrator.norm2(test_point_2d, 0.0)
    
    print(f"  norm1({test_point_1d}, [0.0, 0.0]): {norm1_val:.6f}")
    print(f"  norm2([0.5, -0.5], 0.0): {norm2_val:.6f}")
    
    # Test cubature integrator
    cubature = CubatureStyleIntegrator()
    
    # Test 1D integration
    def test_func_1d(x):
        return x**2
    
    result_1d = cubature.cubintegrate_1d(test_func_1d, (-1, 1))
    print(f" ∫x²dx from -1 to 1: {result_1d['integral']:.6f} (expected: 0.667)")
    
    # Test 2D integration
    def test_func_2d(x, y):
        return x*y
    
    result_2d = cubature.cubintegrate_2d(test_func_2d, [(-1, 1), (-1, 1)])
    print(f" ∫∫xy dxdy: {result_2d['integral']:.6f} (expected: 0.0)")
    
    print("✓ Enhanced integration methods tests completed!")


if __name__ == "__main__":
    test_integration_methods() 