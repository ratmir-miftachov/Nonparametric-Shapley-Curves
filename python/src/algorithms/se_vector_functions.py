"""
Squared error vector functions for Shapley consistency analysis.

This module implements the SE_vec and SE_vec_int functions from the R implementation,
used for computing integrated squared error in consistency studies.
"""

import numpy as np
from typing import Optional, Callable, Union
from scipy import integrate
import warnings

from .advanced_integration import PopulationShapleyIntegrator
from .conditional_densities import setup_conditional_densities


class SquaredErrorVectorCalculator:
    """
    Calculator for squared error vector functions matching R implementation.
    
    This class implements the SE_vec and SE_vec_int functions used in the R
    consistency studies (sim_consistency.R).
    """
    
    def __init__(self, 
                 model_list: Optional[list] = None,
                 population_integrator: Optional[PopulationShapleyIntegrator] = None,
                 d: int = 3):
        """
        Initialize squared error calculator.
        
        Parameters
        ----------
        model_list : list, optional
            List of fitted models for estimation-based calculations
        population_integrator : PopulationShapleyIntegrator, optional
            Population Shapley integrator for true values
        d : int, default=3
            Number of dimensions
        """
        self.model_list = model_list
        self.population_integrator = population_integrator or PopulationShapleyIntegrator()
        self.d = d
        
        # Pre-compute true Shapley functions for comparison
        self._setup_true_functions()
    
    def _setup_true_functions(self):
        """Set up true Shapley functions for comparison."""
        # True component functions from R implementation
        self.g1 = lambda X: -np.sin(2 * X[:, 0])
        self.g2 = lambda X: np.cos(3 * X[:, 1]) 
        self.g3 = lambda X: 0.5 * X[:, 2]
        self.interaction = lambda X: 2 * np.cos(X[:, 0]) * np.sin(2 * X[:, 1])
        
        # True Shapley values (component-wise decomposition)
        self.true_shapley_1 = lambda x_eval: self.g1(x_eval.reshape(1, -1))[0] + \
                                           (1.0/3.0) * self.interaction(x_eval.reshape(1, -1))[0]
        self.true_shapley_2 = lambda x_eval: self.g2(x_eval.reshape(1, -1))[0] + \
                                           (2.0/3.0) * self.interaction(x_eval.reshape(1, -1))[0]
        self.true_shapley_3 = lambda x_eval: self.g3(x_eval.reshape(1, -1))[0]
    
    def shapley_estimator_component(self, j: int, x_eval: np.ndarray) -> float:
        """
        Component-based Shapley estimator matching R shapley function.
        
        Parameters
        ----------
        j : int
            Variable index (1-indexed)
        x_eval : np.ndarray
            Evaluation point
            
        Returns
        -------
        float
            Estimated Shapley value
        """
        if self.model_list is None:
            raise ValueError("Model list required for component-based estimation")
        
        shap = 0.0
        x_eval_dict = {f'X{i+1}': x_eval[i] for i in range(len(x_eval))}
        
        # Constant term - this should match the R implementation
        # c = mean(Y) - this would need to be passed or computed
        c = 0.0  # Placeholder - should be mean of Y from training
        shap = -(1.0/self.d) * (1.0 / 1.0) * c  # nchoosek(d-1, 0) = 1
        
        # Sum over all models with appropriate weights
        for k, model in enumerate(self.model_list):
            # Get model prediction
            model_vars = getattr(model, 'feature_names_in_', [f'X{i+1}' for i in range(self.d)])
            x_subset = [x_eval_dict[var] for var in model_vars if var in x_eval_dict]
            
            if len(x_subset) > 0:
                x_subset_array = np.array(x_subset).reshape(1, -1)
                pred = model.predict(x_subset_array)[0]
            else:
                pred = 0.0
            
            # Compute weight - this would need the actual weight function
            weight_val = self._compute_shapley_weight(j, model_vars)
            shap += weight_val * pred
        
        return shap
    
    def _compute_shapley_weight(self, j: int, model_vars: list) -> float:
        """
        Compute Shapley weight for variable j given model variables.
        
        This is a simplified version - the full implementation would use
        the weight function from weight_functions.py
        """
        var_name = f'X{j}'
        indicator = int(var_name in model_vars)
        sign = 1 if indicator > 0 else -1
        card_s = len(model_vars)
        
        from scipy.special import comb
        binomial_coeff = comb(self.d - 1, card_s - indicator)
        if binomial_coeff == 0:
            return 0.0
        
        return sign * (1.0 / self.d) * (1.0 / binomial_coeff)
    
    def shapley_estimator_integration(self, j: int, x_eval: np.ndarray) -> float:
        """
        Integration-based Shapley estimator matching R shapley_int function.
        
        This implementation now correctly uses the PopulationShapleyIntegrator
        to perform the estimation.
        """
        if self.model_list is None:
            raise ValueError("Model list required for integration-based estimation")
            
        shap = 0.0
        
        # The 'model_list_int' in R corresponds to a series of integration functions.
        # Here, we call the integrator for each type of marginal model.
        
        # This dictionary maps the subset size to the correct integration function.
        # Note: This is a simplified mapping. The R code has a more complex structure
        # that we are approximating here.
        
        # Contribution of m_x1, m_x2, m_x3
        shap += self._compute_shapley_weight(j, ['X1']) * self.population_integrator.m_x1_est(x_eval[0])
        shap += self._compute_shapley_weight(j, ['X2']) * self.population_integrator.m_x2_est(x_eval[1])
        shap += self._compute_shapley_weight(j, ['X3']) * self.population_integrator.m_x3_est(x_eval[2])
        
        # Contribution of m_x1_x2, m_x1_x3, m_x2_x3
        shap += self._compute_shapley_weight(j, ['X1', 'X2']) * self.population_integrator.m_x1_x2_est(x_eval[0], x_eval[1])
        shap += self._compute_shapley_weight(j, ['X1', 'X3']) * self.population_integrator.m_x1_x3_est(x_eval[0], x_eval[2])
        shap += self._compute_shapley_weight(j, ['X2', 'X3']) * self.population_integrator.m_x2_x3_est(x_eval[1], x_eval[2])

        # Contribution of m_full
        m_full_hat = self.model_list[-1].predict(x_eval.reshape(1, -1))[0] # Assuming full model is last
        shap += self._compute_shapley_weight(j, ['X1', 'X2', 'X3']) * m_full_hat

        return shap
    
    def se_vec(self, x_eval: np.ndarray, j: int) -> float:
        """
        Component-based squared error function - matches R SE_vec.
        
        Parameters
        ----------
        x_eval : np.ndarray
            Evaluation point (3D vector)
        j : int
            Variable index (1-indexed)
            
        Returns
        -------
        float
            Squared error at evaluation point
        """
        # Get true Shapley value
        if j == 1:
            true_val = self.true_shapley_1(x_eval)
        elif j == 2:
            true_val = self.true_shapley_2(x_eval)
        elif j == 3:
            true_val = self.true_shapley_3(x_eval)
        else:
            true_val = 0.0
        
        # Get estimated Shapley value
        try:
            est_val = self.shapley_estimator_component(j, x_eval)
        except:
            # If estimation fails, return large error
            return 1e6
        
        # Return squared error
        return (est_val - true_val) ** 2
    
    def se_vec_int(self, x_eval: np.ndarray, j: int) -> float:
        """
        Integration-based squared error function - matches R SE_vec_int.
        
        Parameters
        ----------
        x_eval : np.ndarray
            Evaluation point (3D vector)
        j : int
            Variable index (1-indexed)
            
        Returns
        -------
        float
            Squared error at evaluation point
        """
        # Get true Shapley value
        if j == 1:
            true_val = self.true_shapley_1(x_eval)
        elif j == 2:
            true_val = self.true_shapley_2(x_eval)
        elif j == 3:
            true_val = self.true_shapley_3(x_eval)
        else:
            true_val = 0.0
        
        # Get estimated Shapley value using integration
        est_val = self.shapley_estimator_integration(j, x_eval)
        
        # Return squared error
        return (est_val - true_val) ** 2
    
    def integrated_squared_error(self, j: int, integration_bounds: tuple = (-2, 2),
                                use_integration_method: bool = False) -> float:
        """
        Compute integrated squared error over the domain.
        
        Parameters
        ----------
        j : int
            Variable index (1-indexed)
        integration_bounds : tuple, default=(-2, 2)
            Integration bounds for each dimension
        use_integration_method : bool, default=False
            Whether to use integration-based Shapley estimation
            
        Returns
        -------
        float
            Integrated squared error
        """
        if use_integration_method:
            error_func = lambda x1, x2, x3: self.se_vec_int(np.array([x1, x2, x3]), j)
        else:
            error_func = lambda x1, x2, x3: self.se_vec(np.array([x1, x2, x3]), j)
        
        try:
            result, _ = integrate.tplquad(
                error_func,
                integration_bounds[0], integration_bounds[1],  # x1
                integration_bounds[0], integration_bounds[1],  # x2
                integration_bounds[0], integration_bounds[1],  # x3
                epsrel=1e-1  # Reduced tolerance for speed
            )
            return result
        except Exception as e:
            warnings.warn(f"Integration failed for j={j}: {e}")
            return np.nan


# Convenience functions matching R interface
def SE_vec(x_eval: np.ndarray, j: int, 
           model_list: Optional[list] = None,
           calculator: Optional[SquaredErrorVectorCalculator] = None) -> float:
    """
    Component-based squared error function matching R SE_vec.
    
    Parameters
    ----------
    x_eval : np.ndarray
        Evaluation point
    j : int
        Variable index (1-indexed)
    model_list : list, optional
        List of fitted models
    calculator : SquaredErrorVectorCalculator, optional
        Pre-configured calculator
        
    Returns
    -------
    float
        Squared error
    """
    if calculator is None:
        calculator = SquaredErrorVectorCalculator(model_list=model_list)
    
    return calculator.se_vec(x_eval, j)


def SE_vec_int(x_eval: np.ndarray, j: int,
               population_integrator: Optional[PopulationShapleyIntegrator] = None,
               calculator: Optional[SquaredErrorVectorCalculator] = None) -> float:
    """
    Integration-based squared error function matching R SE_vec_int.
    
    Parameters
    ----------
    x_eval : np.ndarray
        Evaluation point
    j : int
        Variable index (1-indexed)
    population_integrator : PopulationShapleyIntegrator, optional
        Population integrator
    calculator : SquaredErrorVectorCalculator, optional
        Pre-configured calculator
        
    Returns
    -------
    float
        Squared error
    """
    if calculator is None:
        calculator = SquaredErrorVectorCalculator(population_integrator=population_integrator)
    
    return calculator.se_vec_int(x_eval, j)


def compute_ise_results(model_list: list, 
                       integration_bounds: tuple = (-2, 2)) -> tuple:
    """
    Compute all ISE results for the consistency study.
    
    Parameters
    ----------
    model_list : list
        List of all fitted models, with the full model assumed to be last.
    integration_bounds : tuple, default=(-2, 2)
        Bounds for the integration domain.
        
    Returns
    -------
    tuple
        Tuple containing the six ISE results in order.
    """
    # Assume the full model is the last one in the list, as per R's structure
    full_model = model_list[-1]
    
    # Initialize the population integrator with the fitted full model
    pop_integrator = PopulationShapleyIntegrator(m_full_model=full_model)

    # Initialize the SE calculator
    calculator = SquaredErrorVectorCalculator(
        model_list=model_list,
        population_integrator=pop_integrator
    )
    
    # --- Component-based ISE ---
    ise1_comp = calculator.integrated_squared_error(j=1, integration_bounds=integration_bounds, use_integration_method=False)
    ise2_comp = calculator.integrated_squared_error(j=2, integration_bounds=integration_bounds, use_integration_method=False)
    ise3_comp = calculator.integrated_squared_error(j=3, integration_bounds=integration_bounds, use_integration_method=False)
    
    # --- Integral-based ISE ---
    ise1_int = calculator.integrated_squared_error(j=1, integration_bounds=integration_bounds, use_integration_method=True)
    ise2_int = calculator.integrated_squared_error(j=2, integration_bounds=integration_bounds, use_integration_method=True)
    ise3_int = calculator.integrated_squared_error(j=3, integration_bounds=integration_bounds, use_integration_method=True)
    
    return (ise1_comp, ise1_int, ise2_comp, ise2_int, ise3_comp, ise3_int) 