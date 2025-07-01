import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

class AdvancedWildBootstrap:
    """
    Advanced wild bootstrap procedures matching R's sophisticated implementation.
    
    This implements the sophisticated bootstrap from R including:
    - Wild bootstrap with Mammen distribution
    - Bandwidth smoothing with g = b * log(log(n)) * factor
    - Coverage study framework for 1000 bootstrap samples
    """
    
    def __init__(self, 
                 n_bootstrap: int = 1000,
                 wild_type: str = "mammen",
                 bandwidth_factor: float = 1.0,
                 confidence_level: float = 0.95,
                 parallel: bool = True,
                 n_jobs: Optional[int] = None):
        """
        Initialize advanced wild bootstrap.
        
        Parameters
        ----------
        n_bootstrap : int, default=1000
            Number of bootstrap samples (matching R coverage study)
        wild_type : str, default="mammen"
            Type of wild bootstrap ("mammen", "rademacher", "normal")
        bandwidth_factor : float, default=1.0
            Bandwidth scaling factor for smoothing
        confidence_level : float, default=0.95
            Confidence level for intervals
        parallel : bool, default=True
            Whether to use parallel processing
        n_jobs : int, optional
            Number of parallel jobs (-1 for all cores)
        """
        self.n_bootstrap = n_bootstrap
        self.wild_type = wild_type
        self.bandwidth_factor = bandwidth_factor
        self.confidence_level = confidence_level
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        
        # Store bootstrap results
        self.bootstrap_curves_ = None
        self.confidence_intervals_ = None
        self.coverage_diagnostics_ = None
    
    def _mammen_multipliers(self, n: int) -> np.ndarray:
        """
        Generate Mammen distribution multipliers exactly as in R.
        
        R: u_boot <- ((1-sqrt(5))/2) * (rho < (sqrt(5)+1)/(2*sqrt(5))) + ((1+sqrt(5))/2) * (rho >= (sqrt(5)+1)/(2*sqrt(5)))
        """
        rho = np.random.uniform(0, 1, n)
        threshold = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        
        # Mammen distribution values
        val1 = (1 - np.sqrt(5)) / 2  # ≈ -0.618
        val2 = (1 + np.sqrt(5)) / 2  # ≈ 1.618
        
        multipliers = val1 * (rho < threshold) + val2 * (rho >= threshold)
        return multipliers
    
    def _rademacher_multipliers(self, n: int) -> np.ndarray:
        """Generate Rademacher multipliers (+1 or -1 with equal probability)."""
        return 2 * np.random.binomial(1, 0.5, n) - 1
    
    def _normal_multipliers(self, n: int) -> np.ndarray:
        """Generate normal multipliers N(0,1)."""
        return np.random.randn(n)
    
    def _get_multipliers(self, n: int) -> np.ndarray:
        """Get bootstrap multipliers based on wild_type."""
        if self.wild_type == "mammen":
            return self._mammen_multipliers(n)
        elif self.wild_type == "rademacher":
            return self._rademacher_multipliers(n)
        elif self.wild_type == "normal":
            return self._normal_multipliers(n)
        else:
            raise ValueError(f"Unknown wild_type: {self.wild_type}")
    
    def _compute_bandwidth_smoothing(self, n: int, base_bandwidth: float = 1.0) -> float:
        """
        Compute bandwidth with smoothing exactly as in R.
        
        R: g = b * log(log(n)) * factor
        """
        if n <= 2:
            return base_bandwidth
        
        g = base_bandwidth * np.log(np.log(n)) * self.bandwidth_factor
        return max(g, 0.01)  # Ensure positive bandwidth
    
    def wild_bootstrap_single(self, 
                            estimator, 
                            X: pd.DataFrame, 
                            y: np.ndarray,
                            evaluation_points: Dict[str, np.ndarray],
                            seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Perform single wild bootstrap iteration.
        
        This matches R's wild bootstrap procedure with sophisticated multipliers.
        
        Parameters
        ----------
        estimator : ShapleyEstimator
            Fitted Shapley estimator
        X : pd.DataFrame
            Original features
        y : np.ndarray
            Original response
        evaluation_points : Dict[str, np.ndarray]
            Points to evaluate Shapley curves
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        Dict[str, np.ndarray]
            Bootstrap Shapley curves for each variable
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(y)
        
        # Get residuals from original fit
        y_pred = estimator.predict(X)
        residuals = y - y_pred
        
        # Generate wild bootstrap multipliers
        multipliers = self._get_multipliers(n)
        
        # Compute bandwidth smoothing
        bandwidth_smooth = self._compute_bandwidth_smoothing(n)
        
        # Create bootstrap response
        y_boot = y_pred + residuals * multipliers
        
        # Apply bandwidth smoothing (mimicking R's sophisticated procedure)
        if bandwidth_smooth > 0:
            smoothing_noise = np.random.normal(0, bandwidth_smooth * np.std(residuals), n)
            y_boot += smoothing_noise
        
        # Fit estimator to bootstrap data
        try:
            estimator_boot = estimator.__class__(**estimator.get_params())
            estimator_boot.fit(X, y_boot)
            
            # Estimate bootstrap curves
            curves_boot = estimator_boot.estimate_all_curves(evaluation_points)
            
            return curves_boot
            
        except Exception as e:
            warnings.warn(f"Bootstrap iteration failed: {str(e)}")
            # Return NaN curves
            return {var: np.full_like(points, np.nan) 
                   for var, points in evaluation_points.items()}
    
    def wild_bootstrap_parallel(self,
                              estimator,
                              X: pd.DataFrame,
                              y: np.ndarray,
                              evaluation_points: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Perform parallel wild bootstrap exactly matching R's coverage study.
        
        R procedure: 1000 bootstrap samples with parallel processing
        """
        if self.parallel and self.n_jobs > 1:
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Create seeds for reproducibility
                seeds = np.random.randint(0, 2**31, self.n_bootstrap)
                
                # Submit all bootstrap jobs
                futures = []
                for i in range(self.n_bootstrap):
                    future = executor.submit(
                        self.wild_bootstrap_single,
                        estimator, X, y, evaluation_points, seeds[i]
                    )
                    futures.append(future)
                
                # Collect results
                bootstrap_results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per bootstrap
                        bootstrap_results.append(result)
                        
                        if (i + 1) % 100 == 0:
                            print(f"  Completed {i + 1}/{self.n_bootstrap} bootstrap samples")
                            
                    except Exception as e:
                        warnings.warn(f"Bootstrap {i} failed: {str(e)}")
                        bootstrap_results.append({
                            var: np.full_like(points, np.nan) 
                            for var, points in evaluation_points.items()
                        })
                        
                return bootstrap_results
        else:
            # Sequential processing
            bootstrap_results = []
            for i in range(self.n_bootstrap):
                seed = np.random.randint(0, 2**31)
                result = self.wild_bootstrap_single(estimator, X, y, evaluation_points, seed)
                bootstrap_results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"  Completed {i + 1}/{self.n_bootstrap} bootstrap samples")
            
            return bootstrap_results
    
    def compute_confidence_intervals(self, 
                                   bootstrap_curves: List[Dict[str, np.ndarray]],
                                   evaluation_points: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute confidence intervals from bootstrap curves.
        
        Returns both pointwise and simultaneous confidence bands.
        """
        alpha = 1 - self.confidence_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        confidence_intervals = {}
        
        for var in evaluation_points.keys():
            # Stack all bootstrap curves for this variable
            boot_curves = np.array([curves[var] for curves in bootstrap_curves])
            
            # Remove NaN curves
            valid_curves = boot_curves[~np.isnan(boot_curves).any(axis=1)]
            
            if len(valid_curves) == 0:
                warnings.warn(f"No valid bootstrap curves for variable {var}")
                n_points = len(evaluation_points[var])
                confidence_intervals[var] = {
                    'lower': np.full(n_points, np.nan),
                    'upper': np.full(n_points, np.nan),
                    'coverage_prob': 0.0
                }
                continue
            
            # Pointwise confidence intervals
            lower_ci = np.percentile(valid_curves, lower_q * 100, axis=0)
            upper_ci = np.percentile(valid_curves, upper_q * 100, axis=0)
            
            # Coverage probability (fraction of successful bootstrap)
            coverage_prob = len(valid_curves) / len(bootstrap_curves)
            
            confidence_intervals[var] = {
                'lower': lower_ci,
                'upper': upper_ci,
                'coverage_prob': coverage_prob,
                'bootstrap_mean': np.mean(valid_curves, axis=0),
                'bootstrap_std': np.std(valid_curves, axis=0),
                'n_valid': len(valid_curves)
            }
        
        return confidence_intervals
    
    def coverage_study(self,
                      estimator,
                      X: pd.DataFrame,
                      y: np.ndarray,
                      evaluation_points: Dict[str, np.ndarray],
                      true_curves: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Comprehensive coverage study matching R's coverage.R analysis.
        
        This performs the full 1000-bootstrap coverage analysis from R.
        
        Parameters
        ----------
        estimator : ShapleyEstimator
            Fitted Shapley estimator
        X : pd.DataFrame
            Features data
        y : np.ndarray
            Response data
        evaluation_points : Dict[str, np.ndarray]
            Points to evaluate coverage
        true_curves : Dict[str, np.ndarray], optional
            True Shapley curves for coverage evaluation
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive coverage study results
        """
        print(f"Starting coverage study with {self.n_bootstrap} bootstrap samples...")
        print(f"Wild bootstrap type: {self.wild_type}")
        print(f"Confidence level: {self.confidence_level}")
        
        # Perform wild bootstrap
        bootstrap_curves = self.wild_bootstrap_parallel(estimator, X, y, evaluation_points)
        
        # Compute confidence intervals
        confidence_intervals = self.compute_confidence_intervals(bootstrap_curves, evaluation_points)
        
        # Store results
        self.bootstrap_curves_ = bootstrap_curves
        self.confidence_intervals_ = confidence_intervals
        
        # Coverage diagnostics
        coverage_diagnostics = self._compute_coverage_diagnostics(
            confidence_intervals, true_curves, evaluation_points
        )
        self.coverage_diagnostics_ = coverage_diagnostics
        
        print("✓ Coverage study completed!")
        success_rates = [f'{var}: {ci["coverage_prob"]:.1%}' for var, ci in confidence_intervals.items()]
        print(f"  Bootstrap success rates: {success_rates}")
        
        return {
            'bootstrap_curves': bootstrap_curves,
            'confidence_intervals': confidence_intervals,
            'coverage_diagnostics': coverage_diagnostics,
            'study_parameters': {
                'n_bootstrap': self.n_bootstrap,
                'wild_type': self.wild_type,
                'bandwidth_factor': self.bandwidth_factor,
                'confidence_level': self.confidence_level
            }
        }
    
    def _compute_coverage_diagnostics(self,
                                    confidence_intervals: Dict[str, Dict[str, np.ndarray]],
                                    true_curves: Optional[Dict[str, np.ndarray]],
                                    evaluation_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute coverage diagnostics if true curves are available."""
        if true_curves is None:
            return {'coverage_available': False}
        
        diagnostics = {'coverage_available': True}
        
        for var in evaluation_points.keys():
            if var not in true_curves or var not in confidence_intervals:
                continue
            
            true_curve = true_curves[var]
            ci = confidence_intervals[var]
            
            # Check if true curve is within confidence intervals
            within_ci = (true_curve >= ci['lower']) & (true_curve <= ci['upper'])
            coverage_rate = np.mean(within_ci)
            
            diagnostics[var] = {
                'empirical_coverage': coverage_rate,
                'nominal_coverage': self.confidence_level,
                'coverage_difference': coverage_rate - self.confidence_level,
                'points_covered': np.sum(within_ci),
                'total_points': len(within_ci)
            }
        
        return diagnostics 