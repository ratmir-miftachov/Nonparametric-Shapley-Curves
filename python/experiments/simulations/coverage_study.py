"""
Coverage Study Framework matching R's coverage.R analysis.

This module implements the comprehensive coverage probability estimation
and diagnostics from the R implementation with 1000 bootstrap samples.
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.shapley_estimator import ShapleyEstimator
from algorithms.bootstrap_procedures import AdvancedWildBootstrap
from algorithms.population_shapley import PopulationShapleyEstimator
from utils.data_preprocessing import generate_dgp_data


class CoverageStudy:
    """
    Comprehensive coverage study framework matching R's coverage.R.
    
    This implements the full coverage probability analysis with:
    - 1000 bootstrap samples per configuration
    - Multiple sample sizes (n = 200, 300, 400, 500)
    - Both additive and interaction effects
    - Coverage probability estimation
    - Confidence interval diagnostics
    """
    
    def __init__(self,
                 sample_sizes: List[int] = [200, 300, 400, 500],
                 n_replications: int = 100,
                 n_bootstrap: int = 1000,
                 confidence_levels: List[float] = [0.90, 0.95, 0.99],
                 dgp_types: List[str] = ['additive', 'interaction'],
                 evaluation_grid_size: int = 21,
                 parallel: bool = True,
                 save_results: bool = True,
                 results_dir: str = 'results/coverage_study'):
        """
        Initialize coverage study framework.
        
        Parameters
        ----------
        sample_sizes : List[int], default=[200, 300, 400, 500]
            Sample sizes to test (matching R study)
        n_replications : int, default=100
            Number of replications per configuration
        n_bootstrap : int, default=1000
            Bootstrap samples per replication (matching R)
        confidence_levels : List[float]
            Confidence levels to test
        dgp_types : List[str]
            Data generating process types
        evaluation_grid_size : int, default=21
            Size of evaluation grid (-2 to 2, 21 points)
        parallel : bool, default=True
            Use parallel processing
        save_results : bool, default=True
            Save results to disk
        results_dir : str
            Directory for saving results
        """
        self.sample_sizes = sample_sizes
        self.n_replications = n_replications
        self.n_bootstrap = n_bootstrap
        self.confidence_levels = confidence_levels
        self.dgp_types = dgp_types
        self.evaluation_grid_size = evaluation_grid_size
        self.parallel = parallel
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        
        # Create results directory
        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation points (matching R grid)
        self.evaluation_points = {
            'X1': np.linspace(-2, 2, self.evaluation_grid_size),
            'X2': np.linspace(-2, 2, self.evaluation_grid_size),
            'X3': np.linspace(-2, 2, self.evaluation_grid_size)
        }
        
        # Storage for results
        self.coverage_results_ = {}
        self.summary_statistics_ = {}
    
    def run_single_coverage_test(self,
                                n: int,
                                dgp_type: str,
                                confidence_level: float,
                                replication: int,
                                estimator_params: Optional[Dict] = None,
                                bootstrap_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run single coverage test replication.
        
        This matches R's single iteration within the coverage study loop.
        
        Parameters
        ----------
        n : int
            Sample size
        dgp_type : str
            Data generating process type
        confidence_level : float
            Confidence level for intervals
        replication : int
            Replication number
        estimator_params : dict, optional
            Parameters for Shapley estimator
        bootstrap_params : dict, optional
            Parameters for bootstrap procedure
            
        Returns
        -------
        Dict[str, Any]
            Coverage test results
        """
        np.random.seed(replication * 12345)  # Reproducible seeds
        
        # Generate data
        X, y = generate_dgp_data(
            n=n,
            dgp_type=dgp_type,
            noise_level=0.1,
            random_state=replication
        )
        
        # Fit Shapley estimator
        estimator_params = estimator_params or {}
        estimator = ShapleyEstimator(**estimator_params)
        estimator.fit(X, y)
        
        # Estimate Shapley curves
        estimated_curves = estimator.estimate_all_curves(self.evaluation_points)
        
        # Compute true Shapley curves for coverage evaluation
        population_estimator = PopulationShapleyEstimator(dgp_type=dgp_type)
        true_curves = population_estimator.compute_true_shapley_curves(self.evaluation_points)
        
        # Perform bootstrap
        bootstrap_params = bootstrap_params or {}
        bootstrap_params.update({
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': confidence_level,
            'parallel': False  # Sequential within replication for stability
        })
        
        bootstrap = AdvancedWildBootstrap(**bootstrap_params)
        
        try:
            coverage_study_results = bootstrap.coverage_study(
                estimator, X, y, self.evaluation_points, true_curves
            )
            
            # Extract coverage diagnostics
            coverage_diagnostics = coverage_study_results['coverage_diagnostics']
            confidence_intervals = coverage_study_results['confidence_intervals']
            
            # Compute summary statistics
            coverage_rates = {}
            interval_widths = {}
            
            for var in ['X1', 'X2', 'X3']:
                if var in coverage_diagnostics and coverage_diagnostics['coverage_available']:
                    coverage_rates[var] = coverage_diagnostics[var]['empirical_coverage']
                    
                    # Compute average interval width
                    ci = confidence_intervals[var]
                    interval_widths[var] = np.mean(ci['upper'] - ci['lower'])
                else:
                    coverage_rates[var] = np.nan
                    interval_widths[var] = np.nan
            
            return {
                'n': n,
                'dgp_type': dgp_type,
                'confidence_level': confidence_level,
                'replication': replication,
                'coverage_rates': coverage_rates,
                'interval_widths': interval_widths,
                'bootstrap_success': True,
                'bootstrap_diagnostics': {
                    'successful_bootstraps': {
                        var: ci['n_valid'] for var, ci in confidence_intervals.items()
                    },
                    'bootstrap_coverage_prob': {
                        var: ci['coverage_prob'] for var, ci in confidence_intervals.items()
                    }
                }
            }
            
        except Exception as e:
            warnings.warn(f"Bootstrap failed for n={n}, dgp={dgp_type}, rep={replication}: {str(e)}")
            return {
                'n': n,
                'dgp_type': dgp_type,
                'confidence_level': confidence_level,
                'replication': replication,
                'coverage_rates': {'X1': np.nan, 'X2': np.nan, 'X3': np.nan},
                'interval_widths': {'X1': np.nan, 'X2': np.nan, 'X3': np.nan},
                'bootstrap_success': False,
                'error': str(e)
            }
    
    def run_coverage_study(self,
                          estimator_params: Optional[Dict] = None,
                          bootstrap_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive coverage study matching R's coverage.R.
        
        This implements the full nested loop structure from R:
        - Loop over sample sizes
        - Loop over DGP types
        - Loop over confidence levels
        - Loop over replications
        """
        print("🔬 Starting Comprehensive Coverage Study")
        print(f"Sample sizes: {self.sample_sizes}")
        print(f"DGP types: {self.dgp_types}")
        print(f"Confidence levels: {self.confidence_levels}")
        print(f"Replications per config: {self.n_replications}")
        print(f"Bootstrap samples: {self.n_bootstrap}")
        print("=" * 60)
        
        all_results = []
        total_configs = len(self.sample_sizes) * len(self.dgp_types) * len(self.confidence_levels)
        config_counter = 0
        
        for n in self.sample_sizes:
            for dgp_type in self.dgp_types:
                for confidence_level in self.confidence_levels:
                    config_counter += 1
                    
                    print(f"\n📊 Configuration {config_counter}/{total_configs}")
                    print(f"   n={n}, DGP={dgp_type}, α={1-confidence_level:.2f}")
                    
                    config_results = []
                    
                    # Run replications for this configuration
                    for rep in range(self.n_replications):
                        if (rep + 1) % 10 == 0:
                            print(f"     Replication {rep + 1}/{self.n_replications}")
                        
                        result = self.run_single_coverage_test(
                            n, dgp_type, confidence_level, rep,
                            estimator_params, bootstrap_params
                        )
                        
                        config_results.append(result)
                        all_results.append(result)
                    
                    # Compute configuration summary
                    self._summarize_configuration(config_results, n, dgp_type, confidence_level)
        
        # Store all results
        self.coverage_results_ = all_results
        
        # Compute overall summary statistics
        self._compute_summary_statistics()
        
        # Save results if requested
        if self.save_results:
            self._save_coverage_results()
        
        print("\n✅ Coverage Study Completed!")
        self._print_summary()
        
        return {
            'coverage_results': self.coverage_results_,
            'summary_statistics': self.summary_statistics_,
            'study_parameters': {
                'sample_sizes': self.sample_sizes,
                'n_replications': self.n_replications,
                'n_bootstrap': self.n_bootstrap,
                'confidence_levels': self.confidence_levels,
                'dgp_types': self.dgp_types
            }
        }
    
    def _summarize_configuration(self, config_results: List[Dict], n: int, dgp_type: str, confidence_level: float):
        """Summarize results for a single configuration."""
        successful_results = [r for r in config_results if r['bootstrap_success']]
        success_rate = len(successful_results) / len(config_results)
        
        if len(successful_results) > 0:
            # Compute average coverage rates
            avg_coverage = {}
            avg_width = {}
            
            for var in ['X1', 'X2', 'X3']:
                coverage_vals = [r['coverage_rates'][var] for r in successful_results 
                               if not np.isnan(r['coverage_rates'][var])]
                width_vals = [r['interval_widths'][var] for r in successful_results 
                            if not np.isnan(r['interval_widths'][var])]
                
                avg_coverage[var] = np.mean(coverage_vals) if coverage_vals else np.nan
                avg_width[var] = np.mean(width_vals) if width_vals else np.nan
            
            print(f"     Success rate: {success_rate:.1%}")
            print(f"     Avg coverage: X1={avg_coverage['X1']:.3f}, X2={avg_coverage['X2']:.3f}, X3={avg_coverage['X3']:.3f}")
            print(f"     Nominal: {confidence_level:.3f}")
    
    def _compute_summary_statistics(self):
        """Compute comprehensive summary statistics."""
        self.summary_statistics_ = {}
        
        # Group by configuration
        for n in self.sample_sizes:
            for dgp_type in self.dgp_types:
                for confidence_level in self.confidence_levels:
                    config_key = f"n{n}_{dgp_type}_cl{confidence_level:.2f}"
                    
                    config_results = [
                        r for r in self.coverage_results_
                        if r['n'] == n and r['dgp_type'] == dgp_type and r['confidence_level'] == confidence_level
                    ]
                    
                    successful_results = [r for r in config_results if r['bootstrap_success']]
                    
                    summary = {
                        'n': n,
                        'dgp_type': dgp_type,
                        'confidence_level': confidence_level,
                        'n_replications': len(config_results),
                        'success_rate': len(successful_results) / len(config_results),
                        'coverage_statistics': {},
                        'width_statistics': {}
                    }
                    
                    for var in ['X1', 'X2', 'X3']:
                        # Coverage statistics
                        coverage_vals = [r['coverage_rates'][var] for r in successful_results 
                                       if not np.isnan(r['coverage_rates'][var])]
                        
                        if coverage_vals:
                            summary['coverage_statistics'][var] = {
                                'mean': np.mean(coverage_vals),
                                'std': np.std(coverage_vals),
                                'min': np.min(coverage_vals),
                                'max': np.max(coverage_vals),
                                'nominal': confidence_level,
                                'bias': np.mean(coverage_vals) - confidence_level
                            }
                        
                        # Width statistics
                        width_vals = [r['interval_widths'][var] for r in successful_results 
                                    if not np.isnan(r['interval_widths'][var])]
                        
                        if width_vals:
                            summary['width_statistics'][var] = {
                                'mean': np.mean(width_vals),
                                'std': np.std(width_vals),
                                'min': np.min(width_vals),
                                'max': np.max(width_vals)
                            }
                    
                    self.summary_statistics_[config_key] = summary
    
    def _save_coverage_results(self):
        """Save coverage study results to disk."""
        import pickle
        import json
        
        # Save raw results as pickle
        with open(self.results_dir / 'coverage_results.pkl', 'wb') as f:
            pickle.dump({
                'coverage_results': self.coverage_results_,
                'summary_statistics': self.summary_statistics_
            }, f)
        
        # Save summary as JSON
        json_summary = {}
        for key, summary in self.summary_statistics_.items():
            json_summary[key] = {
                k: v for k, v in summary.items() 
                if k not in ['coverage_statistics', 'width_statistics']
            }
            
            # Add simplified statistics
            json_summary[key]['coverage_means'] = {
                var: stats['mean'] for var, stats in summary['coverage_statistics'].items()
            } if summary['coverage_statistics'] else {}
            
            json_summary[key]['coverage_bias'] = {
                var: stats['bias'] for var, stats in summary['coverage_statistics'].items()
            } if summary['coverage_statistics'] else {}
        
        with open(self.results_dir / 'coverage_summary.json', 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        print(f"💾 Results saved to {self.results_dir}")
    
    def _print_summary(self):
        """Print summary of coverage study results."""
        print("\n📈 COVERAGE STUDY SUMMARY")
        print("=" * 50)
        
        for key, summary in self.summary_statistics_.items():
            print(f"\n{key}:")
            print(f"  Success rate: {summary['success_rate']:.1%}")
            
            if summary['coverage_statistics']:
                print("  Coverage rates (mean ± std):")
                for var, stats in summary['coverage_statistics'].items():
                    bias = stats['bias']
                    print(f"    {var}: {stats['mean']:.3f} ± {stats['std']:.3f} (bias: {bias:+.3f})")
    
    def plot_coverage_results(self, save_plots: bool = True):
        """Create coverage study plots matching R visualization."""
        if not self.summary_statistics_:
            print("No results to plot. Run coverage study first.")
            return
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plot 1: Coverage by sample size
        self._plot_coverage_by_sample_size(save_plots)
        
        # Plot 2: Coverage by confidence level
        self._plot_coverage_by_confidence_level(save_plots)
        
        # Plot 3: Interval widths
        self._plot_interval_widths(save_plots)
        
        if save_plots:
            print(f"📊 Plots saved to {self.results_dir}")
    
    def _plot_coverage_by_sample_size(self, save_plots: bool):
        """Plot coverage rates by sample size."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Coverage Rates by Sample Size', fontsize=16)
        
        for i, var in enumerate(['X1', 'X2', 'X3']):
            ax = axes[i]
            
            for dgp_type in self.dgp_types:
                for confidence_level in self.confidence_levels:
                    sample_sizes = []
                    coverage_means = []
                    coverage_stds = []
                    
                    for n in self.sample_sizes:
                        key = f"n{n}_{dgp_type}_cl{confidence_level:.2f}"
                        if key in self.summary_statistics_:
                            summary = self.summary_statistics_[key]
                            if var in summary['coverage_statistics']:
                                sample_sizes.append(n)
                                coverage_means.append(summary['coverage_statistics'][var]['mean'])
                                coverage_stds.append(summary['coverage_statistics'][var]['std'])
                    
                    if sample_sizes:
                        label = f"{dgp_type} (α={1-confidence_level:.2f})"
                        ax.errorbar(sample_sizes, coverage_means, yerr=coverage_stds, 
                                  marker='o', label=label, capsize=5)
            
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Nominal 0.95')
            ax.set_xlabel('Sample Size')
            ax.set_ylabel('Coverage Rate')
            ax.set_title(f'Variable {var}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'coverage_by_sample_size.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_coverage_by_confidence_level(self, save_plots: bool):
        """Plot coverage rates by confidence level."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Coverage Rates by Confidence Level', fontsize=16)
        
        for i, var in enumerate(['X1', 'X2', 'X3']):
            ax = axes[i]
            
            for dgp_type in self.dgp_types:
                for n in self.sample_sizes:
                    confidence_levels = []
                    coverage_means = []
                    
                    for confidence_level in self.confidence_levels:
                        key = f"n{n}_{dgp_type}_cl{confidence_level:.2f}"
                        if key in self.summary_statistics_:
                            summary = self.summary_statistics_[key]
                            if var in summary['coverage_statistics']:
                                confidence_levels.append(confidence_level)
                                coverage_means.append(summary['coverage_statistics'][var]['mean'])
                    
                    if confidence_levels:
                        label = f"{dgp_type} (n={n})"
                        ax.plot(confidence_levels, coverage_means, marker='o', label=label)
            
            # Add diagonal line (perfect coverage)
            ax.plot([0.9, 0.99], [0.9, 0.99], 'k--', alpha=0.5, label='Perfect Coverage')
            ax.set_xlabel('Nominal Coverage')
            ax.set_ylabel('Empirical Coverage')
            ax.set_title(f'Variable {var}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'coverage_by_confidence_level.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_interval_widths(self, save_plots: bool):
        """Plot confidence interval widths."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Confidence Interval Widths by Sample Size', fontsize=16)
        
        for i, var in enumerate(['X1', 'X2', 'X3']):
            ax = axes[i]
            
            for dgp_type in self.dgp_types:
                sample_sizes = []
                width_means = []
                
                for n in self.sample_sizes:
                    # Use first confidence level for width comparison
                    key = f"n{n}_{dgp_type}_cl{self.confidence_levels[0]:.2f}"
                    if key in self.summary_statistics_:
                        summary = self.summary_statistics_[key]
                        if var in summary['width_statistics']:
                            sample_sizes.append(n)
                            width_means.append(summary['width_statistics'][var]['mean'])
                
                if sample_sizes:
                    ax.plot(sample_sizes, width_means, marker='o', label=dgp_type)
            
            ax.set_xlabel('Sample Size')
            ax.set_ylabel('Average Interval Width')
            ax.set_title(f'Variable {var}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'interval_widths.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_quick_coverage_test():
    """Run a quick coverage test to validate the framework."""
    print("🚀 Running Quick Coverage Test...")
    
    study = CoverageStudy(
        sample_sizes=[200],
        n_replications=5,  # Small for testing
        n_bootstrap=100,  # Small for testing
        confidence_levels=[0.95],
        dgp_types=['additive']
    )
    
    results = study.run_coverage_study()
    
    print("✅ Quick test completed!")
    return results


if __name__ == "__main__":
    # Run quick test
    run_quick_coverage_test() 