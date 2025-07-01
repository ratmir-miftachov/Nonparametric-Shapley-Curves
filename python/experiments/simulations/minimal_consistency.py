"""
Minimal consistency study for direct comparison with R.
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import logging
from typing import Dict, List, Tuple, Any

# Add python project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.algorithms.se_vector_functions import compute_ise_results
from src.algorithms.advanced_integration import PopulationShapleyIntegrator
from experiments.simulations.consistency_study import ImprovedConsistencyStudy


def run_minimal_consistency_test():
    """
    Run a single iteration of the consistency study for comparison.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("🔬 MINIMAL CONSISTENCY TEST (PYTHON)")
    
    study = ImprovedConsistencyStudy(
        sample_sizes=[300], 
        n_monte_carlo=1
    )
    
    logging.info("Running single ISE calculation with seed 42. This may take a few minutes...")
    # Run a single ISE calculation with a fixed seed
    # The seed is set inside ise_function as 42 + monte_carlo_id
    # Since monte_carlo_id is 0, the seed will be 42.
    start_time = time.time()
    results = study.ise_function(monte_carlo_id=0, n_samples=300)
    end_time = time.time()
    logging.info(f"Calculation finished in {end_time - start_time:.2f} seconds.")
    
    logging.info("--- Python Results ---")
    logging.info(f"ISE1 (Comp): {results[0]:.6f}, ISE1 (Int): {results[1]:.6f}")
    logging.info(f"ISE2 (Comp): {results[2]:.6f}, ISE2 (Int): {results[3]:.6f}")
    logging.info(f"ISE3 (Comp): {results[4]:.6f}, ISE3 (Int): {results[5]:.6f}")
    
    # For direct comparison with R's output vector
    logging.info("Raw Python output vector:")
    logging.info(results)


if __name__ == "__main__":
    run_minimal_consistency_test() 