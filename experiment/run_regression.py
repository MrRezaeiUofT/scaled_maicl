#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Regression Parameter Sweep
-------------------------------
Runs 018_maicl_regression_biotech.py with different parameter combinations.

Combinations:
- top_k: 10, 30, 50, 75, 100
- ml_mech: linear, xgboost
- scaling: scaling (default), no_scaling
- acceptance_set: test, validation
- num_mechanisms_unknown: 1, 2, 3

Total: 5 * 2 * 2 * 2 * 3 = 120 combinations
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product
import logging

# Base command arguments (fixed for all runs)
BASE_ARGS = {
    'dataset': 'avila',
    'plate_index': 5,
    'iterations': 10,
    'regression_loss': 'r2',
    'max_samples': 500,
    'evaluate_individual_mechanisms': True,
    'use_ml': 1,
    'k_shot': 0,
}

# Parameter combinations to test
PARAM_COMBINATIONS = {
    'top_k': [100, 200, 300, 400],
    'ml_mech': ['linear', 'xgboost'],
    'scaling': ['scaling', 'no_scaling'],  # 'scaling' means default (scaled), 'no_scaling' means --no_scaling flag
    'acceptance_set': ['test', 'validation'],
    'num_mechanisms_unknown': [1, 2],
}

# Generate log file name with dataset extension
LOG_FILE = f'run_regression_sweep_{BASE_ARGS["dataset"]}.log'

# Setup logging with dataset-specific log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def build_command(args_dict):
    """Build command list for subprocess.run"""
    cmd = ['python', '018_maicl_regression_biotech.py']
    
    # Add base arguments
    for key, value in BASE_ARGS.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    # Add parameter combination arguments
    for key, value in args_dict.items():
        if key == 'scaling':
            if value == 'no_scaling':
                cmd.append('--no_scaling')
            # If 'scaling', don't add anything (default behavior)
        elif isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    return cmd

def run_single_combination(combo_dict, combo_num, total_combos):
    """Run a single parameter combination"""
    logger.info("=" * 80)
    logger.info(f"Running combination {combo_num}/{total_combos}")
    logger.info(f"Parameters: {json.dumps(combo_dict, indent=2)}")
    logger.info("=" * 80)
    
    cmd = build_command(combo_dict)
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            timeout=3600  # 1 hour timeout per run
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully completed in {elapsed_time:.2f} seconds")
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'combo_num': combo_num,
                'parameters': combo_dict.copy(),
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,  # Last 1000 chars
                'stderr': result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            }
        else:
            logger.error(f"✗ Failed with exit code {result.returncode} after {elapsed_time:.2f} seconds")
            logger.error(f"Error output: {result.stderr[-500:] if result.stderr else 'No error output'}")
            return {
                'status': 'failed',
                'exit_code': result.returncode,
                'elapsed_time': elapsed_time,
                'combo_num': combo_num,
                'parameters': combo_dict.copy(),
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                'stderr': result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ Timeout after {elapsed_time:.2f} seconds")
        return {
            'status': 'timeout',
            'elapsed_time': elapsed_time,
            'combo_num': combo_num,
            'parameters': combo_dict.copy(),
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ Exception: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time,
            'combo_num': combo_num,
            'parameters': combo_dict.copy(),
        }

def main():
    """Main function to run all parameter combinations"""
    logger.info("=" * 80)
    logger.info("Starting Regression Parameter Sweep")
    logger.info("=" * 80)
    logger.info(f"Base arguments: {json.dumps(BASE_ARGS, indent=2)}")
    logger.info(f"Parameter combinations: {json.dumps(PARAM_COMBINATIONS, indent=2)}")
    
    # Generate all combinations
    param_keys = list(PARAM_COMBINATIONS.keys())
    param_values = list(PARAM_COMBINATIONS.values())
    all_combinations = []
    
    for combo in product(*param_values):
        combo_dict = dict(zip(param_keys, combo))
        all_combinations.append(combo_dict)
    
    total_combos = len(all_combinations)
    logger.info(f"Total combinations to run: {total_combos}")
    logger.info("=" * 80)
    
    # Results tracking
    results = {
        'start_time': datetime.now().isoformat(),
        'base_args': BASE_ARGS,
        'param_combinations': PARAM_COMBINATIONS,
        'total_combinations': total_combos,
        'runs': []
    }
    
    # Run each combination
    successful_runs = 0
    failed_runs = 0
    timeout_runs = 0
    error_runs = 0
    
    for idx, combo_dict in enumerate(all_combinations, 1):
        result = run_single_combination(combo_dict, idx, total_combos)
        results['runs'].append(result)
        
        if result['status'] == 'success':
            successful_runs += 1
        elif result['status'] == 'timeout':
            timeout_runs += 1
            failed_runs += 1
        elif result['status'] == 'error':
            error_runs += 1
            failed_runs += 1
        else:
            failed_runs += 1
        
        # Save intermediate results after each run
        results_file = 'run_regression_sweep_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Progress update every 10 runs
        if idx % 10 == 0:
            logger.info(f"\nProgress: {idx}/{total_combos} ({idx/total_combos*100:.1f}%) | "
                       f"Success: {successful_runs} | Failed: {failed_runs}\n")
        
        # Brief pause between runs to avoid overwhelming the system
        time.sleep(2)
    
    # Final summary
    results['end_time'] = datetime.now().isoformat()
    results['successful_runs'] = successful_runs
    results['failed_runs'] = failed_runs
    results['timeout_runs'] = timeout_runs
    results['error_runs'] = error_runs
    
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total combinations: {total_combos}")
    logger.info(f"Successful: {successful_runs}")
    logger.info(f"Failed: {failed_runs} (Timeouts: {timeout_runs}, Errors: {error_runs})")
    logger.info(f"Success rate: {successful_runs/total_combos*100:.1f}%")
    
    # Save final results
    results_file = 'run_regression_sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Generate summary CSV
    try:
        import pandas as pd
        
        summary_data = []
        for run in results['runs']:
            row = run['parameters'].copy()
            row['status'] = run['status']
            row['elapsed_time'] = run.get('elapsed_time', None)
            row['combo_num'] = run.get('combo_num', None)
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        csv_file = 'run_regression_sweep_summary.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Summary CSV saved to: {csv_file}")
    except ImportError:
        logger.warning("pandas not available, skipping CSV summary")
    except Exception as e:
        logger.warning(f"Failed to generate CSV summary: {e}")
    
    return 0 if failed_runs == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

