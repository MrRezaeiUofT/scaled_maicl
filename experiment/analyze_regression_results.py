#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Regression Parameter Sweep Results
-------------------------------------------
Scans all result directories and creates CSV summary of MA-ICL performance.

Usage:
    # Analyze all results
    python analyze_regression_results.py
    
    # Analyze only gfp_yield results
    python analyze_regression_results.py --dataset gfp_yield
    
    # Analyze protein_expression plate 5 results
    python analyze_regression_results.py --dataset protein_expression --plate_index 5
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Results directory
RESULTS_ROOT = "ma_icl_results"

def parse_run_name(run_name):
    """Parse run name to extract parameters"""
    params = {
        'dataset': None,
        'plate_index': None,
        'ml_mech': None,
        'top_k': None,
        'iterations': None,
        'model_name': None,
        'regression_loss': None,
        'acceptance_set': None,
        'topk_strategy': None,
        'relax_eval': False,
        'num_mechanisms_unknown': 1,
        'k_shot': 0,
        'no_scaling': False,
        'val_size': 0.2,
        'tabicl_bins': None,
    }
    
    # Extract dataset and plate
    if 'protein_expression' in run_name:
        params['dataset'] = 'protein_expression'
        plate_match = re.search(r'plate(\d+)', run_name)
        if plate_match:
            params['plate_index'] = int(plate_match.group(1))
    elif 'gfp_yield' in run_name or 'gfp' in run_name.lower():
        params['dataset'] = 'gfp_yield'
        params['plate_index'] = None  # GFP yield doesn't have plates
    
    # Extract ML mechanism
    if 'mllinear' in run_name:
        params['ml_mech'] = 'linear'
    elif 'mlxgboost' in run_name:
        params['ml_mech'] = 'xgboost'
    elif 'mlkernelridge' in run_name:
        params['ml_mech'] = 'kernelridge'
    elif 'mltabicl' in run_name:
        params['ml_mech'] = 'tabicl'
    elif 'noML' in run_name:
        params['ml_mech'] = 'none'
    
    # Extract top_k
    topk_match = re.search(r'topk(\d+)', run_name)
    if topk_match:
        params['top_k'] = int(topk_match.group(1))
    
    # Extract iterations
    iter_match = re.search(r'iter(\d+)', run_name)
    if iter_match:
        params['iterations'] = int(iter_match.group(1))
    
    # Extract model name
    model_match = re.search(r'model([^_]+)', run_name)
    if model_match:
        params['model_name'] = model_match.group(1).replace('_', '-').replace('gemini', 'gemini')
    
    # Extract loss
    loss_match = re.search(r'loss(r2|mae)', run_name)
    if loss_match:
        params['regression_loss'] = loss_match.group(1)
    
    # Extract acceptance set
    accept_match = re.search(r'accept(test|validation|train)', run_name)
    if accept_match:
        params['acceptance_set'] = accept_match.group(1)
    
    # Extract topk strategy
    if 'topkstratresidual_balanced' in run_name:
        params['topk_strategy'] = 'residual_balanced'
    elif 'topkstratresidual' in run_name:
        params['topk_strategy'] = 'residual'
    
    # Extract flags
    if 'relax' in run_name:
        params['relax_eval'] = True
    
    if 'noscale' in run_name:
        params['no_scaling'] = True
    
    # Extract num_mechanisms_unknown
    nmech_match = re.search(r'nmech(\d+)', run_name)
    if nmech_match:
        params['num_mechanisms_unknown'] = int(nmech_match.group(1))
    
    # Extract k_shot
    kshot_match = re.search(r'kshot(\d+)', run_name)
    if kshot_match:
        params['k_shot'] = int(kshot_match.group(1))
    
    # Extract val_size
    val_match = re.search(r'val(\d+)_(\d+)', run_name)
    if val_match:
        params['val_size'] = float(f"{val_match.group(1)}.{val_match.group(2)}")
    
    # Extract tabicl_bins
    bins_match = re.search(r'bins(\d+)', run_name)
    if bins_match:
        params['tabicl_bins'] = int(bins_match.group(1))
    
    return params

def load_all_results(results_root=RESULTS_ROOT, dataset_filter=None, plate_index=None):
    """Load all final_results.json files from result directories
    
    Args:
        results_root: Root directory containing result directories
        dataset_filter: Optional dataset name to filter by (e.g., 'gfp_yield')
        plate_index: Optional plate index to filter by (for protein_expression)
    """
    results = []
    
    if not os.path.exists(results_root):
        logger.error(f"Results directory not found: {results_root}")
        return results
    
    # Find all directories in results root
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        if not os.path.isdir(item_path):
            continue
        
        # Skip current_run directory
        if item == "current_run":
            continue
        
        # Early filtering: check if directory name matches dataset filter
        if dataset_filter:
            dataset_filter_lower = dataset_filter.lower()
            item_lower = item.lower()
            
            # Check if this directory is for the specified dataset
            if dataset_filter_lower == 'gfp_yield':
                # Only load if it contains 'gfp_yield' or 'gfp' (but not other datasets)
                if 'gfp_yield' not in item_lower and 'gfp' not in item_lower:
                    continue
                # Make sure it's not another dataset
                if any(other_ds in item_lower for other_ds in ['protein_expression', 'diabetes', 'housing']):
                    continue
            elif dataset_filter_lower == 'protein_expression':
                # Only load if it contains 'protein_expression'
                if 'protein_expression' not in item_lower:
                    continue
                # If plate_index is specified, check it matches
                if plate_index is not None:
                    plate_match = re.search(r'plate(\d+)', item_lower)
                    if not plate_match or int(plate_match.group(1)) != plate_index:
                        continue
            else:
                # For other datasets, check if the dataset name is in the directory name
                if dataset_filter_lower not in item_lower:
                    continue
        
        # Look for final_results.json
        results_file = os.path.join(item_path, "final_results.json")
        if not os.path.exists(results_file):
            logger.debug(f"No final_results.json found in {item_path}")
            continue
        
        try:
            with open(results_file, 'r') as f:
                result_data = json.load(f)
            
            # Parse run name to extract parameters
            run_name = result_data.get('run_name', item)
            params = parse_run_name(run_name)
            
            # Additional filtering after parsing (double-check)
            if dataset_filter:
                dataset_filter_lower = dataset_filter.lower()
                result_dataset = (params.get('dataset') or '').lower()
                
                if dataset_filter_lower == 'gfp_yield':
                    if result_dataset != 'gfp_yield':
                        continue
                elif dataset_filter_lower == 'protein_expression':
                    if result_dataset != 'protein_expression':
                        continue
                    if plate_index is not None and params.get('plate_index') != plate_index:
                        continue
                else:
                    if dataset_filter_lower not in result_dataset and result_dataset not in dataset_filter_lower:
                        continue
            
            # Add result data
            result_entry = {
                'run_name': run_name,
                'directory': item_path,
                'parameters': params,
                'results': result_data
            }
            results.append(result_entry)
            logger.info(f"Loaded results from: {run_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {results_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(results)} result files")
    return results

def create_dataframe(results):
    """Create a pandas DataFrame from results"""
    rows = []
    
    for result_entry in results:
        params = result_entry['parameters']
        res = result_entry['results']
        
        row = params.copy()
        
        # Add metrics
        if 'ml_baseline' in res and res['ml_baseline']:
            row['ml_r2'] = res['ml_baseline'].get('r2', np.nan)
            row['ml_mae'] = res['ml_baseline'].get('mae', np.nan)
            row['ml_mse'] = res['ml_baseline'].get('mse', np.nan)
        else:
            row['ml_r2'] = np.nan
            row['ml_mae'] = np.nan
            row['ml_mse'] = np.nan
        
        if 'pre_training' in res:
            row['pre_r2'] = res['pre_training'].get('r2', np.nan)
            row['pre_mae'] = res['pre_training'].get('mae', np.nan)
            row['pre_mse'] = res['pre_training'].get('mse', np.nan)
        
        if 'post_training' in res:
            row['post_r2'] = res['post_training'].get('r2', np.nan)
            row['post_mae'] = res['post_training'].get('mae', np.nan)
            row['post_mse'] = res['post_training'].get('mse', np.nan)
            row['post_rmse'] = res['post_training'].get('rmse', np.nan)
        
        # LLM-only pre-training metrics
        if 'llm_only_pre' in res and res['llm_only_pre']:
            row['llm_only_pre_r2'] = res['llm_only_pre'].get('r2', np.nan)
            row['llm_only_pre_mae'] = res['llm_only_pre'].get('mae', np.nan)
            row['llm_only_pre_mse'] = res['llm_only_pre'].get('mse', np.nan)
            # Compute RMSE from MSE if not available
            rmse_pre = res['llm_only_pre'].get('rmse', np.nan)
            if np.isnan(rmse_pre) and not np.isnan(row['llm_only_pre_mse']):
                rmse_pre = np.sqrt(row['llm_only_pre_mse'])
            row['llm_only_pre_rmse'] = rmse_pre
        else:
            row['llm_only_pre_r2'] = np.nan
            row['llm_only_pre_mae'] = np.nan
            row['llm_only_pre_mse'] = np.nan
            row['llm_only_pre_rmse'] = np.nan
        
        # LLM-only post-training metrics
        if 'llm_only_post' in res and res['llm_only_post']:
            row['llm_only_post_r2'] = res['llm_only_post'].get('r2', np.nan)
            row['llm_only_post_mae'] = res['llm_only_post'].get('mae', np.nan)
            row['llm_only_post_mse'] = res['llm_only_post'].get('mse', np.nan)
            # Compute RMSE from MSE if not available
            rmse_post = res['llm_only_post'].get('rmse', np.nan)
            if np.isnan(rmse_post) and not np.isnan(row['llm_only_post_mse']):
                rmse_post = np.sqrt(row['llm_only_post_mse'])
            row['llm_only_post_rmse'] = rmse_post
        else:
            row['llm_only_post_r2'] = np.nan
            row['llm_only_post_mae'] = np.nan
            row['llm_only_post_mse'] = np.nan
            row['llm_only_post_rmse'] = np.nan
        
        if 'improvements' in res:
            if 'vs_ml_baseline_post' in res['improvements']:
                row['r2_improvement'] = res['improvements']['vs_ml_baseline_post'].get('r2_delta', np.nan)
                row['mae_improvement'] = res['improvements']['vs_ml_baseline_post'].get('mae_delta', np.nan)
                row['mse_improvement'] = res['improvements']['vs_ml_baseline_post'].get('mse_delta', np.nan)
            
            if 'training_improvement' in res['improvements']:
                row['training_r2_delta'] = res['improvements']['training_improvement'].get('r2_delta', np.nan)
                row['training_mae_delta'] = res['improvements']['training_improvement'].get('mae_delta', np.nan)
        
        row['run_name'] = result_entry['run_name']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def create_output_directory(dataset_name, plate_index=None):
    """Create output directory name based on dataset information"""
    # Use the specified dataset name (from args.dataset)
    dataset_info = []
    
    # Clean dataset name
    if dataset_name:
        # Remove common prefixes and clean up
        cleaned_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
        # Fix common typos
        cleaned_name = cleaned_name.replace('protien', 'protein')
        dataset_info.append(cleaned_name)
    else:
        dataset_info.append('unknown_dataset')
    
    # Add plate index if specified
    if plate_index is not None:
        dataset_info.append(f'plate{plate_index}')
    
    # Get date/time for uniqueness
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name
    dir_name = f"analysis_{'_'.join(dataset_info)}_{timestamp}"
    return dir_name


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze Regression Parameter Sweep Results")
    parser.add_argument("--dataset", type=str, default='diabetes',
                        help="Filter results by dataset name (e.g., 'gfp_yield', 'protein_expression'). "
                             "If not specified, analyzes all datasets.")
    parser.add_argument("--plate_index", type=int, default=None,
                        help="Filter results by plate index (only for protein_expression dataset)")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Analyzing Regression Parameter Sweep Results")
    if args.dataset:
        logger.info(f"Filtering for dataset: {args.dataset}")
        if args.plate_index is not None:
            logger.info(f"Filtering for plate_index: {args.plate_index}")
    logger.info("=" * 80)
    
    # Load results with dataset filter applied from the start
    results = load_all_results(dataset_filter=args.dataset, plate_index=args.plate_index)
    
    if len(results) == 0:
        logger.error("No results found. Make sure results are in ma_icl_results/")
        if args.dataset:
            logger.error(f"  for dataset '{args.dataset}'")
            if args.plate_index is not None:
                logger.error(f"  with plate_index {args.plate_index}")
        return 1
    
    if args.dataset:
        logger.info(f"Loaded {len(results)} results for dataset '{args.dataset}'")
        if args.plate_index is not None:
            logger.info(f"  with plate_index {args.plate_index}")
    
    # Create DataFrame
    df = create_dataframe(results)
    
    # Create output directory based on specified dataset name
    output_dir_name = create_output_directory(args.dataset, plate_index=args.plate_index)
    # Get absolute path
    output_dir = os.path.abspath(output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the full path clearly
    logger.info("=" * 80)
    logger.info(f"OUTPUT DIRECTORY (Full Path):")
    logger.info(f"  {output_dir}")
    logger.info("=" * 80)
    
    # Create CSV filename based on dataset
    # Always use the specified dataset name (or default) for the filename
    # This ensures the filename matches the dataset name exactly
    dataset_name_for_file = args.dataset.lower().replace(' ', '_').replace('-', '_')
    # Fix common typos
    dataset_name_for_file = dataset_name_for_file.replace('protien', 'protein')
    # Add plate index if specified
    if args.plate_index is not None:
        dataset_name_for_file = f"{dataset_name_for_file}_plate{args.plate_index}"
    logger.info(f"Using dataset name for CSV filename: {dataset_name_for_file}")
    
    # Create CSV filename
    csv_filename = f'regression_results_summary_{dataset_name_for_file}.csv'
    csv_file = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved results summary to: {csv_file}")
    
    # Save the output directory path to a file for easy reference
    path_file = 'last_analysis_output_path.txt'
    with open(path_file, 'w') as f:
        f.write(output_dir)
    logger.info(f"Output directory path saved to: {os.path.abspath(path_file)}")
    
    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    if 'post_r2' in df.columns:
        logger.info(f"R² Score Statistics:")
        logger.info(f"  Mean: {df['post_r2'].mean():.4f}")
        logger.info(f"  Std:  {df['post_r2'].std():.4f}")
        logger.info(f"  Min:  {df['post_r2'].min():.4f}")
        logger.info(f"  Max:  {df['post_r2'].max():.4f}")
    
    if 'post_mae' in df.columns:
        logger.info(f"\nMAE Statistics:")
        logger.info(f"  Mean: {df['post_mae'].mean():.4f}")
        logger.info(f"  Std:  {df['post_mae'].std():.4f}")
        logger.info(f"  Min:  {df['post_mae'].min():.4f}")
        logger.info(f"  Max:  {df['post_mae'].max():.4f}")
    
    logger.info("=" * 80)
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to:")
    logger.info(f"  {output_dir}")
    logger.info(f"")
    logger.info(f"File created:")
    logger.info(f"  - {csv_filename}")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

