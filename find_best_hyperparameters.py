#!/usr/bin/env python3
"""
Find the best hyperparameters for each dataset in ma_icl_results/BC_Gemini_Flash_2.
Extracts all results, groups by dataset, and finds the best performing configuration.
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_ROOT = "ma_icl_results/BC_Gemini_Flash_2"

def parse_run_name(run_name):
    """Parse run name to extract hyperparameters"""
    params = {
        'dataset': None,
        'plate_index': None,
        'ml_mech': None,
        'top_k': None,
        'iterations': None,
        'model_name': None,
        'loss': None,
        'acceptance_set': None,
        'topk_strategy': None,
        'num_mechanisms_unknown': 1,
        'no_scaling': False,
        'maxsamp': None,
        'evalindiv': False,
    }
    
    run_lower = run_name.lower()
    
    # Extract dataset
    if run_name.startswith('class_'):
        # Classification datasets
        if 'adult' in run_lower:
            params['dataset'] = 'adult'
        elif 'iris' in run_lower:
            params['dataset'] = 'iris'
        elif 'miceprotein' in run_lower or 'mice_protein' in run_lower:
            params['dataset'] = 'miceprotein'
        elif 'colon' in run_lower or 'colon-cancer' in run_lower:
            params['dataset'] = 'colon-cancer'
        elif 'zoo' in run_lower:
            params['dataset'] = 'zoo'
        elif 'ecoli' in run_lower:
            params['dataset'] = 'ecoli'
        elif 'lymphography' in run_lower:
            params['dataset'] = 'lymphography'
        elif 'soybean' in run_lower:
            params['dataset'] = 'soybean'
    elif run_name.startswith('bio_reg_'):
        # Regression datasets
        if 'diabetes' in run_lower:
            params['dataset'] = 'diabetes'
        elif 'gfp_yield' in run_lower or ('gfp' in run_lower and 'yield' in run_lower):
            params['dataset'] = 'gfp_yield'
        elif 'aminotransferase' in run_lower:
            params['dataset'] = 'aminotransferase'
        elif 'esol' in run_lower:
            params['dataset'] = 'esol'
        elif 'lipo' in run_lower:
            params['dataset'] = 'lipo'
        elif 'protein_expression' in run_lower:
            params['dataset'] = 'protein_expression'
            plate_match = re.search(r'plate(\d+)', run_lower)
            if plate_match:
                params['plate_index'] = int(plate_match.group(1))
    
    # Extract ML mechanism
    if 'mllinear' in run_lower:
        params['ml_mech'] = 'linear'
    elif 'mlxgboost' in run_lower:
        params['ml_mech'] = 'xgboost'
    elif 'mllogreg' in run_lower:
        params['ml_mech'] = 'logreg'
    
    # Extract top_k
    topk_match = re.search(r'topk(\d+)', run_lower)
    if topk_match:
        params['top_k'] = int(topk_match.group(1))
    
    # Extract iterations
    iter_match = re.search(r'iter(\d+)', run_lower)
    if iter_match:
        params['iterations'] = int(iter_match.group(1))
    
    # Extract model name
    model_match = re.search(r'model([^_]+)', run_lower)
    if model_match:
        model_str = model_match.group(1).replace('_', '-')
        params['model_name'] = model_str
    
    # Extract loss
    loss_match = re.search(r'loss(r2|mae|f1)', run_lower)
    if loss_match:
        params['loss'] = loss_match.group(1)
    
    # Extract acceptance set
    accept_match = re.search(r'accept(test|validation|train)', run_lower)
    if accept_match:
        params['acceptance_set'] = accept_match.group(1)
    
    # Extract topk strategy
    if 'topkstratresidual' in run_lower:
        params['topk_strategy'] = 'residual'
    
    # Extract flags
    if 'noscale' in run_lower:
        params['no_scaling'] = True
    
    # Extract num_mechanisms_unknown
    nmech_match = re.search(r'nmech(\d+)', run_lower)
    if nmech_match:
        params['num_mechanisms_unknown'] = int(nmech_match.group(1))
    
    # Extract maxsamp
    maxsamp_match = re.search(r'maxsamp(\d+)', run_lower)
    if maxsamp_match:
        params['maxsamp'] = int(maxsamp_match.group(1))
    
    # Extract evalindiv flag
    if 'evalindiv' in run_lower:
        params['evalindiv'] = True
    
    return params

def load_all_results(results_root=RESULTS_ROOT):
    """Load all final_results.json files from result directories"""
    results = []
    
    if not os.path.exists(results_root):
        logger.error(f"Results directory not found: {results_root}")
        return results
    
    # Find all directories in results root
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        if not os.path.isdir(item_path):
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
            
            # Skip if we couldn't identify the dataset
            if not params['dataset']:
                logger.debug(f"Could not identify dataset for: {run_name}")
                continue
            
            # Add result data
            result_entry = {
                'run_name': run_name,
                'directory': item_path,
                'parameters': params,
                'results': result_data
            }
            results.append(result_entry)
            
        except Exception as e:
            logger.warning(f"Failed to load {results_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(results)} result files")
    return results

def get_performance_metric(result_data, task_type):
    """Get the primary performance metric for ranking"""
    if task_type == 'classification':
        # Use F1 score as primary metric for classification
        if 'post_training' in result_data and result_data['post_training']:
            return result_data['post_training'].get('f1', np.nan)
    else:
        # Use R2 score as primary metric for regression
        if 'post_training' in result_data and result_data['post_training']:
            return result_data['post_training'].get('r2', np.nan)
    return np.nan

def detect_task_type(result_data, run_name):
    """Detect if this is a classification or regression task"""
    if run_name.startswith('class_'):
        return 'classification'
    elif run_name.startswith('bio_reg_'):
        return 'regression'
    
    # Check results structure
    if 'ml_baseline' in result_data and result_data['ml_baseline']:
        if 'accuracy' in result_data['ml_baseline'] or 'f1' in result_data['ml_baseline']:
            return 'classification'
        if 'r2' in result_data['ml_baseline'] or 'mae' in result_data['ml_baseline']:
            return 'regression'
    
    return 'regression'  # Default

def find_best_configurations(results):
    """Find the best configuration for each dataset"""
    # Group results by dataset
    dataset_results = defaultdict(list)
    
    for result_entry in results:
        dataset = result_entry['parameters']['dataset']
        if dataset:
            dataset_results[dataset].append(result_entry)
    
    best_configs = []
    
    for dataset, dataset_entries in dataset_results.items():
        logger.info(f"\nProcessing dataset: {dataset} ({len(dataset_entries)} configurations)")
        
        # Determine task type from first entry
        task_type = detect_task_type(dataset_entries[0]['results'], dataset_entries[0]['run_name'])
        
        # Find best configuration
        best_entry = None
        best_performance = -np.inf if task_type == 'regression' else -np.inf
        
        for entry in dataset_entries:
            performance = get_performance_metric(entry['results'], task_type)
            
            if not np.isnan(performance) and performance > best_performance:
                best_performance = performance
                best_entry = entry
        
        if best_entry:
            params = best_entry['parameters']
            res = best_entry['results']
            
            # Extract all relevant metrics
            config_info = {
                'dataset': dataset,
                'task_type': task_type,
                'run_name': best_entry['run_name'],
                'directory': best_entry['directory'],
                'ml_mech': params.get('ml_mech', ''),
                'top_k': params.get('top_k', ''),
                'iterations': params.get('iterations', ''),
                'model_name': params.get('model_name', ''),
                'loss': params.get('loss', ''),
                'acceptance_set': params.get('acceptance_set', ''),
                'topk_strategy': params.get('topk_strategy', ''),
                'num_mechanisms_unknown': params.get('num_mechanisms_unknown', 1),
                'no_scaling': params.get('no_scaling', False),
                'maxsamp': params.get('maxsamp', ''),
                'evalindiv': params.get('evalindiv', False),
            }
            
            # Add performance metrics
            if task_type == 'classification':
                config_info['best_metric'] = 'f1'
                config_info['best_f1'] = best_performance
                if 'post_training' in res:
                    config_info['post_accuracy'] = res['post_training'].get('accuracy', np.nan)
                    config_info['post_f1'] = res['post_training'].get('f1', np.nan)
                if 'ml_baseline' in res and res['ml_baseline']:
                    config_info['ml_accuracy'] = res['ml_baseline'].get('accuracy', np.nan)
                    config_info['ml_f1'] = res['ml_baseline'].get('f1', np.nan)
                if 'improvements' in res and 'vs_ml_baseline_post' in res['improvements']:
                    config_info['accuracy_improvement'] = res['improvements']['vs_ml_baseline_post'].get('acc_delta', np.nan)
                    config_info['f1_improvement'] = res['improvements']['vs_ml_baseline_post'].get('f1_delta', np.nan)
            else:
                config_info['best_metric'] = 'r2'
                config_info['best_r2'] = best_performance
                if 'post_training' in res:
                    config_info['post_r2'] = res['post_training'].get('r2', np.nan)
                    config_info['post_mae'] = res['post_training'].get('mae', np.nan)
                    config_info['post_mse'] = res['post_training'].get('mse', np.nan)
                if 'ml_baseline' in res and res['ml_baseline']:
                    config_info['ml_r2'] = res['ml_baseline'].get('r2', np.nan)
                    config_info['ml_mae'] = res['ml_baseline'].get('mae', np.nan)
                    config_info['ml_mse'] = res['ml_baseline'].get('mse', np.nan)
                if 'improvements' in res and 'vs_ml_baseline_post' in res['improvements']:
                    config_info['r2_improvement'] = res['improvements']['vs_ml_baseline_post'].get('r2_delta', np.nan)
                    config_info['mae_improvement'] = res['improvements']['vs_ml_baseline_post'].get('mae_delta', np.nan)
            
            best_configs.append(config_info)
            
            logger.info(f"  Best configuration: {best_entry['run_name']}")
            if task_type == 'classification':
                logger.info(f"    F1 Score: {best_performance:.4f}")
            else:
                logger.info(f"    R2 Score: {best_performance:.4f}")
        else:
            logger.warning(f"  No valid configuration found for {dataset}")
    
    return best_configs

def main():
    global RESULTS_ROOT
    
    parser = argparse.ArgumentParser(
        description="Find best hyperparameters for each dataset in BC_Gemini_Flash_2 results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path (default: saves in results directory)")
    parser.add_argument("--results-dir", type=str, default=RESULTS_ROOT,
                        help=f"Results directory (default: {RESULTS_ROOT})")
    
    args = parser.parse_args()
    
    RESULTS_ROOT = args.results_dir
    
    # Set default output path to results directory if not specified
    if args.output is None:
        output_path = os.path.join(RESULTS_ROOT, "best_hyperparameters_summary.csv")
    else:
        output_path = args.output
    
    logger.info("=" * 80)
    logger.info("Finding Best Hyperparameters for Each Dataset")
    logger.info(f"Results directory: {RESULTS_ROOT}")
    logger.info("=" * 80)
    
    # Load all results
    results = load_all_results()
    
    if len(results) == 0:
        logger.error(f"No results found in {RESULTS_ROOT}")
        return 1
    
    # Find best configurations
    best_configs = find_best_configurations(results)
    
    if len(best_configs) == 0:
        logger.error("No valid configurations found")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(best_configs)
    
    # Sort by dataset name
    df = df.sort_values('dataset')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved best configurations to: {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF BEST CONFIGURATIONS")
    logger.info("=" * 80)
    
    for _, row in df.iterrows():
        logger.info(f"\nDataset: {row['dataset']} ({row['task_type']})")
        logger.info(f"  Run name: {row['run_name']}")
        logger.info(f"  ML Mechanism: {row['ml_mech']}")
        logger.info(f"  Top K: {row['top_k']}")
        logger.info(f"  Iterations: {row['iterations']}")
        logger.info(f"  Acceptance Set: {row['acceptance_set']}")
        logger.info(f"  No Scaling: {row['no_scaling']}")
        logger.info(f"  Num Mechanisms Unknown: {row['num_mechanisms_unknown']}")
        if row['task_type'] == 'classification':
            logger.info(f"  Best F1: {row['post_f1']:.4f}")
            logger.info(f"  Post Accuracy: {row['post_accuracy']:.4f}")
            logger.info(f"  ML Baseline F1: {row['ml_f1']:.4f}")
            logger.info(f"  F1 Improvement: {row['f1_improvement']:.4f}")
        else:
            logger.info(f"  Best R2: {row['post_r2']:.4f}")
            logger.info(f"  Post MAE: {row['post_mae']:.4f}")
            logger.info(f"  ML Baseline R2: {row['ml_r2']:.4f}")
            logger.info(f"  R2 Improvement: {row['r2_improvement']:.4f}")
    
    logger.info("\n" + "=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(main())

