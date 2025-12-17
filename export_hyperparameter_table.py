#!/usr/bin/env python3
"""
Export hyperparameter search results to CSV table.
Searches ma_icl_results directory for all runs matching a dataset name and exports to CSV.
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from datetime import datetime

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
    
    # Extract dataset - handle various naming patterns
    run_lower = run_name.lower()
    
    # Check for common dataset patterns
    if 'protein_expression' in run_lower:
        params['dataset'] = 'protein_expression'
        plate_match = re.search(r'plate(\d+)', run_lower)
        if plate_match:
            params['plate_index'] = int(plate_match.group(1))
    elif 'gfp_yield' in run_lower or ('gfp' in run_lower and 'yield' in run_lower):
        params['dataset'] = 'gfp_yield'
    elif 'diabetes' in run_lower:
        params['dataset'] = 'diabetes'
    elif 'aminotransferase' in run_lower:
        params['dataset'] = 'aminotransferase'
    elif 'halogenase' in run_lower:
        params['dataset'] = 'halogenase'
    elif 'olea' in run_lower:
        params['dataset'] = 'olea'
    elif 'lipo' in run_lower:
        params['dataset'] = 'lipo'
    elif 'zoo' in run_lower:
        params['dataset'] = 'zoo'
    elif 'colon' in run_lower or 'colon-cancer' in run_lower:
        params['dataset'] = 'colon-cancer'
    
    # Extract ML mechanism
    if 'mllinear' in run_lower:
        params['ml_mech'] = 'linear'
    elif 'mlxgboost' in run_lower:
        params['ml_mech'] = 'xgboost'
    elif 'mllogreg' in run_lower:
        params['ml_mech'] = 'logreg'
    elif 'mlkernelridge' in run_lower:
        params['ml_mech'] = 'kernelridge'
    elif 'mltabicl' in run_lower:
        params['ml_mech'] = 'tabicl'
    elif 'noml' in run_lower:
        params['ml_mech'] = 'none'
    
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
        params['regression_loss'] = loss_match.group(1)
    
    # Extract acceptance set
    accept_match = re.search(r'accept(test|validation|train)', run_lower)
    if accept_match:
        params['acceptance_set'] = accept_match.group(1)
    
    # Extract topk strategy
    if 'topkstratresidual_balanced' in run_lower:
        params['topk_strategy'] = 'residual_balanced'
    elif 'topkstratresidual' in run_lower:
        params['topk_strategy'] = 'residual'
    
    # Extract flags
    if 'relax' in run_lower:
        params['relax_eval'] = True
    
    if 'noscale' in run_lower:
        params['no_scaling'] = True
    
    # Extract num_mechanisms_unknown
    nmech_match = re.search(r'nmech(\d+)', run_lower)
    if nmech_match:
        params['num_mechanisms_unknown'] = int(nmech_match.group(1))
    
    # Extract k_shot
    kshot_match = re.search(r'kshot(\d+)', run_lower)
    if kshot_match:
        params['k_shot'] = int(kshot_match.group(1))
    
    # Extract val_size
    val_match = re.search(r'val(\d+)_(\d+)', run_lower)
    if val_match:
        params['val_size'] = float(f"{val_match.group(1)}.{val_match.group(2)}")
    
    # Extract tabicl_bins
    bins_match = re.search(r'bins(\d+)', run_lower)
    if bins_match:
        params['tabicl_bins'] = int(bins_match.group(1))
    
    return params

def load_all_results(results_root=RESULTS_ROOT, dataset_filter=None):
    """Load all final_results.json files from result directories matching dataset name
    
    Args:
        results_root: Root directory containing result directories
        dataset_filter: Dataset name to filter by (e.g., 'diabetes', 'gfp_yield')
    """
    results = []
    
    if not os.path.exists(results_root):
        logger.error(f"Results directory not found: {results_root}")
        return results
    
    dataset_filter_lower = dataset_filter.lower() if dataset_filter else None
    
    # Find all directories in results root
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        if not os.path.isdir(item_path):
            continue
        
        # Skip current_run directory
        if item == "current_run":
            continue
        
        # Early filtering: check if directory name matches dataset filter
        if dataset_filter_lower:
            item_lower = item.lower()
            
            # Check if this directory is for the specified dataset
            # Handle various naming patterns
            matches = False
            if dataset_filter_lower == 'diabetes':
                matches = 'diabetes' in item_lower
            elif dataset_filter_lower == 'gfp_yield' or dataset_filter_lower == 'gfp':
                matches = ('gfp' in item_lower and 'yield' in item_lower) or ('gfp_yield' in item_lower)
            elif dataset_filter_lower == 'protein_expression':
                matches = 'protein_expression' in item_lower
            elif dataset_filter_lower == 'aminotransferase':
                matches = 'aminotransferase' in item_lower
            elif dataset_filter_lower == 'halogenase':
                matches = 'halogenase' in item_lower
            elif dataset_filter_lower == 'olea':
                matches = 'olea' in item_lower
            elif dataset_filter_lower == 'lipo':
                matches = 'lipo' in item_lower
            elif dataset_filter_lower == 'zoo':
                matches = 'zoo' in item_lower and 'class_' in item_lower
            elif dataset_filter_lower in ['colon', 'colon-cancer', 'colon_cancer']:
                matches = 'colon' in item_lower
            else:
                # Generic match
                matches = dataset_filter_lower in item_lower
            
            if not matches:
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
            if dataset_filter_lower:
                result_dataset = (params.get('dataset') or '').lower()
                
                if dataset_filter_lower == 'diabetes':
                    if result_dataset != 'diabetes':
                        continue
                elif dataset_filter_lower in ['gfp_yield', 'gfp']:
                    if result_dataset != 'gfp_yield':
                        continue
                elif dataset_filter_lower == 'protein_expression':
                    if result_dataset != 'protein_expression':
                        continue
                elif dataset_filter_lower == 'aminotransferase':
                    if result_dataset != 'aminotransferase':
                        continue
                elif dataset_filter_lower == 'halogenase':
                    if result_dataset != 'halogenase':
                        continue
                elif dataset_filter_lower == 'olea':
                    if result_dataset != 'olea':
                        continue
                elif dataset_filter_lower == 'lipo':
                    if result_dataset != 'lipo':
                        continue
                elif dataset_filter_lower == 'zoo':
                    if result_dataset != 'zoo':
                        continue
                elif dataset_filter_lower in ['colon', 'colon-cancer', 'colon_cancer']:
                    if result_dataset not in ['colon', 'colon-cancer', 'colon_cancer']:
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

def detect_task_type(res, run_name):
    """Detect if this is a classification or regression task"""
    # Check run name for classification prefix
    if run_name.startswith('class_'):
        return 'classification'
    
    # Check if results have classification metrics
    if 'ml_baseline' in res and res['ml_baseline']:
        if 'accuracy' in res['ml_baseline'] or 'f1' in res['ml_baseline']:
            return 'classification'
        if 'r2' in res['ml_baseline'] or 'mae' in res['ml_baseline']:
            return 'regression'
    
    # Default to regression for backward compatibility
    return 'regression'

def create_dataframe(results):
    """Create a pandas DataFrame from results matching the CSV format"""
    rows = []
    
    for result_entry in results:
        params = result_entry['parameters']
        res = result_entry['results']
        run_name = result_entry['run_name']
        
        # Detect task type
        task_type = detect_task_type(res, run_name)
        
        # Start with parameter columns
        # Try to get k_shot from results JSON first, then from params
        k_shot_value = res.get('k_shot', params.get('k_shot', 0))
        if k_shot_value is None:
            k_shot_value = 0
        
        row = {
            'task_type': task_type,  # Add task type flag
            'dataset': params.get('dataset', ''),
            'plate_index': params.get('plate_index', ''),
            'ml_mech': params.get('ml_mech', ''),
            'top_k': params.get('top_k', ''),
            'iterations': params.get('iterations', ''),
            'model_name': params.get('model_name', ''),
            'regression_loss': params.get('regression_loss', ''),
            'acceptance_set': params.get('acceptance_set', ''),
            'topk_strategy': params.get('topk_strategy', ''),
            'relax_eval': params.get('relax_eval', False),
            'num_mechanisms_unknown': params.get('num_mechanisms_unknown', 1),
            'k_shot': k_shot_value,
            'no_scaling': params.get('no_scaling', False),
            'val_size': params.get('val_size', 0.2),
            'tabicl_bins': params.get('tabicl_bins', ''),
        }
        
        if task_type == 'classification':
            # CLASSIFICATION METRICS
            
            # ML baseline metrics
            if 'ml_baseline' in res and res['ml_baseline']:
                row['ml_accuracy'] = res['ml_baseline'].get('accuracy', np.nan)
                row['ml_f1'] = res['ml_baseline'].get('f1', np.nan)
                # Precision and recall may not be available
                row['ml_precision'] = res['ml_baseline'].get('precision', '')
                row['ml_recall'] = res['ml_baseline'].get('recall', '')
            else:
                row['ml_accuracy'] = np.nan
                row['ml_f1'] = np.nan
                row['ml_precision'] = ''
                row['ml_recall'] = ''
            
            # Pre-training metrics
            if 'pre_training' in res:
                row['pre_accuracy'] = res['pre_training'].get('accuracy', np.nan)
                row['pre_f1'] = res['pre_training'].get('f1', np.nan)
                row['pre_precision'] = res['pre_training'].get('precision', '')
                row['pre_recall'] = res['pre_training'].get('recall', '')
            else:
                row['pre_accuracy'] = np.nan
                row['pre_f1'] = np.nan
                row['pre_precision'] = ''
                row['pre_recall'] = ''
            
            # Post-training metrics
            if 'post_training' in res:
                row['post_accuracy'] = res['post_training'].get('accuracy', np.nan)
                row['post_f1'] = res['post_training'].get('f1', np.nan)
                row['post_precision'] = res['post_training'].get('precision', '')
                row['post_recall'] = res['post_training'].get('recall', '')
            else:
                row['post_accuracy'] = np.nan
                row['post_f1'] = np.nan
                row['post_precision'] = ''
                row['post_recall'] = ''
            
            # LLM-only pre-training metrics (classification uses llm_only_pre_training)
            if 'llm_only_pre_training' in res and res['llm_only_pre_training']:
                llm_pre = res['llm_only_pre_training']
                row['llm_only_pre_accuracy'] = llm_pre.get('accuracy', np.nan)
                row['llm_only_pre_f1'] = llm_pre.get('f1', np.nan)
                row['llm_only_pre_precision'] = llm_pre.get('precision', '')
                row['llm_only_pre_recall'] = llm_pre.get('recall', '')
            else:
                row['llm_only_pre_accuracy'] = np.nan
                row['llm_only_pre_f1'] = np.nan
                row['llm_only_pre_precision'] = ''
                row['llm_only_pre_recall'] = ''
            
            # LLM-only post-training metrics (classification uses llm_only_post_training)
            if 'llm_only_post_training' in res and res['llm_only_post_training']:
                llm_post = res['llm_only_post_training']
                row['llm_only_post_accuracy'] = llm_post.get('accuracy', np.nan)
                row['llm_only_post_f1'] = llm_post.get('f1', np.nan)
                row['llm_only_post_precision'] = llm_post.get('precision', '')
                row['llm_only_post_recall'] = llm_post.get('recall', '')
            else:
                row['llm_only_post_accuracy'] = np.nan
                row['llm_only_post_f1'] = np.nan
                row['llm_only_post_precision'] = ''
                row['llm_only_post_recall'] = ''
            
            # Add improvements
            if 'improvements' in res:
                if 'vs_ml_baseline_post' in res['improvements']:
                    row['accuracy_improvement'] = res['improvements']['vs_ml_baseline_post'].get('acc_delta', np.nan)
                    row['f1_improvement'] = res['improvements']['vs_ml_baseline_post'].get('f1_delta', np.nan)
                else:
                    row['accuracy_improvement'] = np.nan
                    row['f1_improvement'] = np.nan
                
                if 'training_improvement' in res['improvements']:
                    row['training_accuracy_delta'] = res['improvements']['training_improvement'].get('acc_delta', np.nan)
                    row['training_f1_delta'] = res['improvements']['training_improvement'].get('f1_delta', np.nan)
                else:
                    row['training_accuracy_delta'] = np.nan
                    row['training_f1_delta'] = np.nan
            else:
                row['accuracy_improvement'] = np.nan
                row['f1_improvement'] = np.nan
                row['training_accuracy_delta'] = np.nan
                row['training_f1_delta'] = np.nan
            
            # Set regression metrics to NaN for classification
            row['ml_r2'] = np.nan
            row['ml_mae'] = np.nan
            row['ml_mse'] = np.nan
            row['pre_r2'] = np.nan
            row['pre_mae'] = np.nan
            row['pre_mse'] = np.nan
            row['post_r2'] = np.nan
            row['post_mae'] = np.nan
            row['post_mse'] = np.nan
            row['post_rmse'] = np.nan
            row['llm_only_pre_r2'] = np.nan
            row['llm_only_pre_mae'] = np.nan
            row['llm_only_pre_mse'] = np.nan
            row['llm_only_pre_rmse'] = np.nan
            row['llm_only_post_r2'] = np.nan
            row['llm_only_post_mae'] = np.nan
            row['llm_only_post_mse'] = np.nan
            row['llm_only_post_rmse'] = np.nan
            row['r2_improvement'] = np.nan
            row['mae_improvement'] = np.nan
            row['mse_improvement'] = np.nan
            row['training_r2_delta'] = np.nan
            row['training_mae_delta'] = np.nan
            
        else:
            # REGRESSION METRICS (original code)
            
            # Add ML baseline metrics
            if 'ml_baseline' in res and res['ml_baseline']:
                row['ml_r2'] = res['ml_baseline'].get('r2', np.nan)
                row['ml_mae'] = res['ml_baseline'].get('mae', np.nan)
                row['ml_mse'] = res['ml_baseline'].get('mse', np.nan)
            else:
                row['ml_r2'] = np.nan
                row['ml_mae'] = np.nan
                row['ml_mse'] = np.nan
            
            # Add pre-training metrics
            if 'pre_training' in res:
                row['pre_r2'] = res['pre_training'].get('r2', np.nan)
                row['pre_mae'] = res['pre_training'].get('mae', np.nan)
                row['pre_mse'] = res['pre_training'].get('mse', np.nan)
            else:
                row['pre_r2'] = np.nan
                row['pre_mae'] = np.nan
                row['pre_mse'] = np.nan
            
            # Add post-training metrics
            if 'post_training' in res:
                row['post_r2'] = res['post_training'].get('r2', np.nan)
                row['post_mae'] = res['post_training'].get('mae', np.nan)
                row['post_mse'] = res['post_training'].get('mse', np.nan)
                row['post_rmse'] = res['post_training'].get('rmse', np.nan)
            else:
                row['post_r2'] = np.nan
                row['post_mae'] = np.nan
                row['post_mse'] = np.nan
                row['post_rmse'] = np.nan
            
            # LLM-only pre-training metrics (regression uses llm_only_pre)
            if 'llm_only_pre' in res and res['llm_only_pre']:
                llm_pre = res['llm_only_pre']
                row['llm_only_pre_r2'] = llm_pre.get('r2', np.nan)
                row['llm_only_pre_mae'] = llm_pre.get('mae', np.nan)
                row['llm_only_pre_mse'] = llm_pre.get('mse', np.nan)
                rmse_pre = llm_pre.get('rmse', np.nan)
                if np.isnan(rmse_pre) and not np.isnan(row['llm_only_pre_mse']):
                    rmse_pre = np.sqrt(row['llm_only_pre_mse'])
                row['llm_only_pre_rmse'] = rmse_pre
            else:
                row['llm_only_pre_r2'] = np.nan
                row['llm_only_pre_mae'] = np.nan
                row['llm_only_pre_mse'] = np.nan
                row['llm_only_pre_rmse'] = np.nan
            
            # LLM-only post-training metrics (regression uses llm_only_post)
            if 'llm_only_post' in res and res['llm_only_post']:
                llm_post = res['llm_only_post']
                row['llm_only_post_r2'] = llm_post.get('r2', np.nan)
                row['llm_only_post_mae'] = llm_post.get('mae', np.nan)
                row['llm_only_post_mse'] = llm_post.get('mse', np.nan)
                rmse_post = llm_post.get('rmse', np.nan)
                if np.isnan(rmse_post) and not np.isnan(row['llm_only_post_mse']):
                    rmse_post = np.sqrt(row['llm_only_post_mse'])
                row['llm_only_post_rmse'] = rmse_post
            else:
                row['llm_only_post_r2'] = np.nan
                row['llm_only_post_mae'] = np.nan
                row['llm_only_post_mse'] = np.nan
                row['llm_only_post_rmse'] = np.nan
            
            # Add improvements
            if 'improvements' in res:
                if 'vs_ml_baseline_post' in res['improvements']:
                    row['r2_improvement'] = res['improvements']['vs_ml_baseline_post'].get('r2_delta', np.nan)
                    row['mae_improvement'] = res['improvements']['vs_ml_baseline_post'].get('mae_delta', np.nan)
                    row['mse_improvement'] = res['improvements']['vs_ml_baseline_post'].get('mse_delta', np.nan)
                else:
                    row['r2_improvement'] = np.nan
                    row['mae_improvement'] = np.nan
                    row['mse_improvement'] = np.nan
                
                if 'training_improvement' in res['improvements']:
                    row['training_r2_delta'] = res['improvements']['training_improvement'].get('r2_delta', np.nan)
                    row['training_mae_delta'] = res['improvements']['training_improvement'].get('mae_delta', np.nan)
                else:
                    row['training_r2_delta'] = np.nan
                    row['training_mae_delta'] = np.nan
            else:
                row['r2_improvement'] = np.nan
                row['mae_improvement'] = np.nan
                row['mse_improvement'] = np.nan
                row['training_r2_delta'] = np.nan
                row['training_mae_delta'] = np.nan
            
            # Set classification metrics to NaN for regression
            row['ml_accuracy'] = np.nan
            row['ml_f1'] = np.nan
            row['ml_precision'] = np.nan
            row['ml_recall'] = np.nan
            row['pre_accuracy'] = np.nan
            row['pre_f1'] = np.nan
            row['pre_precision'] = np.nan
            row['pre_recall'] = np.nan
            row['post_accuracy'] = np.nan
            row['post_f1'] = np.nan
            row['post_precision'] = np.nan
            row['post_recall'] = np.nan
            row['llm_only_pre_accuracy'] = np.nan
            row['llm_only_pre_f1'] = np.nan
            row['llm_only_pre_precision'] = np.nan
            row['llm_only_pre_recall'] = np.nan
            row['llm_only_post_accuracy'] = np.nan
            row['llm_only_post_f1'] = np.nan
            row['llm_only_post_precision'] = np.nan
            row['llm_only_post_recall'] = np.nan
            row['accuracy_improvement'] = np.nan
            row['f1_improvement'] = np.nan
            row['training_accuracy_delta'] = np.nan
            row['training_f1_delta'] = np.nan
        
        row['run_name'] = result_entry['run_name']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def create_output_directory(dataset_name):
    """Create output directory name based on dataset information"""
    # Clean dataset name
    cleaned_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
    cleaned_name = cleaned_name.replace('protien', 'protein')
    
    # Get date/time for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name
    dir_name = f"analysis_{cleaned_name}_{timestamp}"
    return dir_name

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Export hyperparameter search results from ma_icl_results to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_hyperparameter_table.py diabetes
  python export_hyperparameter_table.py gfp_yield
  python export_hyperparameter_table.py aminotransferase
        """
    )
    parser.add_argument("dataset", type=str,
                        help="Dataset name to search for (e.g., 'diabetes', 'gfp_yield', 'aminotransferase')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: creates timestamped directory)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Exporting Hyperparameter Search Results")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("=" * 80)
    
    # Load results
    results = load_all_results(dataset_filter=args.dataset)
    
    if len(results) == 0:
        logger.error(f"No results found for dataset '{args.dataset}' in {RESULTS_ROOT}/")
        logger.error("Make sure the dataset name matches the run directory names")
        return 1
    
    logger.info(f"Loaded {len(results)} results for dataset '{args.dataset}'")
    
    # Create DataFrame
    df = create_dataframe(results)
    
    # Create output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir_name = create_output_directory(args.dataset)
        output_dir = os.path.abspath(output_dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"OUTPUT DIRECTORY:")
    logger.info(f"  {output_dir}")
    logger.info("=" * 80)
    
    # Detect task type for filename
    is_classification = 'post_accuracy' in df.columns and df['post_accuracy'].notna().any()
    task_type = 'classification' if is_classification else 'regression'
    
    # Create CSV filename
    dataset_name_for_file = args.dataset.lower().replace(' ', '_').replace('-', '_')
    dataset_name_for_file = dataset_name_for_file.replace('protien', 'protein')
    csv_filename = f'{task_type}_results_summary_{dataset_name_for_file}.csv'
    csv_file = os.path.join(output_dir, csv_filename)
    
    # Save CSV
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved results summary to: {csv_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    if is_classification:
        if 'post_accuracy' in df.columns:
            logger.info(f"Accuracy Statistics:")
            logger.info(f"  Mean: {df['post_accuracy'].mean():.4f}")
            logger.info(f"  Std:  {df['post_accuracy'].std():.4f}")
            logger.info(f"  Min:  {df['post_accuracy'].min():.4f}")
            logger.info(f"  Max:  {df['post_accuracy'].max():.4f}")
        
        if 'post_f1' in df.columns:
            logger.info(f"\nF1 Score Statistics:")
            logger.info(f"  Mean: {df['post_f1'].mean():.4f}")
            logger.info(f"  Std:  {df['post_f1'].std():.4f}")
            logger.info(f"  Min:  {df['post_f1'].min():.4f}")
            logger.info(f"  Max:  {df['post_f1'].max():.4f}")
    else:
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
    logger.info(f"Task type: {task_type}")
    logger.info(f"Total configurations: {len(df)}")
    logger.info(f"CSV file: {csv_file}")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
