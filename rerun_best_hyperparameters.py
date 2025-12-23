#!/usr/bin/env python3
"""
Rerun experiments with best hyperparameters from best_hyperparameters_summary.csv.
Reads the CSV file and runs each experiment with the corresponding best hyperparameters.
"""

import os
import sys
import pandas as pd
import subprocess
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CSV_PATH = "ma_icl_results/BC_Gemini_Flash_2/best_hyperparameters_summary.csv"
CLASSIFICATION_SCRIPT = "017_maicl_classification_residual_topk.py"
REGRESSION_SCRIPT = "018_maicl_regression_biotech.py"

def extract_model_name(row):
    """Extract model name from row, preferring run_name if available"""
    model_name = 'gemini-2.5-pro'  # Default
    run_name = str(row.get('run_name', ''))
    if 'gemini' in run_name.lower():
        # Extract model name from run_name (e.g., modelgemini_2_0_flash -> gemini-2.0-flash)
        import re
        # Match pattern like "modelgemini_2_0_flash" or "modelgemini-2-0-flash"
        model_match = re.search(r'model(gemini[^_]+)', run_name.lower())
        if model_match:
            model_str = model_match.group(1)
            # Replace underscores with dots where appropriate (e.g., 2_0 -> 2.0)
            model_str = re.sub(r'(\d+)_(\d+)', r'\1.\2', model_str)
            # Replace remaining underscores with hyphens
            model_str = model_str.replace('_', '-')
            model_name = model_str
    elif pd.notna(row.get('model_name')):
        model_name = str(row['model_name'])
    return model_name

def build_classification_command(row):
    """Build command for classification experiment"""
    cmd = ['python', CLASSIFICATION_SCRIPT]
    
    # Dataset
    dataset = row['dataset']
    cmd.extend(['--dataset', dataset])
    
    # ML mechanism
    ml_mech = row['ml_mech']
    cmd.extend(['--ml_mech', ml_mech])
    
    # Top K
    top_k = int(row['top_k'])
    cmd.extend(['--top_k', str(top_k)])
    
    # Iterations
    iterations = int(row['iterations'])
    cmd.extend(['--iterations', str(iterations)])
    
    # Model name
    model_name = extract_model_name(row)
    cmd.extend(['--model_name', model_name])
    
    # Acceptance set
    acceptance_set = row['acceptance_set']
    cmd.extend(['--acceptance_set', acceptance_set])
    
    # TopK strategy
    topk_strategy = row.get('topk_strategy', 'residual')
    if pd.notna(topk_strategy):
        cmd.extend(['--topk_strategy', topk_strategy])
    
    # Num mechanisms unknown
    num_mech = int(row['num_mechanisms_unknown'])
    cmd.extend(['--num_mechanisms_unknown', str(num_mech)])
    
    # No scaling
    if row.get('no_scaling', False):
        cmd.append('--no_scaling')
    
    # Max samples
    maxsamp = row.get('maxsamp', '')
    if pd.notna(maxsamp) and maxsamp != '':
        max_samples = int(float(maxsamp))
        cmd.extend(['--max_samples', str(max_samples)])
    
    # Evaluate individual mechanisms
    if row.get('evalindiv', False):
        cmd.append('--evaluate_individual_mechanisms')
    
    # Classification loss
    loss = row.get('loss', 'f1')
    if pd.notna(loss):
        cmd.extend(['--classification_loss', loss])
    
    # Use ML (default to 1)
    cmd.extend(['--use_ml', '1'])
    
    # Val size (default)
    cmd.extend(['--val_size', '0.2'])
    
    return cmd

def build_regression_command(row):
    """Build command for regression experiment"""
    cmd = ['python', REGRESSION_SCRIPT]
    
    # Dataset
    dataset = row['dataset']
    cmd.extend(['--dataset', dataset])
    
    # Plate index for protein_expression
    if dataset == 'protein_expression':
        # Try to extract plate index from run_name or directory
        run_name = str(row.get('run_name', ''))
        if 'plate' in run_name.lower():
            import re
            plate_match = re.search(r'plate(\d+)', run_name.lower())
            if plate_match:
                plate_idx = int(plate_match.group(1))
                cmd.extend(['--plate_index', str(plate_idx)])
    
    # ML mechanism
    ml_mech = row['ml_mech']
    cmd.extend(['--ml_mech', ml_mech])
    
    # Top K
    top_k = int(row['top_k'])
    cmd.extend(['--top_k', str(top_k)])
    
    # Iterations
    iterations = int(row['iterations'])
    cmd.extend(['--iterations', str(iterations)])
    
    # Model name
    model_name = extract_model_name(row)
    cmd.extend(['--model_name', model_name])
    
    # Acceptance set
    acceptance_set = row['acceptance_set']
    cmd.extend(['--acceptance_set', acceptance_set])
    
    # TopK strategy
    topk_strategy = row.get('topk_strategy', 'residual')
    if pd.notna(topk_strategy):
        cmd.extend(['--topk_strategy', topk_strategy])
    
    # Num mechanisms unknown
    num_mech = int(row['num_mechanisms_unknown'])
    cmd.extend(['--num_mechanisms_unknown', str(num_mech)])
    
    # No scaling
    if row.get('no_scaling', False):
        cmd.append('--no_scaling')
    
    # Regression loss
    loss = row.get('loss', 'r2')
    if pd.notna(loss):
        cmd.extend(['--regression_loss', loss])
    
    # Use ML (default to 1)
    cmd.extend(['--use_ml', '1'])
    
    # Val size (default)
    cmd.extend(['--val_size', '0.2'])
    
    # K shot (default to 0)
    cmd.extend(['--k_shot', '0'])
    
    # Evaluate individual mechanisms
    if row.get('evalindiv', False):
        cmd.append('--evaluate_individual_mechanisms')
    
    return cmd

def run_experiment(cmd, dataset, task_type, run_num, total_runs):
    """Run a single experiment"""
    logger.info("=" * 80)
    logger.info(f"Running experiment {run_num}/{total_runs}")
    logger.info(f"Dataset: {dataset} ({task_type})")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 80)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=7200  # 2 hour timeout per run
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully completed in {elapsed_time:.2f} seconds")
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'dataset': dataset,
                'task_type': task_type,
                'command': ' '.join(cmd),
            }
        else:
            logger.error(f"✗ Failed with exit code {result.returncode} after {elapsed_time:.2f} seconds")
            logger.error(f"Error output: {result.stderr[-500:] if result.stderr else 'No error output'}")
            return {
                'status': 'failed',
                'exit_code': result.returncode,
                'elapsed_time': elapsed_time,
                'dataset': dataset,
                'task_type': task_type,
                'command': ' '.join(cmd),
                'stderr': result.stderr[-1000:] if result.stderr else '',
            }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ Timeout after {elapsed_time:.2f} seconds")
        return {
            'status': 'timeout',
            'elapsed_time': elapsed_time,
            'dataset': dataset,
            'task_type': task_type,
            'command': ' '.join(cmd),
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ Exception: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time,
            'dataset': dataset,
            'task_type': task_type,
            'command': ' '.join(cmd),
        }

def main():
    parser = argparse.ArgumentParser(
        description="Rerun experiments with best hyperparameters from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH,
                        help=f"Path to best hyperparameters CSV (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments if output directory already exists")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return 1
    
    # Read CSV
    logger.info(f"Reading best hyperparameters from: {args.csv}")
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return 1
    
    logger.info(f"Found {len(df)} datasets to run")
    logger.info("=" * 80)
    
    # Results tracking
    results = {
        'start_time': datetime.now().isoformat(),
        'csv_file': args.csv,
        'total_datasets': len(df),
        'runs': []
    }
    
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0
    
    # Run each experiment
    for idx, row in df.iterrows():
        dataset = row['dataset']
        task_type = row['task_type']
        
        # Check if we should skip existing results
        if args.skip_existing:
            directory = row.get('directory', '')
            if pd.notna(directory) and os.path.exists(directory):
                logger.info(f"Skipping {dataset} - output directory already exists: {directory}")
                skipped_runs += 1
                results['runs'].append({
                    'status': 'skipped',
                    'dataset': dataset,
                    'task_type': task_type,
                    'reason': 'output_directory_exists'
                })
                continue
        
        # Build command based on task type
        if task_type == 'classification':
            cmd = build_classification_command(row)
        elif task_type == 'regression':
            cmd = build_regression_command(row)
        else:
            logger.warning(f"Unknown task type '{task_type}' for dataset {dataset}, skipping")
            continue
        
        # Dry run mode
        if args.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            results['runs'].append({
                'status': 'dry_run',
                'dataset': dataset,
                'task_type': task_type,
                'command': ' '.join(cmd),
            })
            continue
        
        # Run experiment
        result = run_experiment(cmd, dataset, task_type, idx + 1, len(df))
        results['runs'].append(result)
        
        if result['status'] == 'success':
            successful_runs += 1
        else:
            failed_runs += 1
        
        # Save intermediate results
        results_file = 'rerun_best_hyperparameters_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Brief pause between runs
        if idx < len(df) - 1:  # Don't pause after last run
            time.sleep(2)
    
    # Final summary
    results['end_time'] = datetime.now().isoformat()
    results['successful_runs'] = successful_runs
    results['failed_runs'] = failed_runs
    results['skipped_runs'] = skipped_runs
    
    logger.info("=" * 80)
    logger.info("RERUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total datasets: {len(df)}")
    logger.info(f"Successful: {successful_runs}")
    logger.info(f"Failed: {failed_runs}")
    if skipped_runs > 0:
        logger.info(f"Skipped: {skipped_runs}")
    
    if not args.dry_run:
        success_rate = (successful_runs / (successful_runs + failed_runs) * 100) if (successful_runs + failed_runs) > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Save final results
    results_file = 'rerun_best_hyperparameters_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    return 0 if failed_runs == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

