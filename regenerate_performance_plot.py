#!/usr/bin/env python3
"""
Regenerate performance comparison plot from saved final_results.json
"""

import json
import sys
from pathlib import Path
from visualization import plot_performance_comparison

def regenerate_plot(results_dir):
    """Regenerate performance comparison plot from final_results.json"""
    results_dir = Path(results_dir)
    json_path = results_dir / "final_results.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return
    
    # Load results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    ml_baseline = results.get('ml_baseline', {})
    pre_training = results.get('pre_training', {})
    post_training = results.get('post_training', {})
    llm_only_pre = results.get('llm_only_pre', None)
    llm_only_post = results.get('llm_only_post', None)
    baseline_models = results.get('baseline_models', None)
    
    # Determine task type from dataset name or metrics
    task_type = "regression"  # Default for this script
    
    # Generate plot
    output_path = results_dir / "performance_comparison.png"
    
    print(f"Regenerating performance comparison plot...")
    print(f"  ML Baseline: R²={ml_baseline.get('r2', 0):.4f}, MAE={ml_baseline.get('mae', 0):.4f}")
    print(f"  Pre-training: R²={pre_training.get('r2', 0):.4f}, MAE={pre_training.get('mae', 0):.4f}")
    print(f"  Post-training: R²={post_training.get('r2', 0):.4f}, MAE={post_training.get('mae', 0):.4f}")
    
    if baseline_models:
        print(f"  Baseline models found: {list(baseline_models.keys())}")
        for model_name, metrics in baseline_models.items():
            print(f"    {model_name}: R²={metrics.get('r2', 0):.4f}, MAE={metrics.get('mae', 0):.4f}")
    else:
        print("  No baseline models found in results")
    
    plot_performance_comparison(
        ml_baseline=ml_baseline,
        pre_maicl=pre_training,
        post_maicl=post_training,
        task_type=task_type,
        output_path=str(output_path),
        llm_only_pre=llm_only_pre,
        llm_only_post=llm_only_post,
        baseline_models=baseline_models
    )
    
    print(f"\n✓ Plot saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python regenerate_performance_plot.py <results_directory>")
        print("Example: python regenerate_performance_plot.py ma_icl_results/bio_reg_diabetes_...")
        sys.exit(1)
    
    regenerate_plot(sys.argv[1])

