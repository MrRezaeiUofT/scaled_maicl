#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Mechanism Contributions on Test Set
--------------------------------------------
This script evaluates how different mechanism combinations affect performance:
1. LLM only (no mechanisms)
2. LLM + known mechanisms
3. LLM + all mechanisms (known + unknown)

Usage:
    python evaluate_mechanism_contributions.py --experiment_path ma_icl_results/class_zoo_mlxgboost_topk300_iter5_modelgemini_2_0_flash_lossf1_acceptvalidation_topkstratresidual_balanced_maxsamp150
"""

import os
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import re

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

from maicl_lib_v2 import (
    MinMaxScaler010,
    TrainableMAICL,
    MLModelMechanism,
    BatchedLLM,
    GoogleAPIKeyManager,
    MultiAgentPredictor,
    SCALE_MIN,
    SCALE_MAX,
)
from maicl_llm import OpenAIAPIKeyManager
from evaluate_individual_mechanisms import parse_mechanisms_file

# Import dataset loaders from classification script
import sys
import importlib.util
script_dir = os.path.dirname(os.path.abspath(__file__))
classification_script = os.path.join(script_dir, "017_maicl_classification_residual_topk.py")

# Load the classification script as a module
spec = importlib.util.spec_from_file_location("classification_script", classification_script)
classification_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classification_module)

# Import functions from the module
load_openml_classification = classification_module.load_openml_classification
load_synthetic_classification_5classes = classification_module.load_synthetic_classification_5classes
load_tabarena_classification_dataset = classification_module.load_tabarena_classification_dataset
load_deepchem_classification_dataset = classification_module.load_deepchem_classification_dataset
load_enzyme_classification_dataset = classification_module.load_enzyme_classification_dataset
_get_llm = classification_module._get_llm
_get_pandas = classification_module._get_pandas
RANDOM_STATE = classification_module.RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_task_type(config: Dict[str, Any], run_name: str = "") -> str:
    """Detect if this is a classification or regression task"""
    # Check run name for prefixes
    if run_name.startswith('class_'):
        return 'classification'
    elif run_name.startswith('bio_reg_'):
        return 'regression'
    
    # Check config for task type
    if "task_type" in config:
        return config["task_type"]
    
    # Check ml_baseline metrics
    if "ml_baseline" in config and config.get("ml_baseline"):
        ml_baseline = config["ml_baseline"]
        if isinstance(ml_baseline, dict):
            if "accuracy" in ml_baseline or "f1" in ml_baseline:
                return "classification"
            if "r2" in ml_baseline or "mae" in ml_baseline:
                return "regression"
    
    # Check post_training metrics
    if "post_training" in config and config.get("post_training"):
        post = config["post_training"]
        if isinstance(post, dict):
            if "accuracy" in post or "f1" in post:
                return "classification"
            if "r2" in post or "mae" in post:
                return "regression"
    
    # Default to classification if we can't determine
    return "classification"


def load_experiment_config(experiment_path: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON file"""
    config_file = os.path.join(experiment_path, "experiment_config.json")
    run_name = os.path.basename(experiment_path)
    
    if not os.path.exists(config_file):
        # Try final_results.json as fallback
        config_file = os.path.join(experiment_path, "final_results.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found in {experiment_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Normalize config structure - handle both experiment_config.json and final_results.json formats
    if "dataset" in config and isinstance(config["dataset"], str):
        # final_results.json format - convert to normalized structure
        normalized = {
            "dataset": {
                "name": config.get("dataset_name", config.get("dataset", "")),
                "label": config.get("dataset", ""),
                "max_samples": config.get("n_train", 200) + config.get("n_val", 0) + config.get("n_test", 0)
            },
            "model": {
                "llm_model_name": config.get("model_name", "gemini-2.0-flash"),
                "ml_mechanism": config.get("ml_mechanism", "xgboost")
            },
            "training": {
                "iterations": config.get("iterations", 5),
                "val_size": 0.2,  # Default, will try to infer from n_val/n_train
                "no_scaling": False  # Default
            },
            "system": {
                "random_state": 42  # Default
            },
            "run_name": config.get("run_name", run_name)
        }
        # Try to infer val_size from n_val and n_train
        if "n_val" in config and "n_train" in config:
            if config["n_train"] > 0:
                normalized["training"]["val_size"] = config["n_val"] / (config["n_train"] + config["n_val"])
        config = normalized
    
    # Detect and add task type
    config["task_type"] = detect_task_type(config, config.get("run_name", run_name))
    
    return config


def load_mechanisms(experiment_path: str, iteration: int) -> List[Tuple[str, str]]:
    """Load mechanisms from the specified iteration"""
    mechanisms_file = os.path.join(experiment_path, f"mechanisms_iter_{iteration}.txt")
    if not os.path.exists(mechanisms_file):
        raise FileNotFoundError(f"Mechanisms file not found: {mechanisms_file}")
    
    mechanisms = parse_mechanisms_file(mechanisms_file)
    logger.info(f"Loaded {len(mechanisms)} mechanisms from iteration {iteration}")
    for i, (mech_type, mech_text) in enumerate(mechanisms):
        logger.info(f"  Mechanism {i}: {mech_type.upper()} ({len(mech_text)} chars)")
    
    return mechanisms


def separate_mechanisms(mechanisms: List[Tuple[str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """Separate mechanisms into known, unknown (LLM), and ML"""
    known_mechanisms = []
    unknown_mechanisms = []
    ml_mechanisms = []
    
    for mech_type, mech_text in mechanisms:
        mech_type_lower = mech_type.lower()
        if mech_type_lower == "known":
            known_mechanisms.append(mech_text)
        elif mech_type_lower == "llm":
            unknown_mechanisms.append(mech_text)
        elif mech_type_lower == "ml":
            ml_mechanisms.append(mech_text)
    
    logger.info(f"Separated mechanisms: {len(known_mechanisms)} known, {len(unknown_mechanisms)} unknown (LLM), {len(ml_mechanisms)} ML")
    return known_mechanisms, unknown_mechanisms, ml_mechanisms




def recreate_splits(X_encoded, y_encoded, X_original, config: Dict[str, Any], task_type: str = "classification") -> Tuple:
    """Recreate the same train/val/test splits as the original experiment"""
    # Handle both config formats
    if isinstance(config.get("training"), dict):
        val_size = config.get("training", {}).get("val_size", 0.2)
    else:
        # Infer from n_val and n_train if available
        n_train = config.get("n_train", 64)
        n_val = config.get("n_val", 16)
        if n_train > 0:
            val_size = n_val / (n_train + n_val)
        else:
            val_size = 0.2
    
    if isinstance(config.get("system"), dict):
        random_state = config.get("system", {}).get("random_state", 42)
    else:
        random_state = 42
    
    # First split: train_full vs test (80/20)
    idx = np.arange(len(X_encoded))
    
    if task_type == "classification":
        from collections import Counter
        class_counts = Counter(y_encoded)
        stratify = y_encoded if min(class_counts.values()) >= 2 else None
        idx_tr_full, idx_te = train_test_split(idx, test_size=0.2, random_state=random_state, stratify=stratify)
        
        # Second split: train vs val from train_full
        y_tr_full = y_encoded[idx_tr_full]
        tr_class_counts = Counter(y_tr_full)
        can_stratify = (idx_tr_full.size > 0 and 
                       len(np.unique(y_tr_full)) > 1 and 
                       min(tr_class_counts.values()) >= 2)
        stratify_tr = y_tr_full if can_stratify else None
        idx_tr, idx_va = train_test_split(idx_tr_full, test_size=float(val_size), random_state=random_state, stratify=stratify_tr)
    else:  # regression - use quantile stratification if possible
        try:
            # Try quantile-based stratification for regression
            n_bins = 10
            quantiles = np.quantile(y_encoded, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
            y_bins = np.digitize(y_encoded, quantiles, right=True)
            idx_tr_full, idx_te = train_test_split(idx, test_size=0.2, random_state=random_state, stratify=y_bins)
            
            # Second split with stratification
            y_tr_full = y_encoded[idx_tr_full]
            quantiles_tr = np.quantile(y_tr_full, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
            y_bins_tr = np.digitize(y_tr_full, quantiles_tr, right=True)
            idx_tr, idx_va = train_test_split(idx_tr_full, test_size=float(val_size), random_state=random_state, stratify=y_bins_tr)
        except Exception:
            # Fallback to random split
            idx_tr_full, idx_te = train_test_split(idx, test_size=0.2, random_state=random_state)
            idx_tr, idx_va = train_test_split(idx_tr_full, test_size=float(val_size), random_state=random_state)
    
    X_train, X_val, X_test = X_encoded[idx_tr], X_encoded[idx_va], X_encoded[idx_te]
    y_train, y_val, y_test = y_encoded[idx_tr], y_encoded[idx_va], y_encoded[idx_te]
    X_original_train = [X_original[i] for i in idx_tr]
    X_original_val = [X_original[i] for i in idx_va]
    X_original_test = [X_original[i] for i in idx_te]
    
    logger.info(f"Recreated splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, X_original_train, X_original_val, X_original_test


def setup_scaling(X_train, X_val, X_test, feature_cols, config: Dict[str, Any]) -> Tuple:
    """Setup feature scaling based on config"""
    if isinstance(config.get("training"), dict):
        no_scaling = config.get("training", {}).get("no_scaling", False)
    else:
        no_scaling = False  # Default to scaling enabled
    
    if no_scaling:
        logger.info("Scaling disabled - using raw features")
        scaler = None
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
    else:
        scaler = MinMaxScaler010()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        logger.info(f"Features scaled to [{SCALE_MIN}, {SCALE_MAX}]")
    
    return scaler, X_train_s, X_val_s, X_test_s


def evaluate_llm_only(llm, X_test_s, y_test, X_train_s, y_train, feature_cols, scaler, 
                      task_type: str, class_names=None, X_original_test=None, X_original_train=None,
                      y_scaler=None) -> Dict[str, Any]:
    """Evaluate LLM only (no mechanisms)"""
    logger.info("Evaluating LLM only (no mechanisms)...")
    
    # Create a predictor with no mechanisms
    predictor = MultiAgentPredictor(
        llm, [], [], None, feature_cols, scaler,
        attention_temp=1.0, task_type=task_type, class_names=class_names
    )
    
    # Get predictions
    test_dicts = [{feature_cols[j]: float(X_test_s[i, j]) for j in range(len(feature_cols))} 
                  for i in range(len(X_test_s))]
    batch_outputs = predictor.predict_batch(test_dicts, few_shot_list=[[]] * len(test_dicts))
    predictions = [out[0] for out in batch_outputs]
    predictions = np.array(predictions)
    
    # Handle task-specific post-processing
    if task_type == "classification":
        predictions = np.round(predictions).astype(int)
        if class_names:
            n_classes = len(class_names)
            predictions = np.clip(predictions, 0, n_classes - 1)
        # Compute classification metrics
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        return {
            "accuracy": acc,
            "f1": f1,
            "predictions": predictions.tolist()
        }
    else:  # regression
        predictions = predictions.astype(float)
        # Unscale if needed
        if y_scaler is not None:
            # MinMaxScaler010 doesn't have inverse_transform, so we implement it manually
            # Formula: scaled = scale_ * (original - min_) + feature_range[0]
            # Inverse: original = (scaled - feature_range[0]) / scale_ + min_
            pred_reshaped = predictions.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            
            if hasattr(y_scaler, 'inverse_transform'):
                predictions = y_scaler.inverse_transform(pred_reshaped).ravel()
                y_test_unscaled = y_scaler.inverse_transform(y_test_reshaped).ravel()
            else:
                # Manual inverse transform for MinMaxScaler010
                feature_range = getattr(y_scaler, 'feature_range', (0.0, 1.0))
                scale = y_scaler.scale_
                min_orig = y_scaler.min_
                # Handle division by zero (when scale is 0, feature is constant)
                scale_safe = np.where(scale != 0, scale, 1.0)
                predictions = ((pred_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
                y_test_unscaled = ((y_test_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
        else:
            y_test_unscaled = y_test
        # Compute regression metrics
        r2 = r2_score(y_test_unscaled, predictions)
        mae = mean_absolute_error(y_test_unscaled, predictions)
        mse = mean_squared_error(y_test_unscaled, predictions)
        return {
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "predictions": predictions.tolist()
        }


def evaluate_with_mechanisms(llm, mechanisms: List[str], mechanism_types: List[str],
                            X_test_s, y_test, X_train_s, y_train, feature_cols, scaler,
                            task_type: str, class_names=None, X_original_test=None, X_original_train=None,
                            pretrained_ml_mechanism=None, y_scaler=None) -> Dict[str, Any]:
    """Evaluate with specified mechanisms"""
    logger.info(f"Evaluating with {len(mechanisms)} mechanisms...")
    
    # Create predictor with specified mechanisms
    predictor = MultiAgentPredictor(
        llm, mechanisms, mechanism_types, pretrained_ml_mechanism, feature_cols, scaler,
        attention_temp=1.0, task_type=task_type, class_names=class_names
    )
    
    # Get predictions
    test_dicts = [{feature_cols[j]: float(X_test_s[i, j]) for j in range(len(feature_cols))} 
                  for i in range(len(X_test_s))]
    batch_outputs = predictor.predict_batch(test_dicts, few_shot_list=[[]] * len(test_dicts))
    predictions = [out[0] for out in batch_outputs]
    predictions = np.array(predictions)
    
    # Handle task-specific post-processing
    if task_type == "classification":
        predictions = np.round(predictions).astype(int)
        if class_names:
            n_classes = len(class_names)
            predictions = np.clip(predictions, 0, n_classes - 1)
        # Compute classification metrics
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        return {
            "accuracy": acc,
            "f1": f1,
            "predictions": predictions.tolist()
        }
    else:  # regression
        predictions = predictions.astype(float)
        # Unscale if needed
        if y_scaler is not None:
            # MinMaxScaler010 doesn't have inverse_transform, so we implement it manually
            # Formula: scaled = scale_ * (original - min_) + feature_range[0]
            # Inverse: original = (scaled - feature_range[0]) / scale_ + min_
            pred_reshaped = predictions.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            
            if hasattr(y_scaler, 'inverse_transform'):
                predictions = y_scaler.inverse_transform(pred_reshaped).ravel()
                y_test_unscaled = y_scaler.inverse_transform(y_test_reshaped).ravel()
            else:
                # Manual inverse transform for MinMaxScaler010
                feature_range = getattr(y_scaler, 'feature_range', (0.0, 1.0))
                scale = y_scaler.scale_
                min_orig = y_scaler.min_
                # Handle division by zero (when scale is 0, feature is constant)
                scale_safe = np.where(scale != 0, scale, 1.0)
                predictions = ((pred_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
                y_test_unscaled = ((y_test_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
        else:
            y_test_unscaled = y_test
        # Compute regression metrics
        r2 = r2_score(y_test_unscaled, predictions)
        mae = mean_absolute_error(y_test_unscaled, predictions)
        mse = mean_squared_error(y_test_unscaled, predictions)
        return {
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "predictions": predictions.tolist()
        }


def create_comparison_plot(results: Dict[str, Dict[str, Any]], output_dir: str, task_type: str = "classification"):
    """Create comparison plots for different mechanism combinations"""
    if not _HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    scenarios = list(results.keys())
    
    if task_type == "classification":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract metrics
        accuracies = [results[s].get("accuracy", 0.0) for s in scenarios]
        f1_scores = [results[s].get("f1", 0.0) for s in scenarios]
        
        # Bar plot for accuracy
        bars1 = axes[0].bar(scenarios, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Bar plot for F1 score
        bars2 = axes[1].bar(scenarios, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "mechanism_contribution_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot to {plot_path}")
        
        # Create improvement plot
        if len(scenarios) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            baseline_acc = accuracies[0]  # LLM only
            baseline_f1 = f1_scores[0]
            
            acc_improvements = [(acc - baseline_acc) * 100 for acc in accuracies]
            f1_improvements = [(f1 - baseline_f1) * 100 for f1 in f1_scores]
            
            x = np.arange(len(scenarios))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, acc_improvements, width, label='Accuracy Δ', color='#1f77b4')
            bars2 = ax.bar(x + width/2, f1_improvements, width, label='F1 Score Δ', color='#ff7f0e')
            
            ax.set_ylabel('Improvement (%)', fontsize=12)
            ax.set_title('Performance Improvement vs LLM Only', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if abs(height) > 0.1:  # Only label if significant
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:+.1f}%', ha='center', 
                               va='bottom' if height > 0 else 'top', fontsize=9)
            
            plt.tight_layout()
            improvement_path = os.path.join(output_dir, "mechanism_improvement_comparison.png")
            plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved improvement plot to {improvement_path}")
    
    else:  # regression
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Extract metrics
        r2_scores = [results[s].get("r2", 0.0) for s in scenarios]
        mae_scores = [results[s].get("mae", 0.0) for s in scenarios]
        rmse_scores = [results[s].get("rmse", 0.0) for s in scenarios]
        
        # Bar plot for R2
        bars1 = axes[0].bar(scenarios, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, r2 in zip(bars1, r2_scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Bar plot for MAE
        bars2 = axes[1].bar(scenarios, mae_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('MAE Comparison (lower is better)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mae in zip(bars2, mae_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{mae:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Bar plot for RMSE
        bars3 = axes[2].bar(scenarios, rmse_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[2].set_ylabel('RMSE', fontsize=12)
        axes[2].set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rmse in zip(bars3, rmse_scores):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{rmse:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "mechanism_contribution_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot to {plot_path}")
        
        # Create improvement plot
        if len(scenarios) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            baseline_r2 = r2_scores[0]  # LLM only
            baseline_mae = mae_scores[0]
            
            r2_improvements = [(r2 - baseline_r2) * 100 for r2 in r2_scores]
            mae_improvements = [(baseline_mae - mae) * 100 for mae in mae_scores]  # Negative because lower is better
            
            x = np.arange(len(scenarios))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, r2_improvements, width, label='R² Δ', color='#1f77b4')
            bars2 = ax.bar(x + width/2, mae_improvements, width, label='MAE Δ (improvement)', color='#ff7f0e')
            
            ax.set_ylabel('Improvement (%)', fontsize=12)
            ax.set_title('Performance Improvement vs LLM Only', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if abs(height) > 0.1:  # Only label if significant
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:+.1f}%', ha='center', 
                               va='bottom' if height > 0 else 'top', fontsize=9)
            
            plt.tight_layout()
            improvement_path = os.path.join(output_dir, "mechanism_improvement_comparison.png")
            plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved improvement plot to {improvement_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mechanism contributions on test set")
    parser.add_argument("--experiment_path", type=str, required=True,
                        help="Path to experiment directory (e.g., ma_icl_results/class_zoo_...)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration number to use (default: last iteration from config)")
    args = parser.parse_args()
    
    experiment_path = args.experiment_path
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")
    
    logger.info("=" * 80)
    logger.info("MECHANISM CONTRIBUTION EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Experiment path: {experiment_path}")
    
    # Load experiment configuration
    config = load_experiment_config(experiment_path)
    task_type = config.get("task_type", "classification")
    dataset_label = config.get('dataset', {}).get('label', '') if isinstance(config.get('dataset'), dict) else config.get('dataset', 'unknown')
    model_name = config.get('model', {}).get('llm_model_name', 'unknown') if isinstance(config.get('model'), dict) else config.get('model_name', 'unknown')
    logger.info(f"Task Type: {task_type}")
    logger.info(f"Dataset: {dataset_label}")
    logger.info(f"Model: {model_name}")
    
    # Ensure task_type is set correctly
    if not task_type or task_type not in ["classification", "regression"]:
        # Try to detect from run_name
        run_name = config.get("run_name", os.path.basename(experiment_path))
        if run_name.startswith('class_'):
            task_type = 'classification'
        elif run_name.startswith('bio_reg_'):
            task_type = 'regression'
        else:
            task_type = 'classification'  # Default
        logger.info(f"Detected task type from run_name: {task_type}")
    
    # Determine iteration
    if args.iteration is None:
        if isinstance(config.get("training"), dict):
            iterations = config.get("training", {}).get("iterations", 5)
        else:
            iterations = config.get("iterations", 5)
        args.iteration = iterations
    logger.info(f"Using mechanisms from iteration {args.iteration}")
    
    # Load mechanisms
    mechanisms = load_mechanisms(experiment_path, args.iteration)
    known_mechs, unknown_mechs, ml_mechs = separate_mechanisms(mechanisms)
    
    # Reload dataset
    logger.info("Reloading dataset...")
    # Handle both config formats
    if isinstance(config.get("dataset"), dict):
        dataset_name = config.get("dataset", {}).get("name", "")
        max_samples = config.get("dataset", {}).get("max_samples", 200)
    else:
        dataset_name = config.get("dataset_name", config.get("dataset", ""))
        max_samples = config.get("n_train", 64) + config.get("n_val", 16) + config.get("n_test", 21)
    
    if not dataset_name:
        raise ValueError("Dataset name not found in config")
    
    dataset_name_lower = dataset_name.lower()
    
    # Load dataset based on task type and name
    class_names = None
    y_scaler = None
    
    if task_type == "classification":
        if dataset_name_lower in ("synthetic5", "syn5", "toy5"):
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_synthetic_classification_5classes(
                max_samples, n_features=20, class_sep=1.0, flip_y=0.03
            )
            ds_label = f"Synthetic 5-Class ({dataset_name})"
        elif dataset_name_lower in ("adult", "bank", "income", "wine", "spaceship", "default", "booking", "churn", 
                                     "iris", "breast-cancer", "digits", "mushroom", "diabetes", "credit", "heart", 
                                     "stroke", "employee", "telecom", "customer") or dataset_name_lower.startswith("tabarena_"):
            tabarena_name = dataset_name_lower.replace("tabarena_", "")
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_tabarena_classification_dataset(
                tabarena_name, max_samples
            )
            ds_label = f"TabArena {tabarena_name}"
        elif dataset_name_lower in ("hiv", "bace", "tox21"):
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_deepchem_classification_dataset(
                dataset_name, max_samples
            )
            ds_label = f"DeepChem {dataset_name.upper()}"
        elif dataset_name_lower.endswith("_binary") or dataset_name_lower.endswith("_categorical") or any(
            dataset_name_lower.startswith(prefix) for prefix in ["halogenase", "aminotransferase", "olea", "nitrilase", 
                                                                "phosphatase", "gt_", "duf", "esterase", "davis"]
        ):
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_enzyme_classification_dataset(
                dataset_name, max_samples
            )
            ds_label = f"Enzyme Dataset: {dataset_name}"
        else:
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_openml_classification(
                dataset_name, max_samples
            )
            ds_label = f"OpenML {dataset_name}"
        logger.info(f"Loaded dataset: {len(X_encoded)} samples, {len(feature_cols)} features, {len(class_names)} classes")
    else:  # regression
        # Import regression dataset loaders
        regression_script = os.path.join(script_dir, "018_maicl_regression_biotech.py")
        spec_reg = importlib.util.spec_from_file_location("regression_script", regression_script)
        regression_module = importlib.util.module_from_spec(spec_reg)
        spec_reg.loader.exec_module(regression_module)
        
        # Load regression dataset
        if dataset_name_lower == "gfp_yield":
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_gfp_yield_dataset(max_samples)
        elif dataset_name_lower == "protein_expression":
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_protein_expression_dataset(
                max_samples, plate_file=None, plate_index=None)
        elif dataset_name_lower == "protein_expression_all":
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_protein_expression_all_plates_dataset(max_samples)
        elif dataset_name_lower == "dataset_102":
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_dataset_102(max_samples)
        elif dataset_name_lower in ("esol", "delaney", "lipo", "lipophilicity"):
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_deepchem_regression_dataset(
                dataset_name, max_samples)
        elif dataset_name_lower in ("diabetes", "housing", "bike", "insurance", "concrete", "energy", 
                                      "airfoil", "yacht", "auto", "abalone", "winequality", "students", 
                                      "diamonds", "house-prices", "airbnb"):
            tabarena_name = dataset_name_lower.replace("tabarena_", "")
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_tabarena_regression_dataset(
                tabarena_name, max_samples)
        else:
            # Try enzyme dataset
            X_encoded, y_encoded, X_original, feature_cols, feature_encoders, ds_label = regression_module.load_enzyme_dataset(
                dataset_name, max_samples)
        logger.info(f"Loaded dataset: {len(X_encoded)} samples, {len(feature_cols)} features")
    
    # Recreate splits
    X_train, X_val, X_test, y_train, y_val, y_test, X_original_train, X_original_val, X_original_test = recreate_splits(
        X_encoded, y_encoded, X_original, config, task_type
    )
    
    # Setup scaling
    scaler, X_train_s, X_val_s, X_test_s = setup_scaling(X_train, X_val, X_test, feature_cols, config)
    
    # Setup y_scaler for regression (after splitting, fit on y_train to match original experiment)
    if task_type == "regression":
        no_scaling = config.get("training", {}).get("no_scaling", False) if isinstance(config.get("training"), dict) else False
        if not no_scaling:
            y_scaler = MinMaxScaler010()
            y_scaler.fit(y_train.reshape(-1, 1))
            # Scale y values for consistency with original experiment
            y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
            y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
            logger.info(f"Targets scaled to [{SCALE_MIN}, {SCALE_MAX}] range (y min={y_train_s.min():.3f}, max={y_train_s.max():.3f})")
        else:
            y_scaler = None
            y_train_s = y_train
            y_val_s = y_val
            y_test_s = y_test
            logger.info("Target scaling disabled - using raw values")
    else:
        y_scaler = None
        y_train_s = y_train
        y_val_s = y_val
        y_test_s = y_test
    
    # Initialize LLM
    if isinstance(config.get("model"), dict):
        model_name = config.get("model", {}).get("llm_model_name", "gemini-2.0-flash")
    else:
        model_name = config.get("model_name", "gemini-2.0-flash")
    llm = _get_llm(model_name)
    logger.info(f"Initialized LLM: {model_name}")
    
    # Setup ML mechanism if available
    pretrained_ml = None
    if ml_mechs and len(ml_mechs) > 0:
        if isinstance(config.get("model"), dict):
            ml_mech_name = config.get("model", {}).get("ml_mechanism", "xgboost")
        else:
            ml_mech_name = config.get("ml_mechanism", "xgboost")
        
        if task_type == "classification":
            mech_map = {"logreg": "LogisticRegression", "xgboost": "XGBoost", "tabicl": "TabICL"}
            model_name_ml = mech_map.get(ml_mech_name.lower(), "LogisticRegression")
        else:  # regression
            mech_map = {"linear": "LinearRegression", "xgboost": "XGBoost", "kernelridge": "KernelRidge", "tabicl": "TabICL"}
            model_name_ml = mech_map.get(ml_mech_name.lower(), "KernelRidge")
        
        pretrained_ml = MLModelMechanism(model_name_ml, task_type=task_type)
        pretrained_ml.train(X_train_s, y_train, feature_cols, y_scaler=y_scaler)
        pretrained_ml._maicl_feature_cols = feature_cols
        pretrained_ml._maicl_feature_encoders = feature_encoders
        pretrained_ml._maicl_scaler = scaler
        logger.info(f"Initialized ML mechanism: {model_name_ml}")
    
    # Evaluate three scenarios
    results = {}
    
    # 1. LLM only (no mechanisms)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: LLM Only (No Mechanisms)")
    logger.info("=" * 80)
    results["LLM Only"] = evaluate_llm_only(
        llm, X_test_s, y_test, X_train_s, y_train, feature_cols, scaler,
        task_type, class_names, X_original_test, X_original_train, y_scaler
    )
    if task_type == "classification":
        if 'accuracy' in results['LLM Only']:
            logger.info(f"  Accuracy: {results['LLM Only']['accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['LLM Only']['f1']:.4f}")
        else:
            logger.warning(f"  Classification metrics not found. Got keys: {list(results['LLM Only'].keys())}")
    else:
        if 'r2' in results['LLM Only']:
            logger.info(f"  R²: {results['LLM Only']['r2']:.4f}")
            logger.info(f"  MAE: {results['LLM Only']['mae']:.4f}")
            logger.info(f"  RMSE: {results['LLM Only']['rmse']:.4f}")
        else:
            logger.warning(f"  Regression metrics not found. Got keys: {list(results['LLM Only'].keys())}")
    
    # 2. LLM + Known mechanisms
    if known_mechs:
        logger.info("\n" + "=" * 80)
        logger.info("SCENARIO 2: LLM + Known Mechanisms")
        logger.info("=" * 80)
        mech_types = ["known"] * len(known_mechs)
        results["LLM + Known"] = evaluate_with_mechanisms(
            llm, known_mechs, mech_types, X_test_s, y_test, X_train_s, y_train,
            feature_cols, scaler, task_type, class_names, X_original_test, X_original_train,
            pretrained_ml_mechanism=pretrained_ml, y_scaler=y_scaler
        )
        if task_type == "classification":
            logger.info(f"  Accuracy: {results['LLM + Known']['accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['LLM + Known']['f1']:.4f}")
        else:
            logger.info(f"  R²: {results['LLM + Known']['r2']:.4f}")
            logger.info(f"  MAE: {results['LLM + Known']['mae']:.4f}")
            logger.info(f"  RMSE: {results['LLM + Known']['rmse']:.4f}")
    else:
        logger.info("No known mechanisms found, skipping scenario 2")
    
    # 3. LLM + All mechanisms (known + unknown)
    all_mechs = known_mechs + unknown_mechs
    all_mech_types = ["known"] * len(known_mechs) + ["llm"] * len(unknown_mechs)
    if all_mechs:
        logger.info("\n" + "=" * 80)
        logger.info("SCENARIO 3: LLM + All Mechanisms (Known + Unknown)")
        logger.info("=" * 80)
        results["LLM + All"] = evaluate_with_mechanisms(
            llm, all_mechs, all_mech_types, X_test_s, y_test, X_train_s, y_train,
            feature_cols, scaler, task_type, class_names, X_original_test, X_original_train,
            pretrained_ml_mechanism=pretrained_ml, y_scaler=y_scaler
        )
        if task_type == "classification":
            logger.info(f"  Accuracy: {results['LLM + All']['accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['LLM + All']['f1']:.4f}")
        else:
            logger.info(f"  R²: {results['LLM + All']['r2']:.4f}")
            logger.info(f"  MAE: {results['LLM + All']['mae']:.4f}")
            logger.info(f"  RMSE: {results['LLM + All']['rmse']:.4f}")
    else:
        logger.info("No mechanisms found, skipping scenario 3")
    
    # 4. ML baseline (if available)
    if pretrained_ml:
        logger.info("\n" + "=" * 80)
        logger.info("SCENARIO 4: ML Baseline")
        logger.info("=" * 80)
        test_dicts = [{feature_cols[j]: float(X_test_s[i, j]) for j in range(len(feature_cols))} 
                     for i in range(len(X_test_s))]
        ml_predictions = np.array([pretrained_ml.predict(d, return_class_index=(task_type=="classification")) for d in test_dicts])
        
        if task_type == "classification":
            ml_predictions = np.round(ml_predictions).astype(int)
            if class_names:
                n_classes = len(class_names)
                ml_predictions = np.clip(ml_predictions, 0, n_classes - 1)
            ml_acc = accuracy_score(y_test, ml_predictions)
            ml_f1 = f1_score(y_test, ml_predictions, average='weighted', zero_division=0)
            results["ML Baseline"] = {
                "accuracy": ml_acc,
                "f1": ml_f1,
                "predictions": ml_predictions.tolist()
            }
            logger.info(f"  Accuracy: {results['ML Baseline']['accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['ML Baseline']['f1']:.4f}")
        else:  # regression
            ml_predictions = ml_predictions.astype(float)
            if y_scaler is not None:
                ml_pred_reshaped = ml_predictions.reshape(-1, 1)
                y_test_reshaped = y_test.reshape(-1, 1)
                
                # Inverse transform for MinMaxScaler010
                if hasattr(y_scaler, 'inverse_transform'):
                    ml_predictions = y_scaler.inverse_transform(ml_pred_reshaped).ravel()
                    y_test_unscaled = y_scaler.inverse_transform(y_test_reshaped).ravel()
                else:
                    # Manual inverse transform for MinMaxScaler010
                    feature_range = getattr(y_scaler, 'feature_range', (0.0, 1.0))
                    scale = y_scaler.scale_
                    min_orig = y_scaler.min_
                    # Handle division by zero (when scale is 0, feature is constant)
                    scale_safe = np.where(scale != 0, scale, 1.0)
                    ml_predictions = ((ml_pred_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
                    y_test_unscaled = ((y_test_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
            else:
                y_test_unscaled = y_test
            ml_r2 = r2_score(y_test_unscaled, ml_predictions)
            ml_mae = mean_absolute_error(y_test_unscaled, ml_predictions)
            ml_mse = mean_squared_error(y_test_unscaled, ml_predictions)
            results["ML Baseline"] = {
                "r2": ml_r2,
                "mae": ml_mae,
                "mse": ml_mse,
                "rmse": np.sqrt(ml_mse),
                "predictions": ml_predictions.tolist()
            }
            logger.info(f"  R²: {results['ML Baseline']['r2']:.4f}")
            logger.info(f"  MAE: {results['ML Baseline']['mae']:.4f}")
            logger.info(f"  RMSE: {results['ML Baseline']['rmse']:.4f}")
    
    # Create comparison plots
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING COMPARISON PLOTS")
    logger.info("=" * 80)
    create_comparison_plot(results, experiment_path, task_type=task_type)
    
    # Save results to JSON
    results_file = os.path.join(experiment_path, "mechanism_contribution_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "experiment_path": experiment_path,
            "iteration": args.iteration,
            "dataset": ds_label,
            "test_set_size": len(X_test),
            "results": results,
            "mechanism_counts": {
                "known": len(known_mechs),
                "unknown": len(unknown_mechs),
                "ml": len(ml_mechs),
                "total": len(mechanisms)
            }
        }, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for scenario, metrics in results.items():
        if task_type == "classification":
            logger.info(f"{scenario:20s}: ACC={metrics.get('accuracy', 0.0):.4f}, F1={metrics.get('f1', 0.0):.4f}")
        else:
            logger.info(f"{scenario:20s}: R²={metrics.get('r2', 0.0):.4f}, MAE={metrics.get('mae', 0.0):.4f}, RMSE={metrics.get('rmse', 0.0):.4f}")
    
    if len(results) > 1:
        baseline = results.get("LLM Only", {})
        if baseline:
            logger.info("\nImprovements vs LLM Only:")
            for scenario, metrics in results.items():
                if scenario != "LLM Only":
                    if task_type == "classification":
                        baseline_acc = baseline.get("accuracy", 0.0)
                        baseline_f1 = baseline.get("f1", 0.0)
                        acc_delta = metrics.get('accuracy', 0.0) - baseline_acc
                        f1_delta = metrics.get('f1', 0.0) - baseline_f1
                        logger.info(f"{scenario:20s}: ΔACC={acc_delta:+.4f}, ΔF1={f1_delta:+.4f}")
                    else:
                        baseline_r2 = baseline.get("r2", 0.0)
                        baseline_mae = baseline.get("mae", 0.0)
                        r2_delta = metrics.get('r2', 0.0) - baseline_r2
                        mae_delta = baseline_mae - metrics.get('mae', 0.0)  # Negative because lower is better
                        logger.info(f"{scenario:20s}: ΔR²={r2_delta:+.4f}, ΔMAE={mae_delta:+.4f} (improvement)")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

