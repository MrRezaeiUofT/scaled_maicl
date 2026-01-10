#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA-ICL Regression Runner (Biotech/Biological Datasets)
---------------------------------------------------
- Trains an ML baseline on scaled features
- Computes residuals on train set and selects top-K by residual magnitude
- Trains MA-ICL on the residual-difficult subset (optimization target: MAE or R2, configurable via --regression_loss)
- Reports pre/post: MAE, R2, and MSE on the test set

Available Datasets:
- Experimental: gfp_yield, protein_expression, protein_expression_all, dataset_102
- InaData (SU/EC): pfas_su, ec_fertility, ec_climbing

Usage examples:
  
  # List available protein expression plates:
  python 018_maicl_regression_biotech.py --list_plates
  
  # Select protein expression plate by index (0-based):
  python 018_maicl_regression_biotech.py --dataset protein_expression --plate_index 0 --top_k 50
  
  # Select protein expression plate by filename:
  python 018_maicl_regression_biotech.py --dataset protein_expression --plate_file plate_AL_1_raw_yield_and_std.csv --top_k 50
"""

import os
import argparse
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path
import json
import re
import csv

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    # Import deepchem (TensorFlow will initialize lazily, env vars set above should help)
    import deepchem as dc
    # Test that molnet is accessible
    _ = dc.molnet
    _HAS_DEEPCHEM = True
except ImportError as e:
    _HAS_DEEPCHEM = False
    _DEEPCHEM_ERROR = str(e)
except Exception as e:
    # Other errors (like TensorFlow missing) - still mark as unavailable
    _HAS_DEEPCHEM = False
    _DEEPCHEM_ERROR = str(e)

from maicl_lib_v2 import (
    MinMaxScaler010,
    TrainableMAICL,
    MLModelMechanism,
    BatchedLLM,
    GoogleAPIKeyManager,
    validate_scaled_data,
    compute_ml_residuals,
    get_top_k_residual_samples,
    set_output_dir,
    OUTPUT_DIR,
    create_result_visualizations,
    SCALE_MIN,
    SCALE_MAX,
)

# Optional baseline imports
try:
    from tabpfn import TabPFNRegressor
    _HAS_TABPFN = True
except:
    _HAS_TABPFN = False

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    _HAS_EBM = True
except:
    _HAS_EBM = False

try:
    import shap
    _HAS_SHAP = True
except:
    _HAS_SHAP = False

try:
    import llmlex
    import openai
    _HAS_LLMLEX = True
except:
    _HAS_LLMLEX = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except:
    _HAS_XGB = False

# RDKit for molecular feature extraction
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import hashlib
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not _HAS_RDKIT:
    logger.warning("RDKit not available. RDKit molecular features will be skipped. Install with: pip install rdkit or conda install -c conda-forge rdkit")

MODEL_NAME = os.environ.get("MAICL_MODEL_NAME", "gemini-2.0-flash")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def _load_env():
    """Load environment variables from .env if available (no error if missing)."""
    try:
        from dotenv import load_dotenv, find_dotenv
        here = Path(__file__).resolve().parent
        env_path = here / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
    except Exception:
        pass


def _get_gemini_llm(model_name: str):
    _load_env()
    keys = []
    for env_key in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3"]:
        val = os.environ.get(env_key)
        if val:
            keys.append(val)
    if not keys:
        raise RuntimeError("No Google API key found. Set GOOGLE_API_KEY or GOOGLE_API_KEY_1/2/3")
    km = GoogleAPIKeyManager(keys, model_name=model_name)
    return BatchedLLM(km)


def _subsample_if_needed(X_df: 'pd.DataFrame', y_series: 'pd.Series', max_samples: int):
    if max_samples and len(X_df) > max_samples:
        # Sample and capture original indices BEFORE resetting
        X_df_sampled = X_df.sample(n=max_samples, random_state=RANDOM_STATE)
        original_indices = X_df_sampled.index
        # Reset X_df indices
        X_df = X_df_sampled.reset_index(drop=True)
        # Use original indices to select corresponding y_series values, then reset
        y_series = y_series.loc[original_indices].reset_index(drop=True)
    return X_df, y_series


def save_scatter_plot(y_true, y_pred, title, filename, output_dir):
    """Generate and save a scatter plot of predicted vs actual values."""
    if not _HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed, skipping scatter plot")
        return
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved scatter plot to {save_path}")


def compute_additional_baselines(X_train, y_train, X_test, y_test, task_type="regression", 
                                 feature_cols=None, y_scaler=None, no_scaling=False,
                                 baseline_models=None):
    """Compute additional baseline models: TabPFN, EBM, SHAP-based model, LLM-LEx
    
    Args:
        baseline_models: List of baseline model names to compute. If None, compute all available.
                        Options: 'tabpfn', 'ebm', 'shap', 'llmlex'
    
    Returns:
        Dict with baseline metrics for each model
    """
    baselines = {}
    
    # Default: compute all baselines if not specified
    if baseline_models is None:
        baseline_models = ['tabpfn', 'ebm', 'shap', 'llmlex']
    else:
        # Normalize to lowercase
        baseline_models = [m.lower() for m in baseline_models]
    
    # TabPFN baseline (runs in subprocess to avoid segmentation faults)
    if 'tabpfn' in baseline_models and _HAS_TABPFN and task_type == "regression":
        try:
            logger.info("Computing TabPFN baseline...")
            logger.info("  ℹ️  Note: TabPFN runs in isolated subprocess to avoid crashes")
            
            import subprocess
            import tempfile
            from pathlib import Path
            import sys
            import pandas as pd
            
            # Create temporary directory for data files
            temp_dir = Path(tempfile.mkdtemp(prefix="tabpfn_baseline_"))
            
            # Save data to files
            X_train_df = pd.DataFrame(X_train, columns=feature_cols if feature_cols else [f"feature_{i}" for i in range(X_train.shape[1])])
            X_test_df = pd.DataFrame(X_test, columns=feature_cols if feature_cols else [f"feature_{i}" for i in range(X_test.shape[1])])
            
            X_train_path = temp_dir / "X_train.csv"
            y_train_path = temp_dir / "y_train.npy"
            X_test_path = temp_dir / "X_test.csv"
            output_path = temp_dir / "predictions.json"
            
            X_train_df.to_csv(X_train_path, index=False)
            np.save(y_train_path, y_train)
            X_test_df.to_csv(X_test_path, index=False)
            
            # Find subprocess script
            script_path = Path(__file__).parent / "run_tabpfn_subprocess.py"
            if not script_path.exists():
                raise RuntimeError(f"TabPFN subprocess script not found at {script_path}")
            
            # Run TabPFN in subprocess
            result = subprocess.run(
                [sys.executable, str(script_path), str(X_train_path), str(y_train_path), 
                 str(X_test_path), str(output_path), task_type],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                raise RuntimeError(f"TabPFN subprocess failed: {error_msg}")
            
            if not output_path.exists():
                raise RuntimeError("TabPFN subprocess completed but no output file found")
            
            # Load predictions
            with open(output_path, 'r') as f:
                pred_result = json.load(f)
            
            if not pred_result.get('success', False):
                error_msg = pred_result.get('error', 'Unknown error')
                raise RuntimeError(f"TabPFN failed: {error_msg}")
            
            y_pred = np.array(pred_result['predictions'], dtype=float)
            
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
            # Clip if scaled
            if not no_scaling and y_scaler is not None:
                y_pred = np.clip(y_pred, SCALE_MIN, SCALE_MAX)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            baselines['tabpfn'] = {
                'r2': float(r2),
                'mae': float(mae),
                'mse': float(mse),
                'predictions': y_pred.tolist()
            }
            logger.info(f"  TabPFN: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
        except Exception as e:
            import traceback
            logger.warning(f"Failed to compute TabPFN baseline: {e}")
            logger.debug(f"TabPFN error details: {traceback.format_exc()}")
            # Check if it's a model download issue
            if "huggingface" in str(e).lower() or "model" in str(e).lower():
                logger.warning("  TabPFN model weights may not be downloaded. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5 to accept license and run 'hf auth login'")
    
    # EBM baseline
    if 'ebm' in baseline_models and _HAS_EBM:
        try:
            logger.info("Computing EBM baseline...")
            # Suppress verbose EBM logs
            import logging
            interpret_logger = logging.getLogger('interpret')
            original_level = interpret_logger.level
            interpret_logger.setLevel(logging.WARNING)
            
            try:
                if task_type == "regression":
                    model = ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=1)
                else:
                    from interpret.glassbox import ExplainableBoostingClassifier
                    model = ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=1)
                
                model.fit(X_train, y_train)
            finally:
                # Restore original logging level
                interpret_logger.setLevel(original_level)
            y_pred = model.predict(X_test)
            
            # Clip if scaled
            if not no_scaling and y_scaler is not None:
                y_pred = np.clip(y_pred, SCALE_MIN, SCALE_MAX)
            
            if task_type == "regression":
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                baselines['ebm'] = {
                    'r2': float(r2),
                    'mae': float(mae),
                    'mse': float(mse),
                    'predictions': y_pred.tolist()
                }
                logger.info(f"  EBM: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
            else:
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                baselines['ebm'] = {
                    'accuracy': float(acc),
                    'f1': float(f1),
                    'predictions': y_pred.tolist()
                }
                logger.info(f"  EBM: ACC={acc:.4f}, F1={f1:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute EBM baseline: {e}")
    
    # SHAP-based baseline (using XGBoost with SHAP feature selection)
    if 'shap' in baseline_models and _HAS_SHAP and _HAS_XGB:
        try:
            logger.info("Computing SHAP-based baseline...")
            from xgboost import XGBRegressor, XGBClassifier
            
            if task_type == "regression":
                base_model = XGBRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, n_jobs=1)
            else:
                base_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss')
            
            base_model.fit(X_train, y_train)
            
            # Use SHAP to select top features
            try:
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_train[:min(100, len(X_train))])  # Sample for speed
                
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                if len(shap_values.shape) > 2:
                    shap_values = np.abs(shap_values).mean(axis=0)
                else:
                    shap_values = np.abs(shap_values)
                
                # Get feature importance from SHAP
                feature_importance = np.abs(shap_values).mean(axis=0) if len(shap_values.shape) > 1 else np.abs(shap_values)
                top_features_idx = np.argsort(feature_importance)[-min(10, len(feature_importance)):]
                
                # Retrain on top features
                X_train_selected = X_train[:, top_features_idx]
                X_test_selected = X_test[:, top_features_idx]
                
                if task_type == "regression":
                    model = XGBRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, n_jobs=1)
                else:
                    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss')
                
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                
                # Clip if scaled
                if not no_scaling and y_scaler is not None:
                    y_pred = np.clip(y_pred, SCALE_MIN, SCALE_MAX)
                
                if task_type == "regression":
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    baselines['shap'] = {
                        'r2': float(r2),
                        'mae': float(mae),
                        'mse': float(mse),
                        'predictions': y_pred.tolist()
                    }
                    logger.info(f"  SHAP: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
                else:
                    from sklearn.metrics import accuracy_score, f1_score
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    baselines['shap'] = {
                        'accuracy': float(acc),
                        'f1': float(f1),
                        'predictions': y_pred.tolist()
                    }
                    logger.info(f"  SHAP: ACC={acc:.4f}, F1={f1:.4f}")
            except Exception as e:
                logger.warning(f"SHAP feature selection failed, using full model: {e}")
                # Fallback: use full model without SHAP selection
                y_pred = base_model.predict(X_test)
                if not no_scaling and y_scaler is not None:
                    y_pred = np.clip(y_pred, SCALE_MIN, SCALE_MAX)
                
                if task_type == "regression":
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    baselines['shap'] = {
                        'r2': float(r2),
                        'mae': float(mae),
                        'mse': float(mse),
                        'predictions': y_pred.tolist()
                    }
                else:
                    from sklearn.metrics import accuracy_score, f1_score
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    baselines['shap'] = {
                        'accuracy': float(acc),
                        'f1': float(f1),
                        'predictions': y_pred.tolist()
                    }
        except Exception as e:
            logger.warning(f"Failed to compute SHAP baseline: {e}")
    
    # LLM-LEx symbolic regression baseline
    if 'llmlex' in baseline_models and _HAS_LLMLEX and task_type == "regression":
        try:
            logger.info("Computing LLM-LEx symbolic regression baseline...")
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LinearRegression
            
            # Load environment variables
            _load_env()
            
            # Check for API key
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not found in .env file")
            
            # Create OpenAI client (matches llmlex documentation pattern)
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            
            # Reduce to 1D for symbolic regression using PCA
            if X_train.shape[1] > 1:
                logger.info("  Using PCA to reduce features to 1D for symbolic regression")
                pca = PCA(n_components=1)
                x_train_1d = pca.fit_transform(X_train).ravel()
                x_test_1d = pca.transform(X_test).ravel()
            else:
                x_train_1d = X_train.ravel()
                x_test_1d = X_test.ravel()
            
            # Generate base64 image (matches llmlex documentation pattern)
            logger.info("  Generating visualization for llmlex...")
            fig, ax = plt.subplots()
            ax.scatter(x_train_1d, y_train)
            base64_img = llmlex.images.generate_base64_image(fig, ax, x_train_1d, y_train)
            plt.close(fig)
            
            # Try genetic algorithm first, fallback to single_call
            best_result = None
            best_expression = None
            best_params = None
            
            try:
                logger.info("  Running llmlex genetic algorithm...")
                populations = llmlex.run_genetic(
                    client, base64_img, x_train_1d, y_train,
                    population_size=5,
                    num_of_generations=3,
                    model="openai/gpt-4o"
                )
                
                if populations and len(populations) > 0:
                    last_gen = populations[-1]
                    if last_gen and len(last_gen) > 0:
                        best_result = min(last_gen, key=lambda x: x.get('score', float('inf')))
                        best_expression = best_result.get('ansatz', None)
                        best_params = best_result.get('params', {})
                        logger.info(f"  Best expression: {best_expression}")
            except Exception as e:
                logger.warning(f"  Genetic algorithm failed: {e}, trying single_call...")
                best_result = None
            
            if best_result is None:
                logger.info("  Running llmlex single_call...")
                result = llmlex.single_call(client, base64_img, x_train_1d, y_train, model="openai/gpt-4o")
                best_expression = result.get('ansatz', None)
                best_params = result.get('params', {})
                logger.info(f"  Expression: {best_expression}")
            
            # Try to evaluate the symbolic expression using sympy
            y_pred = None
            try:
                try:
                    import sympy as sp
                    from sympy.parsing.sympy_parser import parse_expr
                except ImportError:
                    raise ImportError("sympy not installed")
                
                expr_str = str(best_expression)
                # Replace parameter placeholders with actual values
                if best_params:
                    for param_name, param_value in best_params.items():
                        expr_str = expr_str.replace(param_name, str(param_value))
                
                # Create symbol for x
                x_sym = sp.Symbol('x')
                # Try to parse the expression
                try:
                    expr = parse_expr(expr_str.replace('x', 'x_sym'), transformations='all')
                except:
                    expr = parse_expr(expr_str, transformations='all')
                
                # Evaluate on test set
                y_pred = np.array([float(expr.subs(x_sym, x_val)) for x_val in x_test_1d])
                logger.info("  Successfully evaluated symbolic expression")
            except Exception as eval_error:
                logger.warning(f"  Failed to evaluate symbolic expression: {eval_error}")
                logger.warning("  Falling back to linear fit on 1D projection")
                # Fallback: use linear regression on 1D projection
                simple_model = LinearRegression()
                simple_model.fit(x_train_1d.reshape(-1, 1), y_train)
                y_pred = simple_model.predict(x_test_1d.reshape(-1, 1))
            
            # Clip if scaled
            if not no_scaling and y_scaler is not None:
                y_pred = np.clip(y_pred, SCALE_MIN, SCALE_MAX)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            baselines['llmlex'] = {
                'r2': float(r2),
                'mae': float(mae),
                'mse': float(mse),
                'predictions': y_pred.tolist(),
                'expression': str(best_expression) if best_expression else None,
                'params': best_params if best_params else None
            }
            logger.info(f"  LLM-LEx: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
        except Exception as e:
            import traceback
            logger.warning(f"Failed to compute LLM-LEx baseline: {e}")
            logger.debug(f"LLM-LEx error details: {traceback.format_exc()}")
            if "api" in str(e).lower() or "key" in str(e).lower():
                logger.warning("  LLM-LEx requires OPENROUTER_API_KEY in .env file")
    
    return baselines


def save_residual_plot(y_true, y_pred, title, filename, output_dir):
    """Generate and save a residual plot."""
    if not _HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed, skipping residual plot")
        return
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title(f'{title} - Residuals vs Predicted', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{title} - Residual Distribution', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved residual plot to {save_path}")


def load_gfp_yield_dataset(max_samples: int):
    """Load GFP Yield Prediction dataset (real experimental data)
    
    Preprocessing matches the reference implementation:
    1. Select numeric columns and drop NA
    2. Group by 'Experiment no' and aggregate (take first row, replace Yield with mean)
    3. Drop 'Experiment no' column
    4. Extract features and target
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    try:
        csv_path = "MohammadFiles/2025_07_25/2025_07_25_GFP_Results_removed_outliers.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Step 1: Select numeric columns and drop NA (before grouping)
        # Note: We need to preserve 'Experiment no' for grouping even if it's numeric
        logger.info(f"Raw dataset shape: {df.shape}, columns: {list(df.columns)}")
        
        # Check if 'Experiment no' exists and its type
        has_experiment_no = 'Experiment no' in df.columns
        if has_experiment_no:
            exp_no_dtype = df['Experiment no'].dtype
            logger.info(f"'Experiment no' column found (dtype: {exp_no_dtype})")
        
        # Select numeric columns (this will include 'Experiment no' if it's numeric)
        df = df.select_dtypes(include=[np.number]).dropna()
        logger.info(f"After selecting numeric columns: {df.shape}, columns: {list(df.columns)}")
        
        # Step 2: Group by 'Experiment no' and aggregate
        if 'Experiment no' in df.columns:
            logger.info(f"Grouping by 'Experiment no': {df['Experiment no'].nunique()} unique experiments")
            logger.info(f"  Samples per experiment: min={df.groupby('Experiment no').size().min()}, "
                       f"max={df.groupby('Experiment no').size().max()}, "
                       f"mean={df.groupby('Experiment no').size().mean():.2f}")
            
            def aggregate_group(group):
                """Aggregate function: take first row, replace Yield with mean"""
                row = group.iloc[0].copy()  # take first row as base
                row['Yield'] = group['Yield'].mean()  # replace yield with mean
                return row
            
            # Apply aggregation (dropna() in case any groups produce NaN)
            aggregated_rows = df.groupby('Experiment no').apply(aggregate_group).dropna().reset_index(drop=True)
            df = aggregated_rows
            logger.info(f"After aggregation: {df.shape} samples")
            
            # Step 3: Drop 'Experiment no' column (it's not a feature)
            df = df.drop(columns=['Experiment no'])
        else:
            logger.warning("'Experiment no' column not found after selecting numeric columns - skipping aggregation")
            logger.warning("  This may indicate 'Experiment no' is non-numeric or was removed during dropna()")
        
        # Step 4: Extract features and target
        feature_cols = [col for col in df.columns if col != 'Yield']
        if not feature_cols:
            raise ValueError("No feature columns found")
        X_df = df[feature_cols].astype(float)
        y_series = df["Yield"].astype(float)
        
        # Log data statistics for debugging
        logger.info(f"Final dataset: {len(X_df)} samples, {len(feature_cols)} features")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Target (Yield) statistics: min={y_series.min():.4f}, max={y_series.max():.4f}, mean={y_series.mean():.4f}, std={y_series.std():.4f}")
        
        # Check for data quality issues
        constant_features = [col for col in feature_cols if X_df[col].nunique() <= 1]
        if constant_features:
            logger.warning(f"⚠️  Constant features detected (will cause issues): {constant_features}")
        
        # Check for low variance features
        low_variance_features = [col for col in feature_cols if X_df[col].std() < 1e-6]
        if low_variance_features:
            logger.warning(f"⚠️  Low variance features detected: {low_variance_features}")
        
        # Check feature correlations (warn if perfect correlation)
        if len(feature_cols) > 1:
            corr_matrix = X_df[feature_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.99:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            if high_corr_pairs:
                logger.warning(f"⚠️  Highly correlated feature pairs (>0.99): {high_corr_pairs}")
        
        logger.info(f"Feature statistics (min/max/mean/std):")
        for col in feature_cols[:5]:  # Show first 5 features
            logger.info(f"  {col}: min={X_df[col].min():.4f}, max={X_df[col].max():.4f}, mean={X_df[col].mean():.4f}, std={X_df[col].std():.4f}")
    except Exception as e:
        logger.warning(f"Failed to load GFP yield data: {e}")
        raise RuntimeError(f"GFP yield dataset not available: {e}")
    
    # Apply max_samples subsampling if needed (handled by _subsample_if_needed)
    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
    X_encoded = X_df.values
    y_values = y_series.values
    X_original = [X_df.iloc[i].astype(float).to_dict() for i in range(len(X_df))]
    feature_encoders: Dict[str, Any] = {}
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, "GFP Yield Prediction"


def get_available_protein_plates():
    """Get list of available protein expression plate files"""
    try:
        import glob
        data_dir = "/Users/mohammadrezarezaei/Desktop/Vector-stuff/ProtienP/active_learning_cell_free/whole_lysate_most_informative_points/data/no_controls"
        csv_files = glob.glob(os.path.join(data_dir, "plate_AL_*_raw_yield_and_std.csv"))
        # Return just the filenames, sorted
        plate_files = [os.path.basename(f) for f in sorted(csv_files)]
        return plate_files
    except Exception as e:
        logger.warning(f"Error getting available plates: {e}")
        return []


def load_protein_expression_dataset(max_samples: int, plate_file: str = None, plate_index: int = None):
    """Load protein expression data from a specific CSV file
    
    Args:
        max_samples: Maximum number of samples to load
        plate_file: Name of the plate file (e.g., 'plate_AL_1_raw_yield_and_std.csv')
        plate_index: Index of the plate (0-based) if plate_file is not provided
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    try:
        import glob
        data_dir = "/Users/mohammadrezarezaei/Desktop/Vector-stuff/ProtienP/active_learning_cell_free/whole_lysate_most_informative_points/data/no_controls"
        csv_files = sorted(glob.glob(os.path.join(data_dir, "plate_AL_*_raw_yield_and_std.csv")))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        if plate_file is None:
            if plate_index is not None:
                if plate_index < 0 or plate_index >= len(csv_files):
                    available = get_available_protein_plates()
                    raise ValueError(f"Plate index {plate_index} out of range. Available indices: 0-{len(csv_files)-1}\n"
                                   f"Available plates: {available}")
                plate_file = csv_files[plate_index]
                logger.info(f"Using plate index {plate_index}: {os.path.basename(plate_file)}")
            else:
                plate_file = csv_files[0]
                logger.info(f"Using default file (first available): {os.path.basename(plate_file)}")
        else:
            if not os.path.isabs(plate_file):
                plate_file = os.path.join(data_dir, plate_file)
            if not os.path.exists(plate_file):
                available = get_available_protein_plates()
                raise FileNotFoundError(f"File not found: {plate_file}\n"
                                       f"Available plates: {available}")
        df = pd.read_csv(plate_file)
        feature_cols = [
            'nad', 'folinic_acid', 'coa', 'nucleo_mix', 'spermidin', 
            'pga', 'aa', 'trna', 'mg_gluta', 'camp', 'K_gluta'
        ]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        X_df = df[feature_cols].astype(float)
        y_series = df['yield'].astype(float)
        valid_mask = ~(np.isnan(X_df).any(axis=1) | np.isnan(y_series))
        X_df = X_df[valid_mask].reset_index(drop=True)
        y_series = y_series[valid_mask].reset_index(drop=True)
        plate_name = os.path.basename(plate_file).replace('_raw_yield_and_std.csv', '')
        dataset_name = f"Protein Expression ({plate_name})"
    except Exception as e:
        logger.warning(f"Failed to load protein expression data: {e}")
        raise RuntimeError(f"Protein expression dataset not available: {e}")
    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
    X_encoded = X_df.values
    y_values = y_series.values
    X_original = [X_df.iloc[i].astype(float).to_dict() for i in range(len(X_df))]
    feature_encoders: Dict[str, Any] = {}
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_name


def load_protein_expression_all_plates_dataset(max_samples: int):
    """Load and combine data from all protein expression plates"""
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    try:
        import glob
        data_dir = "/Users/mohammadrezarezaei/Desktop/Vector-stuff/ProtienP/active_learning_cell_free/whole_lysate_most_informative_points/data/no_controls"
        csv_files = glob.glob(os.path.join(data_dir, "plate_AL_*_raw_yield_and_std.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        all_data = []
        feature_cols = [
            'nad', 'folinic_acid', 'coa', 'nucleo_mix', 'spermidin', 
            'pga', 'aa', 'trna', 'mg_gluta', 'camp', 'K_gluta'
        ]
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                if all(col in df.columns for col in feature_cols + ['yield']):
                    plate_data = df[feature_cols + ['yield']].copy()
                    all_data.append(plate_data)
                    logger.info(f"Loaded {len(plate_data)} samples from {os.path.basename(csv_file)}")
            except Exception as e:
                logger.warning(f"Error loading {os.path.basename(csv_file)}: {e}")
                continue
        if not all_data:
            raise ValueError("No valid data loaded")
        combined_df = pd.concat(all_data, ignore_index=True)
        X_df = combined_df[feature_cols].astype(float)
        y_series = combined_df['yield'].astype(float)
        valid_mask = ~(np.isnan(X_df).any(axis=1) | np.isnan(y_series))
        X_df = X_df[valid_mask].reset_index(drop=True)
        y_series = y_series[valid_mask].reset_index(drop=True)
        dataset_name = "Combined Protein Expression"
    except Exception as e:
        logger.warning(f"Failed to load combined protein expression data: {e}")
        raise RuntimeError(f"Combined protein expression dataset not available: {e}")
    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
    X_encoded = X_df.values
    y_values = y_series.values
    X_original = [X_df.iloc[i].astype(float).to_dict() for i in range(len(X_df))]
    feature_encoders: Dict[str, Any] = {}
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_name


def load_dataset_102(max_samples: int):
    """Load Dataset 102 (train/test split) for protein expression"""
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    try:
        data_dir = "/Users/mohammadrezarezaei/Desktop/Vector-stuff/ProtienP/active_learning_cell_free/whole_lysate_most_informative_points/full_on_102"
        train_file = os.path.join(data_dir, "train.csv")
        test_file = os.path.join(data_dir, "test.csv")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Train or test file not found in {data_dir}")
        df_train = pd.read_csv(train_file, comment='#', header=None)
        df_test = pd.read_csv(test_file, comment='#', header=None)
        column_names = [
            'nad', 'folinic_acid', 'coa', 'nucleo_mix', 'spermidin', 
            'pga', 'aa', 'trna', 'mg_gluta', 'camp', 'K_gluta',
            'y_data', 'y_data_std', 'y_pr', 'y_std_pr'
        ]
        df_train.columns = column_names
        df_test.columns = column_names
        feature_cols = [
            'nad', 'folinic_acid', 'coa', 'nucleo_mix', 'spermidin', 
            'pga', 'aa', 'trna', 'mg_gluta', 'camp', 'K_gluta'
        ]
        X_train = df_train[feature_cols].astype(float)
        y_train = df_train['y_data'].astype(float)
        X_test = df_test[feature_cols].astype(float)
        y_test = df_test['y_data'].astype(float)
        X = np.vstack([X_train.values, X_test.values])
        y = np.hstack([y_train.values, y_test.values])
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        X_df = pd.DataFrame(X, columns=feature_cols)
        y_series = pd.Series(y, name="target")
    except Exception as e:
        logger.warning(f"Failed to load dataset 102: {e}")
        raise RuntimeError(f"Dataset 102 not available: {e}")
    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
    X_encoded = X_df.values
    y_values = y_series.values
    X_original = [X_df.iloc[i].astype(float).to_dict() for i in range(len(X_df))]
    feature_encoders: Dict[str, Any] = {}
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, "Dataset 102"


def _try_parse_float(x):
    """Best-effort conversion to float; returns None if not possible."""
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            v = float(x)
            if np.isnan(v):
                return None
            return v
        s = str(x).strip()
        if not s:
            return None
        # Handle things like "7.0", "N/A (1 dead)", "1 dead in food", etc.
        # Only accept pure numeric tokens.
        s2 = s.replace(",", "")
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", s2):
            v = float(s2)
            if np.isnan(v):
                return None
            return v
        return None
    except Exception:
        return None


def _extract_dah_series_from_row(row_dict: Dict[str, Any], prefix: str):
    """
    Extract a (days, values) series from a row dict where columns look like:
      "Pupation DAH 5", "Eclosure DAH 12", etc.
    Returns (days_sorted, values_sorted) with numeric values only.
    """
    days = []
    vals = []
    for k, v in row_dict.items():
        if not isinstance(k, str):
            continue
        if prefix not in k:
            continue
        # Column examples: "Pupation DAH 5", "Eclosure DAH 12"
        m = re.search(r"DAH\s*(\d+)", k)
        if not m:
            continue
        day = int(m.group(1))
        fv = _try_parse_float(v)
        if fv is None:
            continue
        days.append(day)
        vals.append(fv)
    if not days:
        return [], []
    order = np.argsort(days)
    days_sorted = [int(days[i]) for i in order]
    vals_sorted = [float(vals[i]) for i in order]
    return days_sorted, vals_sorted


def _time_to_fraction(days, cum_values, frac: float = 0.5):
    """
    Given cumulative counts, return the earliest day where cum >= frac * final.
    Uses linear interpolation between adjacent DAH points when needed.
    Returns float day, or None if cannot compute.
    """
    if not days or not cum_values or len(days) != len(cum_values):
        return None
    final = cum_values[-1]
    if final is None or final <= 0:
        return None
    target = frac * final
    for i, y in enumerate(cum_values):
        if y >= target:
            if i == 0:
                return float(days[0])
            x0, y0 = days[i - 1], cum_values[i - 1]
            x1, y1 = days[i], cum_values[i]
            if y1 == y0:
                return float(x1)
            # interpolate day when reaching target
            t = (target - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return float(days[-1])


def load_inadata_pfas_su_dataset(max_samples: int, target: str = "eclosure_total"):
    """
    Load InaData SU PFAS developmental assay from CSV.

    Source: InaData/PFAS Dev Assay Data SU.csv
    Design: InaData/PFAS Dev Assay Exp Design SU.docx

    The CSV contains multiple blocks with repeated headers.
    We parse all blocks and compute per-(chemical, concentration, replicate) summary metrics.

    Targets (target arg):
      - pupation_total: final cumulative pupation count
      - pupation_t50: day when cumulative pupation reaches 50% of final (timing)
      - eclosure_total: final cumulative eclosure total (Sex == Total)
      - eclosure_t50: day when cumulative eclosure reaches 50% of final (timing; Sex == Total)
      - female_ratio_final: female / total at final day (requires Female+Total rows)
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")

    base_dir = Path(__file__).resolve().parent / "InaData"
    csv_path = base_dir / "PFAS Dev Assay Data SU.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"InaData SU CSV not found: {csv_path}")

    # Read raw CSV rows (keeps repeated header blocks)
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = [r for r in reader]

    def _trim_and_make_unique(cols):
        # Trim trailing empty columns (common due to ",,,")
        cols = [("" if c is None else str(c).strip()) for c in cols]
        last = -1
        for i in range(len(cols) - 1, -1, -1):
            if cols[i] != "":
                last = i
                break
        if last >= 0:
            cols = cols[: last + 1]
        else:
            cols = []

        # Ensure uniqueness (pandas concat requires unique column Index)
        seen = {}
        out = []
        for i, c in enumerate(cols):
            name = c if c != "" else f"unnamed_{i}"
            if name in seen:
                seen[name] += 1
                name = f"{name}.{seen[name]}"
            else:
                seen[name] = 0
            out.append(name)
        return out

    # Identify header rows for blocks: starts with "Chemical" and includes "Replicate n"
    header_idxs = []
    for i, r in enumerate(rows):
        if not r:
            continue
        if len(r) >= 3 and str(r[0]).strip() == "Chemical" and any(str(c).strip() == "Replicate n" for c in r):
            header_idxs.append(i)
    if not header_idxs:
        raise ValueError("Could not find any data blocks in SU CSV (no 'Chemical, ... , Replicate n' header rows).")

    # Parse each block into a DataFrame
    block_dfs = []
    for hi, hidx in enumerate(header_idxs):
        header = _trim_and_make_unique(rows[hidx])
        if not header:
            continue
        # Data rows until next blank row or next header
        data = []
        j = hidx + 1
        while j < len(rows):
            r = rows[j]
            # stop at next header
            if j in header_idxs:
                break
            # stop at empty row
            if not r or all((str(x).strip() == "" for x in r)):
                break
            data.append(r)
            j += 1

        if not data:
            continue

        # Normalize row lengths to header length
        norm = []
        for r in data:
            rr = list(r)
            # Trim to header length, or pad if shorter
            if len(rr) > len(header):
                rr = rr[: len(header)]
            elif len(rr) < len(header):
                rr = rr + [""] * (len(header) - len(rr))
            norm.append(rr)

        df = pd.DataFrame(norm, columns=header)
        # Strip whitespace for key columns
        for col in ["Chemical", "Concentration", "Replicate n", "Sex"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        # Try numeric conversion for Replicate n
        if "Replicate n" in df.columns:
            df["Replicate n"] = pd.to_numeric(df["Replicate n"], errors="coerce")
        block_dfs.append(df)

    if not block_dfs:
        raise ValueError("SU CSV parsed but produced no non-empty blocks.")

    su_all = pd.concat(block_dfs, ignore_index=True)

    # Separate pupation and eclosure blocks by presence of "Sex" column and DAH prefixes
    # Pupation blocks: columns contain "Pupation DAH"
    # Eclosure blocks: columns contain "Eclosure DAH"
    pup_mask = [any(isinstance(c, str) and "Pupation DAH" in c for c in su_all.columns)]
    # We'll just detect by column names on the combined DF, then per-row extraction decides actual availability.

    # Build per-sample aggregated table keyed by chemical/concentration/replicate
    group_cols = [c for c in ["Chemical", "Concentration", "Replicate n"] if c in su_all.columns]
    if len(group_cols) < 3:
        raise ValueError(f"SU CSV missing required key columns. Found columns: {list(su_all.columns)}")

    # For eclosure, we want Total/Female rows grouped; for pupation, no Sex.
    records = []
    for (chem, conc, rep), g in su_all.groupby(group_cols, dropna=False):
        # Pupation row(s): pick any row that has Pupation DAH values
        pup_rows = []
        ecl_total_rows = []
        ecl_female_rows = []

        for _, row in g.iterrows():
            rowd = row.to_dict()
            # Pupation series
            d_p, v_p = _extract_dah_series_from_row(rowd, "Pupation DAH")
            if d_p:
                pup_rows.append((d_p, v_p, rowd))
            # Eclosure series
            d_e, v_e = _extract_dah_series_from_row(rowd, "Eclosure DAH")
            if d_e:
                sex = str(rowd.get("Sex", "")).strip()
                if sex.lower() == "total":
                    ecl_total_rows.append((d_e, v_e, rowd))
                elif sex.lower() == "female":
                    ecl_female_rows.append((d_e, v_e, rowd))

        # Helper to pick the "best" row: longest series, then highest final
        def pick_best(rows_list):
            if not rows_list:
                return None
            def keyfn(t):
                d, v, _ = t
                return (len(d), v[-1] if v else -1)
            return sorted(rows_list, key=keyfn, reverse=True)[0]

        pup_best = pick_best(pup_rows)
        ecl_total_best = pick_best(ecl_total_rows)
        ecl_female_best = pick_best(ecl_female_rows)

        pup_total = None
        pup_t50 = None
        if pup_best:
            d, v, _ = pup_best
            pup_total = float(v[-1])
            pup_t50 = _time_to_fraction(d, v, frac=0.5)

        ecl_total = None
        ecl_t50 = None
        if ecl_total_best:
            d, v, _ = ecl_total_best
            ecl_total = float(v[-1])
            ecl_t50 = _time_to_fraction(d, v, frac=0.5)

        female_ratio_final = None
        if ecl_total_best and ecl_female_best:
            dT, vT, _ = ecl_total_best
            dF, vF, _ = ecl_female_best
            # Ratio at final day (use each series' final value)
            if vT and vT[-1] and vT[-1] > 0:
                female_ratio_final = float(vF[-1]) / float(vT[-1])

        rec = {
            "chemical": str(chem).strip(),
            "concentration_label": str(conc).strip(),
            "replicate": float(rep) if rep is not None and not (isinstance(rep, float) and np.isnan(rep)) else None,
            "pupation_total": pup_total,
            "pupation_t50": pup_t50,
            "eclosure_total": ecl_total,
            "eclosure_t50": ecl_t50,
            "female_ratio_final": female_ratio_final,
        }
        records.append(rec)

    df = pd.DataFrame(records)

    # Drop rows without the chosen target
    if target not in df.columns:
        raise ValueError(f"Unknown SU target '{target}'. Available: {list(df.columns)}")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"SU dataset has 0 usable rows after filtering for target={target}")

    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    feature_encoders: Dict[str, Any] = {}
    X_df = df[["chemical", "concentration_label", "replicate"]].copy()
    for col in ["chemical", "concentration_label"]:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        feature_encoders[col] = le
    X_df["replicate"] = pd.to_numeric(X_df["replicate"], errors="coerce").fillna(0.0)

    feature_cols = list(X_df.columns)
    y_series = df[target].astype(float)

    # Subsample if needed
    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)

    X_encoded = X_df.values.astype(float)
    y_values = y_series.values.astype(float)
    X_original = df.iloc[: len(X_df)].to_dict(orient="records")

    dataset_label = f"InaData SU PFAS Dev Assay (target={target})"
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label


def load_inadata_ec_fertility_dataset(max_samples: int, target: str = "total_eclosed"):
    """
    Load InaData EC calibration data (fertility scoring) from Excel.

    Source: InaData/Chemical Screen Calibration Data_EC.xlsx (sheet: 'Fertility Scoring')

    We compute robust summary targets per vial:
      - total_females, total_males, total_pupae, total_eclosed (= females+males)
    Default target is total_eclosed.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")

    base_dir = Path(__file__).resolve().parent / "InaData"
    xlsx_path = base_dir / "Chemical Screen Calibration Data_EC.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"InaData EC XLSX not found: {xlsx_path}")

    df_raw = pd.read_excel(xlsx_path, sheet_name="Fertility Scoring")
    if "Genotype" not in df_raw.columns:
        raise ValueError(f"EC Fertility sheet missing 'Genotype' column. Columns: {list(df_raw.columns)}")

    # Filter real vial rows
    df = df_raw.copy()
    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df = df[(df["Genotype"].notna()) & (df["Genotype"] != "") & (df["Genotype"].str.lower() != "nan")]
    df = df[~df["Genotype"].str.contains("totals", case=False, na=False)]
    df = df.reset_index(drop=True)

    # Identify numeric count columns by prefix (handle duplicated column names like '# Females.1')
    female_cols = [c for c in df.columns if isinstance(c, str) and c.strip().lower().startswith("# females")]
    male_cols = [c for c in df.columns if isinstance(c, str) and c.strip().lower().startswith("# males")]
    pupae_cols = [c for c in df.columns if isinstance(c, str) and c.strip().lower().startswith("# pupae")]

    # Some columns are garbled ("# f# Femalesemales "), include anything containing "females" that starts with '#'
    female_cols += [c for c in df.columns if isinstance(c, str) and c.strip().startswith("#") and ("female" in c.lower()) and (c not in female_cols)]

    def numeric_sum(cols):
        if not cols:
            return pd.Series([0.0] * len(df))
        s = pd.Series([0.0] * len(df))
        for c in cols:
            s = s + pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return s

    df["total_females"] = numeric_sum(female_cols)
    df["total_males"] = numeric_sum(male_cols)
    df["total_pupae"] = numeric_sum(pupae_cols)
    df["total_eclosed"] = df["total_females"] + df["total_males"]

    if target not in df.columns:
        raise ValueError(f"Unknown EC fertility target '{target}'. Available: {[c for c in df.columns if c.startswith('total_')]} + raw columns")

    # Features: genotype + vial number (if present)
    X_df = pd.DataFrame()
    X_df["genotype"] = df["Genotype"].astype(str)
    if "Vial #" in df.columns:
        X_df["vial"] = pd.to_numeric(df["Vial #"], errors="coerce").fillna(0.0)
    else:
        X_df["vial"] = 0.0

    # Encode genotype into numeric
    from sklearn.preprocessing import LabelEncoder
    feature_encoders: Dict[str, Any] = {}
    le = LabelEncoder()
    X_df["genotype"] = le.fit_transform(X_df["genotype"].astype(str))
    feature_encoders["genotype"] = le

    feature_cols = list(X_df.columns)
    y_series = pd.to_numeric(df[target], errors="coerce")

    # Drop missing targets
    valid = y_series.notna()
    X_df = X_df[valid].reset_index(drop=True)
    y_series = y_series[valid].reset_index(drop=True)
    df_valid = df[valid].reset_index(drop=True)

    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
    X_encoded = X_df.values.astype(float)
    y_values = y_series.values.astype(float)
    X_original = df_valid.iloc[: len(X_df)][["Genotype", "Vial #", "total_females", "total_males", "total_pupae", "total_eclosed"]].to_dict(orient="records")
    dataset_label = f"InaData EC Fertility (target={target})"
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label


def load_inadata_ec_climbing_dataset(max_samples: int, target: str = "avg_16s"):
    """
    Load InaData EC calibration data (climbing test) from Excel.

    Source: InaData/Chemical Screen Calibration Data_EC.xlsx (sheet: 'Climbing Test')

    Targets:
      - female_16s: % Female (16 seconds)
      - male_16s: % Male (16 seconds)
      - avg_16s: average of male_16s and female_16s
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")

    base_dir = Path(__file__).resolve().parent / "InaData"
    xlsx_path = base_dir / "Chemical Screen Calibration Data_EC.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"InaData EC XLSX not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name="Climbing Test")
    # Use the right-side columns (Genotype.1 / Vial #.1 / % Male/Female (X seconds))
    if "Genotype.1" not in df.columns:
        raise ValueError(f"EC Climbing sheet missing 'Genotype.1'. Columns: {list(df.columns)}")
    df = df.copy()
    df["Genotype.1"] = df["Genotype.1"].astype(str).str.strip()
    df = df[(df["Genotype.1"].notna()) & (df["Genotype.1"] != "") & (df["Genotype.1"].str.lower() != "nan")].reset_index(drop=True)

    male_16 = pd.to_numeric(df.get("% Male (16 seconds)"), errors="coerce")
    female_16 = pd.to_numeric(df.get("% Female (16 seconds)"), errors="coerce")

    df["male_16s"] = male_16
    df["female_16s"] = female_16
    df["avg_16s"] = (male_16 + female_16) / 2.0

    if target not in df.columns:
        raise ValueError(f"Unknown EC climbing target '{target}'. Available: male_16s, female_16s, avg_16s")

    # Features: genotype + vial
    X_df = pd.DataFrame()
    X_df["genotype"] = df["Genotype.1"].astype(str)
    X_df["vial"] = pd.to_numeric(df.get("Vial #.1"), errors="coerce").fillna(0.0)

    from sklearn.preprocessing import LabelEncoder
    feature_encoders: Dict[str, Any] = {}
    le = LabelEncoder()
    X_df["genotype"] = le.fit_transform(X_df["genotype"].astype(str))
    feature_encoders["genotype"] = le

    y_series = pd.to_numeric(df[target], errors="coerce")
    valid = y_series.notna()
    X_df = X_df[valid].reset_index(drop=True)
    y_series = y_series[valid].reset_index(drop=True)
    df_valid = df[valid].reset_index(drop=True)

    feature_cols = list(X_df.columns)
    X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
    X_encoded = X_df.values.astype(float)
    y_values = y_series.values.astype(float)
    X_original = df_valid.iloc[: len(X_df)][["Genotype.1", "Vial #.1", "male_16s", "female_16s", "avg_16s"]].to_dict(orient="records")
    dataset_label = f"InaData EC Climbing (target={target})"
    return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label


def _extract_sequence_features(seq: str) -> Dict[str, float]:
    """Extract numerical features from protein sequence"""
    if pd.isna(seq) or not seq:
        return {
            'seq_length': 0.0,
            'seq_gc_content': 0.0,
            'seq_aromatic_count': 0.0,
            'seq_charged_count': 0.0,
            'seq_polar_count': 0.0,
            'seq_hydrophobic_count': 0.0
        }
    seq = str(seq).upper()
    aromatic = set('FWY')
    charged = set('DEKRH')
    polar = set('STNQ')
    hydrophobic = set('AILMV')
    
    return {
        'seq_length': float(len(seq)),
        'seq_gc_content': float((seq.count('G') + seq.count('C')) / len(seq)) if len(seq) > 0 else 0.0,
        'seq_aromatic_count': float(sum(1 for aa in seq if aa in aromatic)),
        'seq_charged_count': float(sum(1 for aa in seq if aa in charged)),
        'seq_polar_count': float(sum(1 for aa in seq if aa in polar)),
        'seq_hydrophobic_count': float(sum(1 for aa in seq if aa in hydrophobic))
    }


def _extract_smiles_features(smiles: str) -> Dict[str, float]:
    """Extract numerical features from SMILES string"""
    if pd.isna(smiles) or not smiles:
        return {
            'smiles_length': 0.0,
            'smiles_ring_count': 0.0,
            'smiles_branch_count': 0.0,
            'smiles_double_bond_count': 0.0,
            'smiles_triple_bond_count': 0.0,
            'smiles_aromatic_count': 0.0
        }
    smiles = str(smiles)
    return {
        'smiles_length': float(len(smiles)),
        'smiles_ring_count': float(smiles.count('1') + smiles.count('2') + smiles.count('3') + 
                                   smiles.count('4') + smiles.count('5') + smiles.count('6') + 
                                   smiles.count('7') + smiles.count('8') + smiles.count('9')),
        'smiles_branch_count': float(smiles.count('(') + smiles.count(')')) / 2.0,
        'smiles_double_bond_count': float(smiles.count('=')),
        'smiles_triple_bond_count': float(smiles.count('#')),
        'smiles_aromatic_count': float(sum(1 for c in smiles if c.islower()))
    }


def _compute_rdkit_features(smiles: str) -> Dict[str, float]:
    """
    Compute RDKit molecular features from SMILES string.
    
    Returns:
        Dictionary with RDKit molecular descriptors
    """
    if not _HAS_RDKIT:
        return {
            'rdkit_molecular_weight': 0.0,
            'rdkit_num_rings': 0.0,
            'rdkit_num_hydroxyl_groups': 0.0,
            'rdkit_num_halogen': 0.0,
            'rdkit_num_nitrogen': 0.0,
            'rdkit_num_oxygen': 0.0,
            'rdkit_num_atoms': 0.0,
            'rdkit_is_aromatic': 0.0,
            'rdkit_num_hbd': 0.0
        }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'rdkit_molecular_weight': 0.0,
                'rdkit_num_rings': 0.0,
                'rdkit_num_hydroxyl_groups': 0.0,
                'rdkit_num_halogen': 0.0,
                'rdkit_num_nitrogen': 0.0,
                'rdkit_num_oxygen': 0.0,
                'rdkit_num_atoms': 0.0,
                'rdkit_is_aromatic': 0.0,
                'rdkit_num_hbd': 0.0
            }
        
        # Compute molecular weight
        mw = Descriptors.MolWt(mol)
        
        # Count rings
        num_rings = Descriptors.RingCount(mol)
        
        # Count atoms
        num_atoms = mol.GetNumAtoms()
        
        # Count specific atom types
        num_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        
        # Count halogens (F, Cl, Br, I)
        num_halogen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])
        
        # Count hydroxyl groups (-OH)
        num_hydroxyl = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                # Check if this oxygen is part of a hydroxyl group
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if 'H' in neighbors:
                    num_hydroxyl += 1
        
        # Check aromaticity
        is_aromatic = 1.0 if Descriptors.NumAromaticRings(mol) > 0 else 0.0
        
        # Count hydrogen bond donors (N-H, O-H)
        num_hbd = Descriptors.NumHDonors(mol)
        
        return {
            'rdkit_molecular_weight': float(mw),
            'rdkit_num_rings': float(num_rings),
            'rdkit_num_hydroxyl_groups': float(num_hydroxyl),
            'rdkit_num_halogen': float(num_halogen),
            'rdkit_num_nitrogen': float(num_nitrogen),
            'rdkit_num_oxygen': float(num_oxygen),
            'rdkit_num_atoms': float(num_atoms),
            'rdkit_is_aromatic': is_aromatic,
            'rdkit_num_hbd': float(num_hbd)
        }
    except Exception as e:
        logger.debug(f"RDKit feature computation failed for SMILES '{smiles[:50]}...': {e}")
        return {
            'rdkit_molecular_weight': 0.0,
            'rdkit_num_rings': 0.0,
            'rdkit_num_hydroxyl_groups': 0.0,
            'rdkit_num_halogen': 0.0,
            'rdkit_num_nitrogen': 0.0,
            'rdkit_num_oxygen': 0.0,
            'rdkit_num_atoms': 0.0,
            'rdkit_is_aromatic': 0.0,
            'rdkit_num_hbd': 0.0
        }


def _get_rdkit_cache_path(dataset_name: str, smiles_list: list) -> Path:
    """Generate cache file path based on dataset name and SMILES hash"""
    cache_dir = Path(__file__).parent / ".rdkit_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create hash from dataset name and first/last SMILES for uniqueness
    hash_input = f"{dataset_name}_{smiles_list[0] if smiles_list else ''}_{smiles_list[-1] if len(smiles_list) > 1 else ''}_{len(smiles_list)}"
    cache_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"rdkit_features_{dataset_name}_{cache_hash}.json"
    return cache_file


def _compute_rdkit_features_batch(smiles_list: list, dataset_name: str = "unknown", use_cache: bool = True) -> list:
    """
    Compute RDKit features for a batch of SMILES strings with caching.
    
    Args:
        smiles_list: List of SMILES strings
        dataset_name: Name of dataset (for cache file naming)
        use_cache: Whether to use cache (default: True)
        
    Returns:
        List of dictionaries with RDKit features
    """
    if not _HAS_RDKIT:
        logger.warning("RDKit not available - returning empty RDKit features")
        return [{
            'rdkit_molecular_weight': 0.0,
            'rdkit_num_rings': 0.0,
            'rdkit_num_hydroxyl_groups': 0.0,
            'rdkit_num_halogen': 0.0,
            'rdkit_num_nitrogen': 0.0,
            'rdkit_num_oxygen': 0.0,
            'rdkit_num_atoms': 0.0,
            'rdkit_is_aromatic': 0.0,
            'rdkit_num_hbd': 0.0
        } for _ in smiles_list]
    
    cache_file = _get_rdkit_cache_path(dataset_name, smiles_list)
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if 'features' in cached_data and len(cached_data['features']) == len(smiles_list):
                    logger.info(f"Loaded RDKit features from cache: {cache_file}")
                    return cached_data['features']
                else:
                    logger.info(f"Cache file exists but size mismatch - recomputing")
        except Exception as e:
            logger.warning(f"Failed to load RDKit cache: {e} - recomputing")
    
    # Compute features
    logger.info(f"Computing RDKit features for {len(smiles_list)} SMILES strings...")
    features_list = []
    for i, smiles in enumerate(smiles_list):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(smiles_list)} SMILES...")
        rdkit_features = _compute_rdkit_features(smiles)
        features_list.append(rdkit_features)
    
    logger.info(f"Computed RDKit features for {len(smiles_list)} SMILES strings")
    
    # Save to cache
    if use_cache:
        try:
            cache_data = {
                'dataset_name': dataset_name,
                'n_samples': len(smiles_list),
                'features': features_list
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached RDKit features to: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save RDKit cache: {e}")
    
    return features_list


def load_enzyme_dataset(dataset_name: str, max_samples: int):
    """Load enzyme dataset from enzyme-datasets repository
    
    Args:
        dataset_name: Name of the dataset (e.g., 'halogenase_NaBr', 'aminotransferase', 'olea')
        max_samples: Maximum number of samples to load
        
    Returns:
        X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_name
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    
    try:
        from enzyme_dataset_analysis import EnzymeDatasetLoader
        
        # Initialize loader
        data_dir = os.path.join(os.path.dirname(__file__), "enzyme-datasets", "data")
        if not os.path.exists(data_dir):
            # Try alternative path
            data_dir = "enzyme-datasets/data"
        
        # Check if enzyme-datasets directory exists
        if not os.path.exists(data_dir) and not os.path.exists("enzyme-datasets"):
            error_msg = (
                f"\n❌ Enzyme dataset directory not found!\n"
                f"   Expected location: {os.path.join(os.path.dirname(__file__), 'enzyme-datasets')}\n"
                f"   Or: enzyme-datasets/\n\n"
                f"   To fix this, clone the enzyme-datasets repository:\n"
                f"   git clone https://github.com/samgoldman97/enzyme-datasets.git\n"
                f"   Or download it to the project directory.\n"
            )
            raise FileNotFoundError(error_msg)
        
        loader = EnzymeDatasetLoader(data_dir)
        
        # List available datasets for better error messages
        available_datasets = []
        try:
            datasets_list = loader.list_available_datasets()
            if datasets_list:
                available_datasets = [d.name if hasattr(d, 'name') else str(d) for d in datasets_list]
        except Exception:
            pass
        
        # Also check processed directory for CSV files
        processed_dir = os.path.join(data_dir, "processed")
        if os.path.exists(processed_dir):
            import glob
            csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
            csv_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
            if csv_names:
                available_datasets.extend(csv_names)
                available_datasets = list(set(available_datasets))  # Remove duplicates
        
        # Load dataset - try exact match first, then case-insensitive match
        df = loader.load_dataset(dataset_name)
        loaded_via_case_insensitive = False
        if df is None:
            # Try case-insensitive matching in processed directory
            processed_dir = os.path.join(data_dir, "processed")
            if os.path.exists(processed_dir):
                import glob
                # Look for CSV files that match case-insensitively
                pattern = os.path.join(processed_dir, "*.csv")
                all_csv_files = glob.glob(pattern)
                # Find case-insensitive match
                dataset_name_lower = dataset_name.lower()
                matched_file = None
                for csv_file in all_csv_files:
                    base_name = os.path.splitext(os.path.basename(csv_file))[0]
                    if base_name.lower() == dataset_name_lower:
                        matched_file = csv_file
                        logger.info(f"Found case-insensitive match: {os.path.basename(matched_file)}")
                        break
                
                if matched_file:
                    try:
                        df = pd.read_csv(matched_file, index_col=0)
                        loaded_via_case_insensitive = True
                        logger.info(f"Loaded enzyme dataset via case-insensitive match '{os.path.basename(matched_file)}': {len(df)} samples, {len(df.columns)} columns")
                    except Exception as e:
                        logger.warning(f"Failed to load matched file {matched_file}: {e}")
                        df = None
        
        if df is None:
            error_msg = f"Could not load enzyme dataset: {dataset_name}\n"
            if available_datasets:
                error_msg += f"\nAvailable datasets:\n"
                for ds in sorted(available_datasets):
                    error_msg += f"  - {ds}\n"
            else:
                error_msg += (
                    f"\nNo datasets found in: {data_dir}\n"
                    f"Please ensure the enzyme-datasets repository is cloned and contains data files.\n"
                    f"Clone with: git clone https://github.com/samgoldman97/enzyme-datasets.git\n"
                )
            raise FileNotFoundError(error_msg)
        
        # Only log if we didn't already log via case-insensitive match
        if not loaded_via_case_insensitive:
            logger.info(f"Loaded enzyme dataset '{dataset_name}': {len(df)} samples, {len(df.columns)} columns")
        
        # Fix column names if they were incorrectly parsed (CSV files have index column)
        # Sometimes the first column name includes a comma, e.g., ',SEQ,SUBSTRATES,LogSpActivity'
        if len(df.columns) == 1 and ',' in str(df.columns[0]):
            # Split the single column name by comma and use as column names
            col_str = str(df.columns[0])
            # Remove leading comma if present
            if col_str.startswith(','):
                col_str = col_str[1:]
            new_cols = [col.strip() for col in col_str.split(',')]
            # Reload with proper column names
            processed_dir = os.path.join(data_dir, "processed")
            csv_file = os.path.join(processed_dir, f"{dataset_name}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, index_col=0)  # Use first column as index
                logger.info(f"Reloaded with index_col=0: {len(df)} samples, {len(df.columns)} columns")
        
        # Check required columns
        if 'SEQ' not in df.columns or 'SUBSTRATES' not in df.columns:
            raise ValueError(f"Dataset must have 'SEQ' and 'SUBSTRATES' columns. Found: {df.columns.tolist()}")
        
        # Find target column (exclude SEQ and SUBSTRATES)
        target_cols = [col for col in df.columns if col not in ['SEQ', 'SUBSTRATES', 'Unnamed: 0']]
        if not target_cols:
            raise ValueError(f"No target column found. Available columns: {df.columns.tolist()}")
        
        # Use first numeric column as target, or first column if none are numeric
        target_col = None
        for col in target_cols:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                target_col = col
                break
        if target_col is None:
            target_col = target_cols[0]
        
        logger.info(f"Using '{target_col}' as target column")
        
        # Extract features from sequences and SMILES (for ML training - NO RDKit features)
        logger.info("Extracting features from sequences and SMILES...")
        feature_dicts = []
        for idx, row in df.iterrows():
            seq_features = _extract_sequence_features(row['SEQ'])
            smiles_features = _extract_smiles_features(row['SUBSTRATES'])
            combined = {**seq_features, **smiles_features}
            feature_dicts.append(combined)
        
        # Create feature DataFrame (for ML training - NO RDKit features)
        X_df = pd.DataFrame(feature_dicts)
        feature_cols = X_df.columns.tolist()
        
        # Extract target
        y_series = df[target_col].astype(float)
        
        # Remove rows with missing values
        valid_mask = ~(X_df.isnull().any(axis=1) | y_series.isnull())
        X_df = X_df[valid_mask].reset_index(drop=True)
        y_series = y_series[valid_mask].reset_index(drop=True)
        
        logger.info(f"After removing missing values: {len(X_df)} samples")
        
        # Subsample if needed
        X_df, y_series = _subsample_if_needed(X_df, y_series, max_samples)
        
        # Convert to numpy arrays
        X_encoded = X_df.values.astype(float)
        y_values = y_series.values.astype(float)
        
        # Create original feature dicts (for LLM context) - WITH RDKit features
        X_original = []
        df_valid = df[valid_mask].reset_index(drop=True)
        # Match X_df with df_valid (they should have the same length after subsampling)
        df_valid_subset = df_valid.iloc[:len(X_df)].reset_index(drop=True)
        
        # Extract SMILES strings for RDKit feature computation (for LLM only)
        substrates_list = [str(df_valid_subset.iloc[idx]['SUBSTRATES']) for idx in range(len(X_df)) if idx < len(df_valid_subset)]
        logger.info(f"Computing RDKit features for enzyme dataset '{dataset_name}' ({len(substrates_list)} SMILES) - for LLM only...")
        rdkit_features_list = _compute_rdkit_features_batch(
            substrates_list,
            dataset_name=dataset_name,
            use_cache=True
        )
        
        for idx in range(len(X_df)):
            if idx >= len(df_valid_subset):
                break
            # Include both extracted features and original text for LLM
            feat_dict = X_df.iloc[idx].to_dict()
            feat_dict['SEQ'] = str(df_valid_subset.iloc[idx]['SEQ'])
            feat_dict['SUBSTRATES'] = str(df_valid_subset.iloc[idx]['SUBSTRATES'])
            # Add RDKit features computed from SUBSTRATES (SMILES) - ONLY for LLM
            if idx < len(rdkit_features_list):
                feat_dict.update(rdkit_features_list[idx])
            X_original.append(feat_dict)
        
        n_rdkit_features = len(rdkit_features_list[0]) if rdkit_features_list else 0
        logger.info(f"Added RDKit features to X_original only ({n_rdkit_features} RDKit descriptors per sample) - NOT in ML training data")
        
        feature_encoders: Dict[str, Any] = {}
        dataset_label = f"Enzyme Dataset: {dataset_name}"
        
        logger.info(f"Final dataset: {len(X_df)} samples, {len(feature_cols)} features")
        logger.info(f"Target range: [{y_values.min():.4f}, {y_values.max():.4f}]")
        
        return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
        
    except Exception as e:
        logger.error(f"Failed to load enzyme dataset '{dataset_name}': {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Enzyme dataset '{dataset_name}' not available: {e}")


def load_tabarena_regression_dataset(dataset_name: str, max_samples: int):
    """Load TabArena regression dataset from HuggingFace or sklearn
    
    Args:
        dataset_name: Name of the TabArena dataset (e.g., 'diabetes')
        max_samples: Maximum number of samples to load
        
    Returns:
        X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    
    try:
        from sklearn.preprocessing import LabelEncoder
        
        dataset_name_lower = dataset_name.lower()
        
        # Special case: diabetes dataset from sklearn (not available on HuggingFace)
        if dataset_name_lower == "diabetes":
            from sklearn.datasets import load_diabetes
            logger.info(f"Loading diabetes dataset from sklearn")
            data = load_diabetes(as_frame=True)
            X_df = data.data.astype(float)
            y_series = data.target.astype(float)
            feature_cols = list(X_df.columns)
            
            # Subsample if needed
            if max_samples and len(X_df) > max_samples:
                rng = np.random.RandomState(RANDOM_STATE)
                indices = rng.choice(len(X_df), max_samples, replace=False)
                X_df = X_df.iloc[indices]
                y_series = y_series.iloc[indices]
                logger.info(f"Subsampled to {len(X_df)} samples")
            
            # Create X_original (feature dicts for LLM)
            X_original = []
            for i in range(len(X_df)):
                d = {col: float(X_df.iloc[i][col]) for col in X_df.columns}
                X_original.append(d)
            
            # Convert to numpy
            X_encoded = X_df.values.astype(float)
            y_values = y_series.values.astype(float)
            
            # No categorical features in diabetes dataset, so empty encoders
            feature_encoders: Dict[str, Any] = {}
            
            logger.info(f"Final dataset: {len(X_encoded)} samples, {len(feature_cols)} features")
            logger.info(f"Target range: [{y_values.min():.4f}, {y_values.max():.4f}]")
            
            dataset_label = "sklearn Diabetes"
            
            return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
        
        # Special case: California housing from sklearn (more reliable than HuggingFace)
        if dataset_name_lower == "housing":
            try:
                from sklearn.datasets import fetch_california_housing
                logger.info(f"Loading California housing dataset from sklearn")
                data = fetch_california_housing(as_frame=True)
                X_df = data.data.astype(float)
                y_series = data.target.astype(float)
                feature_cols = list(X_df.columns)
                
                # Subsample if needed
                if max_samples and len(X_df) > max_samples:
                    rng = np.random.RandomState(RANDOM_STATE)
                    indices = rng.choice(len(X_df), max_samples, replace=False)
                    X_df = X_df.iloc[indices]
                    y_series = y_series.iloc[indices]
                    logger.info(f"Subsampled to {len(X_df)} samples")
                
                # Create X_original (feature dicts for LLM)
                X_original = []
                for i in range(len(X_df)):
                    d = {col: float(X_df.iloc[i][col]) for col in X_df.columns}
                    X_original.append(d)
                
                # Convert to numpy
                X_encoded = X_df.values.astype(float)
                y_values = y_series.values.astype(float)
                
                # No categorical features in california housing dataset, so empty encoders
                feature_encoders: Dict[str, Any] = {}
                
                logger.info(f"Final dataset: {len(X_encoded)} samples, {len(feature_cols)} features")
                logger.info(f"Target range: [{y_values.min():.4f}, {y_values.max():.4f}]")
                
                dataset_label = "sklearn California Housing"
                
                return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
            except Exception as e:
                logger.warning(f"Failed to load California housing from sklearn: {e}, trying HuggingFace...")
                # Fall through to HuggingFace loading
        
        # For other datasets, try loading from HuggingFace
        from datasets import load_dataset
        
        # TabArena regression dataset mapping (HuggingFace)
        # Note: Some datasets may not exist on HuggingFace. We'll try multiple paths or use sklearn alternatives
        TABARENA_REGRESSION_DATASETS = {
            "housing": ["scikit-learn/california_housing", "mstz/california-housing"],  # Try sklearn first
            "bike": ["mstz/bike-sharing"],
            "insurance": ["mstz/insurance"],
            "concrete": ["mstz/concrete"],
            "energy": ["mstz/energy-efficiency"],
            "airfoil": ["mstz/airfoil"],
            "yacht": ["mstz/yacht-hydrodynamics"],
            "auto": ["mstz/auto-mpg"],
            "abalone": ["mstz/abalone"],
            "winequality": ["mstz/wine-quality"],
            "students": ["mstz/student-performance"],
            "diamonds": ["mstz/diamonds"],
            "house-prices": ["mstz/house-prices"],
            "airbnb": ["mstz/airbnb-price"],
        }
        
        if dataset_name_lower not in TABARENA_REGRESSION_DATASETS:
            available = ["diabetes"] + list(TABARENA_REGRESSION_DATASETS.keys())
            raise ValueError(f"TabArena regression dataset '{dataset_name}' not found. Available: {available}")
        
        # Try loading from multiple possible paths
        dataset_paths = TABARENA_REGRESSION_DATASETS[dataset_name_lower]
        if not isinstance(dataset_paths, list):
            dataset_paths = [dataset_paths]
        
        dataset = None
        last_error = None
        for path in dataset_paths:
            try:
                logger.info(f"Trying to load TabArena regression dataset: {dataset_name} from {path}")
                dataset = load_dataset(path)
                logger.info(f"Successfully loaded from {path}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load from {path}: {e}")
                continue
        
        if dataset is None:
            available_paths = ", ".join(dataset_paths)
            raise RuntimeError(f"Failed to load TabArena regression dataset '{dataset_name}' from any of: {available_paths}. Last error: {last_error}")
        
        # Convert to pandas DataFrame
        if 'train' in dataset:
            df = pd.DataFrame(dataset['train'])
        else:
            df = pd.DataFrame(dataset[list(dataset.keys())[0]])
        
        logger.info(f"Loaded TabArena dataset: {len(df)} samples, {len(df.columns)} columns")
        
        # Auto-detect target column
        possible_targets = [
            'target', 'label', 'y', 'target_value', 'value',
            'price', 'cnt', 'charges', 'strength', 'heating_load',
            'cooling_load', 'SalePrice', 'MPG', 'rings', 'quality'
        ]
        target_column = None
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            target_column = df.columns[-1]
        
        logger.info(f"Using '{target_column}' as target column")
        
        # Separate features and target
        X_df = df.drop(columns=[target_column])
        y_series = df[target_column]
        
        # Encode categorical features
        feature_encoders: Dict[str, Any] = {}
        categorical_cols = []
        
        for col in X_df.columns:
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                categorical_cols.append(col)
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                feature_encoders[col] = le
        
        # Create X_original (feature dicts for LLM)
        X_original = []
        for i in range(len(X_df)):
            d = {}
            for col in X_df.columns:
                if col in categorical_cols:
                    d[col] = str(df.iloc[i][col])  # Original categorical value
                else:
                    d[col] = float(X_df.iloc[i][col])
            X_original.append(d)
        
        # Check if target is actually regression (continuous values)
        if y_series.dtype == 'object' or y_series.dtype.name == 'category':
            # Try to convert to numeric
            try:
                y_series = pd.to_numeric(y_series, errors='coerce')
            except Exception:
                raise ValueError(f"Dataset '{dataset_name}' appears to be classification (categorical target). "
                               f"Use 017_maicl_classification_residual_topk.py instead.")
        
        # Check unique values to ensure it's regression
        unique_vals = len(np.unique(y_series.dropna()))
        if unique_vals <= 20:
            logger.warning(f"Dataset '{dataset_name}' has only {unique_vals} unique target values. "
                         f"This might be classification. Continuing as regression...")
        
        feature_cols = X_df.columns.tolist()
        
        # Convert to numpy
        X_encoded = X_df.values.astype(float)
        y_values = y_series.values.astype(float)
        
        # Handle missing values
        valid_mask = ~(np.isnan(X_encoded).any(axis=1) | np.isnan(y_values))
        X_encoded = X_encoded[valid_mask]
        y_values = y_values[valid_mask]
        X_original = [X_original[i] for i in range(len(X_original)) if valid_mask[i]]
        
        # Subsample if needed
        if max_samples and len(X_encoded) > max_samples:
            indices = np.random.choice(len(X_encoded), max_samples, replace=False)
            X_encoded = X_encoded[indices]
            y_values = y_values[indices]
            X_original = [X_original[i] for i in indices]
            logger.info(f"Subsampled to {len(X_encoded)} samples")
        
        logger.info(f"Final dataset: {len(X_encoded)} samples, {len(feature_cols)} features")
        logger.info(f"Target range: [{y_values.min():.4f}, {y_values.max():.4f}]")
        
        dataset_label = f"TabArena {dataset_name}"
        
        return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
        
    except ImportError as e:
        raise RuntimeError(f"datasets library not installed. pip install datasets\n"
                         f"Import error: {e}")
    except Exception as e:
        logger.error(f"Failed to load TabArena regression dataset '{dataset_name}': {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"TabArena dataset '{dataset_name}' not available: {e}")


def load_deepchem_regression_dataset(dataset_name: str, max_samples: int):
    """Load DeepChem regression dataset (supports any DeepChem molnet dataset)
    
    Args:
        dataset_name: Name of the dataset. Supports:
            - Common names: 'esol', 'delaney', 'lipo', 'lipophilicity'
            - Any DeepChem molnet loader: will try 'load_{dataset_name}' function
        max_samples: Maximum number of samples to load
        
    Returns:
        X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
        
    Note:
        - ML model receives featurized vectors (ECFP fingerprints)
        - LLM receives SMILES strings in X_original (extracted from dataset.ids)
        - Works with any DeepChem dataset that has SMILES in dataset.ids
    """
    # Ensure deepchem is available
    # Get dc from module globals if available, otherwise import it
    import sys
    current_module = sys.modules[__name__]
    if _HAS_DEEPCHEM and hasattr(current_module, 'dc'):
        # Use module-level dc import
        deepchem_module = current_module.dc
    else:
        # Import deepchem if not available
        try:
            import deepchem
            deepchem_module = deepchem
            _ = deepchem_module.molnet
        except ImportError as e:
            raise RuntimeError(f"DeepChem not installed. pip install deepchem\n"
                             f"Import error: {e}")
        except Exception as e:
            error_msg = f"DeepChem import failed. This may require additional dependencies.\n"
            error_msg += f"Try: pip install deepchem tensorflow\n"
            error_msg += f"Import error: {e}"
            raise RuntimeError(error_msg)
    
    try:
        # Generic approach: try to find loader function dynamically
        # First check common mappings, then try direct function name lookup
        dataset_name_lower = dataset_name.lower()
        
        # Common dataset name mappings
        dataset_mappings = {
            'esol': 'load_delaney',
            'delaney': 'load_delaney',
            'lipo': 'load_lipo',
            'lipophilicity': 'load_lipo',
        }
        
        # Determine loader function name
        if dataset_name_lower in dataset_mappings:
            loader_name = dataset_mappings[dataset_name_lower]
        else:
            # Try direct mapping: dataset name -> load_{dataset_name}
            loader_name = f"load_{dataset_name_lower}"
        
        # Check if loader exists in molnet
        if not hasattr(deepchem_module.molnet, loader_name):
            # List available loaders for better error message
            available_loaders = [attr for attr in dir(deepchem_module.molnet) 
                               if attr.startswith('load_') and callable(getattr(deepchem_module.molnet, attr))]
            raise AttributeError(f"DeepChem loader '{loader_name}' not found. "
                               f"Available loaders: {available_loaders[:10]}...")
        
        loader_func = getattr(deepchem_module.molnet, loader_name)
        logger.info(f"Loading DeepChem regression dataset: {dataset_name_lower} (using {loader_name})")
        
        # Load dataset with ECFP featurization (2048-bit fingerprints)
        tasks, datasets, transformers = loader_func(featurizer='ECFP', splitter='random')
        train_dataset, valid_dataset, test_dataset = datasets
        
        # Combine all splits for now (we'll split later in main())
        all_X = np.vstack([train_dataset.X, valid_dataset.X, test_dataset.X])
        all_y = np.vstack([train_dataset.y, valid_dataset.y, test_dataset.y])
        all_ids = list(train_dataset.ids) + list(valid_dataset.ids) + list(test_dataset.ids)
        
        logger.info(f"Loaded DeepChem {dataset_name_lower}: {len(all_X)} total samples")
        logger.info(f"  Features shape: {all_X.shape} (ECFP fingerprints)")
        logger.info(f"  Target shape: {all_y.shape}")
        logger.info(f"  Tasks: {tasks}")
        
        # Flatten y to 1D for regression
        if all_y.ndim > 1 and all_y.shape[1] == 1:
            all_y = all_y.flatten()
        
        # Remove samples with NaN targets
        valid_mask = ~np.isnan(all_y)
        if isinstance(valid_mask, np.ndarray) and valid_mask.ndim > 0:
            all_X = all_X[valid_mask]
            all_y = all_y[valid_mask]
            all_ids = [all_ids[i] for i in range(len(all_ids)) if valid_mask[i]]
        
        logger.info(f"After removing NaN targets: {len(all_X)} samples")
        
        # Subsample if needed
        if max_samples and len(all_X) > max_samples:
            indices = np.random.choice(len(all_X), max_samples, replace=False)
            all_X = all_X[indices]
            all_y = all_y[indices]
            all_ids = [all_ids[i] for i in indices]
            logger.info(f"Subsampled to {len(all_X)} samples")
        
        # X_encoded: featurized vectors for ML model (ECFP fingerprints)
        X_encoded = all_X.astype(float)
        
        # Create feature column names (ECFP fingerprints are typically 2048 bits)
        n_features = X_encoded.shape[1]
        feature_cols = [f"ecfp_bit_{i}" for i in range(n_features)]
        
        # y_values: regression targets (continuous values)
        y_values = all_y.astype(float)
        
        logger.info(f"Target range: [{y_values.min():.4f}, {y_values.max():.4f}]")
        
        # X_original: SMILES strings + RDKit features for LLM
        # DeepChem stores SMILES in dataset.ids
        logger.info("Computing RDKit features for DeepChem dataset...")
        rdkit_features_list = _compute_rdkit_features_batch(
            [str(smiles) for smiles in all_ids],
            dataset_name=dataset_name_lower,
            use_cache=True
        )
        
        X_original = []
        for i, smiles in enumerate(all_ids):
            # SMILES string is the primary input for LLM
            # Include it as a feature dict that LLM can understand
            feat_dict = {'SMILES': str(smiles)}
            # Add RDKit features
            if i < len(rdkit_features_list):
                feat_dict.update(rdkit_features_list[i])
            X_original.append(feat_dict)
        
        logger.info(f"Added RDKit features to X_original ({len(rdkit_features_list[0])} RDKit descriptors per sample)")
        
        feature_encoders: Dict[str, Any] = {}
        
        # Create dataset label (use friendly names for known datasets, otherwise use dataset name)
        dataset_labels = {
            'esol': 'ESOL (Water Solubility)',
            'delaney': 'Delaney (ESOL)',
            'lipo': 'Lipophilicity (LogP)',
            'lipophilicity': 'Lipophilicity (LogP)',
        }
        # Capitalize first letter and use dataset name if not in mapping
        if dataset_name_lower in dataset_labels:
            dataset_label = dataset_labels[dataset_name_lower]
        else:
            # Use capitalized dataset name
            dataset_label = f"DeepChem {dataset_name_lower.capitalize()}"
        
        logger.info(f"Final dataset: {len(X_encoded)} samples, {len(feature_cols)} features (ECFP)")
        logger.info(f"  ML model will use: {X_encoded.shape} featurized vectors")
        logger.info(f"  LLM will use: SMILES strings from dataset")
        
        return X_encoded, y_values, X_original, feature_cols, feature_encoders, dataset_label
        
    except Exception as e:
        logger.error(f"Failed to load DeepChem regression dataset '{dataset_name}': {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DeepChem dataset '{dataset_name}' not available: {e}")


def _select_topk_residual_balanced(X_tr_s, y_tr_s, residual_vec, k, bins=10):
    """Select top-K residuals with balanced y-quantile coverage (helper for CV)."""
    if k <= 0 or k >= len(y_tr_s):
        return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
    abs_res = np.abs(residual_vec)
    q = np.quantile(y_tr_s, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    yb = np.digitize(y_tr_s, q, right=True)
    idxs = []
    unique_bins, counts = np.unique(yb, return_counts=True)
    proportions = {b: c / len(y_tr_s) for b, c in zip(unique_bins, counts)}
    allocated = {b: max(1, int(round(k * proportions[b]))) for b in unique_bins}
    total_alloc = sum(allocated.values())
    while total_alloc > k:
        bmax = max(allocated, key=lambda b: allocated[b])
        if allocated[bmax] > 1:
            allocated[bmax] -= 1
            total_alloc -= 1
        else:
            break
    while total_alloc < k:
        bmin = min(allocated, key=lambda b: allocated[b])
        allocated[bmin] += 1
        total_alloc += 1
    for b in unique_bins:
        bin_idx = np.where(yb == b)[0]
        if len(bin_idx) == 0:
            continue
        take = min(allocated[b], len(bin_idx))
        top_in_bin = bin_idx[np.argsort(abs_res[bin_idx])[::-1][:take]]
        idxs.extend(top_in_bin.tolist())
    idxs = np.array(sorted(set(idxs)))
    return X_tr_s[idxs], y_tr_s[idxs], idxs

def _select_topk_balanced_y_quantile(X_tr_s, y_tr_s, k, bins=10):
    """Select top-K samples with balanced y-quantile coverage (helper for CV)."""
    if k <= 0 or k >= len(y_tr_s):
        return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
    q = np.quantile(y_tr_s, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    yb = np.digitize(y_tr_s, q, right=True)
    idxs = []
    unique_bins, counts = np.unique(yb, return_counts=True)
    proportions = {b: c / len(y_tr_s) for b, c in zip(unique_bins, counts)}
    allocated = {b: max(1, int(round(k * proportions[b]))) for b in unique_bins}
    total_alloc = sum(allocated.values())
    while total_alloc > k:
        bmax = max(allocated, key=lambda b: allocated[b])
        if allocated[bmax] > 1:
            allocated[bmax] -= 1
            total_alloc -= 1
        else:
            break
    while total_alloc < k:
        bmin = min(allocated, key=lambda b: allocated[b])
        allocated[bmin] += 1
        total_alloc += 1
    rng = np.random.RandomState(RANDOM_STATE)
    for b in unique_bins:
        bin_idx = np.where(yb == b)[0]
        if len(bin_idx) == 0:
            continue
        take = min(allocated[b], len(bin_idx))
        selected = rng.choice(bin_idx, size=take, replace=False)
        idxs.extend(selected.tolist())
    idxs = np.array(sorted(set(idxs)))
    return X_tr_s[idxs], y_tr_s[idxs], idxs


def _select_topk_random(X_tr_s, y_tr_s, residual_vec, k):
    """Select top-K samples randomly from residuals (no sorting)."""
    if k <= 0 or k >= len(y_tr_s):
        return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
    rng = np.random.RandomState(RANDOM_STATE)
    idxs = rng.choice(len(y_tr_s), size=k, replace=False)
    idxs = np.array(sorted(idxs))
    return X_tr_s[idxs], y_tr_s[idxs], idxs


def suggest_optimal_gfp_combinations(
    maicl, X_train_original, feature_cols, scaler, y_scaler_target,
    n_suggestions=20, n_candidates=1000, output_dir=None
):
    """
    Suggest optimal feature combinations for GFP yield maximization.
    
    Uses the trained MAICL model to predict yields for candidate feature combinations
    and returns top suggestions in original (unscaled) scale.
    
    Args:
        maicl: Trained TrainableMAICL model
        X_train_original: Original training data (list of dicts) for reference ranges
        feature_cols: List of feature column names
        scaler: Feature scaler (MinMaxScaler010) used during training
        y_scaler_target: Target scaler (MinMaxScaler010) used during training
        n_suggestions: Number of top suggestions to return (default: 20)
        n_candidates: Number of candidate combinations to evaluate (default: 1000)
        output_dir: Output directory to save suggestions (optional)
    
    Returns:
        List of dicts with feature combinations and predicted yields in original scale
    """
    logger.info("=" * 80)
    logger.info("GENERATING OPTIMAL GFP YIELD COMBINATIONS")
    logger.info("=" * 80)
    
    try:
        # Extract feature ranges from original training data
        if not X_train_original or len(X_train_original) == 0:
            logger.warning("No original training data available for range extraction")
            return []
        
        # Build feature ranges from training data
        feature_ranges = {}
        for col in feature_cols:
            values = [float(x[col]) for x in X_train_original if col in x and x[col] is not None]
            if values:
                feature_ranges[col] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
            else:
                logger.warning(f"Feature {col} not found in original data, skipping")
        
        if len(feature_ranges) != len(feature_cols):
            logger.warning(f"Only found ranges for {len(feature_ranges)}/{len(feature_cols)} features")
        
        logger.info(f"Generating {n_candidates} candidate combinations...")
        logger.info(f"Feature ranges extracted from {len(X_train_original)} training samples")
        
        # Generate candidate combinations
        # Strategy: Sample from feature ranges, with some bias toward high-yield regions
        candidates = []
        rng = np.random.RandomState(RANDOM_STATE)
        
        for i in range(n_candidates):
            candidate = {}
            for col in feature_cols:
                if col in feature_ranges:
                    feat_range = feature_ranges[col]
                    # Mix of random sampling and biased sampling (toward higher values for some features)
                    if i < n_candidates // 2:
                        # Random uniform sampling within observed range
                        val = rng.uniform(feat_range['min'], feat_range['max'])
                    else:
                        # Biased sampling: mix of mean, high values, and random
                        strategy = rng.choice(['mean', 'high', 'random'])
                        if strategy == 'mean':
                            val = feat_range['mean'] + rng.normal(0, feat_range['std'] * 0.5)
                            val = np.clip(val, feat_range['min'], feat_range['max'])
                        elif strategy == 'high':
                            # Sample from upper half of range
                            val = rng.uniform(
                                feat_range['mean'], 
                                feat_range['max']
                            )
                        else:
                            val = rng.uniform(feat_range['min'], feat_range['max'])
                    candidate[col] = float(val)
                else:
                    # Fallback: use mean if range not available
                    candidate[col] = 0.0
            
            candidates.append(candidate)
        
        logger.info(f"Generated {len(candidates)} candidate combinations")
        
        # Convert candidates to scaled format for prediction
        logger.info("Converting candidates to scaled format and predicting yields...")
        X_candidates_scaled = []
        X_candidates_original = []
        
        for candidate in candidates:
            # Convert to array in feature_cols order
            x_array = np.array([candidate[col] for col in feature_cols], dtype=float)
            X_candidates_original.append(candidate.copy())
            
            # Scale features
            if scaler is not None:
                x_scaled = scaler.transform(x_array.reshape(1, -1))[0]
            else:
                x_scaled = x_array
            X_candidates_scaled.append(x_scaled)
        
        X_candidates_scaled = np.array(X_candidates_scaled)
        
        # Check for duplicates with training data (optional but recommended)
        # Build a set of training feature vectors for quick comparison
        training_vectors = set()
        if X_train_original:
            for x_train in X_train_original:
                vec = tuple([round(float(x_train.get(col, 0)), 6) for col in feature_cols])
                training_vectors.add(vec)
            logger.info(f"  Checking {len(candidates)} candidates against {len(training_vectors)} training samples for duplicates...")
            duplicates_removed = 0
            candidates_filtered = []
            X_candidates_original_filtered = []
            for i, candidate in enumerate(candidates):
                vec = tuple([round(float(candidate.get(col, 0)), 6) for col in feature_cols])
                if vec not in training_vectors:
                    candidates_filtered.append(candidate)
                    X_candidates_original_filtered.append(X_candidates_original[i])
                else:
                    duplicates_removed += 1
            if duplicates_removed > 0:
                logger.info(f"  Removed {duplicates_removed} duplicate candidates (matching training samples)")
                candidates = candidates_filtered
                X_candidates_original = X_candidates_original_filtered
                # Rebuild scaled array
                X_candidates_scaled = []
                for candidate in candidates:
                    x_array = np.array([candidate[col] for col in feature_cols], dtype=float)
                    if scaler is not None:
                        x_scaled = scaler.transform(x_array.reshape(1, -1))[0]
                    else:
                        x_scaled = x_array
                    X_candidates_scaled.append(x_scaled)
                X_candidates_scaled = np.array(X_candidates_scaled)
                logger.info(f"  Evaluating {len(candidates)} unique NEW candidate combinations")
        
        # Prepare pool data ONCE (use random sample or full set if small)
        # Use full training set if small enough, otherwise random sample
        pool_size = min(200, len(X_train_original)) if X_train_original else 0
        X_pool_scaled = None
        X_pool_original = None
        
        if X_train_original and len(X_train_original) > 0:
            if len(X_train_original) <= pool_size:
                # Use full training set
                pool_original = X_train_original
                logger.info(f"  Using full training set ({len(pool_original)} samples) as pool for few-shot examples")
            else:
                # Random sample for diversity
                pool_indices = rng.choice(len(X_train_original), size=pool_size, replace=False)
                pool_original = [X_train_original[i] for i in pool_indices]
                logger.info(f"  Using random sample of {len(pool_original)} training samples as pool for few-shot examples")
            
            X_pool_original = pool_original
            if scaler is not None:
                pool_arrays = [np.array([x[col] for col in feature_cols], dtype=float) for x in pool_original]
                X_pool_scaled = np.array([scaler.transform(arr.reshape(1, -1))[0] for arr in pool_arrays])
            else:
                pool_arrays = [np.array([x[col] for col in feature_cols], dtype=float) for x in pool_original]
                X_pool_scaled = np.array(pool_arrays)
            y_pool_dummy = np.zeros(len(X_pool_scaled))
        else:
            logger.warning("  No training data available for pool - predictions may be less accurate")
            X_pool_scaled = X_candidates_scaled[:min(10, len(X_candidates_scaled))] if len(X_candidates_scaled) > 0 else None
            y_pool_dummy = np.zeros(len(X_pool_scaled)) if X_pool_scaled is not None else np.array([])
        
        # Predict yields using trained MAICL
        logger.info("Predicting yields using trained MAICL model...")
        predictions_scaled = []
        
        # Predict in batches to handle large candidate sets
        batch_size = 100
        n_batches = (len(X_candidates_scaled) + batch_size - 1) // batch_size
        logger.info(f"  Processing {len(X_candidates_scaled)} candidates in {n_batches} batch(es) of size {batch_size}")
        
        for i in range(0, len(X_candidates_scaled), batch_size):
            batch = X_candidates_scaled[i:i+batch_size]
            batch_original = X_candidates_original[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            # Use MAICL predict method
            try:
                # Create dummy y for evaluation (not used for prediction)
                y_dummy = np.zeros(len(batch))
                
                # Use evaluate method to get predictions
                # Note: We need to pass X_original for LLM mechanisms
                batch_original_list = batch_original
                
                eval_result = maicl.evaluate(
                    batch, y_dummy, X_pool_scaled, y_pool_dummy,
                    return_details=True, relax_routing=True,
                    X_original=batch_original_list,
                    X_pool_original=X_pool_original
                )
                
                batch_preds = np.array(eval_result.get('predictions', []))
                if len(batch_preds) != len(batch):
                    logger.warning(f"  Batch {batch_num}/{n_batches}: Expected {len(batch)} predictions, got {len(batch_preds)}")
                    # Try to get predictions from the result in a different way
                    if 'y_pred' in eval_result:
                        batch_preds = np.array(eval_result['y_pred'])
                    elif hasattr(eval_result, 'predictions'):
                        batch_preds = np.array(eval_result.predictions)
                    else:
                        logger.error(f"  Batch {batch_num}/{n_batches}: Could not extract predictions from eval_result")
                        raise ValueError(f"Could not extract predictions for batch {batch_num}")
                
                # Log prediction range for first batch to debug
                if i == 0 and len(batch_preds) > 0:
                    logger.info(f"  First batch prediction range (scaled): [{batch_preds.min():.6f}, {batch_preds.max():.6f}], mean={batch_preds.mean():.6f}, std={batch_preds.std():.6f}")
                
                predictions_scaled.extend(batch_preds.tolist())
            except Exception as e:
                logger.error(f"  Batch {batch_num}/{n_batches} prediction failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Don't use fallback - raise error to ensure we catch issues
                raise RuntimeError(f"Failed to predict batch {batch_num}: {e}") from e
        
        predictions_scaled = np.array(predictions_scaled)
        
        # Log prediction statistics before unscaling
        if len(predictions_scaled) > 0:
            unique_preds = len(np.unique(predictions_scaled))
            logger.info(f"  Predictions (scaled): min={predictions_scaled.min():.6f}, max={predictions_scaled.max():.6f}, "
                       f"mean={predictions_scaled.mean():.6f}, std={predictions_scaled.std():.6f}, unique={unique_preds}")
            if unique_preds < 10:
                logger.warning(f"  ⚠️  Only {unique_preds} unique predictions out of {len(predictions_scaled)} candidates - model may be predicting constant values")
        
        # Unscale predictions to original scale
        if y_scaler_target is not None:
            # Inverse transform predictions
            preds_reshaped = predictions_scaled.reshape(-1, 1)
            if hasattr(y_scaler_target, 'inverse_transform'):
                predictions_original = y_scaler_target.inverse_transform(preds_reshaped).ravel()
            else:
                # Manual inverse transform for MinMaxScaler010
                feature_range = getattr(y_scaler_target, 'feature_range', (0.0, 1.0))
                scale = y_scaler_target.scale_
                min_orig = y_scaler_target.min_
                scale_safe = np.where(scale != 0, scale, 1.0)
                predictions_original = ((preds_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
        else:
            predictions_original = predictions_scaled
        
        # Log prediction statistics after unscaling
        if len(predictions_original) > 0:
            unique_preds_orig = len(np.unique(predictions_original))
            logger.info(f"  Predictions (original scale): min={predictions_original.min():.4f}, max={predictions_original.max():.4f}, "
                       f"mean={predictions_original.mean():.4f}, std={predictions_original.std():.4f}, unique={unique_preds_orig}")
        
        # Sort by predicted yield (descending) and get top suggestions
        sorted_indices = np.argsort(predictions_original)[::-1]
        top_indices = sorted_indices[:n_suggestions]
        
        suggestions = []
        for idx in top_indices:
            suggestion = {
                'rank': len(suggestions) + 1,
                'predicted_yield': float(predictions_original[idx]),
                'features': X_candidates_original[idx].copy()
            }
            suggestions.append(suggestion)
        
        logger.info(f"Top {len(suggestions)} suggestions generated")
        pred_range = predictions_original[top_indices]
        logger.info(f"Predicted yield range: {pred_range.min():.4f} - {pred_range.max():.4f}")
        logger.info(f"  NOTE: These are NEW combinations generated by sampling feature ranges, NOT from original dataset")
        logger.info(f"  NOTE: Predictions are made by the trained MAICL model (ensemble of ML + LLM mechanisms)")
        
        # Save suggestions to file
        if output_dir:
            suggestions_file = os.path.join(output_dir, "optimal_gfp_combinations.json")
            with open(suggestions_file, 'w') as f:
                json.dump({
                    'n_suggestions': len(suggestions),
                    'n_candidates_evaluated': n_candidates,
                    'note': 'These are NEW feature combinations generated by MAICL, not from original dataset',
                    'suggestions': suggestions,
                    'feature_ranges_used': feature_ranges
                }, f, indent=2)
            logger.info(f"Saved suggestions to: {suggestions_file}")
            
            # Also save as CSV for easy viewing
            try:
                import pandas as pd
                csv_data = []
                for sug in suggestions:
                    row = {'rank': sug['rank'], 'predicted_yield': sug['predicted_yield']}
                    row.update(sug['features'])
                    csv_data.append(row)
                df_suggestions = pd.DataFrame(csv_data)
                csv_file = os.path.join(output_dir, "optimal_gfp_combinations.csv")
                df_suggestions.to_csv(csv_file, index=False)
                logger.info(f"Saved suggestions CSV to: {csv_file}")
            except Exception as e:
                logger.warning(f"Could not save CSV: {e}")
        
        # Print top 5 suggestions
        logger.info("\n" + "=" * 80)
        logger.info("TOP 5 SUGGESTED GFP YIELD COMBINATIONS (Generated by MAICL)")
        logger.info("=" * 80)
        logger.info("NOTE: These are NEW combinations predicted by the trained MAICL model, not from original dataset")
        for i, sug in enumerate(suggestions[:5], 1):
            logger.info(f"\nRank {i}: Predicted Yield = {sug['predicted_yield']:.4f}")
            logger.info("  Features:")
            for feat, val in sug['features'].items():
                logger.info(f"    {feat}: {val:.4f}")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to generate optimal combinations: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_fold_pipeline_regression(
    X_train, X_val, X_test, y_train, y_val, y_test,
    X_original_train, X_original_val, X_original_test,
    feature_cols, feature_encoders,
    args, llm, ds_name, ds_label, fold_output_dir, fold_num
):
    """
    Run the complete training and evaluation pipeline for a single fold (regression).
    Returns a dictionary with all metrics and results.
    """
    import json
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    logger.info(f"Running fold {fold_num} pipeline...")
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Set output directory for this fold
    original_output_dir = OUTPUT_DIR
    set_output_dir(fold_output_dir)
    
    try:
        # Scale features and targets
        if args.no_scaling:
            scaler = None
            X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
            y_scaler_target = None
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test
        else:
            scaler = MinMaxScaler010()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            y_scaler_target = MinMaxScaler010()
            y_train_s = y_scaler_target.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = y_scaler_target.transform(y_val.reshape(-1, 1)).ravel()
            y_test_s = y_scaler_target.transform(y_test.reshape(-1, 1)).ravel()
        
        # Train ML model
        pretrained_ml = None
        residuals = None
        sorted_idx = None
        ml_preds = None
        
        mech_map = {"linear": "LinearRegression", "xgboost": "XGBoost", "kernelridge": "KernelRidge", "tabicl": "TabICL", "ebm": "EBM", "tabpfn": "TabPFN"}
        model_name = mech_map.get(args.ml_mech.lower(), "KernelRidge")
        
        if args.use_ml:
            pretrained_ml = MLModelMechanism(model_name, task_type="regression")
            if model_name == "TabICL":
                pretrained_ml._tabicl_regression_bins = args.tabicl_bins
            pretrained_ml.train(X_train_s, y_train_s, feature_cols, y_scaler=None)
            pretrained_ml._maicl_feature_cols = feature_cols
            pretrained_ml._maicl_feature_encoders = feature_encoders
            pretrained_ml._maicl_scaler = scaler
            
            sorted_idx, residuals, ml_preds, _ = compute_ml_residuals(
                pretrained_ml, X_train_s, y_train_s, feature_cols, class_names=None, task_type="regression"
            )
        
        # Select top-K samples
        if args.top_k == -1:
            X_topk, y_topk, top_indices = X_train_s, y_train_s, np.arange(len(X_train_s))
            X_original_topk = X_original_train
        else:
            if args.use_ml:
                if args.topk_strategy == "residual_balanced":
                    X_topk, y_topk, top_indices = _select_topk_residual_balanced(
                        X_train_s, y_train_s, residuals, args.top_k, bins=10
                    )
                elif args.topk_strategy == "random":
                    X_topk, y_topk, top_indices = _select_topk_random(
                        X_train_s, y_train_s, residuals, args.top_k
                    )
                else:
                    X_topk, y_topk, top_indices = get_top_k_residual_samples(
                        sorted_idx, residuals, X_train_s, y_train_s, args.top_k,
                        ml_predictions=ml_preds, task_type="regression", selection_strategy="residual"
                    )
            else:
                X_topk, y_topk, top_indices = _select_topk_balanced_y_quantile(
                    X_train_s, y_train_s, args.top_k, bins=10
                )
            
            if len(top_indices) == 0:
                X_topk, y_topk, top_indices = X_train_s, y_train_s, np.arange(len(X_train_s))
                X_original_topk = X_original_train
            else:
                X_original_topk = [X_original_train[i] for i in top_indices]
        
        # Compute residuals on top-K subset
        if args.top_k != -1 and args.use_ml:
            sorted_idx, residuals_topk, ml_preds, _ = compute_ml_residuals(
                pretrained_ml, X_topk, y_topk, feature_cols, class_names=None, task_type="regression"
            )
        else:
            residuals_topk = residuals if residuals is not None else None
        
        # Compute ML baseline on test set
        ml_baseline_metrics = {}
        if args.use_ml and pretrained_ml:
            try:
                # Get predictions (need to handle different model types)
                if model_name == "TabPFN":
                    import subprocess, tempfile, sys, json, pandas as pd
                    from pathlib import Path
                    temp_dir = Path(tempfile.mkdtemp(prefix="tabpfn_ml_baseline_test_"))
                    X_test_df = pd.DataFrame(X_test_s, columns=feature_cols)
                    X_test_path = temp_dir / "X_test.csv"
                    output_path = temp_dir / "pred_result_test.json"
                    X_test_df.to_csv(X_test_path, index=False)
                    script_path = Path(__file__).parent / "run_tabpfn_ml_mechanism.py"
                    result = subprocess.run(
                        [sys.executable, str(script_path), "predict",
                         pretrained_ml._tabpfn_training_data_path['X_train_path'],
                         pretrained_ml._tabpfn_training_data_path['y_train_path'],
                         str(X_test_path), pretrained_ml.task_type, str(output_path)],
                        capture_output=True, text=True, timeout=600
                    )
                    if result.returncode == 0 and output_path.exists():
                        with open(output_path, 'r') as f:
                            pred_result = json.load(f)
                        if pred_result.get('success', False):
                            y_pred = np.array(pred_result['predictions'], dtype=float)
                        else:
                            raise RuntimeError(f"TabPFN failed: {pred_result.get('error', 'Unknown error')}")
                    else:
                        raise RuntimeError("TabPFN subprocess failed")
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                elif model_name == "TabICL":
                    import pandas as pd
                    X_test_df = pd.DataFrame(X_test_s, columns=feature_cols)
                    y_pred = pretrained_ml.model.predict(X_test_df)
                else:
                    y_pred = pretrained_ml.model.predict(X_test_s)
                
                # Unscale predictions if needed
                if y_scaler_target:
                    # MinMaxScaler010 doesn't have inverse_transform, so we implement it manually
                    pred_reshaped = y_pred.reshape(-1, 1)
                    if hasattr(y_scaler_target, 'inverse_transform'):
                        y_pred_unscaled = y_scaler_target.inverse_transform(pred_reshaped).ravel()
                    else:
                        # Manual inverse transform for MinMaxScaler010
                        # Formula: scaled = scale_ * (original - min_) + feature_range[0]
                        # Inverse: original = (scaled - feature_range[0]) / scale_ + min_
                        feature_range = getattr(y_scaler_target, 'feature_range', (0.0, 1.0))
                        scale = y_scaler_target.scale_
                        min_orig = y_scaler_target.min_
                        # Handle division by zero (when scale is 0, feature is constant)
                        scale_safe = np.where(scale != 0, scale, 1.0)
                        y_pred_unscaled = ((pred_reshaped - feature_range[0]) / scale_safe + min_orig).ravel()
                    y_test_unscaled = y_test  # y_test is already unscaled
                else:
                    y_pred_unscaled = y_pred
                    y_test_unscaled = y_test
                
                r2 = r2_score(y_test_unscaled, y_pred_unscaled)
                mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
                mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
                ml_baseline_metrics = {
                    "r2": float(r2), "mae": float(mae), "mse": float(mse),
                    "predictions": y_pred_unscaled.tolist()
                }
            except Exception as e:
                logger.warning(f"Failed to compute ML baseline metrics: {e}")
        
        # Create and train MA-ICL
        maicl = TrainableMAICL(
            llm, feature_cols, scaler,
            use_ml_mechanism=bool(args.use_ml),
            dataset_name=ds_name,
            y_scaler=y_scaler_target,
            pretrained_ml_mechanism=pretrained_ml,
            data_insights=None,
            task_type="regression",
            class_names=None,
            regression_loss_metric=args.regression_loss,
            num_mechanisms_unknown=args.num_mechanisms_unknown,
            use_scaling=not args.no_scaling
        )
        if ml_baseline_metrics:
            try:
                maicl.set_ml_baseline_performance(ml_baseline_metrics)
            except Exception:
                pass
        
        # Pre-training evaluation
        pre = maicl.evaluate(
            X_test_s, y_test_s, X_train_s, y_train_s,
            return_details=True, relax_routing=bool(args.relax_eval),
            X_original=X_original_test, X_pool_original=X_original_train,
        )
        pre_r2 = float(pre.get('r2', 0.0))
        pre_mae = float(pre.get('mae', 0.0))
        pre_loss = float(pre.get('loss', float('inf')))
        
        # Train MA-ICL
        maicl.train(
            X_topk, y_topk, X_val_s, y_val_s,
            iterations=args.iterations,
            ml_residuals=residuals_topk,
            accept_eval_max=None,
            X_test=X_test_s, y_test=y_test_s,
            acceptance_set=args.acceptance_set,
            X_train_original=X_original_topk,
            X_val_original=X_original_val,
            X_test_original=X_original_test,
            output_dir=fold_output_dir,
        )
        
        # Post-training evaluation
        post = maicl.evaluate(
            X_test_s, y_test_s, X_train_s, y_train_s,
            return_details=True, relax_routing=True,
            preserve_mechanism_performance=True,
            X_original=X_original_test, X_pool_original=X_original_train,
        )
        post_r2 = float(post.get('r2', 0.0))
        post_mae = float(post.get('mae', 0.0))
        post_loss = float(post.get('loss', float('inf')))
        
        # LLM-only evaluation
        llm_only_metrics = None
        try:
            llm_only_metrics = maicl.evaluate_llm_only(
                X_test_s, y_test_s, X_train_s, y_train_s,
                return_details=True,
                X_original=X_original_test, X_pool_original=X_original_train,
            )
        except Exception as e:
            logger.warning(f"Failed to evaluate LLM-only: {e}")
        
        # Save fold results
        fold_result = {
            'fold_num': fold_num,
            'n_train': len(X_train_s),
            'n_val': len(X_val_s),
            'n_test': len(X_test_s),
            'ml_baseline_r2': ml_baseline_metrics.get('r2', 0.0),
            'ml_baseline_mae': ml_baseline_metrics.get('mae', 0.0),
            'pre_r2': pre_r2,
            'pre_mae': pre_mae,
            'pre_loss': pre_loss,
            'post_r2': post_r2,
            'post_mae': post_mae,
            'post_loss': post_loss,
        }
        
        if llm_only_metrics:
            fold_result['llm_only_pre_r2'] = float(llm_only_metrics.get('r2', 0.0))
            fold_result['llm_only_pre_mae'] = float(llm_only_metrics.get('mae', 0.0))
            fold_result['llm_only_post_r2'] = float(llm_only_metrics.get('r2', 0.0))
            fold_result['llm_only_post_mae'] = float(llm_only_metrics.get('mae', 0.0))
        
        # Save fold-specific results file
        fold_results_file = os.path.join(fold_output_dir, "fold_results.json")
        with open(fold_results_file, 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        logger.info(f"Fold {fold_num} complete: R2={post_r2:.4f}, MAE={post_mae:.4f}")
        
        return fold_result
        
    finally:
        # Restore original output directory
        set_output_dir(original_output_dir)


def save_cv_summary_regression(cv_results, output_dir, args):
    """
    Aggregate cross-validation results and save summary statistics (regression).
    """
    import json
    import numpy as np
    
    # Aggregate metrics across folds
    metrics_to_aggregate = [
        'ml_baseline_r2', 'ml_baseline_mae',
        'pre_r2', 'pre_mae', 'pre_loss',
        'post_r2', 'post_mae', 'post_loss',
        'llm_only_pre_r2', 'llm_only_pre_mae',
        'llm_only_post_r2', 'llm_only_post_mae'
    ]
    
    summary = {
        'n_folds': len(cv_results),
        'cv_folds': args.cv_folds,
        'fold_results': cv_results,
        'summary_statistics': {}
    }
    
    # Compute mean and std for each metric
    for metric in metrics_to_aggregate:
        values = [r.get(metric, None) for r in cv_results if r.get(metric) is not None]
        if values:
            summary['summary_statistics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
    
    # Save summary
    summary_file = os.path.join(output_dir, "cv_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Number of folds: {len(cv_results)}")
    logger.info("\nSummary Statistics:")
    for metric, stats in summary['summary_statistics'].items():
        logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min={stats['min']:.4f}, max={stats['max']:.4f})")
    logger.info(f"\nFull summary saved to: {summary_file}")
    logger.info(f"Individual fold results saved in: {output_dir}/fold_*/")


def main():
    # Explicitly reference global json module to prevent UnboundLocalError
    # (Python may treat json as local if it sees json.load() calls later in the function)
    global json
    
    parser = argparse.ArgumentParser(description="MA-ICL regression on biotech/experimental datasets with top-K residuals")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name. Options:\n"
                             "  Built-in: gfp_yield, protein_expression, protein_expression_all, dataset_102\n"
                             "  InaData (SU/EC):\n"
                             "    - pfas_su (PFAS developmental assay; pupation/eclosure)\n"
                             "    - ec_fertility (fertility/eclosion counts)\n"
                             "    - ec_climbing (climbing % over time)\n"
                             "  TabArena (HuggingFace/sklearn regression):\n"
                             "    - diabetes (sklearn)\n"
                             "    - housing (sklearn, with HuggingFace fallback)\n"
                             "    - bike, insurance, concrete, energy, airfoil, yacht\n"
                             "    - auto, abalone, winequality, students, diamonds, house-prices, airbnb\n"
                             "    Note: Some datasets may not be available on HuggingFace. The code will try multiple paths.\n"
                             "  DeepChem (molecular regression):\n"
                             "    - esol or delaney (water solubility prediction)\n"
                             "    - lipo or lipophilicity (LogP prediction)\n"
                             "  Enzyme Regression (exact names from enzyme-datasets table):\n"
                             "    - aminotransferase\n"
                             "    - olea\n"
                             "    - halogenase_NaBr\n"
                             "    - halogenase_NaCl\n"
                             "    - phosphatase_achiral\n"
                             "    - phosphatase_chiral\n"
                             "    - davis\n"
                             "    - davis_filtered")
    parser.add_argument("--su_target", type=str, default="female_ratio_final",
                        help="Target for --dataset pfas_su. Options include: pupation_total, pupation_t50, "
                             "eclosure_total, eclosure_t50, female_ratio_final (default: eclosure_total).")
    parser.add_argument("--ec_target", type=str, default=None,
                        help="Target for --dataset ec_fertility or ec_climbing. "
                             "Defaults: ec_fertility -> total_eclosed, ec_climbing -> avg_16s. "
                             "Examples: --ec_target total_pupae (fertility) or --ec_target female_16s (climbing).")
    parser.add_argument("--plate_file", type=str, default=None,
                        help="Plate file name for protein_expression dataset (e.g., 'plate_AL_1_raw_yield_and_std.csv')")
    parser.add_argument("--plate_index", type=int, default=5,
                        help="Plate index (0-based) for protein_expression dataset. Use --list_plates to see available indices.")
    parser.add_argument("--list_plates", action="store_true",
                        help="List all available protein expression plate files and exit")
    parser.add_argument("--list_enzyme_datasets", action="store_true",
                        help="List all available enzyme datasets and exit")
    parser.add_argument("--model_name", default=os.environ.get("MAICL_MODEL_NAME", "gemini-2.0-flash"),
                        help="Gemini model name, e.g., gemini-2.0-flash, gemini-2.5-pro")
    parser.add_argument("--ml_mech", default="linear", help="Regression ML mechanism: linear|xgboost|kernelridge|tabicl|ebm|tabpfn. "
                        "Note: TabPFN may cause segmentation faults when used as ML mechanism. "
                        "Consider using TabPFN only as a baseline (it runs in isolated subprocess) "
                        "or use --ml_mech linear/xgboost for more stable ML mechanism.")
    parser.add_argument("--baseline_models", type=str, nargs='+', default=None,
                        help="Select which baseline models to compute. Options: tabpfn, ebm, shap, llmlex. "
                             "If not specified, all available baselines will be computed. "
                             "Example: --baseline_models tabpfn ebm llmlex")
    parser.add_argument("--tabicl_bins", type=int, default=20,
                        help="Number of bins for TabICL regression quantization (default: 20). "
                             "Only used when --ml_mech=tabicl. Higher values = finer granularity but more classes.")
    parser.add_argument("--use_ml", type=int, default=1, choices=[0,1], help="Include ML mechanism in ensemble")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=1000, help="-1 to use full dataset")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--acceptance_set", type=str, default="validation",
                        choices=["test", "validation", "train"],
                        help="Dataset to use for acceptance evaluation during training. "
                             "Options: 'test' (risks overfitting to test), 'validation' (default), "
                             "or 'train' (may overfit to training data).")
    parser.add_argument("--topk_strategy", choices=["residual", "residual_balanced", "random"], default="residual",
                        help="Top-K selection: 'residual' = global highest | 'residual_balanced' = highest within y-quantile bins | 'random' = random selection (no sorting)")
    parser.add_argument("--relax_eval", type=int, default=0, choices=[0,1],
                        help="Relax ML routing during evaluation to let LLM contribute (default: 1, recommended for regression)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Validation fraction of the training split (0-1). Default: 0.2")
    parser.add_argument("--regression_loss", type=str, default="mae", choices=["mae", "r2"],
                        help="Regression loss metric: 'mae' (mean absolute error) or 'r2' (R-squared). Default: mae")
    parser.add_argument("--evaluate_individual_mechanisms", action="store_true",
                        help="After training, evaluate each mechanism individually on train/test sets")
    parser.add_argument("--num_mechanisms_unknown", type=int, default=1,
                        help="Number of unknown mechanisms to generate (default: 1). "
                             "This controls how many LLM-based mechanisms are created to complement the ML mechanism.")
    parser.add_argument("--k_shot", type=int, default=0,
                        help="Number of few-shot examples to use per prediction (default: 0). "
                             "If > 0, retrieves k_shot examples from training set for each prediction.")
    parser.add_argument("--no_scaling", action="store_true",
                        help="Disable all scaling for both features and targets. Use raw/unscaled data throughout.")
    parser.add_argument("--cv_folds", type=int, default=0,
                        help="Number of cross-validation folds (default: 0 = disabled). "
                             "When > 0, performs k-fold cross-validation and saves results for each fold. "
                             "Example: --cv_folds 5 for 5-fold CV.")
    args = parser.parse_args()

    # Handle --list_plates option
    if args.list_plates:
        plates = get_available_protein_plates()
        if plates:
            print("\nAvailable Protein Expression Plates:")
            print("=" * 60)
            for idx, plate in enumerate(plates):
                print(f"  [{idx}] {plate}")
            print("=" * 60)
            print(f"\nTotal: {len(plates)} plates available")
            print("\nUsage examples:")
            print(f"  # Select by index:")
            print(f"  python {os.path.basename(__file__)} --dataset protein_expression --plate_index 0")
            print(f"  # Select by filename:")
            print(f"  python {os.path.basename(__file__)} --dataset protein_expression --plate_file {plates[0]}")
        else:
            print("No protein expression plates found.")
        return
    
    # Handle --list_enzyme_datasets option
    if args.list_enzyme_datasets:
        try:
            from enzyme_dataset_analysis import EnzymeDatasetLoader
            data_dir = os.path.join(os.path.dirname(__file__), "enzyme-datasets", "data")
            if not os.path.exists(data_dir):
                data_dir = "enzyme-datasets/data"
            
            if not os.path.exists(data_dir) and not os.path.exists("enzyme-datasets"):
                print("\n❌ Enzyme dataset directory not found!")
                print(f"   Expected location: {os.path.join(os.path.dirname(__file__), 'enzyme-datasets')}")
                print(f"   Or: enzyme-datasets/\n")
                print("   To fix this, clone the enzyme-datasets repository:")
                print("   git clone https://github.com/samgoldman97/enzyme-datasets.git")
                print("   Or download it to the project directory.\n")
                return
            
            loader = EnzymeDatasetLoader(data_dir)
            datasets_list = loader.list_available_datasets()
            
            # Also check processed directory
            processed_dir = os.path.join(data_dir, "processed")
            csv_datasets = []
            if os.path.exists(processed_dir):
                import glob
                csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
                csv_datasets = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
            
            print("\nAvailable Enzyme Datasets:")
            print("=" * 60)
            
            if datasets_list:
                print("\nFrom dataset directories:")
                for ds in datasets_list:
                    ds_name = ds.name if hasattr(ds, 'name') else str(ds)
                    print(f"  - {ds_name}")
            
            if csv_datasets:
                print("\nFrom processed CSV files:")
                for ds in sorted(set(csv_datasets)):
                    print(f"  - {ds}")
            
            if not datasets_list and not csv_datasets:
                print("  No datasets found.")
                print(f"\n  Checked: {data_dir}")
                if os.path.exists(processed_dir):
                    print(f"  Checked: {processed_dir}")
            
            print("=" * 60)
            total = len(datasets_list) + len(set(csv_datasets))
            print(f"\nTotal: {total} dataset(s) available")
            print("\nUsage example:")
            if datasets_list or csv_datasets:
                example_ds = datasets_list[0].name if datasets_list else csv_datasets[0]
                print(f"  python {os.path.basename(__file__)} --dataset {example_ds} --top_k 300")
        except Exception as e:
            print(f"\nError listing enzyme datasets: {e}")
            import traceback
            traceback.print_exc()
        return

    # IMPORTANT: allow LLM mechanisms to contribute during regression (don't force-route only to ML)
    os.environ.setdefault("MAICL_REGRESSION_PREFER_ML", "0")

    # Resolve dataset - preserve original case for enzyme datasets, lowercase for built-in datasets
    dataset_name_lower = args.dataset.lower()
    dataset_name_original = args.dataset  # Preserve original case for enzyme datasets
    
    # Extract plate information for protein_expression dataset to include in output directory
    plate_suffix = ""
    if dataset_name_lower == "protein_expression":
        if args.plate_index is not None:
            plate_suffix = f"_plate{args.plate_index}"
        elif args.plate_file is not None:
            # Extract plate name from filename (e.g., "plate_AL_1" from "plate_AL_1_raw_yield_and_std.csv")
            plate_name = os.path.basename(args.plate_file).replace('_raw_yield_and_std.csv', '').replace('.csv', '')
            plate_suffix = f"_{plate_name}"
    
    # Build comprehensive run name with all important arguments
    # Start with base components
    run_name_parts = [
        "bio_reg",
        dataset_name_lower + plate_suffix,
        f"ml{args.ml_mech}" if args.use_ml else "noML",
        f"topk{args.top_k}",
        f"iter{args.iterations}",
        f"model{args.model_name.replace('-', '_').replace('.', '_')}",  # Sanitize model name for filesystem
        f"loss{args.regression_loss}",
        f"accept{args.acceptance_set}",
        f"topkstrat{args.topk_strategy}",
    ]
    
    # Add dataset-specific target arguments
    if dataset_name_lower == "pfas_su":
        # Add su_target for PFAS SU dataset
        su_target_sanitized = str(args.su_target).strip().replace('-', '_').replace('.', '_')
        run_name_parts.append(f"sutarget{su_target_sanitized}")
    elif dataset_name_lower == "ec_fertility":
        # Add ec_target for EC fertility dataset (use default if not specified)
        ec_target = args.ec_target.strip() if isinstance(args.ec_target, str) and args.ec_target.strip() else "total_eclosed"
        ec_target_sanitized = ec_target.replace('-', '_').replace('.', '_')
        run_name_parts.append(f"ectarget{ec_target_sanitized}")
    elif dataset_name_lower == "ec_climbing":
        # Add ec_target for EC climbing dataset (use default if not specified)
        ec_target = args.ec_target.strip() if isinstance(args.ec_target, str) and args.ec_target.strip() else "avg_16s"
        ec_target_sanitized = ec_target.replace('-', '_').replace('.', '_')
        run_name_parts.append(f"ectarget{ec_target_sanitized}")
    
    # Add optional flags
    if args.relax_eval:
        run_name_parts.append("relax")
    if args.num_mechanisms_unknown > 1:
        run_name_parts.append(f"nmech{args.num_mechanisms_unknown}")
    if args.k_shot > 0:
        run_name_parts.append(f"kshot{args.k_shot}")
    if args.no_scaling:
        run_name_parts.append("noscale")
    if args.val_size != 0.2:  # Only include if non-default
        run_name_parts.append(f"val{args.val_size:.2f}".replace('.', '_'))
    if args.ml_mech == "tabicl":  # Include tabicl_bins if using tabicl
        run_name_parts.append(f"bins{args.tabicl_bins}")
    if args.cv_folds > 0:
        run_name_parts.append(f"cv{args.cv_folds}")
    
    # Join all parts with underscores
    run_name = "_".join(run_name_parts)
    output_dir = set_output_dir(run_name)
    logger.info(f"Output directory: {output_dir}")
    
    # Set up file logging to save terminal output
    log_file = os.path.join(output_dir, "run.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    # Add file handler to root logger (will capture all logging)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    # Save experiment configuration for reproducibility
    experiment_config = {
        "run_name": run_name,
        "output_dir": output_dir,
        "dataset": {
            "name": args.dataset,
            "name_lower": dataset_name_lower,
            "name_original": dataset_name_original,
            "plate_file": args.plate_file,
            "plate_index": args.plate_index,
            "max_samples": args.max_samples
        },
        "model": {
            "llm_model_name": args.model_name,
            "ml_mechanism": args.ml_mech,
            "use_ml": bool(args.use_ml),
            "tabicl_bins": args.tabicl_bins if args.ml_mech.lower() == "tabicl" else None
        },
        "training": {
            "top_k": args.top_k,
            "topk_strategy": args.topk_strategy,
            "iterations": args.iterations,
            "acceptance_set": args.acceptance_set,
            "val_size": args.val_size,
            "regression_loss": args.regression_loss,
            "relax_eval": bool(args.relax_eval),
            "num_mechanisms_unknown": args.num_mechanisms_unknown,
            "no_scaling": args.no_scaling
        },
        "evaluation": {
            "evaluate_individual_mechanisms": args.evaluate_individual_mechanisms
        },
        "system": {
            "random_state": RANDOM_STATE,
            "scale_min": SCALE_MIN,
            "scale_max": SCALE_MAX,
            "has_matplotlib": _HAS_MATPLOTLIB,
            "has_pandas": _HAS_PANDAS,
            "has_deepchem": _HAS_DEEPCHEM
        },
        "command_line_args": vars(args)
    }
    
    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    logger.info(f"Saved experiment configuration to {config_file}")

    llm = _get_gemini_llm(args.model_name)

    if dataset_name_lower == "gfp_yield":
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_gfp_yield_dataset(args.max_samples)
    elif dataset_name_lower == "protein_expression":
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_protein_expression_dataset(
            args.max_samples, plate_file=args.plate_file, plate_index=args.plate_index)
    elif dataset_name_lower == "protein_expression_all":
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_protein_expression_all_plates_dataset(args.max_samples)
    elif dataset_name_lower == "dataset_102":
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_dataset_102(args.max_samples)
    elif dataset_name_lower == "pfas_su":
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_inadata_pfas_su_dataset(
            args.max_samples, target=str(args.su_target).strip()
        )
    elif dataset_name_lower == "ec_fertility":
        ec_target = args.ec_target.strip() if isinstance(args.ec_target, str) and args.ec_target.strip() else "total_eclosed"
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_inadata_ec_fertility_dataset(
            args.max_samples, target=ec_target
        )
    elif dataset_name_lower == "ec_climbing":
        ec_target = args.ec_target.strip() if isinstance(args.ec_target, str) and args.ec_target.strip() else "avg_16s"
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_inadata_ec_climbing_dataset(
            args.max_samples, target=ec_target
        )
    elif dataset_name_lower in ("esol", "delaney", "lipo", "lipophilicity"):
        # DeepChem molecular regression datasets
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_deepchem_regression_dataset(
            args.dataset, args.max_samples
        )
    elif dataset_name_lower in ("diabetes", "housing", "bike", "insurance", "concrete", "energy", 
                                  "airfoil", "yacht", "auto", "abalone", "winequality", "students", 
                                  "diamonds", "house-prices", "airbnb") or dataset_name_lower.startswith("tabarena_"):
        # TabArena regression datasets from HuggingFace
        try:
            # Remove tabarena_ prefix if present
            tabarena_name = dataset_name_lower.replace("tabarena_", "")
            X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_tabarena_regression_dataset(
                tabarena_name, args.max_samples
            )
        except Exception as e:
            raise SystemExit(f"Failed to load TabArena regression dataset '{args.dataset}': {e}")
    else:
        # Try loading as enzyme dataset - use original case-preserved name
        try:
            X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_enzyme_dataset(dataset_name_original, args.max_samples)
        except Exception as e:
            raise SystemExit(f"Unknown dataset {args.dataset}. Error: {e}\n"
                           f"Available options: gfp_yield, protein_expression, protein_expression_all, dataset_102, "
                           f"TabArena (diabetes), "
                           f"DeepChem (any molnet dataset, e.g., esol, delaney, lipo, lipophilicity), "
                           f"or enzyme dataset names (e.g., halogenase_NaBr, aminotransferase, olea, phosphatase_achiral)")
    
    # Update experiment config with dataset information after loading
    experiment_config["dataset"]["label"] = ds_label
    experiment_config["dataset"]["n_samples"] = len(X_all)
    experiment_config["dataset"]["n_features"] = len(feature_cols) if feature_cols else X_all.shape[1] if hasattr(X_all, 'shape') else None
    experiment_config["dataset"]["feature_cols"] = feature_cols if feature_cols else None
    experiment_config["dataset"]["y_min"] = float(y_all.min()) if hasattr(y_all, 'min') else None
    experiment_config["dataset"]["y_max"] = float(y_all.max()) if hasattr(y_all, 'max') else None
    experiment_config["dataset"]["y_mean"] = float(y_all.mean()) if hasattr(y_all, 'mean') else None
    experiment_config["dataset"]["y_std"] = float(y_all.std()) if hasattr(y_all, 'std') else None
    
    # Update config file with dataset information
    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    logger.info(f"Updated experiment configuration with dataset information")
    
    from sklearn.model_selection import train_test_split, KFold
    from scipy.stats import binned_statistic
    
    # Check if cross-validation is enabled
    cv_folds = args.cv_folds if args.cv_folds > 0 else 0
    
    if cv_folds > 0:
        logger.info("=" * 80)
        logger.info(f"CROSS-VALIDATION MODE: {cv_folds}-fold CV")
        logger.info("=" * 80)
        
        # For regression, use KFold (can't use StratifiedKFold for continuous targets)
        # Optionally use quantile-based stratification
        try:
            # Try quantile-based stratification for regression
            n_bins = min(10, cv_folds)
            quantiles = np.quantile(y_all, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
            y_bins = np.digitize(y_all, quantiles, right=True)
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(kf.split(X_all, y_bins))
            logger.info("Using quantile-based stratification for CV folds")
        except Exception:
            # Fallback to regular KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(kf.split(X_all))
            logger.info("Using regular KFold for CV (quantile stratification unavailable)")
        
        # Store results for all folds
        cv_results = []
        
        # Run each fold
        for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
            logger.info("=" * 80)
            logger.info(f"FOLD {fold_idx + 1}/{cv_folds}")
            logger.info("=" * 80)
            
            # Create fold-specific output directory
            fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_output_dir, exist_ok=True)
            
            # Split data for this fold
            X_train_val = X_all[train_val_idx]
            y_train_val = y_all[train_val_idx]
            X_original_train_val = [X_original_all[i] for i in train_val_idx]
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]
            X_original_test = [X_original_all[i] for i in test_idx]
            
            # Further split train_val into train and validation
            idx_train_val = np.arange(len(X_train_val))
            try:
                # Try quantile-based stratification for train/val split
                n_bins = 10
                quantiles_tr = np.quantile(y_train_val, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
                y_bins_tr = np.digitize(y_train_val, quantiles_tr, right=True)
                X_train, X_val, y_train, y_val, idx_tr, idx_va = train_test_split(
                    X_train_val, y_train_val, idx_train_val, test_size=float(args.val_size),
                    random_state=RANDOM_STATE, stratify=y_bins_tr
                )
            except Exception:
                # Fallback to random split
                X_train, X_val, y_train, y_val, idx_tr, idx_va = train_test_split(
                    X_train_val, y_train_val, idx_train_val, test_size=float(args.val_size),
                    random_state=RANDOM_STATE
                )
            
            X_original_train = [X_original_train_val[i] for i in idx_tr]
            X_original_val = [X_original_train_val[i] for i in idx_va]
            
            # Run the pipeline for this fold
            fold_result = run_fold_pipeline_regression(
                X_train, X_val, X_test, y_train, y_val, y_test,
                X_original_train, X_original_val, X_original_test,
                feature_cols, feature_encoders,
                args, llm, dataset_name_lower, ds_label, fold_output_dir, fold_idx + 1
            )
            cv_results.append(fold_result)
        
        # Aggregate and save CV summary
        save_cv_summary_regression(cv_results, output_dir, args)
        logger.info("=" * 80)
        logger.info("CROSS-VALIDATION COMPLETE")
        logger.info("=" * 80)
        return
    
    # Standard single run (no CV)
    # Regression-friendly stratification: bin y into quantiles and stratify to preserve target distribution
    split_method = "random"  # Track which split method was used
    try:
        n_bins = 10
        quantiles = np.quantile(y_all, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        y_bins = np.digitize(y_all, quantiles, right=True)
        X_train_full, X_test, y_train_full, y_test, idx_tr_full, idx_te = train_test_split(
            X_all, y_all, np.arange(len(X_all)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_bins
        )
        # Now split train_full into train and validation using stratification on y_train_full
        try:
            quantiles_tr = np.quantile(y_train_full, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
            y_bins_tr = np.digitize(y_train_full, quantiles_tr, right=True)
            X_train, X_val, y_train, y_val, idx_tr, idx_va = train_test_split(
                X_train_full, y_train_full, np.arange(len(X_train_full)), test_size=float(args.val_size), random_state=RANDOM_STATE, stratify=y_bins_tr
            )
            # Split X_original_all using same indices
            X_original_train = [X_original_all[idx_tr_full[i]] for i in idx_tr]
            X_original_val = [X_original_all[idx_tr_full[i]] for i in idx_va]
            X_original_test = [X_original_all[i] for i in idx_te]
            split_method = "quantile_stratified"
            logger.info("Used quantile-stratified train/validation/test split for regression.")
        except Exception:
            X_train, X_val, y_train, y_val, idx_tr, idx_va = train_test_split(
                X_train_full, y_train_full, np.arange(len(X_train_full)), test_size=float(args.val_size), random_state=RANDOM_STATE
            )
            # Split X_original_all using same indices
            X_original_train = [X_original_all[idx_tr_full[i]] for i in idx_tr]
            X_original_val = [X_original_all[idx_tr_full[i]] for i in idx_va]
            X_original_test = [X_original_all[i] for i in idx_te]
            split_method = "partial_stratified"  # First split stratified, second not
            logger.info("Used random train/validation split (stratification unavailable).")
    except Exception:
        # Fallback to regular split if stratification fails (e.g., tiny datasets)
        X_train_full, X_test, y_train_full, y_test, idx_tr_full, idx_te = train_test_split(
            X_all, y_all, np.arange(len(X_all)), test_size=0.2, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val, idx_tr, idx_va = train_test_split(
            X_train_full, y_train_full, np.arange(len(X_train_full)), test_size=float(args.val_size), random_state=RANDOM_STATE
        )
        # Split X_original_all using same indices
        X_original_train = [X_original_all[idx_tr_full[i]] for i in idx_tr]
        X_original_val = [X_original_all[idx_tr_full[i]] for i in idx_va]
        X_original_test = [X_original_all[i] for i in idx_te]
        split_method = "random"
        logger.info("Used regular random train/validation/test split (stratification unavailable).")

    # Update experiment config with split information
    experiment_config["data_splits"] = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_train_full": len(X_train_full),
        "test_size": 0.2,
        "val_size": float(args.val_size),
        "random_state": RANDOM_STATE,
        "split_method": split_method
    }
    # Update config file with split information
    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    logger.info(f"Updated experiment configuration with data split information")

    # Log routing mode for clarity
    if bool(args.relax_eval):
        logger.info("Evaluation routing: RELAXED (LLM mechanisms can contribute).")
    else:
        logger.info("Evaluation routing: STRICT ML preference (LLM contributions limited).")

    # Detect if this is a DeepChem dataset (for special handling of SMILES strings)
    is_deepchem_dataset = dataset_name_lower in ("esol", "delaney", "lipo", "lipophilicity") or (
        feature_cols and len(feature_cols) > 0 and 
        all(feat.startswith('ecfp_bit_') for feat in feature_cols[:10])
    )
    
    # For DeepChem datasets, create index mapping table for SMILES strings
    # This ensures we can map from residual indices to SMILES strings correctly
    if is_deepchem_dataset:
        logger.info("Detected DeepChem dataset - will use SMILES strings (non-vectorized features) for LLM mechanisms")
        # Verify that X_original contains SMILES strings
        if X_original_all and len(X_original_all) > 0:
            first_original = X_original_all[0]
            if isinstance(first_original, dict) and 'SMILES' in first_original:
                logger.info(f"  ✓ SMILES strings available in X_original (e.g., '{first_original['SMILES'][:50]}...')")
            else:
                logger.warning(f"  ⚠ X_original does not contain SMILES strings (type: {type(first_original)})")
        else:
            logger.warning("  ⚠ X_original is empty or None for DeepChem dataset")

    # Conditionally scale features based on --no_scaling flag
    if args.no_scaling:
        logger.info("Scaling DISABLED - using raw/unscaled data for features and targets")
        scaler = None
        X_train_s = X_train
        X_val_s = X_val
        X_test_s = X_test
        logger.info(f"Features NOT scaled (X min={X_train_s.min():.3f}, max={X_train_s.max():.3f})")
        
        # No scaling for targets either
        y_scaler_target = None
        y_train_s = y_train
        y_val_s = y_val
        y_test_s = y_test
        logger.info(f"Targets NOT scaled (y min={y_train_s.min():.3f}, max={y_train_s.max():.3f})")
        logger.info(f"NOTE: All models (ML and LLM) use raw/unscaled data")
    else:
        # Scale features to [SCALE_MIN, SCALE_MAX] using MinMaxScaler010 (default: [0.0, 1.0])
        scaler = MinMaxScaler010()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        validate_scaled_data(X_train_s, feature_cols, scaler=scaler)
        logger.info(f"Features scaled to [{SCALE_MIN}, {SCALE_MAX}] range (X min={X_train_s.min():.3f}, max={X_train_s.max():.3f})")

        # Scale target to [SCALE_MIN, SCALE_MAX] to match feature scaling for consistency
        # Use MinMaxScaler010 to ensure same scaling range as features (default: [0.0, 1.0])
        y_scaler_target = MinMaxScaler010()  # Uses SCALE_MIN and SCALE_MAX from config (default: [0.0, 1.0])
        y_train_s = y_scaler_target.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = y_scaler_target.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = y_scaler_target.transform(y_test.reshape(-1, 1)).ravel()
        logger.info(f"Targets scaled to [{SCALE_MIN}, {SCALE_MAX}] range (y min={y_train_s.min():.3f}, max={y_train_s.max():.3f})")
        logger.info(f"NOTE: All models (ML and LLM) use the same scaled data - features and targets both in [{SCALE_MIN}, {SCALE_MAX}]")

    # Only create and train ML model if use_ml is enabled
    pretrained_ml = None
    residuals = None
    sorted_idx = None
    ml_preds = None
    
    # Define model_name for logging purposes (even when ML is disabled)
    mech_map = {"linear": "LinearRegression", "xgboost": "XGBoost", "kernelridge": "KernelRidge", "tabicl": "TabICL", "ebm": "EBM", "tabpfn": "TabPFN"}
    model_name = mech_map.get(args.ml_mech.lower(), "KernelRidge")
    
    if args.use_ml:

        pretrained_ml = MLModelMechanism(model_name, task_type="regression")
        # Pass tabicl_bins parameter for TabICL regression quantization
        if model_name == "TabICL":
            pretrained_ml._tabicl_regression_bins = args.tabicl_bins
            logger.info(f"Using TabICL for regression with {args.tabicl_bins} quantization bins")
        pretrained_ml.train(X_train_s, y_train_s, feature_cols, y_scaler=None)
        pretrained_ml._maicl_feature_cols = feature_cols
        pretrained_ml._maicl_feature_encoders = feature_encoders
        pretrained_ml._maicl_scaler = scaler

        # Initial residuals computed on full training set (needed for top-K selection)
        sorted_idx, residuals, ml_preds, _ = compute_ml_residuals(
            pretrained_ml, X_train_s, y_train_s, feature_cols, class_names=None, task_type="regression"
        )
        logger.info("ML model trained and residuals computed for top-K selection")
    else:
        logger.info("ML mechanism disabled (--use_ml 0) - using LLM-only learning")

    def _select_topk_residual_balanced(X_tr_s, y_tr_s, residual_vec, k, bins=10):
        if k <= 0 or k >= len(y_tr_s):
            return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
        abs_res = np.abs(residual_vec)
        q = np.quantile(y_tr_s, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        yb = np.digitize(y_tr_s, q, right=True)
        idxs = []
        # Allocate k proportionally to bin sizes, minimum 1 if bin non-empty
        unique_bins, counts = np.unique(yb, return_counts=True)
        proportions = {b: c / len(y_tr_s) for b, c in zip(unique_bins, counts)}
        allocated = {b: max(1, int(round(k * proportions[b]))) for b in unique_bins}
        # Adjust allocation to exactly k
        total_alloc = sum(allocated.values())
        # Trim or add to match k
        while total_alloc > k:
            bmax = max(allocated, key=lambda b: allocated[b])
            if allocated[bmax] > 1:
                allocated[bmax] -= 1
                total_alloc -= 1
            else:
                break
        while total_alloc < k:
            bmin = min(allocated, key=lambda b: allocated[b])
            allocated[bmin] += 1
            total_alloc += 1
        # Pick within each bin
        for b in unique_bins:
            bin_idx = np.where(yb == b)[0]
            if len(bin_idx) == 0:
                continue
            take = min(allocated[b], len(bin_idx))
            top_in_bin = bin_idx[np.argsort(abs_res[bin_idx])[::-1][:take]]
            idxs.extend(top_in_bin.tolist())
        idxs = np.array(sorted(set(idxs)))
        return X_tr_s[idxs], y_tr_s[idxs], idxs
    
    def _select_topk_balanced_y_quantile(X_tr_s, y_tr_s, k, bins=10):
        """Select top-K samples with balanced y-quantile coverage (for LLM-only mode)"""
        if k <= 0 or k >= len(y_tr_s):
            return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
        q = np.quantile(y_tr_s, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        yb = np.digitize(y_tr_s, q, right=True)
        idxs = []
        # Allocate k proportionally to bin sizes, minimum 1 if bin non-empty
        unique_bins, counts = np.unique(yb, return_counts=True)
        proportions = {b: c / len(y_tr_s) for b, c in zip(unique_bins, counts)}
        allocated = {b: max(1, int(round(k * proportions[b]))) for b in unique_bins}
        # Adjust allocation to exactly k
        total_alloc = sum(allocated.values())
        # Trim or add to match k
        while total_alloc > k:
            bmax = max(allocated, key=lambda b: allocated[b])
            if allocated[bmax] > 1:
                allocated[bmax] -= 1
                total_alloc -= 1
            else:
                break
        while total_alloc < k:
            bmin = min(allocated, key=lambda b: allocated[b])
            allocated[bmin] += 1
            total_alloc += 1
        # Pick randomly within each bin (since we don't have residuals)
        rng = np.random.RandomState(RANDOM_STATE)
        for b in unique_bins:
            bin_idx = np.where(yb == b)[0]
            if len(bin_idx) == 0:
                continue
            take = min(allocated[b], len(bin_idx))
            selected = rng.choice(bin_idx, size=take, replace=False)
            idxs.extend(selected.tolist())
        idxs = np.array(sorted(set(idxs)))
        return X_tr_s[idxs], y_tr_s[idxs], idxs

    if args.top_k == -1:
        X_topk, y_topk, top_indices = X_train_s, y_train_s, np.arange(len(X_train_s))
        X_original_topk = X_original_train
    else:
        if args.use_ml:
            # Use ML residuals for top-K selection when ML is enabled
            if args.topk_strategy == "residual_balanced":
                logger.info(f"Selecting top-K residuals with balanced y-quantile coverage (K={args.top_k})")
                X_topk, y_topk, top_indices = _select_topk_residual_balanced(
                    X_train_s, y_train_s, residuals, args.top_k, bins=10
                )
            elif args.topk_strategy == "random":
                logger.info(f"Selecting top-K residuals randomly (no sorting, K={args.top_k})")
                X_topk, y_topk, top_indices = _select_topk_random(
                    X_train_s, y_train_s, residuals, args.top_k
                )
            else:
                X_topk, y_topk, top_indices = get_top_k_residual_samples(
                    sorted_idx, residuals, X_train_s, y_train_s, args.top_k,
                    ml_predictions=ml_preds, task_type="regression", selection_strategy="residual"
                )
            if len(top_indices) == 0:
                logger.warning("No top-K samples found (perfect ML?). Falling back to full dataset.")
                X_topk, y_topk, top_indices = X_train_s, y_train_s, np.arange(len(X_train_s))
                X_original_topk = X_original_train
            else:
                # Extract corresponding X_original samples (SMILES strings for LLM)
                # For DeepChem datasets, this ensures we use SMILES strings instead of vectorized features
                X_original_topk = [X_original_train[i] for i in top_indices]
                
                # For DeepChem datasets, log a sample of SMILES strings to verify correct mapping
                if is_deepchem_dataset and X_original_topk and len(X_original_topk) > 0:
                    sample_smiles = []
                    for idx in top_indices[:min(5, len(top_indices))]:
                        if idx < len(X_original_train):
                            orig = X_original_train[idx]
                            if isinstance(orig, dict) and 'SMILES' in orig:
                                sample_smiles.append(f"idx={idx}: SMILES={orig['SMILES']}")
                    if sample_smiles:
                        logger.info(f"  Sample SMILES strings from top-K residuals: {', '.join(sample_smiles[:3])}")
            logger.info(f"Top-K selection: strategy={args.topk_strategy}, selected={len(top_indices)} examples.")
        else:
            # LLM-only mode: use balanced y-quantile selection (no ML residuals available)
            logger.info(f"LLM-only mode: Selecting top-K samples with balanced y-quantile coverage (K={args.top_k})")
            X_topk, y_topk, top_indices = _select_topk_balanced_y_quantile(
                X_train_s, y_train_s, args.top_k, bins=10
            )
            X_original_topk = [X_original_train[i] for i in top_indices]
            logger.info(f"Top-K selection: balanced y-quantile, selected={len(top_indices)} examples.")
    
    # CRITICAL: All final evaluations (baselines, pre-training, post-training) use TEST set
    # acceptance_set is ONLY used during training iterations to accept/reject updates
    # It does NOT affect final evaluation - all models are always evaluated on TEST set
    X_eval = X_test_s
    y_eval = y_test_s
    X_original_eval = X_original_test if is_deepchem_dataset else None
    eval_set_name = "test"
    
    logger.info(f"Using TEST set for ALL final evaluations (baselines, pre-training, post-training)")
    logger.info(f"  Note: acceptance_set={args.acceptance_set} is only used during training iterations")
    logger.info(f"  All models are evaluated on TEST set for fair comparison")
    
    # ML model is FROZEN after initial training on full training set
    # Compute residuals on the top-K subset using the frozen model (for MA-ICL training)
    residuals_topk = None
    if args.use_ml:
        # ML model stays frozen - trained once on full training set and never retrained
        logger.info(f"ML model is FROZEN (trained on full training set: {len(X_train_s)} samples)")
        logger.info(f"MA-ICL will train on {len(X_topk)} top-K samples, but ML model remains frozen")
        
        if args.top_k != -1:
            # Compute residuals on the subset using the frozen model (trained on full set)
            sorted_idx, residuals_topk, ml_preds, _ = compute_ml_residuals(
                pretrained_ml, X_topk, y_topk, feature_cols, class_names=None, task_type="regression"
            )
            logger.info(f"Computed residuals on top-K subset using frozen ML model (trained on full set)")
        else:
            # Both train on full set, use original residuals
            residuals_topk = residuals
            logger.info(f"Using full training set ({len(X_topk)} samples) - residuals already computed")
        
        # Compute ML baseline metrics on TEST set using frozen model
        # All baselines use TEST set for final evaluation
        X_ml_eval = X_test_s
        y_ml_eval = y_test_s
        X_original_ml_eval = X_original_test if is_deepchem_dataset else None
        eval_set_name_ml = "test"
        
        # Compute ML baseline metrics on TEST set using frozen model
        ml_baseline_metrics: Dict[str, Any] = {}
        try:
            model = pretrained_ml.model
            # TabPFN uses subprocess isolation - model is None, use batch prediction helper
            if model_name == "TabPFN":
                # TabPFN requires batch prediction via subprocess
                import subprocess
                import tempfile
                from pathlib import Path
                import sys
                import pandas as pd
                
                # Create temporary files for prediction
                temp_dir = Path(tempfile.mkdtemp(prefix="tabpfn_ml_baseline_"))
                X_ml_eval_df = pd.DataFrame(X_ml_eval, columns=feature_cols)
                X_ml_eval_path = temp_dir / "X_eval.csv"
                output_path = temp_dir / "pred_result.json"
                
                X_ml_eval_df.to_csv(X_ml_eval_path, index=False)
                
                # Find subprocess script
                script_path = Path(__file__).parent / "run_tabpfn_ml_mechanism.py"
                if not script_path.exists():
                    raise RuntimeError(f"TabPFN subprocess script not found at {script_path}")
                
                # Get training data paths from pretrained_ml
                if not hasattr(pretrained_ml, '_tabpfn_training_data_path') or pretrained_ml._tabpfn_training_data_path is None:
                    raise RuntimeError("TabPFN not properly initialized with subprocess isolation")
                
                # Run prediction in subprocess
                result = subprocess.run(
                    [sys.executable, str(script_path), "predict",
                     pretrained_ml._tabpfn_training_data_path['X_train_path'],
                     pretrained_ml._tabpfn_training_data_path['y_train_path'],
                     str(X_ml_eval_path),
                     pretrained_ml.task_type,
                     str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                    raise RuntimeError(f"TabPFN batch prediction subprocess failed: {error_msg}")
                
                if not output_path.exists():
                    raise RuntimeError("TabPFN batch prediction subprocess completed but no output file found")
                
                # Load prediction result
                with open(output_path, 'r') as f:
                    pred_result = json.load(f)
                
                if not pred_result.get('success', False):
                    error_msg = pred_result.get('error', 'Unknown error')
                    raise RuntimeError(f"TabPFN batch prediction failed: {error_msg}")
                
                y_pred = np.array(pred_result['predictions'], dtype=float)
                
                # Clean up temporary files
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
            else:
                y_pred = model.predict(X_ml_eval).astype(float)
            
            # Clip predictions to [0,1] if targets are scaled (regression with scaling)
            if not args.no_scaling and y_scaler_target is not None:
                y_pred_clipped = np.clip(y_pred, SCALE_MIN, SCALE_MAX)
                n_out_of_range = np.sum((y_pred < SCALE_MIN) | (y_pred > SCALE_MAX))
                if n_out_of_range > 0:
                    logger.warning(f"⚠️  ML predictions out of range [{SCALE_MIN}, {SCALE_MAX}]: {n_out_of_range}/{len(y_pred)} samples")
                    logger.warning(f"   Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
                    logger.warning(f"   Target range: [{y_ml_eval.min():.4f}, {y_ml_eval.max():.4f}]")
                    logger.info(f"   Using clipped predictions for metrics (clipped {n_out_of_range} values)")
                    y_pred = y_pred_clipped
            
            r2 = r2_score(y_ml_eval, y_pred)
            mae = mean_absolute_error(y_ml_eval, y_pred)
            mse = mean_squared_error(y_ml_eval, y_pred)
            
            # Check for multicollinearity (can cause unstable coefficients)
            if len(feature_cols) > 1:
                from sklearn.preprocessing import StandardScaler
                # Compute correlation matrix of features
                feature_corr = np.corrcoef(X_train_s.T)
                high_corr_pairs = []
                for i in range(len(feature_cols)):
                    for j in range(i+1, len(feature_cols)):
                        if abs(feature_corr[i, j]) > 0.9:
                            high_corr_pairs.append((feature_cols[i], feature_cols[j], feature_corr[i, j]))
                if high_corr_pairs:
                    logger.warning(f"⚠️  High multicollinearity detected (|corr| > 0.9):")
                    for feat1, feat2, corr in high_corr_pairs:
                        logger.warning(f"   {feat1} ↔ {feat2}: {corr:.4f}")
                    logger.warning(f"   This can cause unstable/unreliable coefficients in linear models")
            
            # Additional diagnostics for poor performance
            if r2 < 0:
                logger.warning(f"⚠️  Negative R² ({r2:.4f}) indicates model performs worse than predicting the mean")
                logger.warning(f"   This suggests:")
                logger.warning(f"   1. Model may be overfitting or extrapolating poorly")
                logger.warning(f"   2. Non-linear relationships not captured by linear model")
                logger.warning(f"   3. High experimental noise relative to signal")
                logger.warning(f"   4. Small sample size ({len(X_train_s)} samples) may be insufficient")
                
                # Check if predictions are systematically biased
                mean_pred = y_pred.mean()
                mean_true = y_ml_eval.mean()
                pred_std = y_pred.std()
                true_std = y_ml_eval.std()
                logger.info(f"   Prediction mean: {mean_pred:.4f}, Target mean: {mean_true:.4f} (bias: {mean_pred - mean_true:.4f})")
                logger.info(f"   Prediction std: {pred_std:.4f}, Target std: {true_std:.4f}")
                
                # Check correlation
                correlation = np.corrcoef(y_ml_eval, y_pred)[0, 1]
                logger.info(f"   Correlation: {correlation:.4f}")
                
                # CRITICAL: Negative correlation means model predicts opposite direction
                if correlation < 0:
                    logger.error(f"   ❌ NEGATIVE CORRELATION ({correlation:.4f}) - Model predicts OPPOSITE direction!")
                    logger.error(f"   This indicates a fundamental problem with the model or data")
                
                # Check model coefficients to understand what the model learned
                if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                    logger.info(f"   Model coefficients:")
                    logger.info(f"     Intercept: {model.intercept_:.6f}")
                    for i, col in enumerate(feature_cols):
                        logger.info(f"     {col}: {model.coef_[i]:.6f}")
                    
                    # Check if coefficients are all very small (model barely using features)
                    max_coef = np.max(np.abs(model.coef_))
                    if max_coef < 0.1:
                        logger.warning(f"   ⚠️  All coefficients are very small (max={max_coef:.6f})")
                        logger.warning(f"   Model is barely using features - essentially predicting constant")
                    
                    # Check feature correlations with target
                    logger.info(f"   Feature-target correlations (on training set):")
                    for i, col in enumerate(feature_cols):
                        feat_corr = np.corrcoef(X_train_s[:, i], y_train_s)[0, 1]
                        logger.info(f"     {col}: {feat_corr:.4f} (coef: {model.coef_[i]:.6f})")
                        if np.sign(feat_corr) != np.sign(model.coef_[i]) and abs(feat_corr) > 0.1:
                            logger.warning(f"       ⚠️  Coefficient sign mismatch! Feature correlation={feat_corr:.4f}, coef={model.coef_[i]:.6f}")
                
                # Check if predictions are essentially constant
                pred_range = y_pred.max() - y_pred.min()
                true_range = y_ml_eval.max() - y_ml_eval.min()
                if pred_range < 0.1:
                    logger.error(f"   ❌ Predictions are nearly constant (range={pred_range:.4f} vs target range={true_range:.4f})")
                    logger.error(f"   Model is not learning meaningful patterns from features")
            
            ml_baseline_metrics = {
                "r2": r2, "mae": mae, "mse": mse,
                "predictions": y_pred.tolist()
            }
            # Also check training set performance for overfitting diagnosis
            if model_name == "TabPFN":
                # TabPFN batch prediction for training set
                import subprocess
                import tempfile
                from pathlib import Path
                import sys
                import pandas as pd
                
                temp_dir = Path(tempfile.mkdtemp(prefix="tabpfn_ml_baseline_train_"))
                X_train_df = pd.DataFrame(X_train_s, columns=feature_cols)
                X_train_path = temp_dir / "X_train_pred.csv"
                output_path = temp_dir / "pred_result_train.json"
                
                X_train_df.to_csv(X_train_path, index=False)
                
                script_path = Path(__file__).parent / "run_tabpfn_ml_mechanism.py"
                result = subprocess.run(
                    [sys.executable, str(script_path), "predict",
                     pretrained_ml._tabpfn_training_data_path['X_train_path'],
                     pretrained_ml._tabpfn_training_data_path['y_train_path'],
                     str(X_train_path),
                     pretrained_ml.task_type,
                     str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0 and output_path.exists():
                    with open(output_path, 'r') as f:
                        pred_result = json.load(f)
                    if pred_result.get('success', False):
                        y_pred_train = np.array(pred_result['predictions'], dtype=float)
                    else:
                        raise RuntimeError(f"TabPFN training set prediction failed: {pred_result.get('error', 'Unknown error')}")
                else:
                    raise RuntimeError("TabPFN training set prediction subprocess failed")
                
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
            else:
                y_pred_train = model.predict(X_train_s).astype(float)
            if not args.no_scaling and y_scaler_target is not None:
                y_pred_train = np.clip(y_pred_train, SCALE_MIN, SCALE_MAX)
            r2_train = r2_score(y_train_s, y_pred_train)
            mae_train = mean_absolute_error(y_train_s, y_pred_train)
            
            logger.info(f"ML baseline (frozen, trained on full set: {len(X_train_s)} samples):")
            logger.info(f"  TEST:  R2={r2:.4f} MAE={mae:.4f} MSE={mse:.4f}")
            logger.info(f"  TRAIN: R2={r2_train:.4f} MAE={mae_train:.4f}")
            
            if r2_train > 0 and r2 < 0:
                logger.warning(f"⚠️  Model overfitting: Train R²={r2_train:.4f} > 0, but Test R²={r2:.4f} < 0")
                logger.warning(f"   Model fits training data but generalizes poorly to test set")
            elif r2_train < 0 and r2 < 0:
                logger.warning(f"⚠️  Model underfitting: Both Train R²={r2_train:.4f} and Test R²={r2:.4f} are negative")
                logger.warning(f"   Linear model cannot capture the relationships in this dataset")
            
            ml_baseline_metrics["train_r2"] = r2_train
            ml_baseline_metrics["train_mae"] = mae_train
        except Exception as e:
            logger.warning(f"Failed to compute ML baseline metrics: {e}")
    else:
        # LLM-only mode: no ML residuals or baseline metrics
        ml_baseline_metrics = {}
        logger.info(f"LLM-only mode: Training on {len(X_topk)} samples (no ML baseline)")
    
    # Compute additional baselines (TabPFN, EBM, SHAP, LLM-LEx)
    logger.info("=" * 80)
    logger.info("COMPUTING ADDITIONAL BASELINES")
    logger.info("=" * 80)
    if not _HAS_TABPFN:
        logger.warning("⚠ TabPFN not available - skipping TabPFN baseline")
        logger.warning("  Install with: pip install tabpfn")
        logger.warning("  Note: TabPFN requires GPU for datasets >1000 samples")
    if not _HAS_LLMLEX:
        logger.warning("⚠ LLM-LEx not available - skipping LLM-LEx baseline")
        logger.warning("  Install with: pip install llmlex or clone from https://github.com/harveyThomas4692/llmlex")
        logger.warning("  Note: LLM-LEx requires OPENROUTER_API_KEY in .env file")
    # All baselines are evaluated on TEST set (for fair comparison with MA-ICL)
    additional_baselines = compute_additional_baselines(
        X_train_s, y_train_s, X_test_s, y_test_s,
        task_type="regression",
        feature_cols=feature_cols,
        y_scaler=y_scaler_target,
        no_scaling=args.no_scaling,
        baseline_models=args.baseline_models
    )

    maicl = TrainableMAICL(
        llm, feature_cols, scaler,
        use_ml_mechanism=bool(args.use_ml),
        dataset_name=ds_label,  # Use actual dataset label instead of generic "BIO/BIOTECH Regression"
        y_scaler=None,
        pretrained_ml_mechanism=pretrained_ml if args.use_ml else None,  # Only pass ML mechanism if enabled
        data_insights=None,
        task_type="regression",
        class_names=None,
        regression_loss_metric=args.regression_loss,
        num_mechanisms_unknown=args.num_mechanisms_unknown,
        use_scaling=not args.no_scaling  # Enable scaling unless --no_scaling flag is set
    )
    # Set output directory early to ensure all artifacts are saved to the correct location
    maicl.output_dir = output_dir
    if ml_baseline_metrics:
        try:
            maicl.set_ml_baseline_performance(ml_baseline_metrics)
        except Exception:
            pass

    # Pre metrics (MAE as loss; also R2/MSE)
    # Always evaluate on TEST set for final comparison
    logger.info("=" * 80)
    logger.info("PRE-TRAINING EVALUATION")
    logger.info("=" * 80)
    
    pre = maicl.evaluate(X_test_s, y_test_s, X_train_s, y_train_s, return_details=True, relax_routing=bool(args.relax_eval),
                         X_original=X_original_test, X_pool_original=X_original_train)
    pre_r2 = float(pre.get('r2', 0.0))
    pre_mae = float(pre.get('mae', 0.0))
    pre_rmse = float(pre.get('rmse', 0.0))
    pre_mse = float(pre_rmse ** 2)
    logger.info(f"Pre-training: R2={pre_r2:.4f} MAE={pre_mae:.4f} MSE={pre_mse:.4f}")
    
    # Extract pre-training predictions from evaluate results
    y_pred_pre = np.array(pre.get('predictions', [])) if 'predictions' in pre else None
    
    # Track individual mechanism performance (pre-training)
    pre_mechanism_performance = {}
    if hasattr(maicl, 'mechanism_performance_snapshot'):
        pre_mechanism_performance = maicl.mechanism_performance_snapshot.copy()
        logger.info(f"Pre-training mechanism performance: {pre_mechanism_performance}")
    
    # Generate initial LLM mechanism(s) before pre-training evaluation for better LLM-only baseline
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING INITIAL LLM MECHANISM(S) FOR PRE-TRAINING EVALUATION")
    logger.info("=" * 80)
    initial_llm_mechanisms_count = len([t for t in maicl.mechanism_types if t == "llm"])
    if initial_llm_mechanisms_count == 0:
        logger.info("No LLM mechanisms found. Generating initial mechanism(s) for pre-training evaluation...")
        try:
            # Set X_train_original for mechanism generator (needed for DeepChem SMILES strings)
            if is_deepchem_dataset and X_original_topk:
                maicl.mech_generator.X_train_original = X_original_topk
                logger.info("  Set X_train_original for mechanism generator (DeepChem dataset with SMILES)")
            
            # Use ML residuals if available, otherwise use simple baseline
            if args.use_ml and residuals_topk is not None and len(residuals_topk) == len(X_topk):
                prediction_errors = np.abs(residuals_topk)
                ml_residuals_for_init = residuals_topk
            else:
                # Simple baseline: use mean-centered errors
                prediction_errors = np.abs(y_topk - np.mean(y_topk))
                ml_residuals_for_init = None
            
            # Generate initial mechanism(s)
            # Store residual source for logging (initial mechanisms always use ML residuals if available)
            if args.use_ml and residuals_topk is not None:
                residual_source = "ML Model"
            else:
                residual_source = "Mean-Centered Errors (no ML residuals)"
            maicl.mech_generator._last_residual_source = residual_source
            
            initial_mechanisms = maicl.mech_generator.generate_unknown_mechanisms(
                X_topk, y_topk, prediction_errors, ml_residuals=ml_residuals_for_init
            )
            
            if initial_mechanisms and len(initial_mechanisms) > 0:
                # Update mechanisms list
                maicl.mech_generator.unknown_mechanisms = initial_mechanisms
                maicl.mechanisms = maicl.mech_generator.get_all_mechanisms()
                maicl.mechanism_types = maicl.mech_generator.get_mechanism_types()
                logger.info(f"  ✓ Generated {len(initial_mechanisms)} initial LLM mechanism(s) for pre-training evaluation")
                logger.info(f"  Total mechanisms now: {len(maicl.mechanisms)} ({len([t for t in maicl.mechanism_types if t == 'llm'])} LLM + {len([t for t in maicl.mechanism_types if t == 'ml'])} ML)")
                
                # Log latent_zs for initial mechanism generation (iteration 0 / pre-training)
                # Use the output_dir from maicl if available, otherwise use fold_output_dir
                try:
                    output_dir_for_logging = getattr(maicl, 'output_dir', None) or (fold_output_dir if 'fold_output_dir' in locals() else None)
                    maicl._log_latent_zs(iteration=0, output_dir=output_dir_for_logging, residual_source=residual_source)
                except Exception as e:
                    logger.warning(f"  Failed to log latent_zs for initial mechanism generation: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning("  ⚠️  Failed to generate initial LLM mechanisms")
        except Exception as e:
            logger.warning(f"Failed to generate initial LLM mechanisms: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info(f"Found {initial_llm_mechanisms_count} existing LLM mechanism(s), skipping initial generation")
    
    # Evaluate LLM-only mechanisms pre-training (excluding ML) to assess initial LLM performance
    logger.info("\n" + "=" * 80)
    logger.info("LLM-ONLY EVALUATION PRE-TRAINING (excluding ML mechanisms)")
    logger.info("=" * 80)
    llm_only_pre_metrics = None
    try:
        # For DeepChem datasets, pass X_original to use SMILES strings instead of vectorized features
        llm_only_pre_kwargs = {}
        if is_deepchem_dataset:
            llm_only_pre_kwargs['X_original'] = X_original_test
            llm_only_pre_kwargs['X_pool_original'] = X_original_train
        llm_only_pre_metrics = maicl.evaluate_llm_only(X_test_s, y_test_s, X_train_s, y_train_s, 
                                                        return_details=True, k_shot=args.k_shot,
                                                        **llm_only_pre_kwargs)
        llm_only_pre_r2 = float(llm_only_pre_metrics.get('r2', -1.0))
        llm_only_pre_mae = float(llm_only_pre_metrics.get('mae', 1e9))
        llm_only_pre_mse = float(llm_only_pre_metrics.get('mse', 1e9))
        logger.info(f"LLM-only (pre-training): R2={llm_only_pre_r2:.4f} MAE={llm_only_pre_mae:.4f} MSE={llm_only_pre_mse:.4f}")
    except Exception as e:
        logger.warning(f"Failed to evaluate LLM-only mechanisms pre-training: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate pre-training plots
    if y_pred_pre is not None and len(y_pred_pre) > 0:
        save_scatter_plot(y_test_s, y_pred_pre,
                         f"Pre-Training: Predicted vs Actual ({ds_label}, TEST set)",
                         "pre_scatter.png", output_dir)
        save_residual_plot(y_test_s, y_pred_pre,
                          f"Pre-Training ({ds_label}, TEST set)",
                          "pre_residuals.png", output_dir)

    logger.info("TRAINING MA-ICL")
    logger.info("=" * 80)
    logger.info(f"Training on {len(X_topk)} top-K residual samples for {args.iterations} iterations")
    if args.acceptance_set == "test":
        logger.info(f"Using TEST set ({len(X_test_s)} samples) for acceptance evaluation")
    elif args.acceptance_set == "train":
        logger.info(f"Using TRAIN set ({len(X_topk)} samples) for acceptance evaluation")
    else:
        logger.info(f"Using VALIDATION set ({len(X_val_s)} samples) for acceptance evaluation")
    # Use selected set (test/validation/train) for acceptance; pass only TRAIN residuals to LLM
    # accept_eval_max=None means use full acceptance set
    # Use residuals_topk to ensure consistency with training data (X_topk)
    # For DeepChem datasets, pass X_original for validation and test sets to use SMILES strings
    train_kwargs = {'X_train_original': X_original_topk, 'output_dir': output_dir}
    if is_deepchem_dataset:
        train_kwargs['X_val_original'] = X_original_val
        train_kwargs['X_test_original'] = X_original_test
    maicl.train(X_topk, y_topk, X_val_s, y_val_s, iterations=args.iterations, 
                ml_residuals=residuals_topk if args.use_ml else None,  # Only pass residuals if ML is enabled
                accept_eval_max=None, X_test=X_test_s, y_test=y_test_s, 
                acceptance_set=args.acceptance_set, k_shot=args.k_shot, **train_kwargs)

    # Post metrics
    logger.info("=" * 80)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("=" * 80)
    
    # CRITICAL: Final evaluation always uses TEST set (regardless of acceptance_set)
    # acceptance_set is ONLY used during training iterations to accept/reject updates
    # All models (baselines, pre-training, post-training) are evaluated on TEST set
    X_final_eval = X_test_s
    y_final_eval = y_test_s
    X_original_final_eval = X_original_test if is_deepchem_dataset else None
    
    # CRITICAL: Always use relax_routing=True to match training mode
    # Training always uses relax_routing=True, so final evaluation must match
    final_relax_routing = True
    logger.info(f"Using SAME routing as best iteration for exact reproduction")
    logger.info(f"  - Evaluation set: TEST (all models evaluated on TEST set for fair comparison)")
    logger.info(f"  - Note: acceptance_set={args.acceptance_set} was only used during training iterations")
    logger.info(f"  - Routing mode: relax_routing=True (same as training)")
    
    # CRITICAL: Always preserve mechanism performance scores from best snapshot
    # This ensures final evaluation uses the exact same routing weights as the best iteration
    preserve_perf = True
    logger.info(f"Preserving mechanism performance scores from best snapshot (calculated on {args.acceptance_set} set during training)")
    logger.info(f"  Note: This ensures final evaluation uses the same routing as the best iteration")
    logger.info(f"  Note: Final evaluation is on TEST set, but routing weights were determined during training on {args.acceptance_set} set")
    
    # Log final accepted mechanisms for transparency
    final_mechanism_count = len(maicl.mechanisms) if hasattr(maicl, 'mechanisms') else 0
    final_llm_count = sum(1 for t in maicl.mechanism_types if t == "llm") if hasattr(maicl, 'mechanism_types') else 0
    final_ml_count = sum(1 for t in maicl.mechanism_types if t == "ml") if hasattr(maicl, 'mechanism_types') else 0
    logger.info(f"Evaluating with final accepted mechanisms: {final_mechanism_count} total ({final_llm_count} LLM, {final_ml_count} ML)")
    
    # Log best snapshot metrics for comparison
    if hasattr(maicl, '_best_iteration') and maicl._best_iteration is not None:
        best_iter = maicl._best_iteration
        logger.info(f"  [Best Iteration] Using mechanisms from iteration {best_iter}")
        # Try to get best snapshot metrics if available
        if hasattr(maicl, 'training_history') and 'best_snapshot' in str(maicl.training_history):
            # The best snapshot metrics are stored internally, log them if we can access them
            logger.info(f"  [Best Iteration] Best iteration {best_iter} achieved the best performance during training")
    
    # For DeepChem datasets, pass X_original to use SMILES strings instead of vectorized features
    # CRITICAL: Use same evaluation set as acceptance_set for exact reproduction
    # CRITICAL FIX: Explicitly pass k_shot to ensure final evaluation uses same value as training
    post = maicl.evaluate(X_final_eval, y_final_eval, X_train_s, y_train_s, return_details=True, 
                          relax_routing=final_relax_routing, preserve_mechanism_performance=preserve_perf,
                          k_shot=args.k_shot,  # CRITICAL: Use same k_shot as training to ensure consistency
                          X_original=X_original_final_eval,
                          X_pool_original=X_original_train if is_deepchem_dataset else None)
    post_mae = float(post.get('mae', 0.0))
    post_r2 = float(post.get('r2', 0.0))
    post_rmse = float(post.get('rmse', 0.0))
    post_mse = float(post_rmse ** 2)
    logger.info(f"Post-training (TEST set): R2={post_r2:.4f} MAE={post_mae:.4f} MSE={post_mse:.4f}")
    logger.info(f"  Note: All models evaluated on TEST set for fair comparison")
    
    # Compare with best iteration metrics if available
    if hasattr(maicl, '_best_iteration') and maicl._best_iteration is not None:
        best_iter = maicl._best_iteration
        best_r2 = getattr(maicl, '_best_r2', None)
        best_mae = getattr(maicl, '_best_mae', None)
        if best_r2 is not None:
            r2_diff = post_r2 - best_r2
            logger.info(f"  [Comparison] Best iteration {best_iter} had R2={best_r2:.4f} during training")
            logger.info(f"  [Comparison] Final evaluation R2={post_r2:.4f} (difference: {r2_diff:+.4f})")
            if abs(r2_diff) > 0.01:
                logger.warning(f"  [Comparison] ⚠️  Final R2 differs from best iteration by {abs(r2_diff):.4f}")
                logger.warning(f"  [Comparison] This may be due to few-shot example selection randomness or evaluation differences")
        if best_mae is not None:
            mae_diff = post_mae - best_mae
            logger.info(f"  [Comparison] Best iteration {best_iter} had MAE={best_mae:.4f} during training")
            logger.info(f"  [Comparison] Final evaluation MAE={post_mae:.4f} (difference: {mae_diff:+.4f})")
    
    # Extract post-training predictions from evaluate results
    y_pred_post = np.array(post.get('predictions', [])) if 'predictions' in post else None
    
    # Track individual mechanism performance (post-training)
    post_mechanism_performance = {}
    if hasattr(maicl, 'mechanism_performance_snapshot'):
        post_mechanism_performance = maicl.mechanism_performance_snapshot.copy()
        logger.info(f"Post-training mechanism performance: {post_mechanism_performance}")
    
    # Generate post-training plots
    # Use the same evaluation set that was used for evaluation (for consistency)
    if y_pred_post is not None and len(y_pred_post) > 0:
        save_scatter_plot(y_final_eval, y_pred_post,
                         f"Post-Training: Predicted vs Actual ({ds_label}, TEST set)",
                         "post_scatter.png", output_dir)
        save_residual_plot(y_final_eval, y_pred_post,
                          f"Post-Training ({ds_label}, TEST set)",
                          "post_residuals.png", output_dir)
    
    # Evaluate LLM-only mechanisms (excluding ML) to assess LLM learning
    logger.info("\n" + "=" * 80)
    logger.info("LLM-ONLY EVALUATION (excluding ML mechanisms)")
    logger.info("=" * 80)
    llm_only_metrics = None
    y_pred_llm_only = None
    try:
        # For DeepChem datasets, pass X_original to use SMILES strings instead of vectorized features
        # LLM-only evaluation also uses TEST set for consistency
        llm_only_kwargs = {}
        if is_deepchem_dataset:
            llm_only_kwargs['X_original'] = X_original_test
            llm_only_kwargs['X_pool_original'] = X_original_train
        llm_only_metrics = maicl.evaluate_llm_only(X_test_s, y_test_s, X_train_s, y_train_s, 
                                                    return_details=True, k_shot=args.k_shot,
                                                    **llm_only_kwargs)
        llm_only_r2 = float(llm_only_metrics.get('r2', -1.0))
        llm_only_mae = float(llm_only_metrics.get('mae', 1e9))
        llm_only_mse = float(llm_only_metrics.get('mse', 1e9))
        logger.info(f"LLM-only: R2={llm_only_r2:.4f} MAE={llm_only_mae:.4f} MSE={llm_only_mse:.4f}")
        
        y_pred_llm_only = np.array(llm_only_metrics.get('predictions', [])) if 'predictions' in llm_only_metrics else None
        if y_pred_llm_only is not None and len(y_pred_llm_only) > 0:
            save_scatter_plot(y_test_s, y_pred_llm_only,
                             f"LLM-Only: Predicted vs Actual ({ds_label}, TEST set)",
                             "llm_only_scatter.png", output_dir)
            save_residual_plot(y_test_s, y_pred_llm_only,
                              f"LLM-Only ({ds_label}, TEST set)",
                              "llm_only_residuals.png", output_dir)
    except Exception as e:
        logger.warning(f"Failed to evaluate LLM-only mechanisms: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate performance comparison visualizations
    logger.info("\n[Visualizations] Generating performance comparison plots...")
    try:
        create_result_visualizations(
            y_test=y_test_s,
            ml_baseline_metrics=ml_baseline_metrics,
            pre_metrics=pre,
            post_metrics=post,
            task_type="regression",
            class_names=None,
            output_dir=output_dir,
            llm_only_metrics=llm_only_metrics,
            llm_only_pre_metrics=llm_only_pre_metrics,
            additional_baselines=additional_baselines
        )
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")
    
    # Save final metrics
    # Compute improvements vs ML baseline
    ml_r2 = ml_baseline_metrics.get('r2', 0.0)
    ml_mae = ml_baseline_metrics.get('mae', 0.0)
    ml_mse = ml_baseline_metrics.get('mse', 0.0)
    
    final_results = {
        "dataset": ds_label,
        "dataset_name": dataset_name_lower,
        "run_name": run_name,
        "model_name": args.model_name,
        "ml_mechanism": args.ml_mech,
        "top_k": args.top_k,
        "iterations": args.iterations,
        "n_train": len(X_train_s),
        "n_val": len(X_val_s),
        "n_test": len(X_test_s),
        "n_topk_trained": len(X_topk),
        "ml_baseline": ml_baseline_metrics,
        "additional_baselines": additional_baselines,
        "pre_training": {
            "r2": pre_r2,
            "mae": pre_mae,
            "mse": pre_mse,
            "rmse": pre_rmse,
            "mechanism_performance": pre_mechanism_performance
        },
        "post_training": {
            "r2": post_r2,
            "mae": post_mae,
            "mse": post_mse,
            "rmse": post_rmse,
            "mechanism_performance": post_mechanism_performance
        },
        "llm_only_pre": {
            "r2": float(llm_only_pre_metrics.get('r2', -1.0)) if llm_only_pre_metrics else None,
            "mae": float(llm_only_pre_metrics.get('mae', 1e9)) if llm_only_pre_metrics else None,
            "mse": float(llm_only_pre_metrics.get('mse', 1e9)) if llm_only_pre_metrics else None,
            "rmse": float(np.sqrt(llm_only_pre_metrics.get('mse', 1e9))) if llm_only_pre_metrics and 'mse' in llm_only_pre_metrics else None
        } if llm_only_pre_metrics else None,
        "llm_only_post": {
            "r2": float(llm_only_metrics.get('r2', -1.0)) if llm_only_metrics else None,
            "mae": float(llm_only_metrics.get('mae', 1e9)) if llm_only_metrics else None,
            "mse": float(llm_only_metrics.get('mse', 1e9)) if llm_only_metrics else None,
            "rmse": float(np.sqrt(llm_only_metrics.get('mse', 1e9))) if llm_only_metrics and 'mse' in llm_only_metrics else None
        } if llm_only_metrics else None,
        "mechanism_info": {
            "total_mechanisms": len(maicl.mechanisms),
            "mechanism_types": maicl.mechanism_types,
            "ml_mechanism_index": [i for i, t in enumerate(maicl.mechanism_types) if t == "ml"],
            "llm_mechanism_indices": [i for i, t in enumerate(maicl.mechanism_types) if t == "llm"]
        },
        "improvements": {
            "training_improvement": {
                "r2_delta": post_r2 - pre_r2,
                "mae_delta": pre_mae - post_mae,  # Positive is better (reduction)
                "mse_delta": pre_mse - post_mse,   # Positive is better (reduction)
            },
            "vs_ml_baseline_pre": {
                "r2_delta": pre_r2 - ml_r2,
                "mae_delta": ml_mae - pre_mae,  # Positive is better (reduction)
                "mse_delta": ml_mse - pre_mse,   # Positive is better (reduction)
            },
            "vs_ml_baseline_post": {
                "r2_delta": post_r2 - ml_r2,
                "mae_delta": ml_mae - post_mae,  # Positive is better (reduction)
                "mse_delta": ml_mse - post_mse,   # Positive is better (reduction)
            }
        }
    }
    
    results_file = os.path.join(output_dir, "final_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Saved final results to {results_file}")
    
    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Dataset: {ds_label}")
    logger.info(f"ML Mechanism: {args.ml_mech} ({model_name})")
    logger.info(f"Total Mechanisms: {len(maicl.mechanisms)} ({len([t for t in maicl.mechanism_types if t == 'llm'])} LLM + {len([t for t in maicl.mechanism_types if t == 'ml'])} ML)")
    logger.info("")
    logger.info("Performance Comparison:")
    logger.info(f"  ML Baseline:        R2={ml_r2:.4f}, MAE={ml_mae:.4f}, MSE={ml_mse:.4f}")
    
    # Additional baselines
    if additional_baselines:
        if 'tabpfn' in additional_baselines:
            tabpfn_r2 = additional_baselines['tabpfn'].get('r2', 0.0)
            tabpfn_mae = additional_baselines['tabpfn'].get('mae', 0.0)
            tabpfn_mse = additional_baselines['tabpfn'].get('mse', 0.0)
            logger.info(f"  TabPFN Baseline:    R2={tabpfn_r2:.4f}, MAE={tabpfn_mae:.4f}, MSE={tabpfn_mse:.4f}")
        elif not _HAS_TABPFN:
            logger.info(f"  TabPFN Baseline:    [SKIPPED - not installed]")
        if 'ebm' in additional_baselines:
            ebm_r2 = additional_baselines['ebm'].get('r2', 0.0)
            ebm_mae = additional_baselines['ebm'].get('mae', 0.0)
            ebm_mse = additional_baselines['ebm'].get('mse', 0.0)
            logger.info(f"  EBM Baseline:      R2={ebm_r2:.4f}, MAE={ebm_mae:.4f}, MSE={ebm_mse:.4f}")
        if 'shap' in additional_baselines:
            shap_r2 = additional_baselines['shap'].get('r2', 0.0)
            shap_mae = additional_baselines['shap'].get('mae', 0.0)
            shap_mse = additional_baselines['shap'].get('mse', 0.0)
            logger.info(f"  SHAP Baseline:     R2={shap_r2:.4f}, MAE={shap_mae:.4f}, MSE={shap_mse:.4f}")
    
    logger.info(f"  Pre-training:       R2={pre_r2:.4f}, MAE={pre_mae:.4f}, MSE={pre_mse:.4f}")
    if llm_only_pre_metrics:
        llm_only_pre_r2 = float(llm_only_pre_metrics.get('r2', -1.0))
        llm_only_pre_mae = float(llm_only_pre_metrics.get('mae', 1e9))
        llm_only_pre_mse = float(llm_only_pre_metrics.get('mse', 1e9))
        logger.info(f"  LLM-only (pre):    R2={llm_only_pre_r2:.4f}, MAE={llm_only_pre_mae:.4f}, MSE={llm_only_pre_mse:.4f}")
    logger.info(f"  Post-training:      R2={post_r2:.4f}, MAE={post_mae:.4f}, MSE={post_mse:.4f}")
    if llm_only_metrics:
        llm_only_r2 = float(llm_only_metrics.get('r2', -1.0))
        llm_only_mae = float(llm_only_metrics.get('mae', 1e9))
        llm_only_mse = float(llm_only_metrics.get('mse', 1e9))
        logger.info(f"  LLM-only (post):   R2={llm_only_r2:.4f}, MAE={llm_only_mae:.4f}, MSE={llm_only_mse:.4f}")
    logger.info("")
    logger.info("Training Improvement (Post vs Pre):")
    logger.info(f"  ΔR2={final_results['improvements']['training_improvement']['r2_delta']:+.4f}, "
                f"ΔMAE={final_results['improvements']['training_improvement']['mae_delta']:+.4f}, "
                f"ΔMSE={final_results['improvements']['training_improvement']['mse_delta']:+.4f}")
    logger.info("")
    logger.info("Pre-training vs ML Baseline:")
    logger.info(f"  ΔR2={final_results['improvements']['vs_ml_baseline_pre']['r2_delta']:+.4f}, "
                f"ΔMAE={final_results['improvements']['vs_ml_baseline_pre']['mae_delta']:+.4f}, "
                f"ΔMSE={final_results['improvements']['vs_ml_baseline_pre']['mse_delta']:+.4f}")
    logger.info("")
    logger.info("Post-training vs ML Baseline:")
    logger.info(f"  ΔR2={final_results['improvements']['vs_ml_baseline_post']['r2_delta']:+.4f}, "
                f"ΔMAE={final_results['improvements']['vs_ml_baseline_post']['mae_delta']:+.4f}, "
                f"ΔMSE={final_results['improvements']['vs_ml_baseline_post']['mse_delta']:+.4f}")
    logger.info("")
    
    # Show individual mechanism performance if available
    if pre_mechanism_performance or post_mechanism_performance:
        logger.info("Individual Mechanism Performance Scores:")
        ml_indices = [i for i, t in enumerate(maicl.mechanism_types) if t == "ml"]
        llm_indices = [i for i, t in enumerate(maicl.mechanism_types) if t == "llm"]
        
        if ml_indices:
            ml_idx = ml_indices[0]
            pre_ml_perf = pre_mechanism_performance.get(ml_idx, "N/A")
            post_ml_perf = post_mechanism_performance.get(ml_idx, "N/A")
            logger.info(f"  ML Mechanism (idx={ml_idx}):  Pre={pre_ml_perf}, Post={post_ml_perf}")
        
        for llm_idx in llm_indices:
            pre_llm_perf = pre_mechanism_performance.get(llm_idx, "N/A")
            post_llm_perf = post_mechanism_performance.get(llm_idx, "N/A")
            logger.info(f"  LLM Mechanism (idx={llm_idx}): Pre={pre_llm_perf}, Post={post_llm_perf}")
    
    logger.info("=" * 80)
    
    # Optionally evaluate individual mechanisms
    if args.evaluate_individual_mechanisms:
        logger.info("=" * 80)
        logger.info("EVALUATING INDIVIDUAL MECHANISMS")
        logger.info("=" * 80)
        try:
            # Import the evaluation module
            import sys
            eval_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_individual_mechanisms.py")
            if os.path.exists(eval_module_path):
                # Run evaluation for the final iteration
                final_iteration = args.iterations
                mechanisms_file = os.path.join(output_dir, f"mechanisms_iter_{final_iteration}.txt")
                
                if os.path.exists(mechanisms_file):
                    logger.info(f"Evaluating mechanisms from iteration {final_iteration}")
                    
                    # Prepare data for evaluation
                    eval_data = {
                        "X_train": X_train_s,
                        "y_train": y_train_s,
                        "X_test": X_test_s,
                        "y_test": y_test_s,
                        "feature_cols": feature_cols,
                        "scaler": scaler,
                        "y_scaler": y_scaler_target,
                        "dataset_name": ds_label
                    }
                    
                    # Import and run evaluation
                    from evaluate_individual_mechanisms import (
                        parse_mechanisms_file,
                        extract_formula_from_llm_mechanism,
                        evaluate_llm_formula,
                        evaluate_ml_mechanism,
                        save_mechanism_scatter_plot
                    )
                    
                    mechanisms = parse_mechanisms_file(mechanisms_file)
                    individual_results = []
                    
                    for mech_idx, (mech_type, mech_text) in enumerate(mechanisms):
                        logger.info(f"\nEvaluating Mechanism {mech_idx+1}/{len(mechanisms)}: [{mech_type.upper()}]")
                        try:
                            if mech_type == "llm" or mech_type == "known":
                                formula = extract_formula_from_llm_mechanism(mech_text)
                                if formula is None:
                                    formula = mech_text
                                # For DeepChem datasets, pass X_original (SMILES strings) for molecular property computation
                                # X_original_train and X_original_test are defined earlier in this function
                                train_preds = evaluate_llm_formula(formula, X_train_s, feature_cols, scaler, y_scaler_target, X_original=X_original_train)
                                test_preds = evaluate_llm_formula(formula, X_test_s, feature_cols, scaler, y_scaler_target, X_original=X_original_test)
                            elif mech_type == "ml":
                                train_preds, test_preds = evaluate_ml_mechanism(
                                    mech_text, X_train_s, y_train_s, X_test_s, feature_cols, scaler, y_scaler_target
                                )
                            else:
                                continue
                            
                            train_r2 = r2_score(y_train_s, train_preds)
                            train_mae = mean_absolute_error(y_train_s, train_preds)
                            train_mse = mean_squared_error(y_train_s, train_preds)
                            test_r2 = r2_score(y_test_s, test_preds)
                            test_mae = mean_absolute_error(y_test_s, test_preds)
                            test_mse = mean_squared_error(y_test_s, test_preds)
                            
                            individual_results.append({
                                "mechanism_index": mech_idx,
                                "mechanism_type": mech_type,
                                "train_metrics": {"r2": float(train_r2), "mae": float(train_mae), "mse": float(train_mse)},
                                "test_metrics": {"r2": float(test_r2), "mae": float(test_mae), "mse": float(test_mse)}
                            })
                            
                            logger.info(f"  Train: R2={train_r2:.4f}, MAE={train_mae:.4f}, MSE={train_mse:.4f}")
                            logger.info(f"  Test:  R2={test_r2:.4f}, MAE={test_mae:.4f}, MSE={test_mse:.4f}")
                            
                            # Generate scatter plots for train and test
                            save_mechanism_scatter_plot(
                                y_train_s, train_preds,
                                f"Mechanism {mech_idx} ({mech_type.upper()}) - Train Set",
                                f"mechanism_{mech_idx}_{mech_type}_train_scatter.png",
                                output_dir,
                                train_r2, train_mae, train_mse
                            )
                            save_mechanism_scatter_plot(
                                y_test_s, test_preds,
                                f"Mechanism {mech_idx} ({mech_type.upper()}) - Test Set",
                                f"mechanism_{mech_idx}_{mech_type}_test_scatter.png",
                                output_dir,
                                test_r2, test_mae, test_mse
                            )
                        except Exception as e:
                            logger.warning(f"  Failed to evaluate mechanism {mech_idx+1}: {e}")
                    
                    # Save individual mechanism results
                    individual_results_file = os.path.join(output_dir, f"individual_mechanism_performance_iter_{final_iteration}.json")
                    with open(individual_results_file, 'w') as f:
                        json.dump({
                            "iteration": final_iteration,
                            "dataset": ds_label,
                            "n_train": len(X_train_s),
                            "n_test": len(X_test_s),
                            "mechanisms": individual_results
                        }, f, indent=2)
                    logger.info(f"\nIndividual mechanism results saved to: {individual_results_file}")
                    
                    # Also save CSV summary
                    try:
                        import pandas as pd
                        csv_data = []
                        for r in individual_results:
                            csv_data.append({
                                "mechanism_index": r["mechanism_index"],
                                "mechanism_type": r["mechanism_type"],
                                "train_r2": r["train_metrics"]["r2"],
                                "train_mae": r["train_metrics"]["mae"],
                                "train_mse": r["train_metrics"]["mse"],
                                "test_r2": r["test_metrics"]["r2"],
                                "test_mae": r["test_metrics"]["mae"],
                                "test_mse": r["test_metrics"]["mse"],
                            })
                        df = pd.DataFrame(csv_data)
                        csv_file = os.path.join(output_dir, f"individual_mechanism_performance_iter_{final_iteration}.csv")
                        df.to_csv(csv_file, index=False)
                        logger.info(f"CSV summary saved to: {csv_file}")
                    except Exception as e:
                        logger.warning(f"Could not save CSV summary: {e}")
                else:
                    logger.warning(f"Mechanisms file not found: {mechanisms_file}")
            else:
                logger.warning(f"Evaluation module not found: {eval_module_path}")
        except Exception as e:
            logger.warning(f"Failed to evaluate individual mechanisms: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate optimal GFP yield combinations (only for GFP yield dataset, not in CV mode)
    if dataset_name_lower == "gfp_yield" and cv_folds == 0:
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING OPTIMAL GFP YIELD COMBINATIONS")
        logger.info("=" * 80)
        try:
            suggestions = suggest_optimal_gfp_combinations(
                maicl=maicl,
                X_train_original=X_original_train,
                feature_cols=feature_cols,
                scaler=scaler,
                y_scaler_target=y_scaler_target,
                n_suggestions=20,
                n_candidates=1000,
                output_dir=output_dir
            )
            
            # Add suggestions to final results
            if suggestions:
                final_results['optimal_combinations'] = {
                    'n_suggestions': len(suggestions),
                    'suggestions': suggestions[:20]  # Store top 10 in results
                }
                # Update results file
                with open(results_file, 'w') as f:
                    json.dump(final_results, f, indent=2)
                logger.info(f"Updated final_results.json with optimal combinations")
        except Exception as e:
            logger.warning(f"Failed to generate optimal GFP combinations: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()


