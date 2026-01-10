#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA-ICL Classification Runner with Top-K Residuals
---------------------------------------------------
- Trains an ML baseline on scaled features
- Computes residuals on train set and selects top-K by residual magnitude
- Trains MA-ICL on the residual-difficult subset (optimization target: F1 or accuracy, configurable via --classification_loss)
- Reports pre/post: F1, Accuracy on the test set

Usage examples:
  python 017_maicl_classification_residual_topk.py --dataset car --top_k 50 --iterations 5
  python 017_maicl_classification_residual_topk.py --dataset synthetic5 --top_k 75 --ml_mech xgboost
  python 017_maicl_classification_residual_topk.py --dataset iris --top_k 50 --classification_loss acc
"""

import os
import argparse
import numpy as np
from typing import Dict, Any, Optional, List
from collections import Counter
import logging
from pathlib import Path
import json

# Suppress TensorFlow warnings and prevent hangs (if TensorFlow is used by DeepChem)
# Set these BEFORE any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow messages (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that can cause hangs
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL which can cause issues on macOS

try:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*tensorflow.*')
    warnings.filterwarnings('ignore', message='.*mutex.*')
    # Suppress TensorFlow mutex warnings
    import logging as tf_logging
    tf_logging.getLogger('tensorflow').setLevel(tf_logging.CRITICAL)
    tf_logging.getLogger('tensorflow.core').setLevel(tf_logging.CRITICAL)
except Exception:
    pass

from sklearn.metrics import accuracy_score, f1_score

try:
    import openml
    _HAS_OPENML = True
except Exception:
    _HAS_OPENML = False

try:
    import importlib.util as _importlib_util_pandas
    _HAS_PANDAS = _importlib_util_pandas.find_spec("pandas") is not None
except Exception:
    _HAS_PANDAS = False


def _get_pandas():
    """Lazily import pandas (it can segfault in some environments if imported at module import time)."""
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as e:
        raise RuntimeError(
            "pandas is required for this dataset/loader but could not be imported. "
            "Try installing a compatible pandas build for your Python environment."
        ) from e


def _is_missing(x: Any) -> bool:
    """pandas-free missing-value check."""
    if x is None:
        return True
    # NaN
    try:
        if isinstance(x, float) and np.isnan(x):
            return True
    except Exception:
        pass
    # Empty string
    try:
        if isinstance(x, str) and x.strip() == "":
            return True
    except Exception:
        pass
    return False

# IMPORTANT: do NOT import DeepChem at module import time.
# DeepChem may transitively import TensorFlow/RDKit and can crash/segfault in some environments
# even when just running `--help`. We import DeepChem lazily inside
# `load_deepchem_classification_dataset()` when (and only when) needed.
try:
    import importlib.util as _importlib_util
    _HAS_DEEPCHEM = _importlib_util.find_spec("deepchem") is not None
    _DEEPCHEM_ERROR = None
except Exception as e:
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
    find_optimal_threshold,
    create_result_visualizations,
    SCALE_MIN,
    SCALE_MAX,
)
from maicl_llm import OpenAIAPIKeyManager

# Optional baseline imports
try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
except:
    _HAS_TABPFN = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _HAS_EBM = True
except:
    _HAS_EBM = False

try:
    import shap
    _HAS_SHAP = True
except:
    _HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except:
    _HAS_XGB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NOTE: this constant is intentionally aligned with the CLI default below.
# Override via --model_name or MAICL_MODEL_NAME.
MODEL_NAME = os.environ.get("MAICL_MODEL_NAME", "gpt-4o-mini")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def _load_env():
    """Load environment variables from .env if available (no error if missing)."""
    try:
        # Prefer project-local .env next to this file
        from dotenv import load_dotenv, find_dotenv
        here = Path(__file__).resolve().parent
        env_path = here / ".env"
        # Load explicit path first, then fallback to nearest .env
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.debug(f"Loaded .env from: {env_path}")
        else:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
                logger.debug(f"Loaded .env from: {found}")
            else:
                logger.debug("No .env file found")
    except Exception as e:
        logger.debug(f"Failed to load .env file: {e}")
        pass

def _get_gemini_llm(model_name: str):
    # Ensure .env is loaded before reading env vars
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

def _get_openai_llm(model_name: str):
    # Ensure .env is loaded before reading env vars
    _load_env()
    keys = []
    for env_key in ["OPENAI_API_KEY", "OPENAI_API_KEY_1", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"]:
        val = os.environ.get(env_key)
        if val:
            # Strip whitespace and validate key format
            val = val.strip()
            # Remove quotes if present (common .env file issue)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            val = val.strip()  # Strip again after removing quotes
            
            if val:
                # Basic validation: OpenAI keys should start with 'sk-'
                if not val.startswith('sk-'):
                    logger.warning(f"Warning: OpenAI API key from {env_key} doesn't start with 'sk-'. "
                                 f"Key preview: {val[:10]}... (first 10 chars)")
                keys.append(val)
    if not keys:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY or OPENAI_API_KEY_1/2/3 in your .env file or environment variables.")
    
    # Log key info (masked for security)
    key_preview = keys[0][:10] + "..." + keys[0][-4:] if len(keys[0]) > 14 else keys[0][:10] + "..."
    logger.info(f"Using OpenAI API key: {key_preview} (from {len(keys)} key(s))")
    logger.info(f"Key length: {len(keys[0])} characters")
    
    # Validate key format more strictly
    if len(keys[0]) < 20:
        logger.warning(f"Warning: API key seems too short ({len(keys[0])} chars). OpenAI keys are typically 50+ characters.")
    
    km = OpenAIAPIKeyManager(keys, model_name=model_name)
    return BatchedLLM(km)

def _get_llm(model_name: str):
    """Get LLM instance based on model name - automatically detects provider"""
    model_name_lower = model_name.lower()
    if model_name_lower.startswith("gpt") or model_name_lower.startswith("o1") or "openai" in model_name_lower:
        # OpenAI model
        return _get_openai_llm(model_name)
    elif model_name_lower.startswith("gemini"):
        # Gemini model
        return _get_gemini_llm(model_name)
    else:
        # Default to Gemini for backward compatibility
        logger.warning(f"Unknown model provider for '{model_name}', defaulting to Gemini. "
                      f"Use 'gpt-4o-mini' or 'gemini-2.0-flash' for explicit provider selection.")
        return _get_gemini_llm(model_name)

def load_openml_classification(dataset_name: str, max_samples: int):
    if not _HAS_OPENML:
        raise RuntimeError("OpenML not installed. pip install openml")
    dataset_ids = {
        "car": 22, "iris": 61, "wine": 187, "zoo": 62, "glass": 1468, "vehicle": 54,
        "soybean": 41, "lymphography": 10, "ecoli": 40,
        "adult": 1590, "credit": 31, "vote": 56, "mushroom": 24,
        # Datasets with many classes (10+)
        "letter": 6,  # 26 classes - Letter Recognition
        "mfeat": 12,  # 10 classes - Multiple Features (mfeat-factors)
        "pendigits": 32,  # 10 classes - Pen-based recognition of handwritten digits
        "optdigits": 28,  # 10 classes - Optical recognition of handwritten digits
        "avila": 1459,  # 12 classes - Avila dataset
        "amazon": 1457,  # 50 classes - Amazon dataset
        "plant-margin": 1491,  # 100 classes - One-hundred plants margin
        "plant-shape": 1492,  # 100 classes - One-hundred plants shape
        "plant-texture": 1493,  # 100 classes - One-hundred plants texture
        # Gene expression cancer datasets
        "leukemia": 1104,  # Leukemia classification (AML vs ALL) - 72 samples, 7129 genes
        "colon-cancer": 1100,  # Colon cancer classification (tumor vs normal) - 62 samples, 2000 genes
        "colon": 1100,  # Alias for colon-cancer
        # Biological/genomic datasets
        "mice-protein": 40966,  # Mice Protein Expression - 8 classes, 1080 samples, 77 protein expression levels
        "miceprotein": 40966,  # Alias for mice-protein
        "splice": 46,  # Splice (Gene Sequences) - 3 classes, 3175 samples, 60 DNA sequence positions
        # Note: primary-tumor (ID 47) is private and requires authentication
        # Note: TCGA (The Cancer Genome Atlas) datasets require dbGaP authentication and are not directly
        #       accessible via OpenML. To use TCGA data, download from GDC (https://portal.gdc.cancer.gov/)
        #       or GEO (https://www.ncbi.nlm.nih.gov/geo/) and create a custom loader function.
    }
    did = dataset_ids.get(dataset_name.lower(), 22)
    try:
        ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=True)
    except Exception as e:
        # Check if it's a private dataset error (handle both old and new exception types)
        error_msg = str(e).lower()
        if "private" in error_msg or "no access" in error_msg or "access granted" in error_msg:
            raise RuntimeError(
                f"Dataset '{dataset_name}' (OpenML ID {did}) is private and requires authentication. "
                f"Please use a different dataset or configure OpenML authentication. "
                f"Available public datasets: car, iris, wine, zoo, glass, vehicle, soybean, lymphography, ecoli, adult, credit, vote, mushroom, "
                f"letter (26 classes), mfeat (10), pendigits (10), optdigits (10), avila (12), amazon (50), "
                f"plant-margin (100), plant-shape (100), plant-texture (100), "
                f"leukemia (gene expression), colon-cancer/colon (gene expression)"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to load OpenML dataset '{dataset_name}' (ID {did}): {e}. "
                f"Please check your internet connection and OpenML API access."
            ) from e
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    if len(X) > max_samples:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=RANDOM_STATE)
            idx_all = np.arange(len(X))
            idx_sel, _ = next(sss.split(idx_all, y))
            X = X.iloc[idx_sel].reset_index(drop=True)
            y = y.iloc[idx_sel].reset_index(drop=True)
        except Exception:
            X = X.sample(n=max_samples, random_state=RANDOM_STATE).reset_index(drop=True)
            y = y.loc[X.index].reset_index(drop=True)
    categorical, numeric = [], []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical.append(col)
        else:
            numeric.append(col)
    X_original = []
    for i in range(len(X)):
        d = {}
        for col in X.columns:
            d[col] = str(X.iloc[i][col]) if col in categorical else float(X.iloc[i][col])
        X_original.append(d)
    from sklearn.preprocessing import LabelEncoder
    feature_encoders: Dict[str, Any] = {}
    X_parts = []
    for col in X.columns:
        if col in categorical:
            le = LabelEncoder()
            X_parts.append(le.fit_transform(X[col].astype(str)).reshape(-1, 1))
            feature_encoders[col] = le
        else:
            X_parts.append(X[col].values.reshape(-1, 1))
    X_encoded = np.hstack(X_parts)
    
    # Handle missing values (NaN) - remove rows with any missing values
    # This is important for datasets like mice-protein that may have missing values
    # Check for NaN in X_encoded
    x_valid_mask = ~np.isnan(X_encoded).any(axis=1)
    # Check for NaN in y (handle both pandas Series and numpy arrays)
    if hasattr(y, 'isna'):
        # pandas Series - convert to numpy array for boolean operations
        y_valid_mask = ~y.isna().values
    else:
        # numpy array or array-like
        y_array = np.asarray(y)
        y_valid_mask = ~np.isnan(y_array)
    valid_mask = x_valid_mask & y_valid_mask
    if not valid_mask.all():
        n_missing = (~valid_mask).sum()
        logger.info(f"Removing {n_missing} samples with missing values (out of {len(X_encoded)} total)")
        X_encoded = X_encoded[valid_mask]
        # Filter y (pandas Series or numpy array)
        if hasattr(y, 'reset_index'):
            y = y[valid_mask].reset_index(drop=True)
        else:
            y = np.asarray(y)[valid_mask]
        X_original = [X_original[i] for i in range(len(X_original)) if valid_mask[i]]
        logger.info(f"After removing missing values: {len(X_encoded)} samples remaining")
    
    # Encode target labels and get class names
    # Convert to string first to preserve original class names (e.g., "unacc", "acc", "good", "vgood")
    y_str = y.astype(str)
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y_str)
    # LabelEncoder.classes_ gives the original class names in the order they were encoded
    # This preserves the actual class names (e.g., "unacc", "acc", "good", "vgood" for car dataset)
    class_names = y_le.classes_.tolist()
    
    # Get original feature column names (these are the actual feature names from the dataset)
    # For car dataset, these should be the original column names from OpenML
    feature_cols = X.columns.tolist()
    
    return X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders


def compute_additional_baselines(X_train, y_train, X_test, y_test, task_type="classification", 
                                 feature_cols=None, y_scaler=None, no_scaling=False,
                                 baseline_models=None):
    """Compute additional baseline models: TabPFN, EBM, SHAP-based model
    
    Args:
        baseline_models: List of baseline model names to compute. If None, compute all available.
                        Options: 'tabpfn', 'ebm', 'shap'
    
    Returns:
        Dict with baseline metrics for each model
    """
    baselines = {}
    
    # Default: compute all baselines if not specified
    if baseline_models is None:
        baseline_models = ['tabpfn', 'ebm', 'shap']
    else:
        # Normalize to lowercase
        baseline_models = [m.lower() for m in baseline_models]
    
    # TabPFN baseline (runs in subprocess to avoid segmentation faults)
    if 'tabpfn' in baseline_models and _HAS_TABPFN and task_type == "classification":
        try:
            logger.info("Computing TabPFN baseline...")
            logger.info("  ℹ️  Note: TabPFN runs in isolated subprocess to avoid crashes")
            
            import subprocess
            import tempfile
            from pathlib import Path
            import sys
            import json
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
            
            y_pred = np.array(pred_result['predictions'], dtype=int)
            
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            baselines['tabpfn'] = {
                'accuracy': float(acc),
                'f1': float(f1),
                'predictions': y_pred.tolist()
            }
            logger.info(f"  TabPFN: ACC={acc:.4f}, F1={f1:.4f}")
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
                    from interpret.glassbox import ExplainableBoostingRegressor
                    model = ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=1)
                else:
                    model = ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=1)
                
                model.fit(X_train, y_train)
            finally:
                # Restore original logging level
                interpret_logger.setLevel(original_level)
            y_pred = model.predict(X_test)
            
            if task_type == "regression":
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
            if task_type == "regression":
                from xgboost import XGBRegressor
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
                
                if task_type == "regression":
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
                
                if task_type == "regression":
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
    
    return baselines


def load_synthetic_classification_5classes(
    max_samples: int,
    n_features: int = 20,
    class_sep: float = 1.0,
    flip_y: float = 0.03,
    n_informative: Optional[int] = None,
    n_redundant: Optional[int] = None,
    n_clusters_per_class: int = 1,
):
    """
    Generate a synthetic 5-class classification dataset.
    Returns:
      X_encoded: np.ndarray [n_samples, n_features]
      y_encoded: np.ndarray [n_samples]
      X_original: List[Dict[str, float]] original feature dicts
      feature_cols: List[str] names of features
      class_names: List[str] class labels
      feature_encoders: Dict[str, Any] (empty for purely numeric features)
    """
    try:
        from sklearn.datasets import make_classification
    except Exception as e:
        raise RuntimeError("scikit-learn is required for synthetic dataset generation") from e
    n_samples = int(max_samples)
    n_classes = 5
    # Auto-derive informative/redundant if not provided
    if n_informative is None:
        n_informative = max(8, min(n_features - 2, 12))
    if n_redundant is None:
        n_redundant = max(1, min(max(0, n_features - n_informative - 1), n_features // 5))
    # Ensure parameters are compatible with sklearn's constraints
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_informative),
        n_redundant=int(n_redundant),
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=int(n_clusters_per_class),
        class_sep=float(class_sep),
        weights=None,
        flip_y=float(flip_y),
        random_state=RANDOM_STATE,
    )
    feature_cols = [f"x{i}" for i in range(X.shape[1])]
    X_original = [{feature_cols[j]: float(X[i, j]) for j in range(X.shape[1])} for i in range(X.shape[0])]
    y_encoded = y.astype(int)
    class_names = [f"Class {i}" for i in range(n_classes)]
    feature_encoders: Dict[str, Any] = {}
    X_encoded = X.astype(float)
    return X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders


def _extract_sequence_features(seq: str) -> Dict[str, float]:
    """Extract numerical features from protein sequence"""
    if _is_missing(seq) or not seq:
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
    if _is_missing(smiles) or not smiles:
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


def load_enzyme_classification_dataset(dataset_name: str, max_samples: int):
    """Load enzyme dataset from enzyme-datasets repository for classification
    
    Args:
        dataset_name: Name of the dataset (e.g., 'halogenase_NaBr_binary', 'aminotransferase_binary', 
                     'gt_donors_achiral_categorical')
        max_samples: Maximum number of samples to load
        
    Returns:
        X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    pd = _get_pandas()
    
    try:
        from enzyme_dataset_analysis import EnzymeDatasetLoader
        from sklearn.preprocessing import LabelEncoder
        
        # Initialize loader
        data_dir = os.path.join(os.path.dirname(__file__), "enzyme-datasets", "data")
        if not os.path.exists(data_dir):
            # Try alternative path
            data_dir = "enzyme-datasets/data"
        
        loader = EnzymeDatasetLoader(data_dir)
        
        # Load dataset
        df = loader.load_dataset(dataset_name)
        if df is None:
            raise FileNotFoundError(f"Could not load dataset: {dataset_name}")
        
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
        
        # Use first column as target
        target_col = target_cols[0]
        logger.info(f"Using '{target_col}' as target column")
        
        # Extract features from sequences and SMILES
        logger.info("Extracting features from sequences and SMILES...")
        feature_dicts = []
        for idx, row in df.iterrows():
            seq_features = _extract_sequence_features(row['SEQ'])
            smiles_features = _extract_smiles_features(row['SUBSTRATES'])
            combined = {**seq_features, **smiles_features}
            feature_dicts.append(combined)
        
        # Create feature DataFrame
        X_df = pd.DataFrame(feature_dicts)
        feature_cols = X_df.columns.tolist()
        
        # Extract and encode target
        y_raw = df[target_col]
        
        # Remove rows with missing values
        valid_mask = ~(X_df.isnull().any(axis=1) | y_raw.isnull())
        X_df = X_df[valid_mask].reset_index(drop=True)
        y_raw = y_raw[valid_mask].reset_index(drop=True)
        
        logger.info(f"After removing missing values: {len(X_df)} samples")
        
        # Encode target labels
        y_le = LabelEncoder()
        y_encoded = y_le.fit_transform(y_raw.astype(str))
        class_names = y_le.classes_.tolist()
        
        logger.info(f"Found {len(class_names)} classes: {class_names}")
        
        # Subsample if needed (stratified by class)
        if max_samples and len(X_df) > max_samples:
            try:
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(X_df))
                train_indices, _ = train_test_split(
                    indices, train_size=max_samples, 
                    stratify=y_encoded, random_state=RANDOM_STATE
                )
                X_df = X_df.iloc[train_indices].reset_index(drop=True)
                y_encoded = y_encoded[train_indices]
                logger.info(f"Subsampled to {len(X_df)} samples (stratified by class)")
            except Exception:
                # Fallback to random sampling if stratification fails
                indices = np.random.choice(len(X_df), max_samples, replace=False)
                X_df = X_df.iloc[indices].reset_index(drop=True)
                y_encoded = y_encoded[indices]
                logger.info(f"Subsampled to {len(X_df)} samples (random)")
        
        # Convert to numpy arrays
        X_encoded = X_df.values.astype(float)
        y_encoded = y_encoded.astype(int)
        
        # Create original feature dicts (for LLM context)
        X_original = []
        df_valid = df[valid_mask].reset_index(drop=True)
        # Get the indices that were actually used (after subsampling)
        if max_samples and len(df_valid) > max_samples:
            # If we subsampled, we need to track which indices were used
            # The X_df is already subsampled, so we need to match it with df_valid
            # For simplicity, we'll use the first len(X_df) rows of df_valid
            df_valid_subset = df_valid.iloc[:len(X_df)].reset_index(drop=True)
        else:
            df_valid_subset = df_valid
        
        for idx in range(len(X_df)):
            if idx >= len(df_valid_subset):
                break
            # Include both extracted features and original text for LLM
            feat_dict = X_df.iloc[idx].to_dict()
            feat_dict['SEQ'] = str(df_valid_subset.iloc[idx]['SEQ'])
            feat_dict['SUBSTRATES'] = str(df_valid_subset.iloc[idx]['SUBSTRATES'])
            X_original.append(feat_dict)
        
        feature_encoders: Dict[str, Any] = {}
        
        logger.info(f"Final dataset: {len(X_df)} samples, {len(feature_cols)} features, {len(class_names)} classes")
        logger.info(f"Class distribution: {Counter(y_encoded)}")
        
        return X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders
        
    except Exception as e:
        logger.error(f"Failed to load enzyme dataset '{dataset_name}': {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Enzyme dataset '{dataset_name}' not available: {e}")


def load_tabarena_classification_dataset(dataset_name: str, max_samples: int):
    """Load TabArena classification dataset from HuggingFace
    
    Args:
        dataset_name: Name of the TabArena dataset (e.g., 'adult', 'bank', 'wine', 'spaceship')
        max_samples: Maximum number of samples to load
        
    Returns:
        X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    pd = _get_pandas()
    
    try:
        from datasets import load_dataset
        from sklearn.preprocessing import LabelEncoder
        
        # TabArena dataset mapping
        TABARENA_DATASETS = {
            "adult": "scikit-learn/adult-census-income",
            "bank": "mstz/bank",
            "income": "scikit-learn/adult-census-income",
            "wine": "mstz/wine",
            "spaceship": "mstz/spaceship-titanic",
            "default": "mstz/default",
            "booking": "mstz/booking",
            "churn": "mstz/churn",
            "iris": "scikit-learn/iris",
            "breast-cancer": "scikit-learn/breast-cancer",
            "digits": "scikit-learn/digits",
            # Additional classification datasets
            "mushroom": "mstz/mushroom",
            "diabetes": "mstz/diabetes",
            "credit": "mstz/credit-card",
            "heart": "mstz/heart-disease",
            "stroke": "mstz/stroke",
            "employee": "mstz/employee-attrition",
            "telecom": "mstz/telecom-churn",
            "customer": "mstz/customer-churn",
        }
        
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower not in TABARENA_DATASETS:
            available = list(TABARENA_DATASETS.keys())
            raise ValueError(f"TabArena dataset '{dataset_name}' not found. Available: {available}")
        
        logger.info(f"Loading TabArena dataset: {dataset_name} ({TABARENA_DATASETS[dataset_name_lower]})")
        
        dataset = load_dataset(TABARENA_DATASETS[dataset_name_lower])
        
        # Convert to pandas DataFrame
        if 'train' in dataset:
            df = pd.DataFrame(dataset['train'])
        else:
            df = pd.DataFrame(dataset[list(dataset.keys())[0]])
        
        logger.info(f"Loaded TabArena dataset: {len(df)} samples, {len(df.columns)} columns")
        
        # Auto-detect target column
        possible_targets = [
            'target', 'label', 'class', 'income', 'deposit', 
            'Transported', 'Churn', 'Attrition', 'stroke', 'HeartDisease',
            'quality', 'y', 'species', 'class_label'
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
        numeric_cols = []
        
        for col in X_df.columns:
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                categorical_cols.append(col)
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                feature_encoders[col] = le
            else:
                numeric_cols.append(col)
        
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
        
        # Encode target
        y_le = LabelEncoder()
        if y_series.dtype == 'object' or y_series.dtype.name == 'category':
            y_encoded = y_le.fit_transform(y_series.astype(str))
        else:
            # Check if it's classification based on unique values
            unique_vals = len(np.unique(y_series))
            if unique_vals > 20:
                raise ValueError(f"Dataset '{dataset_name}' appears to be regression (>{unique_vals} unique values). "
                               f"Use 018_maicl_regression_biotech.py instead.")
            y_encoded = y_le.fit_transform(y_series.astype(str))
        
        class_names = y_le.classes_.tolist()
        feature_cols = X_df.columns.tolist()
        
        # Convert to numpy
        X_encoded = X_df.values.astype(float)
        y_encoded = y_encoded.astype(int)
        
        # Handle missing values
        valid_mask = ~(np.isnan(X_encoded).any(axis=1) | np.isnan(y_encoded))
        X_encoded = X_encoded[valid_mask]
        y_encoded = y_encoded[valid_mask]
        X_original = [X_original[i] for i in range(len(X_original)) if valid_mask[i]]
        
        # Subsample if needed
        if max_samples and len(X_encoded) > max_samples:
            try:
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(X_encoded))
                train_indices, _ = train_test_split(
                    indices, train_size=max_samples,
                    stratify=y_encoded, random_state=RANDOM_STATE
                )
                X_encoded = X_encoded[train_indices]
                y_encoded = y_encoded[train_indices]
                X_original = [X_original[i] for i in train_indices]
                logger.info(f"Subsampled to {len(X_encoded)} samples (stratified by class)")
            except Exception:
                indices = np.random.choice(len(X_encoded), max_samples, replace=False)
                X_encoded = X_encoded[indices]
                y_encoded = y_encoded[indices]
                X_original = [X_original[i] for i in indices]
                logger.info(f"Subsampled to {len(X_encoded)} samples (random)")
        
        logger.info(f"Final dataset: {len(X_encoded)} samples, {len(feature_cols)} features, {len(class_names)} classes")
        logger.info(f"Class distribution: {Counter(y_encoded)}")
        
        return X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders
        
    except ImportError as e:
        raise RuntimeError(f"datasets library not installed. pip install datasets\n"
                         f"Import error: {e}")
    except Exception as e:
        logger.error(f"Failed to load TabArena classification dataset '{dataset_name}': {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"TabArena dataset '{dataset_name}' not available: {e}")


def load_deepchem_classification_dataset(dataset_name: str, max_samples: int):
    """Load DeepChem classification dataset (HIV, BACE, or Tox21)
    
    Args:
        dataset_name: Name of the dataset ('hiv', 'bace', or 'tox21')
        max_samples: Maximum number of samples to load
        
    Returns:
        X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders
        
    Note:
        - ML model receives featurized vectors (ECFP fingerprints, 2048 dims)
        - LLM receives SMILES strings in X_original
    """
    # Import deepchem in function scope to avoid UnboundLocalError
    # If _HAS_DEEPCHEM is True, dc was already imported at module level, but we need it in function scope
    # If _HAS_DEEPCHEM is False, try to import it here
    if not _HAS_DEEPCHEM:
        # Try importing again to get the actual error message
        try:
            import deepchem as dc
            _ = dc.molnet
        except ImportError as e:
            error_str = str(e)
            if 'tensorflow' in error_str.lower():
                raise RuntimeError(
                    f"DeepChem requires TensorFlow, which is not installed.\n"
                    f"Please install TensorFlow: pip install tensorflow\n"
                    f"Or for CPU-only: pip install tensorflow-cpu\n"
                    f"Original error: {e}"
                )
            else:
                raise RuntimeError(
                    f"DeepChem not installed or missing dependencies.\n"
                    f"Try: pip install deepchem\n"
                    f"Import error: {e}"
                )
        except Exception as e:
            error_str = str(e)
            if 'tensorflow' in error_str.lower():
                raise RuntimeError(
                    f"DeepChem requires TensorFlow, which is not installed.\n"
                    f"Please install TensorFlow: pip install tensorflow\n"
                    f"Or for CPU-only: pip install tensorflow-cpu\n"
                    f"Original error: {e}"
                )
            else:
                error_msg = f"DeepChem import failed. This may require additional dependencies.\n"
                error_msg += f"Try: pip install deepchem tensorflow\n"
                error_msg += f"Import error: {e}"
                raise RuntimeError(error_msg)
        raise RuntimeError("DeepChem not available (unknown error)")
    else:
        # _HAS_DEEPCHEM is True, but we need dc in function scope
        # Import it here to make it available as a local variable
        import deepchem as dc
    
    try:
        from sklearn.preprocessing import LabelEncoder
        
        # Map dataset names to DeepChem loader function names
        # Access loaders lazily to avoid AttributeError for unavailable loaders
        dataset_loader_names = {
            'hiv': 'load_hiv',
            'bace': 'load_bace_c',
            'tox21': 'load_tox21',
        }
        
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower not in dataset_loader_names:
            raise ValueError(f"Unknown DeepChem classification dataset: {dataset_name}. "
                           f"Available: {list(dataset_loader_names.keys())}")
        
        # Get the loader function name and access it dynamically
        loader_name = dataset_loader_names[dataset_name_lower]
        if not hasattr(dc.molnet, loader_name):
            raise AttributeError(f"DeepChem loader '{loader_name}' not available in this version of DeepChem. "
                              f"Available loaders: {[attr for attr in dir(dc.molnet) if attr.startswith('load_')]}")
        
        loader_func = getattr(dc.molnet, loader_name)
        logger.info(f"Loading DeepChem dataset: {dataset_name_lower}")
        
        # Load dataset with ECFP featurization (2048-bit fingerprints)
        # ECFP requires RDKit - check if it's available
        try:
            import rdkit
        except ImportError:
            raise RuntimeError(
                f"RDKit is required for DeepChem molecular datasets but is not installed.\n"
                f"Please install RDKit: conda install -c conda-forge rdkit\n"
                f"Or: pip install rdkit\n"
                f"Note: pip installation may not work on all systems; conda is recommended."
            )
        
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
        
        # Handle multi-task vs single-task
        if len(tasks) > 1:
            # Multi-task (e.g., Tox21 with 12 tasks)
            # For now, use first task only (can be extended later)
            logger.info(f"  Multi-task dataset detected ({len(tasks)} tasks). Using first task: {tasks[0]}")
            all_y = all_y[:, 0:1]  # Take first task only
        
        # Flatten y to 1D for classification
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
        
        # Encode target labels
        y_le = LabelEncoder()
        y_encoded = y_le.fit_transform(all_y.astype(str))
        class_names = y_le.classes_.tolist()
        
        logger.info(f"Classes: {class_names} (n={len(class_names)})")
        logger.info(f"Class distribution: {Counter(y_encoded)}")
        
        # X_original: SMILES strings for LLM
        # DeepChem stores SMILES in dataset.ids
        X_original = []
        for smiles in all_ids:
            # SMILES string is the primary input for LLM
            # Include it as a feature dict that LLM can understand
            feat_dict = {'SMILES': str(smiles)}
            X_original.append(feat_dict)
        
        feature_encoders: Dict[str, Any] = {}
        
        logger.info(f"Final dataset: {len(X_encoded)} samples, {len(feature_cols)} features (ECFP), {len(class_names)} classes")
        logger.info(f"  ML model will use: {X_encoded.shape} featurized vectors")
        logger.info(f"  LLM will use: SMILES strings from dataset")
        
        return X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders
        
    except ImportError as e:
        error_str = str(e)
        if 'rdkit' in error_str.lower() or 'RDKit' in error_str:
            logger.error(f"RDKit is required for DeepChem molecular datasets but is not installed.")
            raise RuntimeError(
                f"RDKit is required for DeepChem dataset '{dataset_name}' but is not installed.\n"
                f"Please install RDKit: conda install -c conda-forge rdkit\n"
                f"Or: pip install rdkit\n"
                f"Note: pip installation may not work on all systems; conda is recommended.\n"
                f"Original error: {e}"
            )
        else:
            logger.error(f"Failed to load DeepChem classification dataset '{dataset_name}': {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DeepChem dataset '{dataset_name}' not available: {e}")
    except Exception as e:
        error_str = str(e)
        if 'rdkit' in error_str.lower() or 'RDKit' in error_str:
            logger.error(f"RDKit is required for DeepChem molecular datasets but is not installed.")
            raise RuntimeError(
                f"RDKit is required for DeepChem dataset '{dataset_name}' but is not installed.\n"
                f"Please install RDKit: conda install -c conda-forge rdkit\n"
                f"Or: pip install rdkit\n"
                f"Note: pip installation may not work on all systems; conda is recommended.\n"
                f"Original error: {e}"
            )
        else:
            logger.error(f"Failed to load DeepChem classification dataset '{dataset_name}': {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DeepChem dataset '{dataset_name}' not available: {e}")


def run_fold_pipeline_classification(
    X_train, X_val, X_test, y_train, y_val, y_test,
    X_original_train, X_original_val, X_original_test,
    feature_cols, class_names, feature_encoders,
    args, llm, ds_name, ds_label, fold_output_dir, fold_num
):
    """
    Run the complete training and evaluation pipeline for a single fold.
    Returns a dictionary with all metrics and results.
    
    This function contains the core training/evaluation logic extracted from main().
    It's called once per fold during cross-validation.
    """
    import json
    # Import here to avoid circular dependencies
    from sklearn.metrics import accuracy_score, f1_score
    from collections import Counter
    
    logger.info(f"Running fold {fold_num} pipeline...")
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Set output directory for this fold (temporarily override global OUTPUT_DIR)
    original_output_dir = OUTPUT_DIR
    set_output_dir(fold_output_dir)
    
    try:
        # Conditionally scale features based on --no_scaling flag
        if not args.no_scaling:
            scaler = MinMaxScaler010()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            validate_scaled_data(X_train_s, feature_cols, scaler=scaler)
            logger.info(f"Features scaled to [{SCALE_MIN}, {SCALE_MAX}] range")
        else:
            scaler = None
            X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
        
        mech_map = {"logreg": "LogisticRegression", "xgboost": "XGBoost", "tabicl": "TabICL", "ebm": "EBM", "tabpfn": "TabPFN"}
        model_name = mech_map.get(args.ml_mech.lower(), "LogisticRegression")
        
        pretrained_ml = MLModelMechanism(model_name, task_type="classification")
        pretrained_ml.train(X_train_s, y_train, feature_cols, y_scaler=None)
        pretrained_ml._maicl_feature_cols = feature_cols
        pretrained_ml._maicl_feature_encoders = feature_encoders
        pretrained_ml._maicl_scaler = scaler
        
        sorted_idx, residuals, ml_preds, ml_proba = compute_ml_residuals(
            pretrained_ml, X_train_s, y_train, feature_cols, class_names, task_type="classification"
        )
        
        def _select_topk_residual_balanced_classification(X_tr_s, y_tr_s, residual_vec, k, class_names_list=None, ml_predictions=None):
            """Select top-K residuals with balanced class coverage for classification."""
            if k <= 0 or k >= len(y_tr_s):
                return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
            unique_classes = np.unique(y_tr_s)
            if len(unique_classes) < 2:
                if len(unique_classes) == 1:
                    idxs = np.argsort(residual_vec)[::-1][:min(k, len(y_tr_s))]
                    return X_tr_s[idxs], y_tr_s[idxs], idxs
                return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
            
            idxs = []
            class_counts = {c: np.sum(y_tr_s == c) for c in unique_classes}
            total_samples = len(y_tr_s)
            n_classes = len(unique_classes)
            min_per_class = 1 if k >= n_classes else 0
            proportions = {c: count / total_samples for c, count in class_counts.items()}
            allocated = {c: max(min_per_class, int(round(k * proportions[c]))) for c in unique_classes}
            
            total_alloc = sum(allocated.values())
            while total_alloc > k:
                cmax = max(allocated, key=lambda c: allocated[c])
                if allocated[cmax] > min_per_class:
                    allocated[cmax] -= 1
                    total_alloc -= 1
                else:
                    break
            while total_alloc < k and total_alloc < total_samples:
                cmin = min(allocated, key=lambda c: allocated[c])
                allocated[cmin] += 1
                total_alloc += 1
            
            for c in unique_classes:
                class_idx = np.where(y_tr_s == c)[0]
                if len(class_idx) == 0:
                    continue
                take = min(allocated[c], len(class_idx))
                top_in_class = class_idx[np.argsort(residual_vec[class_idx])[::-1][:take]]
                idxs.extend(top_in_class.tolist())
            idxs = np.array(sorted(set(idxs)))
            return X_tr_s[idxs], y_tr_s[idxs], idxs
        
        # Select top-K samples
        if args.top_k == -1:
            X_topk, y_topk, top_indices = X_train_s, y_train, np.arange(len(X_train_s))
            X_original_topk = X_original_train
        else:
            if args.topk_strategy == "residual_balanced":
                X_topk, y_topk, top_indices = _select_topk_residual_balanced_classification(
                    X_train_s, y_train, residuals, args.top_k, class_names_list=class_names, ml_predictions=ml_preds
                )
            elif args.topk_strategy == "random":
                X_topk, y_topk, top_indices = _select_topk_random_classification(
                    X_train_s, y_train, residuals, args.top_k
                )
            else:
                X_topk, y_topk, top_indices = get_top_k_residual_samples(
                    sorted_idx, residuals, X_train_s, y_train, args.top_k,
                    ml_predictions=ml_preds, task_type="classification"
                )
            if len(top_indices) == 0:
                X_topk, y_topk, top_indices = X_train_s, y_train, np.arange(len(X_train_s))
                X_original_topk = X_original_train
            else:
                X_original_topk = [X_original_train[i] for i in top_indices]
        
        # Compute residuals on top-K subset
        if args.top_k != -1:
            sorted_idx, residuals_topk, ml_preds, ml_proba = compute_ml_residuals(
                pretrained_ml, X_topk, y_topk, feature_cols, class_names, task_type="classification"
            )
        else:
            residuals_topk = residuals
        
        # Compute ML baseline on test set
        ml_baseline_metrics = {}
        try:
            is_multiclass = class_names is not None and len(class_names) > 2
            model = pretrained_ml.model
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
                        y_pred_cls = np.array(pred_result['predictions'], dtype=int)
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
                y_pred_cls = model.predict(X_test_df).astype(int)
            else:
                if is_multiclass:
                    y_pred_cls = model.predict(X_test_s).astype(int)
                else:
                    y_pred_cls = None
            
            if is_multiclass:
                acc = accuracy_score(y_test, y_pred_cls)
                f1 = f1_score(y_test, y_pred_cls, average='weighted', zero_division=0)
                ml_baseline_metrics = {"accuracy": acc, "f1": f1, "predictions": y_pred_cls.tolist()}
            else:
                if model_name in ["TabPFN", "TabICL"]:
                    if hasattr(model, "predict_proba") and model_name != "TabPFN":
                        y_prob = model.predict_proba(X_test_df)[:, 1]
                    else:
                        y_pred_binary = y_pred_cls.astype(float)
                        y_prob = np.where(y_pred_binary == 1, 0.7, 0.3)
                else:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test_s)[:, 1]
                    else:
                        y_prob = model.predict(X_test_s).astype(float)
                y_prob = np.clip(y_prob, 0.0, 1.0)
                y_pred_05 = (y_prob >= 0.5).astype(int)
                acc_05 = accuracy_score(y_test, y_pred_05)
                f1_05 = f1_score(y_test, y_pred_05, zero_division=0)
                thr_opt = find_optimal_threshold(y_test, y_prob, metric='f1')
                y_pred_opt = (y_prob >= thr_opt).astype(int)
                acc_opt = accuracy_score(y_test, y_pred_opt)
                f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
                ml_baseline_metrics = {
                    "accuracy": acc_opt, "f1": f1_opt, "threshold_opt": float(thr_opt),
                    "acc@0.5": acc_05, "f1@0.5": f1_05, "predictions": y_pred_opt.tolist()
                }
        except Exception as e:
            logger.warning(f"Failed to compute ML baseline metrics: {e}")
        
        # Create and train MA-ICL
        maicl = TrainableMAICL(
            llm, feature_cols, scaler,
            use_ml_mechanism=bool(args.use_ml),
            dataset_name=ds_name,
            y_scaler=None,
            pretrained_ml_mechanism=pretrained_ml,
            data_insights=None,
            task_type="classification",
            class_names=class_names,
            classification_loss_metric=args.classification_loss,
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
            X_test_s, y_test, X_train_s, y_train,
            return_details=True, relax_routing=bool(args.relax_eval),
            X_original=X_original_test, X_pool_original=X_original_train,
        )
        pre_acc = float(pre.get('accuracy', 0.0))
        pre_f1 = float(pre.get('f1', 0.0))
        pre_loss = float(pre.get('loss', 1.0))
        
        # Train MA-ICL
        maicl.train(
            X_topk, y_topk, X_val_s, y_val,
            iterations=args.iterations,
            ml_residuals=residuals_topk,
            accept_eval_max=None,
            X_test=X_test_s, y_test=y_test,
            acceptance_set=args.acceptance_set,
            X_train_original=X_original_topk,
            X_val_original=X_original_val,
            X_test_original=X_original_test,
            output_dir=fold_output_dir,
        )
        
        # Post-training evaluation
        post = maicl.evaluate(
            X_test_s, y_test, X_train_s, y_train,
            return_details=True, relax_routing=True,
            preserve_mechanism_performance=True,
            X_original=X_original_test, X_pool_original=X_original_train,
        )
        post_acc = float(post.get('accuracy', 0.0))
        post_f1 = float(post.get('f1', 0.0))
        post_loss = float(post.get('loss', 1.0))
        
        # LLM-only evaluation
        llm_only_metrics = None
        try:
            llm_only_metrics = maicl.evaluate_llm_only(
                X_test_s, y_test, X_train_s, y_train,
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
            'ml_baseline_accuracy': ml_baseline_metrics.get('accuracy', 0.0),
            'ml_baseline_f1': ml_baseline_metrics.get('f1', 0.0),
            'pre_accuracy': pre_acc,
            'pre_f1': pre_f1,
            'pre_loss': pre_loss,
            'post_accuracy': post_acc,
            'post_f1': post_f1,
            'post_loss': post_loss,
        }
        
        if llm_only_metrics:
            fold_result['llm_only_pre_accuracy'] = float(llm_only_metrics.get('accuracy', 0.0))
            fold_result['llm_only_pre_f1'] = float(llm_only_metrics.get('f1', 0.0))
            fold_result['llm_only_post_accuracy'] = float(llm_only_metrics.get('accuracy', 0.0))
            fold_result['llm_only_post_f1'] = float(llm_only_metrics.get('f1', 0.0))
        
        # Save fold-specific results file
        fold_results_file = os.path.join(fold_output_dir, "fold_results.json")
        with open(fold_results_file, 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        logger.info(f"Fold {fold_num} complete: ACC={post_acc:.4f}, F1={post_f1:.4f}")
        
        return fold_result
        
    finally:
        # Restore original output directory
        set_output_dir(original_output_dir)


def save_cv_summary_classification(cv_results, output_dir, args):
    """
    Aggregate cross-validation results and save summary statistics.
    """
    import json
    import numpy as np
    
    # Aggregate metrics across folds
    metrics_to_aggregate = [
        'ml_baseline_accuracy', 'ml_baseline_f1',
        'pre_accuracy', 'pre_f1', 'pre_loss',
        'post_accuracy', 'post_f1', 'post_loss',
        'llm_only_pre_accuracy', 'llm_only_pre_f1',
        'llm_only_post_accuracy', 'llm_only_post_f1'
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
    import json
    
    parser = argparse.ArgumentParser(description="MA-ICL classification runner with top-K residuals")
    parser.add_argument("--dataset", default="zoo", 
                        help="Classification dataset name. Options:\n"
                             "  OpenML: car (4), iris (3), wine (3), zoo (7), glass (6), vehicle (4), soybean (19), lymphography (4), ecoli (8), adult (2), credit (2), vote (2), mushroom (2)\n"
                             "  OpenML (many classes): letter (26), mfeat (10), pendigits (10), optdigits (10), avila (12), amazon (50), plant-margin (100), plant-shape (100), plant-texture (100)\n"
                             "  OpenML (gene expression cancer):\n"
                             "    - leukemia (2) - Leukemia classification (AML vs ALL), 72 samples, 7129 genes\n"
                             "    - colon-cancer (2) or colon (2) - Colon cancer classification (tumor vs normal), 62 samples, 2000 genes\n"
                             "  OpenML (biological/genomic):\n"
                             "    - mice-protein (8) or miceprotein (8) - Mice Protein Expression, 1080 samples, 77 protein expression levels\n"
                             "    - splice (3) - Splice (Gene Sequences), 3175 samples, 60 DNA sequence positions\n"
                             "  Synthetic: synthetic5 (5)\n"
                             "  TabArena (HuggingFace):\n"
                             "    - adult (2), bank (2), income (2), wine (3), spaceship (2), default (2), booking (3), churn (2)\n"
                             "    - iris (3), breast-cancer (2), digits (10)\n"
                             "    - mushroom (2), diabetes (2), credit (2), heart (2), stroke (2), employee (2), telecom (2), customer (2)\n"
                             "  DeepChem (molecular classification):\n"
                             "    - hiv (HIV protease inhibition, binary)\n"
                             "    - bace (BACE protein binding, binary)\n"
                             "    - tox21 (12-task toxicity prediction, multi-label)\n"
                             "  Enzyme Binary (exact names from enzyme-datasets table):\n"
                             "    - aminotransferase_binary\n"
                             "    - olea_binary\n"
                             "    - gt_donors_achiral_binary\n"
                             "    - gt_donors_chiral_binary\n"
                             "    - nitrilase_binary\n"
                             "    - halogenase_NaBr_binary\n"
                             "    - halogenase_NaCl_binary\n"
                             "    - duf_binary\n"
                             "    - gt_acceptors_achiral_binary\n"
                             "    - gt_acceptors_chiral_binary\n"
                             "    - esterase_binary\n"
                             "    - phosphatase_achiral_binary\n"
                             "    - phosphatase_chiral_binary\n"
                             "  Enzyme Categorical (exact names from enzyme-datasets table):\n"
                             "    - gt_donors_achiral_categorical\n"
                             "    - gt_donors_chiral_categorical\n"
                             "    - gt_acceptors_achiral_categorical\n"
                             "    - gt_acceptors_chiral_categorical")
    parser.add_argument(
        "--model_name",
        default=os.environ.get("MAICL_MODEL_NAME", "gemini-2.0-flash"),
        help="LLM model name. Examples: Gemini: gemini-2.0-flash, gemini-2.0-pro. OpenAI: gpt-4o-mini, gpt-4o, gpt-4.1-mini."
    )
    parser.add_argument("--ml_mech", default="ebm", help="ML mechanism: logreg|xgboost|tabicl|ebm|tabpfn. "
                        "Note: TabPFN may cause segmentation faults when used as ML mechanism. "
                        "Consider using TabPFN only as a baseline (it runs in isolated subprocess) "
                        "or use --ml_mech logreg/xgboost for more stable ML mechanism.")
    parser.add_argument("--baseline_models", type=str, nargs='+', default=None,
                        help="Select which baseline models to compute. Options: tabpfn, ebm, shap. "
                             "If not specified, all available baselines will be computed. "
                             "Example: --baseline_models tabpfn ebm")
    parser.add_argument("--use_ml", type=int, default=1, choices=[0,1], help="Include ML mechanism in ensemble")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=100, help="-1 to use full dataset")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--acceptance_set", type=str, default="validation",
                        choices=["test", "validation", "train"],
                        help="Dataset to use for acceptance evaluation during training. "
                             "Options: 'test' (risks overfitting to test), 'validation' (default), "
                             "or 'train' (may overfit to training data).")
    parser.add_argument("--topk_strategy", choices=["residual", "residual_balanced", "random"], default="residual_balanced",
                        help="Top-K selection: 'residual' = global highest | 'residual_balanced' = highest within class bins (default for classification) | 'random' = random selection (no sorting)")
    parser.add_argument("--relax_eval", type=int, default=0, choices=[0,1],
                        help="Relax ML routing during evaluation to let LLM contribute (default: 1, recommended for classification)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Validation fraction of the training split (0-1). Default: 0.2")
    parser.add_argument("--classification_loss", type=str, default="f1", choices=["f1", "acc", "accuracy"],
                        help="Classification loss metric: 'f1' (F1 score) or 'acc'/'accuracy' (accuracy). Default: f1")
    parser.add_argument("--evaluate_individual_mechanisms", action="store_true",
                        help="After training, evaluate each mechanism individually on train/test sets")
    parser.add_argument("--num_mechanisms_unknown", type=int, default=1,
                        help="Number of unknown mechanisms to generate (default: 1). "
                             "This controls how many LLM-based mechanisms are created to complement the ML mechanism.")
    parser.add_argument("--no_scaling", action="store_true",
                        help="If set, disables all feature scaling. Data will be used in its original range.")
    parser.add_argument("--cv_folds", type=int, default=0,
                        help="Number of cross-validation folds (default: 0 = disabled). "
                             "When > 0, performs k-fold cross-validation and saves results for each fold. "
                             "Example: --cv_folds 5 for 5-fold CV.")
    args = parser.parse_args()

    # Note: Classification loss metric is now configurable via --classification_loss argument
    # Default is F1, but can be set to accuracy if desired

    # Resolve dataset
    ds_name = str(args.dataset).lower()
    
    # Build comprehensive run name with all important arguments
    # Start with base components
    run_name_parts = [
        "class",
        ds_name,
        f"ml{args.ml_mech}" if args.use_ml else "noML",
        f"topk{args.top_k}",
        f"iter{args.iterations}",
        f"model{args.model_name.replace('-', '_').replace('.', '_')}",  # Sanitize model name for filesystem
        f"loss{args.classification_loss}",
        f"accept{args.acceptance_set}",
        f"topkstrat{args.topk_strategy}",
    ]
    
    # Add optional flags and non-default values
    if args.relax_eval:
        run_name_parts.append("relax")
    if args.num_mechanisms_unknown is not None and args.num_mechanisms_unknown > 1:
        run_name_parts.append(f"nmech{args.num_mechanisms_unknown}")
    if args.no_scaling:
        run_name_parts.append("noscale")
    if args.val_size != 0.2:  # Only include if non-default
        run_name_parts.append(f"val{args.val_size:.2f}".replace('.', '_'))
    if args.max_samples != 200:  # Only include if non-default
        run_name_parts.append(f"maxsamp{args.max_samples}")
    if args.evaluate_individual_mechanisms:
        run_name_parts.append("evalindiv")
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

    llm = _get_llm(args.model_name)

    if ds_name in ("synthetic5", "syn5", "toy5"):
        X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_synthetic_classification_5classes(
            args.max_samples,
            n_features=20,
            class_sep=1.0,
            flip_y=0.03,
            n_informative=None,
            n_redundant=None,
            n_clusters_per_class=1,
        )
        ds_label = f"Synthetic 5-Class ({args.dataset})"
    elif ds_name in ("adult", "bank", "income", "wine", "spaceship", "default", "booking", "churn", 
                     "iris", "breast-cancer", "digits", "mushroom", "diabetes", "credit", "heart", 
                     "stroke", "employee", "telecom", "customer") or ds_name.startswith("tabarena_"):
        # TabArena datasets from HuggingFace
        try:
            # Remove tabarena_ prefix if present
            tabarena_name = ds_name.replace("tabarena_", "")
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_tabarena_classification_dataset(
                tabarena_name, args.max_samples
            )
            ds_label = f"TabArena {tabarena_name}"
        except Exception as e:
            logger.warning(f"Failed to load as TabArena dataset: {e}")
            # Fallback to OpenML
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_openml_classification(
                args.dataset, args.max_samples
            )
            ds_label = f"OpenML {args.dataset}"
    elif ds_name in ("hiv", "bace", "tox21"):
        # DeepChem molecular classification datasets
        X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_deepchem_classification_dataset(
            args.dataset, args.max_samples
        )
        ds_label = f"DeepChem {args.dataset.upper()}"
    elif ds_name.endswith("_binary") or ds_name.endswith("_categorical") or any(
        ds_name.startswith(prefix) for prefix in ["halogenase", "aminotransferase", "olea", "nitrilase", 
                                                    "phosphatase", "gt_", "duf", "esterase", "davis"]
    ):
        # Try loading as enzyme dataset
        try:
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_enzyme_classification_dataset(
                args.dataset, args.max_samples
            )
            ds_label = f"Enzyme Dataset: {args.dataset}"
        except Exception as e:
            logger.warning(f"Failed to load as enzyme dataset: {e}")
            # Fallback to OpenML
            X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_openml_classification(
                args.dataset, args.max_samples
            )
            ds_label = f"OpenML {args.dataset}"
    else:
        X_encoded, y_encoded, X_original, feature_cols, class_names, feature_encoders = load_openml_classification(
            args.dataset, args.max_samples
        )
        ds_label = f"OpenML {args.dataset}"

    from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
    
    # Check if cross-validation is enabled
    cv_folds = args.cv_folds if args.cv_folds > 0 else 0
    
    if cv_folds > 0:
        logger.info("=" * 80)
        logger.info(f"CROSS-VALIDATION MODE: {cv_folds}-fold CV")
        logger.info("=" * 80)
        
        # Prepare for cross-validation
        # Use StratifiedKFold for classification to maintain class distribution
        class_counts = Counter(y_encoded)
        if min(class_counts.values()) >= cv_folds:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(skf.split(X_encoded, y_encoded))
        else:
            logger.warning(f"Not enough samples per class for stratified CV. Using regular KFold.")
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(kf.split(X_encoded))
        
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
            X_train_val = X_encoded[train_val_idx]
            y_train_val = y_encoded[train_val_idx]
            X_original_train_val = [X_original[i] for i in train_val_idx]
            X_test = X_encoded[test_idx]
            y_test = y_encoded[test_idx]
            X_original_test = [X_original[i] for i in test_idx]
            
            # Further split train_val into train and validation
            idx_train_val = np.arange(len(X_train_val))
            class_counts_tr = Counter(y_train_val)
            stratify_tr = y_train_val if min(class_counts_tr.values()) >= 2 else None
            can_stratify = (len(idx_train_val) > 0 and 
                           len(np.unique(y_train_val)) > 1 and 
                           min(class_counts_tr.values()) >= 2)
            stratify_tr = y_train_val if can_stratify else None
            idx_tr, idx_va = train_test_split(idx_train_val, test_size=float(args.val_size), random_state=RANDOM_STATE, stratify=stratify_tr)
            
            X_train = X_train_val[idx_tr]
            X_val = X_train_val[idx_va]
            y_train = y_train_val[idx_tr]
            y_val = y_train_val[idx_va]
            X_original_train = [X_original_train_val[i] for i in idx_tr]
            X_original_val = [X_original_train_val[i] for i in idx_va]
            
            # Run the pipeline for this fold (will be implemented below)
            fold_result = run_fold_pipeline_classification(
                X_train, X_val, X_test, y_train, y_val, y_test,
                X_original_train, X_original_val, X_original_test,
                feature_cols, class_names, feature_encoders,
                args, llm, ds_name, ds_label, fold_output_dir, fold_idx + 1
            )
            cv_results.append(fold_result)
        
        # Aggregate and save CV summary
        save_cv_summary_classification(cv_results, output_dir, args)
        logger.info("=" * 80)
        logger.info("CROSS-VALIDATION COMPLETE")
        logger.info("=" * 80)
        return
    
    # Standard single run (no CV)
    class_counts = Counter(y_encoded)
    stratify = y_encoded if min(class_counts.values()) >= 2 else None
    idx = np.arange(len(X_encoded))
    idx_tr_full, idx_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)
    # Split training further into train/validation (stratified if possible)
    # Check if stratification is possible: need at least 2 samples per class in training set
    y_tr_full = y_encoded[idx_tr_full]
    tr_class_counts = Counter(y_tr_full)
    can_stratify = (idx_tr_full.size > 0 and 
                   len(np.unique(y_tr_full)) > 1 and 
                   min(tr_class_counts.values()) >= 2)
    stratify_tr = y_tr_full if can_stratify else None
    idx_tr, idx_va = train_test_split(idx_tr_full, test_size=float(args.val_size), random_state=RANDOM_STATE, stratify=stratify_tr)
    X_train, X_val, X_test = X_encoded[idx_tr], X_encoded[idx_va], X_encoded[idx_te]
    y_train, y_val, y_test = y_encoded[idx_tr], y_encoded[idx_va], y_encoded[idx_te]
    # Split X_original (SMILES strings for LLM) using same indices
    X_original_train = [X_original[i] for i in idx_tr]
    X_original_val = [X_original[i] for i in idx_va]
    X_original_test = [X_original[i] for i in idx_te]

    # Conditionally scale features based on --no_scaling flag
    if not args.no_scaling:
        # Scale features to [0,1]
        scaler = MinMaxScaler010()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        validate_scaled_data(X_train_s, feature_cols, scaler=scaler)
        logger.info(f"Features scaled to [{SCALE_MIN}, {SCALE_MAX}] range (X min={X_train_s.min():.3f}, max={X_train_s.max():.3f})")
    else:
        logger.info("Scaling disabled: Using original feature values.")
        scaler = None
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    mech_map = {"logreg": "LogisticRegression", "xgboost": "XGBoost", "tabicl": "TabICL", "ebm": "EBM", "tabpfn": "TabPFN"}
    model_name = mech_map.get(args.ml_mech.lower(), "LogisticRegression")

    pretrained_ml = MLModelMechanism(model_name, task_type="classification")
    pretrained_ml.train(X_train_s, y_train, feature_cols, y_scaler=None)
    pretrained_ml._maicl_feature_cols = feature_cols
    pretrained_ml._maicl_feature_encoders = feature_encoders
    pretrained_ml._maicl_scaler = scaler

    sorted_idx, residuals, ml_preds, ml_proba = compute_ml_residuals(
        pretrained_ml, X_train_s, y_train, feature_cols, class_names, task_type="classification"
    )

    def _select_topk_residual_balanced_classification(X_tr_s, y_tr_s, residual_vec, k, class_names_list=None, ml_predictions=None):
        """Select top-K residuals with balanced class coverage for classification.
        Residuals are 1.0 for mismatched, 0.0 for matched. Sorting by residual (descending)
        naturally puts mismatched examples first. No filtering needed - just sort and pick top K."""
        if k <= 0 or k >= len(y_tr_s):
            return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
        
        # Residuals are already 1.0 (mismatched) or 0.0 (matched) - no filtering needed
        # Get unique classes
        unique_classes = np.unique(y_tr_s)
        
        # Ensure we have at least 2 classes
        if len(unique_classes) < 2:
            # If only one class in entire dataset, return what we can
            if len(unique_classes) == 1:
                # Sort by residual descending (mismatched first)
                idxs = np.argsort(residual_vec)[::-1][:min(k, len(y_tr_s))]
                return X_tr_s[idxs], y_tr_s[idxs], idxs
            # Should not happen, but handle gracefully
            return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
        
        idxs = []
        # Allocate k proportionally to class sizes, minimum 1 per class to ensure diversity
        class_counts = {c: np.sum(y_tr_s == c) for c in unique_classes}
        total_samples = len(y_tr_s)
        n_classes = len(unique_classes)
        
        # Minimum allocation: at least 1 per class (if k >= n_classes)
        min_per_class = 1 if k >= n_classes else 0
        proportions = {c: count / total_samples for c, count in class_counts.items()}
        
        # Allocate: ensure minimum per class, then distribute remainder proportionally
        allocated = {c: max(min_per_class, int(round(k * proportions[c]))) for c in unique_classes}
        
        # Adjust allocation to exactly k
        total_alloc = sum(allocated.values())
        # Trim or add to match k
        while total_alloc > k:
            cmax = max(allocated, key=lambda c: allocated[c])
            if allocated[cmax] > min_per_class:
                allocated[cmax] -= 1
                total_alloc -= 1
            else:
                break
        while total_alloc < k and total_alloc < total_samples:
            cmin = min(allocated, key=lambda c: allocated[c])
            allocated[cmin] += 1
            total_alloc += 1
        
        # Pick within each class: sort by residual descending (mismatched=1.0 come first)
        for c in unique_classes:
            class_idx = np.where(y_tr_s == c)[0]
            if len(class_idx) == 0:
                continue
            take = min(allocated[c], len(class_idx))
            # Sort by residual descending - mismatched (1.0) come before matched (0.0)
            top_in_class = class_idx[np.argsort(residual_vec[class_idx])[::-1][:take]]
            idxs.extend(top_in_class.tolist())
        idxs = np.array(sorted(set(idxs)))
        return X_tr_s[idxs], y_tr_s[idxs], idxs

    def _select_topk_random_classification(X_tr_s, y_tr_s, residual_vec, k):
        """Select top-K samples randomly from residuals (no sorting)."""
        if k <= 0 or k >= len(y_tr_s):
            return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
        rng = np.random.RandomState(RANDOM_STATE)
        idxs = rng.choice(len(y_tr_s), size=k, replace=False)
        idxs = np.array(sorted(idxs))
        return X_tr_s[idxs], y_tr_s[idxs], idxs

    # Select training subset for MA-ICL (top-K by residual magnitude)
    if args.top_k == -1:
        X_topk, y_topk, top_indices = X_train_s, y_train, np.arange(len(X_train_s))
        X_original_topk = X_original_train
    else:
        if args.topk_strategy == "residual_balanced":
            logger.info(f"Selecting top-K residuals with balanced class coverage (K={args.top_k}), prioritizing mismatched examples")
            X_topk, y_topk, top_indices = _select_topk_residual_balanced_classification(
                X_train_s, y_train, residuals, args.top_k, class_names_list=class_names, ml_predictions=ml_preds
            )
        elif args.topk_strategy == "random":
            logger.info(f"Selecting top-K residuals randomly (no sorting, K={args.top_k})")
            X_topk, y_topk, top_indices = _select_topk_random_classification(
                X_train_s, y_train, residuals, args.top_k
            )
        else:
            # Residuals are 1.0 (mismatched) or 0.0 (matched) - sorting by residual descending
            # naturally puts mismatched examples first - no strategy parameter needed
            X_topk, y_topk, top_indices = get_top_k_residual_samples(
                sorted_idx, residuals, X_train_s, y_train, args.top_k,
                ml_predictions=ml_preds, task_type="classification"
            )
        if len(top_indices) == 0:
            logger.warning("No top-K samples found (perfect ML?). Falling back to full dataset.")
            X_topk, y_topk, top_indices = X_train_s, y_train, np.arange(len(X_train_s))
            X_original_topk = X_original_train
        else:
            # Extract corresponding X_original samples (SMILES strings for LLM)
            X_original_topk = [X_original_train[i] for i in top_indices]
    logger.info(f"Top-K selection: strategy={args.topk_strategy}, selected={len(top_indices)} examples.")
    
    # Validate that selected samples have at least 2 classes (required for classification)
    unique_classes_in_topk = np.unique(y_topk)
    if len(unique_classes_in_topk) < 2:
        logger.warning(f"Top-K selection resulted in only {len(unique_classes_in_topk)} class(es): {unique_classes_in_topk}")
        if args.topk_strategy != "residual_balanced":
            logger.warning("Switching to balanced class selection strategy to ensure class diversity...")
            X_topk, y_topk, top_indices = _select_topk_residual_balanced_classification(
                X_train_s, y_train, residuals, args.top_k, class_names_list=class_names, ml_predictions=ml_preds
            )
            unique_classes_in_topk = np.unique(y_topk)
            logger.info(f"After balanced selection: {len(top_indices)} samples with {len(unique_classes_in_topk)} classes")
        
        # If still only one class, fall back to full dataset or minimum required samples
        if len(unique_classes_in_topk) < 2:
            logger.warning("Balanced selection still resulted in single class. Falling back to full training set.")
            X_topk, y_topk, top_indices = X_train_s, y_train, np.arange(len(X_train_s))
            X_original_topk = X_original_train
            logger.info(f"Using full training set: {len(X_topk)} samples")
    
    # ML model is FROZEN after initial training on full training set
    # Compute residuals on the top-K subset using the frozen model (for MA-ICL training)
    if args.top_k != -1:
        # ML model stays frozen - trained once on full training set and never retrained
        logger.info(f"[Data Alignment] ML model is FROZEN (trained on full training set: {len(X_train_s)} samples)")
        logger.info(f"[Data Alignment] MA-ICL will train on {len(X_topk)} top-K samples, but ML model remains frozen")
        
        # Validate that selected subset has at least 2 classes (for classification)
        unique_classes_final = np.unique(y_topk)
        if len(unique_classes_final) < 2:
            raise ValueError(
                f"Selected {len(X_topk)} samples contain only {len(unique_classes_final)} class(es): {unique_classes_final}. "
                f"Need at least 2 classes for classification. Try: (1) increasing --top_k, (2) using --topk_strategy residual_balanced, "
                f"or (3) using --top_k -1 to use full training set."
            )
        logger.info(f"[Data Alignment] Classes in selected set: {len(unique_classes_final)} classes: {unique_classes_final}")
        
        # Compute residuals on the subset using the frozen model (trained on full set)
        sorted_idx, residuals_topk, ml_preds, ml_proba = compute_ml_residuals(
            pretrained_ml, X_topk, y_topk, feature_cols, class_names, task_type="classification"
        )
        logger.info(f"[Data Alignment] ✓ Computed residuals on top-K subset using frozen ML model (trained on full set)")
        logger.info(f"[Data Alignment] Training data: ML=frozen on {len(X_train_s)} samples, MA-ICL={len(X_topk)} samples")
    else:
        # Both train on full set, use original residuals
        residuals_topk = residuals
        logger.info(f"[Data Alignment] Using full training set ({len(X_topk)} samples) - residuals already computed")
        logger.info(f"[Data Alignment] Training data: ML=frozen on {len(X_train_s)} samples, MA-ICL={len(X_topk)} samples")
    
    # Compute ML baseline metrics on TRAINING set (for comparison with MA-ICL training performance)
    # WARNING: This evaluates on the same data the model was trained on, so high accuracy (often 1.0) 
    # is expected due to overfitting. The test set performance is the real metric.
    ml_baseline_train_metrics: Dict[str, Any] = {}
    try:
        is_multiclass = class_names is not None and len(class_names) > 2
        model = pretrained_ml.model
        # TabPFN and TabICL require DataFrames, not numpy arrays
        if model_name == "TabPFN":
            # TabPFN uses subprocess isolation - model is None, use batch prediction helper
            import subprocess
            import tempfile
            from pathlib import Path
            import sys
            import json
            import pandas as pd
            
            temp_dir = Path(tempfile.mkdtemp(prefix="tabpfn_ml_baseline_train_"))
            X_topk_df = pd.DataFrame(X_topk, columns=feature_cols)
            X_topk_path = temp_dir / "X_topk.csv"
            output_path = temp_dir / "pred_result_train.json"
            
            X_topk_df.to_csv(X_topk_path, index=False)
            
            script_path = Path(__file__).parent / "run_tabpfn_ml_mechanism.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "predict",
                 pretrained_ml._tabpfn_training_data_path['X_train_path'],
                 pretrained_ml._tabpfn_training_data_path['y_train_path'],
                 str(X_topk_path),
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
                    y_pred_train = np.array(pred_result['predictions'], dtype=int)
                else:
                    raise RuntimeError(f"TabPFN training set prediction failed: {pred_result.get('error', 'Unknown error')}")
            else:
                raise RuntimeError("TabPFN training set prediction subprocess failed")
            
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
        elif model_name == "TabICL":
            import pandas as pd
            X_topk_df = pd.DataFrame(X_topk, columns=feature_cols)
            if is_multiclass:
                y_pred_train = model.predict(X_topk_df).astype(int)
            else:
                y_pred_train = model.predict(X_topk_df).astype(int)
        else:
            # Other models can use numpy arrays directly
            if is_multiclass:
                y_pred_train = model.predict(X_topk).astype(int)
            else:
                y_pred_train = None  # Will be computed below for binary
        
        if is_multiclass:
            acc_train = accuracy_score(y_topk, y_pred_train)
            f1_train = f1_score(y_topk, y_pred_train, average='weighted', zero_division=0)
            ml_baseline_train_metrics = {
                "accuracy": acc_train, "f1": f1_train,
                "predictions": y_pred_train.tolist()
            }
            logger.info(f"ML baseline on TRAINING set ({len(X_topk)} samples): ACC={acc_train:.4f} F1={f1_train:.4f}")
            if acc_train >= 0.99:
                logger.warning(f"  ⚠️  Training accuracy is {acc_train:.4f} - this indicates overfitting. "
                              f"The model is being evaluated on the same data it was trained on. "
                              f"Test set performance (see below) is the real metric.")
        else:
            # Binary classification
            if model_name in ["TabPFN", "TabICL"]:
                # TabPFN/TabICL return class predictions directly for binary classification
                if hasattr(model, "predict_proba") and model_name != "TabPFN":
                    # TabICL might have predict_proba
                    y_prob_train = model.predict_proba(X_topk_df)[:, 1]
                else:
                    # TabPFN or models without predict_proba: use predictions as probabilities
                    y_pred_binary = y_pred_train.astype(float)
                    y_prob_train = np.where(y_pred_binary == 1, 0.7, 0.3)
            else:
                if hasattr(model, "predict_proba"):
                    y_prob_train = model.predict_proba(X_topk)[:, 1]
                else:
                    y_prob_train = model.predict(X_topk).astype(float)
            y_prob_train = np.clip(y_prob_train, 0.0, 1.0)
            y_pred_train_05 = (y_prob_train >= 0.5).astype(int)
            acc_train_05 = accuracy_score(y_topk, y_pred_train_05)
            f1_train_05 = f1_score(y_topk, y_pred_train_05, zero_division=0)
            thr_opt_train = find_optimal_threshold(y_topk, y_prob_train, metric='f1')
            y_pred_train_opt = (y_prob_train >= thr_opt_train).astype(int)
            acc_train_opt = accuracy_score(y_topk, y_pred_train_opt)
            f1_train_opt = f1_score(y_topk, y_pred_train_opt, zero_division=0)
            ml_baseline_train_metrics = {
                "accuracy": acc_train_opt, "f1": f1_train_opt,
                "threshold_opt": float(thr_opt_train),
                "acc@0.5": acc_train_05, "f1@0.5": f1_train_05,
                "predictions": y_pred_train_opt.tolist()
            }
            logger.info(f"ML baseline on TRAINING set ({len(X_topk)} samples): ACC={acc_train_opt:.4f} F1={f1_train_opt:.4f} (thr_opt={thr_opt_train:.2f})")
            if acc_train_opt >= 0.99:
                logger.warning(f"  ⚠️  Training accuracy is {acc_train_opt:.4f} - this indicates overfitting. "
                              f"The model is being evaluated on the same data it was trained on. "
                              f"Test set performance (see below) is the real metric.")
    except Exception as e:
        logger.warning(f"Failed to compute ML baseline metrics on training set: {e}")
    
    # Compute ML baseline metrics on test set (using frozen model)
    ml_baseline_metrics: Dict[str, Any] = {}
    try:
        is_multiclass = class_names is not None and len(class_names) > 2
        model = pretrained_ml.model
        # TabPFN and TabICL require DataFrames, not numpy arrays
        if model_name == "TabPFN":
            # TabPFN uses subprocess isolation - model is None, use batch prediction helper
            import subprocess
            import tempfile
            from pathlib import Path
            import sys
            import json
            import pandas as pd
            
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
                 str(X_test_path),
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
                    y_pred_cls = np.array(pred_result['predictions'], dtype=int)
                else:
                    raise RuntimeError(f"TabPFN test set prediction failed: {pred_result.get('error', 'Unknown error')}")
            else:
                raise RuntimeError("TabPFN test set prediction subprocess failed")
            
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
        elif model_name == "TabICL":
            import pandas as pd
            X_test_df = pd.DataFrame(X_test_s, columns=feature_cols)
            if is_multiclass:
                y_pred_cls = model.predict(X_test_df).astype(int)
            else:
                y_pred_cls = model.predict(X_test_df).astype(int)
        else:
            # Other models can use numpy arrays directly
            if is_multiclass:
                y_pred_cls = model.predict(X_test_s).astype(int)
            else:
                y_pred_cls = None  # Will be computed below for binary
        
        if is_multiclass:
            acc = accuracy_score(y_test, y_pred_cls)
            f1 = f1_score(y_test, y_pred_cls, average='weighted', zero_division=0)
            ml_baseline_metrics = {
                "accuracy": acc, "f1": f1,
                "predictions": y_pred_cls.tolist()
            }
            logger.info(f"ML baseline on TEST set: ACC={acc:.4f} F1={f1:.4f}")
        else:
            # Binary classification
            if model_name in ["TabPFN", "TabICL"]:
                # TabPFN/TabICL return class predictions directly for binary classification
                if hasattr(model, "predict_proba") and model_name != "TabPFN":
                    # TabICL might have predict_proba
                    y_prob = model.predict_proba(X_test_df)[:, 1]
                else:
                    # TabPFN or models without predict_proba: use predictions as probabilities
                    y_pred_binary = y_pred_cls.astype(float)
                    # Use a simple heuristic: if prediction is 1, use 0.7, if 0, use 0.3
                    y_prob = np.where(y_pred_binary == 1, 0.7, 0.3)
            else:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_s)[:, 1]
                else:
                    y_prob = model.predict(X_test_s).astype(float)
            y_prob = np.clip(y_prob, 0.0, 1.0)
            y_pred_05 = (y_prob >= 0.5).astype(int)
            acc_05 = accuracy_score(y_test, y_pred_05)
            f1_05 = f1_score(y_test, y_pred_05, zero_division=0)
            thr_opt = find_optimal_threshold(y_test, y_prob, metric='f1')
            y_pred_opt = (y_prob >= thr_opt).astype(int)
            acc_opt = accuracy_score(y_test, y_pred_opt)
            f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
            ml_baseline_metrics = {
                "accuracy": acc_opt, "f1": f1_opt,
                "threshold_opt": float(thr_opt),
                "acc@0.5": acc_05, "f1@0.5": f1_05,
                "predictions": y_pred_opt.tolist()
            }
            logger.info(f"ML baseline (frozen, trained on full set: {len(X_train_s)} samples): ACC={acc_opt:.4f} F1={f1_opt:.4f} (thr_opt={thr_opt:.2f})")
    except Exception as e:
        logger.warning(f"Failed to compute ML baseline metrics: {e}")
    
    # Compute additional baselines (TabPFN, EBM, SHAP)
    logger.info("=" * 80)
    logger.info("COMPUTING ADDITIONAL BASELINES")
    logger.info("=" * 80)
    if not _HAS_TABPFN:
        logger.warning("⚠ TabPFN not available - skipping TabPFN baseline")
        logger.warning("  Install with: pip install tabpfn")
        logger.warning("  Note: TabPFN requires GPU for datasets >1000 samples")
    additional_baselines = compute_additional_baselines(
        X_train_s, y_train, X_test_s, y_test,
        task_type="classification",
        feature_cols=feature_cols,
        y_scaler=None,
        no_scaling=args.no_scaling,
        baseline_models=args.baseline_models
    )

    maicl = TrainableMAICL(
        llm, feature_cols, scaler,  # Pass scaler (can be None)
        use_ml_mechanism=bool(args.use_ml),
        dataset_name=ds_name,  # Use actual dataset name (e.g., "car", "iris") for proper context
        y_scaler=None,
        pretrained_ml_mechanism=pretrained_ml,
        data_insights=None,
        task_type="classification",
        class_names=class_names,
        classification_loss_metric=args.classification_loss,
        num_mechanisms_unknown=args.num_mechanisms_unknown,
        use_scaling=not args.no_scaling  # Pass the new flag
    )
    if ml_baseline_metrics:
        try:
            maicl.set_ml_baseline_performance(ml_baseline_metrics)
        except Exception:
            pass

    # Log routing mode and loss metric for clarity
    logger.info(f"Classification loss metric: {args.classification_loss.upper()}")
    if bool(args.relax_eval):
        logger.info("Evaluation routing: RELAXED (LLM mechanisms can contribute).")
    else:
        logger.info("Evaluation routing: STRICT ML preference (LLM contributions limited).")

    # Pre metrics (loss based on --classification_loss: F1 or accuracy)
    logger.info("=" * 80)
    logger.info("PRE-TRAINING EVALUATION")
    logger.info("=" * 80)
    
    pre = maicl.evaluate(
        X_test_s,
        y_test,
        X_train_s,
        y_train,
        return_details=True,
        relax_routing=bool(args.relax_eval),
        # IMPORTANT (DeepChem molecular classification):
        # - ML uses vectorized features (e.g., ECFP bits in X_*_s)
        # - LLM/TextGrad should see SMILES (X_original_*), not the vectorized bits
        X_original=X_original_test,
        X_pool_original=X_original_train,
    )
    pre_acc = float(pre.get('accuracy', 0.0))
    pre_f1 = float(pre.get('f1', 0.0))
    pre_loss = float(pre.get('loss', 1.0))
    logger.info(f"Pre-training: ACC={pre_acc:.4f} F1={pre_f1:.4f} Loss={pre_loss:.4f}")
    
    # Compare pre-training performance with ML baseline on training set
    if ml_baseline_train_metrics:
        ml_train_acc = ml_baseline_train_metrics.get('accuracy', 0.0)
        ml_train_f1 = ml_baseline_train_metrics.get('f1', 0.0)
        logger.info(f"  [Comparison] ML baseline on training set: ACC={ml_train_acc:.4f} F1={ml_train_f1:.4f}")
        logger.info(f"  [Comparison] MA-ICL pre-training vs ML (train): ΔACC={pre_acc - ml_train_acc:+.4f}, ΔF1={pre_f1 - ml_train_f1:+.4f}")
    
    # Extract pre-training predictions from evaluate results
    y_pred_pre = np.array(pre.get('predictions', [])) if 'predictions' in pre else None
    
    # Track individual mechanism performance (pre-training)
    pre_mechanism_performance = {}
    if hasattr(maicl, 'mechanism_performance_snapshot'):
        pre_mechanism_performance = maicl.mechanism_performance_snapshot.copy()
        logger.info(f"Pre-training mechanism performance: {pre_mechanism_performance}")
    
    # Evaluate with only LLM mechanisms (excluding ML) to assess LLM learning BEFORE training
    logger.info("\n[LLM-only Evaluation (Pre-training)] Evaluating MA-ICL with only LLM mechanisms (excluding ML)...")
    llm_only_pre_metrics = None
    try:
        llm_only_pre_metrics = maicl.evaluate_llm_only(
            X_test_s,
            y_test,
            X_train_s,
            y_train,
            return_details=True,
            X_original=X_original_test,
            X_pool_original=X_original_train,
        )
        llm_only_pre_acc = float(llm_only_pre_metrics.get('accuracy', 0.0))
        llm_only_pre_f1 = float(llm_only_pre_metrics.get('f1', 0.0))
        llm_only_pre_loss = float(llm_only_pre_metrics.get('loss', 1.0))
        logger.info(f"LLM-only (Pre-training, no ML): ACC={llm_only_pre_acc:.4f} F1={llm_only_pre_f1:.4f} Loss={llm_only_pre_loss:.4f}")
    except Exception as e:
        logger.warning(f"Failed to evaluate LLM-only pre-training: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")

    # Train on top-K residuals; TextGrad/evaluation will use selected loss metric (F1 or accuracy)
    logger.info("=" * 80)
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
    maicl.train(
        X_topk,
        y_topk,
        X_val_s,
        y_val,
        iterations=args.iterations,
        ml_residuals=residuals_topk,
        accept_eval_max=None,
        X_test=X_test_s,
        y_test=y_test,
        acceptance_set=args.acceptance_set,
        # IMPORTANT: provide original (non-vectorized) inputs for LLM/TextGrad.
        # For DeepChem these are SMILES dicts: {"SMILES": "..."}.
        X_train_original=X_original_topk,
        X_val_original=X_original_val,
        X_test_original=X_original_test,
        output_dir=output_dir,
    )
    
    # Ensure training artifacts (training progress plots) are exported
    logger.info("\n[Training Artifacts] Exporting training history and progress plots...")
    try:
        if hasattr(maicl, '_export_training_artifacts'):
            maicl._export_training_artifacts(output_dir=output_dir)
            logger.info(f"  ✓ Training progress plots saved to {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to export training artifacts: {e}")
    
    # Post metrics
    logger.info("=" * 80)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("=" * 80)
    
    # CRITICAL FIX: Use the same routing as training to ensure fair comparison
    # Training always uses relax_routing=True, so final evaluation should too
    # This ensures that the performance during iterations matches the final result
    final_relax_routing = True  # Always use relaxed routing to match training
    logger.info("Using relaxed routing for final evaluation (same as training)")
    logger.info("  Note: Training uses relax_routing=True, so final evaluation uses the same for consistency")
    logger.info("  Note: Iteration metrics are on validation set, final evaluation is on test set")
    logger.info("  Note: Mechanism performance will be PRESERVED from best snapshot (not recalculated) to ensure consistent routing")
    # CRITICAL: Explicitly preserve mechanism performance from best snapshot to ensure consistent routing
    # This ensures final evaluation uses the same routing weights as the best iteration during training
    post = maicl.evaluate(
        X_test_s,
        y_test,
        X_train_s,
        y_train,
        return_details=True,
        relax_routing=final_relax_routing,
        preserve_mechanism_performance=True,
        X_original=X_original_test,
        X_pool_original=X_original_train,
    )
    post_acc = float(post.get('accuracy', 0.0))
    post_f1 = float(post.get('f1', 0.0))
    post_loss = float(post.get('loss', 1.0))
    logger.info(f"Post-training: ACC={post_acc:.4f} F1={post_f1:.4f} Loss={post_loss:.4f}")
    
    # Extract post-training predictions from evaluate results
    y_pred_post = np.array(post.get('predictions', [])) if 'predictions' in post else None
    
    # Track individual mechanism performance (post-training)
    post_mechanism_performance = {}
    if hasattr(maicl, 'mechanism_performance_snapshot'):
        post_mechanism_performance = maicl.mechanism_performance_snapshot.copy()
        logger.info(f"Post-training mechanism performance: {post_mechanism_performance}")
    
    # Explicitly save final mechanisms to ensure they're saved
    try:
        from visualization import persist_iteration_artifacts
        final_iteration = args.iterations
        final_mech_snapshot = {
            "mechanisms": maicl.mechanisms,
            "mechanism_types": maicl.mechanism_types
        }
        final_metrics = {
            "iteration": final_iteration,
            "loss": post_loss
        }
        persist_iteration_artifacts(final_iteration, final_mech_snapshot, final_metrics, output_dir=output_dir)
        logger.info(f"  ✓ Final mechanisms saved to {output_dir}/mechanisms_iter_{final_iteration}.txt")
    except Exception as e:
        logger.warning(f"Failed to save final mechanisms: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # Evaluate with only LLM mechanisms (excluding ML) to assess LLM learning
    logger.info("\n[LLM-only Evaluation] Evaluating MA-ICL with only LLM mechanisms (excluding ML)...")
    llm_only_metrics = None
    try:
        llm_only_metrics = maicl.evaluate_llm_only(
            X_test_s,
            y_test,
            X_train_s,
            y_train,
            return_details=True,
            X_original=X_original_test,
            X_pool_original=X_original_train,
        )
        llm_only_acc = float(llm_only_metrics.get('accuracy', 0.0))
        llm_only_f1 = float(llm_only_metrics.get('f1', 0.0))
        llm_only_loss = float(llm_only_metrics.get('loss', 1.0))
        logger.info(f"LLM-only (Post-training, no ML): ACC={llm_only_acc:.4f} F1={llm_only_f1:.4f} Loss={llm_only_loss:.4f}")
    except Exception as e:
        logger.warning(f"Failed to evaluate LLM-only: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # Generate performance comparison visualizations
    logger.info("\n[Visualizations] Generating performance comparison plots and confusion matrices...")
    try:
        # Ensure predictions are included in metrics dicts for visualization
        pre_with_preds = pre.copy()
        if y_pred_pre is not None and 'predictions' not in pre_with_preds:
            pre_with_preds['predictions'] = y_pred_pre.tolist()
        
        post_with_preds = post.copy()
        if y_pred_post is not None and 'predictions' not in post_with_preds:
            post_with_preds['predictions'] = y_pred_post.tolist()
        
        # Ensure ML baseline has predictions
        if 'predictions' not in ml_baseline_metrics:
            logger.warning("ML baseline metrics missing predictions - confusion matrix may not be generated")
        
        create_result_visualizations(
            y_test=y_test,
            ml_baseline_metrics=ml_baseline_metrics,
            pre_metrics=pre_with_preds,
            post_metrics=post_with_preds,
            task_type="classification",
            class_names=class_names,
            output_dir=output_dir,
            llm_only_metrics=llm_only_metrics,
            llm_only_pre_metrics=llm_only_pre_metrics,
            additional_baselines=additional_baselines
        )
        logger.info("  ✓ Performance comparison plot saved")
        logger.info("  ✓ Confusion matrices saved (ML baseline, MA-ICL pre, MA-ICL post, MA-ICL LLM-only)")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # Save final metrics
    # Compute improvements vs ML baseline
    ml_acc = ml_baseline_metrics.get('accuracy', 0.0)
    ml_f1 = ml_baseline_metrics.get('f1', 0.0)
    
    final_results = {
        "dataset": ds_label,
        "dataset_name": ds_name,
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
            "accuracy": pre_acc,
            "f1": pre_f1,
            "loss": pre_loss,
            "mechanism_performance": pre_mechanism_performance
        },
        "post_training": {
            "accuracy": post_acc,
            "f1": post_f1,
            "loss": post_loss,
            "mechanism_performance": post_mechanism_performance
        },
        "llm_only_pre_training": llm_only_pre_metrics if llm_only_pre_metrics is not None else None,
        "llm_only_post_training": llm_only_metrics if llm_only_metrics is not None else None,
        "mechanism_info": {
            "total_mechanisms": len(maicl.mechanisms),
            "mechanism_types": maicl.mechanism_types,
            "ml_mechanism_index": [i for i, t in enumerate(maicl.mechanism_types) if t == "ml"],
            "llm_mechanism_indices": [i for i, t in enumerate(maicl.mechanism_types) if t == "llm"]
        },
        "improvements": {
            "training_improvement": {
                "acc_delta": post_acc - pre_acc,
                "f1_delta": post_f1 - pre_f1,  # Positive is better
                "loss_delta": pre_loss - post_loss,  # Positive is better (reduction)
            },
            "vs_ml_baseline_pre": {
                "acc_delta": pre_acc - ml_acc,
                "f1_delta": pre_f1 - ml_f1,  # Positive is better
            },
            "vs_ml_baseline_post": {
                "acc_delta": post_acc - ml_acc,
                "f1_delta": post_f1 - ml_f1,  # Positive is better
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
    logger.info(f"  ML Baseline (TEST):   ACC={ml_acc:.4f}, F1={ml_f1:.4f}")
    if ml_baseline_train_metrics:
        ml_train_acc = ml_baseline_train_metrics.get('accuracy', 0.0)
        ml_train_f1 = ml_baseline_train_metrics.get('f1', 0.0)
        logger.info(f"  ML Baseline (TRAIN):  ACC={ml_train_acc:.4f}, F1={ml_train_f1:.4f}")
    
    # Additional baselines
    if additional_baselines:
        if 'tabpfn' in additional_baselines:
            tabpfn_acc = additional_baselines['tabpfn'].get('accuracy', 0.0)
            tabpfn_f1 = additional_baselines['tabpfn'].get('f1', 0.0)
            logger.info(f"  TabPFN Baseline:     ACC={tabpfn_acc:.4f}, F1={tabpfn_f1:.4f}")
        elif not _HAS_TABPFN:
            logger.info(f"  TabPFN Baseline:     [SKIPPED - not installed]")
        if 'ebm' in additional_baselines:
            ebm_acc = additional_baselines['ebm'].get('accuracy', 0.0)
            ebm_f1 = additional_baselines['ebm'].get('f1', 0.0)
            logger.info(f"  EBM Baseline:       ACC={ebm_acc:.4f}, F1={ebm_f1:.4f}")
        if 'shap' in additional_baselines:
            shap_acc = additional_baselines['shap'].get('accuracy', 0.0)
            shap_f1 = additional_baselines['shap'].get('f1', 0.0)
            logger.info(f"  SHAP Baseline:      ACC={shap_acc:.4f}, F1={shap_f1:.4f}")
    
    logger.info(f"  Pre-training:  ACC={pre_acc:.4f}, F1={pre_f1:.4f}")
    if llm_only_pre_metrics is not None:
        llm_only_pre_acc = float(llm_only_pre_metrics.get('accuracy', 0.0))
        llm_only_pre_f1 = float(llm_only_pre_metrics.get('f1', 0.0))
        logger.info(f"  LLM-only (Pre, no ML): ACC={llm_only_pre_acc:.4f}, F1={llm_only_pre_f1:.4f}")
    logger.info(f"  Post-training: ACC={post_acc:.4f}, F1={post_f1:.4f}")
    if llm_only_metrics is not None:
        llm_only_acc = float(llm_only_metrics.get('accuracy', 0.0))
        llm_only_f1 = float(llm_only_metrics.get('f1', 0.0))
        logger.info(f"  LLM-only (Post, no ML): ACC={llm_only_acc:.4f}, F1={llm_only_f1:.4f}")
    logger.info("")
    logger.info("Training Improvement (Post vs Pre):")
    logger.info(f"  ΔACC={final_results['improvements']['training_improvement']['acc_delta']:+.4f}, "
                f"ΔF1={final_results['improvements']['training_improvement']['f1_delta']:+.4f}, "
                f"ΔLoss={final_results['improvements']['training_improvement']['loss_delta']:+.4f}")
    logger.info("")
    logger.info("Pre-training vs ML Baseline:")
    logger.info(f"  ΔACC={final_results['improvements']['vs_ml_baseline_pre']['acc_delta']:+.4f}, "
                f"ΔF1={final_results['improvements']['vs_ml_baseline_pre']['f1_delta']:+.4f}")
    logger.info("")
    logger.info("Post-training vs ML Baseline:")
    logger.info(f"  ΔACC={final_results['improvements']['vs_ml_baseline_post']['acc_delta']:+.4f}, "
                f"ΔF1={final_results['improvements']['vs_ml_baseline_post']['f1_delta']:+.4f}")
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
    
    logger.info("")
    logger.info("Generated Visualizations:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("  Training Progress:")
    logger.info("    - training_curves.png (validation loss and metrics over iterations)")
    logger.info("    - training_history.json (detailed training history)")
    logger.info("    - training_metrics.csv (training metrics in CSV format)")
    logger.info("  Performance Comparison:")
    logger.info("    - performance_comparison.png (bar chart: ML baseline vs MA-ICL pre/post)")
    logger.info("  Confusion Matrices:")
    logger.info("    - confusion_matrix_ml_baseline.png (ML model predictions)")
    logger.info("    - confusion_matrix_maicl_pre.png (MA-ICL before training)")
    logger.info("    - confusion_matrix_maicl_post.png (MA-ICL after training)")
    if llm_only_metrics is not None:
        logger.info("    - confusion_matrix_maicl_llm_only_post.png (MA-ICL LLM-only, no ML)")
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
                        "y_train": y_train,
                        "X_test": X_test_s,
                        "y_test": y_test,
                        "feature_cols": feature_cols,
                        "scaler": scaler,
                        "y_scaler": None,  # Classification: y is not scaled
                        "dataset_name": ds_label,
                        "class_names": class_names,
                        "task_type": "classification"
                    }
                    
                    # Import parsing functions (evaluation will be done directly for classification)
                    from evaluate_individual_mechanisms import (
                        parse_mechanisms_file,
                        extract_formula_from_llm_mechanism
                    )
                    
                    mechanisms = parse_mechanisms_file(mechanisms_file)
                    individual_results = []
                    
                    # Helper function to evaluate LLM formula for classification
                    def evaluate_llm_formula_classification(formula: str, X: np.ndarray, feature_cols: List[str], 
                                                           scaler: Any, class_names: Optional[List[str]] = None) -> np.ndarray:
                        """Evaluate LLM formula and convert output to class indices"""
                        try:
                            # Create feature dictionaries for each sample
                            x_dicts = [{feature_cols[j]: float(X[i, j]) for j in range(len(feature_cols))} 
                                      for i in range(len(X))]
                            
                            # Use MA-ICL's predictor to evaluate the formula
                            # Create a temporary predictor with just this mechanism
                            from maicl_lib_v2 import MultiAgentPredictor, BatchedLLM
                            # We need the LLM to evaluate the formula - use the existing LLM instance
                            temp_mechanisms = [formula]
                            temp_types = ["llm"]
                            temp_predictor = MultiAgentPredictor(
                                llm, temp_mechanisms, temp_types, None, feature_cols, scaler,
                                attention_temp=1.0, task_type="classification", class_names=class_names
                            )
                            
                            # Get predictions
                            batch_outputs = temp_predictor.predict_batch(x_dicts, few_shot_list=[[]] * len(x_dicts))
                            predictions = [out[0] for out in batch_outputs]
                            
                            # Convert to class indices
                            preds = np.array(predictions)
                            preds = np.round(preds).astype(int)
                            if class_names:
                                n_classes = len(class_names)
                                preds = np.clip(preds, 0, n_classes - 1)
                            return preds
                        except Exception as e:
                            logger.warning(f"Failed to evaluate LLM formula: {e}")
                            # Return default predictions (all class 0)
                            return np.zeros(len(X), dtype=int)
                    
                    for mech_idx, (mech_type, mech_text) in enumerate(mechanisms):
                        logger.info(f"\nEvaluating Mechanism {mech_idx+1}/{len(mechanisms)}: [{mech_type.upper()}]")
                        try:
                            if mech_type == "llm" or mech_type == "known":
                                formula = extract_formula_from_llm_mechanism(mech_text)
                                if formula is None:
                                    formula = mech_text
                                # For classification, LLM formulas should output class indices
                                train_preds = evaluate_llm_formula_classification(formula, X_train_s, feature_cols, scaler, class_names)
                                test_preds = evaluate_llm_formula_classification(formula, X_test_s, feature_cols, scaler, class_names)
                            elif mech_type == "ml":
                                # For ML mechanism, use the pretrained ML model
                                if pretrained_ml and pretrained_ml.is_trained:
                                    # Create feature dictionaries
                                    train_dicts = [{feature_cols[j]: float(X_train_s[i, j]) for j in range(len(feature_cols))} 
                                                  for i in range(len(X_train_s))]
                                    test_dicts = [{feature_cols[j]: float(X_test_s[i, j]) for j in range(len(feature_cols))} 
                                                 for i in range(len(X_test_s))]
                                    
                                    # Get predictions
                                    train_preds = np.array([pretrained_ml.predict(d, return_class_index=True) for d in train_dicts])
                                    test_preds = np.array([pretrained_ml.predict(d, return_class_index=True) for d in test_dicts])
                                    
                                    # Ensure integer class indices
                                    train_preds = np.round(train_preds).astype(int)
                                    test_preds = np.round(test_preds).astype(int)
                                    if class_names:
                                        n_classes = len(class_names)
                                        train_preds = np.clip(train_preds, 0, n_classes - 1)
                                        test_preds = np.clip(test_preds, 0, n_classes - 1)
                                else:
                                    logger.warning(f"  ML mechanism not trained, skipping")
                                    continue
                            else:
                                continue
                            
                            # Convert predictions to class indices for classification
                            train_preds = np.array(train_preds)
                            test_preds = np.array(test_preds)
                            
                            # Round to nearest integer class index
                            train_preds = np.round(train_preds).astype(int)
                            test_preds = np.round(test_preds).astype(int)
                            
                            # Clip to valid class range
                            if class_names:
                                n_classes = len(class_names)
                                train_preds = np.clip(train_preds, 0, n_classes - 1)
                                test_preds = np.clip(test_preds, 0, n_classes - 1)
                            
                            # Compute classification metrics
                            train_acc = accuracy_score(y_train, train_preds)
                            train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)
                            test_acc = accuracy_score(y_test, test_preds)
                            test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)
                            
                            individual_results.append({
                                "mechanism_index": mech_idx,
                                "mechanism_type": mech_type,
                                "train_metrics": {"accuracy": float(train_acc), "f1": float(train_f1)},
                                "test_metrics": {"accuracy": float(test_acc), "f1": float(test_f1)}
                            })
                            
                            logger.info(f"  Train: ACC={train_acc:.4f}, F1={train_f1:.4f}")
                            logger.info(f"  Test:  ACC={test_acc:.4f}, F1={test_f1:.4f}")
                            
                            # Generate confusion matrices for train and test
                            try:
                                from sklearn.metrics import confusion_matrix
                                import matplotlib.pyplot as plt
                                import matplotlib
                                matplotlib.use('Agg')
                                
                                # Train confusion matrix
                                cm_train = confusion_matrix(y_train, train_preds)
                                plt.figure(figsize=(8, 6))
                                plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title(f"Mechanism {mech_idx} ({mech_type.upper()}) - Train Set\nACC={train_acc:.4f}, F1={train_f1:.4f}")
                                plt.colorbar()
                                tick_marks = np.arange(len(class_names) if class_names else len(np.unique(y_train)))
                                plt.xticks(tick_marks, class_names if class_names else tick_marks)
                                plt.yticks(tick_marks, class_names if class_names else tick_marks)
                                plt.ylabel('True Label')
                                plt.xlabel('Predicted Label')
                                thresh = cm_train.max() / 2.
                                for i, j in np.ndindex(cm_train.shape):
                                    plt.text(j, i, format(cm_train[i, j], 'd'),
                                           horizontalalignment="center",
                                           color="white" if cm_train[i, j] > thresh else "black")
                                plt.tight_layout()
                                plt.savefig(os.path.join(output_dir, f"mechanism_{mech_idx}_{mech_type}_train_confusion.png"), dpi=150)
                                plt.close()
                                
                                # Test confusion matrix
                                cm_test = confusion_matrix(y_test, test_preds)
                                plt.figure(figsize=(8, 6))
                                plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title(f"Mechanism {mech_idx} ({mech_type.upper()}) - Test Set\nACC={test_acc:.4f}, F1={test_f1:.4f}")
                                plt.colorbar()
                                tick_marks = np.arange(len(class_names) if class_names else len(np.unique(y_test)))
                                plt.xticks(tick_marks, class_names if class_names else tick_marks)
                                plt.yticks(tick_marks, class_names if class_names else tick_marks)
                                plt.ylabel('True Label')
                                plt.xlabel('Predicted Label')
                                thresh = cm_test.max() / 2.
                                for i, j in np.ndindex(cm_test.shape):
                                    plt.text(j, i, format(cm_test[i, j], 'd'),
                                           horizontalalignment="center",
                                           color="white" if cm_test[i, j] > thresh else "black")
                                plt.tight_layout()
                                plt.savefig(os.path.join(output_dir, f"mechanism_{mech_idx}_{mech_type}_test_confusion.png"), dpi=150)
                                plt.close()
                            except Exception as e:
                                logger.warning(f"  Failed to generate confusion matrices: {e}")
                        except Exception as e:
                            logger.warning(f"  Failed to evaluate mechanism {mech_idx+1}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Save individual mechanism results
                    individual_results_file = os.path.join(output_dir, f"individual_mechanism_performance_iter_{final_iteration}.json")
                    with open(individual_results_file, 'w') as f:
                        json.dump({
                            "iteration": final_iteration,
                            "dataset": ds_label,
                            "n_train": len(X_train_s),
                            "n_test": len(X_test_s),
                            "task_type": "classification",
                            "class_names": class_names,
                            "mechanisms": individual_results
                        }, f, indent=2)
                    logger.info(f"\nIndividual mechanism results saved to: {individual_results_file}")
                    
                    # Also save CSV summary
                    try:
                        pd = _get_pandas()
                        csv_data = []
                        for r in individual_results:
                            csv_data.append({
                                "mechanism_index": r["mechanism_index"],
                                "mechanism_type": r["mechanism_type"],
                                "train_accuracy": r["train_metrics"]["accuracy"],
                                "train_f1": r["train_metrics"]["f1"],
                                "test_accuracy": r["test_metrics"]["accuracy"],
                                "test_f1": r["test_metrics"]["f1"],
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

if __name__ == "__main__":
    main()


