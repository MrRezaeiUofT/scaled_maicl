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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        X_df = X_df.sample(n=max_samples, random_state=RANDOM_STATE).reset_index(drop=True)
        y_series = y_series.loc[X_df.index].reset_index(drop=True)
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
    """Load GFP Yield Prediction dataset (real experimental data)"""
    if not _HAS_PANDAS:
        raise RuntimeError("pandas not installed. pip install pandas")
    try:
        csv_path = "gfp_yield/2025_07_25_GFP_Results_removed_outliers.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if 'Experiment no' in df.columns:
            df['Yield'] = df.groupby('Experiment no')['Yield'].transform('mean')
            df = df.groupby('Experiment no').first().reset_index(drop=True)
        df = df.select_dtypes(include=[np.number]).dropna()
        feature_cols = [col for col in df.columns if col != 'Yield']
        if not feature_cols:
            raise ValueError("No feature columns found")
        X_df = df[feature_cols].astype(float)
        y_series = df["Yield"].astype(float)
        if len(X_df) > 300:
            logger.info(f"Dataset too large ({len(X_df)} samples), subsampling to 250 samples")
            np.random.seed(51)
            indices = np.random.choice(len(X_df), 250, replace=False)
            X_df = X_df.iloc[indices].reset_index(drop=True)
            y_series = y_series.iloc[indices].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Failed to load GFP yield data: {e}")
        raise RuntimeError(f"GFP yield dataset not available: {e}")
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
        
        loader = EnzymeDatasetLoader(data_dir)
        
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
            raise FileNotFoundError(f"Could not load dataset: {dataset_name}")
        
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
        
        # Create original feature dicts (for LLM context)
        X_original = []
        df_valid = df[valid_mask].reset_index(drop=True)
        # Match X_df with df_valid (they should have the same length after subsampling)
        df_valid_subset = df_valid.iloc[:len(X_df)].reset_index(drop=True)
        
        for idx in range(len(X_df)):
            if idx >= len(df_valid_subset):
                break
            # Include both extracted features and original text for LLM
            feat_dict = X_df.iloc[idx].to_dict()
            feat_dict['SEQ'] = str(df_valid_subset.iloc[idx]['SEQ'])
            feat_dict['SUBSTRATES'] = str(df_valid_subset.iloc[idx]['SUBSTRATES'])
            X_original.append(feat_dict)
        
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
        
        # TabArena regression dataset mapping (HuggingFace/sklearn)
        # Note: Only datasets that are confirmed to exist and are accessible are included
        TABARENA_REGRESSION_DATASETS = {
            "abalone": ["mstz/abalone"],
            "auto": ["scikit-learn/auto-mpg"],
            "diamonds": ["mstz/diamonds"],  # Requires trust_remote_code=True and config selection
        }
        
        if dataset_name_lower not in TABARENA_REGRESSION_DATASETS:
            available = ["diabetes"] + list(TABARENA_REGRESSION_DATASETS.keys())
            raise ValueError(f"TabArena regression dataset '{dataset_name}' not found or not available. Available datasets: {available}")
        
        # Try loading from multiple possible paths
        dataset_paths = TABARENA_REGRESSION_DATASETS[dataset_name_lower]
        if not isinstance(dataset_paths, list):
            dataset_paths = [dataset_paths]
        
        dataset = None
        last_error = None
        
        # Special handling for diamonds dataset (requires trust_remote_code and config)
        if dataset_name_lower == "diamonds":
            # Try different configs for diamonds dataset
            diamonds_configs = ["cut", "cut_binary", "encoding"]
            for config in diamonds_configs:
                try:
                    logger.info(f"Trying to load diamonds dataset with config '{config}' from {dataset_paths[0]}")
                    dataset = load_dataset(dataset_paths[0], config, trust_remote_code=True)
                    logger.info(f"Successfully loaded diamonds dataset with config '{config}' ({len(dataset.get('train', []))} samples)")
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load diamonds with config '{config}': {e}")
                    continue
        else:
            # Standard loading for other datasets
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
        
        # X_original: SMILES strings for LLM
        # DeepChem stores SMILES in dataset.ids
        X_original = []
        for smiles in all_ids:
            # SMILES string is the primary input for LLM
            # Include it as a feature dict that LLM can understand
            feat_dict = {'SMILES': str(smiles)}
            X_original.append(feat_dict)
        
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


def main():
    parser = argparse.ArgumentParser(description="MA-ICL regression on biotech/experimental datasets with top-K residuals")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name. Options:\n"
                             "  Built-in: gfp_yield, protein_expression, protein_expression_all, dataset_102\n"
                             "  TabArena (HuggingFace/sklearn regression):\n"
                             "    - diabetes (sklearn)\n"
                             "    - housing (sklearn, with HuggingFace fallback)\n"
                             "    - abalone (HuggingFace)\n"
                             "    - auto (HuggingFace)\n"
                             "    - diamonds (HuggingFace, requires trust_remote_code)\n"
                             "    Note: Only confirmed available datasets are listed.\n"
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
    parser.add_argument("--plate_file", type=str, default=None,
                        help="Plate file name for protein_expression dataset (e.g., 'plate_AL_1_raw_yield_and_std.csv')")
    parser.add_argument("--plate_index", type=int, default=None,
                        help="Plate index (0-based) for protein_expression dataset. Use --list_plates to see available indices.")
    parser.add_argument("--list_plates", action="store_true",
                        help="List all available protein expression plate files and exit")
    parser.add_argument("--model_name", default=os.environ.get("MAICL_MODEL_NAME", "gemini-2.0-flash"),
                        help="Gemini model name, e.g., gemini-2.0-flash, gemini-2.0-pro")
    parser.add_argument("--ml_mech", default="linear", help="Regression ML mechanism: linear|xgboost|kernelridge|tabicl")
    parser.add_argument("--tabicl_bins", type=int, default=20,
                        help="Number of bins for TabICL regression quantization (default: 20). "
                             "Only used when --ml_mech=tabicl. Higher values = finer granularity but more classes.")
    parser.add_argument("--use_ml", type=int, default=1, choices=[0,1], help="Include ML mechanism in ensemble")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=1000, help="-1 to use full dataset")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--acceptance_set",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help=(
            "Which split to use for acceptance evaluation in TextGrad: "
            "'train', 'val', or 'test'. Defaults to 'val'. "
            "Using 'test' can help align optimization with test performance "
            "but risks overfitting to the test set."
        ),
    )
    parser.add_argument(
        "--use_test_for_acceptance",
        action="store_true",
        help=(
            "[DEPRECATED] Legacy flag to use the test set for acceptance. "
            "Prefer --acceptance_set=test instead. If provided together with "
            "--acceptance_set (left at its default 'val'), this flag will switch "
            "acceptance to the test set for backward compatibility."
        ),
    )
    parser.add_argument("--topk_strategy", choices=["residual", "residual_balanced"], default="residual",
                        help="Top-K selection: 'residual' = global highest | 'residual_balanced' = highest within y-quantile bins")
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
    
    # Include ML baseline model name in output directory
    ml_model_suffix = f"_ml{args.ml_mech}"
    
    # Use lowercase for output directory name (for consistency), but preserve original for enzyme dataset loading
    run_name = f"bio_reg_{dataset_name_lower}{plate_suffix}{ml_model_suffix}_topk{args.top_k}"
    output_dir = set_output_dir(run_name)
    logger.info(f"Output directory: {output_dir}")

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
    elif dataset_name_lower in ("esol", "delaney", "lipo", "lipophilicity"):
        # DeepChem molecular regression datasets
        X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = load_deepchem_regression_dataset(
            args.dataset, args.max_samples
        )
    elif dataset_name_lower in ("diabetes", "housing", "abalone", "auto", "diamonds") or dataset_name_lower.startswith("tabarena_"):
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
                           f"TabArena (diabetes, housing, abalone, auto, diamonds), "
                           f"DeepChem (any molnet dataset, e.g., esol, delaney, lipo, lipophilicity), "
                           f"or enzyme dataset names (e.g., halogenase_NaBr, aminotransferase, olea, phosphatase_achiral)")
    from sklearn.model_selection import train_test_split
    # Regression-friendly stratification: bin y into quantiles and stratify to preserve target distribution
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
            logger.info("Used quantile-stratified train/validation/test split for regression.")
        except Exception:
            X_train, X_val, y_train, y_val, idx_tr, idx_va = train_test_split(
                X_train_full, y_train_full, np.arange(len(X_train_full)), test_size=float(args.val_size), random_state=RANDOM_STATE
            )
            # Split X_original_all using same indices
            X_original_train = [X_original_all[idx_tr_full[i]] for i in idx_tr]
            X_original_val = [X_original_all[idx_tr_full[i]] for i in idx_va]
            X_original_test = [X_original_all[i] for i in idx_te]
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
        logger.info("Used regular random train/validation/test split (stratification unavailable).")

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

    # Scale features to [0, 10] using MinMaxScaler010 (uses SCALE_MIN=0.0, SCALE_MAX=10.0)
    scaler = MinMaxScaler010()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    validate_scaled_data(X_train_s, feature_cols, scaler=scaler)
    logger.info(f"Features scaled to [{SCALE_MIN}, {SCALE_MAX}] range (X min={X_train_s.min():.3f}, max={X_train_s.max():.3f})")

    # Scale target to [0, 10] to match feature scaling for consistency
    # Use MinMaxScaler010 to ensure same scaling range as features
    y_scaler_target = MinMaxScaler010()  # Uses SCALE_MIN=0.0, SCALE_MAX=10.0
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
    mech_map = {"linear": "LinearRegression", "xgboost": "XGBoost", "kernelridge": "KernelRidge", "tabicl": "TabICL"}
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
    
    # Optionally retrain ML model on the same subset that MA-ICL will train on
    # This ensures both models train on the same data for fair comparison
    residuals_topk = None
    if args.use_ml:
        if args.top_k != -1:
            logger.info(f"Retraining ML model on the same {len(X_topk)} samples that MA-ICL will train on...")
            pretrained_ml.train(X_topk, y_topk, feature_cols, y_scaler=None)
            # Recompute residuals on the subset
            sorted_idx, residuals_topk, ml_preds, _ = compute_ml_residuals(
                pretrained_ml, X_topk, y_topk, feature_cols, class_names=None, task_type="regression"
            )
            logger.info(f"ML model and MA-ICL now train on the same {len(X_topk)} samples")
        else:
            # Both train on full set, use original residuals
            residuals_topk = residuals
            logger.info(f"Both ML model and MA-ICL train on the same full training set ({len(X_topk)} samples)")
        
        # Compute ML baseline metrics on test set (after potential retraining)
        ml_baseline_metrics: Dict[str, Any] = {}
        try:
            model = pretrained_ml.model
            y_pred = model.predict(X_test_s).astype(float)
            r2 = r2_score(y_test_s, y_pred)
            mae = mean_absolute_error(y_test_s, y_pred)
            mse = mean_squared_error(y_test_s, y_pred)
            ml_baseline_metrics = {
                "r2": r2, "mae": mae, "mse": mse,
                "predictions": y_pred.tolist()
            }
            logger.info(f"ML baseline (trained on {'subset' if args.top_k != -1 else 'full set'}): R2={r2:.4f} MAE={mae:.4f} MSE={mse:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute ML baseline metrics: {e}")
    else:
        # LLM-only mode: no ML residuals or baseline metrics
        ml_baseline_metrics = {}
        logger.info(f"LLM-only mode: Training on {len(X_topk)} samples (no ML baseline)")

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
        num_mechanisms_unknown=args.num_mechanisms_unknown
    )
    if ml_baseline_metrics:
        try:
            maicl.set_ml_baseline_performance(ml_baseline_metrics)
        except Exception:
            pass

    # Pre metrics (MAE as loss; also R2/MSE)
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
    
    # Generate pre-training plots
    if y_pred_pre is not None and len(y_pred_pre) > 0:
        save_scatter_plot(y_test_s, y_pred_pre, 
                         f"Pre-Training: Predicted vs Actual ({ds_label})", 
                         "pre_scatter.png", output_dir)
        save_residual_plot(y_test_s, y_pred_pre,
                          f"Pre-Training ({ds_label})",
                          "pre_residuals.png", output_dir)

    logger.info("TRAINING MA-ICL")
    logger.info("=" * 80)
    logger.info(f"Training on {len(X_topk)} top-K residual samples for {args.iterations} iterations")
    # Resolve which split is used for acceptance (train/val/test)
    acceptance_set = (args.acceptance_set or "val").lower()
    if acceptance_set not in ("train", "val", "test"):
        logger.warning(f"Unknown --acceptance_set='{acceptance_set}', defaulting to 'val'")
        acceptance_set = "val"
    # Backward compatibility: legacy flag can still request test when acceptance_set is left at default
    if args.use_test_for_acceptance and acceptance_set == "val":
        acceptance_set = "test"
    if acceptance_set == "test":
        logger.info(f"Using TEST set ({len(X_test_s)} samples) for acceptance evaluation")
    elif acceptance_set == "train":
        logger.info(f"Using TRAIN set ({len(X_topk)} samples) for acceptance evaluation")
    else:
        logger.info(f"Using full validation set ({len(X_val_s)} samples) for acceptance evaluation")
    # Use validation or test set for acceptance; pass only TRAIN residuals to LLM
    # accept_eval_max=None means use full acceptance set
    # Use residuals_topk to ensure consistency with training data (X_topk)
    # For DeepChem datasets, pass X_original for validation and test sets to use SMILES strings
    train_kwargs = {'X_train_original': X_original_topk}
    if is_deepchem_dataset:
        train_kwargs['X_val_original'] = X_original_val
        train_kwargs['X_test_original'] = X_original_test
    maicl.train(
        X_topk,
        y_topk,
        X_val_s,
        y_val_s,
        iterations=args.iterations,
        ml_residuals=residuals_topk if args.use_ml else None,  # Only pass residuals if ML is enabled
        accept_eval_max=None,
        # Use few-shot examples during training so LLM mechanisms see rich context
        # This also propagates to LLM-only evaluation (k_shot is stored on the MA-ICL instance)
        k_shot=10,
        X_test=X_test_s,
        y_test=y_test_s,
        use_test_for_acceptance=args.use_test_for_acceptance,
        acceptance_set=acceptance_set,
        output_dir=output_dir,
        **train_kwargs,
    )

    # Post metrics
    logger.info("=" * 80)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("=" * 80)
    
    # For final test metrics: use same routing as training
    # If we optimized on test set, use relaxed routing; otherwise use strict routing to avoid leakage
    optimized_on_test = (acceptance_set == "test")
    final_relax_routing = optimized_on_test or (args.relax_eval == 1)
    if final_relax_routing:
        logger.info("Using relaxed routing for final evaluation (same as training)")
    else:
        logger.info("Using strict routing for final evaluation (to avoid validation leakage)")
    
    # CRITICAL: Mechanism performance will always be recalculated on the test set to ensure
    # accurate routing with the final accepted mechanisms. The preserve_mechanism_performance
    # flag only affects whether we update the snapshot, not whether we recalculate.
    # This ensures post-training results reflect the actual performance of the final mechanisms.
    preserve_perf = args.use_test_for_acceptance
    if preserve_perf:
        logger.info("Recalculating mechanism performance on test set (preserving snapshot for consistency)")
        logger.info("  Note: Performance is recalculated to ensure accurate routing with final accepted mechanisms")
    else:
        logger.info("Recalculating mechanism performance scores on test set (for accurate routing)")
    
    # Log final accepted mechanisms for transparency
    final_mechanism_count = len(maicl.mechanisms) if hasattr(maicl, 'mechanisms') else 0
    final_llm_count = sum(1 for t in maicl.mechanism_types if t == "llm") if hasattr(maicl, 'mechanism_types') else 0
    final_ml_count = sum(1 for t in maicl.mechanism_types if t == "ml") if hasattr(maicl, 'mechanism_types') else 0
    logger.info(f"Evaluating with final accepted mechanisms: {final_mechanism_count} total ({final_llm_count} LLM, {final_ml_count} ML)")
    
    # For DeepChem datasets, pass X_original to use SMILES strings instead of vectorized features
    post = maicl.evaluate(X_test_s, y_test_s, X_train_s, y_train_s, return_details=True, 
                          relax_routing=final_relax_routing, preserve_mechanism_performance=preserve_perf,
                          X_original=X_original_test if is_deepchem_dataset else None,
                          X_pool_original=X_original_train if is_deepchem_dataset else None)
    post_mae = float(post.get('mae', 0.0))
    post_r2 = float(post.get('r2', 0.0))
    post_rmse = float(post.get('rmse', 0.0))
    post_mse = float(post_rmse ** 2)
    logger.info(f"Post-training: R2={post_r2:.4f} MAE={post_mae:.4f} MSE={post_mse:.4f}")
    
    # Extract post-training predictions from evaluate results
    y_pred_post = np.array(post.get('predictions', [])) if 'predictions' in post else None
    
    # Track individual mechanism performance (post-training)
    post_mechanism_performance = {}
    if hasattr(maicl, 'mechanism_performance_snapshot'):
        post_mechanism_performance = maicl.mechanism_performance_snapshot.copy()
        logger.info(f"Post-training mechanism performance: {post_mechanism_performance}")
    
    # Generate post-training plots
    if y_pred_post is not None and len(y_pred_post) > 0:
        save_scatter_plot(y_test_s, y_pred_post,
                         f"Post-Training: Predicted vs Actual ({ds_label})",
                         "post_scatter.png", output_dir)
        save_residual_plot(y_test_s, y_pred_post,
                          f"Post-Training ({ds_label})",
                          "post_residuals.png", output_dir)
    
    # Evaluate LLM-only mechanisms (excluding ML) to assess LLM learning
    logger.info("\n" + "=" * 80)
    logger.info("LLM-ONLY EVALUATION (excluding ML mechanisms)")
    logger.info("=" * 80)
    llm_only_metrics = None
    y_pred_llm_only = None
    try:
        # For DeepChem datasets, pass X_original to use SMILES strings instead of vectorized features
        llm_only_kwargs = {}
        if is_deepchem_dataset:
            llm_only_kwargs['X_original'] = X_original_test
            llm_only_kwargs['X_pool_original'] = X_original_train
        llm_only_metrics = maicl.evaluate_llm_only(X_test_s, y_test_s, X_train_s, y_train_s, 
                                                    return_details=True, k_shot=maicl.k_shot if hasattr(maicl, 'k_shot') else 10,
                                                    **llm_only_kwargs)
        llm_only_r2 = float(llm_only_metrics.get('r2', -1.0))
        llm_only_mae = float(llm_only_metrics.get('mae', 1e9))
        llm_only_mse = float(llm_only_metrics.get('mse', 1e9))
        logger.info(f"LLM-only: R2={llm_only_r2:.4f} MAE={llm_only_mae:.4f} MSE={llm_only_mse:.4f}")
        
        y_pred_llm_only = np.array(llm_only_metrics.get('predictions', [])) if 'predictions' in llm_only_metrics else None
        if y_pred_llm_only is not None and len(y_pred_llm_only) > 0:
            save_scatter_plot(y_test_s, y_pred_llm_only,
                             f"LLM-Only: Predicted vs Actual ({ds_label})",
                             "llm_only_scatter.png", output_dir)
            save_residual_plot(y_test_s, y_pred_llm_only,
                              f"LLM-Only ({ds_label})",
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
            output_dir=output_dir
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
        "llm_only": {
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
    logger.info(f"  ML Baseline:   R2={ml_r2:.4f}, MAE={ml_mae:.4f}, MSE={ml_mse:.4f}")
    logger.info(f"  Pre-training:  R2={pre_r2:.4f}, MAE={pre_mae:.4f}, MSE={pre_mse:.4f}")
    logger.info(f"  Post-training: R2={post_r2:.4f}, MAE={post_mae:.4f}, MSE={post_mse:.4f}")
    if llm_only_metrics:
        llm_only_r2 = float(llm_only_metrics.get('r2', -1.0))
        llm_only_mae = float(llm_only_metrics.get('mae', 1e9))
        llm_only_mse = float(llm_only_metrics.get('mse', 1e9))
        logger.info(f"  LLM-only:     R2={llm_only_r2:.4f}, MAE={llm_only_mae:.4f}, MSE={llm_only_mse:.4f}")
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
                                train_preds = evaluate_llm_formula(formula, X_train_s, feature_cols, scaler, y_scaler_target)
                                test_preds = evaluate_llm_formula(formula, X_test_s, feature_cols, scaler, y_scaler_target)
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


if __name__ == "__main__":
    main()


