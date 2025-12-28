#!/usr/bin/env python3
"""
Subprocess wrapper for TabPFN ML mechanism to isolate crashes.
This script handles both training and prediction for TabPFN when used as ML mechanism.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

def train_tabpfn(X_train_path, y_train_path, task_type, output_path):
    """Train TabPFN and save training data for later predictions."""
    try:
        # Set random seeds for reproducibility
        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        try:
            import torch
            torch.manual_seed(RANDOM_STATE)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_STATE)
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except:
            pass
        
        # Load data
        X_train = pd.read_csv(X_train_path)
        y_train = np.load(y_train_path)
        
        # Import TabPFN
        if task_type == "regression":
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        else:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        
        # Clear PyTorch cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except:
            pass
        
        # Train TabPFN
        model.fit(X_train, y_train)
        
        # Save training data for later predictions (TabPFN needs training data for predictions)
        # We'll save the training data paths so we can reload them for predictions
        results = {
            'success': True,
            'task_type': task_type,
            'X_train_path': str(X_train_path),
            'y_train_path': str(y_train_path),
            'error': None
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f)
        
        return 0
        
    except Exception as e:
        results = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f)
        except:
            pass
        return 1

def predict_tabpfn(X_train_path, y_train_path, X_pred_path, task_type, output_path):
    """Predict using TabPFN with saved training data."""
    try:
        # Set random seeds for reproducibility
        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        try:
            import torch
            torch.manual_seed(RANDOM_STATE)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_STATE)
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except:
            pass
        
        # Load data
        X_train = pd.read_csv(X_train_path)
        y_train = np.load(y_train_path)
        X_pred = pd.read_csv(X_pred_path)
        
        # Import TabPFN
        if task_type == "regression":
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        else:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        
        # Clear PyTorch cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except:
            pass
        
        # Train and predict (TabPFN needs training data for each prediction)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        
        # Save results
        results = {
            'success': True,
            'predictions': y_pred.tolist(),
            'error': None
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f)
        
        return 0
        
    except Exception as e:
        results = {
            'success': False,
            'predictions': None,
            'error': str(e),
            'error_type': type(e).__name__
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f)
        except:
            pass
        return 1

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  Train: run_tabpfn_ml_mechanism.py train <X_train.csv> <y_train.npy> <task_type> <output.json>", file=sys.stderr)
        print("  Predict: run_tabpfn_ml_mechanism.py predict <X_train.csv> <y_train.npy> <X_pred.csv> <task_type> <output.json>", file=sys.stderr)
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "train":
        if len(sys.argv) != 6:
            print("Usage: run_tabpfn_ml_mechanism.py train <X_train.csv> <y_train.npy> <task_type> <output.json>", file=sys.stderr)
            sys.exit(1)
        X_train_path, y_train_path, task_type, output_path = sys.argv[2:6]
        exit_code = train_tabpfn(X_train_path, y_train_path, task_type, output_path)
    elif mode == "predict":
        if len(sys.argv) != 7:
            print("Usage: run_tabpfn_ml_mechanism.py predict <X_train.csv> <y_train.npy> <X_pred.csv> <task_type> <output.json>", file=sys.stderr)
            sys.exit(1)
        X_train_path, y_train_path, X_pred_path, task_type, output_path = sys.argv[2:7]
        exit_code = predict_tabpfn(X_train_path, y_train_path, X_pred_path, task_type, output_path)
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'predict'", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(exit_code)

