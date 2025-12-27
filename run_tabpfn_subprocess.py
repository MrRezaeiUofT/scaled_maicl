#!/usr/bin/env python3
"""
Subprocess wrapper for TabPFN to isolate crashes.
This script runs TabPFN in a separate process so segfaults don't crash the main script.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

def run_tabpfn(X_train_path, y_train_path, X_test_path, output_path):
    """Run TabPFN and save results to JSON file."""
    try:
        # Load data
        X_train = pd.read_csv(X_train_path)
        y_train = np.load(y_train_path)
        X_test = pd.read_csv(X_test_path)
        
        # Import TabPFN
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        
        # Clear PyTorch cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except:
            pass
        
        # Initialize and train TabPFN
        regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        regressor.fit(X_train, y_train)
        
        # Predict
        y_pred = regressor.predict(X_test)
        
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
        # Save error information
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
    if len(sys.argv) != 5:
        print("Usage: run_tabpfn_subprocess.py <X_train.csv> <y_train.npy> <X_test.csv> <output.json>", file=sys.stderr)
        sys.exit(1)
    
    X_train_path, y_train_path, X_test_path, output_path = sys.argv[1:5]
    exit_code = run_tabpfn(X_train_path, y_train_path, X_test_path, output_path)
    sys.exit(exit_code)

