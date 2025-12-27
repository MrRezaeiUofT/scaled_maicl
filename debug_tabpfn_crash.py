#!/usr/bin/env python3
"""
Debug script to identify where TabPFN crashes.
"""

import numpy as np
import pandas as pd
import sys
import traceback

print("=" * 80)
print("TabPFN Crash Debug Script")
print("=" * 80)

# Test 1: Import
print("\n[1] Testing import...")
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    print("  ✓ Import successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create minimal dataset
print("\n[2] Creating minimal dataset...")
X_train = np.random.rand(10, 5).astype(np.float32)
y_train = np.random.rand(10).astype(np.float32)
X_test = np.random.rand(5, 5).astype(np.float32)

X_train_df = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(5)])
X_test_df = pd.DataFrame(X_test, columns=[f'f{i}' for i in range(5)])

print(f"  ✓ Created dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
print(f"  Data types: X_train={X_train.dtype}, y_train={y_train.dtype}")

# Test 3: Initialize model
print("\n[3] Initializing TabPFN v2...")
try:
    regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    print("  ✓ Model initialized")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check model attributes
print("\n[4] Checking model attributes...")
try:
    print(f"  Device: {getattr(regressor, 'device', 'unknown')}")
    print(f"  Model type: {type(regressor)}")
    if hasattr(regressor, 'model_'):
        print(f"  Model_: {type(regressor.model_)}")
except Exception as e:
    print(f"  ⚠️  Could not check attributes: {e}")

# Test 5: Try fit() - this is where it crashes
print("\n[5] Attempting fit() - THIS IS WHERE IT CRASHES...")
print("  Data info:")
print(f"    X_train_df shape: {X_train_df.shape}, dtypes: {X_train_df.dtypes.unique()}")
print(f"    y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
print(f"    y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")

try:
    # Try with DataFrame
    print("  Attempting fit with DataFrame...")
    regressor.fit(X_train_df, y_train)
    print("  ✓ Fit successful!")
except SystemError as e:
    print(f"  ✗ SystemError (possible segfault): {e}")
    traceback.print_exc()
except Exception as e:
    print(f"  ✗ Exception: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug complete")
print("=" * 80)

