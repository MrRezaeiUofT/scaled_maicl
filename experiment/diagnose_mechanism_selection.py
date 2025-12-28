#!/usr/bin/env python3
"""
Diagnostic script to evaluate why LLM mechanism wasn't selected and why it performs poorly.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import from same directory (we're now in the experiment folder)
from evaluate_individual_mechanisms import (
    parse_mechanisms_file,
    extract_formula_from_llm_mechanism,
    evaluate_llm_formula,
    _compute_molecular_properties,
)
# Import from 019_run_deepchem_llm_mechanism_rdkit_validate (filename starts with number, use importlib)
import importlib.util
deepchem_module_path = str(Path(__file__).parent / "019_run_deepchem_llm_mechanism_rdkit_validate.py")
spec = importlib.util.spec_from_file_location("run_deepchem_llm_mechanism_rdkit_validate", deepchem_module_path)
run_deepchem_module = importlib.util.module_from_spec(spec)
sys.modules["run_deepchem_llm_mechanism_rdkit_validate"] = run_deepchem_module
spec.loader.exec_module(run_deepchem_module)  # type: ignore

# Now we can import from it
_run_deepchem = run_deepchem_module
_extract_json_after_marker = _run_deepchem._extract_json_after_marker
_rdkit_descriptor_matrix_from_smiles_dicts = _run_deepchem._rdkit_descriptor_matrix_from_smiles_dicts
_pearson_corr = _run_deepchem._pearson_corr
_sign_label = _run_deepchem._sign_label
_sanitize_formula_single_line = _run_deepchem._sanitize_formula_single_line
_find_unsupported_descriptor_mentions = _run_deepchem._find_unsupported_descriptor_mentions
SUPPORTED_RDKIT_DESCRIPTOR_NAMES = _run_deepchem.SUPPORTED_RDKIT_DESCRIPTOR_NAMES

# Import the DeepChem loader
_load_module_from_path = _run_deepchem._load_module_from_path
here = Path(__file__).resolve().parent
runner_path = str(here / "018_maicl_regression_biotech.py")
runner = _load_module_from_path(runner_path, module_name="_maicl_regression_biotech_018")

from maicl.maicl_lib_v2 import MinMaxScaler010


def main():
    # Load the dataset
    print("Loading ESOL dataset...")
    (
        (X_train, y_train, X_original_train),
        (X_val, y_val, X_original_val),
        (X_test, y_test, X_original_test),
        feature_cols,
        feature_encoders,
        ds_label,
    ) = runner.load_deepchem_regression_dataset(
        "esol",
        200,
        featurizer="ECFP",
        splitter="random",
        return_splits=True,
    )
    
    # Scale y to [0,1]
    y_scaler = MinMaxScaler010()
    y_train_s = y_scaler.fit_transform(np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(np.asarray(y_test, dtype=float).reshape(-1, 1)).ravel()
    y_val_s = np.clip(y_val_s, 0.0, 1.0)
    y_test_s = np.clip(y_test_s, 0.0, 1.0)
    
    # Load mechanisms
    mechanisms_file = "ma_icl_results/rdkit_validate_esol_20251226_114447/mechanisms_iter_3.txt"
    print(f"\nLoading mechanisms from: {mechanisms_file}")
    mechs = parse_mechanisms_file(mechanisms_file)
    
    print(f"\nFound {len(mechs)} mechanisms:")
    for i, (mech_type, mech_text) in enumerate(mechs):
        print(f"  {i+1}. Type: {mech_type.upper()}")
        print(f"     Preview: {mech_text[:100]}...")
    
    # Evaluate each mechanism
    print("\n" + "="*80)
    print("EVALUATING EACH MECHANISM")
    print("="*80)
    
    for mech_type, mech_text in mechs:
        print(f"\n{'='*80}")
        print(f"MECHANISM TYPE: {mech_type.upper()}")
        print(f"{'='*80}")
        
        # Extract formula - try to get the FINAL formula (not intermediate ones)
        formula = extract_formula_from_llm_mechanism(mech_text) or ""
        
        # If multiple "Formula:" lines exist, prefer the one with "ŷ" or in "STEP 4" or "FINAL"
        import re
        all_formula_matches = list(re.finditer(
            r'^\s*(?:Formula|FORMULA|formula)\s*:\s*(.+?)\s*$',
            mech_text,
            re.IGNORECASE | re.MULTILINE
        ))
        
        if len(all_formula_matches) > 1:
            # Find the one with "ŷ" or near "STEP 4" or "FINAL"
            for match in reversed(all_formula_matches):  # Check from end
                formula_candidate = match.group(1).strip()
                # Check if this is the final prediction (has ŷ or clip)
                if 'ŷ' in formula_candidate or 'clip' in formula_candidate.lower():
                    formula = formula_candidate
                    # Remove the "ŷ = " prefix if present
                    formula = re.sub(r'^\s*(?:ŷ|ŷ|y_hat|y)\s*=\s*', '', formula, flags=re.IGNORECASE).strip()
                    break
        
        formula = _sanitize_formula_single_line(formula)
        
        if not formula:
            print("  ❌ No formula found!")
            continue
        
        print(f"\n  Formula: {formula[:150]}...")
        
        # Check for unsupported descriptors
        unsupported = _find_unsupported_descriptor_mentions(formula)
        if unsupported:
            print(f"  ⚠️  Unsupported descriptors: {unsupported}")
            continue
        
        # Check for RDKitValidation block
        rdv = _extract_json_after_marker(mech_text, marker="RDKitValidation")
        has_rdkit_validation = isinstance(rdv, dict) and rdv.get("expected_effects")
        print(f"  RDKitValidation block: {'✓ Found' if has_rdkit_validation else '✗ Missing'}")
        
        if has_rdkit_validation:
            expected = rdv.get("expected_effects", {})
            print(f"  Expected effects: {len(expected)} descriptors")
            for k, v in list(expected.items())[:5]:
                print(f"    - {k}: {v}")
        
        # Evaluate on validation set
        try:
            # Debug: test formula on a single sample first
            if mech_type.lower() == "known":
                print(f"\n  Debugging KNOWN mechanism formula...")
                print(f"  Formula: {formula}")
                # Test on first sample
                test_smiles = X_original_val[0] if X_original_val else None
                if test_smiles:
                    props = _compute_molecular_properties(str(test_smiles))
                    print(f"  Sample SMILES: {test_smiles}")
                    print(f"  Sample properties:")
                    for k, v in props.items():
                        print(f"    {k}: {v}")
            
            y_pred_val = evaluate_llm_formula(
                formula=formula,
                X=np.asarray(X_val, dtype=float),
                feature_cols=feature_cols,
                scaler=None,
                y_scaler=None,
                X_original=X_original_val,
            ).astype(float)
            
            val_metrics = {
                "r2": float(r2_score(y_val_s, y_pred_val)),
                "mae": float(mean_absolute_error(y_val_s, y_pred_val)),
                "mse": float(mean_squared_error(y_val_s, y_pred_val)),
                "rmse": float(np.sqrt(mean_squared_error(y_val_s, y_pred_val))),
            }
            
            print(f"\n  Validation Metrics:")
            print(f"    R²:  {val_metrics['r2']:8.4f}")
            print(f"    MAE: {val_metrics['mae']:8.4f}")
            print(f"    MSE: {val_metrics['mse']:8.4f}")
            print(f"    RMSE: {val_metrics['rmse']:8.4f}")
            
            # Evaluate on test set
            y_pred_test = evaluate_llm_formula(
                formula=formula,
                X=np.asarray(X_test, dtype=float),
                feature_cols=feature_cols,
                scaler=None,
                y_scaler=None,
                X_original=X_original_test,
            ).astype(float)
            
            test_metrics = {
                "r2": float(r2_score(y_test_s, y_pred_test)),
                "mae": float(mean_absolute_error(y_test_s, y_pred_test)),
                "mse": float(mean_squared_error(y_test_s, y_pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_s, y_pred_test))),
            }
            
            print(f"\n  Test Metrics:")
            print(f"    R²:  {test_metrics['r2']:8.4f}")
            print(f"    MAE: {test_metrics['mae']:8.4f}")
            print(f"    MSE: {test_metrics['mse']:8.4f}")
            print(f"    RMSE: {test_metrics['rmse']:8.4f}")
            
            # Analyze prediction distribution
            print(f"\n  Prediction Statistics:")
            print(f"    Min:  {np.min(y_pred_val):.4f}")
            print(f"    Max:  {np.max(y_pred_val):.4f}")
            print(f"    Mean: {np.mean(y_pred_val):.4f}")
            print(f"    Std:  {np.std(y_pred_val):.4f}")
            
            print(f"\n  Target Statistics (validation):")
            print(f"    Min:  {np.min(y_val_s):.4f}")
            print(f"    Max:  {np.max(y_val_s):.4f}")
            print(f"    Mean: {np.mean(y_val_s):.4f}")
            print(f"    Std:  {np.std(y_val_s):.4f}")
            
            # Check if predictions are mostly clipped
            clipped_count = np.sum((y_pred_val <= 0.01) | (y_pred_val >= 0.99))
            clipped_pct = 100 * clipped_count / len(y_pred_val)
            print(f"\n  Clipped predictions: {clipped_count}/{len(y_pred_val)} ({clipped_pct:.1f}%)")
            
            # RDKitValidation sign check if available
            if has_rdkit_validation:
                desc = _rdkit_descriptor_matrix_from_smiles_dicts(X_original_val)
                corr = {k: _pearson_corr(desc[k], y_val_s) for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES}
                
                print(f"\n  RDKitValidation Sign Check:")
                n_match = 0
                n_total = 0
                for k, exp_sign in expected.items():
                    if k not in corr:
                        continue
                    exp_sign_str = str(exp_sign).strip().lower()
                    if exp_sign_str not in ("positive", "negative", "neutral"):
                        continue
                    got = _sign_label(corr[k])
                    ok = (got == exp_sign_str) or (exp_sign_str == "neutral" and got == "neutral")
                    n_total += 1
                    n_match += 1 if ok else 0
                    status = "✓" if ok else "✗"
                    print(f"    {status} {k:25s}: expected={exp_sign_str:8s}, observed={got:8s}, corr={corr[k]:7.4f}")
                
                if n_total > 0:
                    match_rate = n_match / n_total
                    print(f"\n  Match rate: {match_rate:.2%} ({n_match}/{n_total})")
            
        except Exception as e:
            print(f"  ❌ Error evaluating formula: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

