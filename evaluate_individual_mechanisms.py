"""
Module for evaluating individual mechanisms from MA-ICL training.

This module provides functions to:
1. Parse mechanism files
2. Extract formulas from LLM mechanisms
3. Evaluate mechanisms on train/test sets
4. Generate visualization plots
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Try to import RDKit for molecular property computation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_mechanisms_file(filepath: str) -> List[Tuple[str, str]]:
    """
    Parse a mechanisms file into a list of (type, text) tuples.
    
    File format:
        [LLM] mechanism text 1
        
        [ML] mechanism text 2
        
        [KNOWN] mechanism text 3
    
    Also handles cases where multiple [TYPE] markers appear in the same section:
        [KNOWN] [KNOWN] context text
        [LLM] mechanism text
    
    Args:
        filepath: Path to the mechanisms file
        
    Returns:
        List of (mechanism_type, mechanism_text) tuples
    """
    mechanisms = []
    
    if not os.path.exists(filepath):
        return mechanisms
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First, try to find all [TYPE] markers in the entire file
    # This handles cases where multiple markers appear in one section
    type_pattern = r'\[(\w+)\]\s*'
    matches = list(re.finditer(type_pattern, content))
    
    if not matches:
        # Fallback: split by double newlines (original behavior)
        sections = content.split('\n\n')
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            match = re.match(r'\[(\w+)\]\s*(.*)', section, re.DOTALL)
            if match:
                mech_type = match.group(1).lower()
                mech_text = match.group(2).strip()
                mechanisms.append((mech_type, mech_text))
        return mechanisms
    
    # Extract mechanisms based on [TYPE] markers
    for i, match in enumerate(matches):
        mech_type = match.group(1).lower()
        start_pos = match.end()
        
        # Find the end of this mechanism (next [TYPE] marker or end of file)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        mech_text = content[start_pos:end_pos].strip()
        
        # Skip if text is empty or just whitespace
        if mech_text:
            mechanisms.append((mech_type, mech_text))
    
    return mechanisms


def extract_formula_from_llm_mechanism(mechanism_text: str) -> Optional[str]:
    """
    Extract a formula from LLM mechanism text.
    
    Looks for patterns like:
    - "Formula: ..."
    - "ŷ = ..."
    - "score = ..."
    
    Args:
        mechanism_text: The mechanism text to parse
        
    Returns:
        Extracted formula string, or None if not found
    """
    # Look for "Formula:" pattern
    formula_match = re.search(
        r'(?:Formula|FORMULA|formula)[:\s]*(.*?)(?:\n|$|ŷ\s*=|Map|Class)', 
        mechanism_text, 
        re.IGNORECASE | re.DOTALL
    )
    
    if not formula_match:
        # Try to find "score = " or "ŷ = " directly
        formula_match = re.search(
            r'(?:score|ŷ)\s*=\s*(.*?)(?:\n|$|Map|Class|ŷ\s*=|if\s+score)', 
            mechanism_text, 
            re.IGNORECASE | re.DOTALL
        )
    
    if formula_match:
        formula = formula_match.group(1).strip()
        # Remove trailing punctuation or extra text
        formula = re.sub(r'[.;,]$', '', formula)
        return formula
    
    return None


def _compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Compute molecular properties from a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular properties
    """
    if not RDKIT_AVAILABLE:
        # Fallback: return zeros if RDKit is not available
        return {
            'molecular_weight': 0.0,
            'num_rings': 0.0,
            'num_hydroxyl_groups': 0.0,
            'num_halogen': 0.0,
            'num_nitrogen': 0.0,
            'num_oxygen': 0.0,
            'num_atoms': 0.0,
            'is_aromatic': 0.0,
            'num_hbd': 0.0  # hydrogen bond donors
        }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'molecular_weight': 0.0,
                'num_rings': 0.0,
                'num_hydroxyl_groups': 0.0,
                'num_halogen': 0.0,
                'num_nitrogen': 0.0,
                'num_oxygen': 0.0,
                'num_atoms': 0.0,
                'is_aromatic': 0.0,
                'num_hbd': 0.0
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
            'molecular_weight': float(mw),
            'num_rings': float(num_rings),
            'num_hydroxyl_groups': float(num_hydroxyl),
            'num_halogen': float(num_halogen),
            'num_nitrogen': float(num_nitrogen),
            'num_oxygen': float(num_oxygen),
            'num_atoms': float(num_atoms),
            'is_aromatic': is_aromatic,
            'num_hbd': float(num_hbd)
        }
    except Exception:
        return {
            'molecular_weight': 0.0,
            'num_rings': 0.0,
            'num_hydroxyl_groups': 0.0,
            'num_halogen': 0.0,
            'num_nitrogen': 0.0,
            'num_oxygen': 0.0,
            'num_atoms': 0.0,
            'is_aromatic': 0.0,
            'num_hbd': 0.0
        }


def evaluate_llm_formula(
    formula: str,
    X: np.ndarray,
    feature_cols: List[str],
    scaler: Any,
    y_scaler: Optional[Any] = None,
    X_original: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Evaluate an LLM formula on data.
    
    Args:
        formula: The formula string to evaluate
        X: Feature matrix (n_samples, n_features)
        feature_cols: List of feature names
        scaler: Feature scaler (for inverse transform if needed)
        y_scaler: Target scaler (for inverse transform if needed)
        X_original: Optional list of original (non-vectorized) features (e.g., SMILES strings)
        
    Returns:
        Array of predictions
    """
    predictions = []
    
    # Check if this is a DeepChem dataset (formula uses molecular properties)
    is_deepchem = 'molecular_weight(SMILES)' in formula or 'num_rings(SMILES)' in formula or 'num_hydroxyl_groups(SMILES)' in formula
    
    # Detect if we have categorical features in X_original (e.g., SU dataset with chemical, concentration_label)
    has_categorical_features = False
    categorical_feature_names = []
    if X_original is not None and len(X_original) > 0 and isinstance(X_original[0], dict):
        # Check for common categorical feature names that should use original string values
        sample_dict = X_original[0]
        categorical_feature_names = [k for k in sample_dict.keys() 
                                     if k in feature_cols and isinstance(sample_dict[k], str) 
                                     and k not in ['SMILES', 'SEQ', 'SUBSTRATES']]  # Exclude special fields
        has_categorical_features = len(categorical_feature_names) > 0
    
    for i in range(len(X)):
        # For DeepChem datasets, use SMILES from X_original
        if is_deepchem and X_original is not None and i < len(X_original):
            # Extract SMILES string
            if isinstance(X_original[i], dict):
                smiles = X_original[i].get('SMILES', '')
            elif isinstance(X_original[i], str):
                smiles = X_original[i]
            else:
                smiles = str(X_original[i])
            
            # Compute molecular properties from SMILES
            mol_props = _compute_molecular_properties(smiles)
            
            # Create x_dict with molecular properties
            x_dict = mol_props.copy()
            # Also add SMILES as a variable (for formulas that reference it)
            x_dict['SMILES'] = smiles
        elif has_categorical_features and X_original is not None and i < len(X_original):
            # For datasets with categorical features (e.g., SU dataset), use encoded numeric values
            # for feature_cols (so formulas work), but also provide original string values
            x_dict = {}
            orig_dict = X_original[i] if isinstance(X_original[i], dict) else {}
            
            for j, feat_name in enumerate(feature_cols):
                # Always use encoded numeric value for feature_cols (so formulas like 0.667*chemical work)
                x_dict[feat_name] = float(X[i, j]) if j < X.shape[1] else 0.0
                
                # Also provide original string value with _name suffix for conditional logic
                if feat_name in categorical_feature_names and feat_name in orig_dict:
                    x_dict[f"{feat_name}_name"] = orig_dict[feat_name]
        else:
            # For non-DeepChem datasets without categoricals, use vectorized features
            x_dict = {feature_cols[j]: float(X[i, j]) for j in range(len(feature_cols))}
        
        # Evaluate formula
        pred = _execute_formula(formula, x_dict)
        
        if pred is None:
            # If formula execution fails, try to use the raw formula text
            # This is a fallback - in practice, you might want to use LLM execution
            pred = 0.0
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Apply inverse scaling if y_scaler is provided
    if y_scaler is not None:
        try:
            # Reshape for scaler
            pred_reshaped = predictions.reshape(-1, 1)
            predictions = y_scaler.inverse_transform(pred_reshaped).flatten()
        except Exception:
            pass
    
    return predictions


def _execute_formula(formula: str, x_dict: Dict[str, float]) -> Optional[float]:
    """
    Safely execute a formula with feature values.
    
    Handles molecular property functions like molecular_weight(SMILES), num_rings(SMILES), etc.
    by replacing them with their computed values from x_dict.
    
    Args:
        formula: Formula string (may contain function calls like molecular_weight(SMILES))
        x_dict: Dictionary of feature names to values (including molecular properties)
        
    Returns:
        Computed value or None if execution fails
    """
    try:
        # Replace molecular property function calls with their values
        # Pattern: function_name(SMILES) -> function_name
        formula_processed = formula
        
        # List of molecular property functions
        mol_prop_functions = [
            'molecular_weight', 'num_rings', 'num_hydroxyl_groups', 'num_halogen',
            'num_nitrogen', 'num_oxygen', 'num_atoms', 'is_aromatic', 'num_hbd'
        ]
        
        for func_name in mol_prop_functions:
            # Replace function_name(SMILES) with just function_name
            # This handles patterns like: molecular_weight(SMILES) / 100
            pattern = rf'{func_name}\s*\([^)]*\)'
            if func_name in x_dict:
                # Replace the function call with the value from x_dict
                formula_processed = re.sub(pattern, func_name, formula_processed)
        
        # Create safe evaluation environment
        safe_dict = {}
        for key, val in x_dict.items():
            # Sanitize key name (remove special chars, replace with underscore)
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
            
            if isinstance(val, (int, float)):
                # Numeric values: add as float
                safe_dict[safe_key] = float(val)
                # Also add original key if different
                if safe_key != key:
                    safe_dict[key] = float(val)
            elif isinstance(val, str):
                # String values (categorical features with _name suffix): add as string for comparisons
                # These are provided for conditional logic (e.g., if chemical_name == "DMSO")
                safe_dict[safe_key] = val
                if safe_key != key:
                    safe_dict[key] = val
        
        # Add math functions to safe environment
        safe_dict.update({
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pow': pow, '__builtins__': {}
        })
        
        # Add clip function for regression formulas
        def clip_func(x, min_val, max_val):
            return max(min_val, min(max_val, float(x)))
        safe_dict['clip'] = clip_func
        
        # Execute formula
        score = eval(formula_processed, {"__builtins__": {}}, safe_dict)
        return float(score)
        
    except Exception as e:
        return None


def evaluate_ml_mechanism(
    mechanism_text: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: List[str],
    scaler: Any,
    y_scaler: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate an ML mechanism on train and test data.
    
    Note: ML mechanisms are typically stored as text descriptions.
    This function attempts to extract model information or uses a fallback.
    In practice, you may need to retrain the model or use a saved model.
    
    Args:
        mechanism_text: ML mechanism description text
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        feature_cols: List of feature names
        scaler: Feature scaler
        y_scaler: Target scaler
        
    Returns:
        Tuple of (train_predictions, test_predictions)
    """
    # For ML mechanisms, we typically need the actual model object
    # Since we only have text, we'll try to extract model type and retrain
    # This is a simplified version - in practice, you might want to save/load models
    
    # Try to detect model type from text
    model_type = "linear"
    if "xgboost" in mechanism_text.lower() or "xgb" in mechanism_text.lower():
        model_type = "xgboost"
    elif "kernel" in mechanism_text.lower() or "rbf" in mechanism_text.lower():
        model_type = "kernelridge"
    elif "linear" in mechanism_text.lower() or "regression" in mechanism_text.lower():
        model_type = "linear"
    
    # Train a model of the detected type
    from sklearn.linear_model import LinearRegression
    from sklearn.kernel_ridge import KernelRidge
    
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=42, n_estimators=100)
        except ImportError:
            model = LinearRegression()  # Fallback
    elif model_type == "kernelridge":
        model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    else:
        model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Apply inverse scaling if y_scaler is provided
    if y_scaler is not None:
        try:
            train_preds = y_scaler.inverse_transform(train_preds.reshape(-1, 1)).flatten()
            test_preds = y_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        except Exception:
            pass
    
    return train_preds, test_preds


def save_mechanism_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
    output_dir: str,
    r2: Optional[float] = None,
    mae: Optional[float] = None,
    mse: Optional[float] = None
):
    """
    Save a scatter plot comparing true vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        filename: Output filename
        output_dir: Output directory
        r2: Optional R2 score to display
        mae: Optional MAE to display
        mse: Optional MSE to display
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Add diagonal line (perfect predictions)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect predictions')
    
    # Add metrics to title or as text
    metrics_text = []
    if r2 is not None:
        metrics_text.append(f'R² = {r2:.4f}')
    if mae is not None:
        metrics_text.append(f'MAE = {mae:.4f}')
    if mse is not None:
        metrics_text.append(f'MSE = {mse:.4f}')
    
    if metrics_text:
        title_with_metrics = f"{title}\n{', '.join(metrics_text)}"
    else:
        title_with_metrics = title
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title_with_metrics, fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

