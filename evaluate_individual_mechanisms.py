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


def parse_mechanisms_file(filepath: str) -> List[Tuple[str, str]]:
    """
    Parse a mechanisms file into a list of (type, text) tuples.
    
    File format:
        [LLM] mechanism text 1
        
        [ML] mechanism text 2
        
        [KNOWN] mechanism text 3
    
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
    
    # Split by double newlines (mechanism separator)
    sections = content.split('\n\n')
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Match pattern: [TYPE] text
        match = re.match(r'\[(\w+)\]\s*(.*)', section, re.DOTALL)
        if match:
            mech_type = match.group(1).lower()
            mech_text = match.group(2).strip()
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


def evaluate_llm_formula(
    formula: str,
    X: np.ndarray,
    feature_cols: List[str],
    scaler: Any,
    y_scaler: Optional[Any] = None
) -> np.ndarray:
    """
    Evaluate an LLM formula on data.
    
    Args:
        formula: The formula string to evaluate
        X: Feature matrix (n_samples, n_features)
        feature_cols: List of feature names
        scaler: Feature scaler (for inverse transform if needed)
        y_scaler: Target scaler (for inverse transform if needed)
        
    Returns:
        Array of predictions
    """
    predictions = []
    
    for i in range(len(X)):
        # Create feature dictionary
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
    
    Args:
        formula: Formula string
        x_dict: Dictionary of feature names to values
        
    Returns:
        Computed value or None if execution fails
    """
    try:
        # Create safe evaluation environment
        safe_dict = {}
        for key, val in x_dict.items():
            # Sanitize key name (remove special chars, replace with underscore)
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
            safe_dict[safe_key] = float(val)
            # Also add original key if different
            if safe_key != key:
                safe_dict[key] = float(val)
        
        # Replace feature names in formula with safe names
        formula_safe = formula
        for orig_key in x_dict.keys():
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(orig_key))
            # Replace both original and safe versions
            formula_safe = formula_safe.replace(str(orig_key), safe_key)
        
        # Add math functions to safe environment
        safe_dict.update({
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pow': pow, '__builtins__': {}
        })
        
        # Execute formula
        score = eval(formula_safe, {"__builtins__": {}}, safe_dict)
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

