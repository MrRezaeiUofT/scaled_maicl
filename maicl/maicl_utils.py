"""
Utility functions for MA-ICL system.
Contains scalers, metrics, formatting, validation, and helper functions.
"""

import re
import copy
import threading
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from sklearn.metrics import f1_score
from .maicl_config import (
    SCALE_MIN, SCALE_MAX, MAX_PROMPT_PREVIEW_LENGTH,
    MAX_TOP_FEATURES, MAX_TOP_FEATURES_DISPLAY
)

logger = logging.getLogger(__name__)


# =========================
# RESIDUAL FORMATTING
# =========================

def _format_residual_line(
    feat_pairs: str,
    y_true_val,
    y_pred_val,
    task_type: str,
    class_names: Optional[List[str]] = None,
) -> str:
    """Create a single residual example line for the LLM with NO probabilities.
    - Regression: "... -> pred=..., true=..., diff=+/-..."
    - Classification: "... -> pred=<name_or_id>, true=<name_or_id>, diff=<match|mismatch>"
    """
    if task_type == "regression":
        try:
            y_true_f = float(y_true_val)
            y_pred_f = float(y_pred_val)
            diff = y_true_f - y_pred_f
            return f"{feat_pairs} -> pred={y_pred_f:.4f}, true={y_true_f:.4f}, diff={diff:+.4f}"
        except Exception:
            try:
                diff = float(y_true_val) - float(y_pred_val)
            except Exception:
                diff = 0.0
            return f"{feat_pairs} -> pred={y_pred_val}, true={y_true_val}, diff={diff:+.4f}"
    # classification
    try:
        pred_id = int(float(y_pred_val))
    except Exception:
        pred_id = int(y_pred_val)
    try:
        true_id = int(float(y_true_val))
    except Exception:
        true_id = int(y_true_val)
    pred_lbl = class_names[pred_id] if (class_names and 0 <= pred_id < len(class_names)) else str(pred_id)
    true_lbl = class_names[true_id] if (class_names and 0 <= true_id < len(class_names)) else str(true_id)
    diff_str = "match" if (pred_lbl == true_lbl) else "mismatch"
    return f"{feat_pairs} -> pred={pred_lbl}, true={true_lbl}, diff={diff_str}"


# =========================
# SCALERS
# =========================

class MinMaxScaler010:
    """Scale features to [0, 1] range"""
    def __init__(self, feature_range=(SCALE_MIN, SCALE_MAX)):
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.feature_range = feature_range
    
    def fit(self, X):
        X = np.asarray(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = np.where(
            self.max_ - self.min_ != 0,
            (self.feature_range[1] - self.feature_range[0]) / (self.max_ - self.min_),
            1.0
        )
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        X_scaled = self.scale_ * (X - self.min_) + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =========================
# TEXT PROCESSING
# =========================

def clean_json_response(s: str) -> str:
    """Clean LaTeX and special characters from LLM JSON responses"""
    replacements = {
        r'\\times': ' x ', r'\\pi': ' pi ', r'\\cdot': ' * ', r'\\beta': ' beta ',
        r'\\alpha': ' alpha ', r'\\Delta': ' Delta ', r'\\sum': ' sum ',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r'\\(?!["\\/bfnrtu])', '', s)
    return s


# =========================
# METRICS
# =========================

def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for classification"""
    if len(y_true) == 0 or len(y_prob) == 0:
        return float('inf')
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = 'f1') -> float:
    """Find optimal threshold that maximizes specified metric"""
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.5
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        try:
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'f1_balanced':
                score = f1_score(y_true, y_pred, average='weighted')
            else:
                score = f1_score(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        except Exception:
            continue
    return float(best_threshold)


# =========================
# DATA VALIDATION
# =========================

def validate_scaled_data(X: np.ndarray, feature_cols: List[str], scaler: Any = None):
    """Validate scaled data shape and range."""
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X_array.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature count mismatch: X has {X_array.shape[1]}, feature_cols has {len(feature_cols)}"
        )
    if scaler is not None and hasattr(scaler, 'feature_range'):
        expected_min, expected_max = scaler.feature_range
        if not (np.all(X_array >= expected_min - 1e-3) and np.all(X_array <= expected_max + 1e-3)):
            logger.warning(f"Data values outside expected range [{expected_min}, {expected_max}]")
    logger.info(f"Data validation passed - features={len(feature_cols)} samples={len(X_array)}")


# =========================
# FALLBACK TRACKING
# =========================

# Global fallback counter for tracking error handling
_fallback_counts = {}
_fallback_lock = threading.Lock()


def _increment_fallback(key: str, context: str = ""):
    """Increment fallback counter with optional context"""
    with _fallback_lock:
        if key not in _fallback_counts:
            _fallback_counts[key] = {"count": 0, "contexts": []}
        _fallback_counts[key]["count"] += 1
        if context and context not in _fallback_counts[key]["contexts"]:
            _fallback_counts[key]["contexts"].append(context)


def get_fallback_summary() -> Dict[str, Any]:
    """Get summary of fallback counts"""
    with _fallback_lock:
        return copy.deepcopy(_fallback_counts)


def clear_fallback_counts():
    """Clear all fallback counts"""
    with _fallback_lock:
        _fallback_counts.clear()


# =========================
# LOGGING UTILITIES
# =========================

def log_metrics_table(name: str, metrics_dict: Dict[str, Any], tablefmt: str = "github"):
    """
    Log metrics as a formatted table. Works even without tabulate by using simple formatting.
    """
    try:
        from tabulate import tabulate
        rows = [(k, f"{v:.6f}" if isinstance(v, (int, float)) else str(v)) for k, v in metrics_dict.items()]
        table = tabulate(rows, headers=["Metric", name], tablefmt=tablefmt)
        logger.info(f"\n{name} Metrics:\n{table}")
    except ImportError:
        # Fallback: simple formatted output
        logger.info(f"\n{name} Metrics:")
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k}: {v:.6f}")
            else:
                logger.info(f"  {k}: {v}")


# =========================
# DIVERSITY UTILITIES
# =========================

def _tokenize_for_diversity(text: str) -> List[str]:
    """Tokenize text for diversity checking"""
    text = re.sub(r"[^A-Za-z0-9_]+", " ", text)
    return [t for t in text.lower().split() if t]


def _ngrams(words: List[str], n: int = 3) -> set:
    """Generate n-grams from word list"""
    return set(tuple(words[i : i + n]) for i in range(0, max(0, len(words) - n + 1)))


def jaccard_distance(a: set, b: set) -> float:
    """Compute Jaccard distance between two sets (1 - Jaccard similarity)"""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 0.0)


# =========================
# MECHANISM FORMULA EXECUTION
# =========================

def _execute_mechanism_formula_programmatic(mechanism_text: str, x_dict: Dict[str, float], 
                                            task_type: str = "classification",
                                            class_names: Optional[List[str]] = None) -> Optional[float]:
    """
    Programmatically execute a mechanism formula as a fallback when LLM execution fails.
    
    This function attempts to extract and execute the formula from mechanism text.
    Returns None if execution fails (should fall back to LLM or ML).
    """
    try:
        # Extract formula from mechanism text
        # Look for patterns like "Formula:", "ŷ =", "score ="
        formula_match = re.search(r'(?:Formula|FORMULA|formula)[:\s]*(.*?)(?:\n|$|ŷ\s*=|Map|Class)', mechanism_text, re.IGNORECASE | re.DOTALL)
        if not formula_match:
            # Try to find "score = " or "ŷ = " directly
            formula_match = re.search(r'(?:score|ŷ)\s*=\s*(.*?)(?:\n|$|Map|Class|ŷ\s*=|if\s+score)', mechanism_text, re.IGNORECASE | re.DOTALL)
        
        if not formula_match:
            return None
        
        formula = formula_match.group(1).strip()
        
        # Extract threshold mapping if present (for classification)
        threshold_mapping = None
        if task_type == "classification":
            threshold_match = re.search(r'ŷ\s*=\s*0\s+if\s+score\s*<\s*([\d.]+)\s+else\s*\(1\s+if\s+score\s*<\s*([\d.]+)\s+else\s*\(2\s+if\s+score\s*<\s*([\d.]+)\s+else\s*3\)\)', mechanism_text, re.IGNORECASE)
            if threshold_match:
                threshold_mapping = [float(threshold_match.group(1)), float(threshold_match.group(2)), float(threshold_match.group(3))]
        
        # Replace feature names with their values
        # Create a safe evaluation environment
        safe_dict = {}
        for key, val in x_dict.items():
            # Sanitize key name (remove special chars, replace with underscore)
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
            safe_dict[safe_key] = float(val)
            # Also add original key if different
            if safe_key != key:
                safe_dict[key] = float(val)
        
        # Replace feature names in formula with safe names
        for orig_key, val in x_dict.items():
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(orig_key))
            # Replace both original and safe versions
            formula = formula.replace(str(orig_key), safe_key)
        
        # Add math functions to safe environment
        import math
        def clip_func(x, min_val, max_val):
            """Clip function for regression formulas"""
            return max(min_val, min(max_val, float(x)))
        
        safe_dict.update({
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pow': pow, 'clip': clip_func, '__builtins__': {}
        })
        
        # Execute formula
        try:
            score = eval(formula, {"__builtins__": {}}, safe_dict)
            score = float(score)
            
            # Apply threshold mapping for classification
            if task_type == "classification" and threshold_mapping:
                if score < threshold_mapping[0]:
                    class_idx = 0
                elif score < threshold_mapping[1]:
                    class_idx = 1
                elif score < threshold_mapping[2]:
                    class_idx = 2
                else:
                    class_idx = 3
                return float(class_idx)
            else:
                # For regression or no threshold mapping, return score
                return score
        except Exception as e:
            logger.debug(f"Formula execution failed: {e}")
            return None
    except Exception as e:
        logger.debug(f"Formula extraction failed: {e}")
        return None

