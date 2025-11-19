"""
Few-shot example retrieval for MA-ICL system.
Contains functions for retrieving and formatting few-shot examples.
"""

import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from maicl_config import RANDOM_STATE

logger = logging.getLogger(__name__)


def retrieve_few_shot_examples(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    feature_cols: List[str],
    k_shot: int,
    task_type: str = "regression",
    class_names: Optional[List[str]] = None,
    random_state: int = RANDOM_STATE,
    ml_residuals: Optional[np.ndarray] = None,
    prioritize_residuals: bool = True,
    X_pool_original: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve and format few-shot examples from the training pool.
    
    IMPORTANT: X_pool and y_pool MUST be from the training set only, not validation or test sets.
    This ensures no data leakage in few-shot examples.
    
    Args:
        X_pool: Training pool features (n_samples, n_features) - MUST be training data only
        y_pool: Training pool targets (n_samples,) - MUST be training data only
        feature_cols: List of feature column names
        k_shot: Number of few-shot examples to retrieve
        task_type: "classification" or "regression"
        class_names: List of class names for classification tasks
        random_state: Random seed for sampling
        ml_residuals: Optional ML model residuals (n_samples,) - if provided, prioritizes high-residual examples
        prioritize_residuals: If True and ml_residuals provided, sample from high-residual examples first
        X_pool_original: Optional original feature dicts (e.g., with SMILES strings for DeepChem)
    
    Returns:
        List of formatted few-shot examples, each as a dict with 'inputs' and 'output' keys
    """
    if k_shot <= 0:
        return []
    
    # Validate inputs
    X_pool = np.asarray(X_pool)
    y_pool = np.asarray(y_pool)
    
    if len(X_pool) == 0:
        logger.warning("[Few-shot] X_pool is empty, cannot retrieve few-shot examples")
        return []
    
    if len(y_pool) == 0:
        logger.warning("[Few-shot] y_pool is empty, cannot retrieve few-shot examples")
        return []
    
    if len(X_pool) != len(y_pool):
        logger.error(f"[Few-shot] X_pool ({len(X_pool)}) and y_pool ({len(y_pool)}) have mismatched lengths")
        return []
    
    if X_pool.shape[1] != len(feature_cols):
        logger.error(f"[Few-shot] X_pool has {X_pool.shape[1]} features but feature_cols has {len(feature_cols)} columns")
        return []
    
    # Smart sampling: prioritize high-residual examples if available
    n_samples = min(k_shot, len(X_pool))
    rng = np.random.RandomState(random_state)
    
    if ml_residuals is not None and prioritize_residuals and len(ml_residuals) == len(X_pool):
        # Prioritize examples where ML model fails (high residuals)
        # This helps the LLM focus on cases where ML needs correction
        try:
            ml_residuals = np.asarray(ml_residuals)
            abs_residuals = np.abs(ml_residuals)
            
            # Sort by residual magnitude (highest first)
            sorted_indices = np.argsort(abs_residuals)[::-1]
            
            # Take top 50% from high-residual examples, rest randomly from remaining
            high_residual_count = min(n_samples // 2, len(sorted_indices))
            high_residual_indices = sorted_indices[:high_residual_count]
            
            # Remaining samples: mix of medium-residual and random
            remaining_indices = sorted_indices[high_residual_count:]
            if len(remaining_indices) > 0 and n_samples > high_residual_count:
                remaining_count = n_samples - high_residual_count
                if len(remaining_indices) > remaining_count:
                    # Randomly sample from remaining
                    remaining_selected = rng.choice(
                        remaining_indices, 
                        size=remaining_count, 
                        replace=False
                    )
                else:
                    remaining_selected = remaining_indices
                
                indices = np.concatenate([high_residual_indices, remaining_selected])
            else:
                indices = high_residual_indices[:n_samples]
            
            # Shuffle to avoid always showing worst cases first (maintains diversity)
            rng.shuffle(indices)
            indices = indices[:n_samples]
            
            logger.info(f"[Few-shot] Prioritized {high_residual_count} high-residual examples (ML failures) out of {n_samples} total")
        except Exception as e:
            logger.warning(f"[Few-shot] Failed to prioritize by residuals, falling back to random sampling: {e}")
            if len(X_pool) > n_samples:
                indices = rng.choice(len(X_pool), size=n_samples, replace=False)
            else:
                indices = np.arange(len(X_pool))
    else:
        # Fallback to random sampling
        if len(X_pool) > n_samples:
            indices = rng.choice(len(X_pool), size=n_samples, replace=False)
        else:
            indices = np.arange(len(X_pool))
    
    few_shot_examples = []
    for idx in indices:
        try:
            # Use X_pool_original (non-vectorized features like SMILES) if available
            if X_pool_original is not None and idx < len(X_pool_original):
                if isinstance(X_pool_original[idx], dict):
                    # Already a dict (e.g., {'SMILES': '...'} for DeepChem)
                    x_dict = X_pool_original[idx].copy()
                else:
                    # Fallback: convert from vectorized features
                    x_dict = {feature_cols[j]: float(X_pool[idx, j]) for j in range(len(feature_cols))}
            else:
                # Fallback: convert from vectorized features
                x_dict = {feature_cols[j]: float(X_pool[idx, j]) for j in range(len(feature_cols))}
            
            # Format output based on task type
            if task_type == "classification":
                y_val = y_pool[idx]
                # Convert class index to class name if available
                if class_names is not None:
                    try:
                        y_idx = int(y_val)
                        if 0 <= y_idx < len(class_names):
                            output = class_names[y_idx]
                        else:
                            output = str(y_val)
                            logger.warning(f"[Few-shot] Class index {y_idx} out of range for class_names (len={len(class_names)})")
                    except (ValueError, TypeError):
                        output = str(y_val)
                else:
                    output = str(y_val)
            else:
                # Regression: output is numeric
                output = float(y_pool[idx])
            
            few_shot_examples.append({
                "inputs": x_dict,
                "output": output
            })
        except Exception as e:
            logger.warning(f"[Few-shot] Error formatting example at index {idx}: {e}")
            continue
    
    if len(few_shot_examples) > 0:
        logger.info(f"[Few-shot] Retrieved {len(few_shot_examples)} examples from training pool (requested: {k_shot}, available: {len(X_pool)})")
    else:
        logger.warning(f"[Few-shot] Failed to retrieve any few-shot examples (requested: {k_shot}, available: {len(X_pool)})")
    
    return few_shot_examples


def format_few_shot_for_prompt(few_shot_examples: List[Dict[str, Any]], task_type: str = "regression") -> str:
    """
    Format few-shot examples as a string for inclusion in prompts.
    
    Args:
        few_shot_examples: List of few-shot examples (from retrieve_few_shot_examples)
        task_type: "classification" or "regression"
    
    Returns:
        Formatted string with few-shot examples
    """
    if not few_shot_examples:
        return ""
    
    lines = []
    for i, ex in enumerate(few_shot_examples):
        inputs_str = json.dumps(ex["inputs"])
        output_str = str(ex["output"])
        if task_type == "classification":
            lines.append(f"Example {i+1}:\n  Inputs: {inputs_str}\n  Output (class): {output_str}")
        else:
            lines.append(f"Example {i+1}:\n  Inputs: {inputs_str}\n  Output: {output_str}")
    
    return "\n\n".join(lines)

