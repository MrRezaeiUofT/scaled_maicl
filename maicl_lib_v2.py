

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import copy
from collections import Counter
import threading
import logging
import warnings
from datetime import datetime
import subprocess
import sys
import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, recall_score, log_loss, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge

try:
    from sklearn.calibration import calibration_curve
    _HAS_CALIBRATION = True
except:
    _HAS_CALIBRATION = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except:
    _HAS_XGB = False

try:
    from tabicl import TabICLClassifier
    _HAS_TABICL = True
except:
    _HAS_TABICL = False

try:
    from baticl import BaticlClassifier
    _HAS_BATICL = True
except:
    _HAS_BATICL = False

try:
    from langchain.schema import HumanMessage
except Exception:
    from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except:
    _HAS_MPL = False

# Import TextGrad from separate module
from textgrad import TextGrad
# Import ML mechanism classes and functions from separate module
from ml_mechanism import MLModelMechanism, compute_ml_residuals, get_top_k_residual_samples, get_ml_prediction_probability
# Import dataset-specific information and functions from separate module
from dataset_info import (
    get_dataset_description, 
    build_known_mechanism_description, 
    compute_dataset_stats,
    get_dataset_specific_known_mechanisms
)
# Import visualization and result saving functions from separate module
from visualization import (
    plot_confusion_matrix,
    plot_performance_comparison,
    plot_scatter_predictions,
    create_result_visualizations,
    export_training_artifacts,
    export_mechanism_interpretations,
    persist_iteration_artifacts,
    _safe_write_json,
    _safe_append_jsonl,
    _safe_write_text
)

# Import from refactored modules
from maicl_config import (
    SCALE_MIN, SCALE_MAX, MAX_BATCH_SIZE, BATCH_TIMEOUT, EVAL_BATCH_SIZE, GRADIENT_BATCH_SIZE,
    LLM_GROUP_SIZE, LLM_COMBINE_MECHANISMS, K_SHOT_PER_TARGET, ATTENTION_TEMP, RANDOM_STATE,
    IMPROVEMENT_THRESHOLD_CLASSIFICATION, ML_HIGH_CONFIDENCE_THRESHOLD, ML_LOW_CONFIDENCE_THRESHOLD,
    MIN_ML_WEIGHT, MAX_ML_WEIGHT, HARD_ML_GATE_THRESHOLD, OUTPUT_ROOT, RUN_FOLDER_NAME, OUTPUT_DIR,
    MAX_TOP_FEATURES, MAX_TOP_FEATURES_DISPLAY, MAX_TOP_ERROR_FEATURES, MAX_TOP_ERROR_FEATURES_CHECK,
    MAX_FEATURE_INTERACTIONS, MAX_FEATURE_IMPORTANCE_TOP, MAX_WORST_PREDICTIONS_COLLECT,
    MAX_WORST_PREDICTIONS_DISPLAY, MAX_WORST_INDICES, MAX_WORST_RESIDUAL_SAMPLES,
    MAX_FEATURES_IN_DESCRIPTION, MAX_FEATURES_IN_EXAMPLE, MAX_FEATURES_IN_COMPONENT_LIST,
    MAX_FEATURES_IN_FORMULA_DISPLAY, MAX_PROMPT_PREVIEW_LENGTH, MAX_ERROR_MESSAGE_LENGTH,
    MAX_MECHANISM_SUMMARY_LENGTH, MAX_FEATURE_DISPLAY_LENGTH, MAX_MECHANISM_PREVIEW_LENGTH,
    MAX_INTERCEPT_DISPLAY, set_output_dir, load_maicl_config, get_prompt
)
from maicl_utils import (
    _format_residual_line, MinMaxScaler010, clean_json_response,
    compute_expected_calibration_error, find_optimal_threshold, validate_scaled_data,
    _increment_fallback, get_fallback_summary, clear_fallback_counts, log_metrics_table,
    _tokenize_for_diversity, _ngrams, jaccard_distance, _execute_mechanism_formula_programmatic
)
from maicl_llm import GoogleAPIKeyManager, BatchedLLM
from maicl_ml_knowledge import (
    extract_ml_knowledge, generate_ml_guided_mechanism, analyze_ml_failure_patterns,
    enhanced_textgrad_feedback
)
from maicl_few_shot import retrieve_few_shot_examples, format_few_shot_for_prompt

warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class MultiAgentPredictor:
    """Multi-agent predictor that aggregates predictions from multiple mechanisms"""
    def __init__(self, batched_llm: BatchedLLM, mechanisms: List[str], mechanism_types: List[str],
                 ml_mechanism: Optional[MLModelMechanism], feature_cols: List[str], scaler: Any,
                 attention_temp: float, task_type: str = "regression", class_names: Optional[List[str]] = None,
                 hard_ml_gate_threshold: Optional[float] = None,
                 min_ml_weight: Optional[float] = None,
                 max_ml_weight: Optional[float] = None):
        self.llm = batched_llm
        self.mechanisms = mechanisms
        self.mechanism_types = mechanism_types
        self.ml_mechanism = ml_mechanism
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.attention_temp = attention_temp
        self.task_type = task_type
        self.class_names = class_names
        # Allow LLM mechanisms to contribute in regression by default (disable full ML routing)
        # Set MAICL_REGRESSION_PREFER_ML=1 to force 100% ML routing
        self.prefer_ml_for_regression = os.environ.get("MAICL_REGRESSION_PREFER_ML", "0") != "0"
        # Routing overrides (if provided)
        self.hard_ml_gate_threshold = HARD_ML_GATE_THRESHOLD if hard_ml_gate_threshold is None else float(hard_ml_gate_threshold)
        self.min_ml_weight = MIN_ML_WEIGHT if min_ml_weight is None else float(min_ml_weight)
        self.max_ml_weight = MAX_ML_WEIGHT if max_ml_weight is None else float(max_ml_weight)
        self.attention_prompt = self._init_attention_prompt()
        self._setup_parsers()
        self.mechanism_performance = {}
    
    def _scaled_range_from(self, scaler):
        """Get the actual scaling range from scaler or use defaults"""
        if scaler is not None and hasattr(scaler, "feature_range"):
            return tuple(float(x) for x in scaler.feature_range)
        return (SCALE_MIN, SCALE_MAX)  # fallback
    
    def _init_attention_prompt(self) -> str:
        return get_prompt('attention.compute_weights')
    
    def _setup_parsers(self):
        """Setup simple JSON format instructions (no external parser dependency)"""
        if self.task_type == "classification":
            # Classification: always use class name strings
            allowed = ", ".join(self.class_names) if self.class_names else "class0, class1"
            y_hat_desc = f"Predicted class NAME (string; must exactly match one of: {allowed})"
            example_class = self.class_names[0] if self.class_names else "positive"
            example_class_1 = self.class_names[0] if self.class_names else "positive"
            example_class_2 = self.class_names[1] if self.class_names and len(self.class_names) > 1 else "negative"
            self.agent_format = get_prompt('prediction.format_instructions.classification',
                                          y_hat_desc=y_hat_desc,
                                          example_class=example_class,
                                          example_class_1=example_class_1,
                                          example_class_2=example_class_2)
        else:
            # Regression: use numeric scalar in scaled range
            y_hat_desc = "Predicted numeric output (float in the target's scaled range)"
            self.agent_format = get_prompt('prediction.format_instructions.regression',
                                          y_hat_desc=y_hat_desc)
        self.agent_parser = None
    
    def predict_single(self, x_dict: Dict, few_shot: List[Dict] = None) -> Tuple[float, List[float], np.ndarray, Optional[np.ndarray]]:
        """Predict for a single input with confidence-based routing"""
        if few_shot is None:
            few_shot = []
        
        # Get predictions from all mechanisms
        llm_mechanisms, llm_indices, ml_indices = [], [], []
        for i, (mech, mech_type) in enumerate(zip(self.mechanisms, self.mechanism_types)):
            if mech_type == "llm":
                llm_mechanisms.append(mech)
                llm_indices.append(i)
            elif mech_type == "ml":
                ml_indices.append(i)
        
        results = [None] * len(self.mechanisms)
        
        # LLM mechanism predictions
        if len(llm_mechanisms) > 0:
            prediction_prompts = []
            for mechanism in llm_mechanisms:
                if self.task_type == "classification":
                    # Classification: always use class names
                    allowed = ", ".join(self.class_names) if self.class_names else "class0, class1"
                    class_mapping = ", ".join([f"{i}={cn}" for i, cn in enumerate(self.class_names)]) if self.class_names else "0=class0, 1=class1"
                    
                    # Check if mechanism uses threshold mapping (common pattern)
                    has_threshold_mapping = "if" in mechanism.lower() and ("score" in mechanism.lower() or "ŷ" in mechanism.lower())
                    
                    if has_threshold_mapping:
                        task_instr = get_prompt('prediction.task_instructions.classification_with_threshold',
                                               class_mapping=class_mapping,
                                               allowed_classes=allowed)
                    else:
                        task_instr = get_prompt('prediction.task_instructions.classification_direct',
                                               class_mapping=class_mapping,
                                               allowed_classes=allowed)
                else:
                    # Regression: use actual scaling range
                    scale_min, scale_max = self._scaled_range_from(self.scaler)
                    task_instr = get_prompt('prediction.task_instructions.regression',
                                          scale_min=scale_min,
                                          scale_max=scale_max)
                
                # Build prompt with mechanism, few-shot examples, and format instructions
                prompt = f"""{task_instr}

MECHANISM:
{mechanism}

"""
                if few_shot:
                    few_shot_str = format_few_shot_for_prompt(few_shot, self.task_type)
                    prompt += f"""FEW-SHOT EXAMPLES:
{few_shot_str}

"""
                prompt += f"""INPUT:
{json.dumps(x_dict)}

{self.agent_format}

Please provide your prediction in the format specified above."""
                
                prediction_prompts.append(HumanMessage(content=prompt))
            
            # Batch invoke LLM
            llm_responses = self.llm.invoke_batch(prediction_prompts)
            
            # Parse LLM responses
            for i, (response, mech_idx) in enumerate(zip(llm_responses, llm_indices)):
                try:
                    # Try to parse JSON response
                    parsed = clean_json_response(response)
                    if parsed and "y_hat" in parsed:
                        results[mech_idx] = parsed["y_hat"]
                    else:
                        # Fallback: try to extract number or class name from text
                        if self.task_type == "classification":
                            # Look for class name in response
                            for class_name in (self.class_names or []):
                                if class_name.lower() in response.lower():
                                    results[mech_idx] = class_name
                                    break
                            if results[mech_idx] is None:
                                # Try to extract class index
                                match = re.search(r'\b([0-9]+)\b', response)
                                if match:
                                    idx = int(match.group(1))
                                    if self.class_names and 0 <= idx < len(self.class_names):
                                        results[mech_idx] = self.class_names[idx]
                        else:
                            # Regression: extract number
                            match = re.search(r'[-+]?\d*\.?\d+', response)
                            if match:
                                results[mech_idx] = float(match.group(0))
                except Exception as e:
                    logger.warning(f"Failed to parse LLM response for mechanism {mech_idx}: {e}")
                    _increment_fallback("llm_parse_error", f"mechanism_{mech_idx}")
        
        # ML mechanism predictions
        if len(ml_indices) > 0 and self.ml_mechanism is not None and self.ml_mechanism.is_trained:
            try:
                # Convert x_dict to feature vector
                x_array = np.array([[x_dict.get(feat, 0.0) for feat in self.feature_cols]])
                
                # Scale if scaler is available
                if self.scaler is not None:
                    x_array = self.scaler.transform(x_array)
                
                # Get ML prediction
                ml_pred = self.ml_mechanism.predict(x_array)[0]
                
                # Store for all ML mechanism indices
                for ml_idx in ml_indices:
                    results[ml_idx] = float(ml_pred)
            except Exception as e:
                logger.warning(f"Failed to get ML prediction: {e}")
                _increment_fallback("ml_prediction_error")
        
        # Convert results to proper format
        predictions = []
        for i, result in enumerate(results):
            if result is None:
                # Fallback: use default value
                if self.task_type == "classification":
                    predictions.append(self.class_names[0] if self.class_names else 0)
                else:
                    predictions.append(0.0)
                _increment_fallback("prediction_fallback", f"mechanism_{i}")
            else:
                predictions.append(result)
        
        # Compute attention weights
        weights = self._compute_attention_weights(predictions, x_dict)
        
        # Aggregate predictions
        if self.task_type == "classification":
            # Classification: weighted voting
            class_votes = {}
            for pred, weight in zip(predictions, weights):
                if pred not in class_votes:
                    class_votes[pred] = 0.0
                class_votes[pred] += weight
            
            # Return class with highest vote
            final_pred = max(class_votes.items(), key=lambda x: x[1])[0]
            
            # Convert class name to index if needed
            if isinstance(final_pred, str) and self.class_names:
                final_pred_idx = self.class_names.index(final_pred)
            else:
                final_pred_idx = int(final_pred) if isinstance(final_pred, (int, float)) else 0
            
            return final_pred_idx, predictions, weights, None
        else:
            # Regression: weighted average
            final_pred = sum(p * w for p, w in zip(predictions, weights))
            return float(final_pred), predictions, weights, None
    
    def predict_batch(self, x_dicts: List[Dict], few_shot_list: Optional[List[List[Dict]]] = None):
        """Batch prediction across multiple samples (batches by mechanism for efficiency)"""
        if few_shot_list is None:
            few_shot_list = [[]] * len(x_dicts)
        
        if len(x_dicts) != len(few_shot_list):
            raise ValueError("x_dicts and few_shot_list must have same length")
        
        if len(x_dicts) == 0:
            return []
        
        # Get LLM mechanisms
        llm_mechanisms, llm_indices, ml_indices = [], [], []
        for i, (mech, mech_type) in enumerate(zip(self.mechanisms, self.mechanism_types)):
            if mech_type == "llm":
                llm_mechanisms.append(mech)
                llm_indices.append(i)
            elif mech_type == "ml":
                ml_indices.append(i)
        
        # Prepare task instruction
        if self.task_type == "classification":
            # Classification: always use class names
            allowed = ", ".join(self.class_names) if self.class_names else "class0, class1"
            task_instr = get_prompt('prediction.task_instructions.classification_batch',
                                    allowed_classes=allowed)
        else:
            # Regression: use actual scaling range
            ymin, ymax = self._scaled_range_from(self.scaler)
            task_instr = get_prompt('prediction.task_instructions.regression_batch',
                                    ymin=ymin, ymax=ymax)
        
        # Grouped predictions: combine mechanisms per input group to reduce calls by factor M
        all_mechanism_responses: Dict[int, List[Tuple[float, float]]] = {idx: [None] * len(x_dicts) for idx in llm_indices}
        group_size = max(1, min(LLM_GROUP_SIZE, len(x_dicts)))
        if LLM_COMBINE_MECHANISMS and len(llm_mechanisms) > 0:
            mech_block = "\n".join([f"{i+1}) {mech}" for i, mech in enumerate(llm_mechanisms)])
            # Get few-shot examples from first sample (all samples use same few-shot examples)
            few_shot_examples = few_shot_list[0] if few_shot_list and len(few_shot_list) > 0 and len(few_shot_list[0]) > 0 else []
            few_shot_text = format_few_shot_for_prompt(few_shot_examples, task_type=self.task_type) if few_shot_examples else ""
            few_shot_section = get_prompt('few_shot.section_header', few_shot_text=few_shot_text) if few_shot_text else ""
            
            for start in range(0, len(x_dicts), group_size):
                end = min(start + group_size, len(x_dicts))
                inputs_chunk = x_dicts[start:end]
                payload = json.dumps(inputs_chunk)
                prompt = get_prompt('prediction.batch_mechanisms',
                                    num_mechanisms=len(llm_mechanisms),
                                    mechanisms_block=mech_block,
                                    few_shot_section=few_shot_section,
                                    num_inputs=len(inputs_chunk),
                                    inputs_payload=payload,
                                    task_instruction=task_instr)
                resp = self.llm.invoke_single(HumanMessage(content=prompt))
                try:
                    cleaned = clean_json_response(resp)
                    m_obj = re.search(r'\{.*\}', cleaned, re.S)
                    obj = json.loads(m_obj.group(0)) if m_obj else {}
                    results = obj.get("results", [])
                    # Normalize shape and fill defaults
                    if not isinstance(results, list):
                        results = []
                    if len(results) != (end - start):
                        # best effort: truncate/pad rows
                        if len(results) > (end - start):
                            results = results[: (end - start)]
                        else:
                            results = results + ([[]] * ((end - start) - len(results)))
                    for row_idx in range(end - start):
                        row = results[row_idx] if isinstance(results[row_idx], list) else []
                        if len(row) != len(llm_mechanisms):
                            if len(row) > len(llm_mechanisms):
                                row = row[: len(llm_mechanisms)]
                            else:
                                row = row + ([{}] * (len(llm_mechanisms) - len(row)))
                        for mcol, obj_ in enumerate(row):
                            y_hat_raw = obj_.get("y_hat", 0.0) if isinstance(obj_, dict) else 0.0
                            conf = float(obj_.get("confidence", 0.1)) if isinstance(obj_, dict) else 0.1
                            if self.task_type == "classification":
                                # Parse class name string and map to index
                                try:
                                    label = str(y_hat_raw).strip()
                                    class_names_list = self.class_names if self.class_names else []
                                    
                                    # Try exact match first
                                    if label in class_names_list:
                                        class_idx = class_names_list.index(label)
                                    else:
                                        # Fallback: try case-insensitive match
                                        low = [c.lower() for c in class_names_list]
                                        if label.lower() in low:
                                            class_idx = low.index(label.lower())
                                        else:
                                            # Last resort: try parsing as index if it's numeric
                                            try:
                                                class_idx = int(np.round(float(label)))
                                                class_idx = max(0, min(class_idx, len(class_names_list) - 1))
                                            except:
                                                class_idx = 0
                                    
                                    y_hat = float(class_idx)
                                except:
                                    y_hat = 0.0
                                    conf = min(conf, 2.0)
                            else:
                                # Regression: parse numeric and clip to actual range
                                ymin, ymax = self._scaled_range_from(self.scaler)
                                try:
                                    y_hat = float(y_hat_raw)
                                    y_hat = float(np.clip(y_hat, ymin, ymax))
                                    # SAFETY: If prediction is NaN or extreme, mark for ML fallback
                                    if not np.isfinite(y_hat) or abs(y_hat - (ymin + ymax) / 2) > (ymax - ymin) * 0.49:
                                        # Will use ML fallback later if available
                                        pass
                                except:
                                    y_hat = (ymin + ymax) / 2
                            conf = max(0.1, min(10.0, conf))
                            mech_idx = llm_indices[mcol]
                            all_mechanism_responses[mech_idx][start + row_idx] = (y_hat, conf)
                except:
                    for idx_fill in range(start, end):
                        for mcol in range(len(llm_mechanisms)):
                            mech_idx = llm_indices[mcol]
                            all_mechanism_responses[mech_idx][idx_fill] = (0.0, 0.1)
        else:
            # Fallback: per-mechanism grouped calls (already reduced vs per-sample)
            # Get few-shot examples from first sample (all samples use same few-shot examples)
            few_shot_examples = few_shot_list[0] if few_shot_list and len(few_shot_list) > 0 and len(few_shot_list[0]) > 0 else []
            few_shot_text = format_few_shot_for_prompt(few_shot_examples, task_type=self.task_type) if few_shot_examples else ""
            few_shot_section = get_prompt('few_shot.section_header', few_shot_text=few_shot_text) if few_shot_text else ""
            
            for mech_list_idx, mechanism in enumerate(llm_mechanisms):
                mech_idx = llm_indices[mech_list_idx]
                chunk_prompts = []
                chunk_ranges: List[Tuple[int, int]] = []
                for start in range(0, len(x_dicts), group_size):
                    end = min(start + group_size, len(x_dicts))
                    inputs_chunk = x_dicts[start:end]
                    payload = json.dumps(inputs_chunk)
                    prompt = get_prompt('prediction.batch_single_mechanism',
                                        mechanism=mechanism,
                                        few_shot_section=few_shot_section,
                                        num_inputs=len(inputs_chunk),
                                        inputs_payload=payload,
                                        task_instruction=task_instr,
                                        agent_format=self.agent_format)
                    chunk_prompts.append(HumanMessage(content=prompt))
                    chunk_ranges.append((start, end))
                chunk_responses = self.llm.invoke_batch(chunk_prompts)
                for (start, end), response in zip(chunk_ranges, chunk_responses):
                    try:
                        cleaned = clean_json_response(response)
                        parsed = None
                        m_arr = re.search(r'\[.*\]', cleaned, re.S)
                        if m_arr:
                            try:
                                parsed = json.loads(m_arr.group(0))
                            except Exception:
                                parsed = None
                        if parsed is None:
                            m_obj = re.search(r'\{.*\}', cleaned, re.S)
                            if m_obj:
                                try:
                                    single = json.loads(m_obj.group(0))
                                    parsed = [single] * (end - start)
                                except Exception:
                                    parsed = [{}] * (end - start)
                            else:
                                parsed = [{}] * (end - start)
                        if isinstance(parsed, list) and len(parsed) != (end - start):
                            if len(parsed) > (end - start):
                                parsed = parsed[: (end - start)]
                            else:
                                parsed = parsed + ([{}] * ((end - start) - len(parsed)))
                        for offset, obj in enumerate(parsed):
                            y_hat_raw = obj.get("y_hat", 0.0) if isinstance(obj, dict) else 0.0
                            conf = float(obj.get("confidence", 0.1)) if isinstance(obj, dict) else 0.1
                            if self.task_type == "classification":
                                # Parse class name string and map to index
                                try:
                                    label = str(y_hat_raw).strip()
                                    class_names_list = self.class_names if self.class_names else []
                                    
                                    # Try exact match first
                                    if label in class_names_list:
                                        class_idx = class_names_list.index(label)
                                    else:
                                        # Fallback: try case-insensitive match
                                        low = [c.lower() for c in class_names_list]
                                        if label.lower() in low:
                                            class_idx = low.index(label.lower())
                                        else:
                                            # Last resort: try parsing as index if it's numeric
                                            try:
                                                class_idx = int(np.round(float(label)))
                                                class_idx = max(0, min(class_idx, len(class_names_list) - 1))
                                            except:
                                                class_idx = 0
                                    
                                    y_hat = float(class_idx)
                                except:
                                    y_hat = 0.0
                                    conf = min(conf, 2.0)
                            else:
                                # Regression: parse numeric and clip to actual range
                                ymin, ymax = self._scaled_range_from(self.scaler)
                                try:
                                    y_hat = float(y_hat_raw)
                                    y_hat = float(np.clip(y_hat, ymin, ymax))
                                    # SAFETY: If prediction is NaN or extreme, mark for ML fallback
                                    if not np.isfinite(y_hat) or abs(y_hat - (ymin + ymax) / 2) > (ymax - ymin) * 0.49:
                                        # Will use ML fallback later if available
                                        pass
                                except:
                                    y_hat = (ymin + ymax) / 2
                            conf = max(0.1, min(10.0, conf))
                            all_mechanism_responses[mech_idx][start + offset] = (y_hat, conf)
                    except:
                        for idx_fill in range(start, end):
                            all_mechanism_responses[mech_idx][idx_fill] = (0.0, 0.1)
        
        # Process responses and aggregate predictions
        batch_results = []
        for sample_idx in range(len(x_dicts)):
            x_dict = x_dicts[sample_idx]
            results = [None] * len(self.mechanisms)
            
            # Parse LLM responses
            for mech_idx in llm_indices:
                y_hat, conf = all_mechanism_responses.get(mech_idx, [None] * len(x_dicts))[sample_idx] or (0.0, 0.1)
                results[mech_idx] = (y_hat, conf)
            
            # ML mechanism predictions
            for idx in ml_indices:
                if self.ml_mechanism is not None and self.ml_mechanism.is_trained:
                    try:
                        return_class_idx = (self.task_type == "classification" and 
                                          self.class_names is not None and 
                                          len(self.class_names) > 2)
                        y_hat = self.ml_mechanism.predict(x_dict, return_class_index=return_class_idx)
                        # For regression, clip to actual scaling range
                        if self.task_type == "regression":
                            ymin, ymax = self._scaled_range_from(self.scaler)
                            y_hat = float(np.clip(y_hat, ymin, ymax))
                        if hasattr(self.ml_mechanism, 'get_confidence_heuristic_with_boost'):
                            conf = self.ml_mechanism.get_confidence_heuristic_with_boost(x_dict)
                        else:
                            conf = self.ml_mechanism.get_confidence_heuristic(x_dict)
                        results[idx] = (y_hat, conf)
                    except:
                        if self.task_type == "regression":
                            ymin, ymax = self._scaled_range_from(self.scaler)
                            results[idx] = ((ymin + ymax) / 2, 0.1)
                        else:
                            results[idx] = (0.0, 0.1)
                else:
                    if self.task_type == "regression":
                        ymin, ymax = self._scaled_range_from(self.scaler)
                        results[idx] = ((ymin + ymax) / 2, 0.1)
                    else:
                        results[idx] = (0.0, 0.1)
            
            # Aggregate with confidence-based routing (same as predict_single)
            agent_preds = [r[0] if r is not None else 0.0 for r in results]
            confidences = [r[1] if r is not None else 0.1 for r in results]
            
            ml_indices_list = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "ml"]
            ml_confidence = 0.5
            
            if len(ml_indices_list) > 0 and self.ml_mechanism is not None and self.ml_mechanism.is_trained:
                ml_idx = ml_indices_list[0]
                ml_pred = agent_preds[ml_idx]
                ml_predicted_class, ml_max_prob, ml_proba = get_ml_prediction_probability(
                    self.ml_mechanism, x_dict, predicted_class=int(ml_pred) if ml_pred is not None else None, 
                    scaler=self.scaler
                )
                if ml_max_prob is not None:
                    ml_confidence = ml_max_prob
            
            # Compute weights
            if self.task_type == "regression" and self.prefer_ml_for_regression and len(ml_indices_list) > 0:
                weights = np.zeros(len(results))
                for ml_idx in ml_indices_list:
                    weights[ml_idx] = 1.0 / len(ml_indices_list)
                weights = weights / np.sum(weights)
            elif hasattr(self, 'mechanism_performance') and self.mechanism_performance:
                # Use performance-based weighting with temperature scaling for better discrimination
                perf_scores = [self.mechanism_performance.get(i, 1.0) for i in range(len(results))]
                perf_array = np.array(perf_scores)
                
                # Check if ML mechanism is significantly better
                ml_indices_list = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "ml"]
                llm_indices_list = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "llm"]
                
                if len(ml_indices_list) > 0 and len(llm_indices_list) > 0:
                    ml_perf = max([perf_scores[i] for i in ml_indices_list])
                    llm_perf = max([perf_scores[i] for i in llm_indices_list])
                    perf_ratio = ml_perf / (llm_perf + 1e-6)
                    
                    # If ML is significantly better, use higher temperature for sharper weighting
                    if perf_ratio > 1.2:
                        # High temperature: exp(score / temp) with temp < 1 makes differences more pronounced
                        temperature = 0.5  # Lower temp = sharper differences
                        perf_array = np.exp(perf_array / temperature - np.max(perf_array / temperature))
                    else:
                        # Similar performance: use standard softmax
                        perf_array = np.exp(perf_array - np.max(perf_array))
                else:
                    # Standard softmax
                    perf_array = np.exp(perf_array - np.max(perf_array))
                
                weights = perf_array / np.sum(perf_array)
            else:
                ml_indices_list = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "ml"]
                if len(ml_indices_list) > 0 and self.ml_mechanism is not None and self.ml_mechanism.is_trained:
                    num_llm = len(results) - len(ml_indices_list)
                    if ml_confidence >= self.hard_ml_gate_threshold:
                        weights = np.zeros(len(results))
                        for ml_idx in ml_indices_list:
                            weights[ml_idx] = 1.0 / len(ml_indices_list)
                        weights = weights / np.sum(weights)
                    else:
                        if ml_confidence >= ML_HIGH_CONFIDENCE_THRESHOLD:
                            ml_weight = self.max_ml_weight
                        elif ml_confidence <= ML_LOW_CONFIDENCE_THRESHOLD:
                            ml_weight = self.min_ml_weight
                        else:
                            confidence_range = ML_HIGH_CONFIDENCE_THRESHOLD - ML_LOW_CONFIDENCE_THRESHOLD
                            confidence_delta = ml_confidence - ML_LOW_CONFIDENCE_THRESHOLD
                            ml_weight = self.min_ml_weight + (confidence_delta / confidence_range) * (self.max_ml_weight - self.min_ml_weight)
                        if num_llm > 0:
                            llm_weight_total = 1.0 - ml_weight
                            weights = np.ones(len(results)) * llm_weight_total / num_llm
                        else:
                            weights = np.ones(len(results)) / len(results)
                        for ml_idx in ml_indices_list:
                            weights[ml_idx] = ml_weight / len(ml_indices_list)
                        weights = weights / np.sum(weights)
                else:
                    weights = np.ones(len(results)) / len(results)
            
            # Weighted prediction
            if self.task_type == "classification" and self.class_names is not None and len(self.class_names) > 2:
                # Multi-class: per-class score accumulation (soft vote)
                # Each mechanism contributes its weight to the predicted class's score
                num_classes = len(self.class_names)
                class_scores = np.zeros(num_classes, dtype=float)
                for w, pred in zip(weights, agent_preds):
                    try:
                        pred_val = float(pred)
                        # If prediction is already a valid class index (integer in [0, num_classes-1])
                        if 0 <= pred_val < num_classes and abs(pred_val - round(pred_val)) < 0.1:
                            cls = int(round(pred_val))
                            class_scores[cls] += float(w)
                        else:
                            # Continuous value: map to class using thresholds
                            # Divide [0, 10] range into num_classes bins
                            # This handles mechanisms that output continuous scores
                            cls = int(np.clip(np.round(pred_val * (num_classes - 1) / 10.0), 0, num_classes - 1))
                            class_scores[cls] += float(w)
                    except Exception:
                        pass
                final_class = int(np.argmax(class_scores))
                final_conf = float(class_scores[final_class] / (class_scores.sum() + 1e-12))
                y_final = float(final_class)
                class_probs = class_scores / (class_scores.sum() + 1e-12) if class_scores.sum() > 0 else np.ones(num_classes) / num_classes
            else:
                y_final = float(np.sum(weights * np.array(agent_preds)))
                if self.task_type == "classification" and (self.class_names is None or len(self.class_names) == 2):
                    y_final = float(np.clip(y_final, 0.0, 1.0))
                if self.task_type == "classification" and len(self.class_names) == 2:
                    class_probs = np.array([1.0 - y_final, y_final])
                else:
                    class_probs = None
            
            batch_results.append((y_final, agent_preds, weights, class_probs))
        
        return batch_results
    
    def _compute_attention_weights(self, predictions: List[Any], x_dict: Dict) -> np.ndarray:
        """Compute attention weights for mechanisms based on confidence"""
        # Simple uniform weights for now (can be enhanced with confidence-based routing)
        n_mechanisms = len(predictions)
        weights = np.ones(n_mechanisms) / n_mechanisms
        
        # Apply ML confidence-based routing if enabled
        if self.ml_mechanism is not None and self.ml_mechanism.is_trained:
            try:
                x_array = np.array([[x_dict.get(feat, 0.0) for feat in self.feature_cols]])
                if self.scaler is not None:
                    x_array = self.scaler.transform(x_array)
                
                ml_confidence = self.ml_mechanism.get_confidence(x_array)[0]
                
                # Find ML mechanism indices
                ml_indices = [i for i, mech_type in enumerate(self.mechanism_types) if mech_type == "ml"]
                llm_indices = [i for i, mech_type in enumerate(self.mechanism_types) if mech_type == "llm"]
                
                if ml_indices and llm_indices:
                    if ml_confidence >= self.hard_ml_gate_threshold:
                        # High confidence: prefer ML
                        ml_weight = min(self.max_ml_weight, ml_confidence)
                        llm_weight = (1.0 - ml_weight) / len(llm_indices)
                        
                        for ml_idx in ml_indices:
                            weights[ml_idx] = ml_weight / len(ml_indices)
                        for llm_idx in llm_indices:
                            weights[llm_idx] = llm_weight
                    else:
                        # Low confidence: allow LLM to contribute
                        ml_weight = max(self.min_ml_weight, ml_confidence * 0.5)
                        llm_weight = (1.0 - ml_weight) / len(llm_indices)
                        
                        for ml_idx in ml_indices:
                            weights[ml_idx] = ml_weight / len(ml_indices)
                        for llm_idx in llm_indices:
                            weights[llm_idx] = llm_weight
            except Exception as e:
                logger.warning(f"Failed to compute ML confidence-based weights: {e}")
        
        # Normalize weights
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        return weights
    
    def update_mechanism_performance(self, mechanism_idx: int, performance_score: float):
        """Update performance score for a mechanism"""
        self.mechanism_performance[mechanism_idx] = performance_score



class VariationalMechanismGenerator:
    """Generator for discovering and refining mechanisms with residual-based learning"""
    def __init__(self, batched_llm: BatchedLLM, feature_cols: List[str], use_ml_mechanism: bool = False,
                 dataset_name: str = "Dataset", task_type: str = "regression",
                 class_names: List[str] = None, pretrained_ml_mechanism: Optional[Any] = None,
                 data_insights: Optional[Dict[str, Any]] = None,
                 diversity_ngram_n: int = 3, diversity_min_jaccard: float = 0.35,
                 scaler: Any = None, num_mechanisms_unknown: Optional[int] = None):
        self.llm = batched_llm
        self.feature_cols = feature_cols
        self.use_ml_mechanism = use_ml_mechanism
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.class_names = class_names
        self.data_insights = data_insights or {}
        self.scaler = scaler
        # Number of unknown mechanisms to generate (defaults to 1 if not provided)
        self.num_mechanisms_unknown = num_mechanisms_unknown if num_mechanisms_unknown is not None else 1
        self.known_mechanisms = self._init_known_mechanisms()
        self.unknown_mechanisms = self._init_unknown_mechanisms()
        self.encoder_prompt = self._init_encoder_prompt()
        self.decoder_prompt = get_prompt('mechanism_generation.decoder_default')
        self.ml_mechanism = None
        self.predictor = None  # Will be set by TrainableMAICL
        # Diversity checking parameters
        self.diversity_ngram_n = diversity_ngram_n
        self.diversity_min_jaccard = diversity_min_jaccard
        
        if use_ml_mechanism:
            if pretrained_ml_mechanism is not None:
                self.ml_mechanism = pretrained_ml_mechanism
            else:
                model_name = "LinearRegression" if task_type == "regression" else "LogisticRegression"
                self.ml_mechanism = MLModelMechanism(model_name, task_type=self.task_type)
    
    def _is_diverse(self, candidate: str, accepted: List[str]) -> bool:
        """Check if candidate mechanism is diverse enough from accepted mechanisms using Jaccard distance"""
        def enc(s: str):
            return _ngrams(_tokenize_for_diversity(s), n=self.diversity_ngram_n)
        cand = enc(candidate)
        for prev in accepted:
            prev_enc = enc(prev)
            # Jaccard distance: if too similar (distance < threshold), reject
            jaccard_dist = jaccard_distance(cand, prev_enc)
            if jaccard_dist < self.diversity_min_jaccard:
                return False
        return True
    
    def _init_known_mechanisms(self) -> List[str]:
        """Initialize known mechanisms with DATASET-SPECIFIC EXPLICIT FORMULAS using actual feature names"""
        # Get actual scaling range from scaler
        scale_min, scale_max = self._get_scaling_range()
        
        # Get X_train_original for SMILES detection
        X_train_original = getattr(self, 'X_train_original', None)
        predictor = getattr(self, 'predictor', None)
        
        # Use the dataset-specific function from dataset_info module
        return get_dataset_specific_known_mechanisms(
            dataset_name=self.dataset_name,
            feature_cols=self.feature_cols,
            task_type=self.task_type,
            class_names=self.class_names,
            scale_min=scale_min,
            scale_max=scale_max,
            X_train_original=X_train_original,
            predictor=predictor
        )
    
    def _init_unknown_mechanisms(self) -> List[str]:
        """Initialize unknown mechanisms (empty - will be generated during training with ML-guided init if available)"""
        # Note: ML-guided initialization happens in generate_unknown_mechanisms() 
        # after ML mechanism is trained, not here during __init__
        return []
    
    def _init_encoder_prompt(self) -> str:
        """Initialize encoder prompt"""
        dataset_name_clean = self.dataset_name.lower().replace(" dataset", "").strip()
        
        if self.task_type == "classification":
            return get_prompt('mechanism_generation.encoder.classification',
                             dataset_name=dataset_name_clean.upper())
        
        return get_prompt('mechanism_generation.encoder.regression')
    
    def train_ml_mechanism(self, X_train: np.ndarray, y_train: np.ndarray, y_scaler: Any = None):
        """Train the ML mechanism"""
        if self.ml_mechanism is not None:
            self.ml_mechanism.train(X_train, y_train, self.feature_cols, y_scaler)
    
    def encode_latent_space(self, X_sample, y_sample, prediction_errors, ml_residuals: Optional[np.ndarray] = None) -> str:
        """Encode dataset patterns into latent representation"""
        stats = compute_dataset_stats(X_sample, y_sample)
        
        # Check if this is a DeepChem dataset - don't show ECFP features
        is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                      all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
        has_smiles = False
        if is_deepchem:
            X_original_check = None
            if hasattr(self, 'X_train_original') and self.X_train_original is not None:
                X_original_check = self.X_train_original
            elif hasattr(self, 'predictor') and hasattr(self.predictor, 'X_train_original'):
                X_original_check = self.predictor.X_train_original
            
            if X_original_check and len(X_original_check) > 0:
                if isinstance(X_original_check[0], dict) and 'SMILES' in X_original_check[0]:
                    has_smiles = True
        
        if is_deepchem and has_smiles:
            features_desc = "SMILES strings (molecular structures) - ML model uses ECFP fingerprints internally"
        else:
            features_desc = str(self.feature_cols) if self.feature_cols else "features"
        
        data_summary = f"""Dataset statistics:
- Features: {features_desc}
- X mean: {stats['X_mean'].tolist()}
- Sample size: {len(X_sample)}
- Prediction errors (MAE): {np.mean(prediction_errors):.3f}"""
        
        if self.use_ml_mechanism and self.ml_mechanism is not None and self.ml_mechanism.is_trained:
            ml_mech_name = self.ml_mechanism.model_name if hasattr(self.ml_mechanism, 'model_name') else "ML"
            # Include detailed ML mechanism description
            ml_mech_description = self.ml_mechanism.get_description()
            data_summary += f"\n- ML baseline mechanism: {ml_mech_description}"
        
        if ml_residuals is not None and len(ml_residuals) > 0:
            data_summary += f"\n- ML baseline mean|residual|: {float(np.mean(np.abs(ml_residuals))):.3f}"
        
        if self.task_type == "classification":
            task_desc = "Task: Infer a latent mechanism for classification explaining decision boundaries and class separation."
        else:
            task_desc = "Task: Infer a latent mechanism explaining smooth curved tendencies. IMPORTANT: Data is normalized to [0, 1] range for both inputs and outputs."
        
        prompt = get_prompt('mechanism_generation.encoder_with_data',
                           encoder_prompt=self.encoder_prompt,
                           data_summary=data_summary,
                           task_description=task_desc)
        latent_z = self.llm.invoke_single(HumanMessage(content=prompt))
        return latent_z
    
    def generate_unknown_mechanisms(self, X_train: np.ndarray, y_train: np.ndarray, prediction_errors: np.ndarray, ml_residuals: Optional[np.ndarray] = None) -> List[str]:
        """Generate new unknown mechanisms based on data and residuals"""
        # Check if this is a DeepChem dataset (ECFP fingerprints) - skip ML-guided init for these
        is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                      all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
        has_smiles = False
        if is_deepchem:
            # Check if we have SMILES strings available
            X_original_check = None
            if hasattr(self, 'X_train_original') and self.X_train_original is not None:
                X_original_check = self.X_train_original
            elif hasattr(self, 'predictor') and hasattr(self.predictor, 'X_train_original'):
                X_original_check = self.predictor.X_train_original
            
            if X_original_check and len(X_original_check) > 0:
                if isinstance(X_original_check[0], dict) and 'SMILES' in X_original_check[0]:
                    has_smiles = True
        
        # If this is the first generation and ML mechanism is trained, use ML-guided initialization
        # BUT skip ML-guided init for DeepChem datasets (they use ECFP features, not SMILES)
        if (len(self.unknown_mechanisms) == 0 and 
            self.use_ml_mechanism and 
            self.ml_mechanism is not None and 
            hasattr(self.ml_mechanism, 'is_trained') and 
            self.ml_mechanism.is_trained and
            not (is_deepchem and has_smiles)):  # Skip ML-guided init for DeepChem with SMILES
            
            logger.info("  [ML-Guided Init] Using ML-guided initialization for first mechanism generation...")
            
            # Log sample residuals for transparency (shows what the LLM sees during mechanism generation)
            # Use SMILES strings (X_original) for LLM, not vectorized features
            if ml_residuals is not None and len(ml_residuals) > 0:
                try:
                    # For classification: residuals are 1.0 (mismatched) or 0.0 (matched)
                    # Sort descending to put mismatched (1.0) first
                    sorted_indices = np.argsort(ml_residuals)[::-1]  # Sort by residual descending (1.0 first, then 0.0)
                    top_5_idx = sorted_indices[:MAX_WORST_RESIDUAL_SAMPLES]
                    
                    # Check if we have any mismatched examples (residual > 0)
                    mismatched_count = np.sum(ml_residuals > 0)
                    if mismatched_count == 0:
                        logger.warning(f"  [Residual Samples] ⚠️  No mismatched examples found (ML model has 100% accuracy on training set). Showing top samples by residual (all are 0.0):")
                    else:
                        logger.info(f"  [Residual Samples] Top {min(MAX_WORST_RESIDUAL_SAMPLES, mismatched_count)} worst ML residuals (mismatched examples) passed to LLM during mechanism generation:")
                    # Try to get X_train_original (SMILES) from parent TrainableMAICL if available
                    X_original = None
                    if hasattr(self, 'X_train_original') and self.X_train_original is not None:
                        X_original = self.X_train_original
                    elif hasattr(self, 'predictor') and hasattr(self.predictor, 'X_train_original'):
                        X_original = self.predictor.X_train_original
                    
                    for rank, idx in enumerate(top_5_idx, 1):
                        if idx < len(X_train) and idx < len(y_train):
                            # Use SMILES string if available, otherwise fall back to feature names
                            if X_original is not None and idx < len(X_original):
                                original_feat = X_original[idx]
                                if isinstance(original_feat, dict):
                                    # Extract SMILES or other original features
                                    if 'SMILES' in original_feat:
                                        feat_display = f"SMILES={original_feat['SMILES']}"
                                    elif 'SUBSTRATES' in original_feat:
                                        feat_display = f"SUBSTRATES={original_feat['SUBSTRATES']}"
                                    else:
                                        # Use first few key-value pairs
                                        key_vals = list(original_feat.items())[:MAX_FEATURES_IN_COMPONENT_LIST]
                                        feat_display = ', '.join([f"{k}={v}" for k, v in key_vals])
                                else:
                                    feat_display = f"original_feat={str(original_feat)[:MAX_FEATURE_DISPLAY_LENGTH]}"
                            else:
                                # Fallback: use feature names (vectorized)
                                feature_names = self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]
                                feat_vals = {feature_names[j]: float(X_train[idx, j]) for j in range(min(len(feature_names), X_train.shape[1]))}
                                feat_display = ', '.join([f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_COMPONENT_LIST]])
                            
                            if self.task_type == "regression":
                                y_pred = ml_residuals[idx] + y_train[idx] if ml_residuals is not None else y_train[idx]
                                logger.info(f"    {rank}. Sample {idx}: {feat_display} → pred={y_pred:.2f}, true={y_train[idx]:.2f}, residual={ml_residuals[idx]:+.2f}")
                            else:
                                # Classification: show predicted vs true, and residual (1.0=mismatched, 0.0=matched)
                                true_idx = int(y_train[idx])
                                residual_val = ml_residuals[idx]
                                is_mismatched = residual_val > 0.5  # For classification, residual is 1.0 or 0.0
                                
                                # Try to get prediction if available
                                pred_display = ""
                                if hasattr(self, 'ml_mechanism') and self.ml_mechanism is not None:
                                    try:
                                        x_dict = {self.feature_cols[j]: float(X_train[idx, j]) for j in range(len(self.feature_cols))}
                                        pred_idx = self.ml_mechanism.predict(x_dict, return_class_index=True)
                                        class_names = getattr(self, 'class_names', None)
                                        if class_names and pred_idx < len(class_names) and true_idx < len(class_names):
                                            pred_display = f"pred={class_names[pred_idx]}, "
                                        else:
                                            pred_display = f"pred={pred_idx}, "
                                    except:
                                        pass
                                
                                class_names = getattr(self, 'class_names', None)
                                match_status = "mismatched" if is_mismatched else "matched"
                                if class_names and true_idx < len(class_names):
                                    logger.info(f"    {rank}. Sample {idx}: {feat_display} → {pred_display}true={class_names[true_idx]}, residual={residual_val:.2f} ({match_status})")
                                else:
                                    logger.info(f"    {rank}. Sample {idx}: {feat_display} → {pred_display}true={true_idx}, residual={residual_val:.2f} ({match_status})")
                except Exception as e:
                    logger.warning(f"  [Residual Samples] Failed to log residual samples: {e}")
            
            ml_knowledge = extract_ml_knowledge(self.ml_mechanism, self.feature_cols, self.task_type)
            
            # Determine if ML model is linear
            model_name = getattr(self.ml_mechanism, "model_name", "unknown")
            is_linear_ml = model_name in ["LinearRegression", "LogisticRegression"]
            
            # Generate variants (up to num_mechanisms_unknown)
            mechanisms = []
            num_variants = min(3, self.num_mechanisms_unknown)
            
            # If ML model is linear, skip linear variant (variant 0) and use non-linear variants
            # This ensures LLM mechanisms complement rather than replicate the ML model
            if is_linear_ml and num_variants == 1:
                # Only 1 mechanism needed: use non-linear variant (variant 2)
                variant = 2
                mech = generate_ml_guided_mechanism(
                    ml_knowledge, self.feature_cols, self.task_type, self.class_names, variant
                )
                mechanisms.append(mech)
            else:
                # Generate multiple variants
                for variant in range(num_variants):
                    mech = generate_ml_guided_mechanism(
                        ml_knowledge, self.feature_cols, self.task_type, self.class_names, variant
                    )
                    mechanisms.append(mech)
            
            self.unknown_mechanisms = mechanisms
            return mechanisms
        
        # Standard mechanism generation (not ML-guided)
        mechanisms = []
        for _ in range(self.num_mechanisms_unknown):
            latent_z = self.encode_latent_space(X_train, y_train, prediction_errors, ml_residuals)
            mechanism = self.decode_latent_space(latent_z)
            mechanisms.append(mechanism)
        
        self.unknown_mechanisms = mechanisms
        return mechanisms
    
    def decode_latent_space(self, latent_z: str) -> str:
        """Decode latent representation into executable mechanism"""
        prompt = get_prompt('mechanism_generation.decoder',
                           decoder_prompt=self.decoder_prompt,
                           latent_z=latent_z)
        mechanism = self.llm.invoke_single(HumanMessage(content=prompt))
        return mechanism
    
    def get_all_mechanisms(self) -> List[str]:
        """Get all mechanisms (known + unknown)"""
        mechanisms = self.known_mechanisms + self.unknown_mechanisms
        if self.ml_mechanism is not None and self.ml_mechanism.is_trained:
            mechanisms.append(self.ml_mechanism.get_description())
        return mechanisms
    
    def get_mechanism_types(self) -> List[str]:
        """Get mechanism types"""
        # Known mechanisms are labeled as "known", unknown as "llm"
        types = ["known"] * len(self.known_mechanisms) + ["llm"] * len(self.unknown_mechanisms)
        if self.ml_mechanism is not None and self.ml_mechanism.is_trained:
            types.append("ml")
        return types
    
    def get_mechanism_descriptions(self) -> List[str]:
        """Get descriptions of all mechanisms"""
        return [f"Mechanism {i+1}: {mech}" for i, mech in enumerate(self.get_all_mechanisms())]
    
    def _get_scaling_range(self) -> Tuple[float, float]:
        """Get scaling range from scaler"""
        if hasattr(self, 'scaler') and self.scaler is not None:
            if hasattr(self.scaler, 'feature_range_'):
                return self.scaler.feature_range_
            elif hasattr(self.scaler, 'scale_') and hasattr(self.scaler, 'min_'):
                # MinMaxScaler
                return (self.scaler.min_[0] if hasattr(self.scaler.min_, '__len__') else self.scaler.min_,
                        self.scaler.max_[0] if hasattr(self.scaler.max_, '__len__') else self.scaler.max_)
        return (SCALE_MIN, SCALE_MAX)
    
    def _check_mechanism_diversity(self, candidate: str) -> bool:
        """Check if candidate mechanism is diverse enough from existing ones"""
        if len(self.unknown_mechanisms) == 0:
            return True
        
        # Simple diversity check: compare string similarity
        cand_lower = candidate.lower()
        for prev in self.unknown_mechanisms:
            prev_lower = prev.lower()
            # Jaccard similarity on words
            cand_words = set(cand_lower.split())
            prev_words = set(prev_lower.split())
            if len(cand_words) > 0 and len(prev_words) > 0:
                jaccard = len(cand_words & prev_words) / len(cand_words | prev_words)
                if jaccard > 0.8:  # Too similar
                    return False
        return True

# =========================
# TRAINABLE MA-ICL
# =========================

class TrainableMAICL:
    """Trainable MA-ICL system with residual-based learning"""
    def __init__(self, batched_llm: BatchedLLM, feature_cols: List[str], scaler: Any = None,
                 use_ml_mechanism: bool = False, dataset_name: str = "Dataset",
                 y_scaler: Any = None, pretrained_ml_mechanism: Optional[Any] = None,
                 data_insights: Optional[Dict[str, Any]] = None, task_type: str = "regression",
                 class_names: Optional[List[str]] = None, regression_loss_metric: Optional[str] = None,
                 classification_loss_metric: Optional[str] = None,
                 attention_temp: Optional[float] = None,
                 min_ml_weight: Optional[float] = None,
                 max_ml_weight: Optional[float] = None,
                 hard_ml_gate_threshold: Optional[float] = None,
                 num_mechanisms_unknown: Optional[int] = None):
        self.llm = batched_llm
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.use_ml_mechanism = use_ml_mechanism
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.class_names = class_names
        self.output_dir = None  # Will be set during training or via set_output_dir
        # Regression loss metric: "mae" (default) or "r2" (uses 1-R2)
        # IMPORTANT: Only used for regression tasks. Classification uses F1 or accuracy.
        if task_type == "classification":
            # For classification, regression_loss_metric is ignored (we use F1/accuracy)
            if regression_loss_metric is not None:
                logger.warning(f"regression_loss_metric='{regression_loss_metric}' is ignored for classification tasks. "
                             f"Classification uses F1 score or accuracy for loss calculation.")
            self.regression_loss_metric = None  # Not applicable for classification
            # Classification loss metric: "f1" (default) or "acc"/"accuracy"
            if classification_loss_metric is None:
                classification_loss_metric = "f1"
            classification_loss_metric = classification_loss_metric.lower()
            if classification_loss_metric not in ("f1", "acc", "accuracy"):
                logger.warning(f"Invalid classification_loss_metric '{classification_loss_metric}', defaulting to 'f1'")
                classification_loss_metric = "f1"
            # Normalize accuracy variants to "acc"
            if classification_loss_metric in ("accuracy", "acc"):
                classification_loss_metric = "acc"
            self.classification_loss_metric = classification_loss_metric
        else:
            # For regression tasks, validate and set regression_loss_metric
            if regression_loss_metric is None:
                regression_loss_metric = "mae"
            regression_loss_metric = regression_loss_metric.lower()
            if regression_loss_metric not in ("mae", "r2", "r2_score", "r2s", "r2_squared"):
                logger.warning(f"Invalid regression_loss_metric '{regression_loss_metric}', defaulting to 'mae'")
                regression_loss_metric = "mae"
            # Normalize r2 variants to "r2"
            if regression_loss_metric in ("r2_score", "r2s", "r2_squared"):
                regression_loss_metric = "r2"
            self.regression_loss_metric = regression_loss_metric
            # Classification loss metric not applicable for regression
            if classification_loss_metric is not None:
                logger.warning(f"classification_loss_metric='{classification_loss_metric}' is ignored for regression tasks.")
            self.classification_loss_metric = None
        
        self.mech_generator = VariationalMechanismGenerator(
            batched_llm, feature_cols, use_ml_mechanism, dataset_name,
            task_type=task_type, class_names=class_names,
            pretrained_ml_mechanism=pretrained_ml_mechanism,
            data_insights=data_insights,
            scaler=scaler,
            num_mechanisms_unknown=num_mechanisms_unknown
        )
        
        self.textgrad = TextGrad(batched_llm, task_type=task_type, feature_cols=feature_cols, dataset_name=dataset_name, class_names=class_names)
        # Routing parameters (can be overridden)
        self.attention_temp = attention_temp if attention_temp is not None else ATTENTION_TEMP
        self.min_ml_weight = min_ml_weight if min_ml_weight is not None else MIN_ML_WEIGHT
        self.max_ml_weight = max_ml_weight if max_ml_weight is not None else MAX_ML_WEIGHT
        self.hard_ml_gate_threshold = hard_ml_gate_threshold if hard_ml_gate_threshold is not None else HARD_ML_GATE_THRESHOLD
        self.k_shot = K_SHOT_PER_TARGET  # Default: use K_SHOT_PER_TARGET (can be overridden via train() or evaluate())
        self.mechanisms = self.mech_generator.get_all_mechanisms()
        self.mechanism_types = self.mech_generator.get_mechanism_types()
        self.predictor = None
        
        self.training_history = {
            "iterations": [],
            "val_loss": [],
            "discovery_loss": [],
            "accepted_updates": [],
            "rejected_updates": [],
            "mechanism_evolution": [],
            "llm_calls": [],
            "llm_batches": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_r2": [],
            "val_mae": []
        }
        
        self.ml_baseline_performance = None
        self.mechanism_performance_snapshot = {}
        
        logger.info(f"  Initialized MA-ICL with {len(self.mechanisms)} mechanisms:")
        logger.info(f"    • {len([t for t in self.mechanism_types if t == 'llm'])} LLM mechanisms")
        logger.info(f"    • {len([t for t in self.mechanism_types if t == 'ml'])} ML mechanism(s)")
    
    def train_ml_mechanism(self, X_train: np.ndarray, y_train: np.ndarray, y_scaler: Any = None):
        """Train the ML mechanism"""
        if self.use_ml_mechanism:
            logger.info("\n[ML Mechanism] Training ML model on training data...")
            self.mech_generator.train_ml_mechanism(X_train, y_train, y_scaler)
            self.mechanisms = self.mech_generator.get_all_mechanisms()
            self.mechanism_types = self.mech_generator.get_mechanism_types()
            logger.info(f"  ✓ Total mechanisms: {len(self.mechanisms)}")
    
    def _verify_unified_scaling(self, X_train, y_train, X_val, y_val):
        """Verify all data uses unified [0, 10] scaling
        
        Note: For classification tasks, y values are class labels and should NOT be scaled.
        Only X (features) are validated for classification. For regression, both X and y are validated.
        """
        scale_min, scale_max = 0.0, 10.0
        is_classification = getattr(self, 'task_type', 'regression') == 'classification'
        
        logger.info("\n[Scaling Verification]")
        
        # Check X_train
        x_min, x_max = X_train.min(), X_train.max()
        if not (scale_min - 0.01 <= x_min and x_max <= scale_max + 0.01):
            logger.warning(f"  ⚠️ X_train outside [{scale_min}, {scale_max}]: min={x_min:.3f}, max={x_max:.3f}")
        else:
            logger.info(f"  ✓ X_train scaled correctly: [{x_min:.3f}, {x_max:.3f}]")
        
        # Check y_train - only for regression tasks
        if not is_classification:
            y_min, y_max = y_train.min(), y_train.max()
            if not (scale_min - 0.01 <= y_min and y_max <= scale_max + 0.01):
                logger.warning(f"  ⚠️ y_train outside [{scale_min}, {scale_max}]: min={y_min:.3f}, max={y_max:.3f}")
            else:
                logger.info(f"  ✓ y_train scaled correctly: [{y_min:.3f}, {y_max:.3f}]")
        else:
            # For classification, just log that y values are class labels
            logger.info(f"  ✓ y_train contains class labels (not scaled)")
        
        # Check X_val
        x_min, x_max = X_val.min(), X_val.max()
        if not (scale_min - 0.01 <= x_min and x_max <= scale_max + 0.01):
            logger.warning(f"  ⚠️ X_val outside [{scale_min}, {scale_max}]: min={x_min:.3f}, max={x_max:.3f}")
        else:
            logger.info(f"  ✓ X_val scaled correctly: [{x_min:.3f}, {x_max:.3f}]")
        
        # Check y_val - only for regression tasks
        if not is_classification:
            y_min, y_max = y_val.min(), y_val.max()
            if not (scale_min - 0.01 <= y_min and y_max <= scale_max + 0.01):
                logger.warning(f"  ⚠️ y_val outside [{scale_min}, {scale_max}]: min={y_min:.3f}, max={y_max:.3f}")
            else:
                logger.info(f"  ✓ y_val scaled correctly: [{y_min:.3f}, {y_max:.3f}]")
        else:
            # For classification, just log that y values are class labels
            logger.info(f"  ✓ y_val contains class labels (not scaled)")
    
    def set_ml_baseline_performance(self, ml_metrics: Dict[str, float]):
        """Set ML baseline performance for comparison and initialize ML mechanism with high performance"""
        self.ml_baseline_performance = ml_metrics
        
        # Find ML mechanism index and set its initial performance to very high value
        # This ensures ML mechanism dominates pre-training, making MA-ICL start close to ML baseline
        ml_idx = None
        for i, mtype in enumerate(self.mechanism_types):
            if mtype == "ml":
                ml_idx = i
                break
        
        if ml_idx is not None:
            if self.task_type == "classification":
                acc = ml_metrics.get('accuracy', 0.0)
                f1 = ml_metrics.get('f1', 0.0)
                logger.info(f"  [Performance] ML baseline set - ACC: {acc:.4f}, F1: {f1:.4f}")
                
                # Compute initial performance score (matching _evaluate_mechanism_performance formula)
                # For classification: use weighted combination of F1 and accuracy
                classification_loss = getattr(self, "classification_loss_metric", "f1")
                if classification_loss == "acc":
                    # Accuracy-priority: 70% accuracy, 30% F1
                    initial_performance = (acc * 0.7 + f1 * 0.3) * 10.0
                else:
                    # F1-priority (default): 70% F1, 30% accuracy
                    initial_performance = (f1 * 0.7 + acc * 0.3) * 10.0
            else:
                # Regression: use R2 or MAE
                r2 = ml_metrics.get('r2', 0.0)
                mae = ml_metrics.get('mae', 1.0)
                regression_loss = getattr(self, "regression_loss_metric", "mae")
                if regression_loss == "r2":
                    # R2-based: higher R2 = higher performance
                    initial_performance = max(0.0, r2) * 10.0
                else:
                    # MAE-based: lower MAE = higher performance (invert and scale)
                    # Normalize MAE to [0, 1] range, then invert (lower MAE = higher score)
                    initial_performance = max(0.0, (1.0 - min(mae / 10.0, 1.0)) * 10.0)
            
            # CRITICAL: Boost ML performance significantly to ensure it dominates pre-training
            # This ensures MA-ICL starts at baseline performance (not dragged down by known mechanisms)
            # Use 5x multiplier to ensure ML gets ~95-99% weight in attention mechanism
            high_initial_performance = initial_performance * 5.0
            
            # Initialize mechanism performance snapshot if needed
            if not hasattr(self, 'mechanism_performance_snapshot'):
                self.mechanism_performance_snapshot = {}
            
            self.mechanism_performance_snapshot[ml_idx] = high_initial_performance
            logger.info(f"  [Performance] ML mechanism (idx={ml_idx}) initialized with performance={high_initial_performance:.2f} (base={initial_performance:.2f}) to ensure dominance in pre-training")
        else:
            if self.task_type == "classification":
                acc = ml_metrics.get('accuracy', 0.0)
                f1 = ml_metrics.get('f1', 0.0)
                logger.info(f"  [Performance] ML baseline set - ACC: {acc:.4f}, F1: {f1:.4f}")
            else:
                r2 = ml_metrics.get('r2', 0.0)
                mae = ml_metrics.get('mae', 1.0)
                logger.info(f"  [Performance] ML baseline set - R²: {r2:.4f}, MAE: {mae:.4f}")
            logger.warning(f"  [Performance] ML mechanism not found in mechanism list - cannot set initial performance")
    
    def _evaluate_mechanism_performance(self, all_agent_preds: List[List[float]], 
                                       true_values: List[float], predictor: 'MultiAgentPredictor'):
        """
        Evaluate individual mechanism performance for adaptive weighting.
        
        This method was added to fix a critical bug where mechanism performance was never tracked,
        making performance-based weighting non-functional. It calculates how well each mechanism
        predicts on its own, enabling the system to weight better-performing mechanisms higher.
        
        Args:
            all_agent_preds: List of prediction lists, shape (n_samples, n_mechanisms)
            true_values: Ground truth values (n_samples,)
            predictor: MultiAgentPredictor instance to update with performance scores
        
        Effects:
            - Updates predictor.mechanism_performance dict with scores for each mechanism
            - Stores scores in self.mechanism_performance_snapshot for checkpoint restoration
            - Enables data-driven weighting based on empirical performance
        
        Performance Metrics:
            - Classification: F1 score (range [0, 1], higher is better)
            - Regression: 1/(0.01 + MAE) (range (0, 100], higher is better)
        
        Note: This is called automatically during evaluate() to keep performance scores current.
        """
        if len(all_agent_preds) == 0 or len(true_values) == 0:
            return
        
        # Convert to numpy arrays for easier computation
        agent_preds_array = np.array(all_agent_preds)  # Shape: (n_samples, n_mechanisms)
        true_array = np.array(true_values)
        
        # Calculate performance for each mechanism
        for mech_idx in range(agent_preds_array.shape[1]):
            mech_predictions = agent_preds_array[:, mech_idx]
            
            try:
                if self.task_type == "classification":
                    # Get classification loss metric (default: "f1")
                    classification_loss = getattr(self, "classification_loss_metric", "f1")
                    
                    if self.class_names is not None and len(self.class_names) > 2:
                        # Multi-class
                        y_pred = np.round(mech_predictions).astype(int)
                        y_true = true_array.astype(int)
                        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                        acc = accuracy_score(y_true, y_pred)
                        # Use selected metric for performance score
                        if classification_loss == "acc":
                            performance_score = acc  # Range: [0, 1], higher is better
                        else:  # Default to F1
                            performance_score = f1  # Range: [0, 1], higher is better
                        
                        # Store all metrics for TextGrad
                        metrics_dict = {"accuracy": acc, "f1": f1, "performance_score": performance_score}
                    else:
                        # Binary classification
                        y_pred = (mech_predictions >= 0.5).astype(int)
                        y_true = true_array.astype(int)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        acc = accuracy_score(y_true, y_pred)
                        # Use selected metric for performance score
                        if classification_loss == "acc":
                            performance_score = acc  # Range: [0, 1], higher is better
                        else:  # Default to F1
                            performance_score = f1  # Range: [0, 1], higher is better
                        
                        # Store all metrics for TextGrad
                        metrics_dict = {"accuracy": acc, "f1": f1, "performance_score": performance_score}
                else:
                    # Regression: performance aligned with configured loss metric
                    # Only use regression_loss_metric for regression tasks
                    mae = mean_absolute_error(true_array, mech_predictions)
                    try:
                        r2_val = r2_score(true_array, mech_predictions)
                    except Exception:
                        r2_val = -1.0
                    try:
                        mse = mean_squared_error(true_array, mech_predictions)
                    except Exception:
                        mse = float('inf')
                    
                    regression_loss = getattr(self, "regression_loss_metric", "mae")
                    if regression_loss is not None and regression_loss == "r2":
                        # Performance score based on R2: normalize to [0, 1] range for better routing
                        # R2 can be negative, so we use a sigmoid-like transformation
                        # For R2 > 0: use R2 directly (clamped to [0, 1])
                        # For R2 < 0: use a penalty that maps to [0, 0.5]
                        if r2_val > 0:
                            performance_score = min(1.0, max(0.0, float(r2_val)))
                        else:
                            # Negative R2: map to [0, 0.5] range
                            performance_score = max(0.0, 0.5 * (1.0 + r2_val / (1.0 - r2_val)))
                    else:
                        # Default: inverse MAE with better normalization for routing
                        # Normalize MAE relative to a baseline (e.g., mean of all MAEs or a fixed threshold)
                        # Use a more stable formula: exp(-mae/scale) where scale is adaptive
                        # First, compute a reference MAE (mean of all mechanism MAEs or use a fixed scale)
                        # For now, use a scale based on the target range (typically [0, 10] for scaled data)
                        scale = 2.0  # Scale factor: MAE of 2.0 gives ~0.37, MAE of 1.0 gives ~0.61
                        performance_score = np.exp(-mae / scale)  # Range: (0, 1], higher is better
                        # Clamp to reasonable range
                        performance_score = max(0.01, min(1.0, float(performance_score)))
                    
                    # Store all metrics for TextGrad
                    metrics_dict = {"r2": r2_val, "mae": mae, "mse": mse, "performance_score": performance_score}
                
                # Update predictor's mechanism performance
                predictor.update_mechanism_performance(mech_idx, performance_score)
                
                # Store in snapshot for later transfer between evaluations
                # Store both the performance score (for routing) and full metrics (for TextGrad)
                if not hasattr(self, 'mechanism_performance_snapshot'):
                    self.mechanism_performance_snapshot = {}
                if not hasattr(self, 'mechanism_metrics_snapshot'):
                    self.mechanism_metrics_snapshot = {}
                
                self.mechanism_performance_snapshot[mech_idx] = performance_score
                self.mechanism_metrics_snapshot[mech_idx] = metrics_dict
                
                logger.debug(f"  Mechanism {mech_idx} performance: {metrics_dict}")
                
            except Exception as e:
                logger.debug(f"Performance eval failed for mechanism {mech_idx}: {e}")
                # If calculation fails, use default performance
                predictor.update_mechanism_performance(mech_idx, 1.0)
        
        # For regression: normalize performance scores relative to all mechanisms for better routing
        # FIXED: Use proper normalization that preserves performance differences
        # When ML is clearly better, it should get most of the weight
        if self.task_type == "regression":
            all_perf_scores = [predictor.mechanism_performance.get(i, 0.5) for i in range(len(self.mechanism_types))]
            if len(all_perf_scores) > 1 and max(all_perf_scores) > min(all_perf_scores):
                min_perf = min(all_perf_scores)
                max_perf = max(all_perf_scores)
                range_perf = max_perf - min_perf
                if range_perf > 0:
                    # Check if ML mechanism is significantly better than LLM mechanisms
                    ml_indices = [i for i, t in enumerate(self.mechanism_types) if t == "ml"]
                    llm_indices = [i for i, t in enumerate(self.mechanism_types) if t == "llm"]
                    
                    if len(ml_indices) > 0 and len(llm_indices) > 0:
                        ml_perf = max([predictor.mechanism_performance.get(i, 0.5) for i in ml_indices])
                        llm_perf = max([predictor.mechanism_performance.get(i, 0.5) for i in llm_indices])
                        perf_ratio = ml_perf / (llm_perf + 1e-6)
                        
                        # If ML is significantly better (ratio > 1.2), use aggressive weighting
                        # Otherwise, use moderate normalization
                        if perf_ratio > 1.2:
                            # ML is clearly better: use aggressive normalization to give ML most weight
                            # Map to [0.05, 1.0] range to preserve large differences
                            for mech_idx in range(len(self.mechanism_types)):
                                original_perf = predictor.mechanism_performance.get(mech_idx, 0.5)
                                normalized_perf = 0.05 + 0.95 * (original_perf - min_perf) / range_perf
                                normalized_perf = max(0.05, min(1.0, normalized_perf))
                                predictor.update_mechanism_performance(mech_idx, normalized_perf)
                                if hasattr(self, 'mechanism_performance_snapshot'):
                                    self.mechanism_performance_snapshot[mech_idx] = normalized_perf
                        else:
                            # Mechanisms are similar: use moderate normalization
                            # Map to [0.1, 1.0] range
                            for mech_idx in range(len(self.mechanism_types)):
                                original_perf = predictor.mechanism_performance.get(mech_idx, 0.5)
                                normalized_perf = 0.1 + 0.9 * (original_perf - min_perf) / range_perf
                                normalized_perf = max(0.1, min(1.0, normalized_perf))
                                predictor.update_mechanism_performance(mech_idx, normalized_perf)
                                if hasattr(self, 'mechanism_performance_snapshot'):
                                    self.mechanism_performance_snapshot[mech_idx] = normalized_perf
                    else:
                        # No ML or no LLM: use standard normalization
                        for mech_idx in range(len(self.mechanism_types)):
                            original_perf = predictor.mechanism_performance.get(mech_idx, 0.5)
                            normalized_perf = 0.1 + 0.9 * (original_perf - min_perf) / range_perf
                            normalized_perf = max(0.1, min(1.0, normalized_perf))
                            predictor.update_mechanism_performance(mech_idx, normalized_perf)
                            if hasattr(self, 'mechanism_performance_snapshot'):
                                self.mechanism_performance_snapshot[mech_idx] = normalized_perf
        
        # Boost ML mechanism confidence if it clearly outperforms LLM mechanisms
        try:
            ml_indices = [i for i, t in enumerate(self.mechanism_types) if t == "ml"]
            llm_indices = [i for i, t in enumerate(self.mechanism_types) if t == "llm"]
            if len(ml_indices) > 0 and len(llm_indices) > 0:
                ml_perf = np.mean([predictor.mechanism_performance.get(i, 0.5) for i in ml_indices])
                llm_perf = np.mean([predictor.mechanism_performance.get(i, 0.5) for i in llm_indices])
                if np.isfinite(ml_perf) and np.isfinite(llm_perf) and llm_perf > 0:
                    perf_ratio = float(ml_perf / llm_perf)
                    if hasattr(self.mech_generator, "ml_mechanism") and hasattr(self.mech_generator.ml_mechanism, "set_performance_boost"):
                        boost = max(1.0, min(3.0, perf_ratio))
                        self.mech_generator.ml_mechanism.set_performance_boost(boost)
        except Exception:
            pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, X_pool: np.ndarray, y_pool: np.ndarray, 
                 return_details: bool = False, relax_routing: bool = False,
                 attention_temp: Optional[float] = None,
                 min_ml_weight: Optional[float] = None,
                 max_ml_weight: Optional[float] = None,
                 hard_ml_gate_threshold: Optional[float] = None,
                 k_shot: Optional[int] = None,
                 preserve_mechanism_performance: bool = False,
                 **kwargs) -> Dict:
        """
        Evaluate the MA-ICL system
        
        Args:
            X: Features to evaluate on (can be validation or test set)
            y: Targets to evaluate on (can be validation or test set)
            X_pool: Training pool features - MUST be training data only (used for few-shot examples)
            y_pool: Training pool targets - MUST be training data only (used for few-shot examples)
            return_details: If True, return additional metrics
            relax_routing: If True, use relaxed ML routing (allows LLM contributions)
            k_shot: Number of few-shot examples to use (overrides instance default)
            preserve_mechanism_performance: If True, preserve mechanism performance scores from training
                                           snapshot instead of recalculating. Use this when evaluating
                                           on the same test set used for acceptance during training to
                                           ensure consistent routing and results.
            **kwargs: Additional routing parameters
        
        Note: X_pool and y_pool MUST be from the training set only to avoid data leakage.
        """
        # Use provided k_shot or instance default
        if k_shot is not None:
            self.k_shot = k_shot
        
        # Use provided routing parameters or defaults
        att_temp = attention_temp if attention_temp is not None else self.attention_temp
        # Optionally relax ML routing during training/acceptance to allow LLM impact
        if relax_routing:
            hard_gate = hard_ml_gate_threshold if hard_ml_gate_threshold is not None else max(0.90, HARD_ML_GATE_THRESHOLD)
            min_w = min_ml_weight if min_ml_weight is not None else 0.85
            max_w = max_ml_weight if max_ml_weight is not None else 0.95
        else:
            hard_gate = hard_ml_gate_threshold if hard_ml_gate_threshold is not None else HARD_ML_GATE_THRESHOLD
            min_w = min_ml_weight if min_ml_weight is not None else MIN_ML_WEIGHT
            max_w = max_ml_weight if max_ml_weight is not None else MAX_ML_WEIGHT
        predictor = MultiAgentPredictor(
            self.llm, self.mechanisms, self.mechanism_types,
            self.mech_generator.ml_mechanism, self.feature_cols, self.scaler, att_temp,
            task_type=self.task_type, class_names=self.class_names,
            hard_ml_gate_threshold=hard_gate, min_ml_weight=min_w, max_ml_weight=max_w
        )
        
        # Transfer mechanism performance scores
        # Priority order:
        # 1. If preserve_mechanism_performance=True, always use snapshot (for consistency)
        # 2. If relax_routing=True (training mode), use snapshot
        # 3. If snapshot exists and has ML mechanism with high performance, use it (for pre-training baseline)
        # 4. Otherwise, let performance be re-computed on the evaluation set
        
        # Check if we're in pre-training mode (ML has boosted performance > 20.0)
        is_pre_training = False
        ml_idx_check = None
        if hasattr(self, 'mechanism_performance_snapshot') and self.mechanism_performance_snapshot:
            for i, mtype in enumerate(self.mechanism_types):
                if mtype == "ml":
                    ml_idx_check = i
                    break
            if ml_idx_check is not None and ml_idx_check in self.mechanism_performance_snapshot:
                ml_perf = self.mechanism_performance_snapshot[ml_idx_check]
                # If ML performance is high (>20), it was boosted for pre-training
                if ml_perf > 20.0:
                    is_pre_training = True
        
        should_preserve = preserve_mechanism_performance or relax_routing
        
        # Also preserve if snapshot exists and contains ML mechanism with high performance (pre-training case)
        # This ensures pre-training MA-ICL uses the boosted ML performance we set
        if not should_preserve and is_pre_training:
            should_preserve = True
            logger.debug(f"  [Evaluate] Using pre-training performance snapshot (ML performance={ml_perf:.2f} is boosted)")
        
        if should_preserve and hasattr(self, 'mechanism_performance_snapshot') and self.mechanism_performance_snapshot:
            for mech_idx, perf in self.mechanism_performance_snapshot.items():
                predictor.update_mechanism_performance(mech_idx, perf)
            
            # During pre-training: set LLM mechanisms to very low performance to minimize their impact
            # This ensures ML mechanism dominates and MA-ICL starts at baseline performance
            if is_pre_training:
                for mech_idx, mtype in enumerate(self.mechanism_types):
                    if mtype == "llm" and mech_idx not in self.mechanism_performance_snapshot:
                        # Set LLM mechanisms to very low performance (0.01) so they get minimal weight
                        predictor.update_mechanism_performance(mech_idx, 0.01)
                        logger.debug(f"  [Evaluate] Set LLM mechanism {mech_idx} to minimal performance (0.01) for pre-training")
            
            if preserve_mechanism_performance:
                logger.info(f"  [Evaluate] Preserving mechanism performance scores from training snapshot: {self.mechanism_performance_snapshot}")
            elif relax_routing:
                logger.debug(f"  [Evaluate] Using mechanism performance scores from training (relax_routing=True)")
            else:
                logger.debug(f"  [Evaluate] Using pre-training performance snapshot for baseline accuracy")
        
        # Extract X_original and X_pool_original from kwargs if available
        # These contain non-vectorized features (e.g., SMILES strings) for LLM
        X_original = kwargs.get('X_original', None)
        X_pool_original = kwargs.get('X_pool_original', None)
        
        # Use batched prediction for efficiency
        # For DeepChem datasets, X_original contains SMILES strings (dict with 'SMILES' key)
        # For other datasets, X_original may contain original feature dicts or be None
        if X_original is not None and len(X_original) == len(X):
            # Use X_original (non-vectorized features like SMILES) for LLM
            x_dicts = []
            for i in range(len(X)):
                if isinstance(X_original[i], dict):
                    # Already a dict (e.g., {'SMILES': '...'} for DeepChem)
                    x_dicts.append(X_original[i].copy())
                else:
                    # Fallback: convert from vectorized features
                    x_dicts.append({self.feature_cols[j]: float(X[i, j]) for j in range(len(self.feature_cols))})
            logger.debug(f"[Evaluate] Using X_original (non-vectorized features) for LLM predictions")
        else:
            # Fallback: convert from vectorized features
            x_dicts = [{self.feature_cols[j]: float(X[i, j]) for j in range(len(self.feature_cols))} for i in range(len(X))]
            if X_original is not None:
                logger.warning(f"[Evaluate] X_original length ({len(X_original) if X_original else 0}) doesn't match X length ({len(X)}), using vectorized features")
        
        # Get k_shot from instance if available, otherwise default to K_SHOT_PER_TARGET
        k_shot = getattr(self, 'k_shot', K_SHOT_PER_TARGET) if hasattr(self, 'k_shot') else K_SHOT_PER_TARGET
        # Retrieve few-shot examples if k_shot > 0
        # IMPORTANT: X_pool and y_pool MUST be training data only (not validation/test)
        if k_shot > 0:
            if len(X_pool) == 0 or len(y_pool) == 0:
                logger.warning(f"[Few-shot] Training pool is empty (X_pool: {len(X_pool)}, y_pool: {len(y_pool)}), skipping few-shot examples")
                few_shot_list = [[]] * len(x_dicts)
            else:
                # Compute ML residuals on training pool to prioritize high-residual examples
                ml_residuals_pool = None
                if (self.use_ml_mechanism and 
                    self.mech_generator is not None and
                    hasattr(self.mech_generator, 'ml_mechanism') and
                    self.mech_generator.ml_mechanism is not None and
                    hasattr(self.mech_generator.ml_mechanism, 'is_trained') and
                    self.mech_generator.ml_mechanism.is_trained):
                    try:
                        _, _, ml_residuals_pool, _ = compute_ml_residuals(
                            self.mech_generator.ml_mechanism,
                            X_pool, y_pool, self.feature_cols,
                            class_names=self.class_names,
                            task_type=self.task_type
                        )
                        logger.debug(f"[Few-shot] Computed ML residuals on training pool (mean={np.mean(np.abs(ml_residuals_pool)):.4f})")
                    except Exception as e:
                        logger.debug(f"[Few-shot] Could not compute ML residuals for few-shot prioritization: {e}")
                
                few_shot_examples = retrieve_few_shot_examples(
                    X_pool, y_pool, self.feature_cols, k_shot,
                    task_type=self.task_type, class_names=self.class_names,
                    ml_residuals=ml_residuals_pool, prioritize_residuals=True,
                    X_pool_original=X_pool_original
                )
                # Use same few-shot examples for all predictions in the batch
                few_shot_list = [few_shot_examples] * len(x_dicts)
                if len(few_shot_examples) > 0:
                    logger.info(f"[Few-shot] Using {len(few_shot_examples)} examples from training set for {len(x_dicts)} predictions")
        else:
            few_shot_list = [[]] * len(x_dicts)
        
        # Track individual mechanism performance for adaptive weighting
        # CRITICAL FIX: If we're in pre-training mode, preserve the snapshot to maintain baseline performance.
        # Otherwise, recalculate mechanism performance for accurate routing.
        if is_pre_training:
            # Pre-training: preserve snapshot to maintain baseline performance
            # This ensures MA-ICL starts at ML baseline (not dragged down by LLM mechanisms)
            should_recalculate = False
            should_update_snapshot = False
            logger.info(f"  [Evaluate] Preserving pre-training performance snapshot to maintain baseline accuracy")
        elif relax_routing and preserve_mechanism_performance:
            # During training with explicit preservation: recalculate for accuracy but don't update snapshot
            # This ensures accurate routing while maintaining consistency with training logs
            should_recalculate = True
            should_update_snapshot = False  # Don't update snapshot to match training behavior
            logger.info(f"  [Evaluate] Recalculating mechanism performance on evaluation set (preserving snapshot for consistency)")
            if hasattr(self, 'mechanism_performance_snapshot') and self.mechanism_performance_snapshot:
                logger.debug(f"  [Evaluate] Snapshot exists but will be recalculated: {self.mechanism_performance_snapshot}")
        elif relax_routing and not preserve_mechanism_performance:
            # During training without explicit preservation: recalculate and update snapshot
            should_recalculate = True
            should_update_snapshot = True
            logger.debug(f"  [Evaluate] Recalculating mechanism performance scores (training mode, updating snapshot)")
        else:
            # Final evaluation (test set): always recalculate for accuracy
            should_recalculate = True
            should_update_snapshot = True
            logger.info(f"  [Evaluate] Recalculating mechanism performance scores on test set (final evaluation)")
        
        # CRITICAL FIX: If we're going to recalculate, we need to recalculate BEFORE making final predictions
        # to ensure routing uses the correct performance scores. However, we need predictions to calculate
        # performance. Solution: Make initial predictions with snapshot/default scores, recalculate,
        # then re-make predictions with correct routing.
        if should_recalculate:
            # First pass: Make predictions with current routing (snapshot or default scores)
            # This gives us individual mechanism predictions needed for performance calculation
            initial_batch_outputs = predictor.predict_batch(x_dicts, few_shot_list=few_shot_list)
            initial_all_agent_preds = [out[1] for out in initial_batch_outputs]
            initial_true_values = [float(y[i]) for i in range(len(y))]
            
            # Recalculate mechanism performance based on initial predictions
            self._evaluate_mechanism_performance(initial_all_agent_preds, initial_true_values, predictor)
            
            # Update snapshot with new performance scores only if we should update
            if should_update_snapshot and hasattr(predictor, 'mechanism_performance'):
                for mech_idx, perf in predictor.mechanism_performance.items():
                    if not hasattr(self, 'mechanism_performance_snapshot'):
                        self.mechanism_performance_snapshot = {}
                    self.mechanism_performance_snapshot[mech_idx] = perf
            
            # Second pass: Re-make predictions with correct routing weights
            batch_outputs = predictor.predict_batch(x_dicts, few_shot_list=few_shot_list)
            predictions = [out[0] for out in batch_outputs]
            all_agent_preds = [out[1] for out in batch_outputs]
            true_values = [float(y[i]) for i in range(len(y))]
        else:
            # Not recalculating: make predictions once with snapshot scores
            batch_outputs = predictor.predict_batch(x_dicts, few_shot_list=few_shot_list)
            predictions = [out[0] for out in batch_outputs]
            all_agent_preds = [out[1] for out in batch_outputs]
            true_values = [float(y[i]) for i in range(len(y))]
        
        if len(predictions) < 2:
            if self.task_type == "classification":
                return {"accuracy": 0.0, "f1": 0.0, "loss": 1.0, "predictions": predictions, "true_values": true_values}
            return {"r2": -1.0, "mae": 1e9, "loss": 1e9, "predictions": predictions, "true_values": true_values}
        
        if self.task_type == "classification":
            y_probs = np.array(predictions)
            y_true = np.array(true_values).astype(int)
            
            # Get classification loss metric (default: "f1")
            classification_loss = getattr(self, "classification_loss_metric", "f1")
            
            # Handle multi-class vs binary
            if self.class_names is not None and len(self.class_names) > 2:
                y_pred = y_probs.astype(int)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                # Choose loss based on classification_loss_metric
                if classification_loss == "acc":
                    loss = 1.0 - acc
                else:  # Default to F1
                    loss = 1.0 - f1
            else:
                y_pred = (y_probs >= 0.5).astype(int)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                # For binary, use optimal threshold based on the selected metric
                metric_for_threshold = 'f1' if classification_loss == "f1" else 'accuracy'
                optimal_threshold = find_optimal_threshold(y_true, y_probs, metric=metric_for_threshold)
                y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
                acc_optimal = accuracy_score(y_true, y_pred_optimal)
                f1_optimal = f1_score(y_true, y_pred_optimal, zero_division=0)
                # Choose loss based on classification_loss_metric
                if classification_loss == "acc":
                    loss = 1.0 - acc_optimal
                else:  # Default to F1
                    loss = 1.0 - f1_optimal
            
            result = {
                "accuracy": acc,
                "f1": f1,
                "loss": loss,
                "predictions": predictions,
                "true_values": true_values
            }
            
            if return_details:
                result["recall"] = recall_score(y_true, y_pred, zero_division=0, average='weighted' if len(self.class_names) > 2 else 'binary')
            
            return result
        else:
            # Regression task: use regression_loss_metric (MAE or R2)
            y_pred = np.array(predictions)
            y_true = np.array(true_values)
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            # Choose loss metric based on configuration (only for regression)
            regression_loss = getattr(self, "regression_loss_metric", "mae")
            if regression_loss is not None and regression_loss == "r2":
                # Maximize R2 by minimizing |1 - R2|; use abs to handle R2 > 1 cases
                try:
                    loss = abs(1.0 - float(r2))
                except Exception:
                    loss = 1.0
            else:
                # Default to MAE (or if regression_loss_metric is None, use MAE)
                loss = mae
            
            return {
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "loss": loss,
                "predictions": predictions,
                "true_values": true_values
            }
    
    def evaluate_llm_only(self, X: np.ndarray, y: np.ndarray, X_pool: np.ndarray, y_pool: np.ndarray,
                          return_details: bool = False, k_shot: Optional[int] = None,
                          **kwargs) -> Dict:
        """
        Evaluate MA-ICL with only LLM mechanisms (excluding ML mechanisms).
        This helps assess how well the LLM mechanisms learned the task independently.
        
        Args:
            X: Features to evaluate on
            y: Targets to evaluate on
            X_pool: Training pool features (for few-shot examples)
            y_pool: Training pool targets (for few-shot examples)
            return_details: If True, return additional metrics
            k_shot: Number of few-shot examples to use
            **kwargs: Additional parameters
        """
        # Filter to only LLM mechanisms
        llm_mechanisms = []
        llm_mechanism_types = []
        llm_indices = []
        for i, (mech, mech_type) in enumerate(zip(self.mechanisms, self.mechanism_types)):
            if mech_type == "llm":
                llm_mechanisms.append(mech)
                llm_mechanism_types.append(mech_type)
                llm_indices.append(i)
        
        if len(llm_mechanisms) == 0:
            logger.warning("[LLM-only evaluation] No LLM mechanisms found, returning empty results")
            if self.task_type == "classification":
                return {"accuracy": 0.0, "f1": 0.0, "loss": 1.0, "predictions": [], "true_values": []}
            return {"r2": -1.0, "mae": 1e9, "loss": 1e9, "predictions": [], "true_values": []}
        
        logger.info(f"[LLM-only evaluation] Using {len(llm_mechanisms)} LLM mechanism(s) (excluding ML)")
        
        # Use provided k_shot or instance default
        if k_shot is not None:
            self.k_shot = k_shot
        
        # Create predictor with only LLM mechanisms
        att_temp = self.attention_temp
        predictor = MultiAgentPredictor(
            self.llm, llm_mechanisms, llm_mechanism_types,
            None,  # No ML mechanism
            self.feature_cols, self.scaler, att_temp,
            task_type=self.task_type, class_names=self.class_names,
            hard_ml_gate_threshold=0.0,  # No ML routing needed
            min_ml_weight=0.0, max_ml_weight=0.0
        )
        
        # Transfer mechanism performance scores for LLM mechanisms only
        if hasattr(self, 'mechanism_performance_snapshot') and self.mechanism_performance_snapshot:
            # Map original indices to new indices
            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(llm_indices)}
            for old_idx, perf in self.mechanism_performance_snapshot.items():
                if old_idx in old_to_new_idx:
                    new_idx = old_to_new_idx[old_idx]
                    predictor.update_mechanism_performance(new_idx, perf)
        
        # Extract X_original and X_pool_original from kwargs if available
        # These contain non-vectorized features (e.g., SMILES strings) for LLM
        X_original = kwargs.get('X_original', None)
        X_pool_original = kwargs.get('X_pool_original', None)
        
        # Use batched prediction for efficiency
        # For DeepChem datasets, X_original contains SMILES strings (dict with 'SMILES' key)
        # For other datasets, X_original may contain original feature dicts or be None
        if X_original is not None and len(X_original) == len(X):
            # Use X_original (non-vectorized features like SMILES) for LLM
            x_dicts = []
            for i in range(len(X)):
                if isinstance(X_original[i], dict):
                    # Already a dict (e.g., {'SMILES': '...'} for DeepChem)
                    x_dicts.append(X_original[i].copy())
                else:
                    # Fallback: convert from vectorized features
                    x_dicts.append({self.feature_cols[j]: float(X[i, j]) for j in range(len(self.feature_cols))})
            logger.debug(f"[LLM-only Evaluate] Using X_original (non-vectorized features) for LLM predictions")
        else:
            # Fallback: convert from vectorized features
            x_dicts = [{self.feature_cols[j]: float(X[i, j]) for j in range(len(self.feature_cols))} for i in range(len(X))]
            if X_original is not None:
                logger.warning(f"[LLM-only Evaluate] X_original length ({len(X_original) if X_original else 0}) doesn't match X length ({len(X)}), using vectorized features")
        
        k_shot = getattr(self, 'k_shot', K_SHOT_PER_TARGET) if hasattr(self, 'k_shot') else K_SHOT_PER_TARGET
        
        # Retrieve few-shot examples if k_shot > 0
        if k_shot > 0:
            if len(X_pool) == 0 or len(y_pool) == 0:
                logger.warning(f"[Few-shot] Training pool is empty, skipping few-shot examples")
                few_shot_list = [[]] * len(x_dicts)
            else:
                # Compute ML residuals on training pool to prioritize high-residual examples
                ml_residuals_pool = None
                if (self.use_ml_mechanism and 
                    self.mech_generator is not None and
                    hasattr(self.mech_generator, 'ml_mechanism') and
                    self.mech_generator.ml_mechanism is not None and
                    hasattr(self.mech_generator.ml_mechanism, 'is_trained') and
                    self.mech_generator.ml_mechanism.is_trained):
                    try:
                        _, _, ml_residuals_pool, _ = compute_ml_residuals(
                            self.mech_generator.ml_mechanism,
                            X_pool, y_pool, self.feature_cols,
                            class_names=self.class_names,
                            task_type=self.task_type
                        )
                        logger.debug(f"[Few-shot] Computed ML residuals on training pool (mean={np.mean(np.abs(ml_residuals_pool)):.4f})")
                    except Exception as e:
                        logger.debug(f"[Few-shot] Could not compute ML residuals for few-shot prioritization: {e}")
                
                few_shot_examples = retrieve_few_shot_examples(
                    X_pool, y_pool, self.feature_cols, k_shot,
                    task_type=self.task_type, class_names=self.class_names,
                    ml_residuals=ml_residuals_pool, prioritize_residuals=True,
                    X_pool_original=X_pool_original
                )
                few_shot_list = [few_shot_examples] * len(x_dicts)
                if len(few_shot_examples) > 0:
                    logger.info(f"[Few-shot] Using {len(few_shot_examples)} examples from training set for {len(x_dicts)} predictions")
        else:
            few_shot_list = [[]] * len(x_dicts)
        
        batch_outputs = predictor.predict_batch(x_dicts, few_shot_list=few_shot_list)
        predictions = [out[0] for out in batch_outputs]
        true_values = [float(y[i]) for i in range(len(y))]
        
        if len(predictions) < 2:
            if self.task_type == "classification":
                return {"accuracy": 0.0, "f1": 0.0, "loss": 1.0, "predictions": predictions, "true_values": true_values}
            return {"r2": -1.0, "mae": 1e9, "loss": 1e9, "predictions": predictions, "true_values": true_values}
        
        if self.task_type == "classification":
            y_probs = np.array(predictions)
            y_true = np.array(true_values).astype(int)
            
            classification_loss = getattr(self, "classification_loss_metric", "f1")
            
            if self.class_names is not None and len(self.class_names) > 2:
                y_pred = y_probs.astype(int)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                if classification_loss == "acc":
                    loss = 1.0 - acc
                else:
                    loss = 1.0 - f1
            else:
                y_pred = (y_probs >= 0.5).astype(int)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                metric_for_threshold = 'f1' if classification_loss == "f1" else 'accuracy'
                optimal_threshold = find_optimal_threshold(y_true, y_probs, metric=metric_for_threshold)
                y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
                acc_optimal = accuracy_score(y_true, y_pred_optimal)
                f1_optimal = f1_score(y_true, y_pred_optimal, zero_division=0)
                if classification_loss == "acc":
                    loss = 1.0 - acc_optimal
                else:
                    loss = 1.0 - f1_optimal
            
            result = {
                "accuracy": acc,
                "f1": f1,
                "loss": loss,
                "predictions": predictions,
                "true_values": true_values
            }
            
            if return_details:
                result["recall"] = recall_score(y_true, y_pred, zero_division=0, average='weighted' if len(self.class_names) > 2 else 'binary')
            
            return result
        else:
            # Regression
            y_pred = np.array(predictions)
            y_true = np.array(true_values)
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            regression_loss = getattr(self, "regression_loss_metric", "mae")
            if regression_loss is not None and regression_loss == "r2":
                try:
                    loss = abs(1.0 - float(r2))
                except Exception:
                    loss = 1.0
            else:
                loss = mae
            
            return {
                "r2": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "loss": loss,
                "predictions": predictions,
                "true_values": true_values
            }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
              iterations: int = 3, ml_residuals: Optional[np.ndarray] = None,
              accept_eval_max: Optional[int] = None, accept_eval_min: Optional[int] = None,
              k_shot: int = 0, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
              use_test_for_acceptance: bool = False, X_train_original: Optional[List[Dict]] = None,
              X_val_original: Optional[List[Dict]] = None, X_test_original: Optional[List[Dict]] = None,
              output_dir: Optional[str] = None):
        """Train the MA-ICL system
        
        Args:
            accept_eval_max: Maximum samples for acceptance evaluation. If None, uses full validation set.
            accept_eval_min: Minimum samples for acceptance evaluation. If None, uses accept_max // 4 or full set.
            X_test: Optional test set features. Used if use_test_for_acceptance=True.
            y_test: Optional test set targets. Used if use_test_for_acceptance=True.
            use_test_for_acceptance: If True, use test set for acceptance evaluation instead of validation set.
                                    This helps ensure optimization generalizes to test set, but risks overfitting to test.
        """
        logger.info(f"\n[Training] Starting {iterations} iterations...")
        
        # Store output_dir for artifact persistence
        if output_dir is not None:
            self.output_dir = output_dir
        elif self.output_dir is None:
            # Fall back to global OUTPUT_DIR if not set
            from maicl_lib_v2 import OUTPUT_DIR
            self.output_dir = OUTPUT_DIR
        
        # Initialize iteration history tracking for TextGrad memory
        self.textgrad_iteration_history = []  # List of dicts with iteration history
        
        # Store X_train_original (SMILES strings) for use in residual logging
        self.X_train_original = X_train_original
        # Store X_val_original and X_test_original for DeepChem datasets
        self.X_val_original = X_val_original
        self.X_test_original = X_test_original
        # Also make it accessible to mechanism generator
        if self.mech_generator is not None:
            if self.mech_generator.predictor is None:
                self.mech_generator.predictor = self
            # Also store directly on mechanism generator for easier access
            self.mech_generator.X_train_original = X_train_original
        
        # Store k_shot for use in predictions
        self.k_shot = k_shot
        if k_shot > 0:
            logger.info(f"  [K-shot] Using {k_shot} few-shot examples per prediction")
        
        # Choose which set to use for acceptance evaluation
        if use_test_for_acceptance:
            if X_test is None or y_test is None:
                raise ValueError("use_test_for_acceptance=True requires X_test and y_test to be provided")
            X_accept = X_test
            y_accept = y_test
            set_name = "test"
            logger.info(f"  [AcceptEval] Using TEST set for acceptance evaluation (n={len(X_test)} samples)")
        else:
            X_accept = X_val
            y_accept = y_val
            set_name = "validation"
            logger.info(f"  [AcceptEval] Using VALIDATION set for acceptance evaluation (n={len(X_val)} samples)")
        
        # Dynamic acceptance evaluation sample size
        # Default: use full acceptance set (None means use all samples)
        if accept_eval_max is None:
            # Use full acceptance set - no restrictions
            accept_idx = np.arange(len(X_accept))
            accept_max = len(X_accept)
            accept_min = len(X_accept)
        else:
            accept_max = accept_eval_max
            if accept_eval_min is None:
                accept_min = max(10, accept_max // 4)  # Reasonable minimum
            else:
                accept_min = accept_eval_min
            
            # Create consistent acceptance subset for all iterations
            # This ensures current_loss and new_loss are computed on the SAME data
            rng = np.random.RandomState(RANDOM_STATE)
            n_eval = min(len(X_accept), accept_max)
            n_eval = max(n_eval, accept_min)
            
            if len(X_accept) > n_eval:
                accept_idx = rng.choice(len(X_accept), size=n_eval, replace=False)
            else:
                accept_idx = np.arange(len(X_accept))
        
        logger.info(f"  [AcceptEval] Using {len(accept_idx)}/{len(X_accept)} {set_name} samples (min={accept_min}, max={accept_max})")
        if len(accept_idx) <= 20:
            logger.info(f"  [AcceptEval] Selected indices: {accept_idx.tolist()}")
        else:
            logger.info(f"  [AcceptEval] Selected indices (first {MAX_FEATURE_IMPORTANCE_TOP}): {accept_idx[:MAX_FEATURE_IMPORTANCE_TOP].tolist()}...")
        
        # Verify unified scaling
        self._verify_unified_scaling(X_train, y_train, X_val, y_val)
        
        # Optionally focus acceptance subset on hardest samples using meaningful residuals
        try:
            if (self.use_ml_mechanism and 
                self.mech_generator is not None and
                getattr(self.mech_generator, "ml_mechanism", None) is not None and
                getattr(self.mech_generator.ml_mechanism, "is_trained", False) and
                hasattr(self.mech_generator.ml_mechanism, "model")):
                ml_model = self.mech_generator.ml_mechanism.model
                # Use accept_max directly without additional restrictions
                k_accept = accept_max
                # Classification: use class prediction errors (0/1) to define hardness; Regression: absolute error
                if self.task_type == "classification":
                    try:
                        ml_mech = self.mech_generator.ml_mechanism
                        class_to_idx = getattr(ml_mech, "_class_to_idx", None)
                        
                        # For classification: use class predictions only, no probabilities
                        y_pred_lbl = ml_model.predict(X_accept).astype(int)
                        y_true_vals = y_accept  # Keep original type
                        
                        # Compute 0/1 errors based on class predictions
                        residuals_val_full = np.zeros(len(y_true_vals), dtype=float)
                        for i in range(len(y_true_vals)):
                            true_cls = y_true_vals[i]
                            pred_cls_int = int(y_pred_lbl[i])
                            
                            # Map true class to index if needed
                            if class_to_idx is not None and true_cls in class_to_idx:
                                true_cls_int = class_to_idx[true_cls]
                            else:
                                try:
                                    true_cls_int = int(true_cls)
                                except (ValueError, TypeError):
                                    true_cls_int = 0
                            
                            # Residual is 1.0 if wrong, 0.0 if correct
                            residuals_val_full[i] = 1.0 if pred_cls_int != true_cls_int else 0.0
                    except Exception as e:
                        logger.warning(f"[AcceptSubset] Error computing acceptance residuals: {e}")
                        y_pred_lbl = ml_model.predict(X_accept).astype(int)
                        residuals_val_full = (y_pred_lbl != y_accept.astype(int)).astype(float)
                else:
                    try:
                        y_pred_accept_ml_full = ml_model.predict(X_accept).astype(float)
                        residuals_val_full = np.abs(y_accept - y_pred_accept_ml_full)
                    except Exception:
                        residuals_val_full = np.zeros_like(y_accept, dtype=float)
                topk_idx = np.argsort(residuals_val_full)[::-1][:k_accept]
                topk_set = set(topk_idx.tolist())
                accept_idx_hard = np.array([idx for idx in accept_idx if idx in topk_set], dtype=int)
                if len(accept_idx_hard) >= max(8, k_accept // 4):
                    accept_idx = accept_idx_hard
        except Exception:
            pass
        
        # Use this subset consistently for all evaluations
        X_accept_consistent = X_accept[accept_idx]
        y_accept_consistent = y_accept[accept_idx]
        logger.info(f"  Using consistent {set_name} subset: {len(X_accept_consistent)}/{len(X_accept)} samples")
        
        # CRITICAL FIX: Evaluate pre-training state and save as initial checkpoint
        # This ensures we can restore to the pre-training state if no improvements are made
        logger.info("  [Initial Checkpoint] Evaluating pre-training state...")
        routing_kwargs = {}
        if hasattr(self, 'attention_temp'):
            routing_kwargs['attention_temp'] = self.attention_temp
        if hasattr(self, 'min_ml_weight'):
            routing_kwargs['min_ml_weight'] = self.min_ml_weight
        if hasattr(self, 'max_ml_weight'):
            routing_kwargs['max_ml_weight'] = self.max_ml_weight
        if hasattr(self, 'hard_ml_gate_threshold'):
            routing_kwargs['hard_ml_gate_threshold'] = self.hard_ml_gate_threshold
        
        initial_metrics = self.evaluate(X_accept_consistent, y_accept_consistent, X_train, y_train, relax_routing=True,
                                      k_shot=self.k_shot, **routing_kwargs)
        initial_loss = initial_metrics['loss']
        logger.info(f"  [Initial Checkpoint] Pre-training loss: {initial_loss:.4f}")
        
        # Initialize best_snapshot with pre-training state
        best_loss = float(initial_loss)
        best_mechanisms = copy.deepcopy(self.mechanisms)
        # Track best metrics across all iterations (not just loss)
        best_acc = initial_metrics.get('accuracy', 0.0) if self.task_type == "classification" else None
        best_f1 = initial_metrics.get('f1', 0.0) if self.task_type == "classification" else None
        # For regression: track best R2 and MAE (initialize early to avoid AttributeError)
        if self.task_type == "regression":
            self._best_r2 = initial_metrics.get('r2', None)
            self._best_mae = initial_metrics.get('mae', None)
        else:
            self._best_r2 = None
            self._best_mae = None
        best_snapshot = {
            "iteration": 0,
            "loss": float(initial_loss),
            "mechanisms": copy.deepcopy(self.mechanisms),
            "mechanism_types": copy.deepcopy(self.mechanism_types),
            "mechanism_performance": copy.deepcopy(getattr(self, 'mechanism_performance_snapshot', {})),
            "mechanism_metrics": copy.deepcopy(getattr(self, 'mechanism_metrics_snapshot', {}))
        }
        logger.info(f"  [Initial Checkpoint] Saved pre-training state with {len(best_mechanisms)} mechanisms")
        if self.task_type == "classification":
            logger.info(f"  [Best Metrics] Initial: ACC={best_acc:.4f}, F1={best_f1:.4f}, Loss={best_loss:.4f}")
        
        for i in range(iterations):
            logger.info(f"\nIteration {i+1}/{iterations}")
            # LLM usage snapshot (before)
            try:
                stats_before = self.llm.get_stats()
            except Exception:
                stats_before = {"total_calls": 0, "batch_operations": 0, "avg_batch_size": 0}
            
            # Track if mechanisms were rejected during initial generation phase
            # (This happens before TextGrad, so we need to track it separately)
            mechanisms_rejected_at_generation = False
            
            # Generate new mechanisms if needed
            if len(self.mech_generator.unknown_mechanisms) == 0:
                logger.info("  [Mechanism Generation] Generating unknown mechanisms...")
                # Use ML residuals if available, otherwise fall back to simple baseline
                if ml_residuals is not None and len(ml_residuals) == len(X_train):
                    prediction_errors = np.abs(ml_residuals)
                else:
                    prediction_errors = np.abs(y_train - np.mean(y_train))  # Simple baseline
                new_mechanisms = self.mech_generator.generate_unknown_mechanisms(
                    X_train, y_train, prediction_errors, ml_residuals=ml_residuals
                )
                # Store mechanism count before adding new ones
                num_mechanisms_before = len(self.mechanisms)
                
                self.mech_generator.unknown_mechanisms = new_mechanisms
                self.mechanisms = self.mech_generator.get_all_mechanisms()
                self.mechanism_types = self.mech_generator.get_mechanism_types()
                logger.info(f"  ✓ Generated {len(new_mechanisms)} mechanisms")
                
                # CRITICAL: Evaluate newly generated LLM mechanisms in isolation first to get initial performance scores
                # This prevents them from dragging down the full system evaluation due to poor initial routing
                # Find indices of newly generated LLM mechanisms (they're at the end of the list)
                new_llm_indices = []
                for mech_idx in range(num_mechanisms_before, len(self.mechanisms)):
                    if self.mechanism_types[mech_idx] == "llm":
                        new_llm_indices.append(mech_idx)
                
                if len(new_llm_indices) > 0:
                    logger.info(f"  [New Mechanism Init] Evaluating {len(new_llm_indices)} new LLM mechanism(s) in isolation to get initial performance scores...")
                    # Evaluate new mechanisms in isolation (LLM-only) on a small sample
                    # Use a subset for faster evaluation
                    eval_sample_size = min(50, len(X_accept_consistent))
                    if eval_sample_size > 0:
                        sample_indices = np.random.choice(len(X_accept_consistent), eval_sample_size, replace=False)
                        X_sample = X_accept_consistent[sample_indices]
                        y_sample = y_accept_consistent[sample_indices]
                        
                        # Create a temporary MA-ICL instance with only the new LLM mechanisms
                        new_llm_mechanisms = [self.mechanisms[i] for i in new_llm_indices]
                        new_llm_types = ["llm"] * len(new_llm_indices)
                        
                        # Quick evaluation to get performance scores
                        # Import MultiAgentPredictor from the same module
                        temp_predictor = MultiAgentPredictor(
                            self.llm, new_llm_mechanisms, new_llm_types,
                            None,  # No ML mechanism
                            self.feature_cols, self.scaler, self.attention_temp,
                            task_type=self.task_type, class_names=self.class_names,
                            hard_ml_gate_threshold=0.0,
                            min_ml_weight=0.0, max_ml_weight=0.0
                        )
                        
                        # Get predictions from new mechanisms
                        x_dicts = [{self.feature_cols[j]: float(X_sample[i, j]) for j in range(len(self.feature_cols))} for i in range(len(X_sample))]
                        k_shot = getattr(self, 'k_shot', 0)
                        if k_shot > 0 and len(X_train) > 0:
                            # Compute ML residuals on training data to prioritize high-residual examples
                            ml_residuals_train = None
                            if (self.use_ml_mechanism and 
                                self.mech_generator is not None and
                                hasattr(self.mech_generator, 'ml_mechanism') and
                                self.mech_generator.ml_mechanism is not None and
                                hasattr(self.mech_generator.ml_mechanism, 'is_trained') and
                                self.mech_generator.ml_mechanism.is_trained):
                                try:
                                    _, _, ml_residuals_train, _ = compute_ml_residuals(
                                        self.mech_generator.ml_mechanism,
                                        X_train, y_train, self.feature_cols,
                                        class_names=self.class_names,
                                        task_type=self.task_type
                                    )
                                except Exception:
                                    pass
                            
                            few_shot_examples = retrieve_few_shot_examples(
                                X_train, y_train, self.feature_cols, k_shot,
                                task_type=self.task_type, class_names=self.class_names,
                                ml_residuals=ml_residuals_train, prioritize_residuals=True
                            )
                            few_shot_list = [few_shot_examples] * len(x_dicts)
                        else:
                            few_shot_list = [[]] * len(x_dicts)
                        
                        batch_outputs = temp_predictor.predict_batch(x_dicts, few_shot_list=few_shot_list)
                        new_mech_predictions = [out[0] for out in batch_outputs]
                        y_true_sample = np.array([float(y_sample[i]) for i in range(len(y_sample))])
                        
                        # Calculate performance for each new mechanism
                        if self.task_type == "classification":
                            y_pred_sample = np.array(new_mech_predictions)
                            if self.class_names is not None and len(self.class_names) > 2:
                                y_pred_int = np.round(y_pred_sample).astype(int)
                                y_true_int = y_true_sample.astype(int)
                                f1 = f1_score(y_true_int, y_pred_int, average='weighted', zero_division=0)
                                acc = accuracy_score(y_true_int, y_pred_int)
                            else:
                                y_pred_int = (y_pred_sample >= 0.5).astype(int)
                                y_true_int = y_true_sample.astype(int)
                                f1 = f1_score(y_true_int, y_pred_int, zero_division=0)
                                acc = accuracy_score(y_true_int, y_pred_int)
                            
                            classification_loss = getattr(self, "classification_loss_metric", "f1")
                            if classification_loss == "acc":
                                initial_perf_score = acc
                            else:
                                initial_perf_score = f1
                        else:
                            # Regression
                            y_pred_sample = np.array(new_mech_predictions)
                            mae = mean_absolute_error(y_true_sample, y_pred_sample)
                            regression_loss = getattr(self, "regression_loss_metric", "mae")
                            if regression_loss == "r2":
                                try:
                                    r2_val = r2_score(y_true_sample, y_pred_sample)
                                    initial_perf_score = max(0.0, r2_val)
                                except Exception:
                                    initial_perf_score = 1.0 / (0.01 + mae)
                            else:
                                initial_perf_score = 1.0 / (0.01 + mae)
                        
                        # Set initial performance scores for new mechanisms
                        # IMPROVED: Use a more balanced approach that gives new mechanisms a fair chance
                        # Instead of being too conservative, use the isolated performance as a starting point
                        # but ensure it's at least 0.3 (above the minimum routing weight of 0.2)
                        # This allows new mechanisms to contribute while still being weighted appropriately
                        baseline_perf = 0.3  # Minimum to ensure meaningful contribution
                        # Use max of isolated performance and baseline, but cap at reasonable level
                        conservative_perf = max(baseline_perf, min(0.7, initial_perf_score))
                        
                        # Store in snapshot so routing can use it
                        if not hasattr(self, 'mechanism_performance_snapshot'):
                            self.mechanism_performance_snapshot = {}
                        for idx in new_llm_indices:
                            self.mechanism_performance_snapshot[idx] = conservative_perf
                        
                        logger.info(f"  [New Mechanism Init] Set initial performance score: {conservative_perf:.4f} (isolated: {initial_perf_score:.4f}) for {len(new_llm_indices)} new mechanism(s)")
                
                # CRITICAL: Evaluate newly generated mechanisms in full system and reject if they degrade significantly
                # This ensures first iteration starts close to ML baseline
                routing_kwargs_temp = {}
                if hasattr(self, 'attention_temp'):
                    routing_kwargs_temp['attention_temp'] = self.attention_temp
                if hasattr(self, 'min_ml_weight'):
                    routing_kwargs_temp['min_ml_weight'] = self.min_ml_weight
                if hasattr(self, 'max_ml_weight'):
                    routing_kwargs_temp['max_ml_weight'] = self.max_ml_weight
                if hasattr(self, 'hard_ml_gate_threshold'):
                    routing_kwargs_temp['hard_ml_gate_threshold'] = self.hard_ml_gate_threshold
                
                new_mech_metrics = self.evaluate(X_accept_consistent, y_accept_consistent, X_train, y_train, 
                                                relax_routing=True, k_shot=self.k_shot, **routing_kwargs_temp)
                new_mech_loss = new_mech_metrics['loss']
                
                # For regression: allow mechanisms that degrade by up to 0.7 vs best-seen
                # This gives TextGrad a chance to optimize them (they might improve after optimization)
                # The key is that we'll use STRICT criteria when accepting TextGrad-optimized versions
                # For classification: allow mechanisms that degrade by up to 0.3 vs best-seen
                # We'll use stricter criteria when accepting TextGrad updates
                max_degradation_for_new_mech = 0.7 if self.task_type == "regression" else 0.3
                if new_mech_loss > best_loss + max_degradation_for_new_mech:
                    logger.warning(f"  ⚠️  Newly generated mechanisms degrade performance significantly (loss={new_mech_loss:.4f} vs best={best_loss:.4f}, diff={new_mech_loss - best_loss:+.4f})")
                    logger.warning(f"  ⚠️  Rejecting new mechanisms and keeping best state to stay close to ML baseline")
                    # Remove the newly generated mechanisms
                    self.mech_generator.unknown_mechanisms = []
                    self.mechanisms = self.mech_generator.get_all_mechanisms()
                    self.mechanism_types = self.mech_generator.get_mechanism_types()
                    logger.info(f"  ✓ Reverted to best state with {len(self.mechanisms)} mechanisms")
                    # Track rejection (will be used to update counter later)
                    mechanisms_rejected_at_generation = True
                else:
                    if new_mech_loss > best_loss:
                        logger.info(f"  ✓ New mechanisms accepted (loss={new_mech_loss:.4f} vs best={best_loss:.4f}, diff={new_mech_loss - best_loss:+.4f})")
                        logger.info(f"  ✓ Will allow TextGrad to optimize them - using stricter acceptance criteria for optimized versions")
                    else:
                        logger.info(f"  ✓ New mechanisms acceptable (loss={new_mech_loss:.4f} vs best={best_loss:.4f})")
            
            # CRITICAL: Ensure mechanisms are synced with generator before evaluation
            # This ensures current_loss is computed on the correct state
            # Sync mechanisms from generator to ensure consistency
            self.mechanisms = self.mech_generator.get_all_mechanisms()
            self.mechanism_types = self.mech_generator.get_mechanism_types()
            
            # Evaluate current performance ON CONSISTENT SUBSET
            # Use routing parameters from instance if available
            routing_kwargs = {}
            if hasattr(self, 'attention_temp'):
                routing_kwargs['attention_temp'] = self.attention_temp
            if hasattr(self, 'min_ml_weight'):
                routing_kwargs['min_ml_weight'] = self.min_ml_weight
            if hasattr(self, 'max_ml_weight'):
                routing_kwargs['max_ml_weight'] = self.max_ml_weight
            if hasattr(self, 'hard_ml_gate_threshold'):
                routing_kwargs['hard_ml_gate_threshold'] = self.hard_ml_gate_threshold
            
            # For DeepChem datasets, pass X_original for acceptance evaluation
            eval_kwargs = dict(routing_kwargs)
            if hasattr(self, 'X_val_original') and hasattr(self, 'X_test_original'):
                # Determine which X_original to use based on which set is used for acceptance
                if use_test_for_acceptance and self.X_test_original is not None:
                    # Use consistent subset of X_test_original matching accept_idx
                    if len(accept_idx) == len(X_accept):
                        X_accept_original = self.X_test_original
                    else:
                        X_accept_original = [self.X_test_original[i] for i in accept_idx] if len(accept_idx) <= len(self.X_test_original) else None
                    if X_accept_original is not None:
                        eval_kwargs['X_original'] = X_accept_original
                elif not use_test_for_acceptance and self.X_val_original is not None:
                    # Use consistent subset of X_val_original matching accept_idx
                    if len(accept_idx) == len(X_accept):
                        X_accept_original = self.X_val_original
                    else:
                        X_accept_original = [self.X_val_original[i] for i in accept_idx] if len(accept_idx) <= len(self.X_val_original) else None
                    if X_accept_original is not None:
                        eval_kwargs['X_original'] = X_accept_original
                # Always use X_train_original as pool for few-shot examples
                if self.X_train_original is not None:
                    eval_kwargs['X_pool_original'] = self.X_train_original
            
            metrics = self.evaluate(X_accept_consistent, y_accept_consistent, X_train, y_train, relax_routing=True, 
                                     k_shot=self.k_shot, **eval_kwargs)
            current_loss = metrics['loss']
            self._last_eval_loss = current_loss  # Store for restoration decision
            logger.info(f"  Current loss: {current_loss:.4f}")
            
            # Log routing weights at iteration level (from predictor if available)
            # This will be logged by the predictor itself during evaluation
            # Track best-so-far mechanisms by validation loss AND metrics
            current_acc = metrics.get('accuracy', 0.0) if self.task_type == "classification" else None
            current_f1 = metrics.get('f1', 0.0) if self.task_type == "classification" else None
            current_r2 = metrics.get('r2', None) if self.task_type == "regression" else None
            current_mae = metrics.get('mae', None) if self.task_type == "regression" else None
            
            # CRITICAL FIX: Update best_loss and best metrics if current state is better
            # This ensures acceptance logic always compares against the true best seen so far
            # We update best_loss/metrics here for comparison purposes, but best_snapshot is updated after TextGrad
            if current_loss < best_loss:
                best_loss = float(current_loss)
                if self.task_type == "classification":
                    if current_acc is not None and (best_acc is None or current_acc > best_acc):
                        best_acc = current_acc
                    if current_f1 is not None and (best_f1 is None or current_f1 > best_f1):
                        best_f1 = current_f1
                else:
                    # Regression: update best R2 and MAE
                    if current_r2 is not None:
                        if not hasattr(self, '_best_r2') or self._best_r2 is None or current_r2 > self._best_r2:
                            self._best_r2 = current_r2
                    if current_mae is not None:
                        if not hasattr(self, '_best_mae') or self._best_mae is None or current_mae < self._best_mae:
                            self._best_mae = current_mae
            
            # NOTE: Don't update best_snapshot here - wait until after TextGrad updates
            # This ensures the checkpoint reflects the actual accepted state, not a pre-TextGrad state
            # The checkpoint will be updated after TextGrad if the update is accepted (see line 7517-7558)
            
            # NOTE: Do NOT log metrics here - wait until after acceptance/rejection decision
            # to ensure all metrics (loss, r2, mae, accuracy, f1) are from the same state
            
            # Initialize variables for tracking acceptance/rejection and new metrics
            accepted = 0
            rejected = 0
            new_metrics = None  # Will be set if TextGrad updates are attempted
            
            # Update rejected counter if mechanisms were rejected during generation phase
            if mechanisms_rejected_at_generation:
                rejected = 1
            
            # Apply TextGrad optimization
            llm_mechanisms = [m for i, m in enumerate(self.mechanisms) if self.mechanism_types[i] == "llm"]
            if len(llm_mechanisms) > 0:
                logger.info("  [TextGrad] Refining LLM mechanisms...")
                
                # Build enhanced error feedback using TextGrad's diagnostic methods
                try:
                    # Use provided ml_residuals if available (already filtered to top_k if from train_on_residuals)
                    # Otherwise, recompute from full training set
                    ml_residuals_for_feedback = ml_residuals
                    ml_predictions_for_display = None
                    
                    if ml_residuals_for_feedback is None:
                        # Only recompute if not provided (fallback case)
                        if (self.use_ml_mechanism and 
                            self.mech_generator is not None and
                            getattr(self.mech_generator, "ml_mechanism", None) is not None and
                            getattr(self.mech_generator.ml_mechanism, "is_trained", False) and
                            hasattr(self.mech_generator.ml_mechanism, "model") and
                            hasattr(self.mech_generator.ml_mechanism.model, "predict")):
                            
                            ml_model = self.mech_generator.ml_mechanism.model
                            y_pred_train_ml = ml_model.predict(X_train).astype(float)
                            if self.task_type == "classification":
                                # For classification: residual is binary (0 = match, 1 = mismatch)
                                # Compute binary residuals: 1.0 if wrong, 0.0 if correct
                                y_train_int = y_train.astype(int)
                                y_pred_train_ml_int = y_pred_train_ml.astype(int)
                                ml_residuals_for_feedback = (y_train_int != y_pred_train_ml_int).astype(float)
                                ml_predictions_for_display = y_pred_train_ml_int  # Store predictions separately
                            else:
                                # For regression: residual is the difference in values
                                ml_residuals_for_feedback = y_train - y_pred_train_ml
                                ml_predictions_for_display = None
                    else:
                        # ml_residuals provided - use them directly (already filtered to top_k if from train_on_residuals)
                        # For display purposes, we may need predictions, but they're not critical for feedback
                        if (self.use_ml_mechanism and 
                            self.mech_generator is not None and
                            getattr(self.mech_generator, "ml_mechanism", None) is not None and
                            getattr(self.mech_generator.ml_mechanism, "is_trained", False) and
                            hasattr(self.mech_generator.ml_mechanism, "model") and
                            hasattr(self.mech_generator.ml_mechanism.model, "predict") and
                            self.task_type == "classification"):
                            # For classification, we need predictions for display
                            ml_model = self.mech_generator.ml_mechanism.model
                            y_pred_train_ml = ml_model.predict(X_train).astype(int)
                            ml_predictions_for_display = y_pred_train_ml
                    
                    # Use TextGrad's enhanced error feedback builder with current metrics
                    error_feedback_full = self.textgrad._build_enhanced_error_feedback(
                        X_train, y_train, ml_residuals_for_feedback, current_loss, self.dataset_name,
                        current_metrics=metrics,  # Pass current metrics (R2, MAE, etc.)
                        X_train_original=self.X_train_original  # Pass SMILES strings for DeepChem datasets
                    )
                    
                    # Always log sample residuals for transparency (shows what the LLM sees)
                    if ml_residuals_for_feedback is not None and len(ml_residuals_for_feedback) > 0:
                        # ML is enabled - show samples with ML predictions and residuals
                        # For classification, show worst errors (residual=1.0), for regression show highest absolute residuals
                        if self.task_type == "classification":
                            # For classification: show samples where residual=1.0 (mismatched)
                            worst_idx = np.where(ml_residuals_for_feedback > 0.5)[0][:MAX_WORST_RESIDUAL_SAMPLES]  # Show mismatched samples
                            if len(worst_idx) == 0:
                                # If all correct, show any samples
                                worst_idx = np.arange(min(MAX_WORST_RESIDUAL_SAMPLES, len(ml_residuals_for_feedback)))
                        else:
                            # Diversify residual samples: show top worst + some from different quantiles
                            # This prevents bias from always showing the same samples
                            abs_residuals = np.abs(ml_residuals_for_feedback)
                            sorted_indices = np.argsort(abs_residuals)[::-1]
                            
                            # Top 5 worst (increased from 3 for better error pattern coverage)
                            worst_idx = sorted_indices[:MAX_WORST_RESIDUAL_SAMPLES].tolist()
                            
                            # Add samples from different quantiles to show diverse error patterns
                            n_samples = len(ml_residuals_for_feedback)
                            if n_samples > 10:
                                # 75th percentile
                                q75_idx = sorted_indices[max(0, int(0.25 * n_samples)):int(0.35 * n_samples)]
                                if len(q75_idx) > 0:
                                    worst_idx.append(np.random.choice(q75_idx))
                                # 50th percentile (median)
                                q50_idx = sorted_indices[max(0, int(0.45 * n_samples)):int(0.55 * n_samples)]
                                if len(q50_idx) > 0:
                                    worst_idx.append(np.random.choice(q50_idx))
                            
                            # Re-sort by residual magnitude (descending) to ensure proper ordering
                            worst_idx = np.array(worst_idx)
                            worst_idx = worst_idx[np.argsort(abs_residuals[worst_idx])[::-1]][:MAX_WORST_RESIDUAL_SAMPLES].tolist()
                        
                        # Check if this is a DeepChem dataset and we have SMILES strings
                        is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                                      all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
                        use_smiles = (is_deepchem and hasattr(self, 'X_train_original') and 
                                     self.X_train_original is not None and len(self.X_train_original) > 0 and
                                     isinstance(self.X_train_original[0], dict) and 'SMILES' in self.X_train_original[0])
                        
                        feature_names = self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]
                        logger.info(f"  [Residual Samples] Showing top {min(MAX_WORST_RESIDUAL_SAMPLES, len(worst_idx))} worst ML residuals (what LLM sees):")
                        for rank, idx in enumerate(worst_idx[:MAX_WORST_RESIDUAL_SAMPLES], 1):
                            # Use SMILES strings for DeepChem datasets, otherwise use feature values
                            if use_smiles and idx < len(self.X_train_original):
                                original_feat = self.X_train_original[idx]
                                if isinstance(original_feat, dict) and 'SMILES' in original_feat:
                                    key_feats = f"SMILES={original_feat['SMILES']}"
                                else:
                                    # Fallback to feature values
                                    feat_vals = {feature_names[j]: float(X_train[idx, j]) for j in range(min(len(feature_names), X_train.shape[1]))}
                                    key_feats = ', '.join([f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_EXAMPLE]])
                            else:
                                feat_vals = {feature_names[j]: float(X_train[idx, j]) for j in range(min(len(feature_names), X_train.shape[1]))}
                                key_feats = ', '.join([f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_EXAMPLE]])
                            
                            if self.task_type == "regression":
                                y_pred = ml_residuals_for_feedback[idx] + y_train[idx]
                                logger.info(f"    {rank}. {key_feats} → pred={y_pred:.2f}, true={y_train[idx]:.2f}, residual={ml_residuals_for_feedback[idx]:+.2f}")
                            else:
                                # For classification: use stored predictions directly
                                if ml_predictions_for_display is not None:
                                    pred_idx = int(ml_predictions_for_display[idx])
                                else:
                                    # Fallback: if predictions not stored, can't recover
                                    pred_idx = -1
                                true_idx = int(y_train[idx])
                                class_names = getattr(self, 'class_names', None)
                                
                                # Determine if matched or mismatched
                                is_matched = ml_residuals_for_feedback[idx] < 0.5  # residual < 0.5 means match
                                match_status = "matched" if is_matched else "mismatched"
                                
                                if class_names:
                                    pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
                                    true_name = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
                                    logger.info(f"    {rank}. {key_feats} → pred={pred_name}, true={true_name}, {match_status}")
                                else:
                                    logger.info(f"    {rank}. {key_feats} → pred={pred_idx}, true={true_idx}, {match_status}")
                    else:
                        # ML is disabled - show training samples without prediction/residual information
                        # Select diverse samples from training set
                        n_samples = len(X_train)
                        if n_samples > 0:
                            # Select samples from different parts of the dataset for diversity
                            sample_indices = []
                            if n_samples <= MAX_WORST_RESIDUAL_SAMPLES:
                                sample_indices = list(range(n_samples))
                            else:
                                # Select from beginning, middle, and end
                                sample_indices = (
                                    list(range(min(3, n_samples))) +  # First few
                                    list(range(n_samples // 2, n_samples // 2 + min(3, n_samples // 2))) +  # Middle
                                    list(range(max(0, n_samples - 3), n_samples))  # Last few
                                )
                                # Remove duplicates and limit to MAX_WORST_RESIDUAL_SAMPLES
                                sample_indices = list(dict.fromkeys(sample_indices))[:MAX_WORST_RESIDUAL_SAMPLES]
                            
                            # Check if this is a DeepChem dataset and we have SMILES strings
                            is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                                          all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
                            use_smiles = (is_deepchem and hasattr(self, 'X_train_original') and 
                                         self.X_train_original is not None and len(self.X_train_original) > 0 and
                                         isinstance(self.X_train_original[0], dict) and 'SMILES' in self.X_train_original[0])
                            
                            feature_names = self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]
                            logger.info(f"  [Training Samples] Showing {len(sample_indices)} training samples (ML disabled, no residuals available):")
                            for rank, idx in enumerate(sample_indices, 1):
                                # Use SMILES strings for DeepChem datasets, otherwise use feature values
                                if use_smiles and idx < len(self.X_train_original):
                                    original_feat = self.X_train_original[idx]
                                    if isinstance(original_feat, dict) and 'SMILES' in original_feat:
                                        key_feats = f"SMILES={original_feat['SMILES']}"
                                    else:
                                        # Fallback to feature values
                                        feat_vals = {feature_names[j]: float(X_train[idx, j]) for j in range(min(len(feature_names), X_train.shape[1]))}
                                        key_feats = ', '.join([f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_EXAMPLE]])
                                else:
                                    feat_vals = {feature_names[j]: float(X_train[idx, j]) for j in range(min(len(feature_names), X_train.shape[1]))}
                                    key_feats = ', '.join([f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_EXAMPLE]])
                                
                                if self.task_type == "regression":
                                    logger.info(f"    {rank}. {key_feats} → true={y_train[idx]:.2f}")
                                else:
                                    true_idx = int(y_train[idx])
                                    class_names = getattr(self, 'class_names', None)
                                    if class_names:
                                        true_name = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
                                        logger.info(f"    {rank}. {key_feats} → true={true_name}")
                                    else:
                                        logger.info(f"    {rank}. {key_feats} → true={true_idx}")
                except Exception as e:
                    logger.warning(f"  [TextGrad] Failed to build enhanced error feedback: {e}, using simple feedback")
                    error_feedback_full = f"Current loss: {current_loss:.4f}. Improve predictions."
                
                error_feedbacks = [error_feedback_full] * len(llm_mechanisms)
                # Use single-call optimization per mechanism
                # Two-step optimization (gradients + apply) for higher-quality updates
                # Pass ML context for enhanced feedback
                ml_mechanism_for_textgrad = None
                mechanism_performance_for_textgrad = None
                mechanism_metrics_for_textgrad = None
                if (self.use_ml_mechanism and 
                    self.mech_generator is not None and
                    getattr(self.mech_generator, "ml_mechanism", None) is not None):
                    ml_mechanism_for_textgrad = self.mech_generator.ml_mechanism
                    # Get mechanism performance from snapshot if available
                    # Map full mechanism indices to LLM mechanism indices (0, 1, 2...)
                    if hasattr(self, 'mechanism_performance_snapshot') and self.mechanism_performance_snapshot:
                        # Create mapping: llm_idx -> full_idx -> performance
                        llm_indices = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "llm"]
                        mechanism_performance_for_textgrad = {
                            llm_local_idx: self.mechanism_performance_snapshot.get(full_idx, 1.0)
                            for llm_local_idx, full_idx in enumerate(llm_indices)
                        }
                    # Get mechanism metrics (R2, MAE, F1, accuracy) if available
                    if hasattr(self, 'mechanism_metrics_snapshot') and self.mechanism_metrics_snapshot:
                        llm_indices = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "llm"]
                        mechanism_metrics_for_textgrad = {
                            llm_local_idx: self.mechanism_metrics_snapshot.get(full_idx, {})
                            for llm_local_idx, full_idx in enumerate(llm_indices)
                        }
                
                # Build iteration history for TextGrad (memory from previous iterations)
                # Only include history from previous iterations (not current)
                iteration_history_for_textgrad = self.textgrad_iteration_history.copy() if hasattr(self, 'textgrad_iteration_history') else []
                
                updated_mechanisms = self.textgrad.optimize_batch(
                    llm_mechanisms, error_feedbacks, ml_residuals=ml_residuals,
                    X_train=X_train, y_train=y_train,
                    ml_mechanism=ml_mechanism_for_textgrad,
                    mechanism_performance=mechanism_performance_for_textgrad,
                    mechanism_metrics=mechanism_metrics_for_textgrad,
                    iteration_history=iteration_history_for_textgrad
                )
                
                # Keep a copy for potential rollback
                prev_mechanisms = copy.deepcopy(self.mechanisms)
                
                # Tentatively update LLM mechanisms
                llm_idx = 0
                # Log proposed changes for transparency
                try:
                    logger.info("  [TextGrad] Proposed mechanism updates (BEFORE → AFTER):")
                    for i_m, mtype in enumerate(self.mechanism_types):
                        if mtype == "llm":
                            before_text = prev_mechanisms[i_m]
                            after_text = updated_mechanisms[llm_idx]
                            if before_text != after_text:
                                logger.info(f"    • Mechanism idx={i_m}")
                                logger.info("      --- BEFORE ---")
                                logger.info(before_text)
                                logger.info("      --- AFTER ----")
                                logger.info(after_text)
                            llm_idx += 1
                except Exception:
                    pass
                
                llm_idx = 0
                for i_m, mtype in enumerate(self.mechanism_types):
                    if mtype == "llm":
                        self.mechanisms[i_m] = updated_mechanisms[llm_idx]
                        llm_idx += 1
                
                # Re-evaluate ON THE SAME CONSISTENT SUBSET
                new_metrics = self.evaluate(X_accept_consistent, y_accept_consistent, X_train, y_train, relax_routing=True,
                                            k_shot=self.k_shot, **routing_kwargs)
                new_loss = new_metrics['loss']
                
                # Get current metrics for comparison (accuracy, F1)
                current_acc = metrics.get('accuracy', 0.0) if metrics else 0.0
                current_f1 = metrics.get('f1', 0.0) if metrics else 0.0
                new_acc = new_metrics.get('accuracy', 0.0)
                new_f1 = new_metrics.get('f1', 0.0)
                
                # CRITICAL FIX: Compare against BEST-SEEN metrics, not just current iteration start
                # This prevents accepting updates that degrade from the best we've seen
                best_acc_for_comparison = best_acc if self.task_type == "classification" and best_acc is not None else current_acc
                best_f1_for_comparison = best_f1 if self.task_type == "classification" and best_f1 is not None else current_f1
                best_loss_for_comparison = best_loss
                
                # ADAPTIVE acceptance threshold
                # IMPROVED: More lenient threshold for regression to allow mechanism improvements
                # Even if overall loss doesn't improve much, if a mechanism improves significantly, accept it
                # This is especially important when LLM mechanisms have low routing weight
                base_threshold = IMPROVEMENT_THRESHOLD_CLASSIFICATION if self.task_type == "classification" else 0.005
                
                # Reset acceptance/rejection flags (will be set below)
                accepted = 0
                rejected = 0
                
                improvement = current_loss - new_loss
                # Also compute improvement vs best-seen loss
                improvement_vs_best = best_loss_for_comparison - new_loss
                
                # Dynamic threshold based on iteration (more lenient early on)
                # For very poor initial mechanisms (loss > 0.5 for classification), be slightly more lenient
                # BUT: Only allow very small degradation (0.01-0.02) to prevent accepting clearly worse updates
                if self.task_type == "classification" and current_loss > 0.5 and i < 3:
                    # Only in first 3 iterations, allow tiny degradation (0.01) for exploration
                    # This prevents getting stuck but doesn't accept clearly worse updates
                    max_degradation = 0.01  # Allow only 0.01 worse loss (very small)
                    if new_loss <= current_loss + max_degradation:
                        adaptive_threshold = -max_degradation  # Accept only tiny degradation
                    else:
                        adaptive_threshold = base_threshold * (1.0 - 0.1 * i)  # Too much degradation, use normal threshold
                else:
                    adaptive_threshold = base_threshold * (1.0 - 0.1 * i)  # Decreases over iterations
                
                # Multiple acceptance criteria
                accept = False
                reason = ""
                
                # PRIORITY: Check performance metrics FIRST (R²/MAE for regression, ACC/F1 for classification)
                # Only use loss as a secondary criterion
                if self.task_type == "regression":
                    # For regression: prioritize R² and MAE
                    current_r2 = metrics.get('r2', None)
                    current_mae = metrics.get('mae', None)
                    new_r2 = new_metrics.get('r2', None)
                    new_mae = new_metrics.get('mae', None)
                    
                    best_r2 = getattr(self, '_best_r2', current_r2)
                    best_mae = getattr(self, '_best_mae', current_mae)
                    
                    r2_improved_vs_best = (new_r2 is not None and best_r2 is not None and new_r2 > best_r2)
                    mae_improved_vs_best = (new_mae is not None and best_mae is not None and new_mae < best_mae)
                    r2_improved_vs_current = (new_r2 is not None and current_r2 is not None and new_r2 > current_r2)
                    mae_improved_vs_current = (new_mae is not None and current_mae is not None and new_mae < current_mae)
                    
                    # Accept if metrics improve significantly (primary criterion)
                    metric_improvement_threshold_r2 = 0.005  # Require at least 0.005 improvement in R2
                    metric_improvement_threshold_mae = 0.005  # Require at least 0.005 improvement in MAE
                    
                    r2_improvement_vs_best_val = (new_r2 - best_r2) if (new_r2 is not None and best_r2 is not None) else 0.0
                    mae_improvement_vs_best_val = (best_mae - new_mae) if (best_mae is not None and new_mae is not None) else 0.0
                    r2_improvement_vs_current_val = (new_r2 - current_r2) if (new_r2 is not None and current_r2 is not None) else 0.0
                    mae_improvement_vs_current_val = (current_mae - new_mae) if (current_mae is not None and new_mae is not None) else 0.0
                    
                    # CRITICAL: Accept if metrics improve vs best OR vs current (whichever is better)
                    # This ensures we accept improvements even if current state is better than best-seen
                    if (r2_improvement_vs_best_val >= metric_improvement_threshold_r2 or 
                        mae_improvement_vs_best_val >= metric_improvement_threshold_mae or
                        r2_improvement_vs_current_val >= metric_improvement_threshold_r2 or
                        mae_improvement_vs_current_val >= metric_improvement_threshold_mae):
                        accept = True
                        # Prefer logging vs best if both improve, otherwise log vs current
                        if r2_improvement_vs_best_val >= metric_improvement_threshold_r2 and mae_improvement_vs_best_val >= metric_improvement_threshold_mae:
                            reason = f"PRIMARY: metric improvement vs best (R2: {best_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_best_val:.4f}), MAE: {best_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_best_val:.4f}))"
                        elif r2_improvement_vs_best_val >= metric_improvement_threshold_r2:
                            reason = f"PRIMARY: R2 improvement vs best (R2: {best_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_best_val:.4f}))"
                        elif mae_improvement_vs_best_val >= metric_improvement_threshold_mae:
                            reason = f"PRIMARY: MAE improvement vs best (MAE: {best_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_best_val:.4f}))"
                        elif r2_improvement_vs_current_val >= metric_improvement_threshold_r2 and mae_improvement_vs_current_val >= metric_improvement_threshold_mae:
                            reason = f"PRIMARY: metric improvement vs current (R2: {current_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_current_val:.4f}), MAE: {current_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_current_val:.4f}))"
                        elif r2_improvement_vs_current_val >= metric_improvement_threshold_r2:
                            reason = f"PRIMARY: R2 improvement vs current (R2: {current_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_current_val:.4f}))"
                        else:
                            reason = f"PRIMARY: MAE improvement vs current (MAE: {current_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_current_val:.4f}))"
                
                elif self.task_type == "classification":
                    # For classification: prioritize Accuracy and F1
                    acc_improvement_vs_best = new_acc - best_acc_for_comparison
                    f1_improvement_vs_best = new_f1 - best_f1_for_comparison
                    acc_improvement_vs_current = new_acc - current_acc
                    f1_improvement_vs_current = new_f1 - current_f1
                    
                    metric_improvement_threshold = 0.005  # Require at least 0.5% improvement
                    
                    # CRITICAL: Accept if metrics improve vs best OR vs current (whichever is better)
                    # This ensures we accept improvements even if current state is better than best-seen
                    if (acc_improvement_vs_best >= metric_improvement_threshold or 
                        f1_improvement_vs_best >= metric_improvement_threshold or
                        acc_improvement_vs_current >= metric_improvement_threshold or
                        f1_improvement_vs_current >= metric_improvement_threshold):
                        accept = True
                        # Prefer logging vs best if both improve, otherwise log vs current
                        if acc_improvement_vs_best >= metric_improvement_threshold and f1_improvement_vs_best >= metric_improvement_threshold:
                            reason = f"PRIMARY: metric improvement vs best (ACC: {best_acc_for_comparison:.4f} → {new_acc:.4f} (+{acc_improvement_vs_best:.4f}), F1: {best_f1_for_comparison:.4f} → {new_f1:.4f} (+{f1_improvement_vs_best:.4f}))"
                        elif acc_improvement_vs_best >= metric_improvement_threshold:
                            reason = f"PRIMARY: accuracy improvement vs best (ACC: {best_acc_for_comparison:.4f} → {new_acc:.4f} (+{acc_improvement_vs_best:.4f}))"
                        elif f1_improvement_vs_best >= metric_improvement_threshold:
                            reason = f"PRIMARY: F1 improvement vs best (F1: {best_f1_for_comparison:.4f} → {new_f1:.4f} (+{f1_improvement_vs_best:.4f}))"
                        elif acc_improvement_vs_current >= metric_improvement_threshold and f1_improvement_vs_current >= metric_improvement_threshold:
                            reason = f"PRIMARY: metric improvement vs current (ACC: {current_acc:.4f} → {new_acc:.4f} (+{acc_improvement_vs_current:.4f}), F1: {current_f1:.4f} → {new_f1:.4f} (+{f1_improvement_vs_current:.4f}))"
                        elif acc_improvement_vs_current >= metric_improvement_threshold:
                            reason = f"PRIMARY: accuracy improvement vs current (ACC: {current_acc:.4f} → {new_acc:.4f} (+{acc_improvement_vs_current:.4f}))"
                        else:
                            reason = f"PRIMARY: F1 improvement vs current (F1: {current_f1:.4f} → {new_f1:.4f} (+{f1_improvement_vs_current:.4f}))"
                
                # Criterion 1 (SECONDARY): Significant improvement or acceptable degradation in loss
                # Only check if metrics didn't improve
                if not accept and improvement >= adaptive_threshold:
                    # Check if metrics degraded significantly for classification (compared to BEST-SEEN)
                    if self.task_type == "classification":
                        # Compare against best-seen metrics, not just current iteration start
                        acc_degradation_vs_best = best_acc_for_comparison - new_acc
                        f1_degradation_vs_best = best_f1_for_comparison - new_f1
                        acc_degradation_vs_current = current_acc - new_acc
                        f1_degradation_vs_current = current_f1 - new_f1
                        max_metric_degradation = 0.05  # Reject if accuracy or F1 drops by more than 5% vs best
                        max_metric_degradation_current = 0.10  # Allow more degradation vs current (exploration)
                        
                        # Reject if metrics degrade significantly vs best-seen OR vs current (if current is close to best)
                        if (acc_degradation_vs_best > max_metric_degradation or f1_degradation_vs_best > max_metric_degradation or
                            (acc_degradation_vs_current > max_metric_degradation_current or f1_degradation_vs_current > max_metric_degradation_current)):
                            # Reject: metrics degraded too much
                            accept = False
                            reason = f"rejected: metric degradation vs best (ACC: {best_acc_for_comparison:.4f} → {new_acc:.4f} ({acc_degradation_vs_best:+.4f}), F1: {best_f1_for_comparison:.4f} → {new_f1:.4f} ({f1_degradation_vs_best:+.4f}))"
                        else:
                            accept = True
                            if improvement >= 0:
                                reason = f"improvement {improvement:.4f} >= threshold {adaptive_threshold:.4f}"
                            else:
                                reason = f"acceptable degradation {improvement:.4f} (within threshold {adaptive_threshold:.4f}, exploring from poor state)"
                    else:
                        # Regression: use loss-based acceptance, but also check R2 and MAE
                        current_r2 = metrics.get('r2', None)
                        current_mae = metrics.get('mae', None)
                        new_r2 = new_metrics.get('r2', None)
                        new_mae = new_metrics.get('mae', None)
                        
                        # Check if metrics improved
                        r2_improved = (new_r2 is not None and current_r2 is not None and new_r2 > current_r2)
                        mae_improved = (new_mae is not None and current_mae is not None and new_mae < current_mae)
                        
                        # Also compare against best-seen metrics
                        best_r2 = getattr(self, '_best_r2', current_r2)
                        best_mae = getattr(self, '_best_mae', current_mae)
                        r2_improved_vs_best = (new_r2 is not None and best_r2 is not None and new_r2 > best_r2)
                        mae_improved_vs_best = (new_mae is not None and best_mae is not None and new_mae < best_mae)
                        
                        # Reject if metrics degrade significantly vs best-seen
                        # Stricter thresholds: reject if R2 drops by more than 0.02 or MAE increases by more than 0.05
                        max_r2_degradation = 0.02  # Reject if R2 drops by more than 0.02 vs best
                        max_mae_degradation = 0.05  # Reject if MAE increases by more than 0.05 vs best
                        
                        r2_degradation_vs_best = (best_r2 - new_r2) if (best_r2 is not None and new_r2 is not None) else 0.0
                        mae_degradation_vs_best = (new_mae - best_mae) if (best_mae is not None and new_mae is not None) else 0.0
                        
                        if (r2_degradation_vs_best > max_r2_degradation or mae_degradation_vs_best > max_mae_degradation):
                            accept = False
                            reason = f"rejected: metric degradation vs best (R2: {best_r2:.4f} → {new_r2:.4f} ({r2_degradation_vs_best:+.4f}), MAE: {best_mae:.4f} → {new_mae:.4f} ({mae_degradation_vs_best:+.4f}))"
                        else:
                            # For regression, check both overall improvement AND mechanism-level improvements
                            # IMPROVED: Also check if LLM mechanisms improved individually, even if overall loss didn't improve much
                            # This is important when LLM mechanisms have low routing weight - they can improve without affecting overall loss much
                            llm_improved = False
                            if hasattr(self, 'mechanism_metrics_snapshot') and self.mechanism_metrics_snapshot:
                                llm_indices = [i for i, mtype in enumerate(self.mechanism_types) if mtype == "llm"]
                                for llm_idx in llm_indices:
                                    old_metrics = self.mechanism_metrics_snapshot.get(llm_idx, {})
                                    # Re-evaluate new mechanism metrics (will be updated after acceptance)
                                    # For now, check if mechanism performance score improved
                                    old_perf = self.mechanism_performance_snapshot.get(llm_idx, 0.5)
                                    # We'll check new performance after re-evaluation, but for now accept if overall improvement
                            
                            # CRITICAL FIX: Reject if new_loss is significantly worse than current_loss OR best_loss
                            # This prevents accepting updates that degrade from the current state or best we've seen
                            # Allow small degradation (0.01 vs current, 0.05 vs best) only if metrics improve significantly
                            max_loss_degradation_vs_current = 0.01  # Reject if loss is more than 0.01 worse than current
                            max_loss_degradation_vs_best = 0.05  # Reject if loss is more than 0.05 worse than best
                            loss_degradation_vs_current = new_loss - current_loss
                            loss_degradation_vs_best = new_loss - best_loss_for_comparison
                            
                            # Calculate actual improvement values vs best
                            r2_improvement_vs_best_val = (new_r2 - best_r2) if (new_r2 is not None and best_r2 is not None) else 0.0
                            mae_improvement_vs_best_val = (best_mae - new_mae) if (best_mae is not None and new_mae is not None) else 0.0
                            
                            # First check: reject if loss degraded significantly vs current (even if metrics improved)
                            if loss_degradation_vs_current > max_loss_degradation_vs_current:
                                # Loss degraded vs current - reject unless metrics improved VERY significantly vs best
                                # Require larger metric improvements to justify degrading from current state
                                if (r2_improved_vs_best and r2_improvement_vs_best_val >= 0.05) or (mae_improved_vs_best and mae_improvement_vs_best_val >= 0.05):
                                    # Metrics improved very significantly vs best - allow even if loss degraded vs current
                                    accept = True
                                    reason = f"loss degraded vs current ({loss_degradation_vs_current:+.4f}) but metrics improved very significantly vs best (R2: {best_r2:.4f}→{new_r2:.4f} (+{r2_improvement_vs_best_val:.4f}), MAE: {best_mae:.4f}→{new_mae:.4f} (-{mae_improvement_vs_best_val:.4f}))"
                                else:
                                    # Reject: loss degraded too much vs current and metrics didn't improve enough
                                    accept = False
                                    reason = f"rejected: loss degraded vs current ({loss_degradation_vs_current:+.4f} > {max_loss_degradation_vs_current:.4f}, current={current_loss:.4f}, new={new_loss:.4f})"
                            elif loss_degradation_vs_best > max_loss_degradation_vs_best:
                                # New loss is significantly worse than best - reject unless metrics improved significantly vs best
                                if (r2_improved_vs_best and r2_improvement_vs_best_val >= 0.02) or (mae_improved_vs_best and mae_improvement_vs_best_val >= 0.02):
                                    # Metrics improved significantly vs best - allow even if loss degraded
                                    accept = True
                                    reason = f"loss degraded vs best ({loss_degradation_vs_best:+.4f}) but metrics improved significantly vs best (R2: {best_r2:.4f}→{new_r2:.4f} (+{r2_improvement_vs_best_val:.4f}), MAE: {best_mae:.4f}→{new_mae:.4f} (-{mae_improvement_vs_best_val:.4f}))"
                                else:
                                    # Reject: loss degraded too much vs best and metrics didn't improve enough
                                    accept = False
                                    reason = f"rejected: loss degraded vs best ({loss_degradation_vs_best:+.4f} > {max_loss_degradation_vs_best:.4f}, best={best_loss_for_comparison:.4f}, new={new_loss:.4f})"
                            elif improvement > 0:
                                accept = True
                                reason = f"improvement {improvement:.4f} >= threshold {adaptive_threshold:.4f}"
                            elif improvement >= -0.01 and (r2_improved or mae_improved):
                                # Allow tiny degradation (0.01) if R2 or MAE improved - mechanism is getting better
                                accept = True
                                reason = f"tiny degradation {improvement:.4f} but metrics improved (R2: {current_r2:.4f}→{new_r2:.4f}, MAE: {current_mae:.4f}→{new_mae:.4f})"
                            else:
                                # Reject significant degradation for regression
                                accept = False
                                reason = f"rejected: degradation {improvement:.4f} (regression requires improvement or tiny degradation with metric improvement)"
                
                # Criterion 2: Wilcoxon test (if small sample)
                elif len(X_accept_consistent) < 50 and improvement > 0:
                    try:
                        from scipy.stats import wilcoxon
                        old_preds = metrics.get('predictions', [])
                        new_preds = new_metrics.get('predictions', [])
                        y_vals = y_accept_consistent
                        
                        if len(old_preds) == len(new_preds) == len(y_vals):
                            if self.task_type == "classification":
                                old_losses = np.array([1.0 if int(np.round(old_preds[j])) != int(y_vals[j]) else 0.0 for j in range(len(y_vals))])
                                new_losses = np.array([1.0 if int(np.round(new_preds[j])) != int(y_vals[j]) else 0.0 for j in range(len(y_vals))])
                            else:
                                old_losses = np.abs(np.array(old_preds) - np.array(y_vals))
                                new_losses = np.abs(np.array(new_preds) - np.array(y_vals))
                            
                            try:
                                stat, p_value = wilcoxon(new_losses, old_losses, zero_method="wilcox", alternative="less")
                                if p_value < 0.15:  # More lenient p-value
                                    accept = True
                                    reason = f"Wilcoxon p={p_value:.4f} < 0.15"
                            except Exception:
                                pass
                    except ImportError:
                        pass
                    except Exception:
                        pass
                
                # Criterion 3: Any improvement in first 2 iterations (exploration)
                elif i < 2 and improvement > 0:
                    accept = True
                    reason = f"early exploration (iter {i+1}, improvement={improvement:.4f})"
                
                # Criterion 4: For regression, accept if R2 or MAE improve vs BEST-SEEN (even if loss doesn't)
                # This is important because loss might not capture all aspects of regression performance
                if not accept and self.task_type == "regression":
                    metric_improvement_threshold_r2 = 0.01  # Require at least 0.01 improvement in R2
                    metric_improvement_threshold_mae = 0.01  # Require at least 0.01 improvement in MAE
                    
                    current_r2 = metrics.get('r2', None)
                    current_mae = metrics.get('mae', None)
                    new_r2 = new_metrics.get('r2', None)
                    new_mae = new_metrics.get('mae', None)
                    best_r2 = getattr(self, '_best_r2', current_r2)
                    best_mae = getattr(self, '_best_mae', current_mae)
                    
                    r2_improvement_vs_best = (new_r2 - best_r2) if (new_r2 is not None and best_r2 is not None) else 0.0
                    mae_improvement_vs_best = (best_mae - new_mae) if (best_mae is not None and new_mae is not None) else 0.0
                    r2_improvement_vs_current = (new_r2 - current_r2) if (new_r2 is not None and current_r2 is not None) else 0.0
                    mae_improvement_vs_current = (current_mae - new_mae) if (current_mae is not None and new_mae is not None) else 0.0
                    
                    # Accept if metrics improve vs best-seen OR vs current (if significant)
                    if (r2_improvement_vs_best >= metric_improvement_threshold_r2 or 
                        mae_improvement_vs_best >= metric_improvement_threshold_mae or
                        r2_improvement_vs_current >= metric_improvement_threshold_r2 * 2 or 
                        mae_improvement_vs_current >= metric_improvement_threshold_mae * 2):
                        accept = True
                        if r2_improvement_vs_best >= metric_improvement_threshold_r2 and mae_improvement_vs_best >= metric_improvement_threshold_mae:
                            reason = f"metric improvement vs best (R2: {best_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_best:.4f}), MAE: {best_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_best:.4f}))"
                        elif r2_improvement_vs_best >= metric_improvement_threshold_r2:
                            reason = f"R2 improvement vs best (R2: {best_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_best:.4f}))"
                        elif mae_improvement_vs_best >= metric_improvement_threshold_mae:
                            reason = f"MAE improvement vs best (MAE: {best_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_best:.4f}))"
                        else:
                            reason = f"metric improvement vs current (R2: {current_r2:.4f} → {new_r2:.4f} (+{r2_improvement_vs_current:.4f}), MAE: {current_mae:.4f} → {new_mae:.4f} (-{mae_improvement_vs_current:.4f}))"
                
                # Criterion 5: For classification, accept if accuracy or F1 improve vs BEST-SEEN (even if loss doesn't)
                # This is important because loss might not capture all aspects of classification performance
                if not accept and self.task_type == "classification":
                    metric_improvement_threshold = 0.01  # Require at least 1% improvement in accuracy or F1
                    acc_improvement_vs_best = new_acc - best_acc_for_comparison
                    f1_improvement_vs_best = new_f1 - best_f1_for_comparison
                    acc_improvement_vs_current = new_acc - current_acc
                    f1_improvement_vs_current = new_f1 - current_f1
                    
                    # Accept if metrics improve vs best-seen OR vs current (if significant)
                    if (acc_improvement_vs_best >= metric_improvement_threshold or f1_improvement_vs_best >= metric_improvement_threshold or
                        acc_improvement_vs_current >= metric_improvement_threshold * 2 or f1_improvement_vs_current >= metric_improvement_threshold * 2):
                        accept = True
                        if acc_improvement_vs_best >= metric_improvement_threshold and f1_improvement_vs_best >= metric_improvement_threshold:
                            reason = f"metric improvement vs best (ACC: {best_acc_for_comparison:.4f} → {new_acc:.4f} (+{acc_improvement_vs_best:.4f}), F1: {best_f1_for_comparison:.4f} → {new_f1:.4f} (+{f1_improvement_vs_best:.4f}))"
                        elif acc_improvement_vs_best >= metric_improvement_threshold:
                            reason = f"accuracy improvement vs best (ACC: {best_acc_for_comparison:.4f} → {new_acc:.4f} (+{acc_improvement_vs_best:.4f}))"
                        elif f1_improvement_vs_best >= metric_improvement_threshold:
                            reason = f"F1 improvement vs best (F1: {best_f1_for_comparison:.4f} → {new_f1:.4f} (+{f1_improvement_vs_best:.4f}))"
                        else:
                            reason = f"metric improvement vs current (ACC: {current_acc:.4f} → {new_acc:.4f} (+{acc_improvement_vs_current:.4f}), F1: {current_f1:.4f} → {new_f1:.4f} (+{f1_improvement_vs_current:.4f}))"
                
                # Criterion 5: For very poor mechanisms in early iterations, accept tiny degradation only if metrics don't degrade
                # This allows minimal exploration but prevents accepting clearly worse updates
                if not accept and self.task_type == "classification" and current_loss > 0.5 and i < 3:
                    # Only in first 3 iterations, allow tiny degradation (0.01) if metrics don't degrade
                    max_allowed_degradation = 0.01  # Allow only 0.01 worse loss (very small)
                    acc_degradation = current_acc - new_acc
                    f1_degradation = current_f1 - new_f1
                    if (new_loss <= current_loss + max_allowed_degradation and 
                        acc_degradation <= 0.01 and f1_degradation <= 0.01):
                        accept = True
                        reason = f"early exploration (iter {i+1}, tiny degradation {new_loss - current_loss:.4f} <= {max_allowed_degradation}, metrics stable)"
                
                if accept:
                    logger.info(f"  ✓ Accepted update: loss {current_loss:.4f} → {new_loss:.4f} ({reason})")
                    accepted = 1
                    current_loss = new_loss
                    
                    # Track this iteration in history for TextGrad memory
                    # Store mechanism changes and performance metrics
                    llm_indices = [idx for idx, mtype in enumerate(self.mechanism_types) if mtype == "llm"]
                    for llm_local_idx, full_idx in enumerate(llm_indices):
                        if llm_local_idx < len(prev_mechanisms) and llm_local_idx < len(self.mechanisms):
                            mech_before = prev_mechanisms[full_idx] if full_idx < len(prev_mechanisms) else ""
                            mech_after = self.mechanisms[full_idx] if full_idx < len(self.mechanisms) else ""
                            
                            # Build iteration history entry
                            hist_entry = {
                                'iteration': i + 1,
                                'mechanism_index': full_idx,
                                'mechanism_before': mech_before,
                                'mechanism_after': mech_after,
                                'metrics_before': {
                                    'loss': current_loss,
                                    'r2': metrics.get('r2', None),
                                    'mae': metrics.get('mae', None),
                                    'accuracy': metrics.get('accuracy', None),
                                    'f1': metrics.get('f1', None)
                                },
                                'metrics_after': {
                                    'loss': new_loss,
                                    'r2': new_metrics.get('r2', None),
                                    'mae': new_metrics.get('mae', None),
                                    'accuracy': new_metrics.get('accuracy', None),
                                    'f1': new_metrics.get('f1', None)
                                },
                                'accepted': True,
                                'reason': reason
                            }
                            
                            # Initialize history list if needed
                            if not hasattr(self, 'textgrad_iteration_history'):
                                self.textgrad_iteration_history = []
                            
                            # Add to history (keep last 5 iterations to avoid prompt bloat)
                            self.textgrad_iteration_history.append(hist_entry)
                            if len(self.textgrad_iteration_history) > 5:
                                self.textgrad_iteration_history.pop(0)
                    
                    # CRITICAL: Update the generator's mechanisms so changes persist across iterations
                    # Otherwise, get_all_mechanisms() will reset to old mechanisms at the start of next iteration
                    # Mechanism order: known_mechanisms (type="known") + unknown_mechanisms (type="llm") + ML (type="ml")
                    num_known = len(self.mech_generator.known_mechanisms)
                    for i_m, mtype in enumerate(self.mechanism_types):
                        if mtype == "known":
                            # Update known mechanism in generator
                            if i_m < len(self.mech_generator.known_mechanisms):
                                self.mech_generator.known_mechanisms[i_m] = self.mechanisms[i_m]
                        elif mtype == "llm":
                            # Update unknown mechanism in generator
                            # unknown_mechanisms start after known_mechanisms
                            unknown_idx = i_m - num_known
                            if 0 <= unknown_idx < len(self.mech_generator.unknown_mechanisms):
                                self.mech_generator.unknown_mechanisms[unknown_idx] = self.mechanisms[i_m]
                        # ML mechanisms don't need updating (they're not modified by TextGrad)
                    
                    # CRITICAL: Update best snapshot based on PERFORMANCE METRICS (priority), not just loss
                    # This ensures we keep the model with highest R2/MAE (regression) or Accuracy/F1 (classification)
                    is_new_best = False
                    from maicl_config import get_best_snapshot_threshold
                    
                    if self.task_type == "regression":
                        # For regression: PRIORITIZE R2 and MAE over loss
                        new_r2 = new_metrics.get('r2', None)
                        new_mae = new_metrics.get('mae', None)
                        best_r2 = getattr(self, '_best_r2', None)
                        best_mae = getattr(self, '_best_mae', None)
                        
                        r2_threshold = get_best_snapshot_threshold('regression', 'r2')
                        mae_threshold = get_best_snapshot_threshold('regression', 'mae')
                        
                        # Check if performance metrics improved
                        r2_improved = (new_r2 is not None and best_r2 is not None and new_r2 > best_r2 + r2_threshold)
                        mae_improved = (new_mae is not None and best_mae is not None and new_mae < best_mae - mae_threshold)
                        
                        # PRIORITY: Update if metrics improved (even if loss didn't)
                        if r2_improved or mae_improved:
                            is_new_best = True
                            logger.info(f"  [Best Performance] Metrics improved: R2 {best_r2:.4f}→{new_r2:.4f}, MAE {best_mae:.4f}→{new_mae:.4f}")
                        # Also update if loss improved significantly (secondary priority)
                        elif new_loss < best_loss - 0.01:  # Require at least 0.01 improvement in loss
                            is_new_best = True
                            logger.info(f"  [Best Performance] Loss improved: {best_loss:.4f}→{new_loss:.4f}")
                    
                    elif self.task_type == "classification":
                        # For classification: PRIORITIZE Accuracy and F1 over loss
                        acc_threshold = get_best_snapshot_threshold('classification', 'acc')
                        f1_threshold = get_best_snapshot_threshold('classification', 'f1')
                        
                        # Check if performance metrics improved
                        acc_improved = (new_acc > best_acc_for_comparison + acc_threshold)
                        f1_improved = (new_f1 > best_f1_for_comparison + f1_threshold)
                        
                        # PRIORITY: Update if metrics improved (even if loss didn't)
                        if acc_improved or f1_improved:
                            is_new_best = True
                            logger.info(f"  [Best Performance] Metrics improved: ACC {best_acc_for_comparison:.4f}→{new_acc:.4f}, F1 {best_f1_for_comparison:.4f}→{new_f1:.4f}")
                        # Also update if loss improved significantly (secondary priority)
                        elif new_loss < best_loss - 0.01:  # Require at least 0.01 improvement in loss
                            is_new_best = True
                            logger.info(f"  [Best Performance] Loss improved: {best_loss:.4f}→{new_loss:.4f}")
                    
                    if is_new_best:
                        best_loss = float(new_loss)
                        if self.task_type == "classification":
                            best_acc = float(new_acc)
                            best_f1 = float(new_f1)
                        elif self.task_type == "regression":
                            new_r2 = new_metrics.get('r2', None)
                            new_mae = new_metrics.get('mae', None)
                            if new_r2 is not None:
                                self._best_r2 = float(new_r2)
                            if new_mae is not None:
                                self._best_mae = float(new_mae)
                        best_mechanisms = copy.deepcopy(self.mechanisms)
                        # CRITICAL: Capture mechanism performance scores IMMEDIATELY after the evaluation
                        # that produced the best metrics. The snapshot should be from the same evaluation
                        # that produced new_metrics, not from a stale state.
                        # The evaluate() method updates mechanism_performance_snapshot after recalculating
                        # performance, so at this point it should be current.
                        perf_snapshot = copy.deepcopy(getattr(self, 'mechanism_performance_snapshot', {}))
                        metrics_snapshot = copy.deepcopy(getattr(self, 'mechanism_metrics_snapshot', {}))
                        
                        # CRITICAL: Store overall metrics in snapshot for performance-based restoration
                        overall_metrics = {
                            'loss': float(new_loss)
                        }
                        if self.task_type == "regression":
                            new_r2 = new_metrics.get('r2', None)
                            new_mae = new_metrics.get('mae', None)
                            if new_r2 is not None:
                                overall_metrics['r2'] = float(new_r2)
                            if new_mae is not None:
                                overall_metrics['mae'] = float(new_mae)
                        elif self.task_type == "classification":
                            if new_acc is not None:
                                overall_metrics['accuracy'] = float(new_acc)
                            if new_f1 is not None:
                                overall_metrics['f1'] = float(new_f1)
                        
                        best_snapshot = {
                            "iteration": i + 1,
                            "loss": float(new_loss),
                            "overall_metrics": overall_metrics,  # Store overall metrics for performance comparison
                            "mechanisms": copy.deepcopy(self.mechanisms),
                            "mechanism_types": copy.deepcopy(self.mechanism_types),
                            "mechanism_performance": perf_snapshot,
                            "mechanism_metrics": metrics_snapshot
                        }
                        if self.task_type == "classification":
                            logger.info(f"  [Best Snapshot] Updated best-performing model: ACC={best_acc:.4f}, F1={best_f1:.4f}, Loss={best_loss:.4f}")
                        elif self.task_type == "regression":
                            logger.info(f"  [Best Snapshot] Updated best-performing model: R2={self._best_r2:.4f}, MAE={self._best_mae:.4f}, Loss={best_loss:.4f}")
                else:
                    logger.info(f"  ✗ Rejected update: loss {current_loss:.4f} → {new_loss:.4f}")
                    self.mechanisms = prev_mechanisms
                    rejected = 1
                    
                    # CRITICAL: Sync generator state when TextGrad updates are rejected
                    # This ensures mechanisms persist correctly between iterations
                    # Mechanism order: known_mechanisms (type="known") + unknown_mechanisms (type="llm") + ML (type="ml")
                    num_known = len(self.mech_generator.known_mechanisms)
                    for i_m, mtype in enumerate(self.mechanism_types):
                        if mtype == "known":
                            # Update known mechanism in generator
                            if i_m < len(self.mech_generator.known_mechanisms):
                                self.mech_generator.known_mechanisms[i_m] = self.mechanisms[i_m]
                        elif mtype == "llm":
                            # Update unknown mechanism in generator
                            # unknown_mechanisms start after known_mechanisms
                            unknown_idx = i_m - num_known
                            if 0 <= unknown_idx < len(self.mech_generator.unknown_mechanisms):
                                self.mech_generator.unknown_mechanisms[unknown_idx] = self.mechanisms[i_m]
                        # ML mechanisms don't need updating (they're not modified by TextGrad)
                    
                    # Track rejected iteration in history
                    llm_indices = [idx for idx, mtype in enumerate(self.mechanism_types) if mtype == "llm"]
                    for llm_local_idx, full_idx in enumerate(llm_indices):
                        if llm_local_idx < len(prev_mechanisms):
                            mech_before = prev_mechanisms[full_idx] if full_idx < len(prev_mechanisms) else ""
                            mech_after = self.mechanisms[full_idx] if full_idx < len(self.mechanisms) else ""
                            
                            hist_entry = {
                                'iteration': i + 1,
                                'mechanism_index': full_idx,
                                'mechanism_before': mech_before,
                                'mechanism_after': mech_after,
                                'metrics_before': {
                                    'loss': current_loss,
                                    'r2': metrics.get('r2', None),
                                    'mae': metrics.get('mae', None),
                                    'accuracy': metrics.get('accuracy', None),
                                    'f1': metrics.get('f1', None)
                                },
                                'metrics_after': {
                                    'loss': new_loss,
                                    'r2': new_metrics.get('r2', None),
                                    'mae': new_metrics.get('mae', None),
                                    'accuracy': new_metrics.get('accuracy', None),
                                    'f1': new_metrics.get('f1', None)
                                },
                                'accepted': False,
                                'reason': reason
                            }
                            
                            if not hasattr(self, 'textgrad_iteration_history'):
                                self.textgrad_iteration_history = []
                            
                            self.textgrad_iteration_history.append(hist_entry)
                            if len(self.textgrad_iteration_history) > 5:
                                self.textgrad_iteration_history.pop(0)
                    
                    # CRITICAL: After rejecting an update, check if the current state (before rejection) is better than best_snapshot
                    # PRIORITIZE performance metrics over loss to ensure we keep the best-performing model
                    # The current state (metrics) represents the state before the rejected update
                    is_current_best = False
                    from maicl_config import get_best_snapshot_threshold
                    
                    if self.task_type == "regression":
                        current_r2 = metrics.get('r2', None)
                        current_mae = metrics.get('mae', None)
                        best_r2 = getattr(self, '_best_r2', None)
                        best_mae = getattr(self, '_best_mae', None)
                        
                        r2_threshold = get_best_snapshot_threshold('regression', 'r2')
                        mae_threshold = get_best_snapshot_threshold('regression', 'mae')
                        
                        # PRIORITY: Check if performance metrics improved
                        r2_improved = (current_r2 is not None and best_r2 is not None and current_r2 > best_r2 + r2_threshold)
                        mae_improved = (current_mae is not None and best_mae is not None and current_mae < best_mae - mae_threshold)
                        
                        if r2_improved or mae_improved:
                            is_current_best = True
                        # Also check if loss improved significantly (secondary priority)
                        elif current_loss < best_loss - 0.01:
                            is_current_best = True
                    
                    elif self.task_type == "classification":
                        acc_threshold = get_best_snapshot_threshold('classification', 'acc')
                        f1_threshold = get_best_snapshot_threshold('classification', 'f1')
                        
                        # PRIORITY: Check if performance metrics improved
                        acc_improved = (current_acc is not None and best_acc is not None and current_acc > best_acc + acc_threshold)
                        f1_improved = (current_f1 is not None and best_f1 is not None and current_f1 > best_f1 + f1_threshold)
                        
                        if acc_improved or f1_improved:
                            is_current_best = True
                        # Also check if loss improved significantly (secondary priority)
                        elif current_loss < best_loss - 0.01:
                            is_current_best = True
                    
                    if is_current_best:
                        best_loss = float(current_loss)
                        best_mechanisms = copy.deepcopy(self.mechanisms)
                        if self.task_type == "classification":
                            if current_acc is not None:
                                best_acc = float(current_acc)
                            if current_f1 is not None:
                                best_f1 = float(current_f1)
                        elif self.task_type == "regression":
                            current_r2 = metrics.get('r2', None)
                            current_mae = metrics.get('mae', None)
                            if current_r2 is not None:
                                self._best_r2 = float(current_r2)
                            if current_mae is not None:
                                self._best_mae = float(current_mae)
                        perf_snapshot = copy.deepcopy(getattr(self, 'mechanism_performance_snapshot', {}))
                        metrics_snapshot = copy.deepcopy(getattr(self, 'mechanism_metrics_snapshot', {}))
                        
                        # CRITICAL: Store overall metrics in snapshot for performance-based restoration
                        overall_metrics = {
                            'loss': float(current_loss)
                        }
                        if self.task_type == "regression":
                            current_r2 = metrics.get('r2', None)
                            current_mae = metrics.get('mae', None)
                            if current_r2 is not None:
                                overall_metrics['r2'] = float(current_r2)
                            if current_mae is not None:
                                overall_metrics['mae'] = float(current_mae)
                        elif self.task_type == "classification":
                            if current_acc is not None:
                                overall_metrics['accuracy'] = float(current_acc)
                            if current_f1 is not None:
                                overall_metrics['f1'] = float(current_f1)
                        
                        best_snapshot = {
                            "iteration": i + 1,
                            "loss": float(current_loss),
                            "overall_metrics": overall_metrics,  # Store overall metrics for performance comparison
                            "mechanisms": copy.deepcopy(self.mechanisms),
                            "mechanism_types": copy.deepcopy(self.mechanism_types),
                            "mechanism_performance": perf_snapshot,
                            "mechanism_metrics": metrics_snapshot
                        }
                        if self.task_type == "classification":
                            logger.info(f"  [Best Checkpoint] Updated best-performing model at iteration {i+1} after rejection: ACC={current_acc:.4f}, F1={current_f1:.4f}, Loss={current_loss:.4f}")
                        else:
                            current_r2 = metrics.get('r2', None)
                            current_mae = metrics.get('mae', None)
                            logger.info(f"  [Best Checkpoint] Updated best-performing model at iteration {i+1} after rejection: R2={current_r2:.4f}, MAE={current_mae:.4f}, Loss={current_loss:.4f}")
            
            # Update best checkpoint if no TextGrad was attempted OR if TextGrad update was rejected
            # This ensures we capture improvements even when TextGrad isn't used
            if not accepted and new_metrics is None:
                # No TextGrad attempted - check if current state is best
                is_best = False
                if current_loss < best_loss:
                    is_best = True
                elif self.task_type == "classification" and current_acc is not None and best_acc is not None:
                    if current_acc > best_acc + 0.01 or (current_acc > best_acc and current_loss <= best_loss + 0.01):
                        is_best = True
                elif self.task_type == "regression":
                    if (current_r2 is not None and self._best_r2 is not None and current_r2 > self._best_r2 + 0.01) or \
                       (current_mae is not None and self._best_mae is not None and current_mae < self._best_mae - 0.01):
                        is_best = True
                
                if is_best:
                    best_loss = float(current_loss)
                    best_mechanisms = copy.deepcopy(self.mechanisms)
                    if self.task_type == "classification" and current_acc is not None:
                        best_acc = float(current_acc)
                    if self.task_type == "classification" and current_f1 is not None:
                        best_f1 = float(current_f1)
                    if self.task_type == "regression" and current_r2 is not None:
                        self._best_r2 = float(current_r2)
                    if self.task_type == "regression" and current_mae is not None:
                        self._best_mae = float(current_mae)
                    perf_snapshot = copy.deepcopy(getattr(self, 'mechanism_performance_snapshot', {}))
                    metrics_snapshot = copy.deepcopy(getattr(self, 'mechanism_metrics_snapshot', {}))
                    
                    # CRITICAL: Store overall metrics in snapshot for performance-based restoration
                    overall_metrics = {
                        'loss': float(current_loss)
                    }
                    if self.task_type == "regression":
                        if current_r2 is not None:
                            overall_metrics['r2'] = float(current_r2)
                        if current_mae is not None:
                            overall_metrics['mae'] = float(current_mae)
                    elif self.task_type == "classification":
                        if current_acc is not None:
                            overall_metrics['accuracy'] = float(current_acc)
                        if current_f1 is not None:
                            overall_metrics['f1'] = float(current_f1)
                    
                    best_snapshot = {
                        "iteration": i + 1,
                        "loss": float(current_loss),
                        "overall_metrics": overall_metrics,  # Store overall metrics for performance comparison
                        "mechanisms": copy.deepcopy(self.mechanisms),
                        "mechanism_types": copy.deepcopy(self.mechanism_types),
                        "mechanism_performance": perf_snapshot,
                        "mechanism_metrics": metrics_snapshot
                    }
                    if self.task_type == "classification":
                        logger.info(f"  [Best Checkpoint] Updated best-performing model at iteration {i+1}: ACC={current_acc:.4f}, F1={current_f1:.4f}, Loss={current_loss:.4f}")
                    else:
                        logger.info(f"  [Best Checkpoint] Updated best-performing model at iteration {i+1}: R2={current_r2:.4f}, MAE={current_mae:.4f}, Loss={current_loss:.4f}")
            
            # Log fallback summary at end of iteration
            fallback_summary = get_fallback_summary()
            if fallback_summary:
                logger.warning(f"  [Iter={i+1}] Fallback summary: {fallback_summary}")
                clear_fallback_counts()
            
            # Determine which metrics to use: if update was accepted, use new_metrics; otherwise use original metrics
            # This ensures all logged metrics (loss, r2, mae, accuracy, f1) are from the same state
            if accepted and new_metrics is not None:
                # Update was accepted - use metrics from the new state
                final_metrics = new_metrics
            else:
                # Update was rejected or no update attempted - use original metrics
                final_metrics = metrics
            
            # Log metrics table for this iteration (using final_metrics to ensure consistency)
            iter_metrics = {
                "loss": current_loss,
                "accepted": accepted,
                "rejected": rejected
            }
            if self.task_type == "classification":
                iter_metrics["accuracy"] = final_metrics.get('accuracy', 0.0)
                iter_metrics["f1"] = final_metrics.get('f1', 0.0)
            else:
                iter_metrics["r2"] = final_metrics.get('r2', -1.0)
                iter_metrics["mae"] = final_metrics.get('mae', 1e9)
            log_metrics_table(f"Iteration {i+1}", iter_metrics)
            
            # Store training history (using final_metrics to ensure all metrics are from same state)
            self.training_history['iterations'].append(i + 1)
            self.training_history['val_loss'].append(current_loss)
            self.training_history['discovery_loss'].append(0.0)
            self.training_history['accepted_updates'].append(accepted)
            self.training_history['rejected_updates'].append(rejected)
            # Track per-iteration metrics (using final_metrics for consistency)
            if self.task_type == "classification":
                self.training_history['val_accuracy'].append(float(final_metrics.get('accuracy', 0.0)))
                self.training_history['val_f1'].append(float(final_metrics.get('f1', 0.0)))
                self.training_history['val_r2'].append(None)
                self.training_history['val_mae'].append(None)
            else:
                self.training_history['val_accuracy'].append(None)
                self.training_history['val_f1'].append(None)
                self.training_history['val_r2'].append(float(final_metrics.get('r2', -1.0)))
                self.training_history['val_mae'].append(float(final_metrics.get('mae', 1e9)))
            # Save mechanism evolution snapshot and persist artifacts
            mech_snapshot = {
                "iteration": i + 1,
                "accepted": bool(accepted),
                "loss": float(current_loss),
                "mechanisms": self.mechanisms,
                "mechanism_types": self.mechanism_types,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            self.training_history['mechanism_evolution'].append(mech_snapshot)
            try:
                self._persist_iteration_artifacts(i + 1, mech_snapshot, final_metrics, output_dir=self.output_dir)
            except Exception as _:
                pass
            
            # LLM usage snapshot (after) and deltas
            try:
                stats_after = self.llm.get_stats()
            except Exception:
                stats_after = stats_before
            delta_calls = int(stats_after.get("total_calls", 0) - stats_before.get("total_calls", 0))
            delta_batches = int(stats_after.get("batch_operations", 0) - stats_before.get("batch_operations", 0))
            self.training_history['llm_calls'].append(delta_calls)
            self.training_history['llm_batches'].append(delta_batches)
            logger.info(f"  [LLM Usage] calls+={delta_calls}, batches+={delta_batches} (cum calls={stats_after.get('total_calls', 0)})")
        
        logger.info("  ✓ Training complete")
        # CRITICAL: Always restore the best-performing model based on PERFORMANCE METRICS (R2/MAE or Accuracy/F1)
        # This ensures final results use the model with highest performance, not just lowest loss
        try:
            if best_snapshot is not None:
                from maicl_config import get_restoration_config
                
                # Get best metrics from snapshot for comparison
                best_snapshot_metrics = best_snapshot.get('mechanism_metrics', {})
                best_snapshot_loss = best_snapshot.get('loss', float('inf'))
                
                # Get current metrics for comparison
                current_loss = float(getattr(self, '_last_eval_loss', best_snapshot_loss))
                current_has_llm = any(t == "llm" for t in self.mechanism_types)
                best_has_llm = any(t == "llm" for t in best_snapshot.get("mechanism_types", []))
                
                # Compare performance metrics (priority) to determine which is truly best
                should_restore = True
                best_performance_reason = ""
                
                if self.task_type == "regression":
                    # Get best R2 and MAE from snapshot (prefer overall_metrics, fallback to mechanism_metrics)
                    overall_metrics = best_snapshot.get('overall_metrics', {})
                    best_snapshot_r2 = overall_metrics.get('r2', None)
                    best_snapshot_mae = overall_metrics.get('mae', None)
                    
                    # Fallback: extract from mechanism_metrics if overall_metrics not available
                    if best_snapshot_r2 is None or best_snapshot_mae is None:
                        for mech_idx, mech_metrics in best_snapshot_metrics.items():
                            if best_snapshot_r2 is None and mech_metrics.get('r2') is not None:
                                if best_snapshot_r2 is None or mech_metrics['r2'] > best_snapshot_r2:
                                    best_snapshot_r2 = mech_metrics['r2']
                            if best_snapshot_mae is None and mech_metrics.get('mae') is not None:
                                if best_snapshot_mae is None or mech_metrics['mae'] < best_snapshot_mae:
                                    best_snapshot_mae = mech_metrics['mae']
                    
                    # Get current metrics
                    current_r2 = getattr(self, '_best_r2', None)
                    current_mae = getattr(self, '_best_mae', None)
                    
                    # Compare: best snapshot is better if it has higher R2 or lower MAE
                    if best_snapshot_r2 is not None and current_r2 is not None:
                        if best_snapshot_r2 > current_r2 + 0.01:
                            should_restore = True
                            best_performance_reason = f"Best snapshot has better R2 ({best_snapshot_r2:.4f} vs {current_r2:.4f})"
                        elif current_r2 > best_snapshot_r2 + 0.01:
                            should_restore = False
                            best_performance_reason = f"Current state has better R2 ({current_r2:.4f} vs {best_snapshot_r2:.4f})"
                    elif best_snapshot_mae is not None and current_mae is not None:
                        if best_snapshot_mae < current_mae - 0.01:
                            should_restore = True
                            best_performance_reason = f"Best snapshot has better MAE ({best_snapshot_mae:.4f} vs {current_mae:.4f})"
                        elif current_mae < best_snapshot_mae - 0.01:
                            should_restore = False
                            best_performance_reason = f"Current state has better MAE ({current_mae:.4f} vs {best_snapshot_mae:.4f})"
                    
                    # If metrics are similar, check loss
                    if not best_performance_reason:
                        if best_snapshot_loss < current_loss - 0.01:
                            should_restore = True
                            best_performance_reason = f"Best snapshot has better loss ({best_snapshot_loss:.4f} vs {current_loss:.4f})"
                        elif current_loss < best_snapshot_loss - 0.01:
                            should_restore = False
                            best_performance_reason = f"Current state has better loss ({current_loss:.4f} vs {best_snapshot_loss:.4f})"
                
                elif self.task_type == "classification":
                    # Get best Accuracy and F1 from snapshot (prefer overall_metrics, fallback to mechanism_metrics)
                    overall_metrics = best_snapshot.get('overall_metrics', {})
                    best_snapshot_acc = overall_metrics.get('accuracy', None)
                    best_snapshot_f1 = overall_metrics.get('f1', None)
                    
                    # Fallback: extract from mechanism_metrics if overall_metrics not available
                    if best_snapshot_acc is None or best_snapshot_f1 is None:
                        for mech_idx, mech_metrics in best_snapshot_metrics.items():
                            if best_snapshot_acc is None and mech_metrics.get('accuracy') is not None:
                                if best_snapshot_acc is None or mech_metrics['accuracy'] > best_snapshot_acc:
                                    best_snapshot_acc = mech_metrics['accuracy']
                            if best_snapshot_f1 is None and mech_metrics.get('f1') is not None:
                                if best_snapshot_f1 is None or mech_metrics['f1'] > best_snapshot_f1:
                                    best_snapshot_f1 = mech_metrics['f1']
                    
                    # Get current metrics
                    current_acc = getattr(self, '_best_acc', None)
                    current_f1 = getattr(self, '_best_f1', None)
                    
                    # Compare: best snapshot is better if it has higher Accuracy or F1
                    if best_snapshot_acc is not None and current_acc is not None:
                        if best_snapshot_acc > current_acc + 0.01:
                            should_restore = True
                            best_performance_reason = f"Best snapshot has better Accuracy ({best_snapshot_acc:.4f} vs {current_acc:.4f})"
                        elif current_acc > best_snapshot_acc + 0.01:
                            should_restore = False
                            best_performance_reason = f"Current state has better Accuracy ({current_acc:.4f} vs {best_snapshot_acc:.4f})"
                    elif best_snapshot_f1 is not None and current_f1 is not None:
                        if best_snapshot_f1 > current_f1 + 0.01:
                            should_restore = True
                            best_performance_reason = f"Best snapshot has better F1 ({best_snapshot_f1:.4f} vs {current_f1:.4f})"
                        elif current_f1 > best_snapshot_f1 + 0.01:
                            should_restore = False
                            best_performance_reason = f"Current state has better F1 ({current_f1:.4f} vs {best_snapshot_f1:.4f})"
                    
                    # If metrics are similar, check loss
                    if not best_performance_reason:
                        if best_snapshot_loss < current_loss - 0.01:
                            should_restore = True
                            best_performance_reason = f"Best snapshot has better loss ({best_snapshot_loss:.4f} vs {current_loss:.4f})"
                        elif current_loss < best_snapshot_loss - 0.01:
                            should_restore = False
                            best_performance_reason = f"Current state has better loss ({current_loss:.4f} vs {best_snapshot_loss:.4f})"
                
                # Log restoration decision
                if should_restore:
                    logger.info(f"  [Restoration] Restoring best-performing model from iteration {best_snapshot['iteration']}: {best_performance_reason}")
                else:
                    logger.info(f"  [Restoration] Keeping current state: {best_performance_reason}")
                
                if should_restore:
                    # CRITICAL FIX: Preserve LLM mechanisms that were generated during training
                    # Even if we restore to an earlier iteration, we should keep LLM mechanisms
                    # since they're part of the training process and shouldn't be lost
                    current_llm_mechanisms = []
                    current_llm_types = []
                    current_llm_performance = {}
                    current_llm_metrics = {}
                    
                    # Extract LLM mechanisms from current state (before restoration)
                    for i, (mech, mech_type) in enumerate(zip(self.mechanisms, self.mechanism_types)):
                        if mech_type == "llm":
                            llm_idx = len(current_llm_mechanisms)  # Index in the preserved LLM list
                            current_llm_mechanisms.append(mech)
                            current_llm_types.append(mech_type)
                            if i in getattr(self, 'mechanism_performance_snapshot', {}):
                                current_llm_performance[llm_idx] = self.mechanism_performance_snapshot[i]
                            if i in getattr(self, 'mechanism_metrics_snapshot', {}):
                                current_llm_metrics[llm_idx] = self.mechanism_metrics_snapshot[i]
                    
                    # Restore mechanisms from best snapshot
                    self.mechanisms = best_snapshot["mechanisms"]
                    self.mechanism_types = best_snapshot["mechanism_types"]
                    
                    # Append preserved LLM mechanisms to the restored state
                    if current_llm_mechanisms:
                        num_llm_before = len([t for t in self.mechanism_types if t == "llm"])
                        self.mechanisms.extend(current_llm_mechanisms)
                        self.mechanism_types.extend(current_llm_types)
                        logger.info(f"  [Restoration] Preserved {len(current_llm_mechanisms)} LLM mechanism(s) from training (total LLM: {num_llm_before} → {len([t for t in self.mechanism_types if t == 'llm'])})")
                        
                        # Update the generator's unknown_mechanisms to match
                        num_known = len(self.mech_generator.known_mechanisms)
                        num_restored_llm = len([t for t in best_snapshot.get("mechanism_types", []) if t == "llm"])
                        # Update unknown_mechanisms: skip any that were in the snapshot, add the preserved ones
                        self.mech_generator.unknown_mechanisms = current_llm_mechanisms
                    
                    # CRITICAL FIX: Filter mechanism_performance_snapshot to only include indices that exist in restored mechanisms
                    # This prevents stale performance scores from mechanisms that were removed during training
                    restored_performance = best_snapshot.get("mechanism_performance", {})
                    filtered_performance = {}
                    for mech_idx in range(len(self.mechanisms)):
                        if mech_idx in restored_performance:
                            filtered_performance[mech_idx] = restored_performance[mech_idx]
                        elif mech_idx >= len(best_snapshot["mechanisms"]):
                            # This is a preserved LLM mechanism - use its performance from current state
                            llm_idx = mech_idx - len(best_snapshot["mechanisms"])
                            if llm_idx in current_llm_performance:
                                filtered_performance[mech_idx] = current_llm_performance[llm_idx]
                            else:
                                filtered_performance[mech_idx] = 1.0
                        else:
                            # If mechanism exists but no performance score, set default
                            filtered_performance[mech_idx] = 1.0
                    self.mechanism_performance_snapshot = filtered_performance
                    
                    # Also restore mechanism metrics (R2, MAE, F1, accuracy)
                    restored_metrics = best_snapshot.get("mechanism_metrics", {})
                    filtered_metrics = {}
                    for mech_idx in range(len(self.mechanisms)):
                        if mech_idx in restored_metrics:
                            filtered_metrics[mech_idx] = restored_metrics[mech_idx]
                        elif mech_idx >= len(best_snapshot["mechanisms"]):
                            # This is a preserved LLM mechanism - use its metrics from current state
                            llm_idx = mech_idx - len(best_snapshot["mechanisms"])
                            if llm_idx in current_llm_metrics:
                                filtered_metrics[mech_idx] = current_llm_metrics[llm_idx]
                            else:
                                filtered_metrics[mech_idx] = {}
                        else:
                            # If mechanism exists but no metrics, set empty dict
                            filtered_metrics[mech_idx] = {}
                    self.mechanism_metrics_snapshot = filtered_metrics
                    
                    logger.info(f"  [Restoration] Restored mechanism performance scores: {filtered_performance}")
                    if filtered_metrics:
                        logger.info(f"  [Restoration] Restored mechanism metrics: {filtered_metrics}")
                    
                    logger.info(f"  ✓ Restored {len(self.mechanisms)} mechanisms with {len(filtered_performance)} performance scores")
                else:
                    # Keep current state - just log what we have
                    current_perf = getattr(self, 'mechanism_performance_snapshot', {})
                    current_metrics = getattr(self, 'mechanism_metrics_snapshot', {})
                    logger.info(f"  [Restoration] Kept current state with {len(self.mechanisms)} mechanisms (including LLM)")
                    logger.info(f"  [Restoration] Current mechanism performance scores: {current_perf}")
                    if current_metrics:
                        logger.info(f"  [Restoration] Current mechanism metrics: {current_metrics}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not restore best snapshot: {e}")
            pass
        try:
            self._export_training_artifacts()
        except Exception as _:
            pass

    # =========================
    # ARTIFACTS: SAVE / PLOT
    # =========================
    def _persist_iteration_artifacts(self, iteration: int, mech_snapshot: Dict[str, Any], metrics: Dict[str, Any], output_dir: Optional[str] = None):
        """Save per-iteration mechanism and metric snapshots to OUTPUT_DIR"""
        persist_iteration_artifacts(iteration, mech_snapshot, metrics, output_dir=output_dir)

    def _export_training_artifacts(self, output_dir: Optional[str] = None):
        """Export training history (JSON/CSV) and training plots"""
        export_training_artifacts(
            self.training_history,
            self.mechanisms,
            self.mechanism_types,
            self.mechanism_performance_snapshot,
            self.task_type,
            self.use_ml_mechanism,
            self.mech_generator.ml_mechanism if self.use_ml_mechanism else None,
            self.feature_cols,
            output_dir,
            MAX_FEATURE_IMPORTANCE_TOP
        )

    def _export_mechanism_interpretations(self, output_dir: str):
        """Export human-readable mechanism interpretations"""
        export_mechanism_interpretations(
            self.mechanisms,
            self.mechanism_types,
            self.mechanism_performance_snapshot,
            self.training_history,
            self.task_type,
            self.use_ml_mechanism,
            self.mech_generator.ml_mechanism if self.use_ml_mechanism else None,
            self.feature_cols,
            output_dir,
            MAX_FEATURE_IMPORTANCE_TOP
        )
    
    def get_mechanism_explanations(self) -> List[Dict[str, str]]:
        """Get interpretable explanations for all mechanisms"""
        explanations = []
        for idx, (mech, mtype) in enumerate(zip(self.mechanisms, self.mechanism_types)):
            explanation = {
                "index": idx,
                "type": mtype,
                "mechanism": mech,
                "performance": self.mechanism_performance_snapshot.get(idx, None) if hasattr(self, 'mechanism_performance_snapshot') else None
            }
            explanations.append(explanation)
        return explanations

    def train_on_residuals(self, X_full: np.ndarray, y_full: np.ndarray, top_k: int = -1,
                           iterations: int = 3, X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None, use_test_for_acceptance: bool = False) -> Dict[str, Any]:
        """
        Convenience training that computes ML residuals, selects top-K, and trains.
        Requires an ML mechanism to be present and trained when use_ml_mechanism=True.
        
        Args:
            X_test: Optional test set features. Used if use_test_for_acceptance=True.
            y_test: Optional test set targets. Used if use_test_for_acceptance=True.
            use_test_for_acceptance: If True, use test set for acceptance evaluation instead of validation set.
        """
        if self.use_ml_mechanism and (self.mech_generator.ml_mechanism is None or not self.mech_generator.ml_mechanism.is_trained):
            raise RuntimeError("ML mechanism is not trained. Call train_ml_mechanism(...) first or disable use_ml_mechanism.")
        
        class_names = self.class_names if self.task_type == "classification" else None
        sorted_idx, residuals, ml_preds, _ = compute_ml_residuals(
            self.mech_generator.ml_mechanism if self.use_ml_mechanism else self,  # fallback won't be used
            X_full, y_full, self.feature_cols, class_names, task_type=self.task_type
        )
        if top_k == -1:
            X_sel, y_sel, top_indices = X_full, y_full, np.arange(len(X_full))
            residual_sel = residuals
        else:
            # For classification: residuals are 1.0 (mismatched) or 0.0 (matched)
            # Sorting by residual descending naturally puts mismatched first - no strategy needed
            # For regression: uses residual ranking (absolute errors)
            X_sel, y_sel, top_indices = get_top_k_residual_samples(
                sorted_idx, residuals, X_full, y_full, top_k, ml_predictions=ml_preds, task_type=self.task_type
            )
            if len(top_indices) == 0:
                logger.warning("No top-K residual samples found; falling back to full dataset.")
                X_sel, y_sel, top_indices = X_full, y_full, np.arange(len(X_full))
                residual_sel = residuals
            else:
                # Select residuals corresponding to the top-K samples
                residual_sel = residuals[top_indices]
        self.train(X_sel, y_sel, X_full, y_full, iterations=iterations, ml_residuals=residual_sel,
                   X_test=X_test, y_test=y_test, use_test_for_acceptance=use_test_for_acceptance)
        return {
            "top_indices": top_indices,
            "residuals": residuals.tolist() if hasattr(residuals, "tolist") else residuals,
        }

# Visualization functions moved to visualization.py


# Export all classes
__all__ = [
    'BatchedLLM', 'GoogleAPIKeyManager', 'MLModelMechanism', 'TextGrad', 'MultiAgentPredictor',
    'VariationalMechanismGenerator', 'TrainableMAICL', 'MinMaxScaler010',
    'clean_json_response', 'compute_dataset_stats', 'build_known_mechanism_description',
    'compute_expected_calibration_error', 'find_optimal_threshold',
    'get_ml_prediction_probability', 'validate_scaled_data', 'compute_ml_residuals',
    'get_top_k_residual_samples', 'set_output_dir', 'OUTPUT_DIR', 'OUTPUT_ROOT',
    'plot_confusion_matrix', 'plot_performance_comparison', 'plot_scatter_predictions', 'create_result_visualizations',
    # ML-guided initialization and enhanced TextGrad functions
    'extract_ml_knowledge', 'generate_ml_guided_mechanism', 'analyze_ml_failure_patterns',
    'enhanced_textgrad_feedback',
    # Constants
    'SCALE_MIN', 'SCALE_MAX', 'MAX_BATCH_SIZE', 'EVAL_BATCH_SIZE',
    'ML_HIGH_CONFIDENCE_THRESHOLD', 'ML_LOW_CONFIDENCE_THRESHOLD',
    'MIN_ML_WEIGHT', 'MAX_ML_WEIGHT', 'HARD_ML_GATE_THRESHOLD'
]