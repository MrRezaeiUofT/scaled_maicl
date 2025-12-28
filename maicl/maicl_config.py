"""
Configuration module for MA-ICL system.
Contains constants, configuration loading, and prompt retrieval.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# CONFIGURATION CONSTANTS
# =========================

SCALE_MIN = 0.0
SCALE_MAX = 1.0
MAX_BATCH_SIZE = 128
BATCH_TIMEOUT = 60
EVAL_BATCH_SIZE = 128
GRADIENT_BATCH_SIZE = 128
LLM_GROUP_SIZE = 128  # group multiple inputs per LLM call for prediction
LLM_COMBINE_MECHANISMS = True  # combine all LLM mechanisms per input group into one call
K_SHOT_PER_TARGET = 10  # Default number of few-shot examples to include in prompts (set to 0 to disable)

ATTENTION_TEMP = 1.0
RANDOM_STATE = 42
IMPROVEMENT_THRESHOLD_CLASSIFICATION = 0.005

# Confidence-based routing thresholds
ML_HIGH_CONFIDENCE_THRESHOLD = 0.75
ML_LOW_CONFIDENCE_THRESHOLD = 0.30
# Routing weight caps for ML mechanism
MIN_ML_WEIGHT = 0.5
MAX_ML_WEIGHT = 0.9
# Hard gate: if ML confidence >= this, route fully to ML
HARD_ML_GATE_THRESHOLD = 0.80

# Output directory
OUTPUT_ROOT = "ma_icl_results"
RUN_FOLDER_NAME = "current_run"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_FOLDER_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Slice Limits Configuration (avoid hardcoded numbers) ===
# Feature selection and analysis limits
MAX_TOP_FEATURES = 20  # Maximum number of top features to use in mechanisms/formulas
MAX_TOP_FEATURES_DISPLAY = 20  # Maximum features to display in descriptions
MAX_TOP_ERROR_FEATURES = 20  # Maximum error-correlated features to analyze
MAX_TOP_ERROR_FEATURES_CHECK = 5  # Maximum error features for specific checks
MAX_FEATURE_INTERACTIONS = 10  # Maximum interaction terms to include
MAX_FEATURE_IMPORTANCE_TOP = 20  # Maximum features for importance ranking

# Prediction and residual analysis limits
# NOTE: These are for DISPLAY/LOGGING purposes only (limiting examples in prompts/logs).
# Actual filtering to top_k residuals is done via get_top_k_residual_samples() and passed via ml_residuals parameter.
# When ml_residuals is already filtered to top_k, these constants limit how many examples are shown in error feedback.
MAX_WORST_PREDICTIONS_COLLECT = 20  # Maximum worst predictions to collect for analysis
MAX_WORST_PREDICTIONS_DISPLAY = 3  # Maximum worst predictions to display in prompts
MAX_WORST_INDICES = 20  # Maximum worst indices for analysis in error feedback
MAX_WORST_RESIDUAL_SAMPLES = 10  # Maximum worst residual samples to show in logs/prompts

# Feature display limits
MAX_FEATURES_IN_DESCRIPTION = 5  # Maximum features in dataset descriptions
MAX_FEATURES_IN_EXAMPLE = 20  # Maximum features to show in examples
MAX_FEATURES_IN_COMPONENT_LIST = 3  # Maximum features in component lists
MAX_FEATURES_IN_FORMULA_DISPLAY = 5  # Maximum features in formula display

# Display/truncation limits (for UI/logging)
MAX_PROMPT_PREVIEW_LENGTH = 1000  # Maximum characters for prompt preview
MAX_ERROR_MESSAGE_LENGTH = 500  # Maximum characters for error messages
MAX_MECHANISM_SUMMARY_LENGTH = 200  # Maximum characters for mechanism summary
MAX_FEATURE_DISPLAY_LENGTH = 50  # Maximum characters for feature display
MAX_MECHANISM_PREVIEW_LENGTH = 100  # Maximum characters for mechanism preview
MAX_INTERCEPT_DISPLAY = 3  # Maximum intercepts to display

# =========================
# ML MECHANISM CONFIGURATION
# =========================

# XGBoost parameters
XGBOOST_N_ESTIMATORS = 200  # Number of boosting rounds
XGBOOST_MAX_DEPTH = 4  # Maximum tree depth
XGBOOST_LEARNING_RATE = 0.1  # Learning rate
XGBOOST_RANDOM_STATE = RANDOM_STATE  # Random state for reproducibility

# LogisticRegression parameters
LOGISTIC_REGRESSION_MAX_ITER = 1000  # Maximum iterations for convergence

# TabICL parameters
TABICL_N_ESTIMATORS = 32  # Number of ensemble members
TABICL_RANDOM_STATE = RANDOM_STATE  # Random state for reproducibility

# Baticl parameters
BATICL_BATCH_SIZE = 4  # Batch size for Baticl inference
BATICL_DEVICE = "cpu"  # Device to use ("cpu" or "cuda")
BATICL_N_JOBS = 1  # Number of parallel jobs
BATICL_USE_AMP = False  # Whether to use automatic mixed precision
BATICL_CHECKPOINT_VERSION = "baticl-classifier-v1.1-0506.ckpt"  # Checkpoint version

# =========================
# TEXTGRAD CONFIGURATION
# =========================

# TextGrad routing guidance thresholds
TEXTGRAD_MIN_WEIGHT_THRESHOLD = 0.3  # Minimum weight to contribute meaningfully

# TextGrad error pattern detection thresholds
TEXTGRAD_OVERESTIMATION_THRESHOLD = 0.6  # Threshold for overestimation pattern detection (lowered from 0.7)
TEXTGRAD_FEATURE_ERROR_THRESHOLD = 1.3  # Threshold for feature-specific errors (lowered from 1.5)
TEXTGRAD_FEATURE_ERROR_THRESHOLD_STRICT = 1.4  # Stricter threshold for feature-specific errors (lowered from 1.8)

# TextGrad iteration history
TEXTGRAD_MAX_ITERATION_HISTORY = 3  # Maximum number of previous iterations to show in context


def set_output_dir(run_folder_name: str):
    """Set the output directory for MA-ICL results"""
    global RUN_FOLDER_NAME, OUTPUT_DIR
    RUN_FOLDER_NAME = run_folder_name
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_FOLDER_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# =========================
# CONFIG LOADER
# =========================

_MAICL_CONFIG = None


def load_maicl_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load MA-ICL configuration from YAML file"""
    global _MAICL_CONFIG
    if _MAICL_CONFIG is not None:
        return _MAICL_CONFIG
    
    if config_path is None:
        # Default to maicl_config.yaml in the same directory as this file
        config_path = Path(__file__).parent / "maicl_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            _MAICL_CONFIG = yaml.safe_load(f)
        return _MAICL_CONFIG
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Error loading config file: {e}, using defaults")
        return {}


def get_prompt(key_path: str, **kwargs) -> str:
    """Get a prompt from config by key path (e.g., 'prediction.single_mechanism')"""
    config = load_maicl_config()
    keys = key_path.split('.')
    value = config.get('prompts', {})
    for key in keys:
        value = value.get(key, '')
        if not value:
            logger.warning(f"Prompt not found: {key_path}")
            return ''
    
    # Format the prompt with provided kwargs
    if kwargs:
        try:
            return value.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing placeholder in prompt {key_path}: {e}")
            return value
    
    return value


def get_training_config(key_path: str, default=None):
    """Get a training configuration value by key path (e.g., 'metric_improvement.primary.regression.r2')"""
    config = load_maicl_config()
    keys = key_path.split('.')
    value = config.get('training', {})
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
        if value is None:
            return default
    return value if value is not None else default


# Convenience functions for common training config values
def get_metric_improvement_threshold(task_type: str, metric: str, level: str = "primary") -> float:
    """Get metric improvement threshold for acceptance
    
    Args:
        task_type: 'regression' or 'classification'
        metric: 'r2', 'mae', 'acc', or 'f1'
        level: 'primary' or 'secondary'
    
    Returns:
        Threshold value (default: 0.005 for primary, 0.01 for secondary)
    """
    key = f"metric_improvement.{level}.{task_type}.{metric}"
    default = 0.005 if level == "primary" else 0.01
    return get_training_config(key, default)


def get_loss_degradation_threshold(comparison: str = "vs_current") -> float:
    """Get loss degradation threshold
    
    Args:
        comparison: 'vs_current' or 'vs_best'
    
    Returns:
        Threshold value (default: 0.01 for vs_current, 0.05 for vs_best)
    """
    key = f"loss_degradation.{comparison}"
    default = 0.01 if comparison == "vs_current" else 0.05
    return get_training_config(key, default)


def get_metric_degradation_threshold(task_type: str, metric: str, comparison: str = "vs_best") -> float:
    """Get metric degradation threshold for rejection
    
    Args:
        task_type: 'regression' or 'classification'
        metric: 'r2', 'mae', 'acc', or 'f1'
        comparison: 'vs_best' or 'vs_current'
    
    Returns:
        Threshold value
    """
    if comparison == "vs_current" and task_type == "classification":
        key = f"metric_degradation.{task_type}.max_{metric}_degradation_current"
        default = 0.10
    else:
        key = f"metric_degradation.{task_type}.max_{metric}_degradation"
        default = 0.02 if metric in ["r2"] else 0.05
    return get_training_config(key, default)


def get_adaptive_threshold_config() -> dict:
    """Get adaptive threshold configuration"""
    config = load_maicl_config()
    training = config.get('training', {})
    adaptive = training.get('adaptive_threshold', {})
    return {
        'base_regression': adaptive.get('base_regression', 0.005),
        'base_classification': adaptive.get('base_classification', 0.005),
        'iteration_decay': adaptive.get('iteration_decay', 0.1),
        'max_degradation_early': adaptive.get('max_degradation_early', 0.01),
        'early_iteration_limit': adaptive.get('early_iteration_limit', 3)
    }


def get_best_snapshot_threshold(task_type: str, metric: str) -> float:
    """Get threshold for updating best snapshot"""
    key = f"best_snapshot.{task_type}.{metric}_improvement"
    # Default: match the *acceptance* improvement threshold (primary) so that any
    # accepted update that improves the best metric can also become the "best snapshot".
    # This avoids unintuitive behavior where an update is accepted (e.g., MAE improves),
    # but the run still restores an older snapshot because the best-snapshot threshold
    # was stricter than acceptance.
    default = get_metric_improvement_threshold(task_type, metric, level="primary")
    return get_training_config(key, default)


def get_restoration_config(task_type: str) -> float:
    """Get max allowed degradation for restoration"""
    key = f"restoration.max_allowed_degradation_{task_type}"
    default = 0.2 if task_type == "regression" else 0.05
    return get_training_config(key, default)


def get_performance_score_config() -> dict:
    """Get performance score configuration"""
    config = load_maicl_config()
    training = config.get('training', {})
    perf = training.get('performance_scores', {})
    return {
        'min': perf.get('min_performance', 0.01),
        'min_ml_dominant': perf.get('min_performance_ml_dominant', 0.05),
        'min_similar': perf.get('min_performance_similar', 0.1),
        'max': perf.get('max_performance', 1.0),
        'mae_epsilon': perf.get('mae_epsilon', 0.01)
    }


def get_confidence_config() -> dict:
    """Get confidence configuration"""
    config = load_maicl_config()
    training = config.get('training', {})
    conf = training.get('confidence', {})
    return {
        'default': conf.get('default', 0.1),
        'min': conf.get('min', 0.1),
        'max': conf.get('max', 10.0)
    }


def get_scaling_tolerance() -> float:
    """Get scaling verification tolerance"""
    return get_training_config('scaling.tolerance', 0.01)


def get_llm_temperature() -> float:
    """Get LLM temperature from config
    
    Returns:
        Temperature value (default: 0.8)
    """
    config = load_maicl_config()
    llm_config = config.get('llm', {})
    return llm_config.get('temperature', 0.8)

