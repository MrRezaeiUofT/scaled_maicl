"""
ML Mechanism module for MA-ICL.

This module contains the MLModelMechanism class and related helper functions
for training and using ML models as mechanisms in the MA-ICL system.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge

# Optional dependencies
# NOTE: Do NOT import heavy optional libs (xgboost/tabicl) at module import time.
# Some environments can segfault during import; import lazily only if requested.
_HAS_XGB = False
_HAS_TABICL = False

# NOTE: Do NOT import baticl at module import time.
# Some environments can segfault during import (cannot be caught). Import it lazily only if requested.
_HAS_BATICL = False

logger = logging.getLogger(__name__)

# Import constants from maicl_config
try:
    from maicl_config import (
        RANDOM_STATE,
        MAX_TOP_FEATURES,
        MAX_TOP_FEATURES_DISPLAY,
        MAX_INTERCEPT_DISPLAY,
        XGBOOST_N_ESTIMATORS,
        XGBOOST_MAX_DEPTH,
        XGBOOST_LEARNING_RATE,
        XGBOOST_RANDOM_STATE,
        LOGISTIC_REGRESSION_MAX_ITER,
        TABICL_N_ESTIMATORS,
        TABICL_RANDOM_STATE,
        BATICL_BATCH_SIZE,
        BATICL_DEVICE,
        BATICL_N_JOBS,
        BATICL_USE_AMP,
        BATICL_CHECKPOINT_VERSION
    )
except ImportError:
    # Fallback defaults if maicl_config is not available
    RANDOM_STATE = 42
    MAX_TOP_FEATURES = 20
    MAX_TOP_FEATURES_DISPLAY = 20
    MAX_INTERCEPT_DISPLAY = 3
    XGBOOST_N_ESTIMATORS = 200
    XGBOOST_MAX_DEPTH = 4
    XGBOOST_LEARNING_RATE = 0.1
    XGBOOST_RANDOM_STATE = 42
    LOGISTIC_REGRESSION_MAX_ITER = 1000
    TABICL_N_ESTIMATORS = 32
    TABICL_RANDOM_STATE = 42
    BATICL_BATCH_SIZE = 4
    BATICL_DEVICE = "cpu"
    BATICL_N_JOBS = 1
    BATICL_USE_AMP = False
    BATICL_CHECKPOINT_VERSION = "baticl-classifier-v1.1-0506.ckpt"


# =========================
# ML MODEL MECHANISM
# =========================

class MLModelMechanism:
    """Wrapper for ML models to work as mechanisms in MA-ICL"""
    def __init__(self, model_name: str = "LinearRegression", task_type: str = "classification"):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_cols = None
        self.task_type = task_type
        self._tabicl_feature_cols = None  # For TabICL DataFrame conversion
        self._tabicl_regression_bins = None  # Number of bins for TabICL regression quantization
        self._tabicl_bin_edges = None  # Bin edges for regression quantization
        self._tabicl_bin_centers = None  # Bin centers for regression dequantization
        self._tabpfn_use_subprocess = False  # Use subprocess isolation for TabPFN
        self._tabpfn_training_data_path = None  # Path to saved training data for TabPFN subprocess
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_cols: List[str], y_scaler: Any = None):
        """Train the ML model with calibration and stable class mapping for classification"""
        self.feature_cols = feature_cols
        
        if self.model_name == "LinearRegression":
            self.model = LinearRegression()
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression(max_iter=LOGISTIC_REGRESSION_MAX_ITER, random_state=RANDOM_STATE)
        elif self.model_name == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif self.model_name == "KernelRidge":
            self.model = KernelRidge(alpha=1.0, kernel='rbf')
        elif self.model_name == "XGBoost":
            try:
                from xgboost import XGBClassifier, XGBRegressor  # type: ignore
            except Exception as e:
                logger.warning("XGBoost not installed or failed to import, falling back to baseline")
                if self.task_type == "regression":
                    self.model_name = "LinearRegression"
                    self.model = LinearRegression()
                else:
                    self.model_name = "LogisticRegression"
                    self.model = LogisticRegression(max_iter=LOGISTIC_REGRESSION_MAX_ITER, random_state=RANDOM_STATE)
            else:
                if self.task_type == "regression":
                    self.model = XGBRegressor(
                        n_estimators=XGBOOST_N_ESTIMATORS,
                        max_depth=XGBOOST_MAX_DEPTH,
                        learning_rate=XGBOOST_LEARNING_RATE,
                        random_state=XGBOOST_RANDOM_STATE,
                    )
                else:
                    self.model = XGBClassifier(
                        use_label_encoder=False,
                        eval_metric='logloss',
                        n_estimators=XGBOOST_N_ESTIMATORS,
                        max_depth=XGBOOST_MAX_DEPTH,
                        learning_rate=XGBOOST_LEARNING_RATE,
                        random_state=XGBOOST_RANDOM_STATE,
                    )
        elif self.model_name == "TabICL":
            try:
                from tabicl import TabICLClassifier  # type: ignore
            except Exception as e:
                raise RuntimeError("TabICL not installed or failed to import. Install with: pip install tabicl") from e
            
            # TabICL can be used for regression via quantization
            if self.task_type != "classification" and not hasattr(self, '_tabicl_regression_bins'):
                raise ValueError("TabICL for regression requires _tabicl_regression_bins to be set. "
                               "Use MLModelMechanism with _tabicl_regression_bins parameter.")
            
            logger.info("Initializing TabICL...")
            try:
                self.model = TabICLClassifier(
                    n_estimators=TABICL_N_ESTIMATORS,  # number of ensemble members
                    random_state=TABICL_RANDOM_STATE,
                    verbose=True
                )
                logger.info("✓ TabICL initialized successfully")
            except Exception as e:
                raise RuntimeError(
                    f"TabICL initialization failed: {e}. "
                    f"Please use --ml_mech logreg or --ml_mech xgboost instead."
                ) from e
            # Store feature columns for DataFrame conversion
            self._tabicl_feature_cols = feature_cols
        elif self.model_name == "TabPFN":
            # Use subprocess isolation for TabPFN to prevent segmentation faults
            logger.info("=" * 80)
            logger.info("ℹ️  TabPFN ML mechanism: Using subprocess isolation for safety")
            logger.info("=" * 80)
            logger.info("TabPFN will run in isolated subprocess to prevent segmentation faults.")
            logger.info("This is slower than direct execution but ensures stability.")
            logger.info("=" * 80)
            # Store feature columns for DataFrame conversion (TabPFN expects DataFrames)
            self._tabpfn_feature_cols = feature_cols
            self._tabpfn_use_subprocess = True
            # Model will be None - we'll use subprocess for all operations
            self.model = None
        elif self.model_name == "Baticl":
            # Lazy import to avoid hard crashes on environments where baticl import is unstable.
            try:
                from baticl import BaticlClassifier  # type: ignore
            except Exception as e:
                raise RuntimeError("Baticl not installed or failed to import. Install with: pip install baticl") from e
            logger.warning("Baticl can cause segmentation faults on some systems. If you encounter crashes, use --ml_mech logreg instead.")
            try:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                self.model = BaticlClassifier(
                    checkpoint_version=BATICL_CHECKPOINT_VERSION,
                    device=BATICL_DEVICE, n_jobs=BATICL_N_JOBS, use_amp=BATICL_USE_AMP, batch_size=BATICL_BATCH_SIZE,
                    allow_auto_download=True, verbose=False
                )
            except Exception as e:
                raise RuntimeError("Baticl initialization failed. Try using --ml_mech logreg or --ml_mech xgboost instead.") from e
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Fit the model with error handling
        try:
            # TabICL and TabPFN expect DataFrame, convert if needed
            if self.model_name == "TabICL":
                X_train_df = pd.DataFrame(X_train, columns=self._tabicl_feature_cols)
            elif self.model_name == "TabPFN":
                # Use subprocess isolation for TabPFN training
                X_train_df = pd.DataFrame(X_train, columns=self._tabpfn_feature_cols)
                logger.info(f"Training TabPFN on {len(X_train_df)} samples using subprocess isolation...")
                
                try:
                    import subprocess
                    import tempfile
                    from pathlib import Path
                    import sys
                    
                    # Create temporary directory for data exchange
                    temp_dir = tempfile.mkdtemp(prefix="tabpfn_ml_")
                    temp_path = Path(temp_dir)
                    
                    X_train_path = temp_path / "X_train.csv"
                    y_train_path = temp_path / "y_train.npy"
                    output_path = temp_path / "train_result.json"
                    
                    # Save training data
                    X_train_df.to_csv(X_train_path, index=False)
                    np.save(y_train_path, y_train)
                    
                    # Find subprocess script
                    script_path = Path(__file__).parent / "run_tabpfn_ml_mechanism.py"
                    if not script_path.exists():
                        raise RuntimeError(f"TabPFN subprocess script not found at {script_path}")
                    
                    # Run training in subprocess
                    result = subprocess.run(
                        [sys.executable, str(script_path), "train",
                         str(X_train_path), str(y_train_path), 
                         self.task_type, str(output_path)],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout for training
                    )
                    
                    if result.returncode != 0:
                        error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                        raise RuntimeError(f"TabPFN training subprocess failed: {error_msg}")
                    
                    if not output_path.exists():
                        raise RuntimeError("TabPFN training subprocess completed but no output file found")
                    
                    # Load training result
                    with open(output_path, 'r') as f:
                        train_result = json.load(f)
                    
                    if not train_result.get('success', False):
                        error_msg = train_result.get('error', 'Unknown error')
                        raise RuntimeError(f"TabPFN training failed: {error_msg}")
                    
                    # Store paths for later predictions
                    self._tabpfn_training_data_path = {
                        'X_train_path': str(X_train_path),
                        'y_train_path': str(y_train_path),
                        'temp_dir': temp_dir
                    }
                    
                    logger.info("  ✓ TabPFN training completed successfully (via subprocess)")
                    
                except subprocess.TimeoutExpired:
                    raise RuntimeError("TabPFN training subprocess timed out (>10 minutes)")
                except Exception as e:
                    raise RuntimeError(f"TabPFN training via subprocess failed: {e}. Try using --ml_mech linear or --ml_mech xgboost instead.") from e
            else:
                self.model.fit(X_train, y_train)
        except Exception as e:
            if self.model_name in ["TabICL", "TabPFN", "Baticl"]:
                raise RuntimeError(
                    f"{self.model_name} training failed. This may be due to a segmentation fault "
                    f"(which cannot be caught in Python) or other system-level issues. "
                    f"Try using --ml_mech linear or --ml_mech xgboost instead. "
                    f"Original error: {e}"
                ) from e
            raise
        
        # For classification: ensure calibrated probabilities and build stable class mapping
        if self.task_type == "classification":
            # TabPFN uses subprocess, so skip calibration (handled in subprocess)
            # TabICL typically has predict_proba, so skip calibration for it
            # Check if model has predict_proba; if not, wrap with calibration
            if self.model_name == "TabPFN":
                # TabPFN uses subprocess - class mapping will be inferred from training data
                unique_train = np.unique(y_train)
                self._class_order = sorted(unique_train.tolist())
                self._class_to_idx = {c: i for i, c in enumerate(self._class_order)}
                logger.info(f"  [ClassMap] Built class mapping for TabPFN (subprocess): {len(self._class_order)} classes")
            elif self.model_name != "TabICL" and not hasattr(self.model, "predict_proba"):
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                    n_classes = len(np.unique(y_train))
                    if n_classes > 2:
                        # Multi-class: use isotonic calibration
                        calibration_method = "isotonic"
                    else:
                        # Binary: use sigmoid calibration
                        calibration_method = "sigmoid"
                    logger.info(f"  [Calibration] Wrapping {self.model_name} with CalibratedClassifierCV (method={calibration_method})")
                    self.model = CalibratedClassifierCV(self.model, method=calibration_method, cv=min(3, len(X_train) // 10 + 1))
                    self.model.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"  [Calibration] Could not add calibration wrapper: {e}")
            
            # Build stable class→index mapping from the trained estimator
            if self.model_name == "TabPFN":
                # TabPFN class mapping already handled above
                pass
            elif hasattr(self.model, "classes_"):
                self._class_order = self.model.classes_.tolist()
                self._class_to_idx = {c: i for i, c in enumerate(self._class_order)}
                logger.info(f"  [ClassMap] Built class mapping: {len(self._class_order)} classes")
                # Validate training labels are in the mapping
                unique_train = np.unique(y_train)
                missing = [yy for yy in unique_train if yy not in self._class_to_idx]
                if missing:
                    logger.warning(f"  [ClassMap] Missing classes in classifier mapping: {missing}")
            else:
                # Fallback: infer from unique training labels
                unique_train = np.unique(y_train)
                self._class_order = sorted(unique_train.tolist())
                self._class_to_idx = {c: i for i, c in enumerate(self._class_order)}
                logger.warning(f"  [ClassMap] Model lacks classes_ attribute; inferred from training data: {len(self._class_order)} classes")
        else:
            # Regression: no class mapping needed
            self._class_order = None
            self._class_to_idx = None
        
        # Extract trained model information for description
        self._model_info = self._extract_model_info()
        
        self.is_trained = True
        logger.info(f"  ✓ Trained {self.model_name} model on {len(X_train)} samples")
    
    def predict(self, x_dict: Dict[str, float], return_class_index: bool = False) -> float:
        """Predict output for given input"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        x_array = np.array([x_dict[col] for col in self.feature_cols]).reshape(1, -1)

        # For regression tasks, handle TabICL quantization specially
        if getattr(self, "task_type", "classification") == "regression":
            if self.model_name == "TabICL" and hasattr(self, '_tabicl_bin_centers') and self._tabicl_bin_centers is not None:
                # TabICL regression: predict bin, then map to continuous value
                x_df = pd.DataFrame(x_array, columns=self._tabicl_feature_cols)
                try:
                    # Try to use probabilities for weighted prediction (more accurate)
                    if hasattr(self.model, 'predict_proba'):
                        try:
                            proba = self.model.predict_proba(x_df)[0]
                            # Get class indices (bins) that the model knows about
                            if hasattr(self.model, 'classes_'):
                                class_indices = self.model.classes_
                                # Map probabilities to bin centers and compute weighted average
                                weighted_sum = 0.0
                                total_prob = 0.0
                                for i, class_idx in enumerate(class_indices):
                                    if 0 <= class_idx < len(self._tabicl_bin_centers):
                                        weighted_sum += proba[i] * self._tabicl_bin_centers[class_idx]
                                        total_prob += proba[i]
                                if total_prob > 0:
                                    prediction = weighted_sum / total_prob
                                    return float(prediction)
                        except Exception:
                            # Fall back to hard prediction if probability prediction fails
                            pass
                    
                    # Fallback: use hard prediction (most likely bin)
                    bin_idx = int(self.model.predict(x_df)[0])
                    # Clip to valid range
                    bin_idx = np.clip(bin_idx, 0, len(self._tabicl_bin_centers) - 1)
                    # Map bin index to continuous value (use bin center)
                    prediction = float(self._tabicl_bin_centers[bin_idx])
                    return prediction
                except Exception as e:
                    logger.warning(f"TabICL regression prediction failed: {e}")
                    # Return middle bin center as fallback
                    mid_idx = len(self._tabicl_bin_centers) // 2
                    return float(self._tabicl_bin_centers[mid_idx])
            else:
                # Standard regression: return raw prediction
                prediction = self.model.predict(x_array)[0]
                return float(prediction)

        is_multi_class = (hasattr(self.model, 'classes_') and len(self.model.classes_) > 2)
        
        if self.model_name in ("LogisticRegression", "XGBoost"):
            # For classification: always return class index, never probabilities
            if self.task_type == "classification":
                class_idx = int(self.model.predict(x_array)[0])
                return float(class_idx)
            else:
                # Regression: return raw prediction
                prediction = self.model.predict(x_array)[0]
                return float(prediction)
        
        if self.model_name == "TabICL":
            # TabICL expects DataFrame, convert if needed
            try:
                x_df = pd.DataFrame(x_array, columns=self._tabicl_feature_cols)
                class_idx = int(self.model.predict(x_df)[0])
                return float(class_idx)
            except Exception as e:
                logger.warning(f"TabICL prediction failed: {e}")
                return 0.0
        
        if self.model_name == "TabPFN":
            # TabPFN uses subprocess isolation for predictions
            if not self._tabpfn_use_subprocess or self._tabpfn_training_data_path is None:
                logger.warning("TabPFN not properly initialized with subprocess isolation")
                return 0.0
            
            try:
                import subprocess
                import tempfile
                from pathlib import Path
                import sys
                import json
                
                # Convert to DataFrame
                x_df = pd.DataFrame(x_array, columns=self._tabpfn_feature_cols)
                
                # Create temporary file for prediction data
                temp_dir = Path(self._tabpfn_training_data_path['temp_dir'])
                X_pred_path = temp_dir / "X_pred.csv"
                output_path = temp_dir / "pred_result.json"
                
                # Save prediction data
                x_df.to_csv(X_pred_path, index=False)
                
                # Find subprocess script
                script_path = Path(__file__).parent / "run_tabpfn_ml_mechanism.py"
                if not script_path.exists():
                    logger.warning("TabPFN subprocess script not found")
                    return 0.0
                
                # Run prediction in subprocess
                result = subprocess.run(
                    [sys.executable, str(script_path), "predict",
                     self._tabpfn_training_data_path['X_train_path'],
                     self._tabpfn_training_data_path['y_train_path'],
                     str(X_pred_path),
                     self.task_type,
                     str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout for prediction
                )
                
                if result.returncode != 0:
                    logger.warning(f"TabPFN prediction subprocess failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                    return 0.0
                
                if not output_path.exists():
                    logger.warning("TabPFN prediction subprocess completed but no output file found")
                    return 0.0
                
                # Load prediction result
                with open(output_path, 'r') as f:
                    pred_result = json.load(f)
                
                if not pred_result.get('success', False):
                    error_msg = pred_result.get('error', 'Unknown error')
                    logger.warning(f"TabPFN prediction failed: {error_msg}")
                    return 0.0
                
                predictions = pred_result.get('predictions', [])
                if not predictions:
                    logger.warning("TabPFN prediction returned empty results")
                    return 0.0
                
                return float(predictions[0])
                
            except subprocess.TimeoutExpired:
                logger.warning("TabPFN prediction subprocess timed out")
                return 0.0
            except Exception as e:
                logger.warning(f"TabPFN prediction failed: {e}")
                return 0.0
        
        if self.model_name == "Baticl":
            # For classification: always return class index, never probabilities
            if self.task_type == "classification":
                try:
                    class_idx = int(self.model.predict(x_array)[0])
                    return float(class_idx)
                except:
                    return 0.0
            else:
                # Regression: return raw prediction
                try:
                    prediction = self.model.predict(x_array)[0]
                    return float(prediction)
                except:
                    return 0.0
        
        if self.model_name == "KernelRidge":
            score = float(self.model.predict(x_array)[0])
            # For regression, return raw KernelRidge prediction.
            # Only squash to [0,1] when explicitly used for classification.
            if self.task_type == "classification":
                return float(1.0 / (1.0 + np.exp(-score)))
            return score
        
        prediction = self.model.predict(x_array)[0]
        return float(prediction)
    
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract information from trained model for description"""
        info = {
            "coefficients": None,
            "feature_importance": None,
            "intercept": None,
            "n_features": len(self.feature_cols) if self.feature_cols else 0
        }
        
        if not self.is_trained or self.model is None:
            return info
        
        try:
            # Get the underlying model (unwrap if calibrated)
            underlying_model = self.model
            # Unwrap CalibratedClassifierCV or other wrappers
            if hasattr(self.model, "base_estimator") and self.model.base_estimator is not None:
                underlying_model = self.model.base_estimator
            elif hasattr(self.model, "estimator") and self.model.estimator is not None:
                underlying_model = self.model.estimator
            
            # LinearRegression: extract coefficients and intercept
            if self.model_name == "LinearRegression":
                if hasattr(underlying_model, "coef_"):
                    coef = underlying_model.coef_
                    intercept = underlying_model.intercept_ if hasattr(underlying_model, "intercept_") else None
                    info["coefficients"] = coef.flatten() if coef.ndim > 1 else coef
                    info["intercept"] = float(intercept) if intercept is not None else None
            
            # LogisticRegression: extract coefficients and intercept
            elif self.model_name == "LogisticRegression":
                if hasattr(underlying_model, "coef_"):
                    coef = underlying_model.coef_
                    intercept = underlying_model.intercept_ if hasattr(underlying_model, "intercept_") else None
                    # For multi-class, coef_ is 2D (n_classes, n_features)
                    if coef.ndim == 2:
                        info["coefficients"] = coef.tolist()
                    else:
                        info["coefficients"] = coef.flatten().tolist()
                    info["intercept"] = intercept.tolist() if hasattr(intercept, "tolist") else (float(intercept) if intercept is not None else None)
            
            # XGBoost: extract feature importance
            elif self.model_name == "XGBoost":
                if hasattr(underlying_model, "feature_importances_"):
                    importance = underlying_model.feature_importances_
                    info["feature_importance"] = importance.tolist() if hasattr(importance, "tolist") else importance
                elif hasattr(underlying_model, "get_booster"):
                    try:
                        booster = underlying_model.get_booster()
                        importance_dict = booster.get_score(importance_type='weight')
                        info["feature_importance"] = importance_dict
                    except Exception:
                        pass
            
            # KernelRidge: extract dual coefficients (if available)
            elif self.model_name == "KernelRidge":
                if hasattr(underlying_model, "dual_coef_"):
                    info["coefficients"] = underlying_model.dual_coef_.flatten().tolist() if underlying_model.dual_coef_.ndim > 1 else underlying_model.dual_coef_.tolist()
                if hasattr(underlying_model, "alpha"):
                    info["regularization"] = float(underlying_model.alpha)
                    
        except Exception as e:
            logger.debug(f"Could not extract model info for {self.model_name}: {e}")
        
        return info
    
    def get_description(self) -> str:
        """Get mechanism description with trained model features"""
        base_descriptions = {
            "LinearRegression": "ML Mechanism (LinearRegression): Data-driven linear regression model",
            "LogisticRegression": "ML Mechanism (LogisticRegression): Data-driven logistic regression classifier",
            "XGBoost": "ML Mechanism (XGBoost): Gradient-boosted decision trees with calibrated probabilities",
            "TabICL": "ML Mechanism (TabICL): Tabular in-context learning baseline using transformer architecture",
            "TabPFN": "ML Mechanism (TabPFN): Prior-data Fitted Networks for tabular data using transformer architecture",
            "Baticl": "ML Mechanism (Baticl): Tabular in-context learning baseline using transformer architecture",
            "KNN": "ML Mechanism (KNN): K-nearest neighbors classifier using distance-based similarity",
            "KernelRidge": "ML Mechanism (KernelRidge): Kernel-based ridge regression with RBF kernel"
        }
        
        base_desc = base_descriptions.get(self.model_name, f"ML Mechanism ({self.model_name}): Data-driven baseline")
        
        # If model is trained, add detailed information
        if self.is_trained and hasattr(self, "_model_info") and self._model_info:
            info = self._model_info
            details = []
            
            # Add feature count
            if info.get("n_features", 0) > 0:
                details.append(f"{info['n_features']} features")
            
            # LinearRegression: add coefficients
            if self.model_name == "LinearRegression" and info.get("coefficients") is not None:
                coef = info["coefficients"]
                intercept = info.get("intercept")
                
                # Get top features by absolute coefficient value
                if self.feature_cols and len(coef) == len(self.feature_cols):
                    coef_abs = np.abs(coef)
                    top_k = min(5, len(coef))
                    top_indices = np.argsort(coef_abs)[::-1][:top_k]
                    
                    coef_strs = []
                    for idx in top_indices:
                        feat_name = self.feature_cols[idx]
                        weight = float(coef[idx])
                        coef_strs.append(f"{feat_name}={weight:.4f}")
                    
                    details.append(f"top weights: {', '.join(coef_strs)}")
                    if intercept is not None:
                        details.append(f"intercept={intercept:.4f}")
            
            # LogisticRegression: add coefficients
            elif self.model_name == "LogisticRegression" and info.get("coefficients") is not None:
                coef = info["coefficients"]
                intercept = info.get("intercept")
                
                # Handle multi-class (2D) vs binary (1D) coefficients
                if isinstance(coef, list) and len(coef) > 0:
                    if isinstance(coef[0], list):
                        # Multi-class: show first class coefficients
                        coef_flat = np.array(coef[0])
                    else:
                        # Binary: use directly
                        coef_flat = np.array(coef)
                    
                    if self.feature_cols and len(coef_flat) == len(self.feature_cols):
                        coef_abs = np.abs(coef_flat)
                        top_k = min(5, len(coef_flat))
                        top_indices = np.argsort(coef_abs)[::-1][:top_k]
                        
                        coef_strs = []
                        for idx in top_indices:
                            feat_name = self.feature_cols[idx]
                            weight = float(coef_flat[idx])
                            coef_strs.append(f"{feat_name}={weight:.4f}")
                        
                        details.append(f"top weights: {', '.join(coef_strs)}")
                        if intercept is not None:
                            if isinstance(intercept, list):
                                details.append(f"intercepts={[float(x) for x in intercept[:MAX_INTERCEPT_DISPLAY]]}...")
                            else:
                                details.append(f"intercept={float(intercept):.4f}")
            
            # XGBoost: add feature importance
            elif self.model_name == "XGBoost" and info.get("feature_importance") is not None:
                importance = info["feature_importance"]
                
                # Check if this is a DeepChem dataset (ECFP fingerprints)
                is_deepchem_ml = (self.feature_cols and len(self.feature_cols) > 0 and 
                                 all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
                
                if isinstance(importance, dict):
                    # Dictionary format (from get_booster)
                    if is_deepchem_ml:
                        # For DeepChem: don't show specific ECFP bit names
                        details.append(f"uses ECFP fingerprints ({len(self.feature_cols)} bits) - LLM should work with SMILES strings, not ECFP bits")
                    else:
                        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:MAX_TOP_FEATURES]
                        imp_strs = [f"{k}={v:.2f}" for k, v in sorted_imp]
                        details.append(f"top importance: {', '.join(imp_strs)}")
                elif isinstance(importance, (list, np.ndarray)):
                    # Array format
                    importance_arr = np.array(importance)
                    if self.feature_cols and len(importance_arr) == len(self.feature_cols):
                        if is_deepchem_ml:
                            # For DeepChem: don't show specific ECFP bit names
                            details.append(f"uses ECFP fingerprints ({len(self.feature_cols)} bits) - LLM should work with SMILES strings, not ECFP bits")
                        else:
                            top_k = min(5, len(importance_arr))
                            top_indices = np.argsort(importance_arr)[::-1][:top_k]
                            
                            imp_strs = []
                            for idx in top_indices:
                                feat_name = self.feature_cols[idx]
                                imp_val = float(importance_arr[idx])
                                imp_strs.append(f"{feat_name}={imp_val:.4f}")
                            
                            details.append(f"top importance: {', '.join(imp_strs)}")
            
            # KernelRidge: add regularization info
            elif self.model_name == "KernelRidge":
                if info.get("regularization") is not None:
                    details.append(f"alpha={info['regularization']:.4f}")
            
            # Combine base description with details
            if details:
                return f"{base_desc}. Trained model: {', '.join(details)}."
        
        return base_desc
    
    def get_confidence_heuristic(self, x_dict: Dict[str, float]) -> float:
        """Get confidence score for prediction"""
        return 8.0
    
    def set_performance_boost(self, performance_ratio: float):
        """Set performance boost multiplier"""
        self._performance_boost = max(1.0, min(3.0, performance_ratio))
    
    def get_confidence_heuristic_with_boost(self, x_dict: Dict[str, float]) -> float:
        """Get confidence score with performance boost"""
        base_conf = self.get_confidence_heuristic(x_dict)
        boost = getattr(self, '_performance_boost', 1.0)
        return base_conf * boost


# =========================
# HELPER FUNCTIONS FOR ML MECHANISMS
# =========================

def compute_ml_residuals(
    ml_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: List[str],
    class_names: Optional[List[str]] = None,
    task_type: str = "classification",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]]:
    """
    Compute ML predictions and residuals on training set.
    - classification: residual = 1.0 if prediction is wrong, 0.0 if correct (0/1 error, no probabilities used)
    - regression: residual = |y_true - y_pred|
    Returns: (sorted_indices_desc, residuals, predictions, probabilities_or_None)
    Note: For classification, probabilities are not extracted - only class predictions are used.
    """
    logger.info("\n[Residuals] Computing ML residuals...")
    
    # Get stable class mapping from the trained model
    class_to_idx = getattr(ml_model, "_class_to_idx", None)
    class_order = getattr(ml_model, "_class_order", None)
    
    # Validate class mapping for classification
    if task_type == "classification" and class_to_idx is not None:
        unique_y = np.unique(y_train)
        missing = [yy for yy in unique_y if yy not in class_to_idx]
        if missing:
            logger.warning(f"[ProbaMap] Missing classes in classifier mapping: {missing}")
    
    predictions = []
    probas: List[Any] = []
    is_multiclass = task_type == "classification" and class_names is not None and len(class_names) > 2
    
    # For both classification and regression: use batch prediction when possible
    try:
        # TabPFN and TabICL require DataFrames
        requires_dataframe = ml_model.model_name in ["TabPFN", "TabICL"]
        
        # TabPFN uses subprocess isolation, so model is None - must use per-sample prediction
        # Check if model exists and is not None before attempting batch prediction
        can_use_batch = (
            hasattr(ml_model, "model") 
            and ml_model.model is not None 
            and hasattr(ml_model.model, "predict")
            and ml_model.model_name != "TabPFN"  # TabPFN always uses subprocess, model is None
        )
        
        if task_type == "classification":
            # Classification: only get class predictions, no probabilities
            if can_use_batch:
                # Direct batch prediction for efficiency
                if requires_dataframe:
                    # TabICL needs DataFrame
                    X_train_df = pd.DataFrame(X_train, columns=ml_model._tabicl_feature_cols)
                    predictions_batch = ml_model.model.predict(X_train_df)
                else:
                    predictions_batch = ml_model.model.predict(X_train)
                predictions = predictions_batch.tolist()
                # No probabilities needed for classification
                probas = [None] * len(predictions)
            else:
                # Fallback to per-sample prediction (required for TabPFN, or when model is None)
                logger.info(f"[Residuals] Using per-sample prediction for {ml_model.model_name} (model={'None' if ml_model.model is None else 'available'})")
                for i in range(len(X_train)):
                    x_dict = {feature_cols[j]: float(X_train[i, j]) for j in range(len(feature_cols))}
                    try:
                        pred = ml_model.predict(x_dict, return_class_index=True)
                    except Exception:
                        pred = 0.0
                    predictions.append(pred)
                    probas.append(None)  # No probabilities for classification
        else:
            # Regression: use batch prediction for efficiency and accuracy (matching baseline evaluation)
            if can_use_batch:
                # Direct batch prediction for efficiency (same as baseline evaluation)
                if requires_dataframe:
                    # TabICL needs DataFrame
                    X_train_df = pd.DataFrame(X_train, columns=ml_model._tabicl_feature_cols)
                    predictions_batch = ml_model.model.predict(X_train_df)
                else:
                    predictions_batch = ml_model.model.predict(X_train)
                predictions = predictions_batch.tolist()
                probas = [None] * len(predictions)
            else:
                # Fallback to per-sample prediction (required for TabPFN, or when model is None)
                logger.info(f"[Residuals] Using per-sample prediction for {ml_model.model_name} (model={'None' if ml_model.model is None else 'available'})")
                for i in range(len(X_train)):
                    x_dict = {feature_cols[j]: float(X_train[i, j]) for j in range(len(feature_cols))}
                    try:
                        pred = ml_model.predict(x_dict, return_class_index=False)
                    except Exception:
                        pred = 0.0
                    predictions.append(pred)
                    probas.append(None)
    except Exception as e:
        logger.warning(f"[Residuals] Batch prediction failed, falling back to per-sample: {e}")
        # Fallback to per-sample
        for i in range(len(X_train)):
            x_dict = {feature_cols[j]: float(X_train[i, j]) for j in range(len(feature_cols))}
            try:
                if task_type == "classification":
                    pred = ml_model.predict(x_dict, return_class_index=True)
                else:
                    pred = ml_model.predict(x_dict, return_class_index=False)
            except Exception:
                pred = 0.0
            predictions.append(pred)
            probas.append(None)
    
    predictions = np.array(predictions, dtype=float)
    y_true = np.array(y_train)
    residuals = np.zeros_like(y_true, dtype=float)
    
    if task_type == "classification":
        # Classification: use simple 0/1 error based on class predictions only
        for i in range(len(y_true)):
            true_cls = y_true[i]  # Keep as original type (could be int, str, etc.)
            pred_cls = int(np.round(predictions[i]))
            
            # Map true class to index if needed
            if class_to_idx is not None and true_cls in class_to_idx:
                true_cls_int = class_to_idx[true_cls]
            else:
                # Fallback: try direct integer conversion
                try:
                    true_cls_int = int(true_cls)
                except (ValueError, TypeError):
                    # If conversion fails, use 0 as default
                    true_cls_int = 0
            
            # Residual is 1.0 if prediction is wrong, 0.0 if correct
            residuals[i] = 1.0 if pred_cls != true_cls_int else 0.0
    else:
        # Regression: absolute error
        residuals = np.abs(y_true - predictions)
    
    sorted_indices = np.argsort(residuals)[::-1]
    return sorted_indices, residuals, predictions, (probas if any(p is not None for p in probas) else None)


def get_top_k_residual_samples(
    sorted_indices: np.ndarray,
    residuals: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int,
    ml_predictions: Optional[np.ndarray] = None,
    task_type: str = "classification",
    selection_strategy: str = "errors",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select top-K samples guided by ML residuals.
    - classification: residuals are 1.0 for mismatched, 0.0 for matched.
                     Sorting by residual (descending) naturally puts mismatched examples first.
                     Simply pick top-K from sorted_indices (already sorted by residual descending).
    - regression: uses top absolute errors by residual ranking
    Returns: (X_topk, y_topk, top_k_indices)
    """
    if top_k == -1 or top_k >= len(X_train):
        indices = np.arange(len(X_train))
        return X_train[indices], y_train[indices], indices
    
    if task_type == "classification":
        # For classification: residuals are already 1.0 (mismatched) or 0.0 (matched)
        # sorted_indices is already sorted descending by residual, so mismatched (1.0) come first
        # Just pick top-K directly - no filtering needed
        top_k_indices = sorted_indices[:min(top_k, len(sorted_indices))]
    else:
        # Regression: use residual ranking (filter to non-zero residuals)
        non_zero = sorted_indices[residuals[sorted_indices] > 0]
        if len(non_zero) == 0:
            return X_train[:0], y_train[:0], np.array([], dtype=int)
        top_k_indices = non_zero[:min(top_k, len(non_zero))]
    
    return X_train[top_k_indices], y_train[top_k_indices], top_k_indices


def get_ml_prediction_probability(ml_mechanism, x_dict, predicted_class=None, scaler=None, context: str = ""):
    """
    Extract ML model's prediction probability for confidence-based routing.
    For classification: returns predicted class and fixed confidence (no probability extraction).
    For regression: may use probabilities if available.
    Uses stable class mapping from the trained estimator.
    Returns (predicted_class, confidence_score, all_probabilities_or_None)
    
    Note: This function may use _increment_fallback from maicl_lib_v2, which should be imported
    if needed. For now, we'll handle fallbacks locally.
    """
    if ml_mechanism is None or not ml_mechanism.is_trained:
        return None, 0.5, None
    
    # Check if this is a classification task
    task_type = getattr(ml_mechanism, "task_type", "classification")
    
    try:
        feature_order = getattr(ml_mechanism, "_maicl_feature_cols", None)
        feature_encoders = getattr(ml_mechanism, "_maicl_feature_encoders", None)
        mech_scaler = getattr(ml_mechanism, "_maicl_scaler", None)
        underlying_model = ml_mechanism.model if hasattr(ml_mechanism, 'model') else ml_mechanism
        
        # Get stable class mapping
        class_to_idx = getattr(ml_mechanism, "_class_to_idx", None)
        class_order = getattr(ml_mechanism, "_class_order", None)
        
        # For classification: only get class prediction, use fixed confidence
        if task_type == "classification":
            # Get class prediction directly without probabilities
            try:
                if isinstance(x_dict, dict):
                    if feature_order is None:
                        feature_order = sorted(x_dict.keys())
                    encoded_vals = []
                    for col in feature_order:
                        if feature_encoders and col in feature_encoders:
                            try:
                                encoded_val = float(feature_encoders[col].transform([str(x_dict.get(col, ""))])[0])
                            except:
                                encoded_val = 0.0
                        else:
                            try:
                                encoded_val = float(x_dict.get(col, 0.0))
                            except:
                                encoded_val = 0.0
                        encoded_vals.append(encoded_val)
                    x_array = np.array(encoded_vals, dtype=float).reshape(1, -1)
                    if mech_scaler is not None and hasattr(mech_scaler, "transform"):
                        try:
                            apply_scaling = True
                            if hasattr(mech_scaler, "feature_range"):
                                exp_min, exp_max = mech_scaler.feature_range
                                if np.all(x_array >= exp_min - 1e-6) and np.all(x_array <= exp_max + 1e-6):
                                    apply_scaling = False
                            if apply_scaling:
                                x_array = mech_scaler.transform(x_array)
                        except:
                            pass
                else:
                    x_array = np.array(x_dict, dtype=float).reshape(1, -1)
                    if mech_scaler is not None and hasattr(mech_scaler, "transform"):
                        try:
                            apply_scaling = True
                            if hasattr(mech_scaler, "feature_range"):
                                exp_min, exp_max = mech_scaler.feature_range
                                if np.all(x_array >= exp_min - 1e-6) and np.all(x_array <= exp_max + 1e-6):
                                    apply_scaling = False
                            if apply_scaling:
                                x_array = mech_scaler.transform(x_array)
                        except:
                            pass
                
                # Get class prediction directly
                pred_class_idx = int(underlying_model.predict(x_array)[0])
                
                # Map to class name if available
                if class_order is not None and 0 <= pred_class_idx < len(class_order):
                    pred_class = class_order[pred_class_idx]
                else:
                    pred_class = pred_class_idx
                
                # Use fixed confidence for classification (no probability extraction)
                confidence = 0.8  # Fixed confidence score for classification
                
                return pred_class, confidence, None
            except Exception as e:
                logger.debug(f"[ClassPred] Error getting class prediction: {e}")
                return predicted_class, 0.5, None
        
        # For regression: may use probabilities if available (keep existing behavior)
        if not hasattr(underlying_model, "predict_proba"):
            return predicted_class, 0.5, None
        
        # Build encoded, scaled row
        if isinstance(x_dict, dict):
            if feature_order is None:
                feature_order = sorted(x_dict.keys())
            encoded_vals = []
            for col in feature_order:
                if feature_encoders and col in feature_encoders:
                    try:
                        encoded_val = float(feature_encoders[col].transform([str(x_dict.get(col, ""))])[0])
                    except:
                        encoded_val = 0.0
                else:
                    try:
                        encoded_val = float(x_dict.get(col, 0.0))
                    except:
                        encoded_val = 0.0
                encoded_vals.append(encoded_val)
            x_array = np.array(encoded_vals, dtype=float).reshape(1, -1)
            # Apply scaling only if input appears unscaled (avoid double-scaling)
            if mech_scaler is not None and hasattr(mech_scaler, "transform"):
                try:
                    apply_scaling = True
                    if hasattr(mech_scaler, "feature_range"):
                        exp_min, exp_max = mech_scaler.feature_range
                        if np.all(x_array >= exp_min - 1e-6) and np.all(x_array <= exp_max + 1e-6):
                            apply_scaling = False
                    if apply_scaling:
                        x_array = mech_scaler.transform(x_array)
                except:
                    pass
        else:
            x_array = np.array(x_dict, dtype=float).reshape(1, -1)
            # Apply scaling only if input appears unscaled (avoid double-scaling)
            if mech_scaler is not None and hasattr(mech_scaler, "transform"):
                try:
                    apply_scaling = True
                    if hasattr(mech_scaler, "feature_range"):
                        exp_min, exp_max = mech_scaler.feature_range
                        if np.all(x_array >= exp_min - 1e-6) and np.all(x_array <= exp_max + 1e-6):
                            apply_scaling = False
                    if apply_scaling:
                        x_array = mech_scaler.transform(x_array)
                except:
                    pass
        
        proba = underlying_model.predict_proba(x_array)[0]
        
        # Use class mapping to get predicted class if available
        if class_order is not None and len(class_order) > 0:
            # proba shape should match class_order length
            if len(proba) == len(class_order):
                pred_idx = int(np.argmax(proba))
                pred_class = class_order[pred_idx] if pred_idx < len(class_order) else pred_idx
            else:
                # Mismatch: fallback to index
                pred_class = int(np.argmax(proba))
                logger.warning(f"[ProbaMap] Probability shape {proba.shape} doesn't match class_order length {len(class_order)}")
        else:
            # No class mapping: use index directly
            pred_class = int(np.argmax(proba))
        
        max_prob = float(np.max(proba))
        
        return pred_class, max_prob, proba
    except (AttributeError, ValueError, IndexError, KeyError, TypeError) as e:
        # Note: _increment_fallback is in maicl_lib_v2, but we'll handle fallbacks locally
        logger.warning(f"[ProbaFallback] {context or 'get_ml_prediction_probability'} -> {type(e).__name__}: {e}")
        return predicted_class, 0.5, None
    except Exception as e:
        logger.warning(f"[ProbaFallback] {context or 'get_ml_prediction_probability'} -> Unexpected {type(e).__name__}: {e}")
        return predicted_class, 0.5, None

