"""
TextGrad optimizer for textual mechanism refinement.

This module contains the TextGrad class and all prompts used for optimizing
textual mechanisms through LLM-based gradient updates.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import yaml
import os

try:
    from langchain.schema import HumanMessage
except Exception:
    from langchain_core.messages import HumanMessage

# Import dependencies from maicl_lib_v2
# These will be imported at runtime to avoid circular dependencies
# We'll use a lazy import pattern

logger = logging.getLogger(__name__)

# Import constants from maicl_config
try:
    from maicl_config import (
        MAX_TOP_FEATURES,
        MAX_WORST_INDICES,
        MAX_FEATURES_IN_EXAMPLE,
        MAX_TOP_ERROR_FEATURES_CHECK,
        MAX_MECHANISM_SUMMARY_LENGTH,
        MAX_FEATURE_IMPORTANCE_TOP,
        MAX_TOP_FEATURES_DISPLAY,
        TEXTGRAD_MIN_WEIGHT_THRESHOLD,
        TEXTGRAD_OVERESTIMATION_THRESHOLD,
        TEXTGRAD_FEATURE_ERROR_THRESHOLD,
        TEXTGRAD_FEATURE_ERROR_THRESHOLD_STRICT,
        TEXTGRAD_MAX_ITERATION_HISTORY
    )
except ImportError:
    # Fallback defaults if maicl_config is not available
    MAX_TOP_FEATURES = 20
    MAX_WORST_INDICES = 20
    MAX_FEATURES_IN_EXAMPLE = 5
    MAX_TOP_ERROR_FEATURES_CHECK = 5
    MAX_MECHANISM_SUMMARY_LENGTH = 200
    MAX_FEATURE_IMPORTANCE_TOP = 20
    MAX_TOP_FEATURES_DISPLAY = 20
    TEXTGRAD_MIN_WEIGHT_THRESHOLD = 0.3
    TEXTGRAD_OVERESTIMATION_THRESHOLD = 0.6
    TEXTGRAD_FEATURE_ERROR_THRESHOLD = 1.3
    TEXTGRAD_FEATURE_ERROR_THRESHOLD_STRICT = 1.4
    TEXTGRAD_MAX_ITERATION_HISTORY = 3


class TextGrad:
    """TextGrad optimizer for textual mechanism refinement"""
    def __init__(self, batched_llm, task_type: str = "regression", feature_cols: Optional[List[str]] = None, dataset_name: Optional[str] = None, class_names: Optional[List[str]] = None, config_path: Optional[str] = None, scale_min: Optional[float] = None, scale_max: Optional[float] = None, use_scaling: bool = True):
        self.llm = batched_llm
        self.task_type = task_type
        self.feature_cols = feature_cols
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.config = self._load_config(config_path)
        self.use_scaling = use_scaling  # Flag to indicate if scaling is enabled
        # Store scale range for output clipping (defaults to [0, 1] if not provided)
        # Always set scale_min and scale_max, even for classification (needed for template formatting)
        try:
            from maicl_config import SCALE_MIN, SCALE_MAX
            self.scale_min = scale_min if scale_min is not None else SCALE_MIN
            self.scale_max = scale_max if scale_max is not None else SCALE_MAX
        except ImportError:
            self.scale_min = scale_min if scale_min is not None else 0.0
            self.scale_max = scale_max if scale_max is not None else 1.0
        # Ensure they are never None (fallback to defaults)
        if self.scale_min is None:
            self.scale_min = 0.0
        if self.scale_max is None:
            self.scale_max = 1.0
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Default to textgrad_config.yaml in the same directory as this file
            config_path = os.path.join(os.path.dirname(__file__), "textgrad_config.yaml")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded TextGrad config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using empty config")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using empty config")
            return {}
    
    def _build_enhanced_error_feedback(self, X_train, y_train, ml_residuals, current_loss, dataset_name, 
                                       current_metrics: Optional[Dict[str, float]] = None,
                                       target_loss: Optional[float] = None,
                                       mechanism_performance: Optional[Dict[int, float]] = None,
                                       mechanism_idx: Optional[int] = None,
                                       X_train_original: Optional[List[Dict[str, Any]]] = None):
        """Build rich, diagnostic error feedback for TextGrad
        
        Args:
            current_metrics: Optional dict with 'r2', 'mae', 'mse' for regression or 'accuracy', 'f1' for classification
            target_loss: Optional target loss to achieve (for quantitative guidance)
            mechanism_performance: Optional dict mapping mechanism index to performance score
            mechanism_idx: Optional index of current mechanism (for routing-aware guidance)
        """
        # 1. Feature-level error correlation analysis
        feature_error_correlations = {}
        if ml_residuals is not None and len(ml_residuals) > 0:
            for i, feat_name in enumerate(self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]):
                try:
                    if i < X_train.shape[1]:
                        corr = np.corrcoef(X_train[:, i], np.abs(ml_residuals))[0, 1]
                        if np.isfinite(corr):
                            feature_error_correlations[feat_name] = float(corr)
                except:
                    pass
        
        # Sort by absolute correlation (most predictive of errors)
        top_error_features = sorted(
            feature_error_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:MAX_TOP_FEATURES]
        
        # 2. Error pattern detection
        error_patterns = self._detect_error_patterns(X_train, y_train, ml_residuals)
        
        # 3. Build comprehensive feedback with metrics
        metrics_str = ""
        if current_metrics is not None:
            metrics_templates = self.config.get('templates', {}).get('metrics', {})
            if self.task_type == "regression":
                r2 = current_metrics.get('r2', None)
                mae = current_metrics.get('mae', None)
                mse = current_metrics.get('mse', None)
                if r2 is not None:
                    metrics_str += metrics_templates.get('regression_r2', "R²: {r2:.4f} (target: maximize, higher is better)\n").format(r2=r2)
                if mae is not None:
                    metrics_str += metrics_templates.get('regression_mae', "MAE: {mae:.4f} (target: minimize, lower is better)\n").format(mae=mae)
                if mse is not None:
                    metrics_str += metrics_templates.get('regression_mse', "MSE: {mse:.4f} (target: minimize, lower is better)\n").format(mse=mse)
            else:
                acc = current_metrics.get('accuracy', None)
                f1 = current_metrics.get('f1', None)
                if acc is not None:
                    metrics_str += metrics_templates.get('classification_accuracy', "Accuracy: {acc:.4f} (target: maximize, higher is better)\n").format(acc=acc)
                if f1 is not None:
                    metrics_str += metrics_templates.get('classification_f1', "F1: {f1:.4f} (target: maximize, higher is better)\n").format(f1=f1)
        
        # 4. Add quantitative improvement targets
        improvement_target_str = ""
        if target_loss is not None and current_loss > target_loss:
            improvement_needed = (current_loss - target_loss) / current_loss
            performance_target_template = self.config.get('templates', {}).get('performance_target', {}).get('template', "")
            if performance_target_template:
                improvement_target_str = performance_target_template.format(
                    current_loss=current_loss,
                    target_loss=target_loss,
                    improvement_needed=improvement_needed
                )
            else:
                improvement_target_str = f"""
PERFORMANCE TARGET:
- Current loss: {current_loss:.4f}
- Target loss: {target_loss:.4f}
- Improvement needed: {improvement_needed:.1%} reduction
- Focus on achieving at least {target_loss:.4f} to meet target

"""
        
        # 5. Add routing-aware guidance if mechanism performance available
        routing_guidance_str = ""
        if mechanism_performance is not None and mechanism_idx is not None:
            my_perf = mechanism_performance.get(mechanism_idx, None)
            if my_perf is not None:
                all_perfs = [v for v in mechanism_performance.values() if v is not None]
                if len(all_perfs) > 1:
                    avg_perf = np.mean(all_perfs)
                    min_weight_threshold = TEXTGRAD_MIN_WEIGHT_THRESHOLD
                    routing_templates = self.config.get('templates', {}).get('routing_guidance', {})
                    if my_perf < min_weight_threshold:
                        routing_guidance_str = routing_templates.get('low_performance', "").format(
                            my_perf=my_perf,
                            min_weight_threshold=min_weight_threshold,
                            avg_perf=avg_perf
                        )
                    elif my_perf < avg_perf:
                        routing_guidance_str = routing_templates.get('below_average', "").format(
                            my_perf=my_perf,
                            avg_perf=avg_perf
                        )
        
        # 6. Adaptive complexity guidance (data-driven, not prescriptive)
        ml_model_type = ""
        if hasattr(self, 'mech_generator') and hasattr(self.mech_generator, 'ml_mechanism'):
            ml_mech = getattr(self.mech_generator, 'ml_mechanism', None)
            if ml_mech is not None:
                model_name = getattr(ml_mech, 'model_name', 'unknown')
                is_linear = 'linear' in model_name.lower() or 'logistic' in model_name.lower()
                
                # Check if errors show nonlinear patterns
                shows_nonlinearity = False
                if ml_residuals is not None and len(ml_residuals) > 10:
                    # Check for curvature in residuals vs feature values
                    try:
                        for feat_name, corr in top_error_features[:MAX_TOP_ERROR_FEATURES_CHECK]:  # Check top error-correlated features
                            feat_idx = self.feature_cols.index(feat_name) if self.feature_cols else None
                            if feat_idx is not None and feat_idx < X_train.shape[1]:
                                feat_vals = X_train[:, feat_idx]
                                # Check if errors increase nonlinearly with feature values
                                # Split into low/med/high and check if error increases faster than linear
                                q33, q67 = np.percentile(feat_vals, [33, 67])
                                low_mask = feat_vals < q33
                                mid_mask = (feat_vals >= q33) & (feat_vals <= q67)
                                high_mask = feat_vals > q67
                                
                                if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
                                    error_low = np.mean(np.abs(ml_residuals[low_mask]))
                                    error_high = np.mean(np.abs(ml_residuals[high_mask]))
                                    # If error increases faster than linear (more than 2x), suggest nonlinearity
                                    if error_high > 2.0 * error_low or error_low > 2.0 * error_high:
                                        shows_nonlinearity = True
                                        break
                    except:
                        pass
                
                complexity_templates = self.config.get('templates', {}).get('complexity_guidance', {})
                if is_linear and shows_nonlinearity:
                    ml_model_type = complexity_templates.get('linear_with_nonlinear_errors', "")
                elif is_linear and not shows_nonlinearity:
                    ml_model_type = complexity_templates.get('linear_with_linear_errors', "")
                elif not is_linear:
                    ml_model_type = complexity_templates.get('nonlinear_baseline', "")
        
        # Get feedback template from config
        feedback_template = self.config.get('templates', {}).get('error_feedback', {}).get('performance_analysis_header', 
            """PERFORMANCE ANALYSIS:
Current Loss: {current_loss:.4f}
{metrics_str}{improvement_target_str}{routing_guidance_str}{ml_model_type}

🎯 YOUR MISSION: The ML mechanism has systematic failures. Your mechanism must correct these errors.
The examples below show WHERE and WHY the ML model fails. Design your mechanism to handle these cases correctly.

CRITICAL ERROR PATTERNS (ML Model Failures):

{error_patterns}

FEATURES MOST CORRELATED WITH ML ERRORS (focus on these):

""")
        
        feedback = feedback_template.format(
            current_loss=current_loss,
            metrics_str=metrics_str,
            improvement_target_str=improvement_target_str,
            routing_guidance_str=routing_guidance_str,
            ml_model_type=ml_model_type,
            error_patterns=error_patterns
        )
        
        # For DeepChem datasets, don't show ECFP bit correlations - they're not interpretable
        # Instead, provide guidance about molecular properties
        is_deepchem_feedback = (self.feature_cols and len(self.feature_cols) > 0 and 
                               all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
        has_smiles_feedback = (is_deepchem_feedback and X_train_original is not None and 
                              len(X_train_original) > 0 and
                              isinstance(X_train_original[0], dict) and 'SMILES' in X_train_original[0])
        
        if is_deepchem_feedback and has_smiles_feedback:
            # For DeepChem: provide molecular property guidance instead of ECFP bit correlations
            deepchem_guidance_template = self.config.get('templates', {}).get('error_feedback', {}).get('deepchem_molecular_guidance', "")
            # Format with scale information if placeholders are present
            try:
                deepchem_guidance = deepchem_guidance_template.format(
                    scale_min=self.scale_min,
                    scale_max=self.scale_max
                )
            except (KeyError, ValueError):
                # If formatting fails, use template as-is and append scale info
                deepchem_guidance = deepchem_guidance_template
                if self.use_scaling:
                    deepchem_guidance += f"\n\nCRITICAL: All outputs must be clipped to [{self.scale_min}, {self.scale_max}] using clip(expression, {self.scale_min}, {self.scale_max})"
            feedback += deepchem_guidance
        else:
            # For other datasets: show feature correlations as usual
            for feat_name, corr in top_error_features:
                direction = "increases" if corr > 0 else "decreases"
                feedback += f"  • {feat_name}: {abs(corr):.3f} correlation (error {direction} with this feature)\n"
        
        # 4. Specific residual examples with feature names
        if ml_residuals is not None and len(ml_residuals) > 0:
            # Get worst predictions
            worst_idx = np.argsort(np.abs(ml_residuals))[::-1][:MAX_WORST_INDICES]
            
            worst_predictions_header = self.config.get('templates', {}).get('error_feedback', {}).get('worst_predictions_header', "")
            feedback += worst_predictions_header
            # Check if this is a DeepChem dataset and we have SMILES strings
            is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                          all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
            use_smiles = (is_deepchem and X_train_original is not None and len(X_train_original) > 0 and
                         isinstance(X_train_original[0], dict) and 'SMILES' in X_train_original[0])
            
            feature_names = self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]
            for idx in worst_idx:
                # Use SMILES strings for DeepChem datasets, otherwise use feature values
                if use_smiles and idx < len(X_train_original):
                    original_feat = X_train_original[idx]
                    if isinstance(original_feat, dict) and 'SMILES' in original_feat:
                        key_feats = [f"SMILES={original_feat['SMILES']}"]
                    else:
                        # Fallback to feature values
                        feat_vals = {feature_names[j]: float(X_train[idx, j]) 
                                    for j in range(min(len(feature_names), X_train.shape[1]))}
                        key_feats = [f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_EXAMPLE]]
                else:
                    feat_vals = {feature_names[j]: float(X_train[idx, j]) 
                                for j in range(min(len(feature_names), X_train.shape[1]))}
                    # Highlight key features
                    key_feats = [f"{k}={v:.2f}" for k, v in list(feat_vals.items())[:MAX_FEATURES_IN_EXAMPLE]]
                
                if self.task_type == "regression":
                    y_pred_val = ml_residuals[idx] + y_train[idx]
                    feedback += f"  • {', '.join(key_feats)} → pred={y_pred_val:.2f}, true={y_train[idx]:.2f}, error={ml_residuals[idx]:+.2f}\n"
                else:
                    # Classification: use class names if available
                    # For classification, ml_residuals is binary (1.0 = wrong, 0.0 = correct)
                    # We need to infer the predicted class from the residual
                    true_idx = int(y_train[idx])
                    is_correct = abs(ml_residuals[idx]) < 0.5
                    
                    # Try to get class names from the mechanism or context
                    class_names = getattr(self, 'class_names', None)
                    
                    if is_correct:
                        pred_idx = true_idx
                        match_str = '✓ CORRECT'
                    else:
                        # Wrong prediction - try to infer what it was predicted as
                        # This is approximate since we only have residual magnitude
                        # For now, show it was wrong without specific prediction
                        pred_idx = -1  # Unknown
                        match_str = '✗ WRONG (misclassified)'
                    
                    if class_names and len(class_names) > 0:
                        true_name = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
                        if pred_idx >= 0 and pred_idx < len(class_names):
                            pred_name = class_names[pred_idx]
                            feedback += f"  • {', '.join(key_feats)} → pred={pred_name}, true={true_name}, {match_str}\n"
                        else:
                            feedback += f"  • {', '.join(key_feats)} → true={true_name}, {match_str}\n"
                    else:
                        if pred_idx >= 0:
                            feedback += f"  • {', '.join(key_feats)} → pred={pred_idx}, true={true_idx}, {match_str}\n"
                        else:
                            feedback += f"  • {', '.join(key_feats)} → true={true_idx}, {match_str}\n"
        
        return feedback
    
    def _detect_error_patterns(self, X_train, y_train, ml_residuals):
        """Detect systematic error patterns with higher sensitivity"""
        patterns = []
        
        if ml_residuals is None or len(ml_residuals) < 5:
            return "Insufficient data for pattern detection"
        
        try:
            # Pattern 1: Overestimation vs underestimation (LOWER threshold)
            overestimate = np.sum(ml_residuals < 0)
            underestimate = np.sum(ml_residuals > 0)
            total = len(ml_residuals)
            
            # Use configurable threshold for more sensitivity
            if overestimate > TEXTGRAD_OVERESTIMATION_THRESHOLD * total:
                bias = np.mean(ml_residuals)
                patterns.append(f"⚠️ SYSTEMATIC OVERESTIMATION: {overestimate}/{total} ({100*overestimate/total:.0f}%) predictions too high (bias={bias:.3f})")
            elif underestimate > TEXTGRAD_OVERESTIMATION_THRESHOLD * total:
                bias = np.mean(ml_residuals)
                patterns.append(f"⚠️ SYSTEMATIC UNDERESTIMATION: {underestimate}/{total} ({100*underestimate/total:.0f}%) predictions too low (bias={bias:.3f})")
            
            # Pattern 2: Per-class error analysis for classification (ENHANCED with actionable guidance)
            if self.task_type == "classification":
                try:
                    y_true_int = y_train.astype(int)
                    
                    # Per-class error rate analysis with confusion matrix
                    # For classification, ml_residuals is binary: 1.0 = wrong, 0.0 = correct
                    if self.class_names and len(self.class_names) > 0:
                        # Build confusion matrix to identify class confusion pairs
                        confusion_pairs = {}  # (true_class, pred_class) -> count
                        class_error_rates = {}
                        class_discriminative_features = {}
                        
                        for true_class_idx in range(len(self.class_names)):
                            true_mask = (y_true_int == true_class_idx)
                            if np.sum(true_mask) > 0:
                                # Compute error rate (how many were misclassified)
                                errors_for_class = ml_residuals[true_mask]
                                error_rate = np.mean(errors_for_class > 0.5)  # Residual > 0.5 means wrong
                                class_error_rates[true_class_idx] = error_rate
                                
                                # Find which features best discriminate this class
                                if error_rate > 0.1 and X_train.shape[1] > 0:
                                    # Compute feature means for this class vs others
                                    this_class_features = X_train[true_mask]
                                    other_class_features = X_train[~true_mask]
                                    
                                    if len(other_class_features) > 0:
                                        feature_diffs = {}
                                        for feat_idx in range(min(5, X_train.shape[1])):  # Check top 5 features
                                            try:
                                                this_mean = np.mean(this_class_features[:, feat_idx])
                                                other_mean = np.mean(other_class_features[:, feat_idx])
                                                diff = abs(this_mean - other_mean)
                                                feat_name = self.feature_cols[feat_idx] if self.feature_cols else f"x{feat_idx}"
                                                feature_diffs[feat_name] = diff
                                            except:
                                                pass
                                        
                                        # Get top 2 most discriminative features
                                        if feature_diffs:
                                            top_features = sorted(feature_diffs.items(), key=lambda x: x[1], reverse=True)[:MAX_TOP_ERROR_FEATURES_CHECK]
                                            class_discriminative_features[true_class_idx] = top_features
                                
                                if error_rate > 0.1:  # Only report if >10% error rate
                                    true_name = self.class_names[true_class_idx]
                                    n_samples = np.sum(true_mask)
                                    n_errors = np.sum(errors_for_class > 0.5)
                                    
                                    # Build actionable guidance
                                    guidance_parts = []
                                    
                                    # Add discriminative features
                                    if true_class_idx in class_discriminative_features:
                                        top_feats = class_discriminative_features[true_class_idx]
                                        feat_names = [f[0] for f in top_feats]
                                        guidance_parts.append(f"Discriminative features: {', '.join(feat_names)}")
                                    
                                    # Check if this class is confused with others
                                    # (This would require prediction info, which we don't have in residuals alone)
                                    # For now, just report the error rate with guidance
                                    
                                    guidance_str = f" - {', '.join(guidance_parts)}" if guidance_parts else ""
                                    patterns.append(f"⚠️ {true_name.upper()} CLASS: {error_rate:.1%} error rate ({n_errors}/{n_samples} samples misclassified){guidance_str}")
                                    
                                    # Add specific fix suggestion if we have discriminative features
                                    if true_class_idx in class_discriminative_features and len(class_discriminative_features[true_class_idx]) > 0:
                                        top_feat_name, top_feat_diff = class_discriminative_features[true_class_idx][0]
                                        # Compute threshold suggestion
                                        this_class_vals = X_train[true_mask, self.feature_cols.index(top_feat_name) if self.feature_cols else 0]
                                        threshold_suggestion = np.median(this_class_vals)
                                        patterns.append(f"   → SUGGESTION: Add decision boundary using {top_feat_name} (median for {true_name}: {threshold_suggestion:.2f})")
                                    
                        # Find most confused class pairs (if we had prediction info, we'd use it here)
                        # For now, identify classes with highest error rates that might be confused
                        if len(class_error_rates) > 1:
                            sorted_classes = sorted(class_error_rates.items(), key=lambda x: x[1], reverse=True)
                            if len(sorted_classes) >= 2:
                                worst_class_idx, worst_error = sorted_classes[0]
                                second_worst_idx, second_error = sorted_classes[1]
                                if worst_error > 0.2 and second_error > 0.15:  # Both have significant errors
                                    worst_name = self.class_names[worst_class_idx] if worst_class_idx < len(self.class_names) else str(worst_class_idx)
                                    second_name = self.class_names[second_worst_idx] if second_worst_idx < len(self.class_names) else str(second_worst_idx)
                                    patterns.append(f"⚠️ CLASS CONFUSION: {worst_name} and {second_name} have high error rates ({worst_error:.1%} and {second_error:.1%}) - may be confused with each other")
                except Exception as e:
                    logger.debug(f"Per-class error analysis failed: {e}")
            
            # Pattern 2: Error distribution by target value (MORE DETAILED)
            if self.task_type == "regression":
                # Quartile-based analysis
                q25 = np.percentile(y_train, 25)
                q75 = np.percentile(y_train, 75)
                
                low_y = y_train < q25
                mid_y = (y_train >= q25) & (y_train <= q75)
                high_y = y_train > q75
                
                if np.sum(low_y) > 0 and np.sum(high_y) > 0:
                    error_low = np.mean(np.abs(ml_residuals[low_y]))
                    error_mid = np.mean(np.abs(ml_residuals[mid_y])) if np.sum(mid_y) > 0 else 0
                    error_high = np.mean(np.abs(ml_residuals[high_y]))
                    
                    # Use appropriate metric label based on task type
                    if self.task_type == "classification":
                        metric_label = "error_rate"
                    else:
                        metric_label = "MAE"
                    
                    # Use configurable threshold
                    if error_low > TEXTGRAD_FEATURE_ERROR_THRESHOLD * error_high:
                        patterns.append(f"⚠️ POOR LOW-VALUE PREDICTIONS: {metric_label.upper()}={error_low:.3f} for bottom 25% vs {error_high:.3f} for top 25%")
                    elif error_high > TEXTGRAD_FEATURE_ERROR_THRESHOLD * error_low:
                        patterns.append(f"⚠️ POOR HIGH-VALUE PREDICTIONS: {metric_label.upper()}={error_high:.3f} for top 25% vs {error_low:.3f} for bottom 25%")
                    
                    # Add mid-range analysis
                    if error_mid > 0 and error_mid > TEXTGRAD_FEATURE_ERROR_THRESHOLD * min(error_low, error_high):
                        patterns.append(f"⚠️ POOR MID-RANGE PREDICTIONS: {metric_label.upper()}={error_mid:.3f} for middle 50%")
            
            # Pattern 3: Feature-specific errors (LOWER threshold + more features)
            feature_names = self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]
            for feat_idx in range(min(8, len(feature_names), X_train.shape[1])):  # Check up to 8 features
                feat_name = feature_names[feat_idx]
                try:
                    feat_vals = X_train[:, feat_idx]
                    
                    q25_feat = np.percentile(feat_vals, 25)
                    q75_feat = np.percentile(feat_vals, 75)
                    
                    low_feat = feat_vals < q25_feat
                    high_feat = feat_vals > q75_feat
                    
                    if np.sum(low_feat) > 0 and np.sum(high_feat) > 0:
                        error_low_feat = np.mean(np.abs(ml_residuals[low_feat]))
                        error_high_feat = np.mean(np.abs(ml_residuals[high_feat]))
                        
                        # Use appropriate metric label based on task type
                        if self.task_type == "classification":
                            metric_label = "error_rate"
                        else:
                            metric_label = "MAE"
                        
                        # Use configurable threshold
                        if error_high_feat > TEXTGRAD_FEATURE_ERROR_THRESHOLD_STRICT * error_low_feat:
                            patterns.append(f"⚠️ ERRORS INCREASE WHEN {feat_name} IS HIGH: {metric_label.upper()}={error_high_feat:.3f} (high) vs {error_low_feat:.3f} (low) - ratio {error_high_feat/error_low_feat:.2f}x")
                        elif error_low_feat > TEXTGRAD_FEATURE_ERROR_THRESHOLD_STRICT * error_high_feat:
                            patterns.append(f"⚠️ ERRORS INCREASE WHEN {feat_name} IS LOW: {metric_label.upper()}={error_low_feat:.3f} (low) vs {error_high_feat:.3f} (high) - ratio {error_low_feat/error_high_feat:.2f}x")
                except:
                    pass
            
            # Pattern 4: Feature interaction effects (NEW)
            if len(feature_names) >= 2:
                try:
                    # Check for interaction between top 2 error-correlated features
                    feat1_idx = 0
                    feat2_idx = 1
                    
                    feat1_vals = X_train[:, feat1_idx]
                    feat2_vals = X_train[:, feat2_idx]
                    
                    # Quadrant analysis
                    feat1_high = feat1_vals > np.median(feat1_vals)
                    feat2_high = feat2_vals > np.median(feat2_vals)
                    
                    quadrants = [
                        (feat1_high & feat2_high, f"both {feature_names[feat1_idx]} and {feature_names[feat2_idx]} high"),
                        (~feat1_high & feat2_high, f"{feature_names[feat1_idx]} low but {feature_names[feat2_idx]} high"),
                        (feat1_high & ~feat2_high, f"{feature_names[feat1_idx]} high but {feature_names[feat2_idx]} low"),
                        (~feat1_high & ~feat2_high, f"both {feature_names[feat1_idx]} and {feature_names[feat2_idx]} low")
                    ]
                    
                    quad_errors = []
                    for mask, desc in quadrants:
                        if np.sum(mask) > 0:
                            error_quad = np.mean(np.abs(ml_residuals[mask]))
                            quad_errors.append((error_quad, desc, np.sum(mask)))
                    
                    if len(quad_errors) >= 4:
                        quad_errors.sort(reverse=True)  # Highest error first
                        worst_error, worst_desc, worst_n = quad_errors[0]
                        best_error, best_desc, best_n = quad_errors[-1]
                        
                        # Use appropriate metric label based on task type
                        if self.task_type == "classification":
                            metric_label = "error_rate"
                        else:
                            metric_label = "MAE"
                        
                        if worst_error > 1.5 * best_error:
                            patterns.append(f"⚠️ INTERACTION EFFECT: Errors worst when {worst_desc} ({metric_label.upper()}={worst_error:.3f}, n={worst_n}) vs when {best_desc} ({metric_label.upper()}={best_error:.3f}, n={best_n})")
                except Exception:
                    pass
        
        except Exception as e:
            logger.debug(f"Error pattern detection failed: {e}")
            return "Error pattern detection failed"
        
        if not patterns:
            # If no patterns detected, add a helpful default
            mae_overall = np.mean(np.abs(ml_residuals))
            std_overall = np.std(np.abs(ml_residuals))
            patterns.append(f"No strong systematic patterns (MAE={mae_overall:.3f}±{std_overall:.3f}), but individual errors range from {np.min(np.abs(ml_residuals)):.3f} to {np.max(np.abs(ml_residuals)):.3f}")
        
        return "\n".join(patterns)
    
    def _build_iteration_history_context(self, iteration_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build context from previous iterations to help TextGrad learn from past changes
        
        Args:
            iteration_history: List of dicts, each containing:
                - iteration: int
                - mechanism_before: str (mechanism text before optimization)
                - mechanism_after: str (mechanism text after optimization)
                - metrics_before: Dict with loss, r2, mae, etc.
                - metrics_after: Dict with loss, r2, mae, etc.
                - accepted: bool (whether update was accepted)
                - reason: str (acceptance/rejection reason)
        """
        if not iteration_history or len(iteration_history) == 0:
            return ""
        
        # Get templates from config
        iteration_templates = self.config.get('templates', {}).get('iteration_history', {})
        header = iteration_templates.get('header', "")
        context_parts = []
        if header:
            context_parts.append(header)
        else:
            context_parts.append("\n" + "="*80)
            context_parts.append("📚 ITERATION HISTORY - LEARN FROM PREVIOUS CHANGES")
            context_parts.append("="*80)
            context_parts.append("\nThe following shows what changed in previous iterations and what worked/didn't work.")
            context_parts.append("Use this to avoid repeating mistakes and build on successful changes.\n")
        
        # Show last N iterations (most recent first) - configurable
        recent_history = iteration_history[-TEXTGRAD_MAX_ITERATION_HISTORY:] if len(iteration_history) > TEXTGRAD_MAX_ITERATION_HISTORY else iteration_history
        
        for hist in reversed(recent_history):  # Show most recent first
            iter_num = hist.get('iteration', '?')
            accepted = hist.get('accepted', False)
            reason = hist.get('reason', '')
            
            iter_separator = iteration_templates.get('iteration_separator', f"\n--- Iteration {iter_num} ---")
            context_parts.append(iter_separator.format(iter_num=iter_num))
            
            if accepted:
                accepted_label = iteration_templates.get('accepted_label', f"✓ ACCEPTED: {reason}")
                context_parts.append(accepted_label.format(reason=reason))
                
                # Show performance change
                metrics_before = hist.get('metrics_before', {})
                metrics_after = hist.get('metrics_after', {})
                
                if self.task_type == "regression":
                    loss_before = metrics_before.get('loss', None)
                    loss_after = metrics_after.get('loss', None)
                    r2_before = metrics_before.get('r2', None)
                    r2_after = metrics_after.get('r2', None)
                    mae_before = metrics_before.get('mae', None)
                    mae_after = metrics_after.get('mae', None)
                    
                    perf_changes = []
                    if loss_before is not None and loss_after is not None:
                        loss_change = loss_before - loss_after
                        perf_changes.append(f"Loss: {loss_before:.4f} → {loss_after:.4f} ({loss_change:+.4f})")
                    if r2_before is not None and r2_after is not None:
                        r2_change = r2_after - r2_before
                        perf_changes.append(f"R²: {r2_before:.4f} → {r2_after:.4f} ({r2_change:+.4f})")
                    if mae_before is not None and mae_after is not None:
                        mae_change = mae_before - mae_after
                        perf_changes.append(f"MAE: {mae_before:.4f} → {mae_after:.4f} ({mae_change:+.4f})")
                    
                    if perf_changes:
                        context_parts.append("  Performance: " + ", ".join(perf_changes))
                else:
                    acc_before = metrics_before.get('accuracy', None)
                    acc_after = metrics_after.get('accuracy', None)
                    f1_before = metrics_before.get('f1', None)
                    f1_after = metrics_after.get('f1', None)
                    
                    perf_changes = []
                    if acc_before is not None and acc_after is not None:
                        acc_change = acc_after - acc_before
                        perf_changes.append(f"Accuracy: {acc_before:.4f} → {acc_after:.4f} ({acc_change:+.4f})")
                    if f1_before is not None and f1_after is not None:
                        f1_change = f1_after - f1_before
                        perf_changes.append(f"F1: {f1_before:.4f} → {f1_after:.4f} ({f1_change:+.4f})")
                    
                    if perf_changes:
                        perf_label = iteration_templates.get('performance_label', "  Performance: {perf_changes}")
                        context_parts.append(perf_label.format(perf_changes=", ".join(perf_changes)))
                
                # Show what changed in the mechanism
                mech_before = hist.get('mechanism_before', '')
                mech_after = hist.get('mechanism_after', '')
                
                if mech_before and mech_after and mech_before != mech_after:
                    # Show a summary of the change (first 200 chars of each)
                    before_summary = mech_before[:MAX_MECHANISM_SUMMARY_LENGTH] + "..." if len(mech_before) > MAX_MECHANISM_SUMMARY_LENGTH else mech_before
                    after_summary = mech_after[:MAX_MECHANISM_SUMMARY_LENGTH] + "..." if len(mech_after) > MAX_MECHANISM_SUMMARY_LENGTH else mech_after
                    mech_changed_label = iteration_templates.get('mechanism_changed_label', "")
                    if mech_changed_label:
                        context_parts.append(mech_changed_label.format(
                            before_summary=before_summary,
                            after_summary=after_summary
                        ))
                    else:
                        context_parts.append("\n  Mechanism changed:")
                        context_parts.append(f"    Before: {before_summary}")
                        context_parts.append(f"    After:  {after_summary}")
                        context_parts.append("  ✓ This change improved performance - consider similar improvements")
            else:
                rejected_label = iteration_templates.get('rejected_label', f"✗ REJECTED: {reason}")
                context_parts.append(rejected_label.format(reason=reason))
                
                # Show why it was rejected
                metrics_before = hist.get('metrics_before', {})
                metrics_after = hist.get('metrics_after', {})
                
                if self.task_type == "regression":
                    loss_before = metrics_before.get('loss', None)
                    loss_after = metrics_after.get('loss', None)
                    if loss_before is not None and loss_after is not None:
                        loss_change = loss_before - loss_after
                        context_parts.append(f"  Loss: {loss_before:.4f} → {loss_after:.4f} ({loss_change:+.4f})")
                else:
                    acc_before = metrics_before.get('accuracy', None)
                    acc_after = metrics_after.get('accuracy', None)
                    if acc_before is not None and acc_after is not None:
                        acc_change = acc_after - acc_before
                        context_parts.append(f"  Accuracy: {acc_before:.4f} → {acc_after:.4f} ({acc_change:+.4f})")
                
                rejected_warning = iteration_templates.get('rejected_warning', "  ⚠️ This change did NOT improve performance - avoid similar changes")
                context_parts.append(rejected_warning)
        
        # Analyze patterns
        accepted_count = sum(1 for h in recent_history if h.get('accepted', False))
        rejected_count = len(recent_history) - accepted_count
        
        insights = []
        if accepted_count > 0:
            insights.append(f"  • {accepted_count} recent change(s) were ACCEPTED - these worked")
        if rejected_count > 0:
            insights.append(f"  • {rejected_count} recent change(s) were REJECTED - avoid similar approaches")
        
        # Find common patterns in accepted changes
        accepted_changes = [h for h in recent_history if h.get('accepted', False)]
        if len(accepted_changes) >= 2:
            insights.append("  • Multiple successful changes - build on these patterns")
        
        footer = iteration_templates.get('footer', "")
        if footer:
            context_parts.append(footer.format(insights="\n".join(insights)))
        else:
            context_parts.append("\n" + "="*80)
            context_parts.append("KEY INSIGHTS:")
            context_parts.extend(insights)
            context_parts.append("\nYOUR TASK: Learn from this history. Build on successful changes and avoid repeating rejected approaches.")
            context_parts.append("="*80 + "\n")
        
        return "\n".join(context_parts)
    
    def _get_dataset_optimization_context(self) -> str:
        """Get dataset-specific optimization context"""
        dataset_lower = self.dataset_name.lower() if self.dataset_name else ""
        features_str = ', '.join(self.feature_cols) if self.feature_cols else 'input features'
        
        # Get scale range information for output
        if self.use_scaling:
            scale_range_str = f"[{self.scale_min}, {self.scale_max}]"
            scale_info = f"All data (inputs and outputs) are scaled to {scale_range_str} range."
        else:
            scale_range_str = "raw/unscaled"
            scale_info = "Data is NOT normalized; use raw/unscaled feature and target values."
        
        # Get dataset contexts from config
        dataset_contexts = self.config.get('templates', {}).get('dataset_contexts', {})
        
        # Match dataset name
        for key, template in dataset_contexts.items():
            if key != 'generic' and key in dataset_lower:
                # Format template with features and scale info
                if key == 'enzyme':
                    features_str = ', '.join(self.feature_cols) if self.feature_cols else 'enzyme kinetic parameters'
                elif key == 'protein':
                    features_str = ', '.join(self.feature_cols) if self.feature_cols else 'expression system parameters'
                elif key == 'concrete':
                    features_str = ', '.join(self.feature_cols) if self.feature_cols else 'concrete composition parameters'
                # Try to format with scale info, fallback if not in template
                try:
                    return template.format(features=features_str, scale_min=self.scale_min, scale_max=self.scale_max, scale_range=scale_range_str, scale_info=scale_info)
                except (KeyError, ValueError) as e:
                    # If template doesn't have all placeholders, try with just features first
                    try:
                        formatted = template.format(features=features_str)
                        # Check if there are still unformatted placeholders
                        if '{scale_min}' in formatted or '{scale_max}' in formatted:
                            # Replace any remaining scale placeholders
                            formatted = formatted.replace('{scale_min}', str(self.scale_min))
                            formatted = formatted.replace('{scale_max}', str(self.scale_max))
                            formatted = formatted.replace('{scale_range}', scale_range_str)
                            formatted = formatted.replace('{scale_info}', scale_info)
                        return formatted + f"\n\nCRITICAL SCALING INFORMATION:\n{scale_info}\nOutput must be clipped to {scale_range_str}."
                    except (KeyError, ValueError):
                        # Last resort: just append scale info
                        return template + f"\n\nCRITICAL SCALING INFORMATION:\n{scale_info}\nOutput must be clipped to {scale_range_str}."
        
        # Generic fallback
        generic_template = dataset_contexts.get('generic', 
            """DATASET: {dataset_name}
FEATURES: {features}
OPTIMIZATION GUIDANCE:
- Use actual feature names (not generic x1, x2, etc.)
- Model nonlinear effects (saturation, thresholds, synergies)
- Include domain-relevant interactions
- {scale_info}
- Ensure output stays in valid range {scale_range}
""")
        return generic_template.format(
            dataset_name=self.dataset_name, 
            features=features_str,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            scale_range=scale_range_str,
            scale_info=scale_info
        )
    
    def optimize_batch_single_call(self, mechanisms: List[str], error_feedbacks: List[str],
                                   ml_residuals: Optional[np.ndarray] = None,
                                   X_train: Optional[np.ndarray] = None,
                                   y_train: Optional[np.ndarray] = None,
                                   ml_mechanism: Optional[Any] = None,
                                   mechanism_performance: Optional[Dict[int, float]] = None,
                                   mechanism_metrics: Optional[Dict[int, Dict[str, float]]] = None,
                                   iteration_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Enhanced single-call optimization with diagnostic prompting and ML failure analysis"""
        # Lazy import to avoid circular dependency
        from maicl_lib_v2 import extract_ml_knowledge, analyze_ml_failure_patterns, enhanced_textgrad_feedback
        
        if len(mechanisms) != len(error_feedbacks):
            raise ValueError("mechanisms and error_feedbacks must have same length")
        if len(mechanisms) == 0:
            return []
        
        # Enhance feedback with ML failure analysis if available
        enhanced_feedbacks = error_feedbacks
        if (X_train is not None and y_train is not None and 
            ml_mechanism is not None and mechanism_performance is not None):
            try:
                # Extract ML knowledge
                ml_knowledge = extract_ml_knowledge(ml_mechanism, self.feature_cols, self.task_type)
                
                # Analyze ML failure patterns
                ml_failure_analysis = analyze_ml_failure_patterns(
                    ml_mechanism, X_train, y_train, self.feature_cols, 
                    self.task_type, getattr(self, "class_names", None)
                )
                
                # Add residual distribution summary if ml_residuals provided and matches length
                residual_summary = ""
                if (ml_residuals is not None and len(ml_residuals) == len(X_train) and len(ml_residuals) > 0):
                    try:
                        abs_residuals = np.abs(ml_residuals)
                        residual_summary = f"\n\n📊 ML RESIDUAL DISTRIBUTION (on training samples):\n"
                        residual_summary += f"  - Mean absolute residual: {np.mean(abs_residuals):.4f}\n"
                        residual_summary += f"  - Median absolute residual: {np.median(abs_residuals):.4f}\n"
                        residual_summary += f"  - Max absolute residual: {np.max(abs_residuals):.4f}\n"
                        residual_summary += f"  - Min absolute residual: {np.min(abs_residuals):.4f}\n"
                        if len(abs_residuals) > 1:
                            residual_summary += f"  - Std absolute residual: {np.std(abs_residuals):.4f}\n"
                        # Add quantiles for better understanding
                        if len(abs_residuals) >= 10:
                            q25, q75 = np.percentile(abs_residuals, [25, 75])
                            residual_summary += f"  - 25th percentile: {q25:.4f}, 75th percentile: {q75:.4f}\n"
                        residual_summary += f"\nFocus on samples with high residuals when refining the mechanism."
                    except Exception:
                        pass  # Silently skip if residual summary fails
                
                # Build enhanced feedback for each mechanism
                enhanced_feedbacks = []
                for mech_idx, (mechanism, error_feedback) in enumerate(zip(mechanisms, error_feedbacks)):
                    # Extract current loss from error_feedback if possible
                    match = re.search(r'Current Loss: ([\d.]+)', error_feedback)
                    current_loss = float(match.group(1)) if match else 0.0
                    
                    # Get enhanced feedback with full metrics
                    enhanced_fb = enhanced_textgrad_feedback(
                        ml_failure_analysis, current_loss, mechanism_performance,
                        mech_idx, ml_knowledge, mechanism_metrics
                    )
                    
                    # Append residual summary if available
                    if residual_summary:
                        enhanced_fb = enhanced_fb + residual_summary
                    
                    # IMPROVED: Add routing-aware guidance
                    # If this mechanism has low performance score, emphasize that it needs to improve significantly
                    # to contribute meaningfully to the ensemble
                    routing_guidance = ""
                    if mechanism_performance is not None and mech_idx in mechanism_performance:
                        mech_perf = mechanism_performance[mech_idx]
                        routing_templates = self.config.get('templates', {}).get('routing_guidance', {})
                        if mech_perf < 0.5:
                            routing_guidance = routing_templates.get('routing_awareness_low', "").format(mech_perf=mech_perf)
                        elif mech_perf < 0.7:
                            routing_guidance = routing_templates.get('routing_awareness_moderate', "").format(mech_perf=mech_perf)
                        else:
                            routing_guidance = routing_templates.get('routing_awareness_good', "").format(mech_perf=mech_perf)
                    
                    # Combine with original error feedback
                    enhanced_feedbacks.append(error_feedback + "\n\n" + enhanced_fb + routing_guidance)
                
                logger.info("  [TextGrad] Using enhanced feedback with ML failure analysis")
            except Exception as e:
                logger.warning(f"  [TextGrad] Failed to enhance feedback: {e}, using original")
                enhanced_feedbacks = error_feedbacks
        
        prompts = []
        for mechanism, error_feedback in zip(mechanisms, enhanced_feedbacks):
            # Extract dataset-specific context
            dataset_context = self._get_dataset_optimization_context()
            
            # Build iteration history context (memory from previous iterations)
            iteration_history_context = self._build_iteration_history_context(iteration_history)
            
            # Build feature instruction with actual names
            # For DeepChem datasets, don't show ECFP features - use SMILES instead
            feature_instruction = ""
            is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                          all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
            
            feature_templates = self.config.get('templates', {}).get('feature_instruction', {})
            
            if is_deepchem:
                feature_instruction = feature_templates.get('deepchem', "")
            elif self.feature_cols and len(self.feature_cols) > 0:
                feature_list = self.feature_cols[:MAX_FEATURE_IMPORTANCE_TOP]
                feature_list_str = ', '.join(feature_list)
                feature_example = f"{feature_list[0]} + {feature_list[1]}" if len(feature_list) >= 2 else feature_list[0]
                feature_instruction = feature_templates.get('standard', "").format(
                    feature_list=feature_list_str,
                    feature_example=feature_example
                )
            
            # Add class distribution context for classification
            class_dist_context = ""
            if self.task_type == "classification" and X_train is not None and y_train is not None:
                try:
                    class_dist = np.bincount(y_train.astype(int))
                    if self.class_names and len(self.class_names) > 0:
                        class_dist_templates = self.config.get('templates', {}).get('class_distribution', {})
                        class_dist_header = class_dist_templates.get('header', "\n\nCLASS DISTRIBUTION IN TRAINING DATA:\n")
                        class_dist_footer = class_dist_templates.get('footer', "\n\nIMPORTANT: Your mechanism must correctly predict ALL classes, not just the majority classes. Pay special attention to classes with fewer samples.")
                        class_line_template = class_dist_templates.get('class_line', "  - {class_name}: {count} samples ({percentage:.1f}%)")
                        
                        class_dist_lines = []
                        for i in range(len(self.class_names)):
                            if i < len(class_dist):
                                class_dist_lines.append(class_line_template.format(
                                    class_name=self.class_names[i],
                                    count=class_dist[i],
                                    percentage=100*class_dist[i]/len(y_train)
                                ))
                        if class_dist_lines:
                            class_dist_context = class_dist_header + "\n".join(class_dist_lines) + class_dist_footer
                except Exception:
                    pass
            
            # Build optimization prompt
            main_template = self.config.get('prompts', {}).get('optimization', {}).get('main_template', "")
            formula_instructions = self.config.get('formula_instructions', {})
            if self.task_type == "classification":
                formula_instruction = formula_instructions.get(
                    'classification',
                    "- For classification: Formula MUST output a CLASS INDEX (integer: 0, 1, 2, ...), NOT a probability or continuous value. Use thresholds, argmax, or conditional logic as needed."
                )
            else:
                # For regression, use actual scale range instead of hardcoded [0, 10]
                if self.use_scaling:
                    default_regression = f"- For regression: Clipped to [{self.scale_min}, {self.scale_max}]"
                else:
                    default_regression = "- For regression: Output numeric values (no clipping, use raw/unscaled values)"
                formula_instruction = formula_instructions.get('regression', default_regression)
                # If the config has a template with {scale_min} and {scale_max}, format it
                # This handles both old configs (hardcoded [0, 10]) and new configs (with placeholders)
                try:
                    if '{scale_min}' in formula_instruction or '{scale_max}' in formula_instruction:
                        formula_instruction = formula_instruction.format(scale_min=self.scale_min, scale_max=self.scale_max)
                except (KeyError, ValueError):
                    # If formatting fails (e.g., old config without placeholders), use default
                    formula_instruction = default_regression
            
            # Format main_template with all required placeholders
            # Handle both old templates (without scale placeholders) and new ones (with scale placeholders)
            # Ensure scale_min and scale_max are always set (not None)
            scale_min_val = self.scale_min if self.scale_min is not None else 0.0
            scale_max_val = self.scale_max if self.scale_max is not None else 1.0
            
            # Replace any unformatted scale placeholders in sub-strings to prevent KeyError
            # This handles cases where dataset_context or other strings have unformatted placeholders
            dataset_context = str(dataset_context).replace('{scale_min}', str(scale_min_val)).replace('{scale_max}', str(scale_max_val))
            feature_instruction = str(feature_instruction).replace('{scale_min}', str(scale_min_val)).replace('{scale_max}', str(scale_max_val))
            formula_instruction = str(formula_instruction).replace('{scale_min}', str(scale_min_val)).replace('{scale_max}', str(scale_max_val))
            
            format_kwargs = {
                'dataset_context': dataset_context,
                'iteration_history_context': iteration_history_context,
                'mechanism': mechanism,
                'error_feedback': error_feedback,
                'class_dist_context': class_dist_context,
                'feature_instruction': feature_instruction,
                'formula_instruction': formula_instruction,
                'scale_min': scale_min_val,
                'scale_max': scale_max_val
            }
            try:
                prompt = main_template.format(**format_kwargs)
            except KeyError as e:
                # If template has placeholders we don't recognize, log and re-raise
                # This helps identify missing placeholders in templates
                logger.warning(f"Template formatting error: {e}. Missing placeholder in main_template?")
                logger.warning(f"Available format_kwargs keys: {list(format_kwargs.keys())}")
                raise
            prompts.append(HumanMessage(content=prompt))
        
        gradients = self.llm.invoke_batch(prompts)
        return gradients
    
    def apply_gradients_batch(self, mechanisms: List[str], gradients: List[str]) -> List[str]:
        """Apply textual gradients to update mechanisms (batched)"""
        if len(mechanisms) != len(gradients):
            raise ValueError("mechanisms and gradients must have same length")
        if len(mechanisms) == 0:
            return []
        
        update_prompts = []
        update_instructions = self.config.get('prompts', {}).get('update_instructions', {})
        update_template = self.config.get('prompts', {}).get('update_mechanism', {}).get('template', "")
        
        for mechanism, gradient in zip(mechanisms, gradients):
            if self.task_type == "classification":
                update_instruction = update_instructions.get('classification', "")
            else:
                update_instruction = update_instructions.get('regression', "")
            
            prompt = update_template.format(
                mechanism=mechanism,
                gradient=gradient,
                update_instruction=update_instruction
            )
            update_prompts.append(HumanMessage(content=prompt))
        
        updated_mechanisms = self.llm.invoke_batch(update_prompts)
        return [upd if upd else orig for upd, orig in zip(updated_mechanisms, mechanisms)]
    
    def optimize_batch(self, mechanisms: List[str], error_feedbacks: List[str], 
                       ml_residuals: Optional[np.ndarray] = None,
                       X_train: Optional[np.ndarray] = None,
                       y_train: Optional[np.ndarray] = None,
                       ml_mechanism: Optional[Any] = None,
                       mechanism_performance: Optional[Dict[int, float]] = None,
                       mechanism_metrics: Optional[Dict[int, Dict[str, float]]] = None,
                       iteration_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Optimize mechanisms using textual gradients with optional ML-enhanced feedback and iteration history"""
        return self.optimize_batch_single_call(mechanisms, error_feedbacks, ml_residuals,
                                              X_train, y_train, ml_mechanism, mechanism_performance, mechanism_metrics, iteration_history)

