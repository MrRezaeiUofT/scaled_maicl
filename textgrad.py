"""
TextGrad optimizer for textual mechanism refinement.

Simplified version that focuses on descriptive prompts and learns from residual samples.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import yaml
import os

try:
    from langchain.schema import HumanMessage
except Exception:
    from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class TextGrad:
    """TextGrad optimizer for textual mechanism refinement"""
    
    def __init__(self, batched_llm, task_type: str = "regression", 
                 feature_cols: Optional[List[str]] = None, 
                 dataset_name: Optional[str] = None, 
                 class_names: Optional[List[str]] = None, 
                 config_path: Optional[str] = None, 
                 scale_min: Optional[float] = None, 
                 scale_max: Optional[float] = None, 
                 use_scaling: bool = True):
        self.llm = batched_llm
        self.task_type = task_type
        self.feature_cols = feature_cols
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.config = self._load_config(config_path)
        self.use_scaling = use_scaling
        if not self.use_scaling:
            self.scale_min = None
            self.scale_max = None
        else:
            try:
                from maicl_config import SCALE_MIN, SCALE_MAX
                self.scale_min = scale_min if scale_min is not None else SCALE_MIN
                self.scale_max = scale_max if scale_max is not None else SCALE_MAX
            except ImportError:
                self.scale_min = scale_min if scale_min is not None else 0.0
                self.scale_max = scale_max if scale_max is not None else 1.0
            if self.scale_min is None:
                self.scale_min = 0.0
            if self.scale_max is None:
                self.scale_max = 1.0

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
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
    
    def _build_enhanced_error_feedback(self, X_train: np.ndarray, y_train: np.ndarray, 
                                       ml_residuals: np.ndarray, current_loss: float,
                                       dataset_name: Optional[str] = None,
                                       current_metrics: Optional[Dict[str, float]] = None,
                                       target_loss: Optional[float] = None,
                                       mechanism_performance: Optional[Dict[int, float]] = None,
                                       mechanism_idx: Optional[int] = None,
                                       X_train_original: Optional[List[Dict[str, Any]]] = None,
                                       latent_z: Optional[str] = None) -> str:
        """Build error feedback for TextGrad (simplified version)"""
        feedback_parts = []
        
        # Performance summary
        feedback_parts.append(f"PERFORMANCE ANALYSIS:")
        
        if current_metrics:
            if self.task_type == "regression":
                r2 = current_metrics.get('r2', None)
                mae = current_metrics.get('mae', None)
                if r2 is not None:
                    feedback_parts.append(f"R²: {r2:.4f} (target: maximize, higher is better)")
                if mae is not None:
                    feedback_parts.append(f"MAE: {mae:.4f} (target: minimize, lower is better)")
            else:
                acc = current_metrics.get('accuracy', None)
                f1 = current_metrics.get('f1', None)
                if acc is not None:
                    feedback_parts.append(f"Accuracy: {acc:.4f} (target: maximize, higher is better)")
                if f1 is not None:
                    feedback_parts.append(f"F1: {f1:.4f} (target: maximize, higher is better)")
        
        feedback_parts.append("\n🎯 YOUR MISSION: Improve the mechanism based on the insights below.\n")
        
        # Add residual context - prefer latent_z if available (contains LLM's analysis of residual patterns)
        if latent_z:
            # Use the pre-computed latent_z that contains LLM's analysis of residual patterns
            logger.info("  [TextGrad] Using pre-computed latent_z (LLM's analysis of residual patterns) instead of raw samples")
            residual_context = self._build_residual_context_from_latent_z(latent_z, ml_residuals)
        else:
            # Fallback to raw residual samples if latent_z not available
            logger.debug("  [TextGrad] No latent_z available, using raw residual samples")
            residual_context = self._build_residual_context(X_train, y_train, ml_residuals, X_train_original)
        
        if residual_context:
            feedback_parts.append(residual_context)
        
        return "\n".join(feedback_parts)
    
    def _build_residual_context_from_latent_z(self, latent_z: str, ml_residuals: Optional[np.ndarray] = None) -> str:
        """Build residual context from pre-computed latent_z (LLM's analysis of residual patterns) - NO raw samples"""
        context_parts = []
        
        # Add the latent_z analysis (LLM's insights about residual patterns) - this is the key information
        context_parts.append("\nLATENT MECHANISM INSIGHTS:")
        context_parts.append("=" * 80)
        context_parts.append(latent_z)
        context_parts.append("=" * 80)
        
        return "\n".join(context_parts)
    
    def _build_residual_context(self, X_train: np.ndarray, y_train: np.ndarray, 
                                ml_residuals: np.ndarray, 
                                X_train_original: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build descriptive context from raw residual samples for optimization (fallback when latent_z not available)"""
        if ml_residuals is None or len(ml_residuals) == 0:
            return ""
        
        context_parts = []
        
        # Overall residual statistics
        abs_residuals = np.abs(ml_residuals)
        context_parts.append("RESIDUAL ANALYSIS:")
        context_parts.append(f"  Mean absolute error: {np.mean(abs_residuals):.4f}")
        context_parts.append(f"  Median absolute error: {np.median(abs_residuals):.4f}")
        context_parts.append(f"  Max absolute error: {np.max(abs_residuals):.4f}")
        context_parts.append(f"  Min absolute error: {np.min(abs_residuals):.4f}")
        if len(abs_residuals) > 1:
            context_parts.append(f"  Standard deviation: {np.std(abs_residuals):.4f}")
        
        # Direction of errors
        overestimate = np.sum(ml_residuals < 0)
        underestimate = np.sum(ml_residuals > 0)
        total = len(ml_residuals)
        context_parts.append(f"\nERROR DIRECTION:")
        context_parts.append(f"  Overestimations: {overestimate}/{total} ({100*overestimate/total:.1f}%)")
        context_parts.append(f"  Underestimations: {underestimate}/{total} ({100*underestimate/total:.1f}%)")
        if overestimate > 0 or underestimate > 0:
            mean_bias = np.mean(ml_residuals)
            context_parts.append(f"  Mean bias: {mean_bias:+.4f}")
        
        # Worst prediction examples
        if len(ml_residuals) > 0:
            worst_indices = np.argsort(abs_residuals)[::-1]
            context_parts.append(f"\nWORST PREDICTION EXAMPLES (where the model fails most):")
            
            feature_names = self.feature_cols if self.feature_cols else [f"x{j}" for j in range(X_train.shape[1])]
            use_original = (X_train_original is not None and len(X_train_original) > 0 and
                           isinstance(X_train_original[0], dict))
            
            # Log how many features are being included (for verification)
            num_features_included = len(feature_names) if feature_names else X_train.shape[1]
            logger.debug(f"  [TextGrad] Including {num_features_included} features in residual context examples")
            
            for idx in worst_indices:
                if use_original and idx < len(X_train_original):
                    original_feat = X_train_original[idx]
                    if isinstance(original_feat, dict):
                        # Show key features from original dict
                        num_feats = {k: v for k, v in original_feat.items() 
                                    if k not in ['SEQ', 'SUBSTRATES', 'SMILES'] and isinstance(v, (int, float))}
                        key_feats = [f"{k}={v:.2f}" for k, v in list(num_feats.items())]
                        if 'SMILES' in original_feat:
                            key_feats.insert(0, f"SMILES={original_feat['SMILES']}")
                else:
                    # Use feature values from X_train - include ALL features (no truncation)
                    feat_vals = {feature_names[j]: float(X_train[idx, j]) 
                                for j in range(min(len(feature_names), X_train.shape[1]))}
                    # Include ALL features in the prompt (no limit here - this goes to the LLM)
                    key_feats = [f"{k}={v:.2f}" for k, v in list(feat_vals.items())]
                
                if self.task_type == "regression":
                    y_pred_val = ml_residuals[idx] + y_train[idx]
                    context_parts.append(f"  Example: {', '.join(key_feats)} → predicted={y_pred_val:.2f}, actual={y_train[idx]:.2f}, error={ml_residuals[idx]:+.2f}")
                else:
                    # Classification
                    true_idx = int(y_train[idx])
                    is_correct = abs(ml_residuals[idx]) < 0.5
                    if self.class_names and len(self.class_names) > 0:
                        true_name = self.class_names[true_idx] if 0 <= true_idx < len(self.class_names) else str(true_idx)
                        status = "CORRECT" if is_correct else "WRONG"
                        context_parts.append(f"  Example: {', '.join(key_feats)} → actual={true_name}, {status}")
                    else:
                        status = "CORRECT" if is_correct else "WRONG"
                        context_parts.append(f"  Example: {', '.join(key_feats)} → actual={true_idx}, {status}")
        
        return "\n".join(context_parts)
    
    def _get_dataset_context(self) -> str:
        """Get dataset-specific optimization context"""
        dataset_lower = self.dataset_name.lower() if self.dataset_name else ""
        features_str = ', '.join(self.feature_cols) if self.feature_cols else 'input features'
        
        if self.use_scaling:
            scale_range_str = f"[{self.scale_min}, {self.scale_max}]"
            scale_info = f"""All data (inputs and outputs) are scaled to {scale_range_str} range.

CRITICAL: ALL constants in your formula must also be in the {scale_range_str} scaled range.
- If you see feature values like Temperature=0.93 or Potassium=0.60, these are SCALED values
- ALL constants in your formula (like optimal temperature, Km values, thresholds) must be SCALED accordingly
- Example: If optimal temperature is 34°C and range is 22-37°C, use 0.8 (scaled) NOT 34 (unscaled)
- Example: If Km for Mg is 8-12 mM and range is 2.5-12.5 mM, use 0.55-0.76 (scaled) NOT 8-12 (unscaled)
- Base your constants on the SCALED sample values you see in the residual examples, not on raw physical units"""
        else:
            scale_range_str = "raw (no clipping)"
            scale_info = "Data is NOT normalized; use raw feature and target values. Do NOT clip outputs to any fixed range."
        
        dataset_contexts = self.config.get('templates', {}).get('dataset_contexts', {})
        
        # Try task-specific key first
        if dataset_lower:
            task_suffix = self.task_type.lower() if self.task_type else ""
            if task_suffix:
                direct_key = f"{dataset_lower}_{task_suffix}"
                if direct_key in dataset_contexts:
                    try:
                        return dataset_contexts[direct_key].format(
                            dataset_name=self.dataset_name,
                            features=features_str,
                            scale_min=self.scale_min if self.use_scaling else "N/A",
                            scale_max=self.scale_max if self.use_scaling else "N/A",
                            scale_range=scale_range_str,
                            scale_info=scale_info,
                        )
                    except Exception:
                        return str(dataset_contexts[direct_key])
        
        # Try dataset name match
        for key, template in dataset_contexts.items():
            if key != 'generic' and key in dataset_lower:
                try:
                    return template.format(
                        dataset_name=self.dataset_name,
                        features=features_str,
                        scale_min=self.scale_min if self.use_scaling else "N/A",
                        scale_max=self.scale_max if self.use_scaling else "N/A",
                        scale_range=scale_range_str,
                        scale_info=scale_info,
                    )
                except Exception:
                    return str(template)
        
        # Generic fallback
        generic_template = dataset_contexts.get('generic', 
            """DATASET: {dataset_name}
FEATURES: {features}
OPTIMIZATION GUIDANCE:
- Use actual feature names (not generic x1, x2, etc.)
- Model nonlinear effects (saturation, thresholds, synergies) when patterns suggest them
- Include domain-relevant interactions
- {scale_info}
""")
        return generic_template.format(
            dataset_name=self.dataset_name, 
            features=features_str,
            scale_min=self.scale_min if self.use_scaling else "N/A",
            scale_max=self.scale_max if self.use_scaling else "N/A",
            scale_range=scale_range_str,
            scale_info=scale_info
        )
    
    def _build_iteration_history_context(self, iteration_history: Optional[List[Dict[str, Any]]] = None, 
                                         mechanism_idx: Optional[int] = None) -> str:
        """Build concise iteration history showing only what changed, not full mechanisms"""
        if not iteration_history or len(iteration_history) == 0:
            return ""
        
        # Filter history for this mechanism if mechanism_idx provided
        if mechanism_idx is not None:
            filtered_history = [h for h in iteration_history 
                              if h.get('mechanism_local_index') == mechanism_idx or 
                                 h.get('mechanism_index') == mechanism_idx]
        else:
            filtered_history = iteration_history
        
        if not filtered_history:
            return ""
        
        # Show only last 3 iterations (most recent first)
        recent_history = filtered_history[-3:] if len(filtered_history) > 3 else filtered_history
        
        context_parts = []
        context_parts.append("\n" + "=" * 80)
        context_parts.append("ITERATION HISTORY (Previous Changes):")
        context_parts.append("=" * 80)
        
        for hist in reversed(recent_history):
            iter_num = hist.get('iteration', '?')
            accepted = hist.get('accepted', False)
            reason = hist.get('reason', '')
            
            # Extract only the formula from mechanisms (not full descriptions)
            mech_before = hist.get('mechanism_before', '')
            mech_after = hist.get('mechanism_after', '')
            
            # Extract formula lines only
            def extract_formula(mech_text: str) -> str:
                lines = mech_text.split('\n')
                for line in lines:
                    if 'Formula:' in line or 'formula:' in line.lower():
                        return line.strip()
                # Fallback: return first line if no formula found
                return lines[0].strip() if lines else ""
            
            formula_before = extract_formula(mech_before)
            formula_after = extract_formula(mech_after)
            
            # Get metrics
            metrics_before = hist.get('metrics_before', {})
            metrics_after = hist.get('metrics_after', {})
            
            status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
            context_parts.append(f"\nIteration {iter_num}: {status} - {reason}")
            
            if self.task_type == "regression":
                r2_before = metrics_before.get('r2', None)
                mae_before = metrics_before.get('mae', None)
                r2_after = metrics_after.get('r2', None)
                mae_after = metrics_after.get('mae', None)
                if r2_before is not None and r2_after is not None and mae_before is not None and mae_after is not None:
                    perf_str = f"R²: {r2_before:.3f}→{r2_after:.3f}, MAE: {mae_before:.3f}→{mae_after:.3f}"
                else:
                    perf_str = "Metrics: updated"
            else:
                acc_before = metrics_before.get('accuracy', None)
                f1_before = metrics_before.get('f1', None)
                acc_after = metrics_after.get('accuracy', None)
                f1_after = metrics_after.get('f1', None)
                if acc_before is not None and acc_after is not None and f1_before is not None and f1_after is not None:
                    perf_str = f"ACC: {acc_before:.3f}→{acc_after:.3f}, F1: {f1_before:.3f}→{f1_after:.3f}"
                else:
                    perf_str = "Metrics: updated"
            
            context_parts.append(f"  Performance: {perf_str}")
            if formula_before and formula_after:
                # Truncate long formulas
                before_short = formula_before[:100] + "..." if len(formula_before) > 100 else formula_before
                after_short = formula_after[:100] + "..." if len(formula_after) > 100 else formula_after
                context_parts.append(f"  Formula: {before_short} → {after_short}")
        
        context_parts.append("\n" + "=" * 80)
        context_parts.append("Use this history to avoid repeating mistakes and build on successful changes.")
        context_parts.append("=" * 80 + "\n")
        
        return "\n".join(context_parts)
    
    def optimize_batch(self, mechanisms: List[str], error_feedbacks: List[str], 
                       ml_residuals: Optional[np.ndarray] = None,
                       X_train: Optional[np.ndarray] = None,
                       y_train: Optional[np.ndarray] = None,
                       ml_mechanism: Optional[Any] = None,
                       mechanism_performance: Optional[Dict[int, float]] = None,
                       mechanism_metrics: Optional[Dict[int, Dict[str, float]]] = None,
                       iteration_history: Optional[List[Dict[str, Any]]] = None,
                       X_train_original: Optional[List[Dict[str, Any]]] = None,
                       output_dir: Optional[str] = None,
                       iteration: Optional[int] = None,
                       latent_zs: Optional[List[Optional[str]]] = None) -> List[str]:
        """Optimize mechanisms using textual gradients based on residual samples or pre-computed latent_zs"""
        if len(mechanisms) != len(error_feedbacks):
            raise ValueError("mechanisms and error_feedbacks must have same length")
        if len(mechanisms) == 0:
            return []
        
        # Note: error_feedbacks are already built with latent_zs (if available) in maicl_lib_v2.py
        # The latent_zs parameter is passed here for logging/future use
        if latent_zs is not None:
            num_with_latent_z = sum(1 for z in latent_zs if z is not None)
            if num_with_latent_z > 0:
                logger.info(f"  [TextGrad] Using pre-computed latent_zs for {num_with_latent_z}/{len(mechanisms)} mechanism(s)")
        
        # Note: residual_context is already included in error_feedback via _build_enhanced_error_feedback
        # So we set it to empty string to avoid duplication in the prompt
        residual_context = ""
        
        # Get dataset-specific context
        dataset_context = self._get_dataset_context()
        
        # Build feature instruction
        feature_instruction = ""
        is_deepchem = (self.feature_cols and len(self.feature_cols) > 0 and 
                      all(feat.startswith('ecfp_bit_') for feat in self.feature_cols[:10]))
        
        feature_templates = self.config.get('templates', {}).get('feature_instruction', {})
        if is_deepchem:
            feature_instruction = feature_templates.get('deepchem', "")
        elif self.feature_cols and len(self.feature_cols) > 0:
            # Include ALL features in the feature instruction (no truncation)
            feature_list = ', '.join(self.feature_cols)
            feature_example = f"{self.feature_cols[0]} + {self.feature_cols[1]}" if len(self.feature_cols) >= 2 else self.feature_cols[0]
            logger.debug(f"  [TextGrad] Feature instruction includes {len(self.feature_cols)} features: {feature_list[:100]}...")
            feature_instruction = feature_templates.get('standard', "").format(
                feature_list=feature_list,
                feature_example=feature_example
            )
        
        # Build optimization prompts
        main_template = self.config.get('prompts', {}).get('optimization', {}).get('main_template', "")
        formula_instructions = self.config.get('formula_instructions', {})
        
        if self.task_type == "classification":
            formula_instruction = formula_instructions.get(
                'classification',
                "- For classification: Formula MUST output a CLASS INDEX (integer: 0, 1, 2, ...), NOT a probability or continuous value."
            )
        else:
            if self.use_scaling:
                formula_instruction = formula_instructions.get('regression', 
                    f"- For regression: Clipped to [{self.scale_min}, {self.scale_max}]\n- CRITICAL: ALL constants in your formula must be in [{self.scale_min}, {self.scale_max}] scaled range. Base constants on scaled sample values (e.g., if you see Temperature=0.93, use 0.93 not 34°C).")
            else:
                formula_instruction = "- For regression: Output numeric values (no clipping, use raw target values)"
        
        prompts = []
        prompt_saved = False  # Track if we've logged the file save message
        for mech_idx, (mechanism, error_feedback) in enumerate(zip(mechanisms, error_feedbacks)):
            # Build iteration history for this specific mechanism
            mech_history_context = ""
            if iteration_history:
                mech_history_context = self._build_iteration_history_context(iteration_history, mechanism_idx=mech_idx)
            
            # Format main template
            format_kwargs = {
                'dataset_context': dataset_context,
                'mechanism': mechanism,
                'error_feedback': error_feedback,
                'residual_context': "",  # Empty - residual context is already in error_feedback to avoid duplication
                'iteration_history': mech_history_context,  # Add iteration history
                'feature_instruction': feature_instruction,
                'formula_instruction': formula_instruction,
                'scale_min': self.scale_min if self.use_scaling else "",
                'scale_max': self.scale_max if self.use_scaling else ""
            }
            
            try:
                prompt = main_template.format(**format_kwargs)
            except KeyError as e:
                logger.warning(f"Template formatting error: {e}")
                # Fallback to simple prompt
                prompt = f"""{dataset_context}

CURRENT MECHANISM:
{mechanism}

{error_feedback}

{residual_context}

{feature_instruction}

YOUR TASK: Optimize this mechanism based on the residual patterns shown above. Learn from the examples where the model fails and improve the mechanism to handle these cases correctly.

{formula_instruction}

Provide an improved mechanism with a formula that addresses the errors shown in the residual analysis.
"""
            
            # Validate prompt is not empty
            if not prompt or not prompt.strip():
                logger.error(f"  [TextGrad] ERROR: Generated prompt is empty! This will cause LLM to return empty response.")
                logger.error(f"  [TextGrad] Template loaded: {bool(main_template)}")
                logger.error(f"  [TextGrad] Mechanism: {mechanism[:100] if mechanism else 'None'}...")
                logger.error(f"  [TextGrad] Error feedback length: {len(error_feedback) if error_feedback else 0}")
                # Skip this mechanism - return original
                prompts.append(None)  # Mark as invalid
                continue
            
            # Validate prompt is not empty
            if not prompt or not prompt.strip():
                logger.error(f"  [TextGrad] ERROR: Generated prompt is empty! This will cause LLM to return empty response.")
                logger.error(f"  [TextGrad] Template loaded: {bool(main_template)}")
                logger.error(f"  [TextGrad] Mechanism: {mechanism[:100] if mechanism else 'None'}...")
                logger.error(f"  [TextGrad] Error feedback length: {len(error_feedback) if error_feedback else 0}")
                # Skip this mechanism - will keep original
                prompts.append(None)  # Mark as invalid
                continue
            
            # Log the complete prompt for debugging/verification
            logger.info("=" * 80)
            logger.info(f"  [TextGrad] COMPLETE PROMPT SENT TO LLM (Dataset: {self.dataset_name}, Task: {self.task_type})")
            logger.info("=" * 80)
            logger.info(prompt)
            logger.info("=" * 80)
            
            # Save prompt to file if output_dir and iteration are provided
            if output_dir is not None and iteration is not None:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    prompt_file = os.path.join(output_dir, f"textgrad_prompts_iter_{iteration}.txt")
                    
                    # Count valid prompts (not None) to determine if this is the first valid one
                    valid_prompt_count = len([p for p in prompts if p is not None])
                    
                    # Write to file (overwrite, not append - one file per iteration with all mechanisms)
                    mode = 'w' if valid_prompt_count == 1 else 'a'  # First valid mechanism: write, others: append
                    with open(prompt_file, mode, encoding='utf-8') as f:
                        if valid_prompt_count == 1:
                            # First mechanism - write header
                            f.write("=" * 80 + "\n")
                            f.write(f"TEXTGRAD PROMPTS - ITERATION {iteration}\n")
                            f.write(f"Dataset: {self.dataset_name}, Task: {self.task_type}\n")
                            f.write("=" * 80 + "\n\n")
                        f.write("=" * 80 + "\n")
                        f.write(f"PROMPT FOR MECHANISM {valid_prompt_count}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(prompt)
                        f.write("\n\n" + "=" * 80 + "\n\n")
                    
                    if not prompt_saved:  # Only log once per iteration
                        logger.info(f"  ✓ Saved TextGrad prompt to {prompt_file}")
                        prompt_saved = True
                except Exception as e:
                    logger.warning(f"  Failed to save TextGrad prompt to file: {e}")
            
            prompts.append(HumanMessage(content=prompt))
        
        # Filter out None prompts (invalid/empty prompts)
        valid_prompts = [p for p in prompts if p is not None]
        if len(valid_prompts) == 0:
            logger.error("  [TextGrad] ERROR: All prompts are empty! Cannot call LLM.")
            # Return original mechanisms as fallback
            return mechanisms
        
        gradients = self.llm.invoke_batch(valid_prompts)
        
        # Map gradients back to original mechanism positions (accounting for skipped None prompts)
        full_gradients = []
        valid_idx = 0
        for p in prompts:
            if p is None:
                # Keep original mechanism for invalid prompts
                full_gradients.append(None)
            else:
                full_gradients.append(gradients[valid_idx] if valid_idx < len(gradients) else None)
                valid_idx += 1
        gradients = full_gradients
        
        # Extract content from gradients and validate
        extracted_mechanisms = []
        for i, gradient in enumerate(gradients):
            if gradient is None:
                # Keep original mechanism if gradient is None (invalid prompt)
                extracted_mechanisms.append(mechanisms[i] if i < len(mechanisms) else "")
                logger.warning(f"  [TextGrad] Mechanism {i+1}: No gradient (invalid prompt), keeping original")
                continue
            
            # Extract content from gradient (handle both Message and string types)
            if hasattr(gradient, 'content'):
                content = gradient.content
            else:
                content = str(gradient)
            
            # Validate content is not empty
            if not content or not content.strip():
                logger.warning(f"  [TextGrad] Mechanism {i+1}: LLM returned empty response, keeping original mechanism")
                extracted_mechanisms.append(mechanisms[i] if i < len(mechanisms) else "")
            else:
                extracted_mechanisms.append(content)
        
        # Log the complete optimized mechanisms returned by LLM
        logger.info("=" * 80)
        logger.info(f"  [TextGrad] COMPLETE OPTIMIZED MECHANISMS FROM LLM (Dataset: {self.dataset_name}, Task: {self.task_type})")
        logger.info("=" * 80)
        for i, mech in enumerate(extracted_mechanisms):
            logger.info(f"\n--- Optimized Mechanism {i+1} ---")
            logger.info(mech[:500] + "..." if len(mech) > 500 else mech)
        logger.info("=" * 80)
        
        # Save optimized mechanisms to file if output_dir and iteration are provided
        if output_dir is not None and iteration is not None:
            try:
                os.makedirs(output_dir, exist_ok=True)
                optimized_file = os.path.join(output_dir, f"textgrad_optimized_iter_{iteration}.txt")
                
                with open(optimized_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"OPTIMIZED MECHANISMS FROM LLM (Iteration {iteration})\n")
                    f.write(f"Dataset: {self.dataset_name}, Task: {self.task_type}\n")
                    f.write("=" * 80 + "\n\n")
                    for i, mech in enumerate(extracted_mechanisms):
                        f.write(f"--- Optimized Mechanism {i+1} ---\n")
                        f.write(mech)
                        f.write("\n\n" + "=" * 80 + "\n\n")
                
                logger.info(f"  ✓ Saved optimized mechanisms to {optimized_file}")
            except Exception as e:
                logger.warning(f"  Failed to save optimized mechanisms to file: {e}")
        
        return extracted_mechanisms
    
    def optimize_routing_weights(self, current_ml_weight: float, current_llm_weight: float,
                                  current_metrics: Dict[str, float],
                                  ml_performance: Optional[float] = None,
                                  llm_performance: Optional[float] = None) -> Tuple[float, float]:
        """
        Simple routing weight optimization based on performance.
        Returns weights that sum to 1.0.
        """
        # Simple approach: if we have performance scores, use them to set weights
        if ml_performance is not None and llm_performance is not None:
            # Normalize performance scores to get weights
            total_perf = ml_performance + llm_performance
            if total_perf > 0:
                ml_weight = ml_performance / total_perf
                llm_weight = llm_performance / total_perf
            else:
                ml_weight = current_ml_weight
                llm_weight = current_llm_weight
        else:
            # No performance scores: use LLM to optimize based on metrics
            try:
                import json
                import re
                
                # Simple prompt
                if self.task_type == "regression":
                    r2 = current_metrics.get('r2', 0.0)
                    mae = current_metrics.get('mae', 1.0)
                    prompt = f"""Current performance: R²={r2:.4f}, MAE={mae:.4f}
Current weights: ML={current_ml_weight:.4f}, LLM={current_llm_weight:.4f}

Recommend optimal weights (must sum to 1.0). JSON: {{"ml_weight": <0.0-1.0>, "llm_weight": <0.0-1.0>}}"""
                else:
                    acc = current_metrics.get('accuracy', 0.0)
                    f1 = current_metrics.get('f1', 0.0)
                    prompt = f"""Current performance: Accuracy={acc:.4f}, F1={f1:.4f}
Current weights: ML={current_ml_weight:.4f}, LLM={current_llm_weight:.4f}

Recommend optimal weights (must sum to 1.0). JSON: {{"ml_weight": <0.0-1.0>, "llm_weight": <0.0-1.0>}}"""
                
                response = self.llm.invoke_single(HumanMessage(content=prompt))
                cleaned = re.search(r'\{.*\}', response, re.S)
                if cleaned:
                    result = json.loads(cleaned.group(0))
                    ml_weight = float(result.get('ml_weight', current_ml_weight))
                    llm_weight = float(result.get('llm_weight', current_llm_weight))
                    # Normalize
                    total = ml_weight + llm_weight
                    if total > 0:
                        ml_weight = ml_weight / total
                        llm_weight = llm_weight / total
                    else:
                        ml_weight = current_ml_weight
                        llm_weight = current_llm_weight
                else:
                    ml_weight = current_ml_weight
                    llm_weight = current_llm_weight
            except Exception as e:
                logger.warning(f"  [Routing] Error: {e}, keeping current weights")
                ml_weight = current_ml_weight
                llm_weight = current_llm_weight
        
        # Ensure valid range and sum to 1.0
        ml_weight = max(0.0, min(1.0, ml_weight))
        llm_weight = max(0.0, min(1.0, llm_weight))
        total = ml_weight + llm_weight
        if total > 0:
            ml_weight = ml_weight / total
            llm_weight = llm_weight / total
        
        logger.info(f"  [Routing] Weights: ML={ml_weight:.4f}, LLM={llm_weight:.4f}")
        return ml_weight, llm_weight