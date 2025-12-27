"""
ML knowledge extraction and mechanism generation for MA-ICL system.
Contains functions for extracting knowledge from ML models and generating ML-guided mechanisms.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from maicl_config import (
    MAX_TOP_FEATURES, MAX_TOP_FEATURES_DISPLAY, MAX_FEATURE_INTERACTIONS,
    MAX_WORST_PREDICTIONS_COLLECT, MAX_WORST_PREDICTIONS_DISPLAY,
    MAX_TOP_ERROR_FEATURES, MAX_FEATURES_IN_EXAMPLE,
    SCALE_MIN, SCALE_MAX
)

logger = logging.getLogger(__name__)


def extract_ml_knowledge(ml_mechanism, feature_cols: List[str], task_type: str = "regression") -> Dict[str, Any]:
    """
    Extract interpretable knowledge from trained ML model.
    
    Returns:
        Dictionary with:
        - feature_importance: Dict mapping feature names to importance scores
        - coefficients: Model coefficients (if linear model)
        - top_features: List of most important features
        - feature_interactions: Suggested feature pairs for interactions
        - model_formula: Approximate formula representation
    """
    knowledge = {
        "feature_importance": {},
        "coefficients": None,
        "top_features": [],
        "feature_interactions": [],
        "model_formula": ""
    }
    
    if ml_mechanism is None or not ml_mechanism.is_trained:
        return knowledge
    
    try:
        model = ml_mechanism.model
        model_name = getattr(ml_mechanism, "model_name", "unknown")
        
        # Extract feature importance/coefficients based on model type
        if model_name in ["LinearRegression", "LogisticRegression"]:
            # Linear models: use coefficients
            if hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for binary/multiclass
                
                knowledge["coefficients"] = coef.tolist()
                
                # Map to feature importance (absolute value)
                for i, feat in enumerate(feature_cols):
                    if i < len(coef):
                        knowledge["feature_importance"][feat] = float(abs(coef[i]))
                
                # Generate formula
                intercept = model.intercept_ if hasattr(model, "intercept_") else 0
                if isinstance(intercept, np.ndarray):
                    intercept = intercept[0]
                
                terms = []
                for i, feat in enumerate(feature_cols):
                    if i < len(coef) and abs(coef[i]) > 0.01:
                        terms.append(f"{coef[i]:.3f}*{feat}")
                
                if task_type == "regression":
                    knowledge["model_formula"] = f"ŷ = {intercept:.3f} + " + " + ".join(terms[:MAX_TOP_FEATURES])
                else:
                    knowledge["model_formula"] = f"score = {intercept:.3f} + " + " + ".join(terms[:MAX_TOP_FEATURES])
        
        elif model_name == "XGBoost":
            # XGBoost: use feature importance
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                for i, feat in enumerate(feature_cols):
                    if i < len(importance):
                        knowledge["feature_importance"][feat] = float(importance[i])
        
        elif model_name == "KNN":
            # KNN: no direct feature importance, use correlation with target
            # This requires training data, so skip for now
            pass
        
        # Sort features by importance
        if knowledge["feature_importance"]:
            sorted_feats = sorted(
                knowledge["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            knowledge["top_features"] = [f[0] for f in sorted_feats[:MAX_TOP_FEATURES]]
            
            # Suggest interactions between top features
            if len(knowledge["top_features"]) >= 2:
                knowledge["feature_interactions"] = [
                    (knowledge["top_features"][0], knowledge["top_features"][1]),
                    (knowledge["top_features"][0], knowledge["top_features"][2]) if len(knowledge["top_features"]) > 2 else None
                ]
                knowledge["feature_interactions"] = [x for x in knowledge["feature_interactions"] if x is not None]
        
    except Exception as e:
        logger.warning(f"Failed to extract ML knowledge: {e}")
    
    return knowledge


def generate_ml_guided_mechanism(ml_knowledge: Dict[str, Any],
                                  feature_cols: List[str],
                                  task_type: str = "regression",
                                  class_names: Optional[List[str]] = None,
                                  variant: int = 0,
                                  use_scaling: bool = True) -> str:
    """
    Generate an LLM mechanism initialized from ML model knowledge.
    
    Args:
        ml_knowledge: Output from extract_ml_knowledge()
        variant: Which variant to generate (0=linear-like, 1=with interactions, 2=non-linear)
    
    Returns:
        Mechanism description string
    """
    # Check if this is a DeepChem dataset - if so, use molecular properties instead of ECFP features
    is_deepchem = (feature_cols and len(feature_cols) > 0 and 
                  all(feat.startswith('ecfp_bit_') for feat in feature_cols[:10]))
    
    top_features = ml_knowledge.get("top_features", [])
    if not top_features:
        # Use up to 5 features as fallback (matching variant 0 usage)
        # For DeepChem, don't use ECFP features - will use molecular properties instead
        if not is_deepchem:
            top_features = feature_cols[:min(5, len(feature_cols))]
        else:
            # For DeepChem, use molecular properties instead
            top_features = ["molecular_weight(SMILES)", "num_rings(SMILES)", "num_hydroxyl_groups(SMILES)"]
    
    feature_importance = ml_knowledge.get("feature_importance", {})
    interactions = ml_knowledge.get("feature_interactions", [])
    
    # Get scaling range
    scale_min, scale_max = SCALE_MIN, SCALE_MAX
    
    # Build mechanism based on variant
    if variant == 0:
        # Variant 0: Nonlinear transformations with intermediate variables (complements ML baseline)
        if is_deepchem:
            # For DeepChem: use molecular properties, not ECFP features
            mechanism = f"""[ML-GUIDED LINEAR] This mechanism uses molecular properties derived from SMILES strings.

MECHANISM: The output is a weighted combination of molecular properties that can be computed from SMILES strings.

KEY MOLECULAR PROPERTIES:

"""
            mechanism += "  1. molecular_weight(SMILES) - overall molecular size\n"
            mechanism += "  2. num_rings(SMILES) - ring count\n"
            mechanism += "  3. num_hydroxyl_groups(SMILES) - hydrogen bonding capacity\n"
            
            if use_scaling:
                mechanism += f"\nFORMULA: ŷ = clip((molecular_weight(SMILES) + num_rings(SMILES) + num_hydroxyl_groups(SMILES)) / 3, {scale_min}, {scale_max})"
            else:
                mechanism += f"\nFORMULA: ŷ = (molecular_weight(SMILES) + num_rings(SMILES) + num_hydroxyl_groups(SMILES)) / 3"
        else:
            # For non-DeepChem: use actual features
            # IMPORTANT: Don't use feature importance as coefficients - they're not mathematically equivalent
            # Instead, use normalized weights as a starting point, and let TextGrad optimize them
            mechanism = f"""[ML-GUIDED WITH NONLINEAR TRANSFORMATIONS] This mechanism uses the most important features identified by the ML model with advanced nonlinear transformations.

MECHANISM: The output uses nonlinear transformations (saturation effects, intermediate variables, nonlinear interactions) of the top features. The ML model indicates these features are most important, but you must learn proper coefficients and nonlinear forms from the data through optimization. This mechanism should work well as a standalone predictor - optimize coefficients, saturation parameters, and interaction forms to maximize independent predictive performance (high R², low MAE when used alone). Use intermediate variables to capture complex nonlinear relationships between features. Learn the nonlinearity of interactions themselves - interactions may have saturation effects.

KEY FEATURES (by ML relative importance - use as guide, NOT as coefficients):

"""
            for i, feat in enumerate(top_features[:MAX_TOP_FEATURES_DISPLAY]):
                imp = feature_importance.get(feat, 0.5)
                mechanism += f"  {i+1}. {feat} (relative importance: {imp:.3f} - this is NOT a coefficient!)\n"
            
            # Use normalized weights based on relative importance, not raw importance values
            # Normalize so weights sum to a reasonable range
            total_importance = sum(feature_importance.get(feat, 0.5) for feat in top_features[:MAX_TOP_FEATURES])
            if total_importance > 0:
                # Normalize to sum to 1.0, then scale to reasonable coefficient range
                normalized_weights = [feature_importance.get(feat, 0.5) / total_importance for feat in top_features[:MAX_TOP_FEATURES]]
                # Scale to reasonable range (0.1 to 1.0) for initial coefficients
                scaled_weights = [max(0.1, min(1.0, w * 2.0)) for w in normalized_weights]
            else:
                # Fallback: equal weights
                scaled_weights = [0.5] * len(top_features[:MAX_TOP_FEATURES])
            
            mechanism += f"\nINITIAL FORMULA (starting point - optimize coefficients based on performance metrics R² and MAE):\n"
            mechanism += f"ŷ = ("
            terms = []
            for i, feat in enumerate(top_features[:MAX_TOP_FEATURES]):
                weight = scaled_weights[i] if i < len(scaled_weights) else 0.5
                terms.append(f"{weight:.3f}*{feat}")
            mechanism += " + ".join(terms)
            mechanism += f") / {len(top_features[:MAX_TOP_FEATURES])}"
            mechanism += f"\n\nCRITICAL: The coefficients above (e.g., {scaled_weights[0]:.3f}, {scaled_weights[1] if len(scaled_weights) > 1 else 0.5:.3f}) are normalized starting values, NOT feature importance values. Feature importance values (like 0.392, 0.150) are NOT coefficients - they only indicate which features are important. You MUST learn proper coefficients (typically 0.1-2.0 range) by optimizing based on R² and MAE performance metrics, not by copying importance values."
        
        if task_type == "classification":
            if class_names and len(class_names) > 0:
                n_classes = len(class_names)
                class_mapping = ", ".join([f"{i}={cn}" for i, cn in enumerate(class_names)])

                # Detect lymphography-style feature set to tailor the initial mechanism (avoids filler text).
                feat_set = set([str(f) for f in (feature_cols or [])])
                is_lymphography = all(k in feat_set for k in ("lymphatics", "block_of_affere", "bl_of_lymph_s", "bl_of_lymph_c", "by_pass"))
                
                # Generate class-specific equations instead of thresholding
                mechanism += f"\n\nCLASSIFICATION FORMULATION:\n"
                if is_lymphography and n_classes == 4:
                    mechanism += (
                        "This is a 4-way lymphography diagnosis task with many ordinal/discrete features (often {0.00, 0.33, 0.67, 1.00}). "
                        "Use per-class scores with saturation and a few robust interactions, then pick the argmax.\n\n"
                    )
                else:
                    mechanism += (
                        "Use separate per-class score equations with nonlinear transforms and interactions, then select the class with the highest score.\n\n"
                    )
                
                # Generate textual interpretations and equations for each class
                for i, class_name in enumerate(class_names):
                    mechanism += f"CLASS {i} ({class_name}):\n"
                    
                    # Pick a couple features for concrete signatures
                    primary_feat = top_features[i % len(top_features)] if len(top_features) > 0 else "feature_0"
                    secondary_feat = top_features[(i + 1) % len(top_features)] if len(top_features) > 1 else primary_feat

                    # Concrete signatures for lymphography (if detected)
                    if is_lymphography and n_classes == 4:
                        if str(class_name).lower() == "normal":
                            signature = "lymphatics high (>=0.67) AND block_of_affere low (<=0.33); unlike metastases which shows stronger node defects"
                        elif str(class_name).lower() == "fibrosis":
                            signature = "lym_nodes_dimin high (>=0.67) AND regeneration_of moderate/high; unlike normal which has high lymphatics"
                        elif "metast" in str(class_name).lower():
                            signature = "defect_in_node high (>=0.67) OR bl_of_lymph_s/by_pass interaction strong; unlike malign_lymph which relies more on regeneration_of"
                        else:  # malign_lymph
                            signature = "regeneration_of high (>=0.67) with low lymphatics; unlike fibrosis which has stronger lym_nodes_dimin"
                        mechanism += f"  SIGNATURE: {signature}\n"
                    else:
                        mechanism += f"  SIGNATURE: {primary_feat} high (>=0.67) with {secondary_feat} low (<=0.33) (adjust thresholds as needed)\n"
                    
                    # Add equation after interpretation - USE ADVANCED NONLINEAR TRANSFORMATIONS
                    mechanism += f"  EQUATION (with nonlinear transformations and interactions):\n"
                    # Use different feature combinations for each class to encourage diversity
                    # Apply sophisticated nonlinear transformations including intermediate variables
                    if len(top_features) >= 2:
                        # Alternate which features are emphasized for each class
                        w1 = scaled_weights[i % len(scaled_weights)] if len(scaled_weights) > 0 else 0.5
                        w2 = scaled_weights[(i + 1) % len(scaled_weights)] if len(scaled_weights) > 1 else 0.3
                        saturation_param = 0.3 + (i * 0.1)  # Vary saturation parameter per class
                        
                        # Create intermediate variable for nonlinear interaction
                        mechanism += f"    # Intermediate variable capturing nonlinear interaction\n"
                        mechanism += f"    intermediate_{i} = {primary_feat} * {secondary_feat} / (0.2 + {primary_feat} * {secondary_feat})\n"
                        mechanism += f"    # Main equation with saturation and nonlinear interaction\n"
                        mechanism += f"    score_{i} = {w1:.3f} * {primary_feat} / ({saturation_param:.2f} + {primary_feat}) + {w2:.3f} * {secondary_feat} / (0.4 + {secondary_feat}) + 0.25 * intermediate_{i}"
                        
                        if len(top_features) >= 3:
                            tertiary_feat = top_features[(i + 2) % len(top_features)]
                            w3 = scaled_weights[(i + 2) % len(scaled_weights)] if len(scaled_weights) > 2 else 0.2
                            # Add three-way interaction with nonlinearity
                            mechanism += f" + {w3:.3f} * {primary_feat} * {tertiary_feat} / (0.3 + {primary_feat} * {tertiary_feat})"
                        mechanism += f"\n"
                    else:
                        saturation_param = 0.3
                        mechanism += f"    score_{i} = {scaled_weights[0] if len(scaled_weights) > 0 else 0.5:.3f} * {top_features[0]} / ({saturation_param:.2f} + {top_features[0]})\n"
                    
                    mechanism += f"    NOTE: Intermediate variables (like intermediate_{i}) capture complex nonlinear interactions between features. "
                    mechanism += f"The saturation terms (feature / (K + feature)) model diminishing returns. "
                    mechanism += f"Nonlinear interactions (feature1 * feature2 / (K + feature1 * feature2)) capture how features interact nonlinearly. "
                    mechanism += f"Optimize saturation parameters, interaction coefficients, and all weights to improve performance.\n"
                
                mechanism += f"\nFINAL PREDICTION:\n"
                mechanism += f"ŷ = argmax([score_0, score_1"
                if n_classes > 2:
                    for i in range(2, n_classes):
                        mechanism += f", score_{i}"
                mechanism += f"])\n"
                mechanism += f"Class mapping: {class_mapping}\n"
                mechanism += f"\nCRITICAL NONLINEARITY AND INTERACTION GUIDANCE:\n"
                mechanism += f"- Each class equation uses INTERMEDIATE VARIABLES to capture complex nonlinear interactions\n"
                mechanism += f"- Learn the NONLINEARITY of interactions: interactions themselves may have saturation effects (feature1 * feature2 / (K + feature1 * feature2))\n"
                mechanism += f"- Discover which features interact and HOW they interact nonlinearly (multiplicative, ratio-based, threshold-based)\n"
                mechanism += f"- Use intermediate variables to model complex relationships: intermediate = f(feature1, feature2) where f is nonlinear\n"
                mechanism += f"- Think about feature relationships: do features interact multiplicatively, additively, or through more complex patterns?\n"
                mechanism += f"- Optimize BOTH the form of interactions AND their coefficients - the nonlinearity of interactions matters\n"
                mechanism += f"- Linear combinations fail - you must learn nonlinear relationships and nonlinear interactions\n"
                mechanism += f"\n🧠 REASONING AND DOMAIN KNOWLEDGE:\n"
                mechanism += f"- Use your understanding of the domain to guide feature selection and interaction design\n"
                mechanism += f"- Think about what makes each class unique: what characteristics distinguish one class from another?\n"
                mechanism += f"- Reason about relationships: 'If class A has high feature X and class B has low feature X, then feature X is a discriminator'\n"
                mechanism += f"- Create meaningful intermediate variables that capture domain concepts (e.g., 'boxiness', 'elongation_factor')\n"
                mechanism += f"- Use analogical reasoning: 'This is similar to [domain concept], so I should model it as [mathematical form]'"
            else:
                mechanism += f"\n\nCLASSIFICATION FORMULATION:\n"
                mechanism += f"This mechanism uses separate equations for each class with nonlinear transformations.\n\n"
                mechanism += f"# Intermediate variable for nonlinear interaction\n"
                mechanism += f"intermediate_0 = {top_features[0] if len(top_features) > 0 else 'feature_0'} * {top_features[1] if len(top_features) > 1 else 'feature_1'} / (0.2 + {top_features[0] if len(top_features) > 0 else 'feature_0'} * {top_features[1] if len(top_features) > 1 else 'feature_1'})\n"
                mechanism += f"CLASS 0 SCORE: score_0 = {scaled_weights[0] if len(scaled_weights) > 0 else 0.5:.3f} * {top_features[0] if len(top_features) > 0 else 'feature_0'} / (0.3 + {top_features[0] if len(top_features) > 0 else 'feature_0'}) + 0.25 * intermediate_0\n"
                mechanism += f"CLASS 1 SCORE: score_1 = {scaled_weights[1] if len(scaled_weights) > 1 else 0.3:.3f} * {top_features[1] if len(top_features) > 1 else 'feature_1'} / (0.4 + {top_features[1] if len(top_features) > 1 else 'feature_1'}) + 0.25 * intermediate_0\n"
                mechanism += f"ŷ = argmax([score_0, score_1])\n"
        else:
            mechanism += f"\nClipped to [{scale_min}, {scale_max}]"
    
    elif variant == 1:
        # Variant 1: Add feature interactions
        if is_deepchem:
            # For DeepChem: use molecular properties with interactions
            mechanism = f"""[ML-GUIDED WITH INTERACTIONS] This mechanism uses molecular properties with interactions.

MECHANISM: Uses molecular properties derived from SMILES strings with multiplicative interactions.

KEY MOLECULAR PROPERTIES:

"""
            mechanism += "  1. molecular_weight(SMILES) - overall molecular size\n"
            mechanism += "  2. num_rings(SMILES) - ring count\n"
            mechanism += "  3. num_hydroxyl_groups(SMILES) - hydrogen bonding capacity\n"
            mechanism += "  4. num_halogen(SMILES) - halogen atom count\n"
            
            if use_scaling:
                mechanism += f"\nFORMULA: ŷ = clip((molecular_weight(SMILES) + num_rings(SMILES) + num_hydroxyl_groups(SMILES) - 0.1 * num_halogen(SMILES) + 0.05 * molecular_weight(SMILES) * num_rings(SMILES)) / 3, {scale_min}, {scale_max})"
            else:
                mechanism += f"\nFORMULA: ŷ = (molecular_weight(SMILES) + num_rings(SMILES) + num_hydroxyl_groups(SMILES) - 0.1 * num_halogen(SMILES) + 0.05 * molecular_weight(SMILES) * num_rings(SMILES)) / 3"
        else:
            # For non-DeepChem: use actual features
            mechanism = f"""[ML-GUIDED WITH INTERACTIONS] This mechanism uses the ML model's top features with interactions.

MECHANISM: Uses the most important features from the ML model and adds multiplicative interactions between them. This mechanism should work well as a standalone predictor - optimize coefficients to maximize independent predictive performance. Coefficients are normalized based on relative importance as starting values - learn proper values through optimization.

KEY FEATURES:

"""
            for i, feat in enumerate(top_features[:MAX_TOP_FEATURES_DISPLAY]):
                imp = feature_importance.get(feat, 0.5)
                mechanism += f"  {i+1}. {feat} (relative importance: {imp:.3f})\n"
            
            if interactions:
                mechanism += f"\nKEY INTERACTIONS:\n"
                for f1, f2 in interactions:
                    mechanism += f"  - {f1} × {f2}\n"
            
            # Normalize weights properly
            total_importance = sum(feature_importance.get(feat, 0.5) for feat in top_features[:MAX_TOP_FEATURES])
            if total_importance > 0:
                normalized_weights = [feature_importance.get(feat, 0.5) / total_importance for feat in top_features[:MAX_TOP_FEATURES]]
                scaled_weights = [max(0.1, min(1.0, w * 2.0)) for w in normalized_weights]
            else:
                scaled_weights = [0.5] * len(top_features[:MAX_TOP_FEATURES])
            
            mechanism += f"\nINITIAL FORMULA (starting point - optimize coefficients based on performance metrics R² and MAE):\n"
            mechanism += f"ŷ = ("
            terms = []
            # Use up to 5 features (matching variant 0 for consistency)
            for i, feat in enumerate(top_features[:MAX_TOP_FEATURES]):
                weight = scaled_weights[i] if i < len(scaled_weights) else 0.5
                terms.append(f"{weight:.3f}*{feat}")
            
            # Add interactions with smaller weights
            if interactions:
                # Use top 3 interactions (increased from 2 to match increased feature count)
                for f1, f2 in interactions[:MAX_FEATURE_INTERACTIONS]:
                    # Use smaller interaction coefficient (0.1-0.3 range)
                    interaction_weight = 0.2
                    terms.append(f"{interaction_weight:.3f}*{f1}*{f2}")
            
            mechanism += " + ".join(terms)
            mechanism += f") / {len(terms)}"
            mechanism += f"\n\nCRITICAL: The coefficients above are normalized starting values, NOT feature importance values. Feature importance values (like 0.392, 0.150) are NOT coefficients - they only indicate which features are important. You MUST learn proper coefficients (typically 0.1-2.0 range) by optimizing based on R² and MAE performance metrics, not by copying importance values."
        
        if task_type == "classification":
            if class_names and len(class_names) > 0:
                n_classes = len(class_names)
                class_mapping = ", ".join([f"{i}={cn}" for i, cn in enumerate(class_names)])
                
                # Generate class-specific equations instead of thresholding
                mechanism += f"\n\nCLASSIFICATION FORMULATION:\n"
                mechanism += f"This mechanism uses separate equations for each class. First, interpret each class textually based on its label and relationship to input features. Then compute a score for each class, and select the class with the highest score.\n\n"
                
                # Generate textual interpretations and equations for each class
                for i, class_name in enumerate(class_names):
                    mechanism += f"CLASS {i} ({class_name}):\n"
                    
                    # Add textual interpretation for the class
                    primary_feat = top_features[i % len(top_features)] if len(top_features) > 0 else "feature_0"
                    secondary_feat = top_features[(i + 1) % len(top_features)] if len(top_features) > 1 else primary_feat
                    
                    # Generate meaningful textual description based on class name
                    mechanism += f"  INTERPRETATION: "
                    if class_name and len(class_name) > 0 and not class_name.isdigit():
                        # Class has meaningful label - provide interpretation with domain knowledge
                        mechanism += f"The class '{class_name}' is characterized by "
                        if len(top_features) >= 2:
                            mechanism += f"specific patterns in {primary_feat} and {secondary_feat}. "
                        else:
                            mechanism += f"patterns in {primary_feat}. "
                        mechanism += f"Examples belonging to this class typically exhibit "
                        mechanism += f"distinctive NONLINEAR relationships between these features that distinguish '{class_name}' from other classes. "
                        mechanism += f"Think about what makes '{class_name}' unique: use your knowledge of the domain to understand why these features matter. "
                        mechanism += f"The INTERACTION between these features is also nonlinear - they don't just multiply, but interact in complex ways. "
                        mechanism += f"The mechanism models this class using intermediate variables and nonlinear transformations of {primary_feat}"
                        if len(top_features) >= 2:
                            mechanism += f" and {secondary_feat}"
                        mechanism += f" in the scoring equation. "
                        mechanism += f"Use your understanding of '{class_name}' to guide how these features should be combined.\n"
                    else:
                        # Generic class label - provide feature-based interpretation
                        mechanism += f"This class is characterized by specific patterns in {primary_feat}"
                        if len(top_features) >= 2:
                            mechanism += f" and {secondary_feat}"
                        mechanism += f". Examples belonging to this class typically exhibit distinctive NONLINEAR relationships between these features. "
                        mechanism += f"The INTERACTION between features is nonlinear. "
                        mechanism += f"The mechanism models this class using intermediate variables and nonlinear transformations of {primary_feat}"
                        if len(top_features) >= 2:
                            mechanism += f" and {secondary_feat}"
                        mechanism += f" in the scoring equation.\n"
                    
                    # Add equation after interpretation
                    mechanism += f"  EQUATION:\n"
                    # Use different feature combinations for each class to encourage diversity
                    if len(top_features) >= 2:
                        # Alternate which features are emphasized for each class
                        w1 = scaled_weights[i % len(scaled_weights)] if len(scaled_weights) > 0 else 0.5
                        w2 = scaled_weights[(i + 1) % len(scaled_weights)] if len(scaled_weights) > 1 else 0.3
                        mechanism += f"    score_{i} = {w1:.3f}*{primary_feat} + {w2:.3f}*{secondary_feat}"
                        if len(top_features) >= 3:
                            tertiary_feat = top_features[(i + 2) % len(top_features)]
                            w3 = scaled_weights[(i + 2) % len(scaled_weights)] if len(scaled_weights) > 2 else 0.2
                            mechanism += f" + {w3:.3f}*{tertiary_feat}"
                        mechanism += f"\n"
                    else:
                        mechanism += f"    score_{i} = {scaled_weights[0] if len(scaled_weights) > 0 else 0.5:.3f}*{top_features[0]}\n"
                
                mechanism += f"\nFINAL PREDICTION:\n"
                mechanism += f"ŷ = argmax([score_0, score_1"
                if n_classes > 2:
                    for i in range(2, n_classes):
                        mechanism += f", score_{i}"
                mechanism += f"])\n"
                mechanism += f"Class mapping: {class_mapping}\n"
                mechanism += f"\nCRITICAL NONLINEARITY AND INTERACTION GUIDANCE:\n"
                mechanism += f"- Each class equation uses INTERMEDIATE VARIABLES to capture complex nonlinear interactions\n"
                mechanism += f"- Learn the NONLINEARITY of interactions: interactions themselves may have saturation effects (feature1 * feature2 / (K + feature1 * feature2))\n"
                mechanism += f"- Discover which features interact and HOW they interact nonlinearly (multiplicative, ratio-based, threshold-based)\n"
                mechanism += f"- Use intermediate variables to model complex relationships: intermediate = f(feature1, feature2) where f is nonlinear\n"
                mechanism += f"- Think about feature relationships: do features interact multiplicatively, additively, or through more complex patterns?\n"
                mechanism += f"- Optimize BOTH the form of interactions AND their coefficients - the nonlinearity of interactions matters\n"
                mechanism += f"- Linear combinations fail - you must learn nonlinear relationships and nonlinear interactions\n"
                mechanism += f"\n🧠 REASONING AND DOMAIN KNOWLEDGE:\n"
                mechanism += f"- Use your understanding of the domain to guide feature selection and interaction design\n"
                mechanism += f"- Think about what makes each class unique: what characteristics distinguish one class from another?\n"
                mechanism += f"- Reason about relationships: 'If class A has high feature X and class B has low feature X, then feature X is a discriminator'\n"
                mechanism += f"- Create meaningful intermediate variables that capture domain concepts (e.g., 'boxiness', 'elongation_factor')\n"
                mechanism += f"- Use analogical reasoning: 'This is similar to [domain concept], so I should model it as [mathematical form]'"
            else:
                mechanism += f"\n\nCLASSIFICATION FORMULATION:\n"
                mechanism += f"CLASS 0 SCORE: score_0 = {scaled_weights[0] if len(scaled_weights) > 0 else 0.5:.3f}*{top_features[0] if len(top_features) > 0 else 'feature_0'}\n"
                mechanism += f"CLASS 1 SCORE: score_1 = {scaled_weights[1] if len(scaled_weights) > 1 else 0.3:.3f}*{top_features[1] if len(top_features) > 1 else 'feature_1'}\n"
                mechanism += f"ŷ = argmax([score_0, score_1])\n"
        else:
            mechanism += f"\nClipped to [{scale_min}, {scale_max}]"
    
    else:
        # Variant 2: Non-linear transformations (IMPROVED: Start closer to ML baseline)
        if is_deepchem:
            # For DeepChem: use molecular properties with non-linear transformations
            mechanism = f"""[ML-GUIDED NON-LINEAR] This mechanism uses non-linear transformations of molecular properties.

MECHANISM: Applies saturation and threshold effects to molecular properties derived from SMILES strings.

KEY MOLECULAR PROPERTIES:

"""
            mechanism += "  1. molecular_weight(SMILES) - overall molecular size\n"
            mechanism += "  2. num_rings(SMILES) - ring count\n"
            mechanism += "  3. num_hydroxyl_groups(SMILES) - hydrogen bonding capacity\n"
            mechanism += "  4. num_halogen(SMILES) - halogen atom count\n"
            
            if use_scaling:
                mechanism += f"\nFORMULA: ŷ = clip(molecular_weight(SMILES) / (100 + molecular_weight(SMILES)) + 0.1 * num_rings(SMILES) / (1 + num_rings(SMILES)) + 0.15 * num_hydroxyl_groups(SMILES) - 0.05 * num_halogen(SMILES), {scale_min}, {scale_max})"
            else:
                mechanism += f"\nFORMULA: ŷ = molecular_weight(SMILES) / (100 + molecular_weight(SMILES)) + 0.1 * num_rings(SMILES) / (1 + num_rings(SMILES)) + 0.15 * num_hydroxyl_groups(SMILES) - 0.05 * num_halogen(SMILES)"
        else:
            # For non-DeepChem: use actual features
            if task_type == "classification":
                # For classification: add descriptive text explaining the classification task
                class_desc = ""
                if class_names and len(class_names) > 0:
                    class_list = ", ".join(class_names)
                    class_desc = f"This mechanism classifies examples into {len(class_names)} classes: {class_list}. "
                
                mechanism = f"""[ML-GUIDED NON-LINEAR] This mechanism uses non-linear transformations of the ML model's top features.

MECHANISM DESCRIPTION:
{class_desc}The mechanism applies saturation and threshold effects to the most important features identified by the ML model. It uses non-linear transformations to capture complex decision boundaries between classes. The mechanism is designed to be a strong, independent predictor that works well on its own, while also complementing the linear ML baseline. Key features are combined using saturation functions (where very high feature values have diminishing returns) and weighted combinations to distinguish between different classes. Uses normalized coefficients based on relative importance as starting values - optimize these to maximize standalone predictive performance.

KEY FEATURES:

"""
                for i, feat in enumerate(top_features[:MAX_TOP_FEATURES_DISPLAY]):
                    imp = feature_importance.get(feat, 0.5)
                    mechanism += f"  {i+1}. {feat} (relative importance: {imp:.3f})\n"
                
                # Normalize weights properly
                total_importance = sum(feature_importance.get(feat, 0.5) for feat in top_features[:MAX_TOP_FEATURES])
                if total_importance > 0:
                    normalized_weights = [feature_importance.get(feat, 0.5) / total_importance for feat in top_features[:MAX_TOP_FEATURES]]
                    scaled_weights = [max(0.1, min(1.0, w * 2.0)) for w in normalized_weights]
                else:
                    scaled_weights = [0.5] * len(top_features[:MAX_TOP_FEATURES])
                
                mechanism += f"\nINITIAL FORMULA (starting point - optimize coefficients and saturation parameters based on performance metrics R² and MAE):\n"
                mechanism += f"ŷ = "
                
                if len(top_features) >= 2:
                    f1, f2 = top_features[0], top_features[1]
                    w1 = scaled_weights[0] if len(scaled_weights) > 0 else 0.5
                    w2 = scaled_weights[1] if len(scaled_weights) > 1 else 0.5
                    
                    # Use normalized weights with saturation for top feature
                    mechanism += f"{w1:.3f} * {f1} / (0.3 + {f1})"
                    
                    # Add second feature with interaction or addition
                    if interactions and len(interactions) > 0:
                        mechanism += f" * (1 + {w2*0.8:.3f}*{f2})"
                    else:
                        mechanism += f" + {w2*0.8:.3f}*{f2}"
                    
                    # Add third feature if available (with reduced weight to avoid overfitting)
                    if len(top_features) >= 3:
                        f3 = top_features[2]
                        w3 = scaled_weights[2] if len(scaled_weights) > 2 else 0.5
                        mechanism += f" + {w3*0.4:.3f}*{f3}"
                    
                    # Add fourth and fifth features with even smaller weights for stability
                    if len(top_features) >= 4:
                        f4 = top_features[3]
                        w4 = scaled_weights[3] if len(scaled_weights) > 3 else 0.5
                        mechanism += f" + {w4*0.2:.3f}*{f4}"
                    if len(top_features) >= 5:
                        f5 = top_features[4]
                        w5 = scaled_weights[4] if len(scaled_weights) > 4 else 0.5
                        mechanism += f" + {w5*0.1:.3f}*{f5}"
                else:
                    mechanism += f"{top_features[0]} / (0.5 + {top_features[0]})"
                
                mechanism += f"\n\nCRITICAL: Adjust coefficients, saturation parameters (e.g., 0.3 in denominator), and feature combinations based on R² and MAE performance metrics. Feature importance values are NOT coefficients - learn proper values through optimization."
            else:
                # For regression: keep original format
                mechanism = f"""[ML-GUIDED NON-LINEAR] This mechanism uses non-linear transformations of the ML model's top features.

MECHANISM: Applies saturation and threshold effects to the most important features from the ML model. This mechanism is designed to be a strong, independent predictor that works well on its own, while also complementing the linear ML baseline. Uses normalized coefficients based on relative importance as starting values - optimize these to maximize standalone predictive performance.

KEY FEATURES:

"""
                for i, feat in enumerate(top_features[:MAX_TOP_FEATURES_DISPLAY]):
                    imp = feature_importance.get(feat, 0.5)
                    mechanism += f"  {i+1}. {feat} (relative importance: {imp:.3f})\n"
                
                # Normalize weights properly
                total_importance = sum(feature_importance.get(feat, 0.5) for feat in top_features[:MAX_TOP_FEATURES])
                if total_importance > 0:
                    normalized_weights = [feature_importance.get(feat, 0.5) / total_importance for feat in top_features[:MAX_TOP_FEATURES]]
                    scaled_weights = [max(0.1, min(1.0, w * 2.0)) for w in normalized_weights]
                else:
                    scaled_weights = [0.5] * len(top_features[:MAX_TOP_FEATURES])
                
                mechanism += f"\nINITIAL FORMULA (starting point - optimize coefficients and saturation parameters based on performance metrics R² and MAE):\n"
                mechanism += f"ŷ = "
                
                if len(top_features) >= 2:
                    f1, f2 = top_features[0], top_features[1]
                    w1 = scaled_weights[0] if len(scaled_weights) > 0 else 0.5
                    w2 = scaled_weights[1] if len(scaled_weights) > 1 else 0.5
                    
                    # Use normalized weights with saturation for top feature
                    mechanism += f"{w1:.3f} * {f1} / (0.3 + {f1})"
                    
                    # Add second feature with interaction or addition
                    if interactions and len(interactions) > 0:
                        mechanism += f" * (1 + {w2*0.8:.3f}*{f2})"
                    else:
                        mechanism += f" + {w2*0.8:.3f}*{f2}"
                    
                    # Add third feature if available (with reduced weight to avoid overfitting)
                    if len(top_features) >= 3:
                        f3 = top_features[2]
                        w3 = scaled_weights[2] if len(scaled_weights) > 2 else 0.5
                        mechanism += f" + {w3*0.4:.3f}*{f3}"
                    
                    # Add fourth and fifth features with even smaller weights for stability
                    if len(top_features) >= 4:
                        f4 = top_features[3]
                        w4 = scaled_weights[3] if len(scaled_weights) > 3 else 0.5
                        mechanism += f" + {w4*0.2:.3f}*{f4}"
                    if len(top_features) >= 5:
                        f5 = top_features[4]
                        w5 = scaled_weights[4] if len(scaled_weights) > 4 else 0.5
                        mechanism += f" + {w5*0.1:.3f}*{f5}"
                else:
                    mechanism += f"{top_features[0]} / (0.5 + {top_features[0]})"
                
                mechanism += f"\n\nCRITICAL: Adjust coefficients, saturation parameters (e.g., 0.3 in denominator), and feature combinations based on R² and MAE performance metrics. Feature importance values are NOT coefficients - learn proper values through optimization."
        
        if task_type == "classification":
            if class_names and len(class_names) > 0:
                n_classes = len(class_names)
                class_mapping = ", ".join([f"{i}={cn}" for i, cn in enumerate(class_names)])
                
                # Generate class-specific equations instead of thresholding
                mechanism += f"\n\nCLASSIFICATION FORMULATION:\n"
                mechanism += f"This mechanism uses separate equations for each class. First, interpret each class textually based on its label and relationship to input features. Then compute a score for each class, and select the class with the highest score.\n\n"
                
                # Generate textual interpretations and equations for each class with non-linear transformations
                for i, class_name in enumerate(class_names):
                    mechanism += f"CLASS {i} ({class_name}):\n"
                    
                    # Add textual interpretation for the class
                    primary_feat = top_features[i % len(top_features)] if len(top_features) > 0 else "feature_0"
                    secondary_feat = top_features[(i + 1) % len(top_features)] if len(top_features) > 1 else primary_feat
                    
                    # Generate meaningful textual description based on class name
                    mechanism += f"  INTERPRETATION: "
                    if class_name and len(class_name) > 0 and not class_name.isdigit():
                        # Class has meaningful label - provide interpretation
                        mechanism += f"The class '{class_name}' is characterized by "
                        if len(top_features) >= 2:
                            mechanism += f"specific patterns in {primary_feat} and {secondary_feat}. "
                        else:
                            mechanism += f"patterns in {primary_feat}. "
                        mechanism += f"Examples belonging to this class typically exhibit "
                        mechanism += f"distinctive relationships between these features that distinguish '{class_name}' from other classes. "
                        mechanism += f"The mechanism models this class using non-linear transformations (saturation effects) on {primary_feat}"
                        if len(top_features) >= 2:
                            mechanism += f" and linear contributions from {secondary_feat}"
                        mechanism += f" in the scoring equation.\n"
                    else:
                        # Generic class label - provide feature-based interpretation
                        mechanism += f"This class is characterized by specific patterns in {primary_feat}"
                        if len(top_features) >= 2:
                            mechanism += f" and {secondary_feat}"
                        mechanism += f". Examples belonging to this class typically exhibit distinctive relationships between these features. "
                        mechanism += f"The mechanism models this class using non-linear transformations (saturation effects) on {primary_feat}"
                        if len(top_features) >= 2:
                            mechanism += f" and linear contributions from {secondary_feat}"
                        mechanism += f" in the scoring equation.\n"
                    
                    # Add equation after interpretation - USE NONLINEAR TRANSFORMATIONS
                    mechanism += f"  EQUATION (with nonlinear transformations):\n"
                    # Use different feature combinations for each class to encourage diversity
                    # Apply nonlinear transformations to capture complex relationships
                    if len(top_features) >= 2:
                        # Alternate which features are emphasized for each class
                        w1 = scaled_weights[i % len(scaled_weights)] if len(scaled_weights) > 0 else 0.5
                        w2 = scaled_weights[(i + 1) % len(scaled_weights)] if len(scaled_weights) > 1 else 0.3
                        # Use saturation for primary feature to capture nonlinear relationships
                        saturation_param = 0.3 + (i * 0.1)  # Vary saturation parameter per class
                        mechanism += f"    score_{i} = {w1:.3f} * {primary_feat} / ({saturation_param:.2f} + {primary_feat}) + {w2:.3f} * {secondary_feat}"
                        if len(top_features) >= 3:
                            tertiary_feat = top_features[(i + 2) % len(top_features)]
                            w3 = scaled_weights[(i + 2) % len(scaled_weights)] if len(scaled_weights) > 2 else 0.2
                            # Add interaction term for nonlinearity
                            mechanism += f" + {w3:.3f} * {primary_feat} * {tertiary_feat}"
                        mechanism += f"\n"
                    else:
                        saturation_param = 0.3
                        mechanism += f"    score_{i} = {scaled_weights[0] if len(scaled_weights) > 0 else 0.5:.3f} * {top_features[0]} / ({saturation_param:.2f} + {top_features[0]})\n"
                    mechanism += f"    NOTE: The saturation term ({primary_feat} / ({saturation_param:.2f} + {primary_feat})) captures nonlinear relationships where high feature values have diminishing returns. The interaction term ({primary_feat} * {tertiary_feat if len(top_features) >= 3 else 'feature'}) captures synergistic effects. Optimize saturation parameters and coefficients to improve performance.\n"
                
                mechanism += f"\nFINAL PREDICTION:\n"
                mechanism += f"ŷ = argmax([score_0, score_1"
                if n_classes > 2:
                    for i in range(2, n_classes):
                        mechanism += f", score_{i}"
                mechanism += f"])\n"
                mechanism += f"Class mapping: {class_mapping}\n"
                mechanism += f"\nCRITICAL NONLINEARITY AND INTERACTION GUIDANCE:\n"
                mechanism += f"- Each class equation uses INTERMEDIATE VARIABLES to capture complex nonlinear interactions\n"
                mechanism += f"- Learn the NONLINEARITY of interactions: interactions themselves may have saturation effects (feature1 * feature2 / (K + feature1 * feature2))\n"
                mechanism += f"- Discover which features interact and HOW they interact nonlinearly (multiplicative, ratio-based, threshold-based)\n"
                mechanism += f"- Use intermediate variables to model complex relationships: intermediate = f(feature1, feature2) where f is nonlinear\n"
                mechanism += f"- Think about feature relationships: do features interact multiplicatively, additively, or through more complex patterns?\n"
                mechanism += f"- The saturation terms (feature / (K + feature)) capture diminishing returns and nonlinear relationships\n"
                mechanism += f"- Nonlinear interactions (feature1 * feature2 / (K + feature1 * feature2)) capture how features interact nonlinearly\n"
                mechanism += f"- Optimize BOTH the form of interactions AND their coefficients - the nonlinearity of interactions matters\n"
                mechanism += f"- Consider threshold effects: are there critical values where the relationship changes?\n"
                mechanism += f"- Linear combinations fail - you must learn nonlinear relationships and nonlinear interactions\n"
                mechanism += f"\n🧠 REASONING AND DOMAIN KNOWLEDGE:\n"
                mechanism += f"- Use your understanding of the domain to guide feature selection and interaction design\n"
                mechanism += f"- Think about what makes each class unique: what characteristics distinguish one class from another?\n"
                mechanism += f"- Reason about relationships: 'If class A has high feature X and class B has low feature X, then feature X is a discriminator'\n"
                mechanism += f"- Create meaningful intermediate variables that capture domain concepts (e.g., 'boxiness', 'elongation_factor')\n"
                mechanism += f"- Use analogical reasoning: 'This is similar to [domain concept], so I should model it as [mathematical form]'"
            else:
                mechanism += f"\n\nCLASSIFICATION FORMULATION:\n"
                mechanism += f"CLASS 0 SCORE: score_0 = {scaled_weights[0] if len(scaled_weights) > 0 else 0.5:.3f} * {top_features[0] if len(top_features) > 0 else 'feature_0'} / (0.3 + {top_features[0] if len(top_features) > 0 else 'feature_0'})\n"
                mechanism += f"CLASS 1 SCORE: score_1 = {scaled_weights[1] if len(scaled_weights) > 1 else 0.3:.3f} * {top_features[1] if len(top_features) > 1 else 'feature_1'} / (0.3 + {top_features[1] if len(top_features) > 1 else 'feature_1'})\n"
                mechanism += f"ŷ = argmax([score_0, score_1])\n"
        else:
            # Only mention clipping if scaling is enabled
            if use_scaling:
                mechanism += f"\nClipped to [{scale_min}, {scale_max}]"
    
    return mechanism


def analyze_ml_failure_patterns(ml_mechanism,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_cols: List[str],
                                 task_type: str = "regression",
                                 class_names: Optional[List[str]] = None,
                                 num_counterfactuals: int = 5) -> Dict[str, Any]:
    """
    Analyze where and why the ML model fails.
    
    Args:
        ml_mechanism: Trained ML mechanism
        X: Feature matrix
        y: Target values
        feature_cols: List of feature names
        task_type: "regression" or "classification"
        class_names: Optional list of class names for classification
        num_counterfactuals: Number of worst predictions to generate counterfactuals for (default: 5)
    
    Returns:
        Dictionary with:
        - error_by_feature: Per-feature error correlation
        - worst_predictions: Indices and details of worst predictions
        - error_patterns: Systematic error patterns
        - counterfactual_examples: Examples of how to fix errors
    """
    analysis = {
        "error_by_feature": {},
        "worst_predictions": [],
        "error_patterns": [],
        "counterfactual_examples": []
    }
    
    if ml_mechanism is None or not ml_mechanism.is_trained:
        return analysis
    
    try:
        model = ml_mechanism.model
        
        # Get predictions and compute errors
        if task_type == "classification":
            y_pred = model.predict(X).astype(int)
            y_true = y.astype(int)
            errors = (y_pred != y_true).astype(float)
            error_magnitude = errors  # 0 or 1
        else:
            y_pred = model.predict(X).astype(float)
            y_true = y.astype(float)
            errors = y_true - y_pred
            error_magnitude = np.abs(errors)
        
        # 1. Error correlation by feature
        for i, feat in enumerate(feature_cols):
            try:
                corr = np.corrcoef(X[:, i], error_magnitude)[0, 1]
                if np.isfinite(corr):
                    analysis["error_by_feature"][feat] = float(corr)
            except:
                pass
        
        # 2. Worst predictions
        worst_idx = np.argsort(error_magnitude)[::-1][:MAX_WORST_PREDICTIONS_COLLECT]
        for idx in worst_idx:
            true_val = float(y_true[idx])
            pred_val = float(y_pred[idx])
            
            # For classification, convert indices to class names if available
            if task_type == "classification" and class_names and len(class_names) > 0:
                true_idx = int(np.round(true_val))
                pred_idx = int(np.round(pred_val))
                true_name = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
                pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
                example = {
                    "index": int(idx),
                    "features": {feat: float(X[idx, j]) for j, feat in enumerate(feature_cols)},
                    "true": true_name,  # Store class name
                    "true_index": true_idx,  # Also store index for compatibility
                    "predicted": pred_name,  # Store class name
                    "predicted_index": pred_idx,  # Also store index for compatibility
                    "error": float(errors[idx])
                }
            else:
                example = {
                    "index": int(idx),
                    "features": {feat: float(X[idx, j]) for j, feat in enumerate(feature_cols)},
                    "true": float(true_val),
                    "predicted": float(pred_val),
                    "error": float(errors[idx])
                }
            analysis["worst_predictions"].append(example)
        
        # 3. Error patterns
        # Pattern: errors by feature value ranges
        # Analyze top 5 error-correlated features (increased from 3 for better coverage)
        top_error_features = sorted(
            analysis["error_by_feature"].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:MAX_TOP_FEATURES]
        
        for feat, corr in top_error_features:
            feat_idx = feature_cols.index(feat)
            feat_vals = X[:, feat_idx]
            
            # Split by median
            median_val = np.median(feat_vals)
            low_mask = feat_vals < median_val
            high_mask = feat_vals >= median_val
            
            if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
                # For classification, error_magnitude is 0/1, so mean is error rate
                # For regression, error_magnitude is absolute error, so mean is MAE
                error_low = np.mean(error_magnitude[low_mask])
                error_high = np.mean(error_magnitude[high_mask])
                
                if task_type == "classification":
                    metric_label = "error_rate"
                else:
                    metric_label = "MAE"
                
                if error_high > error_low * 1.3:
                    pattern = f"High {feat} values (>{median_val:.2f}) have {error_high/error_low:.1f}x higher {metric_label} ({metric_label}={error_high:.3f} vs {error_low:.3f})"
                    analysis["error_patterns"].append(pattern)
                elif error_low > error_high * 1.3:
                    pattern = f"Low {feat} values (<{median_val:.2f}) have {error_low/error_high:.1f}x higher {metric_label} ({metric_label}={error_low:.3f} vs {error_high:.3f})"
                    analysis["error_patterns"].append(pattern)
        
        # 4. Enhanced counterfactual examples with interactions and bounds checking
        # For worst predictions, suggest how to adjust features to fix prediction
        num_cf = min(num_counterfactuals, len(analysis["worst_predictions"]))
        for example in analysis["worst_predictions"][:num_cf]:
            idx = example["index"]
            counterfactual = {
                "original": example,
                "suggestions": []
            }
            
            # Get feature bounds for validation (assume [0, 1] scaled range, but check actual data)
            feat_bounds = {}
            for feat in feature_cols:
                feat_idx = feature_cols.index(feat)
                feat_vals = X[:, feat_idx]
                feat_bounds[feat] = (np.min(feat_vals), np.max(feat_vals))
            
            # Find top 2 error-correlated features for interaction analysis
            if top_error_features:
                # Primary feature adjustment
                feat1, corr1 = top_error_features[0]
                feat1_idx = feature_cols.index(feat1)
                current_val1 = float(X[idx, feat1_idx])
                feat1_min, feat1_max = feat_bounds[feat1]
                
                # Compute suggested value with bounds checking
                if corr1 > 0:
                    # Positive correlation: reduce feature to reduce error
                    suggested_val1 = max(feat1_min, current_val1 * 0.7)
                    if suggested_val1 < feat1_min:
                        suggested_val1 = feat1_min + 0.1 * (feat1_max - feat1_min)  # Move toward lower bound
                else:
                    # Negative correlation: increase feature to reduce error
                    suggested_val1 = min(feat1_max, current_val1 * 1.3)
                    if suggested_val1 > feat1_max:
                        suggested_val1 = feat1_max - 0.1 * (feat1_max - feat1_min)  # Move toward upper bound
                
                suggestion1 = f"Adjust {feat1} from {current_val1:.2f} to ~{suggested_val1:.2f} (within bounds [{feat1_min:.2f}, {feat1_max:.2f}])"
                counterfactual["suggestions"].append(suggestion1)
                
                # If we have a second feature, suggest interaction
                if len(top_error_features) > 1:
                    feat2, corr2 = top_error_features[1]
                    feat2_idx = feature_cols.index(feat2)
                    current_val2 = float(X[idx, feat2_idx])
                    feat2_min, feat2_max = feat_bounds[feat2]
                    
                    # Check if there's an interaction effect
                    # If both features are high/low and errors are high, suggest interaction
                    feat1_high = current_val1 > np.median(X[:, feat1_idx])
                    feat2_high = current_val2 > np.median(X[:, feat2_idx])
                    
                    # Suggest interaction term if both are on same side of median
                    if feat1_high == feat2_high:
                        interaction_suggestion = f"Consider adding interaction term {feat1} * {feat2} - both features are {'high' if feat1_high else 'low'} ({feat1}={current_val1:.2f}, {feat2}={current_val2:.2f})"
                        counterfactual["suggestions"].append(interaction_suggestion)
                    else:
                        # Features are on opposite sides - suggest conditional logic
                        conditional_suggestion = f"Consider conditional logic: when {feat1} is {'high' if feat1_high else 'low'} ({current_val1:.2f}) AND {feat2} is {'high' if feat2_high else 'low'} ({current_val2:.2f}), errors increase"
                        counterfactual["suggestions"].append(conditional_suggestion)
                
                # Add mechanistic explanation
                if task_type == "classification":
                    true_class = example.get("true", "unknown")
                    pred_class = example.get("predicted", "unknown")
                    if true_class != pred_class:
                        counterfactual["suggestions"].append(f"Mechanism fix: Add decision boundary to correctly classify {true_class} instead of {pred_class} using {feat1} and potentially {feat2 if len(top_error_features) > 1 else 'other features'}")
                else:
                    error_mag = abs(example.get("error", 0))
                    if error_mag > 0:
                        direction = "overestimate" if example.get("error", 0) < 0 else "underestimate"
                        counterfactual["suggestions"].append(f"Mechanism fix: Currently {direction}s by {error_mag:.2f} - adjust {feat1} coefficient or add {feat1} interaction to correct this")
            
            analysis["counterfactual_examples"].append(counterfactual)
    
    except Exception as e:
        logger.warning(f"Failed to analyze ML failure patterns: {e}")
    
    return analysis


def enhanced_textgrad_feedback(ml_failure_analysis: Dict[str, Any],
                                 current_loss: float,
                                 mechanism_performance: Dict[int, float],
                                 mechanism_idx: int,
                                 ml_knowledge: Dict[str, Any],
                                 mechanism_metrics: Optional[Dict[int, Dict[str, float]]] = None,
                                 target_loss: Optional[float] = None) -> str:
    """
    Build enhanced TextGrad feedback with ML failure patterns and counterfactuals.
    
    Args:
        ml_failure_analysis: Output from analyze_ml_failure_patterns()
        current_loss: Current validation loss
        mechanism_performance: Dict mapping mechanism index to performance score
        mechanism_idx: Index of mechanism being optimized
        ml_knowledge: Output from extract_ml_knowledge()
        mechanism_metrics: Optional dict mapping mechanism index to full metrics dict (R2, MAE, F1, accuracy)
        target_loss: Optional target loss to achieve (for quantitative guidance)
    
    Returns:
        Enhanced feedback string for TextGrad
    """
    # Get performance metrics
    my_performance = mechanism_performance.get(mechanism_idx, None)
    all_performances = [v for k, v in mechanism_performance.items() if v is not None]
    my_metrics = mechanism_metrics.get(mechanism_idx, {}) if mechanism_metrics else {}
    
    # Build performance context with detailed metrics
    performance_context = ""
    if my_performance is not None:
        # Start with detailed metrics if available
        metrics_str = ""
        if my_metrics:
            if "r2" in my_metrics:
                # Regression metrics
                metrics_str = f"R²: {my_metrics['r2']:.4f}, MAE: {my_metrics['mae']:.4f}, MSE: {my_metrics.get('mse', 0):.4f}"
            elif "accuracy" in my_metrics:
                # Classification metrics
                metrics_str = f"Accuracy: {my_metrics['accuracy']:.4f}, F1: {my_metrics['f1']:.4f}"
        
        if len(all_performances) > 1:
            avg_perf = np.mean(all_performances)
            max_perf = np.max(all_performances)
            min_perf = np.min(all_performances)
            min_weight_threshold = 0.3  # Minimum weight to contribute meaningfully
            
            performance_context = f"""MECHANISM PERFORMANCE:
Your Performance Score: {my_performance:.4f}
"""
            if metrics_str:
                performance_context += f"Detailed Metrics: {metrics_str}\n"
            performance_context += f"""- Average across all mechanisms: {avg_perf:.4f}
- Best mechanism: {max_perf:.4f}
- Worst mechanism: {min_perf:.4f}
- Your rank: {sorted(all_performances, reverse=True).index(my_performance) + 1} of {len(all_performances)}

"""
            # Add quantitative targets
            if my_performance < min_weight_threshold:
                improvement_needed = (min_weight_threshold - my_performance) / my_performance if my_performance > 0 else 1.0
                performance_context += f"⚠️ CRITICAL: Your mechanism performance ({my_performance:.4f}) is below the contribution threshold ({min_weight_threshold:.2f}). You need {improvement_needed:.1%} improvement to contribute meaningfully to the ensemble.\n\n"
            elif my_performance < avg_perf:
                improvement_needed = (avg_perf - my_performance) / my_performance if my_performance > 0 else 1.0
                performance_context += f"⚠️ Your mechanism is BELOW average ({my_performance:.4f} vs {avg_perf:.4f}). Target: improve by {improvement_needed:.1%} to reach average performance.\n\n"
            elif my_performance > avg_perf * 1.1:
                performance_context += f"✓ Your mechanism is ABOVE average ({my_performance:.4f} vs {avg_perf:.4f}). Refine edge cases and systematic errors.\n\n"
        else:
            performance_context = f"""MECHANISM PERFORMANCE:
Your Performance Score: {my_performance:.4f}
"""
            if metrics_str:
                performance_context += f"Detailed Metrics: {metrics_str}\n"
            performance_context += "\n"
    
    # Add quantitative improvement targets
    improvement_target_str = ""
    if target_loss is not None and current_loss > target_loss:
        improvement_needed = (current_loss - target_loss) / current_loss
        improvement_target_str = f"""
PERFORMANCE TARGET:
- Current loss: {current_loss:.4f}
- Target loss: {target_loss:.4f}
- Improvement needed: {improvement_needed:.1%} reduction
- Focus on achieving at least {target_loss:.4f} to meet target

"""
    
    feedback = f"""PERFORMANCE ANALYSIS:
Current Loss: {current_loss:.4f}
{improvement_target_str}{performance_context}"""
    
    # Add ML model knowledge
    if ml_knowledge.get("top_features"):
        feedback += f"""ML MODEL INSIGHTS:
The ML baseline identifies these as the most important features:

"""
        for i, feat in enumerate(ml_knowledge["top_features"][:MAX_TOP_FEATURES_DISPLAY]):
            imp = ml_knowledge["feature_importance"].get(feat, 0)
            feedback += f"  {i+1}. {feat} (importance: {imp:.3f})\n"
        
        if ml_knowledge.get("model_formula"):
            feedback += f"\nML baseline formula: {ml_knowledge['model_formula']}\n"
        
        feedback += "\n"
    
    # Add error patterns
    if ml_failure_analysis.get("error_patterns"):
        feedback += "CRITICAL ERROR PATTERNS (where ML fails):\n"
        for pattern in ml_failure_analysis["error_patterns"]:
            feedback += f"  ⚠️ {pattern}\n"
        feedback += "\n"
    
    # Add feature-error correlations
    if ml_failure_analysis.get("error_by_feature"):
        feedback += "FEATURES MOST CORRELATED WITH ERRORS:\n"
        sorted_features = sorted(
            ml_failure_analysis["error_by_feature"].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:MAX_TOP_FEATURES]
        
        for feat, corr in sorted_features:
            direction = "increases" if corr > 0 else "decreases"
            feedback += f"  • {feat}: {abs(corr):.3f} correlation (error {direction} with this feature)\n"
        feedback += "\n"
    
    # Add worst prediction examples
    if ml_failure_analysis.get("worst_predictions"):
        feedback += "WORST PREDICTION EXAMPLES (use these to improve):\n"
        for i, example in enumerate(ml_failure_analysis["worst_predictions"][:MAX_WORST_PREDICTIONS_DISPLAY]):
            # Format features for display
            feat_items = list(example['features'].items())[:MAX_FEATURES_IN_EXAMPLE]
            feat_str = ', '.join([f"{k}={v:.2f}" for k, v in feat_items])
            
            # For classification, use class names; for regression, use numeric values
            if isinstance(example['true'], str) and isinstance(example['predicted'], str):
                # Classification with class names
                match_str = '✓' if abs(example.get('error', 1.0)) < 0.5 else '✗ WRONG'
                feedback += f"  • {feat_str} → pred={example['predicted']}, true={example['true']}, {match_str}\n"
            else:
                # Regression or classification without class names
                feedback += f"  • {feat_str} → pred={example['predicted']:.2f}, true={example['true']:.2f}, error={example['error']:+.2f}\n"
        feedback += "\n"
    
    # Add counterfactual suggestions
    if ml_failure_analysis.get("counterfactual_examples"):
        feedback += "COUNTERFACTUAL SUGGESTIONS (how to fix errors):\n"
        # Use all available counterfactual examples (default is 5, matching worst_predictions display)
        for i, cf in enumerate(ml_failure_analysis["counterfactual_examples"]):
            if cf.get("suggestions"):
                feedback += f"  For worst example {i+1}:\n"
                for suggestion in cf["suggestions"]:
                    feedback += f"    → {suggestion}\n"
        feedback += "\n"
    
    feedback += """YOUR TASK: Build a STRONG, INDEPENDENT PREDICTIVE MECHANISM

🎯 PRIMARY GOAL: Your mechanism must work WELL ON ITS OWN as a standalone predictor.
While it may complement other mechanisms in an ensemble, prioritize independent predictive performance.
Your mechanism should achieve high R² and low MAE when evaluated alone (LLM-only evaluation).

🧠 CRITICAL: USE YOUR INTERNAL KNOWLEDGE AND REASONING
- Think deeply about the domain: What do you know about this problem that could help?
- Reason about relationships: "If class A has high feature X and class B has low feature X, then feature X discriminates"
- Use analogical reasoning: "This is similar to [domain concept], so I should model it as [mathematical form]"
- Think step-by-step: Understand the problem → Design mechanism → Optimize coefficients
- Consider domain-specific knowledge: e.g., for vehicles, think about what makes buses vs vans vs sedans distinctive
- Create meaningful intermediate variables: Model domain concepts (e.g., "boxiness", "elongation_factor") as variables

KEY STRATEGIES:
1. BUILD INDEPENDENT PREDICTIVE POWER: Focus on making your mechanism work well as a standalone predictor. 
   Include all features and interactions that improve independent performance, not just those that complement other mechanisms.
   If a feature or pattern improves standalone R²/MAE, include it even if it overlaps with other mechanisms.
   USE DOMAIN KNOWLEDGE: Think about what features should matter for this problem based on your understanding.

2. LEARN NEW COEFFICIENT VALUES: Your current formula has coefficients (like 0.783, 0.301, 0.194, etc.) that are starting values. 
   You MUST change these to new values based on performance metrics. Don't just keep the same coefficients and add interaction terms - 
   actually try different coefficient values (e.g., 0.5, 1.0, 1.2, 1.5, 2.0) to find what maximizes R² and minimizes MAE. 
   If R² is low, try increasing important feature coefficients. If MAE is high, adjust coefficients systematically.
   REASON ABOUT COEFFICIENTS: "If feature X is very important for class A, its coefficient should be larger"

3. OPTIMIZE COEFFICIENTS BASED ON PERFORMANCE METRICS: Don't just copy feature importance values - adjust coefficients based on 
   R² and MAE (for regression) or Accuracy and F1 (for classification). Feature importance (e.g., 0.392) is NOT a coefficient - 
   it's just a measure of contribution. Learn proper coefficients (typically 0.1-2.0 range) by optimizing performance metrics, not loss.

4. USE ML INSIGHTS TO IMPROVE INDEPENDENT PERFORMANCE: The error patterns identified above show where the ML model fails. 
   Use these to improve your mechanism's independent predictive power. Design your mechanism to handle these cases correctly 
   and achieve better standalone performance.
   REASON ABOUT ERRORS: "The ML model confuses class A and B when feature X is low - I should emphasize feature X for class A"

5. CRITICAL: ADD ADVANCED NONLINEAR TRANSFORMATIONS AND LEARN NONLINEAR INTERACTIONS:
   - Use INTERMEDIATE VARIABLES to capture complex nonlinear interactions: intermediate = f(feature1, feature2) where f is nonlinear
   - LEARN THE NONLINEARITY OF INTERACTIONS: interactions themselves may have saturation (feature1 * feature2 / (K + feature1 * feature2))
   - DISCOVER which features interact and HOW they interact nonlinearly (multiplicative, ratio-based, threshold-based)
   - Use saturation effects: feature / (K + feature) to capture diminishing returns
   - Use NONLINEAR interactions: not just feature1 * feature2, but feature1 * feature2 / (K + feature1 * feature2)
   - Use threshold effects: max(0, feature - threshold) or conditional logic
   - Think about how features interact: do high values of multiple features create different effects? How do they interact nonlinearly?
   - Consider saturation: do very high feature values have diminishing returns?
   - The ML model is linear - you must use nonlinearity to capture patterns it misses
   - Each class equation should use different nonlinear transformations and intermediate variables to model what makes that class unique
   - Create intermediate variables that capture complex relationships: e.g., intermediate = feature1 * feature2 / (0.2 + feature1 * feature2)
   - Optimize BOTH the form of interactions AND their coefficients - the nonlinearity of interactions matters

6. USE COUNTERFACTUALS: The suggestions above show which features to emphasize and how to adjust them to improve standalone performance.
   REASON ABOUT COUNTERFACTUALS: "If I increase feature X for this example, it should move toward class Y - does that make sense?"

7. ITERATE BASED ON PERFORMANCE METRICS: Change coefficients, add/remove terms, or modify formulas based on R² and MAE 
   (for regression) or Accuracy and F1 (for classification). Focus on improving these metrics for standalone performance, not just reducing loss.
   THINK ABOUT WHY CHANGES WORK: "This change improved performance because it better captures the relationship between features X and Y"

8. REASON ABOUT FEATURE INTERACTIONS: Don't add random interactions - think about which features should interact and why
   - "Features A and B both relate to [concept], so they should interact"
   - "If feature A is high AND feature B is low, that might indicate class X"
   - Use domain knowledge to guide interaction design

CRITICAL: Your mechanism formula should use proper mathematical coefficients (typically 0.1-2.0 range), NOT raw feature importance values. 
The initial formula uses normalized weights as a starting point - you should optimize these based on performance metrics 
(R²/MAE for regression, Accuracy/F1 for classification), NOT by copying feature importance values.

REMEMBER: Your mechanism should be a STRONG, INDEPENDENT PREDICTOR that works well on its own.
While it may complement the ML model, prioritize building standalone predictive power. Use all relevant features and patterns 
that improve independent performance. Adjust coefficients iteratively based on performance metrics (R², MAE, Accuracy, F1), not just loss.

"""
    
    return feedback

