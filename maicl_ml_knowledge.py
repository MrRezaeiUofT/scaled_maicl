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
                                  variant: int = 0) -> str:
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
    
    # Build mechanism - ALL variants focus on learning data patterns and non-linearities
    # NO LINEAR MECHANISMS - only pattern-based, descriptive, non-linear mechanisms
    if variant == 0:
        # Variant 0: Pattern-based mechanism with basic non-linearities
        if is_deepchem:
            # For DeepChem: use molecular properties, not ECFP features
            mechanism = f"""[PATTERN-BASED] This mechanism learns structural patterns from molecular properties.

================================================================================
DATA-DRIVEN PATTERN RECOGNITION
================================================================================

This mechanism identifies structural patterns in molecules by analyzing key molecular
properties and their non-linear relationships.

MOLECULAR FEATURE SPACE:

"""
            mechanism += "  • molecular_weight(SMILES): Overall molecular size - heavier molecules often have different properties\n"
            mechanism += "  • num_rings(SMILES): Aromatic/cyclic structure count - affects rigidity and binding\n"
            mechanism += "  • num_hydroxyl_groups(SMILES): Hydrogen bonding capacity - impacts solubility\n"
            
            mechanism += f"\n\nPATTERN DISCOVERY APPROACH:\n"
            mechanism += f"The mechanism looks for SATURATION effects (diminishing returns with increasing values),\n"
            mechanism += f"INTERACTION patterns (synergies between features), and THRESHOLD behaviors (step changes).\n\n"
            
            mechanism += f"FORMULA: ŷ = clip(molecular_weight(SMILES) / (100 + molecular_weight(SMILES)) + 0.3 * num_rings(SMILES) / (1 + num_rings(SMILES)) + 0.2 * num_hydroxyl_groups(SMILES), 0, 10)"
        else:
            # For non-DeepChem: Generate HIGHLY DESCRIPTIVE pattern-based mechanisms
            total_importance = sum(feature_importance.get(feat, 0.5) for feat in top_features[:MAX_TOP_FEATURES])
            if total_importance > 0:
                normalized_weights = [feature_importance.get(feat, 0.5) / total_importance for feat in top_features[:MAX_TOP_FEATURES]]
                scaled_weights = [max(0.1, min(1.0, w * 2.0)) for w in normalized_weights]
            else:
                scaled_weights = [0.5] * len(top_features[:MAX_TOP_FEATURES])
            
            mechanism = f"""[PATTERN-BASED] This mechanism learns data patterns through non-linear feature relationships.

================================================================================
DATA-DRIVEN PATTERN RECOGNITION
================================================================================

This mechanism discovers underlying patterns in the data by analyzing feature interactions,
non-linear transformations, and class-specific signatures.

FEATURE LANDSCAPE & PATTERNS:

"""
            for i, feat in enumerate(top_features[:min(6, len(top_features))]):
                imp = feature_importance.get(feat, 0.5)
                mechanism += f"  • {feat} (importance: {imp:.3f})\n"
                if "VARIANCE" in feat or "SCATTER" in feat:
                    mechanism += f"    Pattern: Likely exhibits SPREAD patterns - high values may indicate dispersed/variable shapes\n"
                elif "ASPECT_RATIO" in feat or "LENGTH" in feat:
                    mechanism += f"    Pattern: Captures ELONGATION - ratio effects may create thresholds between classes\n"
                elif "CIRCULARITY" in feat or "COMPACTNESS" in feat:
                    mechanism += f"    Pattern: Measures ROUNDNESS - interaction with other shape features reveals class boundaries\n"
                elif "RADIUS" in feat:
                    mechanism += f"    Pattern: Describes SIZE/EXTENT - may have saturation effects at extreme values\n"
                else:
                    mechanism += f"    Pattern: Contributes to multi-dimensional feature space partitioning\n"
            
            mechanism += f"\n\nNON-LINEAR PATTERN TYPES TO DISCOVER:\n"
            mechanism += f"  1. SATURATION: Features that plateau (x/(k+x)) - effectiveness diminishes at high values\n"
            mechanism += f"  2. SYNERGIES: Multiplicative interactions (x*y) - features amplify each other\n"
            mechanism += f"  3. THRESHOLDS: Step-function behaviors - sharp transitions at specific values\n"
            mechanism += f"  4. RATIOS: Relative relationships (x/y) - balance between features matters\n"
            mechanism += f"  5. TRANSFORMATIONS: Non-linear mappings (sqrt, square, exp) - reshape feature space\n\n"
            
            if task_type == "classification" and class_names and len(class_names) > 0:
                mechanism += f"CLASS-SPECIFIC PATTERN HYPOTHESIS:\n"
                mechanism += f"Each class likely has a unique signature in the feature space.\n"
                mechanism += f"The mechanism will learn distinct patterns for: {', '.join(class_names)}\n\n"
            
            mechanism += f"INITIAL PATTERN FORMULA (discover and refine through optimization):\n"
            mechanism += f"This formula encodes initial pattern hypotheses using saturation and interactions.\n"
            mechanism += f"TextGrad will refine these patterns based on observed data.\n\n"
            
            # Generate pattern-based formula with saturations and interactions
            mechanism += f"ŷ = "
            if len(top_features) >= 2:
                f1, f2 = top_features[0], top_features[1]
                w1 = scaled_weights[0] if len(scaled_weights) > 0 else 0.5
                w2 = scaled_weights[1] if len(scaled_weights) > 1 else 0.5
                mechanism += f"{w1:.3f} * {f1} / (0.3 + {f1})  # saturation pattern\n"
                mechanism += f"    + {w2:.3f} * {f2} * {f1}  # synergy pattern\n"
                
                if len(top_features) >= 3:
                    f3 = top_features[2]
                    w3 = scaled_weights[2] if len(scaled_weights) > 2 else 0.5
                    mechanism += f"    + {w3*0.6:.3f} * sqrt({f3})  # transformation pattern\n"
                    
                if len(top_features) >= 4:
                    f4 = top_features[3]
                    w4 = scaled_weights[3] if len(scaled_weights) > 3 else 0.5
                    mechanism += f"    + {w4*0.4:.3f} * {f4} / (0.5 + {f4})  # ratio pattern\n"
            else:
                mechanism += f"{top_features[0]} / (0.5 + {top_features[0]})"
        
        if task_type == "classification":
            if class_names and len(class_names) > 0:
                n_classes = len(class_names)
                class_mapping = ", ".join([f"{i}={cn}" for i, cn in enumerate(class_names)])
                # Add explicit threshold-based class mapping formula
                bin_size = (scale_max - scale_min) / n_classes
                thresholds = [scale_min + (i + 1) * bin_size for i in range(n_classes - 1)]
                if len(thresholds) == 1:
                    class_formula = f"ŷ = 0 if score < {thresholds[0]:.2f} else 1"
                elif len(thresholds) == 2:
                    class_formula = f"ŷ = 0 if score < {thresholds[0]:.2f} else (1 if score < {thresholds[1]:.2f} else 2)"
                elif len(thresholds) == 3:
                    class_formula = f"ŷ = 0 if score < {thresholds[0]:.2f} else (1 if score < {thresholds[1]:.2f} else (2 if score < {thresholds[2]:.2f} else 3))"
                else:
                    # General case
                    class_formula = f"ŷ = 0"
                    for i in range(len(thresholds)):
                        class_formula = f"({class_formula} if score < {thresholds[i]:.2f} else {i+1})"
                    class_formula = f"({class_formula} else {n_classes-1})"
                mechanism += f"\nMap score to class index: {class_formula}\nClass mapping: {class_mapping}"
            else:
                mechanism += f"\nMap score to class index: ŷ = 0 if score < {(scale_min + scale_max) / 2:.2f} else 1"
        else:
            mechanism += f"\nClipped to [{scale_min}, {scale_max}]"
    
    elif variant == 1:
        # Variant 1: Pattern-based with rich feature interactions
        if is_deepchem:
            # For DeepChem: discover molecular interaction patterns
            mechanism = f"""[INTERACTION PATTERN DISCOVERY] This mechanism learns molecular interaction patterns.

================================================================================
MOLECULAR INTERACTION PATTERN LEARNING
================================================================================

This mechanism discovers how molecular properties interact and combine to produce outcomes.
It focuses on SYNERGISTIC effects where properties amplify or dampen each other.

MOLECULAR PROPERTY INTERACTIONS:

"""
            mechanism += "  • molecular_weight(SMILES) × num_rings(SMILES): Size-structure synergy\n"
            mechanism += "    Hypothesis: Larger molecules with more rings may have distinct behavior\n\n"
            mechanism += "  • num_hydroxyl_groups(SMILES) / molecular_weight(SMILES): Polarity density\n"
            mechanism += "    Hypothesis: Concentration of polar groups matters more than absolute count\n\n"
            mechanism += "  • num_halogen(SMILES): Halogenation effect\n"
            mechanism += "    Hypothesis: Halogens may inhibit or enhance activity non-linearly\n\n"
            
            mechanism += f"INTERACTION PATTERN FORMULA:\n"
            mechanism += f"ŷ = clip(\n"
            mechanism += f"    molecular_weight(SMILES) / (100 + molecular_weight(SMILES))  # saturation\n"
            mechanism += f"    + 0.3 * num_rings(SMILES) * (1 - exp(-molecular_weight(SMILES)/200))  # synergy\n"
            mechanism += f"    + 0.2 * num_hydroxyl_groups(SMILES) / (1 + molecular_weight(SMILES)/100)  # ratio\n"
            mechanism += f"    - 0.1 * num_halogen(SMILES) / (1 + num_halogen(SMILES))  # dampening\n"
            mechanism += f", 0, 10)"
        else:
            # For non-DeepChem: discover feature interaction patterns
            mechanism = f"""[INTERACTION PATTERN DISCOVERY] This mechanism learns feature interaction patterns from data.

================================================================================
FEATURE INTERACTION PATTERN LEARNING
================================================================================

This mechanism discovers how features interact, combining effects through multiplication,
division, and conditional logic to reveal hidden patterns.

KEY FEATURE INTERACTIONS TO EXPLORE:

"""
            for i, feat in enumerate(top_features[:min(6, len(top_features))]):
                imp = feature_importance.get(feat, 0.5)
                mechanism += f"  • {feat} (importance: {imp:.3f})\n"
            
            if interactions:
                mechanism += f"\n\nDISCOVERED INTERACTION PATTERNS:\n"
                for i, (f1, f2) in enumerate(interactions[:3]):
                    mechanism += f"  {i+1}. {f1} × {f2}: Multiplicative synergy pattern\n"
                    mechanism += f"     Hypothesis: When both features are high, effect amplifies\n"
            
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
                # Add explicit threshold-based class mapping formula
                bin_size = (scale_max - scale_min) / n_classes
                thresholds = [scale_min + (i + 1) * bin_size for i in range(n_classes - 1)]
                if len(thresholds) == 1:
                    class_formula = f"ŷ = 0 if score < {thresholds[0]:.2f} else 1"
                elif len(thresholds) == 2:
                    class_formula = f"ŷ = 0 if score < {thresholds[0]:.2f} else (1 if score < {thresholds[1]:.2f} else 2)"
                elif len(thresholds) == 3:
                    class_formula = f"ŷ = 0 if score < {thresholds[0]:.2f} else (1 if score < {thresholds[1]:.2f} else (2 if score < {thresholds[2]:.2f} else 3))"
                else:
                    # General case
                    class_formula = f"ŷ = 0"
                    for i in range(len(thresholds)):
                        class_formula = f"({class_formula} if score < {thresholds[i]:.2f} else {i+1})"
                    class_formula = f"({class_formula} else {n_classes-1})"
                mechanism += f"\nMap score to class index: {class_formula}\nClass mapping: {class_mapping}"
            else:
                mechanism += f"\nMap score to class index: ŷ = 0 if score < {(scale_min + scale_max) / 2:.2f} else 1"
        else:
            mechanism += f"\nClipped to [{scale_min}, {scale_max}]"
    
    else:
        # Variant 2: Deep pattern learning with rich non-linearities
        if is_deepchem:
            # For DeepChem: deep pattern learning in molecular space
            mechanism = f"""[DEEP PATTERN LEARNING] This mechanism discovers complex non-linear patterns in molecular structures.

================================================================================
DEEP MOLECULAR PATTERN RECOGNITION
================================================================================

This mechanism learns sophisticated non-linear relationships between molecular properties,
identifying complex patterns like saturation effects, synergistic interactions, threshold
behaviors, and inhibitory relationships.

MOLECULAR PATTERN SPACE:

"""
            mechanism += "  • molecular_weight(SMILES): Overall molecular size\n"
            mechanism += "    Pattern: Saturation effects - very large molecules plateau in effectiveness\n\n"
            mechanism += "  • num_rings(SMILES): Aromatic/cyclic structures\n"
            mechanism += "    Pattern: Diminishing returns with additional rings\n\n"
            mechanism += "  • num_hydroxyl_groups(SMILES): Hydrogen bonding sites\n"
            mechanism += "    Pattern: May enhance or inhibit depending on molecular context\n\n"
            mechanism += "  • num_halogen(SMILES): Halogenation level\n"
            mechanism += "    Pattern: Non-linear dampening effect\n\n"
            
            mechanism += f"COMPLEX PATTERN FORMULA:\n"
            mechanism += f"ŷ = clip(\n"
            mechanism += f"    molecular_weight(SMILES) / (100 + molecular_weight(SMILES))  # Michaelis-Menten saturation\n"
            mechanism += f"    + 0.1 * num_rings(SMILES) / (1 + num_rings(SMILES))  # diminishing returns\n"
            mechanism += f"    + 0.15 * num_hydroxyl_groups(SMILES) * (1 - num_halogen(SMILES)/10)  # context-dependent\n"
            mechanism += f"    - 0.05 * num_halogen(SMILES)  # inhibitory effect\n"
            mechanism += f", 0, 10)"
        else:
            # For non-DeepChem: deep pattern learning with VERY RICH descriptions
            mechanism = f"""[DEEP PATTERN LEARNING] This mechanism discovers complex non-linear patterns in the data.

================================================================================
DEEP PATTERN RECOGNITION & NON-LINEAR RELATIONSHIPS
================================================================================

This mechanism learns sophisticated patterns by analyzing non-linear transformations,
class-specific signatures, feature synergies, and adaptive thresholds in the data.

FEATURE PATTERN ANALYSIS:

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
            
            # For classification: use class-specific non-linear scoring with RICH TEXTUAL DESCRIPTIONS
            if task_type == "classification" and class_names and len(class_names) > 0:
                mechanism += f"\n{'='*80}\n"
                mechanism += f"TEXTUAL MECHANISM: CLASS-SPECIFIC PATTERN RECOGNITION\n"
                mechanism += f"{'='*80}\n\n"
                
                mechanism += f"This mechanism distinguishes between {len(class_names)} vehicle classes ({', '.join(class_names)}) "
                mechanism += f"by identifying unique geometric and shape signatures for each class.\n\n"
                
                # Add feature context
                mechanism += f"FEATURE LANDSCAPE:\n"
                for i, feat in enumerate(top_features[:6]):
                    imp = feature_importance.get(feat, 0.5)
                    mechanism += f"  • {feat}: importance={imp:.3f} - "
                    if "VARIANCE" in feat or "SCATTER" in feat:
                        mechanism += "measures spread and distribution of shape\n"
                    elif "ASPECT_RATIO" in feat or "LENGTH" in feat:
                        mechanism += "captures elongation and proportions\n"
                    elif "CIRCULARITY" in feat or "COMPACTNESS" in feat:
                        mechanism += "quantifies roundness vs. angularity\n"
                    elif "RADIUS" in feat:
                        mechanism += "describes size and extent from center\n"
                    else:
                        mechanism += "contributes to shape characterization\n"
                
                mechanism += f"\nCLASS-SPECIFIC DECISION LOGIC:\n\n"
                
                # Generate rich textual descriptions for each class
                for i, cls_name in enumerate(class_names[:4]):
                    f1 = top_features[0] if len(top_features) > 0 else "feature1"
                    f2 = top_features[1] if len(top_features) > 1 else "feature2"
                    f3 = top_features[2] if len(top_features) > 2 else "feature3"
                    f4 = top_features[3] if len(top_features) > 3 else "feature4"
                    
                    mechanism += f"CLASS '{cls_name.upper()}':\n"
                    
                    if i == 0:
                        mechanism += f"  Recognition Pattern: Exhibits SATURATION behavior in {f1} - effectiveness plateaus at high values,\n"
                        mechanism += f"  suggesting this class has a characteristic threshold beyond which additional {f1} doesn't help.\n"
                        mechanism += f"  Strong SYNERGISTIC interaction between {f1} and {f2}: when both are elevated together,\n"
                        mechanism += f"  the class signature is amplified (multiplicative effect).\n"
                        mechanism += f"  \n"
                        mechanism += f"  Decision Heuristic: High {f1} (>0.5) with proportional {f2} → strong {cls_name} signal\n"
                        mechanism += f"  Secondary cue: Look for {f3} in moderate range (0.3-0.7) to confirm\n"
                        
                    elif i == 1:
                        mechanism += f"  Recognition Pattern: PRIMARY DOMINANCE of {f1} - this feature alone is highly discriminative.\n"
                        mechanism += f"  {f2} shows SATURATION DAMPENING: high values are penalized due to inverse relationship,\n"
                        mechanism += f"  creating a natural ceiling that prevents over-scoring.\n"
                        mechanism += f"  This class prefers balanced feature values rather than extremes.\n"
                        mechanism += f"  \n"
                        mechanism += f"  Decision Heuristic: Moderate-to-high {f1} (0.4-0.8) with controlled {f2} (<0.5) → {cls_name}\n"
                        mechanism += f"  Watch for {f4} as a tiebreaker when features are ambiguous\n"
                        
                    elif i == 2:
                        mechanism += f"  Recognition Pattern: SQUARE-ROOT TRANSFORMATION of {f1} suggests diminishing returns -\n"
                        mechanism += f"  early increases matter more than later ones (concave relationship).\n"
                        mechanism += f"  STRONG AMPLIFICATION via {f1}×{f2} interaction: this class emerges when BOTH features\n"
                        mechanism += f"  are simultaneously elevated, indicating a complex multi-feature signature.\n"
                        mechanism += f"  \n"
                        mechanism += f"  Decision Heuristic: {f1} > 0.6 AND {f2} > 0.5 together → {cls_name} likely\n"
                        mechanism += f"  Consider {f3} and {f4} ratio for fine-grained discrimination\n"
                        
                    else:
                        mechanism += f"  Recognition Pattern: LINEAR DOMINANCE of {f1} with RATIO-BASED modulation by {f2}.\n"
                        mechanism += f"  The {f2} ratio term (division by 0.5 + {f2}) creates a ceiling effect,\n"
                        mechanism += f"  preventing runaway scores and ensuring robustness to outliers.\n"
                        mechanism += f"  This class is characterized by MODERATE feature values across the board.\n"
                        mechanism += f"  \n"
                        mechanism += f"  Decision Heuristic: Middling {f1} (0.2-0.5) with stable {f2} → {cls_name}\n"
                        mechanism += f"  If uncertain, examine {f3}/{f2} ratio for confirmation\n"
                    
                    mechanism += f"\n"
                
                mechanism += f"PREDICTION STRATEGY:\n"
                mechanism += f"1. Compute class-specific confidence score for EACH class using their respective patterns\n"
                mechanism += f"2. Each class score captures its unique geometric signature and feature interactions\n"
                mechanism += f"3. The class with MAXIMUM confidence score wins (argmax selection)\n"
                mechanism += f"4. In case of ties, defer to {f1} as the primary discriminator\n\n"
                
                mechanism += f"ADAPTIVE THRESHOLDS:\n"
                mechanism += f"The mechanism uses soft thresholds and continuous scoring rather than hard cutoffs.\n"
                mechanism += f"This allows graceful degradation when features are noisy or ambiguous.\n"
                mechanism += f"Saturation terms (e.g., x/(a+x)) ensure numerical stability and prevent extreme scores.\n\n"
                
                # Add simplified formula for LLM execution
                mechanism += f"EXECUTABLE FORMULA (for LLM prediction engine):\n"
                mechanism += f"```python\n"
                mechanism += f"def classify_{class_names[0].lower()}({', '.join(top_features[:MAX_TOP_FEATURES])}):\n"
                mechanism += f"    import numpy as np\n"
                mechanism += f"    # Implement the textual logic described above\n"
                
                for i, cls_name in enumerate(class_names[:4]):
                    f1 = top_features[0] if len(top_features) > 0 else "feature1"
                    f2 = top_features[1] if len(top_features) > 1 else "feature2"
                    w1 = scaled_weights[0] if len(scaled_weights) > 0 else 0.5
                    w2 = scaled_weights[1] if len(scaled_weights) > 1 else 0.5
                    
                    # Use more complex non-linear transformations
                    if i == 0:
                        mechanism += f"    {cls_name.lower()}_score = {w1:.3f} * {f1} / (0.3 + {f1}) + {w2*0.8:.3f} * {f2} * {f1}\n"
                    elif i == 1:
                        mechanism += f"    {cls_name.lower()}_score = {w1*1.2:.3f} * {f1} + {w2:.3f} * {f2} / (0.4 + {f2})\n"
                    elif i == 2:
                        mechanism += f"    {cls_name.lower()}_score = {w1*0.8:.3f} * np.sqrt(max(0, {f1})) + {w2*1.1:.3f} * {f2} * {f1}\n"
                    else:
                        mechanism += f"    {cls_name.lower()}_score = {w1*0.6:.3f} * {f1} + {w2*0.9:.3f} * {f2} / (0.5 + {f2})\n"
                
                mechanism += f"    scores = [{', '.join([cn.lower() + '_score' for cn in class_names[:4]])}]\n"
                mechanism += f"    return scores.index(max(scores))\n"
                mechanism += f"```\n"
            else:
                # For regression: use non-linear transformations
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
            
            mechanism += f"\n\n{'='*80}\n"
            mechanism += f"OPTIMIZATION GUIDANCE:\n"
            mechanism += f"{'='*80}\n"
            mechanism += f"The textual descriptions above contain the TRUE mechanism logic.\n"
            mechanism += f"The Python formula is a STARTING POINT that implements these textual patterns.\n\n"
            mechanism += f"CRITICAL ADJUSTMENTS during TextGrad optimization:\n"
            mechanism += f"  1. Modify saturation parameters (denominators like 0.3, 0.4) to control feature sensitivity\n"
            mechanism += f"  2. Adjust coefficient weights to reflect actual class discrimination patterns\n"
            mechanism += f"  3. Add/remove feature interactions based on which combinations are truly predictive\n"
            mechanism += f"  4. Introduce conditional logic (if-then rules) when appropriate\n"
            mechanism += f"  5. Use feature ratios (e.g., {top_features[0]}/{top_features[1]}) to capture relative relationships\n\n"
            mechanism += f"REMEMBER: Feature importance values are NOT coefficients!\n"
            mechanism += f"Importance tells you WHICH features matter, not HOW MUCH weight to give them.\n"
            mechanism += f"Learn proper weights (typically 0.1-2.0) through iterative optimization.\n\n"
            mechanism += f"TEXTUAL ENRICHMENT:\n"
            mechanism += f"Prefer adding rich conditional descriptions over complex formulas.\n"
            mechanism += f"Example: 'When {top_features[0]} exceeds 0.7 AND {top_features[1]} is below 0.3, this strongly\n"
            mechanism += f"indicates class X due to the characteristic shape signature.'\n"
        
        if task_type == "classification":
            if class_names and len(class_names) > 0:
                n_classes = len(class_names)
                class_mapping = ", ".join([f"{i}={cn}" for i, cn in enumerate(class_names)])
                # For variant 2 (non-linear), the class-specific scoring is already included above
                # Just add the class mapping reference
                mechanism += f"\nClass mapping: {class_mapping}"
                mechanism += f"\n\nThe mechanism uses class-specific non-linear scoring (already defined above). Each class has its own formula with saturation effects and interactions, allowing for complex non-linear decision boundaries."
            else:
                mechanism += f"\nMap score to class index: ŷ = 0 if score < {(scale_min + scale_max) / 2:.2f} else 1"
        else:
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
            feat_str = ', '.join([f"{k}={v:.4f}" for k, v in feat_items])
            
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
    
    feedback += """YOUR TASK: Use the above ML insights and failure patterns to improve your mechanism.

KEY STRATEGIES:
1. USE NON-LINEAR FORMS: Avoid simple linear combinations like "score = a*feature1 + b*feature2". Instead, use:
   - Saturation effects: feature / (K + feature)
   - Multiplicative interactions: feature1 * feature2
   - Threshold effects: max(0, feature - threshold)
   - Conditional logic: if feature > threshold then ... else ...
   - Non-linear transformations: sqrt(feature), log(1 + feature), feature^2
   - Class-specific scoring: Different non-linear formulas for each class (classification)

2. TEXTUAL DESCRIPTIONS WHEN FORMULAS ARE TOO COMPLEX: If the mechanism is too complex for a simple formula, use a rich textual description instead. The LLM can interpret detailed textual mechanisms better than overly complex formulas. For example:
   "When COMPACTNESS is high (>0.7) AND CIRCULARITY is moderate (0.4-0.6), the vehicle is likely a bus. When COMPACTNESS is low (<0.3) AND DISTANCE_CIRCULARITY is high (>0.8), it's likely a van. The interaction between SCALED_VARIANCE_MINOR and MAX.LENGTH_ASPECT_RATIO creates a non-linear decision boundary..."

3. LEARN NEW COEFFICIENT VALUES: Your current formula has coefficients (like 0.783, 0.301, 0.194, etc.) that are starting values. You MUST change these to new values based on performance metrics. Don't just keep the same coefficients and add interaction terms - actually try different coefficient values (e.g., 0.5, 1.0, 1.2, 1.5, 2.0) to find what maximizes R² and minimizes MAE. If R² is low, try increasing important feature coefficients. If MAE is high, adjust coefficients systematically.

4. OPTIMIZE COEFFICIENTS BASED ON PERFORMANCE METRICS: Don't just copy feature importance values - adjust coefficients based on R² and MAE (for regression) or Accuracy and F1 (for classification). Feature importance (e.g., 0.392) is NOT a coefficient - it's just a measure of contribution. Learn proper coefficients (typically 0.1-2.0 range) by optimizing performance metrics, not loss.

5. FIX ML WEAKNESSES: Target the error patterns identified above - these show where the ML model fails

6. USE COUNTERFACTUALS: The suggestions above show which features to emphasize and how to adjust them

7. ITERATE BASED ON PERFORMANCE METRICS: Change coefficients, add/remove terms, or modify formulas based on R² and MAE (for regression) or Accuracy and F1 (for classification). Focus on improving these metrics, not just reducing loss.

CRITICAL: Your mechanism formula should use proper mathematical coefficients (typically 0.1-2.0 range), NOT raw feature importance values. The initial formula uses normalized weights as a starting point - you should optimize these based on performance metrics (R²/MAE for regression, Accuracy/F1 for classification), NOT by copying feature importance values.

REMEMBER: Your mechanism should COMPLEMENT the ML model, not replicate it.
Focus on the error patterns and use different mathematical forms. If formulas become too complex, use rich textual descriptions that the LLM can interpret. Adjust coefficients iteratively based on performance metrics (R², MAE, Accuracy, F1), not just loss.

"""
    
    return feedback

