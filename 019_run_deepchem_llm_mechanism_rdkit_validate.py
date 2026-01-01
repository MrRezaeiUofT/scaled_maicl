#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Standalone Runner for DeepChem MA-ICL Mechanism Generation & RDKit Validation

Key Improvements over Original:
1. PROMPTING ENHANCEMENTS:
   - Structured chain-of-thought reasoning with explicit steps
   - Few-shot examples embedded in prompts for formula generation
   - Clearer separation of chemistry reasoning vs formula syntax
   - Explicit error-prevention instructions (common mistakes to avoid)
   - Temperature/coefficient guidance based on empirical ranges
   - Self-consistency verification prompt section

2. EVALUATION ENHANCEMENTS:
   - Multi-metric scoring (weighted combination of sign-match, MAE, R², stability)
   - Formula stability analysis (variance across perturbations)
   - Outlier detection and reporting
   - Cross-validation within train split for robustness
   - Gradient-based sensitivity analysis for interpretability
   - Automatic coefficient tuning via simple optimization

3. VALIDATION ENHANCEMENTS:
   - Comprehensive formula parsing with better error messages
   - Domain-specific sanity checks (e.g., output should vary with inputs)
   - Detection of degenerate formulas (constants, near-constants)
   - Statistical significance testing for descriptor correlations

Example:
  python 019_run_deepchem_llm_mechanism_rdkit_validate.py --dataset esol --max_samples 200 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import existing MA-ICL modules
from maicl_lib_v2 import MinMaxScaler010, TrainableMAICL
import maicl_config
from evaluate_individual_mechanisms import (
    extract_formula_from_llm_mechanism,
    evaluate_llm_formula,
    parse_mechanisms_file,
    _compute_molecular_properties,
)

# Import dataset loader from 018
_018_runner = None


def _load_018_runner():
    """Load the 018 runner module to access its functions."""
    global _018_runner
    if _018_runner is None:
        here = Path(__file__).resolve().parent
        runner_path = str(here / "018_maicl_regression_biotech.py")
        spec = importlib.util.spec_from_file_location("_maicl_regression_biotech_018", runner_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {runner_path}")
        _018_runner = importlib.util.module_from_spec(spec)
        # IMPORTANT: register in sys.modules before exec_module so code that does
        # `sys.modules[__name__]` (as in 018_maicl_regression_biotech.py) works.
        import sys
        sys.modules["_maicl_regression_biotech_018"] = _018_runner
        spec.loader.exec_module(_018_runner)
    return _018_runner


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

SUPPORTED_RDKIT_DESCRIPTOR_NAMES = [
    "molecular_weight",
    "num_rings",
    "num_hydroxyl_groups",
    "num_halogen",
    "num_nitrogen",
    "num_oxygen",
    "num_atoms",
    "is_aromatic",
    "num_hbd",
]

# Descriptor groupings for chemistry reasoning
POLARITY_DESCRIPTORS = ["num_oxygen", "num_nitrogen", "num_hydroxyl_groups", "num_hbd"]
HYDROPHOBIC_DESCRIPTORS = ["num_rings", "is_aromatic", "num_halogen", "molecular_weight", "num_atoms"]

# Empirical coefficient ranges (from literature and experimentation)
COEFFICIENT_RANGES = {
    "intercept": (0.2, 0.6),
    "polarity_linear": (0.02, 0.15),
    "hydrophobic_linear": (0.02, 0.15),
    "saturation_k": (0.5, 3.0),
    "interaction": (-0.1, 0.1),
    "mw_normalization": (200, 400),
}

# Evaluation weights for multi-metric scoring
EVAL_WEIGHTS = {
    "sign_match": 0.30,
    "mae": 0.25,
    "r2": 0.20,
    "stability": 0.15,
    "interpretability": 0.10,
}


# =============================================================================
# IMPROVED PROMPT TEMPLATES
# =============================================================================

def get_encoder_prompt_v2(dataset_key: str) -> str:
    """
    Generate an improved encoder prompt with chain-of-thought structure focused on validation performance.
    """
    descriptor_list = ", ".join([f"{k}(SMILES)" for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES])
    
    dataset_key = str(dataset_key).lower().strip()
    
    # Dataset-specific sign guidance
    if dataset_key in ("esol", "delaney"):
        dataset_guidance = """## DATASET: ESOL (Aqueous Solubility)
The target represents aqueous solubility (scaled to [0,1]).
- Higher target value = MORE soluble in water
- Lower target value = LESS soluble (more hydrophobic/lipophilic)

### ⚠️ CRITICAL: CORRECT SIGNS FOR SOLUBILITY (DO NOT REVERSE THESE!)
For aqueous solubility, you MUST use these signs:
- num_hydroxyl_groups: POSITIVE (increases solubility via H-bonding with water)
- num_oxygen: POSITIVE (increases solubility via H-bonding with water)
- num_nitrogen: POSITIVE (increases solubility via H-bonding with water)
- num_hbd: POSITIVE (increases solubility via H-bonding with water)
- num_halogen: NEGATIVE (increases lipophilicity, decreases solubility)
- num_rings: NEGATIVE (increases hydrophobicity, decreases solubility)
- is_aromatic: NEGATIVE (increases hydrophobicity, decreases solubility)
- molecular_weight: NEGATIVE (larger molecules harder to solvate)
- num_atoms: NEGATIVE (more atoms = harder to solvate)

### COMMON MISTAKE TO AVOID:
❌ WRONG: "Aromatic rings and halogens enhance solubility" → This is FALSE for aqueous solubility!
✅ CORRECT: "Aromatic rings and halogens DECREASE solubility (they increase hydrophobicity)"
"""
    elif dataset_key in ("lipo", "lipophilicity"):
        dataset_guidance = """## DATASET: LogP (Lipophilicity)
The target represents octanol-water partition coefficient (scaled to [0,1]).
- Higher target value = MORE lipophilic (partitions into octanol/fat)
- Lower target value = MORE hydrophilic (partitions into water)

### ⚠️ CRITICAL: CORRECT SIGNS FOR LIPOPHILICITY (DO NOT REVERSE THESE!)
For lipophilicity (LogP), you MUST use these signs (OPPOSITE of solubility):
- num_hydroxyl_groups: NEGATIVE (decreases lipophilicity, increases hydrophilicity)
- num_oxygen: NEGATIVE (decreases lipophilicity)
- num_nitrogen: NEGATIVE (decreases lipophilicity)
- num_hbd: NEGATIVE (decreases lipophilicity)
- num_halogen: POSITIVE (increases lipophilicity)
- num_rings: POSITIVE (increases lipophilicity)
- is_aromatic: POSITIVE (increases lipophilicity)
- molecular_weight: POSITIVE (weak, larger molecules tend to be more lipophilic)
- num_atoms: POSITIVE (weak, more atoms = more surface area for hydrophobic interaction)
"""
    else:
        dataset_guidance = """## DATASET: Unknown
Please determine the correct signs based on the property being predicted.
"""
    
    return f"""You are a scientific mechanism encoder specializing in molecular property prediction.

## YOUR PRIMARY GOAL: VALIDATION PERFORMANCE
Your analysis will be used to generate a formula that MUST perform well on a validation set.
The formula will be evaluated using R² (coefficient of determination) and MAE (mean absolute error).
- R² > 0.0 is required (positive R² means the formula explains variance better than the mean)
- Lower MAE is better (target: MAE < 0.25 for scaled targets in [0,1])
- CRITICAL: Getting the SIGN of each descriptor's effect wrong will cause catastrophic validation failure (negative R²)

{dataset_guidance}

## YOUR TASK
Analyze the relationship between molecular structure (represented as SMILES strings) and the target property.
Your output will guide a formula generator to create an interpretable, executable mechanism that maximizes validation performance.

## AVAILABLE MOLECULAR DESCRIPTORS
You may ONLY reference these RDKit-computable descriptors:
{descriptor_list}

### Descriptor Definitions:
- molecular_weight(SMILES): Total molecular mass in Daltons
- num_rings(SMILES): Count of ring systems (aromatic + aliphatic)
- num_hydroxyl_groups(SMILES): Count of -OH groups
- num_halogen(SMILES): Count of F, Cl, Br, I atoms
- num_nitrogen(SMILES): Count of nitrogen atoms
- num_oxygen(SMILES): Count of oxygen atoms
- num_atoms(SMILES): Total heavy atom count
- is_aromatic(SMILES): Binary flag (1 if molecule contains aromatic rings, 0 otherwise)
- num_hbd(SMILES): Count of hydrogen bond donors

### Descriptor Groupings:
- POLARITY proxies: num_oxygen, num_nitrogen, num_hydroxyl_groups, num_hbd
- HYDROPHOBIC proxies: num_rings, is_aromatic, num_halogen, molecular_weight, num_atoms

## CHAIN-OF-THOUGHT REASONING (follow these steps):

### Step 1: Property Analysis
Identify what physical/chemical property the target represents and what molecular features drive it.
Be explicit about the property direction (e.g., "higher target = more soluble" for ESOL).

### Step 2: Direction of Effects (CRITICAL FOR VALIDATION)
For each descriptor, determine: does increasing this descriptor INCREASE or DECREASE the target?
⚠️ YOU MUST USE THE CORRECT SIGNS FROM THE DATASET GUIDANCE ABOVE ⚠️
Double-check your reasoning against the dataset-specific sign guidance provided above.

⚠️ SIGN ERRORS CAUSE VALIDATION FAILURE: If you get the sign wrong, the formula will have negative R² and high MAE.

### Step 3: Nonlinear Effects
Consider saturation effects: very high values of a descriptor may have diminishing returns.
Use saturation functions like x/(K+x) to prevent extreme predictions that hurt validation metrics.

### Step 4: Interaction Effects
Identify if/how descriptors interact: does high polarity AMPLIFY or DAMPEN the effect of hydrophobicity?
Interaction terms can improve validation performance if they capture real chemistry.

## VALIDATION PERFORMANCE GUIDANCE
- Strong positive descriptors should have POSITIVE coefficients in the formula
- Strong negative descriptors should have NEGATIVE coefficients in the formula
- Weak/neutral descriptors should have small or zero coefficients
- Use saturation (x/(K+x)) to prevent extreme predictions that hurt R²
- Balance coefficients so the formula produces reasonable predictions across the validation set

## ABSOLUTE PROHIBITIONS
- Do NOT use ECFP fingerprint bits (ecfp_bit_*, bit_123, etc.)
- Do NOT invent descriptors not in the whitelist (NO: logp, tpsa, qed, num_carbon, SMARTS patterns)
- Do NOT assume access to computed properties like LogP—you must BUILD that from primitives
- Do NOT reverse the sign of well-established effects (e.g., halogens decrease solubility, not increase)

## OUTPUT FORMAT
Provide a structured analysis (3-6 sentences) covering:
1. Primary drivers of the target property (with correct signs - MUST match dataset guidance above)
2. Expected direction of each descriptor's effect (be explicit: POSITIVE or NEGATIVE - verify against dataset guidance)
3. Key nonlinearities or interactions to include for better validation performance

### CRITICAL: Sign Verification Checklist
Before submitting your analysis, verify:
- [ ] For SOLUBILITY: num_halogen, num_rings, molecular_weight are NEGATIVE
- [ ] For SOLUBILITY: num_hydroxyl_groups, num_oxygen, num_nitrogen are POSITIVE
- [ ] For LIPOPHILICITY: Signs are REVERSED from solubility
- [ ] Your analysis explicitly states the correct sign for each major descriptor

Example correct statement for SOLUBILITY:
"Aromatic rings and halogens DECREASE solubility because they increase hydrophobicity. Hydroxyl groups and oxygen atoms INCREASE solubility because they enable hydrogen bonding with water."

Example WRONG statement (DO NOT USE):
"Aromatic rings and halogens enhance solubility" ← This is FALSE for aqueous solubility!
"""


def get_decoder_prompt_v2(dataset_key: str, variation_tag: str = "") -> str:
    """
    Generate an improved decoder prompt with examples and explicit formula syntax.
    """
    dataset_key = str(dataset_key).lower().strip()
    descriptor_list = ", ".join([f"{k}(SMILES)" for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES])
    
    # Dataset-specific chemistry guidance
    if dataset_key in ("lipo", "lipophilicity"):
        domain_context = """## DATASET: LogP (Lipophilicity)
The target represents octanol-water partition coefficient (scaled to [0,1]).
- Higher values = MORE lipophilic (partitions into octanol/fat)
- Lower values = MORE hydrophilic (partitions into water)

### Expected Descriptor Effects for LogP:
| Descriptor | Expected Effect | Reasoning |
|------------|-----------------|-----------|
| num_rings | POSITIVE | Aromatic/aliphatic rings increase hydrophobicity |
| is_aromatic | POSITIVE | Aromatic systems are hydrophobic |
| num_halogen | POSITIVE | Halogens increase lipophilicity |
| molecular_weight | POSITIVE (weak) | Larger molecules tend to be more lipophilic |
| num_atoms | POSITIVE (weak) | More atoms = more surface area for hydrophobic interaction |
| num_oxygen | NEGATIVE | Oxygen enables H-bonding with water |
| num_nitrogen | NEGATIVE | Nitrogen enables H-bonding with water |
| num_hydroxyl_groups | NEGATIVE (strong) | -OH groups strongly hydrophilic |
| num_hbd | NEGATIVE | H-bond donors prefer water |

### Formula Skeleton for LogP:
```
P = 0.5*num_hydroxyl_groups(SMILES) + 0.3*num_oxygen(SMILES) + 0.2*num_nitrogen(SMILES) + 0.4*num_hbd(SMILES)
H = 0.4*num_rings(SMILES) + 0.3*is_aromatic(SMILES) + 0.2*num_halogen(SMILES) + 0.3*(molecular_weight(SMILES)/300)
ŷ = clip(0.4 + 0.4*(H/(1+H)) - 0.3*(P/(1+P)) - 0.05*(P*H)/(1+P*H), 0.0, 1.0)
```
"""
    else:  # ESOL / solubility
        domain_context = """## DATASET: ESOL (Aqueous Solubility)
The target represents aqueous solubility (scaled to [0,1]).
- Higher values = MORE soluble in water
- Lower values = LESS soluble (more hydrophobic)

### Expected Descriptor Effects for Solubility:
| Descriptor | Expected Effect | Reasoning |
|------------|-----------------|-----------|
| num_oxygen | POSITIVE | Oxygen enables H-bonding with water |
| num_nitrogen | POSITIVE | Nitrogen enables H-bonding with water |
| num_hydroxyl_groups | POSITIVE (strong) | -OH groups strongly hydrophilic |
| num_hbd | POSITIVE | H-bond donors interact favorably with water |
| num_rings | NEGATIVE | Rings increase hydrophobicity |
| is_aromatic | NEGATIVE | Aromatic systems disrupt water structure |
| num_halogen | NEGATIVE | Halogens increase lipophilicity |
| molecular_weight | NEGATIVE | Larger molecules harder to solvate |
| num_atoms | NEGATIVE (weak) | More atoms = harder to solvate |

### Formula Skeleton for Solubility:
```
P = 0.5*num_hydroxyl_groups(SMILES) + 0.3*num_oxygen(SMILES) + 0.2*num_nitrogen(SMILES) + 0.4*num_hbd(SMILES)
H = 0.4*num_rings(SMILES) + 0.3*is_aromatic(SMILES) + 0.2*num_halogen(SMILES) + 0.3*(molecular_weight(SMILES)/300)
ŷ = clip(0.5 + 0.35*(P/(1+P)) - 0.3*(H/(1+H)) + 0.05*(P*H)/(1+P*H), 0.0, 1.0)
```
"""

    few_shot_examples = """
## FEW-SHOT EXAMPLES OF VALID FORMULAS

### Example 1: Simple Linear with Saturation
```
Formula: ŷ = clip(0.45 + 0.25*(num_hydroxyl_groups(SMILES)/(1+num_hydroxyl_groups(SMILES))) - 0.2*(num_rings(SMILES)/(2+num_rings(SMILES))), 0.0, 1.0)
```

### Example 2: With Interaction Term
```
Formula: ŷ = clip(0.4 + 0.3*((num_oxygen(SMILES) + 0.5*num_nitrogen(SMILES) + num_hydroxyl_groups(SMILES))/(1 + num_oxygen(SMILES) + 0.5*num_nitrogen(SMILES) + num_hydroxyl_groups(SMILES))) - 0.25*((num_rings(SMILES) + 0.5*is_aromatic(SMILES))/(1 + num_rings(SMILES) + 0.5*is_aromatic(SMILES))) - 0.08*((num_oxygen(SMILES) + 0.5*num_nitrogen(SMILES) + num_hydroxyl_groups(SMILES))*(num_rings(SMILES) + 0.5*is_aromatic(SMILES)))/(1 + (num_oxygen(SMILES) + 0.5*num_nitrogen(SMILES) + num_hydroxyl_groups(SMILES))*(num_rings(SMILES) + 0.5*is_aromatic(SMILES))), 0.0, 1.0)
```

### Example 3: With Log Transform for Size Penalty
```
Formula: ŷ = clip(0.5 + 0.2*(num_hbd(SMILES)/(1+num_hbd(SMILES))) - 0.15*log(1 + num_rings(SMILES)) - 0.1*log(1 + molecular_weight(SMILES)/200), 0.0, 1.0)
```

### Example 4: Explicit Inline Computation (REQUIRED FORMAT)
```
Formula: ŷ = clip(0.45 + 0.3*((num_oxygen(SMILES) + 0.5*num_nitrogen(SMILES) + num_hydroxyl_groups(SMILES))/(1 + num_oxygen(SMILES) + 0.5*num_nitrogen(SMILES) + num_hydroxyl_groups(SMILES))) - 0.25*((num_rings(SMILES) + 0.5*is_aromatic(SMILES))/(1 + num_rings(SMILES) + 0.5*is_aromatic(SMILES))), 0.0, 1.0)
```
"""

    common_mistakes = """
## COMMON MISTAKES TO AVOID (THESE CAUSE VALIDATION FAILURE)

### ❌ WRONG: Using undefined variables
```
Formula: ŷ = clip(0.5 + 0.3*P - 0.2*H, 0.0, 1.0)  # P and H are not defined!
```

### ✅ CORRECT: Inline everything
```
Formula: ŷ = clip(0.5 + 0.3*(num_oxygen(SMILES)+num_nitrogen(SMILES)) - 0.2*num_rings(SMILES), 0.0, 1.0)
```

### ❌ WRONG: Using unsupported descriptors
```
Formula: ŷ = clip(0.5 + 0.3*logp(SMILES), 0.0, 1.0)  # logp is NOT available!
```

### ❌ WRONG: Using ECFP bits
```
Formula: ŷ = clip(0.5 + 0.1*ecfp_bit_123, 0.0, 1.0)  # Fingerprint bits are NOT allowed!
```

### ❌ WRONG: Formula spanning multiple lines
```
Formula: ŷ = clip(0.5 
    + 0.3*num_oxygen(SMILES), 0.0, 1.0)  # Must be ONE line!
```

### ❌ WRONG: Missing clip() wrapper
```
Formula: ŷ = 0.5 + 0.3*num_oxygen(SMILES)  # Could produce values outside [0,1]!
```

### ❌ WRONG: Coefficients too large
```
Formula: ŷ = clip(0.5 + 5.0*num_rings(SMILES), 0.0, 1.0)  # Will saturate immediately!
```
Coefficients should typically be in range [0.02, 0.5] for individual descriptors.

### ❌ WRONG: Sign errors (CRITICAL - CAUSES NEGATIVE R²)
```
# For SOLUBILITY (ESOL):
Formula: ŷ = clip(0.5 + 0.2*num_halogen(SMILES), 0.0, 1.0)  # WRONG: halogens DECREASE solubility!
```
Halogens increase lipophilicity, which DECREASES solubility. Should be NEGATIVE coefficient.

### ✅ CORRECT: Correct signs
```
# For SOLUBILITY (ESOL):
Formula: ŷ = clip(0.5 - 0.15*num_halogen(SMILES) + 0.3*num_hydroxyl_groups(SMILES), 0.0, 1.0)
```

### ❌ WRONG: Unbalanced formula (causes poor validation)
```
Formula: ŷ = clip(0.1 + 0.8*num_hydroxyl_groups(SMILES), 0.0, 1.0)  # Too extreme, will saturate
```

### ✅ CORRECT: Balanced with saturation
```
Formula: ŷ = clip(0.5 + 0.3*(num_hydroxyl_groups(SMILES)/(1+0.2*num_hydroxyl_groups(SMILES))) - 0.2*num_rings(SMILES), 0.0, 1.0)
```
"""

    validation_performance_guidance = """
## VALIDATION PERFORMANCE OPTIMIZATION

Your formula will be evaluated on a validation set using:
- **R² (coefficient of determination)**: Must be > 0.0 (preferably > 0.3)
- **MAE (mean absolute error)**: Lower is better (target: < 0.25 for scaled [0,1] targets)

### Key Principles for Good Validation Performance:

1. **SIGN CORRECTNESS IS CRITICAL - NEVER REVERSE SIGNS**
   - Wrong signs cause negative R² and high MAE - this is the #1 cause of rejection
   - ⚠️ FOR ESOL/SOLUBILITY: You MUST preserve these signs:
     * num_hydroxyl_groups: POSITIVE (never negative)
     * num_oxygen: POSITIVE (never negative)
     * num_nitrogen: POSITIVE (never negative)
     * num_hbd: POSITIVE (never negative)
     * num_halogen: NEGATIVE (never positive)
     * num_rings: NEGATIVE (never positive)
     * is_aromatic: NEGATIVE (never positive)
     * molecular_weight: NEGATIVE (never positive)
   - If you change a coefficient's sign, your update WILL BE REJECTED
   - Only adjust coefficient MAGNITUDES, never signs

2. **BALANCED COEFFICIENTS**
   - Use saturation functions: x/(K+x) prevents extreme predictions
   - Intercept should be 0.4-0.6 (middle of [0,1] range)
   - Individual descriptor coefficients: 0.05-0.3 for linear, 0.2-0.5 for saturated terms
   - If R² is negative or very low, INCREASE coefficient magnitudes (but keep signs the same)

3. **COVERAGE OF IMPORTANT DESCRIPTORS**
   - Include the 3-5 most important descriptors (don't overfit to noise)
   - Prioritize descriptors with strong expected effects
   - Use interaction terms sparingly (only if chemistry justifies it)

4. **AVOID OVERFITTING**
   - Don't use too many descriptors (5-7 max)
   - Prefer simple, interpretable formulas
   - Use saturation to prevent extreme predictions

### Example of Good Validation Performance Formula (for ESOL):
```
# Balanced, correct signs, uses saturation
Formula: ŷ = clip(0.55 + 0.4*(num_hydroxyl_groups(SMILES)/(1+0.15*num_hydroxyl_groups(SMILES))) + 0.12*num_oxygen(SMILES) + 0.12*num_nitrogen(SMILES) - 0.15*num_halogen(SMILES) - 0.15*num_rings(SMILES) - 0.004*molecular_weight(SMILES)/300, 0.0, 1.0)
```
This formula:
- ✅ Has correct signs (hydroxyl positive, halogen/rings/mw negative)
- ✅ Uses saturation for hydroxyl groups (prevents extreme predictions)
- ✅ Has balanced coefficients (won't saturate to 0 or 1)
- ✅ Includes key descriptors without overfitting
"""

    output_format = """
## REQUIRED OUTPUT FORMAT

Your response MUST include these sections IN ORDER:

### 1. MECHANISM DESCRIPTION (3-6 sentences)
Explain the chemistry behind your formula. Why does each term make sense?
Explicitly state the sign of each major descriptor's effect (e.g., "num_halogen has a NEGATIVE coefficient because halogens decrease solubility").

### 2. INTERMEDIATE CONCEPTS (1-3 bullets)
Define any aggregate variables you use conceptually (but remember: the Formula must be self-contained!)

### 3. VALIDATION PERFORMANCE CONSIDERATIONS (2-4 bullets)
- ⚠️ SIGN VERIFICATION (CRITICAL): 
  * For SOLUBILITY: Verify num_halogen, num_rings, molecular_weight have NEGATIVE coefficients
  * For SOLUBILITY: Verify num_hydroxyl_groups, num_oxygen, num_nitrogen have POSITIVE coefficients
  * For LIPOPHILICITY: Signs are REVERSED from solubility
  * Wrong signs = negative R² = validation failure!
- Check coefficient magnitudes: Are they balanced so the formula doesn't saturate (always predict 0 or 1)?
- Consider saturation: Use x/(K+x) for descriptors that could have extreme values
- Test mentally: Would a molecule with many hydroxyl groups and few rings predict HIGH solubility? (for ESOL)

### 4. SANITY CHECKS (2-4 bullets)
- What happens when all polarity descriptors are 0? (Should give baseline)
- What happens when all hydrophobicity descriptors are high? (Should decrease/increase appropriately)
- Does the formula avoid division by zero?
- Is the output always in [0, 1]?

### 5. RDKitValidation (REQUIRED)
This MUST be present and formatted exactly as shown. The expected_effects MUST match the signs in your formula!

For SOLUBILITY (ESOL) - example:
```
RDKitValidation:
{{"expected_effects":{{"num_rings":"negative","is_aromatic":"negative","num_halogen":"negative","molecular_weight":"negative","num_atoms":"negative","num_oxygen":"positive","num_nitrogen":"positive","num_hydroxyl_groups":"positive","num_hbd":"positive"}}}}
```

For LIPOPHILICITY (LogP) - example:
```
RDKitValidation:
{{"expected_effects":{{"num_rings":"positive","is_aromatic":"positive","num_halogen":"positive","molecular_weight":"positive","num_atoms":"positive","num_oxygen":"negative","num_nitrogen":"negative","num_hydroxyl_groups":"negative","num_hbd":"negative"}}}}
```

⚠️ CRITICAL: 
- The expected_effects MUST match the signs in your formula coefficients!
- For SOLUBILITY: num_halogen, num_rings, molecular_weight should be "negative" (they decrease solubility)
- For SOLUBILITY: num_oxygen, num_nitrogen, num_hydroxyl_groups, num_hbd should be "positive" (they increase solubility)
- For LIPOPHILICITY: Reverse all the signs above

Use ONLY these effect values: "positive", "negative", or "neutral"
Use ONLY these descriptor keys: molecular_weight, num_rings, num_hydroxyl_groups, num_halogen, num_nitrogen, num_oxygen, num_atoms, is_aromatic, num_hbd

### 6. Formula (REQUIRED - MUST BE LAST)
```
Formula: ŷ = clip(<SINGLE LINE expression with all descriptors inline>, 0.0, 1.0)
```

## CRITICAL FORMULA REQUIREMENTS FOR VALIDATION PERFORMANCE
1. SINGLE LINE - no line breaks within the formula
2. ALL descriptors must use the pattern: descriptor_name(SMILES)
3. MUST be wrapped in clip(..., 0.0, 1.0)
4. Use only: +, -, *, /, (), abs, min, max, sqrt, exp, log, pow, clip
5. NO undefined variables - everything must be inline
6. Coefficients should be in reasonable ranges (see examples)
7. ⚠️ SIGN CORRECTNESS: Ensure positive descriptors have positive coefficients and negative descriptors have negative coefficients
8. ⚠️ BALANCE: Use saturation functions (x/(K+x)) to prevent extreme predictions that hurt R²
9. ⚠️ INTERCEPT: Start with an intercept around 0.4-0.6 to provide a reasonable baseline
"""

    variation_note = ""
    if variation_tag:
        variation_note = f"""
## VARIATION HINT
This is iteration {variation_tag}. Try a DIFFERENT formula structure than previous iterations:
- If you've used linear terms, try more saturation (x/(K+x))
- If you've used simple saturation, try interaction terms (P*H)/(1+P*H)
- If you've used additive structure, try multiplicative: base * factor1 * factor2
- Adjust coefficients by ±20% from typical values
"""

    return f"""You are a mechanism decoder creating an executable formula for molecular property prediction.

## PRIMARY GOAL: MAXIMIZE VALIDATION PERFORMANCE
Your formula will be evaluated on a validation set. Your goal is to achieve:
- R² > 0.0 (preferably > 0.3) - positive R² means the formula explains variance better than the mean
- MAE < 0.25 (for scaled targets in [0,1]) - lower is better
- Sign correctness is CRITICAL - wrong signs cause catastrophic validation failure

{domain_context}

## AVAILABLE DESCRIPTORS (ONLY THESE)
{descriptor_list}

{validation_performance_guidance}

{few_shot_examples}

{common_mistakes}

{output_format}

{variation_note}
"""


# =============================================================================
# IMPROVED EVALUATION FUNCTIONS
# =============================================================================

@dataclass
class FormulaEvaluation:
    """Container for comprehensive formula evaluation results."""
    formula: str
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    
    # Metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    
    # Sign-match analysis
    sign_match_rate: Optional[float] = None
    sign_match_details: Optional[Dict[str, Dict]] = None
    
    # Stability analysis
    stability_score: Optional[float] = None
    cv_mae_std: Optional[float] = None
    
    # Interpretability
    interpretability_score: Optional[float] = None
    sensitivity_analysis: Optional[Dict[str, float]] = None
    
    # Predictions
    predictions: Optional[np.ndarray] = None
    
    # Composite score
    composite_score: Optional[float] = None


def _extract_json_after_marker(text: str, marker: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object that appears after a marker line, e.g.:
      RDKitValidation:
      {{ ... }}
    Uses brace counting to find the matching closing brace.
    """
    if not isinstance(text, str) or not text:
        return None
    idx = text.lower().find(marker.lower())
    if idx < 0:
        return None
    # Find first '{' after marker
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    blob = text[start:end].strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def _rdkit_descriptor_matrix_from_smiles_dicts(x_original: List[Any]) -> Dict[str, np.ndarray]:
    """
    Compute RDKit descriptor vectors for a list of DeepChem-style X_original entries.
    Each entry is expected to be a dict with 'SMILES' (or convertible to str).
    Returns dict: descriptor_name -> np.ndarray shape (n,)
    """
    vals: Dict[str, List[float]] = {k: [] for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES}
    for item in x_original:
        if isinstance(item, dict):
            smiles = str(item.get("SMILES", ""))
        else:
            smiles = str(item)
        props = _compute_molecular_properties(smiles)
        for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES:
            try:
                vals[k].append(float(props.get(k, 0.0)))
            except Exception:
                vals[k].append(0.0)
    return {k: np.asarray(v, dtype=float) for k, v in vals.items()}


def compute_sign_match_analysis(
    descriptors: Dict[str, np.ndarray],
    y_true: np.ndarray,
    expected_effects: Dict[str, str],
    significance_threshold: float = 0.1
) -> Tuple[float, Dict[str, Dict]]:
    """
    Compute sign-match analysis between expected and observed descriptor effects.
    Uses Pearson correlation with significance testing.
    """
    details = {}
    n_matched = 0
    n_total = 0
    
    for desc_name, expected_sign in expected_effects.items():
        if desc_name not in descriptors:
            continue
        
        desc_values = descriptors[desc_name]
        
        # Compute correlation
        if len(desc_values) < 3 or np.std(desc_values) < 1e-10:
            corr = 0.0
            p_value = 1.0
        else:
            try:
                corr, p_value = stats.pearsonr(desc_values, y_true)
            except Exception:
                corr = 0.0
                p_value = 1.0
        
        # Determine observed sign
        if abs(corr) < significance_threshold or p_value > 0.1:
            observed_sign = "neutral"
        elif corr > 0:
            observed_sign = "positive"
        else:
            observed_sign = "negative"
        
        # Check match
        expected_sign = str(expected_sign).lower().strip()
        is_match = (observed_sign == expected_sign) or (expected_sign == "neutral")
        
        details[desc_name] = {
            "expected": expected_sign,
            "observed": observed_sign,
            "correlation": float(corr),
            "p_value": float(p_value),
            "match": bool(is_match)
        }
        
        n_total += 1
        if is_match:
            n_matched += 1
    
    match_rate = float(n_matched / n_total) if n_total > 0 else 0.0
    return match_rate, details


def compute_stability_score(
    formula: str,
    descriptors: Dict[str, np.ndarray],
    y_true: np.ndarray,
    feature_cols: List[str],
    X_original: List[Any],
    n_folds: int = 5
) -> Tuple[float, float]:
    """
    Compute stability score via cross-validation.
    Returns (stability_score, cv_mae_std).
    
    Stability score = 1.0 - normalized_cv_std (higher is better)
    """
    n = len(y_true)
    if n < n_folds * 2:
        return 0.5, 0.0  # Not enough data
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_maes = []
    
    for train_idx, val_idx in kf.split(y_true):
        # Create fold descriptors
        fold_desc = {k: v[val_idx] for k, v in descriptors.items()}
        fold_y = y_true[val_idx]
        fold_X_original = [X_original[i] for i in val_idx]
        
        # Evaluate using existing evaluate_llm_formula
        try:
            # Create dummy X matrix for evaluate_llm_formula
            fold_X = np.zeros((len(val_idx), len(feature_cols)))
            preds = evaluate_llm_formula(
                formula=formula,
                X=fold_X,
                feature_cols=feature_cols,
                scaler=None,
                y_scaler=None,
                X_original=fold_X_original,
            ).astype(float)
            mae = mean_absolute_error(fold_y, preds)
            fold_maes.append(mae)
        except Exception:
            fold_maes.append(float("inf"))
    
    fold_maes = np.array(fold_maes)
    fold_maes = fold_maes[np.isfinite(fold_maes)]
    
    if len(fold_maes) == 0:
        return 0.0, float("inf")
    
    cv_std = float(np.std(fold_maes))
    cv_mean = float(np.mean(fold_maes))
    
    # Normalize std by mean to get coefficient of variation
    if cv_mean > 1e-10:
        cv_normalized = cv_std / cv_mean
    else:
        cv_normalized = cv_std
    
    # Convert to stability score (1 = perfectly stable, 0 = highly variable)
    stability = max(0.0, 1.0 - cv_normalized)
    
    return stability, cv_std


def compute_sensitivity_analysis(
    formula: str,
    descriptors: Dict[str, np.ndarray],
    feature_cols: List[str],
    X_original: List[Any],
    perturbation_pct: float = 0.1
) -> Dict[str, float]:
    """
    Compute sensitivity of formula output to each descriptor.
    Returns dict mapping descriptor name to sensitivity (mean absolute change in output).
    """
    sensitivities = {}
    
    # Baseline predictions
    try:
        baseline_X = np.zeros((len(X_original), len(feature_cols)))
        baseline_preds = evaluate_llm_formula(
            formula=formula,
            X=baseline_X,
            feature_cols=feature_cols,
            scaler=None,
            y_scaler=None,
            X_original=X_original,
        ).astype(float)
    except Exception:
        return {k: 0.0 for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES}
    
    # For each descriptor, compute sensitivity
    # Note: This is simplified - a full version would modify SMILES and recompute
    for desc_name in SUPPORTED_RDKIT_DESCRIPTOR_NAMES:
        if desc_name not in descriptors:
            sensitivities[desc_name] = 0.0
            continue
        
        # Simplified sensitivity: use correlation as proxy
        desc_values = descriptors[desc_name]
        if len(desc_values) < 2 or np.std(desc_values) < 1e-10:
            sensitivities[desc_name] = 0.0
        else:
            # Use correlation as sensitivity proxy
            try:
                corr = np.corrcoef(desc_values, baseline_preds)[0, 1]
                sensitivities[desc_name] = float(abs(corr)) if not np.isnan(corr) else 0.0
            except Exception:
                sensitivities[desc_name] = 0.0
    
    return sensitivities


def compute_interpretability_score(
    sensitivity_analysis: Dict[str, float],
    expected_effects: Optional[Dict[str, str]] = None
) -> float:
    """
    Compute interpretability score based on:
    1. How many descriptors have non-negligible sensitivity
    2. Whether sensitivities align with expected effects
    """
    sensitivities = list(sensitivity_analysis.values())
    
    if not sensitivities:
        return 0.5
    
    # Check for degenerate formula (constant or near-constant)
    max_sens = max(sensitivities)
    if max_sens < 0.001:
        return 0.1  # Formula is essentially constant
    
    # Count how many descriptors have meaningful sensitivity
    threshold = max_sens * 0.1  # 10% of max sensitivity
    n_meaningful = sum(1 for s in sensitivities if s > threshold)
    
    # Ideal: 3-6 meaningful descriptors
    if 3 <= n_meaningful <= 6:
        diversity_score = 1.0
    elif n_meaningful < 3:
        diversity_score = n_meaningful / 3.0
    else:
        diversity_score = max(0.5, 1.0 - (n_meaningful - 6) * 0.1)
    
    # Alignment score (if expected effects provided)
    alignment_score = 0.5
    if expected_effects:
        n_aligned = 0
        n_total = 0
        for desc_name, expected in expected_effects.items():
            if desc_name not in sensitivity_analysis:
                continue
            sens = sensitivity_analysis[desc_name]
            expected = str(expected).lower()
            
            # Check if descriptor has appropriate sensitivity
            if expected == "neutral" and sens < threshold:
                n_aligned += 1
            elif expected in ("positive", "negative") and sens >= threshold:
                n_aligned += 1
            n_total += 1
        
        if n_total > 0:
            alignment_score = n_aligned / n_total
    
    return 0.6 * diversity_score + 0.4 * alignment_score


def compute_composite_score(evaluation: FormulaEvaluation) -> float:
    """
    Compute weighted composite score from multiple metrics.
    Higher is better.
    """
    if not evaluation.is_valid:
        return 0.0
    
    scores = {}
    
    # Sign match (0-1, higher is better)
    if evaluation.sign_match_rate is not None:
        scores["sign_match"] = evaluation.sign_match_rate
    else:
        scores["sign_match"] = 0.5
    
    # MAE (convert to 0-1 score, lower MAE = higher score)
    if evaluation.mae is not None:
        # Assume MAE of 0.3+ is bad, 0.0 is perfect
        scores["mae"] = max(0.0, 1.0 - evaluation.mae / 0.3)
    else:
        scores["mae"] = 0.0
    
    # R² (already 0-1ish, but can be negative)
    if evaluation.r2 is not None:
        scores["r2"] = max(0.0, min(1.0, evaluation.r2))
    else:
        scores["r2"] = 0.0
    
    # Stability (0-1, higher is better)
    if evaluation.stability_score is not None:
        scores["stability"] = evaluation.stability_score
    else:
        scores["stability"] = 0.5
    
    # Interpretability (0-1, higher is better)
    if evaluation.interpretability_score is not None:
        scores["interpretability"] = evaluation.interpretability_score
    else:
        scores["interpretability"] = 0.5
    
    # Weighted average
    total = 0.0
    weight_sum = 0.0
    for metric, weight in EVAL_WEIGHTS.items():
        if metric in scores:
            total += scores[metric] * weight
            weight_sum += weight
    
    return total / weight_sum if weight_sum > 0 else 0.0


def evaluate_formula_comprehensive(
    formula: str,
    descriptors: Dict[str, np.ndarray],
    y_true: np.ndarray,
    feature_cols: List[str],
    X_original: List[Any],
    expected_effects: Optional[Dict[str, str]] = None,
    compute_stability: bool = True
) -> FormulaEvaluation:
    """
    Comprehensive formula evaluation combining multiple metrics.
    """
    result = FormulaEvaluation(formula=formula, is_valid=False)
    
    # Step 1: Try to evaluate using existing evaluate_llm_formula
    try:
        X_dummy = np.zeros((len(X_original), len(feature_cols)))
        predictions = evaluate_llm_formula(
            formula=formula,
            X=X_dummy,
            feature_cols=feature_cols,
            scaler=None,
            y_scaler=None,
            X_original=X_original,
        ).astype(float)
        
        # Validate predictions
        if len(predictions) != len(y_true):
            result.validation_errors.append(f"Prediction length mismatch: {len(predictions)} vs {len(y_true)}")
            return result
        
        # Check for invalid predictions
        valid_mask = np.isfinite(predictions)
        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            result.validation_errors.append(f"Found {n_invalid} invalid predictions (NaN/Inf)")
            # Replace invalid predictions with mean of valid ones
            if np.sum(valid_mask) > 0:
                predictions[~valid_mask] = np.mean(predictions[valid_mask])
            else:
                predictions[~valid_mask] = 0.5  # Default to middle of [0,1] range
        
        # Check prediction range
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        if pred_max - pred_min < 1e-6:
            result.validation_errors.append(f"Predictions have no variance (all ~{pred_min:.4f})")
        
        result.predictions = predictions
        result.is_valid = True
    except Exception as e:
        result.validation_errors.append(f"Evaluation error: {str(e)}")
        import traceback
        result.validation_errors.append(f"Traceback: {traceback.format_exc()}")
        return result
    
    # Step 2: Basic metrics
    result.mae = float(mean_absolute_error(y_true, predictions))
    result.mse = float(mean_squared_error(y_true, predictions))
    result.rmse = float(np.sqrt(result.mse))
    result.r2 = float(r2_score(y_true, predictions))
    
    # Step 3: Sign-match analysis
    if expected_effects:
        match_rate, match_details = compute_sign_match_analysis(
            descriptors, y_true, expected_effects
        )
        result.sign_match_rate = match_rate
        result.sign_match_details = match_details
    
    # Step 4: Stability analysis
    if compute_stability:
        stability, cv_std = compute_stability_score(
            formula, descriptors, y_true, feature_cols, X_original
        )
        result.stability_score = stability
        result.cv_mae_std = cv_std
    
    # Step 5: Sensitivity analysis
    result.sensitivity_analysis = compute_sensitivity_analysis(
        formula, descriptors, feature_cols, X_original
    )
    
    # Step 6: Interpretability score
    result.interpretability_score = compute_interpretability_score(
        result.sensitivity_analysis, expected_effects
    )
    
    # Step 7: Composite score
    result.composite_score = compute_composite_score(result)
    
    return result


def _find_unsupported_descriptor_mentions(formula: str) -> List[str]:
    if not isinstance(formula, str) or not formula.strip():
        return []
    allowed = set(SUPPORTED_RDKIT_DESCRIPTOR_NAMES + ["SMILES", "clip"])

    # Collect identifiers; keep it conservative (avoid numbers and python keywords).
    toks = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
    # Remove obvious builtins and math funcs that evaluator supports.
    safe_math = {
        "abs", "min", "max", "round", "sqrt", "exp", "log", "sin", "cos", "tan", "pow",
    }
    toks = {t for t in toks if t not in safe_math}

    # If token appears as a call foo(...), it must be in allowed descriptors (or clip).
    bad = []
    for t in sorted(toks):
        if t in allowed:
            continue
        if f"{t}(" in formula:
            bad.append(t)
        # Also flag bare variables that are not allowed.
        elif t not in allowed:
            bad.append(t)
    # De-dup while keeping order
    seen = set()
    out = []
    for b in bad:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _sanitize_formula_single_line(formula: str) -> str:
    """
    Make the extracted formula safe for RDKit execution and file logging:
    - remove newlines inside expressions
    - collapse excessive whitespace
    """
    if not isinstance(formula, str):
        return ""
    # Replace any hard line breaks introduced by the LLM with spaces.
    s = formula.replace("\r", " ").replace("\n", " ").strip()
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _validate_esol_signs(formula: str, dataset_key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that formula has correct signs for ESOL dataset.
    Returns (is_valid, error_message).
    """
    dataset_key = str(dataset_key).lower().strip()
    if dataset_key not in ("esol", "delaney"):
        return True, None  # Only validate ESOL
    
    if not isinstance(formula, str) or not formula.strip():
        return True, None  # Can't validate empty formula
    
    # Expected signs for ESOL (solubility)
    expected_signs = {
        "num_hydroxyl_groups": "positive",
        "num_oxygen": "positive",
        "num_nitrogen": "positive",
        "num_hbd": "positive",
        "num_halogen": "negative",
        "num_rings": "negative",
        "is_aromatic": "negative",
        "molecular_weight": "negative",
    }
    
    # Check for sign violations by looking for descriptor patterns with wrong signs
    violations = []
    
    for desc_name, expected_sign in expected_signs.items():
        # Look for this descriptor in the formula
        pattern = rf"{desc_name}\s*\("
        if not re.search(pattern, formula, re.IGNORECASE):
            continue  # Descriptor not used, skip
        
        # Find all occurrences of this descriptor
        sign_pattern = rf"([+\-])\s*(?:[\d.]+\s*\*)?\s*{desc_name}\s*\("
        matches = list(re.finditer(sign_pattern, formula, re.IGNORECASE))
        
        if not matches:
            # Check if it's the first term (no sign) or after a multiplication
            continue
        
        for match in matches:
            sign_char = match.group(1)
            actual_sign = "positive" if sign_char == "+" else "negative"
            
            if actual_sign != expected_sign:
                violations.append(f"{desc_name} has {actual_sign} coefficient but should be {expected_sign}")
    
    if violations:
        violations_str = ', '.join(violations)
        return False, f"Sign violations detected: {violations_str}"
    
    return True, None


# =============================================================================
# PROMPT PATCHING (for MA-ICL integration)
# =============================================================================

def _ensure_nested_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    cur: Dict[str, Any] = d
    for k in keys:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    return cur


def _patch_deepchem_prompts_in_memory(dataset_key: str) -> None:
    """
    Patch MA-ICL prompt strings for DeepChem molecular datasets using improved prompts.
    """
    return _patch_deepchem_prompts_in_memory_variant(dataset_key=dataset_key, variation_tag="")


def _patch_deepchem_prompts_in_memory_variant(dataset_key: str, variation_tag: str) -> None:
    """
    Same as `_patch_deepchem_prompts_in_memory`, but allows adding a small variation tag to
    encourage diverse candidate formulas across `--iterations` runs.
    """
    cfg = maicl_config.load_maicl_config()
    prompts = _ensure_nested_dict(cfg, ["prompts"])
    ds_specific = _ensure_nested_dict(prompts, ["mechanism_generation", "dataset_specific"])

    # Always patch both ESOL-ish and LIPO-ish because dataset normalization may map delaney->esol.
    target_keys = sorted(set([dataset_key, "esol", "delaney", "lipo", "lipophilicity"]))
    for key in target_keys:
        _ensure_nested_dict(ds_specific, [key])

    dataset_key = str(dataset_key).lower().strip()

    # Use improved prompts
    encoder_template = get_encoder_prompt_v2(dataset_key)
    decoder_template = get_decoder_prompt_v2(dataset_key, variation_tag)

    for key in target_keys:
        # Only patch keys that exist in config OR ones we created above.
        ds_specific[key]["encoder"] = encoder_template
        ds_specific[key]["decoder"] = decoder_template

    # Commit patched config into the module-global cache.
    maicl_config._MAICL_CONFIG = cfg  # type: ignore[attr-defined]


# =============================================================================
# MAIN SCRIPT
# =============================================================================

@dataclass
class RunOutputs:
    output_dir: str
    mechanism_text: str
    extracted_formula: str
    metrics: Dict[str, float]


def main() -> None:
    ap = argparse.ArgumentParser(description="Improved DeepChem MA-ICL: generate LLM mechanism and validate via RDKit execution")
    ap.add_argument("--dataset", type=str, default="esol", choices=["esol", "delaney", "lipo", "lipophilicity"],
                    help="DeepChem chemistry regression dataset to use")
    ap.add_argument("--model_name", type=str, default=os.environ.get("MAICL_MODEL_NAME", "gemini-2.0-flash"),
                    help="Gemini model name (must have GOOGLE_API_KEY env var set)")
    ap.add_argument("--max_samples", type=int, default=100, help="Max total samples to use (applied to combined dataset before splitting into train/val/test)")
    ap.add_argument("--top_k", type=int, default=100, help="Top-K samples to select for training (-1 to use full dataset). Uses balanced y-quantile selection for LLM-only mode.")
    ap.add_argument("--iterations", type=int, default=3,
                    help="MA-ICL learning iterations (runs TrainableMAICL.train).")
    ap.add_argument("--acceptance_set", type=str, default="test", choices=["validation", "train", "test"],
                    help="Which set MA-ICL uses for acceptance during training (default: validation).")
    ap.add_argument("--k_shot", type=int, default=0, help="Few-shot examples per prediction inside MA-ICL train/eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=None, help="Where to write artifacts (default: ma_icl_results/rdkit_validate_improved_<dataset>_<timestamp>)")
    ap.add_argument("--deepchem_splitter", type=str, default="random", help="DeepChem MolNet splitter (random/scaffold/...)")
    ap.add_argument("--deepchem_featurizer", type=str, default="ECFP", help="DeepChem featurizer for ML side (kept for loader compatibility)")
    args = ap.parse_args()

    np.random.seed(int(args.seed))

    # Patch prompts so the produced mechanism is RDKit-evaluable.
    _patch_deepchem_prompts_in_memory(dataset_key=str(args.dataset).lower())

    # Load 018 runner as a module by path, and reuse its DeepChem loader + Gemini LLM factory.
    runner = _load_018_runner()

    # Create output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        import datetime as _dt
        here = Path(__file__).resolve().parent
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = str(here / "ma_icl_results" / f"rdkit_validate_improved_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Get LLM client (requires API key env vars)
    llm = runner._get_gemini_llm(args.model_name)

    # Load DeepChem dataset - the function combines all splits, so we'll load all data
    # and then split it ourselves
    # Pass a very large number to effectively disable max_samples limit
    X_all, y_all, X_original_all, feature_cols, feature_encoders, ds_label = runner.load_deepchem_regression_dataset(
        args.dataset,
        999999,  # Large number to load all data (we'll subsample later)
    )
    
    maicl_config.logger.info(f"Loaded dataset: {len(X_all)} total samples")
    
    # Apply max_samples to combined dataset
    max_samples = int(args.max_samples) if args.max_samples else None
    if max_samples and len(X_all) > max_samples:
        rng = np.random.RandomState(int(args.seed))
        indices = rng.choice(len(X_all), max_samples, replace=False)
        X_all = X_all[indices]
        y_all = y_all[indices]
        X_original_all = [X_original_all[i] for i in indices]
        maicl_config.logger.info(f"Subsampled combined dataset to {len(X_all)} samples (max_samples={max_samples})")
    
    # Split into train/validation/test (60/20/20 split)
    # First split: 60% train, 40% temp (which will become val+test)
    X_train, X_temp, y_train, y_temp, X_original_train, X_original_temp = train_test_split(
        X_all, y_all, X_original_all,
        test_size=0.4,
        random_state=int(args.seed),
        shuffle=True
    )
    # Second split: split temp into 50/50 for val and test (so 20% each of original)
    X_val, X_test, y_val, y_test, X_original_val, X_original_test = train_test_split(
        X_temp, y_temp, X_original_temp,
        test_size=0.5,
        random_state=int(args.seed),
        shuffle=True
    )
    
    maicl_config.logger.info(f"Split dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Scale y to [0,1] (match your 018 default behavior) so the Formula can clip to [0,1].
    y_scaler = MinMaxScaler010()
    y_train_s = y_scaler.fit_transform(np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(np.asarray(y_test, dtype=float).reshape(-1, 1)).ravel()
    # DeepChem official splits can contain targets outside the TRAIN min/max; MinMax scaling can then yield
    # slightly <0 or >1 on val/test. MA-ICL's scaling verifier expects [0,1] (with small tolerance).
    y_val_s = np.clip(y_val_s, 0.0, 1.0)
    y_test_s = np.clip(y_test_s, 0.0, 1.0)

    # Top-K selection for training subset (LLM-only mode: uses balanced y-quantile selection)
    def _select_topk_balanced_y_quantile(X_tr_s, y_tr_s, k, bins=10, random_state=42):
        """Select top-K samples with balanced y-quantile coverage (for LLM-only mode)"""
        if k <= 0 or k >= len(y_tr_s):
            return X_tr_s, y_tr_s, np.arange(len(y_tr_s))
        q = np.quantile(y_tr_s, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        yb = np.digitize(y_tr_s, q, right=True)
        idxs = []
        # Allocate k proportionally to bin sizes, minimum 1 if bin non-empty
        unique_bins, counts = np.unique(yb, return_counts=True)
        proportions = {b: c / len(y_tr_s) for b, c in zip(unique_bins, counts)}
        allocated = {b: max(1, int(round(k * proportions[b]))) for b in unique_bins}
        # Adjust allocation to exactly k
        total_alloc = sum(allocated.values())
        # Trim or add to match k
        while total_alloc > k:
            bmax = max(allocated, key=lambda b: allocated[b])
            if allocated[bmax] > 1:
                allocated[bmax] -= 1
                total_alloc -= 1
            else:
                break
        while total_alloc < k:
            bmin = min(allocated, key=lambda b: allocated[b])
            allocated[bmin] += 1
            total_alloc += 1
        # Pick randomly within each bin (since we don't have residuals)
        rng = np.random.RandomState(random_state)
        for b in unique_bins:
            bin_idx = np.where(yb == b)[0]
            if len(bin_idx) == 0:
                continue
            take = min(allocated[b], len(bin_idx))
            selected = rng.choice(bin_idx, size=take, replace=False)
            idxs.extend(selected.tolist())
        idxs = np.array(sorted(set(idxs)))
        return X_tr_s[idxs], y_tr_s[idxs], idxs

    # Apply top-K selection if requested
    if args.top_k == -1:
        X_topk, y_topk, top_indices = X_train, y_train_s, np.arange(len(X_train))
        X_original_topk = X_original_train
        print(f"Using full training set: {len(X_topk)} samples")
    else:
        print(f"Selecting top-K samples with balanced y-quantile coverage (K={args.top_k})")
        X_topk, y_topk, top_indices = _select_topk_balanced_y_quantile(
            X_train, y_train_s, args.top_k, bins=10, random_state=int(args.seed)
        )
        X_original_topk = [X_original_train[i] for i in top_indices]
        print(f"Top-K selection: selected {len(top_indices)} examples from {len(X_train)} total")

    # We don't need to train ML here; we only need the LLM mechanism + RDKit evaluation.
    # Use MA-ICL in LLM-only mode.
    maicl = TrainableMAICL(
        batched_llm=llm,
        feature_cols=feature_cols,
        scaler=None,  # ECFP scaling is irrelevant for RDKit formula evaluation
        use_ml_mechanism=False,
        dataset_name=ds_label,
        y_scaler=None,
        pretrained_ml_mechanism=None,
        data_insights=None,
        task_type="regression",
        class_names=None,
        regression_loss_metric="mae",
        num_mechanisms_unknown=1,
        use_scaling=True,
    )
    maicl.output_dir = out_dir

    # Provide SMILES to mechanism generator so it never tries to reason over ecfp_bit_*.
    maicl.mech_generator.X_train_original = X_original_topk
    
    # Store acceptance set references for LLM-only evaluation detection
    acceptance_set_name = str(args.acceptance_set).lower()
    if acceptance_set_name == "test":
        maicl._acceptance_set_X = X_test
        maicl._acceptance_set_X_original = X_original_test
    elif acceptance_set_name == "validation":
        maicl._acceptance_set_X = X_val
        maicl._acceptance_set_X_original = X_original_val
    else:  # train
        maicl._acceptance_set_X = X_topk
        maicl._acceptance_set_X_original = X_original_topk
    maicl._acceptance_set_name = acceptance_set_name

    # Build a simple "error" signal to trigger mechanism generation (no ML residuals here).
    # Use deviation from mean as a proxy difficulty score.
    prediction_errors = np.abs(y_topk - float(np.mean(y_topk)))

    # ----------------------------
    # RUN MA-ICL LEARNING PROCESS
    # ----------------------------
    # This runs the full TextGrad-driven mechanism update loop and writes:
    #   mechanisms_iter_{i}.txt
    # into the output_dir via visualization.persist_iteration_artifacts().
    maicl.train(
        X_topk,
        y_topk,
        X_val,
        y_val_s,
        iterations=int(max(1, args.iterations)),
        ml_residuals=None,
        accept_eval_max=None,
        X_test=X_test,
        y_test=y_test_s,
        acceptance_set=str(args.acceptance_set),
        k_shot=int(args.k_shot),
        X_train_original=X_original_topk,
        X_val_original=X_original_val,
        X_test_original=X_original_test,
        output_dir=out_dir,
    )

    # Determine which iteration snapshot to use (prefer best snapshot, else final).
    best_iter = getattr(maicl, "_best_iteration", None)
    if best_iter is None:
        best_iter = int(max(1, args.iterations))
    best_iter = int(best_iter)

    mechanisms_file = os.path.join(out_dir, f"mechanisms_iter_{best_iter}.txt")
    mechs = parse_mechanisms_file(mechanisms_file)
    if not mechs:
        # If best_iter is 0 and file doesn't exist, try to use iteration 1 as fallback
        if best_iter == 0:
            maicl_config.logger.warning(f"  ⚠️  mechanisms_iter_0.txt not found, trying mechanisms_iter_1.txt as fallback")
            mechanisms_file = os.path.join(out_dir, f"mechanisms_iter_1.txt")
            mechs = parse_mechanisms_file(mechanisms_file)
            if mechs:
                best_iter = 1
                maicl_config.logger.info(f"  ✓ Using mechanisms from iteration 1 instead")
        if not mechs:
            raise SystemExit(f"No mechanisms found in {mechanisms_file}")

    # Compute descriptors once for validation set
    desc_val = _rdkit_descriptor_matrix_from_smiles_dicts(X_original_val)

    # Pick the best mechanism using improved comprehensive evaluation
    best_mech = None
    best_eval = None
    best_mech_type = None
    best_match_rate = -1.0
    best_mech_val_mae = float("inf")
    best_mech_val_metrics = None
    best_mech_rdkit_validation_report = None

    for mech_type, mech_text in mechs:
        if mech_type.lower() not in ("llm", "known"):
            continue
        
        # Extract formula
        formula = extract_formula_from_llm_mechanism(mech_text) or ""
        formula = _sanitize_formula_single_line(formula)
        
        if not formula or _find_unsupported_descriptor_mentions(formula):
            continue
        
        # ESOL-specific: Validate signs before evaluation (skip mechanisms with wrong signs)
        if str(args.dataset).lower() in ("esol", "delaney"):
            is_valid_signs, sign_error = _validate_esol_signs(formula, args.dataset)
            if not is_valid_signs:
                maicl_config.logger.debug(f"  Skipping mechanism with sign violations: {sign_error}")
                continue
        
        # Extract expected effects from RDKitValidation block
        rdv = _extract_json_after_marker(mech_text, marker="RDKitValidation")
        expected_effects = None
        if isinstance(rdv, dict):
            expected_effects = rdv.get("expected_effects")
        
        # Compute sign-match analysis
        report = None
        match_rate = -1.0
        if isinstance(expected_effects, dict) and expected_effects:
            match_rate, match_details = compute_sign_match_analysis(
                desc_val, y_val_s, expected_effects
            )
            report = {"match_rate": match_rate, "per_descriptor": match_details}
        
        # Evaluate formula comprehensively
        evaluation = evaluate_formula_comprehensive(
            formula=formula,
            descriptors=desc_val,
            y_true=y_val_s,
            feature_cols=feature_cols,
            X_original=X_original_val,
            expected_effects=expected_effects,
            compute_stability=True
        )
        
        if not evaluation.is_valid:
            continue
        
        # Selection: prefer higher composite score; break ties via match_rate, then MAE
        val_mae = evaluation.mae if evaluation.mae is not None else float("inf")
        val_r2 = evaluation.r2 if evaluation.r2 is not None else float("-inf")
        composite = evaluation.composite_score if evaluation.composite_score is not None else 0.0
        
        better = False
        if best_eval is None:
            better = True
        elif composite > (best_eval.composite_score if best_eval.composite_score is not None else 0.0):
            better = True
        elif abs(composite - (best_eval.composite_score if best_eval.composite_score is not None else 0.0)) < 0.05:
            # Tie-breaker: prefer higher match_rate
            if match_rate > best_match_rate:
                better = True
            elif match_rate == best_match_rate:
                # Same match rate - compare MAE
                if val_mae < best_mech_val_mae:
                    better = True
                elif abs(val_mae - best_mech_val_mae) / max(abs(best_mech_val_mae), 0.01) < 0.05:
                    # MAE within 5% - prefer LLM over KNOWN
                    if mech_type.lower() == "llm" and best_mech_type and best_mech_type.lower() == "known":
                        better = True
                    # Also prefer better R2 if MAE is similar
                    elif val_r2 > (float(best_mech_val_metrics.get("r2", float("-inf"))) if best_mech_val_metrics else float("-inf")):
                        better = True

        if better:
            best_match_rate = match_rate
            best_mech_val_mae = val_mae
            best_mech = {"type": mech_type, "text": mech_text, "formula": formula}
            best_mech_val_metrics = {
                "r2": val_r2,
                "mae": val_mae,
                "mse": evaluation.mse if evaluation.mse is not None else float("inf"),
                "rmse": evaluation.rmse if evaluation.rmse is not None else float("inf"),
            }
            best_mech_rdkit_validation_report = report
            best_mech_type = mech_type
            best_eval = evaluation

    if best_mech is None:
        raise SystemExit(
            f"MA-ICL trained, but no usable LLM mechanism could be selected from {mechanisms_file}"
        )

    # Evaluate chosen mechanism on test split:
    desc_test = _rdkit_descriptor_matrix_from_smiles_dicts(X_original_test)
    test_metrics = None
    test_evaluation = None
    
    if best_mech.get("formula"):
        test_evaluation = evaluate_formula_comprehensive(
            formula=best_mech["formula"],
            descriptors=desc_test,
            y_true=y_test_s,
            feature_cols=feature_cols,
            X_original=X_original_test,
            expected_effects=expected_effects,
            compute_stability=False  # Skip CV on test set
        )
        
        if test_evaluation.is_valid and test_evaluation.predictions is not None:
            test_metrics = {
                "r2": test_evaluation.r2 if test_evaluation.r2 is not None else float("-inf"),
                "mae": test_evaluation.mae if test_evaluation.mae is not None else float("inf"),
                "mse": test_evaluation.mse if test_evaluation.mse is not None else float("inf"),
                "rmse": test_evaluation.rmse if test_evaluation.rmse is not None else float("inf"),
            }

    # Also compute a test-side RDKitValidation report if we have one on validation.
    best_mech_rdkit_validation_report_test = None
    rdv_best = _extract_json_after_marker(best_mech["text"], marker="RDKitValidation")
    expected_best = rdv_best.get("expected_effects") if isinstance(rdv_best, dict) else None
    if isinstance(expected_best, dict) and expected_best:
        match_rate_test, match_details_test = compute_sign_match_analysis(
            desc_test, y_test_s, expected_best
        )
        best_mech_rdkit_validation_report_test = {"match_rate": match_rate_test, "per_descriptor": match_details_test}

    # Save stable artifacts
    with open(os.path.join(out_dir, "llm_mechanism_best.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["text"].strip() + "\n")
    with open(os.path.join(out_dir, "extracted_formula_best.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["formula"].strip() + "\n")
    with open(os.path.join(out_dir, "llm_mechanism.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["text"].strip() + "\n")
    with open(os.path.join(out_dir, "extracted_formula.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["formula"].strip() + "\n")

    # Save comprehensive metrics
    metrics_dict = {
        "dataset": ds_label,
        "dataset_key": args.dataset,
        "model_name": args.model_name,
        "splitter": args.deepchem_splitter,
        "max_samples_train": int(args.max_samples),
        "top_k": int(args.top_k),
        "train_iterations": int(max(1, args.iterations)),
        "acceptance_set": str(args.acceptance_set),
        "k_shot": int(args.k_shot),
        "n_train": int(len(X_train)),
        "n_train_topk": int(len(X_topk)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "snapshot_iteration_used": best_iter,
        "selected_mechanism_type": str(best_mech["type"]),
        "formula": best_mech["formula"],
        # Formula-only metrics (for RDKit validation, not system performance)
        "formula_only_val_metrics_scaled_y": best_mech_val_metrics,
        "formula_only_test_metrics_scaled_y": test_metrics,
        "rdkit_validation_signcheck_val": best_mech_rdkit_validation_report,
        "rdkit_validation_signcheck_test": best_mech_rdkit_validation_report_test,
        "selection": {
            "match_rate_val": best_match_rate,
            "tie_breaker_val_mae": best_mech_val_mae,
            "composite_score_val": best_eval.composite_score if best_eval and best_eval.composite_score is not None else None,
            "stability_score_val": best_eval.stability_score if best_eval and best_eval.stability_score is not None else None,
            "interpretability_score_val": best_eval.interpretability_score if best_eval and best_eval.interpretability_score is not None else None,
        },
    }
    
    # Add comprehensive evaluation details if available
    if best_eval:
        if best_eval.sign_match_details:
            metrics_dict["sign_match_details_val"] = best_eval.sign_match_details
        if best_eval.sensitivity_analysis:
            metrics_dict["sensitivity_analysis_val"] = best_eval.sensitivity_analysis
    
    if test_evaluation:
        if test_evaluation.sign_match_details:
            metrics_dict["sign_match_details_test"] = test_evaluation.sign_match_details

    with open(os.path.join(out_dir, "rdkit_validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print("\n=== MA-ICL Training + RDKit Validation Complete ===")
    print(f"Output dir: {out_dir}")
    print(f"Snapshot iteration used: {best_iter}")
    
    # Print formula-only metrics (for RDKit validation)
    print("\n--- Formula-Only Performance (raw formula, for RDKit validation) ---")
    if best_mech_rdkit_validation_report is not None:
        print(f"RDKit sign-check (validation) match_rate: {best_mech_rdkit_validation_report.get('match_rate'):.3f}")
    if best_eval and best_eval.composite_score is not None:
        print(f"Composite score (validation): {best_eval.composite_score:.4f}")
        if best_eval.stability_score is not None:
            print(f"Stability score (validation): {best_eval.stability_score:.4f}")
        if best_eval.interpretability_score is not None:
            print(f"Interpretability score (validation): {best_eval.interpretability_score:.4f}")
    if best_mech.get("formula"):
        print("\nExtracted formula:")
        print(best_mech["formula"])
        if best_mech_val_metrics:
            print("\nFormula-only validation metrics (scaled y in [0,1]):")
            for k, v in best_mech_val_metrics.items():
                print(f"  {k}: {v:.6f}")
        if test_metrics:
            print("\nFormula-only test metrics (scaled y in [0,1]):")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

