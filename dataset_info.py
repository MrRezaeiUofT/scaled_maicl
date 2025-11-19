"""
Dataset-specific information and functions for MA-ICL.

This module contains dataset descriptions, known mechanisms, and helper functions
for handling different datasets in the MA-ICL system.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Constants
MAX_FEATURES_IN_DESCRIPTION = 5  # Maximum features in dataset descriptions
MAX_FEATURES_IN_COMPONENT_LIST = 3  # Maximum features in component lists


# =========================
# DATASET DESCRIPTIONS
# =========================

def get_dataset_description(dataset_name: str) -> str:
    """Get dataset-specific description for mechanism context"""
    descriptions = {
        "car": """CAR EVALUATION DATASET:
This dataset contains car attributes (buying price, maintenance cost, doors, persons, luggage boot, safety)
and the task is to classify cars into acceptability categories: unacc, acc, good, vgood.
Key insights:
- Categories represent safety and economic value assessment
- Higher safety and lower costs generally indicate better evaluations
- Multiple categorical features interact to determine final rating
- The classes are ordinal in nature (unacc < acc < good < vgood)""",
        
        "iris": """IRIS FLOWER CLASSIFICATION DATASET:
This dataset contains measurements of iris flowers (sepal length, sepal width, petal length, petal width) 
and the task is to classify them into three species: Setosa, Versicolor, and Virginica.
Key insights:
- Setosa is typically well-separated from the other two species
- Versicolor and Virginica are more similar and may require careful distinction
- Petal measurements are usually more discriminative than sepal measurements
- The classes represent distinct biological species with measurable differences""",
        
        "wine": """WINE QUALITY CLASSIFICATION DATASET:
This dataset contains chemical analysis measurements of wine samples
and the task is to classify wines into different quality categories.
Key insights:
- Features represent chemical properties (alcohol, acidity, sulfates, etc.)
- Class quality often correlates with chemical composition
- Multiple chemical properties interact to determine wine quality
- The classification is based on expert wine quality assessment""",
        
        "vehicle": """VEHICLE CLASSIFICATION DATASET:
This dataset contains shape and measurement features of vehicles
and the task is to classify them into vehicle types: bus, opel, saab, van.
Key insights:
- Features describe vehicle dimensions and shape characteristics
- Different vehicle types have distinct size and shape profiles
- Geometric features help distinguish vehicle categories
- The classes represent different vehicle manufacturers and types""",
        
        "adult": """ADULT INCOME PREDICTION DATASET:
This dataset contains demographic and employment features
and the task is to predict whether income exceeds $50K/year.
Key insights:
- Features include age, education, occupation, workclass, relationship status
- Income prediction relates to multiple socioeconomic factors
- Class imbalance may exist (more examples in one category)
- The classification has real-world economic significance""",
        
        "glass": """GLASS CLASSIFICATION DATASET:
This dataset contains chemical composition measurements of glass samples
and the task is to classify them into different glass types.
Key insights:
- Features represent chemical element percentages (Na, Mg, Al, Si, etc.)
- Different glass types have distinct chemical compositions
- Chemical ratios and interactions determine glass classification
- The classes represent different manufacturing processes or uses""",
        
        "zoo": """ZOO ANIMAL CLASSIFICATION DATASET:
This dataset contains animal characteristics and the task is to classify them into categories:
mammal, bird, reptile, fish, amphibian, insect, invertebrate.
Key insights:
- Features describe physical and behavioral characteristics
- Different animal classes have distinct feature patterns
- Biological features help distinguish animal categories""",
        
        "soybean": """SOYBEAN DATASET (Many Classes, Few Samples):
This dataset contains soybean disease diagnosis data with many disease types.
Key insights:
- Large number of classes (~19 disease types)
- Limited samples per class (few-shot classification challenge)
- Features include plant characteristics, environmental conditions, and disease symptoms
- Requires careful few-shot learning to handle class imbalance""",
        
        "primary-tumor": """PRIMARY TUMOR DATASET:
This dataset contains medical diagnosis data for primary tumor classification.
Key insights:
- Many classes (~22 tumor types)
- Limited samples per class
- Features include diagnostic attributes and patient characteristics""",
        
        "lymphography": """LYMPHOGRAPHY DATASET:
This dataset contains lymphography diagnosis data.
Key insights:
- 4 classes with limited samples
- Features include diagnostic attributes
- Medical classification task""",
        
        "ecoli": """ECOLI DATASET:
This dataset contains protein localization sites in E.coli bacteria.
Key insights:
- 8 classes representing different localization sites
- Features include sequence and composition attributes
- Biological classification task""",
        
        "credit": """CREDIT APPROVAL DATASET:
This dataset contains credit application data.
Key insights:
- Binary classification (approved/not approved)
- Features include financial and personal attributes
- Real-world financial decision task""",
        
        "vote": """CONGRESSIONAL VOTING RECORDS DATASET:
This dataset contains voting records of U.S. House of Representatives.
Key insights:
- Binary classification (republican/democrat)
- Features represent votes on various issues
- Political classification task""",
        
        "mushroom": """MUSHROOM DATASET:
This dataset contains mushroom characteristics.
Key insights:
- Binary classification (edible/poisonous)
- Features include physical and habitat characteristics
- Safety-critical classification task"""
    }
    
    return descriptions.get(dataset_name.lower(), f"Classification dataset: {dataset_name}")


# =========================
# DATASET STATISTICS
# =========================

def compute_dataset_stats(X, y):
    """Compute dataset statistics for mechanism generation"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "X_mean": X.mean(axis=0),
        "X_std": X.std(axis=0) + 1e-8,
        "y_mean": y.mean(),
        "y_std": y.std() + 1e-8,
        "corr": np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.eye(X.shape[1])
    }


# =========================
# MECHANISM DESCRIPTIONS
# =========================

def build_known_mechanism_description(
    dataset_name: str,
    feature_cols: List[str],
    task_type: str,
    mechanism_formula: str,
    problem_description: Optional[str] = None
) -> str:
    """
    Build a rich known mechanism description with dataset, feature, and task context.
    
    Args:
        dataset_name: Name of the dataset
        feature_cols: List of feature names
        task_type: "classification" or "regression"
        mechanism_formula: The executable formula for the mechanism
        problem_description: Optional description of what problem this mechanism solves
    
    Returns:
        A formatted mechanism description string
    """
    # Build feature description
    if feature_cols and len(feature_cols) > 0:
        if len(feature_cols) <= 5:
            feature_desc = ", ".join(feature_cols)
        else:
            feature_desc = ", ".join(feature_cols[:MAX_FEATURES_IN_DESCRIPTION]) + f", ... ({len(feature_cols)} total)"
    else:
        feature_desc = "features"
    
    # Build task description
    task_desc = "classification" if task_type == "classification" else "regression"
    if task_type == "classification":
        task_detail = "predicting class labels"
    else:
        task_detail = "predicting numeric values"
    
    # Build problem description if not provided
    if problem_description is None:
        problem_description = f"This mechanism addresses {task_detail} in the {dataset_name} dataset"
    
    # Combine into rich description
    description = f"""{problem_description}.

Dataset: {dataset_name}
Task: {task_desc} ({task_detail})
Features: {feature_desc}

Mechanism formula:
{mechanism_formula}

This mechanism uses actual feature names from the dataset and is designed to capture domain-specific patterns."""
    
    return description


# =========================
# DATASET TYPE DETECTION
# =========================

def is_deepchem_dataset(feature_cols: Optional[List[str]]) -> bool:
    """Check if this is a DeepChem dataset (ECFP fingerprints)"""
    if not feature_cols or len(feature_cols) == 0:
        return False
    return all(feat.startswith('ecfp_bit_') for feat in feature_cols[:10])


def has_smiles_data(X_original: Optional[List[Any]]) -> bool:
    """Check if X_original contains SMILES strings"""
    if not X_original or len(X_original) == 0:
        return False
    return isinstance(X_original[0], dict) and 'SMILES' in X_original[0]


def get_feature_info(
    feature_cols: Optional[List[str]],
    is_deepchem: bool,
    scale_range_str: str
) -> str:
    """Get feature information string for mechanism descriptions"""
    if is_deepchem:
        return f" Input: SMILES strings (molecular structures). The ML model uses ECFP fingerprints ({len(feature_cols)} bits internally), but YOU should work with SMILES strings and molecular properties, NOT ECFP bit features."
    elif feature_cols is not None and len(feature_cols) > 0:
        all_features_str = ", ".join(feature_cols)
        return f" All available features ({len(feature_cols)} total): {all_features_str}."
    else:
        return " Features: (feature names not available)."


# =========================
# DATASET-SPECIFIC KNOWN MECHANISMS
# =========================

def get_dataset_specific_known_mechanisms(
    dataset_name: str,
    feature_cols: Optional[List[str]],
    task_type: str,
    class_names: Optional[List[str]],
    scale_min: float,
    scale_max: float,
    X_train_original: Optional[List[Any]] = None,
    predictor: Optional[Any] = None
) -> List[str]:
    """
    Get dataset-specific known mechanisms with explicit formulas using actual feature names.
    
    Args:
        dataset_name: Name of the dataset
        feature_cols: List of feature column names
        task_type: "classification" or "regression"
        class_names: Optional list of class names for classification
        scale_min: Minimum value in scaling range
        scale_max: Maximum value in scaling range
        X_train_original: Optional original training data (for SMILES detection)
        predictor: Optional predictor object (for SMILES detection)
    
    Returns:
        List of known mechanism description strings
    """
    scale_range_str = f"[{scale_min}, {scale_max}]"
    
    # Check if this is a DeepChem dataset
    is_deepchem = is_deepchem_dataset(feature_cols)
    has_smiles = False
    if is_deepchem:
        X_original_check = X_train_original
        if X_original_check is None and predictor is not None:
            X_original_check = getattr(predictor, 'X_train_original', None)
        has_smiles = has_smiles_data(X_original_check)
    
    # Get feature info
    feature_info = get_feature_info(feature_cols, is_deepchem, scale_range_str)
    
    if task_type == "classification":
        # Get dataset description
        dataset_desc = get_dataset_description(dataset_name)
        
        # Build class names string
        class_names_str = ""
        class_list = []
        if class_names is not None and len(class_names) > 0:
            formatted_class_names = [str(cn) for cn in class_names]
            class_names_str = f" Available classes: {', '.join(formatted_class_names)}."
            class_list = formatted_class_names
        
        # Build formula with explicit class mapping
        if feature_cols and len(feature_cols) > 0:
            feature_sum = " + ".join(feature_cols[:min(5, len(feature_cols))])
            n_feat = min(5, len(feature_cols))
            if class_names_str and len(class_list) > 0:
                n_classes = len(class_list)
                bin_size = (scale_max - scale_min) / n_classes
                thresholds = [scale_min + (i + 1) * bin_size for i in range(n_classes - 1)]
                
                # Create nested if-else structure for explicit class mapping
                if len(thresholds) == 1:
                    class_mapping = f"ŷ = 0 if score < {thresholds[0]:.2f} else 1"
                elif len(thresholds) == 2:
                    class_mapping = f"ŷ = 0 if score < {thresholds[0]:.2f} else (1 if score < {thresholds[1]:.2f} else 2)"
                elif len(thresholds) == 3:
                    class_mapping = f"ŷ = 0 if score < {thresholds[0]:.2f} else (1 if score < {thresholds[1]:.2f} else (2 if score < {thresholds[2]:.2f} else 3))"
                else:
                    class_mapping = f"ŷ = 0"
                    for i in range(len(thresholds)):
                        class_mapping = f"({class_mapping} if score < {thresholds[i]:.2f} else {i+1})"
                    class_mapping = f"({class_mapping} else {n_classes-1})"
                
                formula = f"score = ({feature_sum}) / {n_feat}; {class_mapping}"
            else:
                formula = f"score = ({feature_sum}) / {n_feat}; ŷ = 0 if score < {(scale_min + scale_max) / 2:.2f} else 1"
        else:
            formula = f"score = (sum of all features) / (number of features); ŷ = 0 if score < {(scale_min + scale_max) / 2:.2f} else 1"
        
        # Build class output description
        class_output_desc = ""
        if class_names_str:
            class_list = [cn.strip() for cn in class_names_str.replace("Available classes:", "").replace("Classes:", "").split(",")]
            class_list = [cn for cn in class_list if cn]
            if class_list:
                class_indices = ", ".join([f"{i}={cn}" for i, cn in enumerate(class_list)])
                class_output_desc = f"\n- Output must be a CLASS INDEX (integer: 0, 1, 2, ...) corresponding to one of these classes: {class_indices}"
            else:
                class_output_desc = f"\n- Output must be a CLASS INDEX (integer: 0, 1, 2, ...) for classification"
        else:
            class_output_desc = f"\n- Output must be a CLASS INDEX (integer: 0, 1, 2, ...) for classification"
        
        mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Classification task using all available features.

{feature_info}{class_names_str}

MECHANISM DESCRIPTION:
{formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}{class_output_desc}
- The mechanism should compute a value that maps to a class index (use argmax, thresholds, or direct class selection)
- Use actual feature names from the dataset when referencing features
- The score computation should account for the feature scaling range [{scale_min}, {scale_max}]"""
        
        return [mechanism_desc]
    
    else:
        # REGRESSION: Dataset-specific mechanisms
        dataset_desc = get_dataset_description(dataset_name)
        if dataset_desc.startswith("Classification dataset:"):
            dataset_desc = f"Regression dataset: {dataset_name}. This dataset predicts continuous numeric values from input features."
        
        # Dataset-specific known mechanisms
        if "Yacht" in dataset_name:
            if feature_cols and len(feature_cols) >= 4:
                f1, f2, f3, f4 = feature_cols[0], feature_cols[1], feature_cols[2], feature_cols[3]
                formula = f"ŷ = ({f1} + {f2} + {f3}) / 3 + 0.2 * {f1} * {f2} - 0.1 * {f4}"
            else:
                formula = "ŷ = (sum of first 3 features) / 3 + 0.2 * feature1 * feature2 - 0.1 * feature4"
            
            dataset_desc = """YACHT HYDRODYNAMICS DATASET:
This dataset predicts yacht total resistance based on hull geometry parameters.
Key insights:
- Output depends on hull geometry parameters and their interactions
- Nonlinear effects from Froude number and displacement ratios
- Hydrodynamic resistance is influenced by multiple geometric factors"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting yacht total resistance using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "Concrete" in dataset_name:
            if feature_cols and len(feature_cols) >= 3:
                f1, f2 = feature_cols[0], feature_cols[1]
                f3 = feature_cols[2] if len(feature_cols) > 2 else f1
                formula = f"ŷ = ({f1} + {f2} + sum of remaining features) / {len(feature_cols)} + 0.15 * {f1} * {f3} - 0.1 * {f2}"
            else:
                formula = "ŷ = (sum of all features) / (number of features) + 0.15 * feature1 * feature3 - 0.1 * feature2"
            
            dataset_desc = """CONCRETE STRENGTH DATASET:
This dataset predicts concrete compressive strength from material composition.
Key insights:
- Output depends on cement content, water-cement ratio, and aggregate properties
- Nonlinear interactions between components affect compressive strength
- Material composition ratios determine final strength"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting concrete compressive strength using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "Energy" in dataset_name:
            if feature_cols and len(feature_cols) >= 2:
                f1, f2 = feature_cols[0], feature_cols[1]
                formula = f"ŷ = ({f1} + {f2} + sum of remaining features) / {len(feature_cols)} + 0.1 * {f1} * {f2}"
            else:
                formula = "ŷ = (sum of all features) / (number of features) + 0.1 * feature1 * feature2"
            
            dataset_desc = """ENERGY EFFICIENCY DATASET:
This dataset predicts building energy efficiency (heating/cooling load) from architectural and thermal properties.
Key insights:
- Output depends on building orientation, glazing area, and thermal properties
- Seasonal variations and heat transfer interactions affect energy consumption
- Architectural design parameters influence heating and cooling loads"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting building energy efficiency using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "Protein" in dataset_name and "Expression" not in dataset_name and "Structure" not in dataset_name:
            # Protein Binding (not Protein Expression or Protein Structure)
            if feature_cols and len(feature_cols) >= 2:
                f1, f2 = feature_cols[0], feature_cols[1]
                formula = f"ŷ = (sum of all features) / {len(feature_cols)} + 0.2 * {f1} * {f2}"
            else:
                formula = "ŷ = (sum of all features) / (number of features) + 0.2 * feature1 * feature2"
            
            dataset_desc = """PROTEIN-LIGAND BINDING AFFINITY DATASET:
This dataset predicts protein-ligand binding affinity from molecular descriptors.
Key insights:
- Output depends on molecular descriptors with hydrophobic interactions
- Hydrogen bonding and steric effects determine binding affinity
- Molecular properties influence protein-ligand interactions"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting protein-ligand binding affinity using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "Protein Expression" in dataset_name or "protein_expression" in dataset_name.lower():
            # Protein Expression - use actual feature names
            if feature_cols and all(f in feature_cols for f in ['nad', 'trna', 'aa']):
                formula = "ŷ = (nad + trna + aa + pga) / 4 + 0.15 * nad * trna - 0.1 * coa"
            else:
                if feature_cols and len(feature_cols) >= 3:
                    f1, f2, f3 = feature_cols[0], feature_cols[1], feature_cols[2]
                    formula = f"ŷ = ({f1} + {f2} + {f3}) / {len(feature_cols)} + 0.15 * {f1} * {f2} - 0.1 * {f3}"
                else:
                    formula = "ŷ = (sum of all input features) / (number of features) + 0.15 * feature1 * feature2 - 0.1 * feature3"
            
            dataset_desc = """PROTEIN EXPRESSION YIELD DATASET:
This dataset predicts protein production yield from cell-free expression systems.
Key insights:
- Output depends on reaction conditions (nucleotides, tRNA, amino acids, cofactors)
- Enzyme-substrate interactions, optimal temperature profiles, and template-primer binding kinetics determine yield
- Synthesis pathway interactions affect protein production efficiency"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting protein production yield using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "CASP" in dataset_name or ("Protein" in dataset_name and "Structure" in dataset_name):
            # Protein Structure (CASP)
            if feature_cols and len(feature_cols) >= 2:
                f1, f2 = feature_cols[0], feature_cols[1]
                penalty_feat = feature_cols[2] if len(feature_cols) > 2 else feature_cols[0]
                formula = f"ŷ = ({f1} + {f2}) / 2 - 0.1 * {penalty_feat}"
            else:
                formula = "ŷ = (sum of first 2 features) / 2 - 0.1 * structural_penalty_feature"
            
            dataset_desc = """PROTEIN STRUCTURE QUALITY DATASET (CASP):
This dataset predicts protein tertiary structure RMSD from physicochemical properties.
Key insights:
- Output depends on surface area properties and structural constraints
- Surface exposure and structural penalties affect RMSD
- Physicochemical properties influence protein structure quality"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting protein structure RMSD using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "Crop" in dataset_name:
            if feature_cols and len(feature_cols) >= 3:
                # Try to find nutrient-related features
                nutrient_feats = [f for f in feature_cols if any(n in f.lower() for n in ['nitrogen', 'phosphorus', 'potassium', 'n', 'p', 'k'])]
                climate_feats = [f for f in feature_cols if any(c in f.lower() for c in ['rainfall', 'temperature', 'temp', 'rain', 'climate'])]
                if nutrient_feats and climate_feats:
                    n1, n2 = nutrient_feats[0], nutrient_feats[1] if len(nutrient_feats) > 1 else nutrient_feats[0]
                    c1 = climate_feats[0]
                    formula = f"ŷ = ({n1} + {n2} + {c1}) / 3 + 0.15 * {n1} * {n2} - 0.1 * {c1}"
                else:
                    f1, f2, f3 = feature_cols[0], feature_cols[1], feature_cols[2]
                    formula = f"ŷ = ({f1} + {f2} + {f3}) / 3 + 0.15 * {f1} * {f2} - 0.1 * {f3}"
            else:
                formula = "ŷ = (sum of features) / (number of features) + 0.15 * feature1 * feature2 - 0.1 * feature3"
            
            dataset_desc = """CROP YIELD DATASET:
This dataset predicts agricultural crop yield from soil and climate conditions.
Key insights:
- Output depends on soil properties and climate factors
- Nutrient interactions and optimal growth conditions affect yield
- Environmental factors influence agricultural productivity"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting crop yield using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        elif "Drug" in dataset_name:
            if feature_cols and len(feature_cols) >= 4:
                # Try to find molecular property features
                mol_feats = [f for f in feature_cols if any(m in f.lower() for m in ['weight', 'logp', 'hbd', 'hba', 'tpsa', 'rot', 'aromatic'])]
                if len(mol_feats) >= 4:
                    f1, f2, f3, f4 = mol_feats[0], mol_feats[1], mol_feats[2], mol_feats[3]
                    feat_desc = f"({f1}, {f2}, {f3}, {f4})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4}) / 4 - 0.2 * {f2} + 0.1 * {f3} * {f4}"
                else:
                    f1, f2, f3, f4 = feature_cols[0], feature_cols[1], feature_cols[2], feature_cols[3]
                    feat_desc = f"({f1}, {f2}, {f3}, {f4})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4}) / 4 - 0.2 * {f2} + 0.1 * {f3} * {f4}"
            else:
                feat_desc = "molecular properties"
                formula = "ŷ = (sum of features) / (number of features) - 0.2 * logp + 0.1 * hbd * hba"
            return [
                f"Drug solubility mechanism: This dataset predicts aqueous drug solubility from molecular properties. Output depends on molecular properties {feat_desc}, with lipophilicity and hydrogen bonding determining aqueous solubility. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "Enzyme" in dataset_name or "enzyme" in dataset_name.lower() or "biochemical proxy" in dataset_name.lower():
            # Check if this is actually wine data (biochemical proxy) or real enzyme data
            is_wine_proxy = feature_cols and any(f in feature_cols for f in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
            is_real_enzyme = feature_cols and any(f in feature_cols for f in ['substrate_concentration', 'substrate', 'inhibitor_conc', 'inhibitor', 'cofactor_conc', 'cofactor'])
            
            if is_wine_proxy and not is_real_enzyme:
                # This is wine quality data used as enzyme proxy - use wine mechanism
                if feature_cols and len(feature_cols) >= 3:
                    # Try to find wine-related features
                    acid_feats = [f for f in feature_cols if any(a in f.lower() for a in ['acid', 'acidity', 'ph'])]
                    alcohol_feats = [f for f in feature_cols if 'alcohol' in f.lower()]
                    sugar_feats = [f for f in feature_cols if any(s in f.lower() for s in ['sugar', 'sweet', 'glucose', 'fructose'])]
                    
                    if acid_feats and alcohol_feats:
                        f1, f2 = acid_feats[0], alcohol_feats[0]
                        f3 = sugar_feats[0] if sugar_feats else feature_cols[2]
                        feat_desc = f"acidity ({f1}), alcohol content ({f2}), sugar levels ({f3})"
                        formula = f"ŷ = ({f1} + {f2} + {f3}) / {len(feature_cols)} + 0.15 * {f2} * {f1}"
                    else:
                        f1, f2, f3 = feature_cols[0], feature_cols[1], feature_cols[2]
                        feat_desc = f"({f1}, {f2}, {f3})"
                        formula = f"ŷ = ({f1} + {f2} + {f3}) / {len(feature_cols)} + 0.15 * {f2} * {f1}"
                else:
                    feat_desc = "acidity, alcohol content, sugar levels"
                    formula = "ŷ = (sum of features) / (number of features) + 0.15 * alcohol * acidity"
                
                dataset_desc = """WINE QUALITY DATASET (Biochemical Proxy):
This dataset predicts wine quality from chemical composition (used as biochemical proxy for enzyme kinetics).
Key insights:
- Output depends on chemical properties (acidity, alcohol, sugar, pH, etc.)
- Chemical balance and interactions affect quality
- Used as a proxy dataset for biochemical regression tasks"""
                
                mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting wine quality (biochemical proxy) using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
                
                return [mechanism_desc]
            else:
                # Real enzyme activity data - Enhanced with Michaelis-Menten kinetics
                if not feature_cols or len(feature_cols) < 3:
                    # Fallback to generic
                    if feature_cols and len(feature_cols) > 0:
                        top_features = feature_cols[:min(5, len(feature_cols))]
                        feature_sum = " + ".join(top_features)
                        n_features = len(top_features)
                        f1 = feature_cols[0]
                        f2 = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
                        formula = f"ŷ = ({feature_sum}) / {n_features} + 0.1 * {f1} * {f2}"
                    else:
                        formula = "ŷ = (sum of all input features) / (number of features) + 0.1 * feature1 * feature2"
                    
                    dataset_desc = f"""REGRESSION DATASET: {dataset_name}
This dataset predicts continuous numeric values from input features.
Key insights:
- Output depends on feature interactions and cross-feature effects
- Non-linear relationships may exist between features and target
- Domain-specific patterns influence predictions"""
                    
                    mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
                    
                    return [mechanism_desc]
                
                # Extract likely substrate, cofactor, temperature features
                substrate_feat = feature_cols[0]  # x1
                cofactor_feat = feature_cols[1]   # x2
                temp_feat = feature_cols[2] if len(feature_cols) > 2 else substrate_feat
                ph_feat = feature_cols[3] if len(feature_cols) > 3 else temp_feat
                inhibitor_feat = feature_cols[4] if len(feature_cols) > 4 else substrate_feat
                
                mechanism = f"""[KNOWN] ENZYME KINETICS MECHANISM

DATASET: Enzyme Activity Prediction

DOMAIN: Enzyme catalysis follows Michaelis-Menten kinetics with cofactor enhancement

MECHANISTIC PRINCIPLES:

1. MICHAELIS-MENTEN SATURATION:

   V_max * [S] / (K_m + [S])

   - Substrate binding saturates enzyme active sites

   - Hyperbolic response curve

   

2. COFACTOR ENHANCEMENT:

   Activity * (1 + k_cof * [Cofactor])

   - Cofactors (NAD+, ATP, Mg2+) enhance turnover

   - Linear or slightly cooperative

   

3. TEMPERATURE OPTIMUM:

   exp(-({temp_feat} - T_opt)^2 / (2*σ^2))

   - Activity peaks at optimal temperature (~310K = 37°C)

   - Drops outside due to denaturation

   

4. pH OPTIMUM:

   exp(-({ph_feat} - pH_opt)^2 / (2*σ^2))

   - Most enzymes optimal at pH 6-8

   - Protonation affects catalysis

   

5. COMPETITIVE INHIBITION:

   V_max * [S] / (K_m*(1 + [I]/K_i) + [S])

   - Inhibitor competes for active site

   - Increases apparent K_m

FEATURES ({len(feature_cols)} total):

{', '.join(feature_cols)}

KNOWN MECHANISM FORMULA:

ŷ = clip(

    # Base Michaelis-Menten

    6.0 * {substrate_feat} / (0.3 + {substrate_feat})

    

    # Cofactor enhancement

    * (1 + 0.5 * {cofactor_feat})

    

    # Temperature optimum (scaled range 0-10, optimum at 5)

    * exp(-({temp_feat} - 5.0)^2 / 8.0)

    

    # pH optimum (scaled range 0-10, optimum at 5)

    * exp(-({ph_feat} - 5.0)^2 / 8.0)

    

    # Competitive inhibition

    / (1 + 0.3 * {inhibitor_feat}),

    

    0, 10

)

INTERPRETATION:

- Term 1: Substrate saturation (Michaelis-Menten)

- Term 2: Cofactor increases turnover rate

- Term 3: Temperature optimum curve

- Term 4: pH optimum curve

- Term 5: Competitive inhibition reduces activity

This mechanism encodes fundamental enzyme biochemistry.

"""
                
                return [mechanism]
        
        elif "Material" in dataset_name:
            if feature_cols and len(feature_cols) >= 5:
                # Try to find composition and processing features
                comp_feats = [f for f in feature_cols if any(c in f.lower() for c in ['ratio', 'element', 'composition', 'comp'])]
                proc_feats = [f for f in feature_cols if any(p in f.lower() for p in ['temp', 'temperature', 'pressure', 'process'])]
                if comp_feats and proc_feats:
                    f1, f2 = comp_feats[0], comp_feats[1] if len(comp_feats) > 1 else comp_feats[0]
                    f3, f4 = proc_feats[0], proc_feats[1] if len(proc_feats) > 1 else proc_feats[0]
                    f5 = feature_cols[4] if len(feature_cols) > 4 else f4
                    feat_desc = f"elemental composition ({f1}, {f2}), processing conditions ({f3}, {f4})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4} + {f5}) / 5 + 0.15 * {f1} * {f2} - 0.1 * {f3}"
                else:
                    f1, f2, f3, f4, f5 = feature_cols[0], feature_cols[1], feature_cols[2], feature_cols[3], feature_cols[4]
                    feat_desc = f"({f1}, {f2}, {f3}, {f4}, {f5})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4} + {f5}) / 5 + 0.15 * {f1} * {f2} - 0.1 * {f3}"
            else:
                feat_desc = "elemental composition ratios, processing conditions"
                formula = "ŷ = (sum of features) / (number of features) + 0.15 * element_a * element_b - 0.1 * temperature"
            return [
                f"Material properties mechanism: This dataset predicts material properties from composition and processing conditions. Output depends on {feat_desc}, and crystal structure. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "Biomarker" in dataset_name:
            if feature_cols and len(feature_cols) >= 5:
                # Try to find clinical and lifestyle features
                clinical_feats = [f for f in feature_cols if any(c in f.lower() for c in ['age', 'bmi', 'glucose', 'cholesterol', 'blood', 'pressure', 'bp'])]
                if len(clinical_feats) >= 3:
                    f1, f2, f3 = clinical_feats[0], clinical_feats[1], clinical_feats[2]
                    f4 = clinical_feats[3] if len(clinical_feats) > 3 else feature_cols[3]
                    f5 = clinical_feats[4] if len(clinical_feats) > 4 else feature_cols[4]
                    feat_desc = f"clinical parameters ({', '.join(clinical_feats[:MAX_FEATURES_IN_DESCRIPTION])})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4} + {f5}) / 5 + 0.1 * {f3} * {f4} - 0.15 * {f2}"
                else:
                    f1, f2, f3, f4, f5 = feature_cols[0], feature_cols[1], feature_cols[2], feature_cols[3], feature_cols[4]
                    feat_desc = f"({f1}, {f2}, {f3}, {f4}, {f5})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4} + {f5}) / 5 + 0.1 * {f3} * {f4} - 0.15 * {f2}"
            else:
                feat_desc = "clinical parameters, lifestyle factors"
                formula = "ŷ = (sum of features) / (number of features) + 0.1 * glucose * cholesterol - 0.15 * bmi"
            return [
                f"Biomarker prediction mechanism: This dataset predicts biomarker levels from clinical and lifestyle parameters. Output depends on {feat_desc}, and genetic predisposition. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "GFP" in dataset_name:
            if feature_cols and len(feature_cols) >= 2:
                # Try to find reaction condition features
                temp_feats = [f for f in feature_cols if any(t in f.lower() for t in ['temp', 'temperature'])]
                comp_feats = [f for f in feature_cols if any(c in f.lower() for c in ['potassium', 'magnesium', 'ntp', 'dna', 'concentration'])]
                if temp_feats and comp_feats:
                    f1, f2 = temp_feats[0], comp_feats[0]
                    feat_desc = f"reaction conditions (temperature: {f1}, components: {', '.join(comp_feats[:MAX_FEATURES_IN_COMPONENT_LIST])})"
                    formula = f"ŷ = ({f1} + {f2} + sum of remaining) / {len(feature_cols)} + 0.2 * {f1} * {f2}"
                else:
                    f1, f2 = feature_cols[0], feature_cols[1]
                    feat_desc = f"({f1}, {f2})"
                    formula = f"ŷ = ({f1} + {f2} + sum of remaining) / {len(feature_cols)} + 0.2 * {f1} * {f2}"
            else:
                feat_desc = "reaction conditions"
                formula = "ŷ = (sum of features) / (number of features) + 0.2 * temperature * component"
            return [
                f"GFP yield mechanism: This dataset predicts GFP (green fluorescent protein) production yield from reaction conditions. Output depends on {feat_desc}, with enzyme-substrate interactions, optimal temperature profiles, and template-primer binding kinetics determining GFP protein production yield. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "Diabetes" in dataset_name:
            if feature_cols and len(feature_cols) >= 2:
                f1, f2 = feature_cols[0], feature_cols[1]
                feat_desc = ", ".join(feature_cols[:min(5, len(feature_cols))])
                formula = f"ŷ = (sum of all features) / {len(feature_cols)} + 0.1 * {f1} * {f2}"
            else:
                feat_desc = "clinical factors"
                formula = "ŷ = (sum of all features) / (number of features) + 0.1 * feature1 * feature2"
            return [
                f"Diabetes progression mechanism: This dataset predicts diabetes disease progression from clinical measurements. Output depends on multiple clinical factors ({feat_desc}) with interactions affecting disease progression. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "California" in dataset_name or "Housing" in dataset_name:
            if feature_cols and len(feature_cols) >= 4:
                # Try to find housing-related features
                income_feats = [f for f in feature_cols if any(i in f.lower() for i in ['income', 'median'])]
                age_feats = [f for f in feature_cols if 'age' in f.lower()]
                room_feats = [f for f in feature_cols if 'room' in f.lower()]
                pop_feats = [f for f in feature_cols if 'pop' in f.lower()]
                
                if income_feats and room_feats:
                    f1, f2 = income_feats[0], room_feats[0]
                    f3 = age_feats[0] if age_feats else feature_cols[2]
                    f4 = pop_feats[0] if pop_feats else feature_cols[3]
                    feat_desc = f"income ({f1}), age ({f3}), rooms ({f2}), population ({f4})"
                    formula = f"ŷ = ({f1} + {f3} + {f2} + {f4}) / 4 + 0.2 * {f1} * {f2}"
                else:
                    f1, f2, f3, f4 = feature_cols[0], feature_cols[1], feature_cols[2], feature_cols[3]
                    feat_desc = f"({f1}, {f2}, {f3}, {f4})"
                    formula = f"ŷ = ({f1} + {f2} + {f3} + {f4}) / 4 + 0.2 * {f1} * {f2}"
            else:
                feat_desc = "income, age, rooms, population"
                formula = "ŷ = (median_income + house_age + rooms + population) / 4 + 0.2 * median_income * rooms"
            return [
                f"Housing price mechanism: This dataset predicts California housing prices from location and property features. Output depends on {feat_desc} with interactions affecting housing values. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "Synthetic" in dataset_name and "Linear" in dataset_name:
            if feature_cols and len(feature_cols) > 0:
                feature_sum = " + ".join(feature_cols[:min(5, len(feature_cols))])
                n_features = len(feature_cols)
                formula = f"ŷ = ({feature_sum}) / {n_features}"
            else:
                formula = "ŷ = (sum of all input features) / (number of features)"
            return [
                f"Synthetic linear mechanism: This is a synthetic dataset with linear relationships between features and target. Output varies proportionally with inputs in a simple linear fashion. COMPUTE prediction as: {formula}. All inputs and outputs are normalized to {scale_range_str}. Return ŷ clipped to {scale_range_str}."
            ]
        
        elif "Synthetic" in dataset_name and "Nonlinear" in dataset_name:
            if feature_cols and len(feature_cols) > 0:
                feature_sum = " + ".join(feature_cols[:min(5, len(feature_cols))])
                n_features = len(feature_cols)
                formula = f"ŷ = sigmoid(({feature_sum}) / {n_features} - 0.5) * 0.5 + 0.5"
            else:
                formula = "ŷ = sigmoid((sum of all features) / (number of features) - 0.5) * 0.5 + 0.5"
            return [
                f"Synthetic nonlinear mechanism: This is a synthetic dataset with nonlinear relationships (sinusoidal and polynomial terms) between features and target. Output depends on nonlinear transformations of feature combinations. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "Wine" in dataset_name:
            if feature_cols and len(feature_cols) >= 3:
                # Try to find wine-related features
                acid_feats = [f for f in feature_cols if any(a in f.lower() for a in ['acid', 'acidity', 'ph'])]
                alcohol_feats = [f for f in feature_cols if 'alcohol' in f.lower()]
                sugar_feats = [f for f in feature_cols if any(s in f.lower() for s in ['sugar', 'sweet', 'glucose', 'fructose'])]
                
                if acid_feats and alcohol_feats:
                    f1, f2 = acid_feats[0], alcohol_feats[0]
                    f3 = sugar_feats[0] if sugar_feats else feature_cols[2]
                    feat_desc = f"acidity ({f1}), alcohol content ({f2}), sugar levels ({f3})"
                    formula = f"ŷ = ({f1} + {f2} + {f3}) / {len(feature_cols)} + 0.15 * {f2} * {f1}"
                else:
                    f1, f2, f3 = feature_cols[0], feature_cols[1], feature_cols[2]
                    feat_desc = f"({f1}, {f2}, {f3})"
                    formula = f"ŷ = ({f1} + {f2} + {f3}) / {len(feature_cols)} + 0.15 * {f2} * {f1}"
            else:
                feat_desc = "acidity, alcohol content, sugar levels"
                formula = "ŷ = (sum of features) / (number of features) + 0.15 * alcohol * acidity"
            return [
                f"Wine quality mechanism: This dataset predicts wine quality from chemical composition (used as biochemical proxy). Output depends on {feat_desc}, and other chemical properties with balance interactions affecting quality. COMPUTE prediction as: {formula}. All inputs/outputs in {scale_range_str}. Clip ŷ to {scale_range_str}."
            ]
        
        elif "Dataset 102" in dataset_name or "dataset_102" in dataset_name.lower():
            if feature_cols and all(f in feature_cols for f in ['nad', 'trna', 'aa']):
                # Use actual feature names
                nad, trna, aa = 'nad', 'trna', 'aa'
                pga = 'pga' if 'pga' in feature_cols else feature_cols[3] if len(feature_cols) > 3 else 'pga'
                folinic = 'folinic_acid' if 'folinic_acid' in feature_cols else feature_cols[4] if len(feature_cols) > 4 else 'folinic_acid'
                coa = 'coa' if 'coa' in feature_cols else feature_cols[5] if len(feature_cols) > 5 else 'coa'
                formula = f"ŷ = ({nad} + {trna} + {aa} + {pga} + {folinic}) / 5 + 0.2 * {nad} * {trna} - 0.15 * {coa}"
            else:
                if feature_cols and len(feature_cols) >= 3:
                    f1, f2, f3 = feature_cols[0], feature_cols[1], feature_cols[2]
                    formula = f"ŷ = ({f1} + {f2} + {f3}) / {len(feature_cols)} + 0.15 * {f1} * {f2} - 0.1 * {f3}"
                else:
                    formula = "ŷ = (sum of all input features) / (number of features) + 0.15 * feature1 * feature2 - 0.1 * feature3"
            
            dataset_desc = """PROTEIN EXPRESSION YIELD DATASET (Dataset 102):
This dataset predicts protein expression yield from cell-free expression systems with train/test split.
Key insights:
- Output depends on reaction components (nucleotides, tRNA, amino acids, cofactors)
- Synthesis pathway interactions determine protein production yield
- Enzyme-substrate interactions and optimal conditions affect yield"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task predicting protein expression yield using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]
        
        else:
            # Generic fallback - For DeepChem, use molecular properties; otherwise use actual feature names
            is_deepchem_fallback = is_deepchem_dataset(feature_cols)
            has_smiles_fallback = False
            if is_deepchem_fallback:
                X_original_check = X_train_original
                if X_original_check is None and predictor is not None:
                    X_original_check = getattr(predictor, 'X_train_original', None)
                has_smiles_fallback = has_smiles_data(X_original_check)
            
            # For DeepChem datasets, ALWAYS use molecular properties, never ECFP bits
            if is_deepchem_fallback:
                # For DeepChem: use molecular properties derived from SMILES, not ECFP bits
                formula = "ŷ = clip((molecular_weight(SMILES) + num_rings(SMILES) + num_hydroxyl_groups(SMILES)) / 3 + 0.1 * molecular_weight(SMILES) * num_rings(SMILES), 0, 10)"
            elif feature_cols and len(feature_cols) > 0:
                # Use first few features for formula (only for non-DeepChem datasets)
                top_features = feature_cols[:min(5, len(feature_cols))]
                feature_sum = " + ".join(top_features)
                n_features = len(top_features)
                f1 = feature_cols[0]
                f2 = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
                formula = f"ŷ = ({feature_sum}) / {n_features} + 0.1 * {f1} * {f2}"
            else:
                formula = "ŷ = (sum of all input features) / (number of features) + 0.1 * feature1 * feature2"
            
            # Use the dataset description we got earlier, or create a generic one
            if dataset_desc.startswith("Regression dataset:"):
                # Already have a generic regression description
                pass
            else:
                dataset_desc = f"""REGRESSION DATASET: {dataset_name}
This dataset predicts continuous numeric values from input features.
Key insights:
- Output depends on feature interactions and cross-feature effects
- Non-linear relationships may exist between features and target
- Domain-specific patterns influence predictions"""
            
            mechanism_desc = f"""[KNOWN] DATASET CONTEXT:
{dataset_desc}

TASK: Regression task using all available features.

{feature_info}

MECHANISM FORMULA:
COMPUTE prediction as: {formula}

CONSTRAINTS:
- All inputs are normalized to {scale_range_str}
- Output must be a numeric value in {scale_range_str}
- Use actual feature names from the dataset when referencing features
- Clip ŷ to {scale_range_str}"""
            
            return [mechanism_desc]

