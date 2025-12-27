#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone runner:
- loads a DeepChem *chemistry* regression dataset (default: ESOL/Delaney)
- forces generation of an LLM mechanism (via MA-ICL mechanism generator)
- extracts the single-line `Formula:` and validates it by executing on RDKit descriptors from SMILES

Key constraints (per request):
- Do NOT modify MA-ICL code. This script only patches prompt strings in-memory.
- Uses existing repo utilities: `evaluate_individual_mechanisms.py` for RDKit execution and metrics.

Example:
  python run_deepchem_llm_mechanism_rdkit_validate.py --dataset esol --max_samples 200 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import maicl_config
from maicl_lib_v2 import MinMaxScaler010, TrainableMAICL
from evaluate_individual_mechanisms import (
    extract_formula_from_llm_mechanism,
    evaluate_llm_formula,
    parse_mechanisms_file,
)
from evaluate_individual_mechanisms import _compute_molecular_properties  # RDKit-backed (returns zeros if RDKit missing)


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

def _extract_json_after_marker(text: str, marker: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object that appears after a marker line, e.g.:
      RDKitValidation:
      { ... }
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


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _sign_label(v: float, eps: float = 0.05) -> str:
    if v > eps:
        return "positive"
    if v < -eps:
        return "negative"
    return "neutral"

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


def _load_module_from_path(py_path: str, module_name: str) -> Any:
    """Import a Python file by absolute path (works even if filename starts with digits)."""
    # IMPORTANT: register in sys.modules before exec_module so code that does
    # `sys.modules[__name__]` (as in 018_maicl_regression_biotech.py) works.
    import sys
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


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
    Patch MA-ICL prompt strings for DeepChem molecular datasets so that the LLM:
    - emits a `Formula:` line that is executable by `evaluate_llm_formula()` (RDKit-backed)
    - only uses the RDKit descriptor names supported by `evaluate_individual_mechanisms.py`
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

    # Encoder: keep it tight and only mention supported descriptors.
    encoder_template = f"""You are a mechanism encoder for a DeepChem molecular regression dataset.
CRITICAL:
- You will receive SMILES strings as the human-readable input.
- The ML model uses ECFP fingerprints internally, but YOUR mechanism must use SMILES-derived molecular descriptors.
- Outputs are scaled to [0, 1]; the final mechanism output must be kept in [0, 1].

You may ONLY rely on these RDKit-computable descriptor names (either as variables or as function calls with SMILES):
{", ".join([f"{k}(SMILES)" for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES])}

ABSOLUTE PROHIBITIONS:
- Do NOT use ECFP bits or any identifier like ecfp_bit_123.
- Do NOT use descriptors that are not in the whitelist above (e.g., logp, tpsa, qed, num_aromatic_rings, num_carbon, fragments, substructure SMARTS).

Think about (a) polarity vs hydrophobicity, (b) size/complexity penalties, (c) saturating effects,
and (d) at least one nonlinear interaction between polarity and hydrophobic bulk.

Be concise (3-5 sentences)."""

    # Dataset-specific decoder guidance (still obeying the strict descriptor whitelist).
    # ESOL: higher polarity -> higher solubility; LIPO: higher hydrophobe -> higher LogP.
    if dataset_key in ("lipo", "lipophilicity"):
        domain_guidance = """DATASET CONTEXT (LogP / lipophilicity):
- Target is scaled to [0,1]. Higher values mean MORE lipophilic (more hydrophobic / partitions into octanol).
- Hydrophobicity tends to INCREASE with: num_rings, is_aromatic, num_halogen, molecular_weight, num_atoms.
- Polarity tends to DECREASE LogP (counterweight) with: num_oxygen, num_nitrogen, num_hydroxyl_groups, num_hbd.
- A good interaction: polarity should *dampen* hydrophobic gain (e.g., hydrophobe/(K+hydrophobe) multiplied by 1/(1+alpha*polarity)).
COEFFICIENT HINTS (scaled output):
- Use small per-count coefficients (0.02–0.20); avoid exploding with raw molecular_weight (normalize it, e.g., molecular_weight/300).

RDKit VALIDATION (CRITICAL):
- You MUST include a section exactly titled:
  RDKitValidation:
  { ...valid JSON... }
- The JSON must include an 'expected_effects' object mapping ONLY these descriptor names to one of:
  "positive", "negative", or "neutral"
  Allowed keys: molecular_weight, num_rings, num_hydroxyl_groups, num_halogen, num_nitrogen, num_oxygen, num_atoms, is_aromatic, num_hbd
- For this dataset, a reasonable starting hypothesis is:
  hydrophobe descriptors -> positive, polarity descriptors -> negative
  (but you can deviate if you justify it)
Example:
RDKitValidation:
{"expected_effects":{"num_rings":"positive","is_aromatic":"positive","num_halogen":"positive","molecular_weight":"positive","num_atoms":"positive","num_oxygen":"negative","num_nitrogen":"negative","num_hydroxyl_groups":"negative","num_hbd":"negative"}}
"""
    else:
        domain_guidance = """DATASET CONTEXT (ESOL / solubility):
- Target is scaled to [0,1]. Higher values mean MORE soluble (more polar / better water interactions).
- Solubility tends to INCREASE with: num_oxygen, num_nitrogen, num_hydroxyl_groups, num_hbd.
- Solubility tends to DECREASE with: num_rings, is_aromatic, num_halogen, molecular_weight, num_atoms.
- A good interaction: hydrophobic bulk should dampen the benefit of polarity (or vice versa).
COEFFICIENT HINTS (scaled output):
- Use small per-count coefficients (0.02–0.25); normalize molecular_weight, e.g., molecular_weight/300.

RDKit VALIDATION (CRITICAL):
- You MUST include a section exactly titled:
  RDKitValidation:
  { ...valid JSON... }
- The JSON must include an 'expected_effects' object mapping ONLY these descriptor names to one of:
  "positive", "negative", or "neutral"
  Allowed keys: molecular_weight, num_rings, num_hydroxyl_groups, num_halogen, num_nitrogen, num_oxygen, num_atoms, is_aromatic, num_hbd
- For this dataset, a reasonable starting hypothesis is:
  polarity descriptors -> positive, hydrophobe descriptors -> negative
Example:
RDKitValidation:
{"expected_effects":{"num_oxygen":"positive","num_nitrogen":"positive","num_hydroxyl_groups":"positive","num_hbd":"positive","num_rings":"negative","is_aromatic":"negative","num_halogen":"negative","molecular_weight":"negative","num_atoms":"negative"}}
"""

    # Decoder: enforce the Formula line format + descriptor whitelist.
    decoder_template = f"""You are a mechanism decoder for a DeepChem molecular regression dataset.

HARD REQUIREMENTS:
1) Your mechanism MUST be interpretable and grounded in chemistry.
2) Your final output MUST include exactly ONE single-line formula starting with:
   Formula: ŷ = <python-math-expression>
3) The formula MUST be executable by a simple evaluator. Use only:
   - arithmetic: + - * / ( )
   - math functions: abs, min, max, round, sqrt, exp, log, sin, cos, tan, pow
   - clip(x, 0.0, 1.0) to keep the output in [0, 1]
4) You may ONLY use these SMILES-derived descriptor names (as variables or as function calls with SMILES):
   {", ".join([f"{k}(SMILES)" for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES])}
   DO NOT invent any other descriptors.
   ABSOLUTE PROHIBITIONS (will be rejected): any token like ecfp_bit_*, any fingerprint bits, logp, tpsa, qed,
   fragments/substructure SMARTS, or any descriptor name not in the whitelist above.
5) Include at least:
   - one nonlinearity (e.g., saturation x/(K+x), log(1+x), exp(-x))
   - one interaction term (e.g., (polarity * hydrophobe)/(K + polarity * hydrophobe))

RECOMMENDED CHEMISTRY INTUITION (generic):
- Polarity proxies: num_oxygen, num_nitrogen, num_hydroxyl_groups, num_hbd
- Hydrophobic/complexity proxies: num_rings, is_aromatic, molecular_weight, num_atoms, num_halogen

HIGH-SIGNAL FORMULA SKELETONS (pick ONE skeleton and tune coefficients; keep it simple and stable):
1) ESOL-like (solubility ↑ with polarity, ↓ with hydrophobe), with saturation + interaction:
   ŷ = clip(b
            + aP * (P/(1+P))
            - aH * (H/(1+H))
            + aI * (P*H)/(1 + P*H), 0.0, 1.0)
   where P = num_hydroxyl_groups(SMILES) + 0.6*num_oxygen(SMILES) + 0.4*num_nitrogen(SMILES) + 0.8*num_hbd(SMILES)
         H = 0.7*num_rings(SMILES) + 0.8*is_aromatic(SMILES) + 0.6*num_halogen(SMILES) + 0.8*(molecular_weight(SMILES)/300)

2) LIPO-like (lipophilicity ↑ with hydrophobe, ↓ with polarity), with damping:
   ŷ = clip(b
            + aH * (H/(1+H)) * (1/(1 + aD*P))
            - aP * (P/(1+P)), 0.0, 1.0)

3) Size penalty with diminishing returns:
   ŷ = clip(b + a1*(P/(1+P)) - a2*log(1 + H) - a3*log(1 + molecular_weight(SMILES)/200), 0.0, 1.0)

{domain_guidance}

OUTPUT FORMAT (keep it short):
MECHANISM DESCRIPTION: 3-6 sentences.
INTERMEDIATE CONCEPTS: 1-3 bullets (name + meaning).
SANITY CHECKS: 2-4 bullets.
RDKitValidation:
<one JSON object, see constraints above>
Formula: ŷ = clip(<SINGLE LINE expression; no newlines>, 0.0, 1.0)

VARIATION TAG (use this only to pick a different *shape* of formula; do not mention it explicitly in output):
{variation_tag}
"""

    for key in target_keys:
        # Only patch keys that exist in config OR ones we created above.
        ds_specific[key]["encoder"] = encoder_template
        ds_specific[key]["decoder"] = decoder_template

    # Commit patched config into the module-global cache.
    maicl_config._MAICL_CONFIG = cfg  # type: ignore[attr-defined]


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


@dataclass
class RunOutputs:
    output_dir: str
    mechanism_text: str
    extracted_formula: str
    metrics: Dict[str, float]


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepChem MA-ICL: generate LLM mechanism and validate via RDKit execution")
    ap.add_argument("--dataset", type=str, default="esol", choices=["esol", "delaney", "lipo", "lipophilicity"],
                    help="DeepChem chemistry regression dataset to use")
    ap.add_argument("--model_name", type=str, default=os.environ.get("MAICL_MODEL_NAME", "gemini-2.0-flash"),
                    help="Gemini model name (must have GOOGLE_API_KEY env var set)")
    ap.add_argument("--max_samples", type=int, default=200, help="Max TRAIN samples (applied to DeepChem train split only)")
    ap.add_argument("--iterations", type=int, default=5,
                    help="MA-ICL learning iterations (runs TrainableMAICL.train).")
    ap.add_argument("--acceptance_set", type=str, default="test", choices=["validation", "train", "test"],
                    help="Which set MA-ICL uses for acceptance during training (default: validation).")
    ap.add_argument("--k_shot", type=int, default=0, help="Few-shot examples per prediction inside MA-ICL train/eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=None, help="Where to write artifacts (default: ma_icl_results/rdkit_validate_<dataset>_<timestamp>)")
    ap.add_argument("--deepchem_splitter", type=str, default="random", help="DeepChem MolNet splitter (random/scaffold/...)")
    ap.add_argument("--deepchem_featurizer", type=str, default="ECFP", help="DeepChem featurizer for ML side (kept for loader compatibility)")
    args = ap.parse_args()

    np.random.seed(int(args.seed))

    # Patch prompts so the produced mechanism is RDKit-evaluable.
    # (We may repatch with a small variation tag per iteration.)
    _patch_deepchem_prompts_in_memory(dataset_key=str(args.dataset).lower())

    # Load 018 runner as a module by path, and reuse its DeepChem loader + Gemini LLM factory.
    here = Path(__file__).resolve().parent
    runner_path = str(here / "018_maicl_regression_biotech.py")
    runner = _load_module_from_path(runner_path, module_name="_maicl_regression_biotech_018")

    # Create output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = str(here / "ma_icl_results" / f"rdkit_validate_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Get LLM client (requires API key env vars)
    llm = runner._get_gemini_llm(args.model_name)

    # Load DeepChem official splits (train/valid/test) with SMILES in X_original.
    (
        (X_train, y_train, X_original_train),
        (X_val, y_val, X_original_val),
        (X_test, y_test, X_original_test),
        feature_cols,
        feature_encoders,
        ds_label,
    ) = runner.load_deepchem_regression_dataset(
        args.dataset,
        int(args.max_samples),
        featurizer=args.deepchem_featurizer,
        splitter=args.deepchem_splitter,
        return_splits=True,
    )

    # Scale y to [0,1] (match your 018 default behavior) so the Formula can clip to [0,1].
    y_scaler = MinMaxScaler010()
    y_train_s = y_scaler.fit_transform(np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(np.asarray(y_test, dtype=float).reshape(-1, 1)).ravel()
    # DeepChem official splits can contain targets outside the TRAIN min/max; MinMax scaling can then yield
    # slightly <0 or >1 on val/test. MA-ICL's scaling verifier expects [0,1] (with small tolerance).
    y_val_s = np.clip(y_val_s, 0.0, 1.0)
    y_test_s = np.clip(y_test_s, 0.0, 1.0)

    # We don't need to train ML here; we only need the LLM mechanism + RDKit evaluation.
    # Use MA-ICL in LLM-only mode.
    maicl = TrainableMAICL(
        batched_llm=llm,
        feature_cols=feature_cols,
        scaler=None,  # ECFP scaling is irrelevant for RDKit formula evaluation
        use_ml_mechanism=False,
        dataset_name=ds_label,
        # IMPORTANT: In this repo version, TrainableMAICL.__init__ references a missing helper
        # `_scaled_range_from()` when y_scaler is not None. The main runners typically pass y_scaler=None.
        # We keep y scaling for *evaluation* in this script (y_*_s), and enforce [0,1] output via prompts.
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
    maicl.mech_generator.X_train_original = X_original_train

    # ---------------------------------------------------------------------
    # DeepChem safety shim (NO MA-ICL CODE CHANGES):
    # MA-ICL's `train()` calls `self.evaluate(...)` in a few places without forwarding
    # the DeepChem `X_original` kwargs, which triggers a hard error in evaluate().
    # We monkeypatch the instance method to auto-inject matching SMILES lists.
    # ---------------------------------------------------------------------
    _orig_evaluate = maicl.evaluate

    def _evaluate_with_smiles_autofill(X, y_true, X_pool, y_pool, *args, **kwargs):
        is_deepchem = bool(maicl.feature_cols) and all(
            str(f).startswith("ecfp_bit_") for f in maicl.feature_cols[: min(10, len(maicl.feature_cols))]
        )
        if is_deepchem:
            if kwargs.get("X_original", None) is None:
                # Pick the stored original list that matches the current X length.
                candidates = [
                    getattr(maicl, "X_val_original", None),
                    getattr(maicl, "X_train_original", None),
                    getattr(maicl, "X_test_original", None),
                ]
                for c in candidates:
                    if c is not None and hasattr(c, "__len__") and len(c) == len(X):
                        kwargs["X_original"] = c
                        break
            if kwargs.get("X_pool_original", None) is None:
                c = getattr(maicl, "X_train_original", None)
                if c is not None and hasattr(c, "__len__") and len(c) == len(X_pool):
                    kwargs["X_pool_original"] = c
        return _orig_evaluate(X, y_true, X_pool, y_pool, *args, **kwargs)

    maicl.evaluate = _evaluate_with_smiles_autofill  # type: ignore[assignment]

    # Build a simple "error" signal to trigger mechanism generation (no ML residuals here).
    # Use deviation from mean as a proxy difficulty score.
    prediction_errors = np.abs(y_train_s - float(np.mean(y_train_s)))

    # ----------------------------
    # RUN MA-ICL LEARNING PROCESS
    # ----------------------------
    # This runs the full TextGrad-driven mechanism update loop and writes:
    #   mechanisms_iter_{i}.txt
    # into the output_dir via visualization.persist_iteration_artifacts().
    maicl.train(
        X_train,
        y_train_s,
        X_val,
        y_val_s,
        iterations=int(max(1, args.iterations)),
        ml_residuals=None,
        accept_eval_max=None,
        X_test=X_test,
        y_test=y_test_s,
        acceptance_set=str(args.acceptance_set),
        k_shot=int(args.k_shot),
        X_train_original=X_original_train,
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
        raise SystemExit(f"No mechanisms found in {mechanisms_file}")

    # Pick the best mechanism by:
    # 1) RDKitValidation sign-match score (preferred, formula not required conceptually)
    # 2) If tie/absent, fall back to formula-based RDKit execution MAE on validation split.
    # 3) Prefer LLM mechanisms over KNOWN mechanisms when metrics are similar
    best_mech = None
    best_match_rate = -1.0
    best_mech_val_mae = float("inf")
    best_mech_val_metrics = None
    best_mech_rdkit_validation_report = None
    best_mech_type = None

    for mech_type, mech_text in mechs:
        if mech_type.lower() not in ("llm", "known"):
            continue
        # 1) RDKitValidation block (descriptor sign expectations)
        rdv = _extract_json_after_marker(mech_text, marker="RDKitValidation")
        expected = None
        if isinstance(rdv, dict):
            expected = rdv.get("expected_effects")
        report = None
        match_rate = -1.0
        if isinstance(expected, dict) and expected:
            # Compute descriptor correlations on validation split using RDKit.
            desc = _rdkit_descriptor_matrix_from_smiles_dicts(X_original_val)
            corr = {k: _pearson_corr(desc[k], y_val_s) for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES}
            per_key = {}
            n = 0
            n_match = 0
            for k, exp_sign in expected.items():
                if k not in corr:
                    continue
                exp_sign_str = str(exp_sign).strip().lower()
                if exp_sign_str not in ("positive", "negative", "neutral"):
                    continue
                got = _sign_label(corr[k])
                ok = (got == exp_sign_str) or (exp_sign_str == "neutral" and got == "neutral")
                per_key[k] = {"expected": exp_sign_str, "corr": float(corr[k]), "observed": got, "match": bool(ok)}
                n += 1
                n_match += 1 if ok else 0
            if n > 0:
                match_rate = float(n_match / n)
                report = {"match_rate": match_rate, "per_descriptor": per_key}

        # 2) Optional formula metrics (still useful)
        formula = extract_formula_from_llm_mechanism(mech_text) or ""
        formula = _sanitize_formula_single_line(formula)
        val_metrics = None
        if formula and not _find_unsupported_descriptor_mentions(formula):
            y_pred_val = evaluate_llm_formula(
                formula=formula,
                X=np.asarray(X_val, dtype=float),
                feature_cols=feature_cols,
                scaler=None,
                y_scaler=None,
                X_original=X_original_val,
            ).astype(float)
            val_metrics = {
                "r2": float(r2_score(y_val_s, y_pred_val)),
                "mae": float(mean_absolute_error(y_val_s, y_pred_val)),
                "mse": float(mean_squared_error(y_val_s, y_pred_val)),
                "rmse": float(np.sqrt(mean_squared_error(y_val_s, y_pred_val))),
            }

        # Selection: prefer higher match_rate; break ties via lower val_mae if available.
        # Also prefer LLM over KNOWN when metrics are similar (within 5% MAE difference)
        val_mae = float(val_metrics["mae"]) if isinstance(val_metrics, dict) and "mae" in val_metrics else float("inf")
        val_r2 = float(val_metrics["r2"]) if isinstance(val_metrics, dict) and "r2" in val_metrics else float("-inf")
        
        better = False
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
            best_mech_val_metrics = val_metrics
            best_mech_rdkit_validation_report = report
            best_mech_type = mech_type

    if best_mech is None:
        raise SystemExit(
            f"MA-ICL trained, but no usable LLM mechanism could be selected from {mechanisms_file}"
        )

    # Evaluate chosen mechanism on test split:
    # - Always compute RDKitValidation sign-match report (if present)
    # - Compute formula metrics if formula exists
    test_metrics = None
    if best_mech.get("formula"):
        y_pred_test = evaluate_llm_formula(
            formula=best_mech["formula"],
            X=np.asarray(X_test, dtype=float),
            feature_cols=feature_cols,
            scaler=None,
            y_scaler=None,
            X_original=X_original_test,
        ).astype(float)
        test_metrics = {
            "r2": float(r2_score(y_test_s, y_pred_test)),
            "mae": float(mean_absolute_error(y_test_s, y_pred_test)),
            "mse": float(mean_squared_error(y_test_s, y_pred_test)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_s, y_pred_test))),
        }

    # Also compute a test-side RDKitValidation report if we have one on validation.
    best_mech_rdkit_validation_report_test = None
    rdv_best = _extract_json_after_marker(best_mech["text"], marker="RDKitValidation")
    expected_best = rdv_best.get("expected_effects") if isinstance(rdv_best, dict) else None
    if isinstance(expected_best, dict) and expected_best:
        desc_te = _rdkit_descriptor_matrix_from_smiles_dicts(X_original_test)
        corr_te = {k: _pearson_corr(desc_te[k], y_test_s) for k in SUPPORTED_RDKIT_DESCRIPTOR_NAMES}
        per_key_te = {}
        n = 0
        n_match = 0
        for k, exp_sign in expected_best.items():
            if k not in corr_te:
                continue
            exp_sign_str = str(exp_sign).strip().lower()
            if exp_sign_str not in ("positive", "negative", "neutral"):
                continue
            got = _sign_label(corr_te[k])
            ok = (got == exp_sign_str) or (exp_sign_str == "neutral" and got == "neutral")
            per_key_te[k] = {"expected": exp_sign_str, "corr": float(corr_te[k]), "observed": got, "match": bool(ok)}
            n += 1
            n_match += 1 if ok else 0
        if n > 0:
            best_mech_rdkit_validation_report_test = {"match_rate": float(n_match / n), "per_descriptor": per_key_te}

    # Save stable artifacts
    with open(os.path.join(out_dir, "llm_mechanism_best.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["text"].strip() + "\n")
    with open(os.path.join(out_dir, "extracted_formula_best.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["formula"].strip() + "\n")
    with open(os.path.join(out_dir, "llm_mechanism.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["text"].strip() + "\n")
    with open(os.path.join(out_dir, "extracted_formula.txt"), "w", encoding="utf-8") as f:
        f.write(best_mech["formula"].strip() + "\n")

    with open(os.path.join(out_dir, "rdkit_validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": ds_label,
                "dataset_key": args.dataset,
                "model_name": args.model_name,
                "splitter": args.deepchem_splitter,
                "max_samples_train": int(args.max_samples),
                "train_iterations": int(max(1, args.iterations)),
                "acceptance_set": str(args.acceptance_set),
                "k_shot": int(args.k_shot),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "snapshot_iteration_used": best_iter,
                "selected_mechanism_type": str(best_mech["type"]),
                "formula": best_mech["formula"],
                "val_metrics_scaled_y": best_mech_val_metrics,
                "test_metrics_scaled_y": test_metrics,
                "rdkit_validation_signcheck_val": best_mech_rdkit_validation_report,
                "rdkit_validation_signcheck_test": best_mech_rdkit_validation_report_test,
                "selection": {
                    "match_rate_val": best_match_rate,
                    "tie_breaker_val_mae": best_mech_val_mae,
                },
            },
            f,
            indent=2,
        )

    print("\n=== MA-ICL Training + RDKit Validation Complete ===")
    print(f"Output dir: {out_dir}")
    print(f"Snapshot iteration used: {best_iter}")
    if best_mech_rdkit_validation_report is not None:
        print(f"RDKit sign-check (validation) match_rate: {best_mech_rdkit_validation_report.get('match_rate'):.3f}")
    if best_mech.get("formula"):
        print("Extracted formula:")
        print(best_mech["formula"])
        if best_mech_val_metrics:
            print("Validation metrics (scaled y in [0,1]):")
            for k, v in best_mech_val_metrics.items():
                print(f"  {k}: {v:.6f}")
        if test_metrics:
            print("Test metrics (scaled y in [0,1]):")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()


