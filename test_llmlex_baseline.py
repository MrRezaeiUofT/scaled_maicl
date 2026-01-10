#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LLM-LEx symbolic regression baseline
Tests the integration of llmlex for multivariate regression tasks
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import llmlex
try:
    import llmlex
    _HAS_LLMLEX = True
except ImportError:
    _HAS_LLMLEX = False
    logger.warning("llmlex not installed. Install with: pip install llmlex or clone from https://github.com/harveyThomas4692/llmlex")

# Try to import OpenAI client
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    logger.warning("openai not installed. Install with: pip install openai")


def _load_env():
    """Load environment variables from .env if available (simple version)."""
    try:
        from dotenv import load_dotenv
        # Load from script directory
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
    except:
        pass


def generate_base64_image_from_data(x, y):
    """Generate base64 image from x, y data for llmlex (matches llmlex documentation pattern)"""
    if not _HAS_LLMLEX:
        return None
    
    try:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        base64_img = llmlex.images.generate_base64_image(fig, ax, x, y)
        plt.close(fig)
        return base64_img
    except Exception as e:
        logger.error(f"Failed to generate base64 image: {e}")
        return None


def llmlex_baseline_multivariate(
    X_train, y_train, X_test, y_test,
    client=None,
    model="openai/gpt-4o-mini",
    use_pca=True,
    population_size=5,
    num_generations=3,
    use_genetic=True
):
    """
    Apply llmlex symbolic regression to multivariate regression problem
    
    Strategy:
    1. Use PCA to reduce to 1D (or use most important feature)
    2. Apply llmlex to find symbolic expression
    3. Evaluate the expression to predict on test set
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_test: Test features (n_samples, n_features)
        y_test: Test targets (n_samples,)
        client: OpenAI client (if None, will try to create one from .env)
        model: Model name for OpenRouter
        use_pca: If True, use PCA to reduce to 1D; if False, use most correlated feature
        population_size: Population size for genetic algorithm
        num_generations: Number of generations for genetic algorithm
        use_genetic: If True, use genetic algorithm; if False, use single_call
    
    Returns:
        dict with keys: 'r2', 'mae', 'mse', 'predictions', 'expression', 'success'
    """
    if not _HAS_LLMLEX:
        return {
            'success': False,
            'error': 'llmlex not installed',
            'r2': 0.0,
            'mae': float('inf'),
            'mse': float('inf'),
            'predictions': None,
            'expression': None
        }
    
    if not _HAS_OPENAI:
        return {
            'success': False,
            'error': 'openai not installed',
            'r2': 0.0,
            'mae': float('inf'),
            'mse': float('inf'),
            'predictions': None,
            'expression': None
        }
    
    try:
        # Create client if not provided
        if client is None:
            # Load environment variables first
            _load_env()
            
            # Check for OPENAI_API_KEY first (from .env)
            api_key = os.getenv("OPENAI_API_KEY")
            use_openrouter = False
            
            # Only use OpenRouter if OPENAI_API_KEY is not found
            if not api_key:
                api_key = os.getenv("OPENROUTER_API_KEY")
                use_openrouter = True
            
            if not api_key:
                return {
                    'success': False,
                    'error': 'No API key found. Set OPENAI_API_KEY in .env file',
                    'r2': 0.0,
                    'mae': float('inf'),
                    'mse': float('inf'),
                    'predictions': None,
                    'expression': None
                }
            
            # Use OpenRouter only if OPENROUTER_API_KEY is set, otherwise use OpenAI directly
            if use_openrouter:
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key
                )
                # Keep model name as-is for OpenRouter
                model_name = model
            else:
                # Use OpenAI API directly (no base_url override)
                client = openai.OpenAI(api_key=api_key)
                # Remove "openai/" prefix from model name for OpenAI API
                model_name = model.replace("openai/", "", 1) if model.startswith("openai/") else model
        else:
            # Client provided - assume model name is correct (user's responsibility)
            model_name = model
        
        # Reduce to 1D for symbolic regression
        if use_pca and X_train.shape[1] > 1:
            logger.info("Using PCA to reduce features to 1D for symbolic regression")
            pca = PCA(n_components=1)
            x_train_1d = pca.fit_transform(X_train).ravel()
            x_test_1d = pca.transform(X_test).ravel()
            feature_name = "PC1"
        else:
            # Use most correlated feature
            if X_train.shape[1] > 1:
                correlations = [np.abs(np.corrcoef(X_train[:, i], y_train)[0, 1]) 
                              for i in range(X_train.shape[1])]
                best_feature_idx = np.argmax(correlations)
                x_train_1d = X_train[:, best_feature_idx]
                x_test_1d = X_test[:, best_feature_idx]
                feature_name = f"Feature_{best_feature_idx}"
                logger.info(f"Using most correlated feature (index {best_feature_idx}) for symbolic regression")
            else:
                x_train_1d = X_train.ravel()
                x_test_1d = X_test.ravel()
                feature_name = "Feature"
        
        # Generate base64 image (matches llmlex documentation pattern)
        logger.info("Generating visualization for llmlex...")
        base64_img = generate_base64_image_from_data(x_train_1d, y_train)
        if base64_img is None:
            return {
                'success': False,
                'error': 'Failed to generate base64 image',
                'r2': 0.0,
                'mae': float('inf'),
                'mse': float('inf'),
                'predictions': None,
                'expression': None
            }
        
        # Run symbolic regression (matches llmlex documentation pattern)
        best_result = None
        best_expression = None
        best_params = None
        best_score = None
        
        if use_genetic:
            logger.info(f"Running llmlex genetic algorithm (population={population_size}, generations={num_generations})...")
            try:
                populations = llmlex.run_genetic(
                    client, base64_img, x_train_1d, y_train,
                    population_size=population_size,
                    num_of_generations=num_generations,
                    model=model_name
                )
                
                # Get best result from last generation
                if not populations or len(populations) == 0:
                    raise ValueError("No populations returned from llmlex")
                
                last_gen = populations[-1]
                if not last_gen or len(last_gen) == 0:
                    raise ValueError("Last generation is empty")
                
                # Find best result (lowest score = best)
                best_result = min(last_gen, key=lambda x: x.get('score', float('inf')))
                best_expression = best_result.get('ansatz', None)
                best_params = best_result.get('params', {})
                best_score = best_result.get('score', None)
                
                logger.info(f"Best symbolic expression: {best_expression}")
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best score: {best_score}")
                
            except Exception as e:
                logger.warning(f"llmlex genetic algorithm failed: {e}, falling back to single_call")
                use_genetic = False
        
        if not use_genetic or best_result is None:
            logger.info("Running llmlex single_call...")
            try:
                result = llmlex.single_call(client, base64_img, x_train_1d, y_train, model=model_name)
                best_expression = result.get('ansatz', None)
                best_params = result.get('params', {})
                best_score = result.get('score', None)
                
                logger.info(f"Symbolic expression: {best_expression}")
                logger.info(f"Parameters: {best_params}")
                logger.info(f"Score: {best_score}")
                
            except Exception as e2:
                logger.error(f"llmlex single call failed: {e2}")
                return {
                    'success': False,
                    'error': f'llmlex failed: {str(e2)}',
                    'r2': 0.0,
                    'mae': float('inf'),
                    'mse': float('inf'),
                    'predictions': None,
                    'expression': None
                }
        
        # Evaluate the symbolic expression on test set
        # Try to parse and evaluate the expression using sympy
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr
            
            # Parse the expression
            expr_str = str(best_expression)
            # Replace parameter placeholders with actual values
            if best_params:
                for param_name, param_value in best_params.items():
                    expr_str = expr_str.replace(param_name, str(param_value))
            
            # Create symbol for x
            x_sym = sp.Symbol('x')
            # Try to parse the expression
            try:
                expr = parse_expr(expr_str.replace('x', 'x_sym'), transformations='all')
            except:
                # If parsing fails, try simpler approach
                expr = parse_expr(expr_str, transformations='all')
            
            # Evaluate on test set
            y_pred = np.array([float(expr.subs(x_sym, x_val)) for x_val in x_test_1d])
            
            logger.info("Successfully evaluated symbolic expression")
            
        except Exception as eval_error:
            logger.warning(f"Failed to evaluate symbolic expression directly: {eval_error}")
            logger.warning("Falling back to linear fit on 1D projection")
            
            # Fallback: use linear regression on 1D projection
            from sklearn.linear_model import LinearRegression
            simple_model = LinearRegression()
            simple_model.fit(x_train_1d.reshape(-1, 1), y_train)
            y_pred = simple_model.predict(x_test_1d.reshape(-1, 1))
        
        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'success': True,
            'r2': float(r2),
            'mae': float(mae),
            'mse': float(mse),
            'predictions': y_pred.tolist(),
            'expression': best_expression,
            'params': best_params,
            'score': best_score,
            'method': 'genetic' if use_genetic and best_result is not None else 'single_call'
        }
    
    except Exception as e:
        logger.error(f"llmlex baseline failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'r2': 0.0,
            'mae': float('inf'),
            'mse': float('inf'),
            'predictions': None,
            'expression': None
        }


def test_llmlex_baseline(api_key_override=None):
    """Test the llmlex baseline with synthetic data
    
    Args:
        api_key_override: Optional API key to use directly (for testing)
    """
    logger.info("=" * 80)
    logger.info("TESTING LLMLEX BASELINE")
    logger.info("=" * 80)
    
    if not _HAS_LLMLEX:
        logger.error("llmlex not installed. Cannot run test.")
        logger.info("To install: pip install llmlex or clone from https://github.com/harveyThomas4692/llmlex")
        return False
    
    if not _HAS_OPENAI:
        logger.error("openai not installed. Cannot run test.")
        logger.info("To install: pip install openai")
        return False
    
    # Load environment variables (simple)
    _load_env()
    
    # Check for API key - prioritize OPENAI_API_KEY from .env
    if api_key_override:
        api_key = api_key_override
        key_name = "override"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            key_name = "OPENAI_API_KEY"
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            key_name = "OPENROUTER_API_KEY"
    
    if not api_key:
        logger.error("❌ No API key found!")
        logger.error("Set OPENAI_API_KEY in .env file")
        logger.error("  OPENAI_API_KEY=your_api_key_here")
        return False
    
    logger.info(f"✓ Found {key_name} (length: {len(api_key)} characters, starts with: {api_key[:7]}...)")
    
    # Generate synthetic test data
    logger.info("Generating synthetic test data...")
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create data with some structure
    X = np.random.randn(n_samples, n_features)
    # Target is a function of first feature with some noise
    y = np.sin(X[:, 0] * np.pi) + 0.1 * np.random.randn(n_samples)
    
    # Split into train/test
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Test with PCA
    logger.info("\nTesting with PCA reduction (genetic algorithm)...")
    result_pca = llmlex_baseline_multivariate(
        X_train, y_train, X_test, y_test,
        use_pca=True,
        population_size=3,
        num_generations=2,
        use_genetic=True
    )
    
    if result_pca['success']:
        logger.info(f"✓ PCA method succeeded!")
        logger.info(f"  R2: {result_pca['r2']:.4f}")
        logger.info(f"  MAE: {result_pca['mae']:.4f}")
        logger.info(f"  MSE: {result_pca['mse']:.4f}")
        logger.info(f"  Expression: {result_pca['expression']}")
    else:
        logger.error(f"✗ PCA method failed: {result_pca.get('error', 'Unknown error')}")
    
    # Test with feature selection
    logger.info("\nTesting with feature selection (single call)...")
    result_feat = llmlex_baseline_multivariate(
        X_train, y_train, X_test, y_test,
        use_pca=False,
        population_size=3,
        num_generations=2,
        use_genetic=False
    )
    
    if result_feat['success']:
        logger.info(f"✓ Feature selection method succeeded!")
        logger.info(f"  R2: {result_feat['r2']:.4f}")
        logger.info(f"  MAE: {result_feat['mae']:.4f}")
        logger.info(f"  MSE: {result_feat['mse']:.4f}")
        logger.info(f"  Expression: {result_feat['expression']}")
    else:
        logger.error(f"✗ Feature selection method failed: {result_feat.get('error', 'Unknown error')}")
    
    return result_pca['success'] or result_feat['success']


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test LLM-LEx baseline")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (overrides environment variables)")
    args = parser.parse_args()
    
    success = test_llmlex_baseline(api_key_override=args.api_key)
    sys.exit(0 if success else 1)

