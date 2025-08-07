#!/usr/bin/env python
"""
Run experiments on synthetic data with known ground truth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import argparse
import json
from datetime import datetime

from src.causal_ddpm import CausalConfig, CausalInferenceFramework
from src.utils import generate_synthetic_data, visualize_effects, check_balance
from src.evaluation import evaluate_ate, evaluate_cate, compare_methods

# For baseline comparisons
try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("Warning: scikit-learn not available for baseline comparisons")


def propensity_score_matching(df, treatment_col, outcome_col, confounder_cols):
    """Simple propensity score matching baseline"""
    if not BASELINES_AVAILABLE:
        return None, None
    
    X = df[confounder_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, T)
    propensity = ps_model.predict_proba(X)[:, 1]
    
    # Simple 1:1 matching
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    matched_effects = []
    for t_idx in treated_idx:
        # Find closest match in control group
        distances = np.abs(propensity[control_idx] - propensity[t_idx])
        best_match = control_idx[np.argmin(distances)]
        
        # Calculate effect
        effect = Y[t_idx] - Y[best_match]
        matched_effects.append(effect)
    
    ate = np.mean(matched_effects)
    ate_std = np.std(matched_effects)
    
    return ate, ate_std


def linear_regression_baseline(df, treatment_col, outcome_col, confounder_cols):
    """Linear regression with treatment and confounders"""
    if not BASELINES_AVAILABLE:
        return None, None
    
    X = df[confounder_cols].values
    T = df[treatment_col].values.reshape(-1, 1)
    Y = df[outcome_col].values
    
    # Combine treatment and confounders
    X_combined = np.hstack([T, X])
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X_combined, Y)
    
    # Treatment effect is the coefficient for treatment
    ate = model.coef_[0]
    
    # Bootstrap for standard error
    bootstrap_effects = []
    for _ in range(100):
        idx = np.random.choice(len(df), len(df), replace=True)
        X_boot = X_combined[idx]
        Y_boot = Y[idx]
        model_boot = LinearRegression()
        model_boot.fit(X_boot, Y_boot)
        bootstrap_effects.append(model_boot.coef_[0])
    
    ate_std = np.std(bootstrap_effects)
    
    return ate, ate_std


def main(args):
    print("="*80)
    print("DDPM Causal Inference - Synthetic Data Experiment")
    print("="*80)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(
        n_samples=args.n_samples,
        treatment_effect=args.true_effect,
        heterogeneous=args.heterogeneous,
        seed=args.seed
    )
    
    print(f"   - Samples: {len(df)}")
    print(f"   - Treatment rate: {df['treatment'].mean():.2%}")
    print(f"   - True ATE: {args.true_effect}")
    
    # Check covariate balance
    print("\n2. Checking covariate balance...")
    balance = check_balance(df, 'treatment', ['confounder_1', 'confounder_2', 'confounder_3'])
    print(balance)
    
    # Initialize DDPM framework
    print("\n3. Training DDPM model...")
    config = CausalConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps
    )
    
    framework = CausalInferenceFramework(config)
    
    # Prepare data
    dataset = framework.prepare_data(
        df,
        treatment_col='treatment',
        outcome_col='outcome',
        confounder_cols=['confounder_1', 'confounder_2', 'confounder_3']
    )
    
    # Train model
    framework.train(dataset)
    
    # Estimate effects
    print("\n4. Estimating treatment effects...")
    ate_ddpm, ate_std_ddpm = framework.estimate_ate(dataset, n_samples=args.n_mc_samples)
    
    print(f"\n   DDPM Results:")
    print(f"   - Estimated ATE: {ate_ddpm:.3f} ± {ate_std_ddpm:.3f}")
    print(f"   - True ATE: {args.true_effect:.3f}")
    print(f"   - Absolute Error: {abs(ate_ddpm - args.true_effect):.3f}")
    
    results = {
        'method': 'DDPM',
        'ate': ate_ddpm,
        'ate_std': ate_std_ddpm,
        'true_ate': args.true_effect,
        'absolute_error': abs(ate_ddpm - args.true_effect)
    }
    
    # Run baseline methods
    all_results = {'DDPM': evaluate_ate(args.true_effect, ate_ddpm, ate_std_ddpm)}
    
    if args.run_baselines and BASELINES_AVAILABLE:
        print("\n5. Running baseline methods...")
        
        # Propensity Score Matching
        ate_psm, ate_std_psm = propensity_score_matching(
            df, 'treatment', 'outcome', ['confounder_1', 'confounder_2', 'confounder_3']
        )
        if ate_psm is not None:
            print(f"\n   PSM Results:")
            print(f"   - Estimated ATE: {ate_psm:.3f} ± {ate_std_psm:.3f}")
            print(f"   - Absolute Error: {abs(ate_psm - args.true_effect):.3f}")
            all_results['PSM'] = evaluate_ate(args.true_effect, ate_psm, ate_std_psm)
        
        # Linear Regression
        ate_lr, ate_std_lr = linear_regression_baseline(
            df, 'treatment', 'outcome', ['confounder_1', 'confounder_2', 'confounder_3']
        )
        if ate_lr is not None:
            print(f"\n   Linear Regression Results:")
            print(f"   - Estimated ATE: {ate_lr:.3f} ± {ate_std_lr:.3f}")
            print(f"   - Absolute Error: {abs(ate_lr - args.true_effect):.3f}")
            all_results['LinearReg'] = evaluate_ate(args.true_effect, ate_lr, ate_std_lr)
    
    # Compare methods
    if len(all_results) > 1:
        print("\n6. Method Comparison:")
        comparison_df = compare_methods(all_results)
        print(comparison_df[['method', 'estimated_ate', 'absolute_error', 'bias']])
    
    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/synthetic_experiment_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'args': vars(args),
                'results': {k: {kk: float(vv) if isinstance(vv, (np.number, float)) else vv 
                              for kk, vv in v.items()} 
                          for k, v in all_results.items()}
            }, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDPM causal inference on synthetic data")
    
    # Data parameters
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--true_effect', type=float, default=3.0,
                       help='True treatment effect')
    parser.add_argument('--heterogeneous', action='store_true',
                       help='Include heterogeneous treatment effects')
    
    # Model parameters
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                       help='Number of diffusion steps')
    parser.add_argument('--n_mc_samples', type=int, default=100,
                       help='Number of Monte Carlo samples for inference')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--run_baselines', action='store_true',
                       help='Run baseline methods for comparison')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    main(args)