#!/usr/bin/env python
"""
Analyze AI adoption effects using DDPM causal inference
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
from src.utils import prepare_economic_data, visualize_effects, check_balance, plot_propensity_overlap
from src.evaluation import evaluate_ate, bootstrap_confidence_interval


def generate_ai_adoption_data(n_industries=500, seed=42):
    """
    Generate synthetic AI adoption data mimicking real economic patterns
    """
    np.random.seed(seed)
    
    # Industry characteristics
    industry_size = np.random.lognormal(10, 2, n_industries)  # Company size (log-scale)
    rd_intensity = np.random.beta(2, 5, n_industries)  # R&D spending as % of revenue
    workforce_education = np.random.beta(3, 2, n_industries)  # % with higher education
    capital_intensity = np.random.gamma(2, 2, n_industries)  # Capital per worker
    tech_readiness = np.random.beta(2, 3, n_industries)  # Technology infrastructure
    
    # AI adoption is not random - depends on industry characteristics
    adoption_propensity = (
        0.3 * (industry_size - industry_size.min()) / (industry_size.max() - industry_size.min()) +
        0.25 * rd_intensity +
        0.2 * workforce_education +
        0.15 * (capital_intensity - capital_intensity.min()) / (capital_intensity.max() - capital_intensity.min()) +
        0.1 * tech_readiness
    )
    
    # Add noise and create binary adoption decision
    adoption_propensity = 1 / (1 + np.exp(-5 * (adoption_propensity - 0.5)))
    ai_adoption = np.random.binomial(1, adoption_propensity)
    
    # Productivity outcome (with heterogeneous treatment effect)
    base_productivity = (
        np.log(industry_size) * 0.5 +
        rd_intensity * 2 +
        workforce_education * 1.5 +
        np.log(capital_intensity + 1) * 0.3 +
        tech_readiness * 1
    )
    
    # AI effect varies by industry characteristics
    ai_effect = ai_adoption * (1.5 + 0.5 * workforce_education + 0.3 * tech_readiness)
    
    # Final productivity with noise
    productivity = base_productivity + ai_effect + np.random.normal(0, 0.3, n_industries)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ai_adoption': ai_adoption,
        'productivity': productivity,
        'industry_size': industry_size,
        'rd_intensity': rd_intensity,
        'workforce_education': workforce_education,
        'capital_intensity': capital_intensity,
        'tech_readiness': tech_readiness,
        'adoption_propensity': adoption_propensity,
        'true_ai_effect': ai_effect / np.maximum(ai_adoption, 1)  # Individual treatment effect
    })
    
    return df


def main(args):
    print("="*80)
    print("DDPM Causal Inference - AI Adoption Impact Analysis")
    print("="*80)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load or generate data
    if args.data_path:
        print(f"\n1. Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
    else:
        print("\n1. Generating synthetic AI adoption data...")
        df = generate_ai_adoption_data(n_industries=args.n_industries, seed=args.seed)
    
    print(f"   - Industries: {len(df)}")
    print(f"   - AI adoption rate: {df['ai_adoption'].mean():.2%}")
    print(f"   - Mean productivity (adopted): {df[df['ai_adoption']==1]['productivity'].mean():.3f}")
    print(f"   - Mean productivity (not adopted): {df[df['ai_adoption']==0]['productivity'].mean():.3f}")
    print(f"   - Naive difference: {df[df['ai_adoption']==1]['productivity'].mean() - df[df['ai_adoption']==0]['productivity'].mean():.3f}")
    
    # Define confounders
    confounder_cols = ['industry_size', 'rd_intensity', 'workforce_education', 
                       'capital_intensity', 'tech_readiness']
    
    # Check covariate balance
    print("\n2. Checking covariate balance before adjustment...")
    balance = check_balance(df, 'ai_adoption', confounder_cols)
    print(balance[['covariate', 'treated_mean', 'control_mean', 'smd', 'balanced']])
    
    # Plot propensity score overlap
    if 'adoption_propensity' in df.columns:
        print("\n3. Checking propensity score overlap...")
        plot_propensity_overlap(df, 'ai_adoption', 'adoption_propensity')
    
    # Initialize DDPM framework
    print("\n4. Training DDPM model...")
    config = CausalConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        hidden_dim=256,
        n_layers=6
    )
    
    framework = CausalInferenceFramework(config)
    
    # Prepare data
    dataset = framework.prepare_data(
        df,
        treatment_col='ai_adoption',
        outcome_col='productivity',
        confounder_cols=confounder_cols
    )
    
    # Train model
    framework.train(dataset)
    
    # Estimate effects
    print("\n5. Estimating AI adoption effects...")
    ate_ddpm, ate_std_ddpm = framework.estimate_ate(dataset, n_samples=args.n_mc_samples)
    
    print(f"\n   DDPM Results:")
    print(f"   - Estimated ATE: {ate_ddpm:.3f} ± {ate_std_ddpm:.3f}")
    print(f"   - 95% CI: [{ate_ddpm - 1.96*ate_std_ddpm:.3f}, {ate_ddpm + 1.96*ate_std_ddpm:.3f}]")
    
    if 'true_ai_effect' in df.columns:
        true_ate = df[df['ai_adoption']==1]['true_ai_effect'].mean()
        print(f"   - True ATE: {true_ate:.3f}")
        print(f"   - Absolute Error: {abs(ate_ddpm - true_ate):.3f}")
    
    # Heterogeneous effects analysis
    if args.analyze_heterogeneity:
        print("\n6. Analyzing heterogeneous effects...")
        
        # Create covariate grid for CATE estimation
        n_points = 20
        education_range = np.linspace(df['workforce_education'].min(), 
                                    df['workforce_education'].max(), n_points)
        
        # Create synthetic confounders for CATE
        synthetic_confounders = []
        for edu in education_range:
            confounder_vec = torch.tensor([
                df['industry_size'].mean(),
                df['rd_intensity'].mean(),
                edu,
                df['capital_intensity'].mean(),
                df['tech_readiness'].mean()
            ], dtype=torch.float32)
            synthetic_confounders.append(confounder_vec)
        
        synthetic_confounders = torch.stack(synthetic_confounders)
        
        # Estimate CATE
        cate_estimates = framework.estimate_heterogeneous_effects(dataset, synthetic_confounders)
        
        # Visualize heterogeneous effects
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(education_range, cate_estimates, 'b-', linewidth=2)
        plt.xlabel('Workforce Education Level')
        plt.ylabel('Estimated Treatment Effect')
        plt.title('Heterogeneous AI Adoption Effects by Education Level')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/ai_adoption_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        results = {
            'estimated_ate': float(ate_ddpm),
            'ate_std': float(ate_std_ddpm),
            'n_industries': len(df),
            'adoption_rate': float(df['ai_adoption'].mean()),
            'args': vars(args)
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AI adoption effects using DDPM")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file (if None, generates synthetic data)')
    parser.add_argument('--n_industries', type=int, default=500,
                       help='Number of industries (for synthetic data)')
    
    # Model parameters
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                       help='Number of diffusion steps')
    parser.add_argument('--n_mc_samples', type=int, default=100,
                       help='Number of Monte Carlo samples for inference')
    
    # Analysis parameters
    parser.add_argument('--analyze_heterogeneity', action='store_true',
                       help='Analyze heterogeneous treatment effects')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    main(args)