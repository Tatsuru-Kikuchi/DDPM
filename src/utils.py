"""
Utility functions for data processing and visualization
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def generate_synthetic_data(n_samples: int = 1000, 
                          treatment_effect: float = 3.0,
                          heterogeneous: bool = True,
                          seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data with known causal structure for testing
    
    Args:
        n_samples: Number of samples to generate
        treatment_effect: Base treatment effect
        heterogeneous: Whether to include heterogeneous treatment effects
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(seed)
    
    # Generate confounders
    X1 = np.random.normal(0, 1, n_samples)  # Continuous confounder
    X2 = np.random.normal(0, 1, n_samples)  # Continuous confounder
    X3 = np.random.binomial(1, 0.5, n_samples)  # Binary confounder
    
    # Non-random treatment assignment based on confounders
    propensity = 1 / (1 + np.exp(-0.5 * X1 - 0.3 * X2 - 0.8 * X3))
    T = np.random.binomial(1, propensity)
    
    # Generate outcome with treatment effect
    if heterogeneous:
        # Heterogeneous treatment effect
        tau = treatment_effect * (1 + 0.5 * X1)  # Effect varies with X1
    else:
        # Constant treatment effect
        tau = treatment_effect
    
    # Outcome equation
    Y = 2 * X1 + 1.5 * X2 + X3 + T * tau + np.random.normal(0, 0.5, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'treatment': T,
        'outcome': Y,
        'confounder_1': X1,
        'confounder_2': X2,
        'confounder_3': X3,
        'propensity_score': propensity
    })
    
    # Add true treatment effect for evaluation
    df['true_effect'] = tau if isinstance(tau, (int, float)) else tau
    
    return df


def prepare_economic_data(df: pd.DataFrame,
                         treatment_col: str,
                         outcome_col: str,
                         confounder_cols: List[str],
                         normalize: bool = True) -> Dict[str, torch.Tensor]:
    """
    Prepare economic data for DDPM training
    
    Args:
        df: Input DataFrame
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names
        normalize: Whether to normalize the data
    
    Returns:
        Dictionary of tensors
    """
    # Extract data
    treatment = torch.tensor(df[treatment_col].values, dtype=torch.float32)
    outcome = torch.tensor(df[outcome_col].values, dtype=torch.float32)
    confounders = torch.tensor(df[confounder_cols].values, dtype=torch.float32)
    
    # Normalize if requested
    if normalize:
        outcome_mean = outcome.mean()
        outcome_std = outcome.std()
        outcome = (outcome - outcome_mean) / outcome_std
        
        confounder_mean = confounders.mean(dim=0)
        confounder_std = confounders.std(dim=0)
        confounders = (confounders - confounder_mean) / confounder_std
    else:
        outcome_mean = outcome_std = None
        confounder_mean = confounder_std = None
    
    return {
        'treatment': treatment,
        'outcome': outcome,
        'confounders': confounders,
        'outcome_mean': outcome_mean,
        'outcome_std': outcome_std,
        'confounder_mean': confounder_mean,
        'confounder_std': confounder_std
    }


def visualize_effects(true_effects: np.ndarray,
                      estimated_effects: np.ndarray,
                      title: str = "Treatment Effects Comparison",
                      save_path: Optional[str] = None):
    """
    Visualize true vs estimated treatment effects
    
    Args:
        true_effects: Array of true treatment effects
        estimated_effects: Array of estimated treatment effects
        title: Plot title
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(true_effects, estimated_effects, alpha=0.5)
    axes[0].plot([true_effects.min(), true_effects.max()], 
                 [true_effects.min(), true_effects.max()], 
                 'r--', label='Perfect estimation')
    axes[0].set_xlabel('True Effect')
    axes[0].set_ylabel('Estimated Effect')
    axes[0].set_title('True vs Estimated Effects')
    axes[0].legend()
    
    # Distribution comparison
    axes[1].hist(true_effects, bins=30, alpha=0.5, label='True', density=True)
    axes[1].hist(estimated_effects, bins=30, alpha=0.5, label='Estimated', density=True)
    axes[1].set_xlabel('Effect Size')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Effect Distributions')
    axes[1].legend()
    
    # Error distribution
    errors = estimated_effects - true_effects
    axes[2].hist(errors, bins=30, alpha=0.7, color='green')
    axes[2].axvline(0, color='red', linestyle='--', label='Zero error')
    axes[2].set_xlabel('Estimation Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    
    # Calculate metrics
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)
    
    fig.suptitle(f"{title}\nRMSE: {rmse:.3f}, MAE: {mae:.3f}, Bias: {bias:.3f}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {'rmse': rmse, 'mae': mae, 'bias': bias}


def plot_propensity_overlap(df: pd.DataFrame,
                           treatment_col: str = 'treatment',
                           propensity_col: str = 'propensity_score',
                           save_path: Optional[str] = None):
    """
    Plot propensity score overlap between treatment and control groups
    
    Args:
        df: DataFrame with treatment and propensity scores
        treatment_col: Name of treatment column
        propensity_col: Name of propensity score column
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    treated = df[df[treatment_col] == 1][propensity_col]
    control = df[df[treatment_col] == 0][propensity_col]
    
    ax.hist(control, bins=30, alpha=0.5, label='Control', density=True)
    ax.hist(treated, bins=30, alpha=0.5, label='Treated', density=True)
    
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Overlap')
    ax.legend()
    
    # Add overlap region
    overlap_min = max(control.min(), treated.min())
    overlap_max = min(control.max(), treated.max())
    ax.axvspan(overlap_min, overlap_max, alpha=0.2, color='gray', label='Overlap region')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def check_balance(df: pd.DataFrame,
                  treatment_col: str,
                  covariate_cols: List[str],
                  weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Check covariate balance between treatment and control groups
    
    Args:
        df: Input DataFrame
        treatment_col: Name of treatment column
        covariate_cols: List of covariate column names
        weights: Optional weights for weighted balance
    
    Returns:
        DataFrame with balance statistics
    """
    balance_stats = []
    
    for col in covariate_cols:
        treated = df[df[treatment_col] == 1][col]
        control = df[df[treatment_col] == 0][col]
        
        if weights is not None:
            treated_weights = weights[df[treatment_col] == 1]
            control_weights = weights[df[treatment_col] == 0]
            
            treated_mean = np.average(treated, weights=treated_weights)
            control_mean = np.average(control, weights=control_weights)
            
            treated_std = np.sqrt(np.average((treated - treated_mean)**2, weights=treated_weights))
            control_std = np.sqrt(np.average((control - control_mean)**2, weights=control_weights))
        else:
            treated_mean = treated.mean()
            control_mean = control.mean()
            treated_std = treated.std()
            control_std = control.std()
        
        # Standardized mean difference
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        balance_stats.append({
            'covariate': col,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'smd': smd,
            'balanced': abs(smd) < 0.1  # Common threshold
        })
    
    return pd.DataFrame(balance_stats)


def save_results(results: Dict,
                save_path: str):
    """
    Save experiment results to file
    
    Args:
        results: Dictionary of results
        save_path: Path to save results
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_results[key] = value.cpu().numpy().tolist()
        else:
            serializable_results[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {save_path}")