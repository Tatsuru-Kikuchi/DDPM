"""
Evaluation metrics and comparison methods for causal inference
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings


def evaluate_ate(true_ate: float,
                estimated_ate: float,
                estimated_std: Optional[float] = None) -> Dict[str, float]:
    """
    Evaluate Average Treatment Effect estimation
    
    Args:
        true_ate: True average treatment effect
        estimated_ate: Estimated average treatment effect
        estimated_std: Standard deviation of estimate
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'true_ate': true_ate,
        'estimated_ate': estimated_ate,
        'absolute_error': abs(estimated_ate - true_ate),
        'relative_error': abs(estimated_ate - true_ate) / abs(true_ate) if true_ate != 0 else np.inf,
        'bias': estimated_ate - true_ate
    }
    
    if estimated_std is not None:
        metrics['estimated_std'] = estimated_std
        # Check if true value is within confidence interval
        ci_lower = estimated_ate - 1.96 * estimated_std
        ci_upper = estimated_ate + 1.96 * estimated_std
        metrics['ci_lower'] = ci_lower
        metrics['ci_upper'] = ci_upper
        metrics['ci_coverage'] = ci_lower <= true_ate <= ci_upper
    
    return metrics


def evaluate_cate(true_cate: np.ndarray,
                 estimated_cate: np.ndarray,
                 X: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate Conditional Average Treatment Effect estimation
    
    Args:
        true_cate: True conditional treatment effects
        estimated_cate: Estimated conditional treatment effects
        X: Covariates (optional, for weighted metrics)
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Basic metrics
    mse = mean_squared_error(true_cate, estimated_cate)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_cate, estimated_cate)
    
    # Correlation
    correlation, p_value = stats.pearsonr(true_cate, estimated_cate)
    
    # R-squared
    ss_res = np.sum((true_cate - estimated_cate) ** 2)
    ss_tot = np.sum((true_cate - np.mean(true_cate)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'correlation_pvalue': p_value,
        'r2': r2,
        'mean_bias': np.mean(estimated_cate - true_cate),
        'std_bias': np.std(estimated_cate - true_cate)
    }
    
    # Quantile-based metrics
    for q in [0.25, 0.5, 0.75]:
        q_true = np.quantile(true_cate, q)
        q_est = np.quantile(estimated_cate, q)
        metrics[f'quantile_{int(q*100)}_error'] = q_est - q_true
    
    return metrics


def compare_methods(results_dict: Dict[str, Dict],
                   metric_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare results from different causal inference methods
    
    Args:
        results_dict: Dictionary with method names as keys and result dicts as values
        metric_names: List of metrics to compare (None for all)
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for method_name, results in results_dict.items():
        row = {'method': method_name}
        
        if metric_names is None:
            # Use all available metrics
            row.update(results)
        else:
            # Use only specified metrics
            for metric in metric_names:
                if metric in results:
                    row[metric] = results[metric]
                else:
                    row[metric] = np.nan
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Add ranking for each metric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'error' in col.lower() or 'mse' in col.lower() or 'mae' in col.lower():
            # Lower is better
            df[f'{col}_rank'] = df[col].rank()
        elif 'correlation' in col.lower() or 'r2' in col.lower():
            # Higher is better
            df[f'{col}_rank'] = df[col].rank(ascending=False)
    
    return df


def bootstrap_confidence_interval(estimator_func,
                                 data: pd.DataFrame,
                                 n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95,
                                 **kwargs) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for an estimator
    
    Args:
        estimator_func: Function that estimates the causal effect
        data: Input data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for interval
        **kwargs: Additional arguments for estimator_func
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = data.sample(n=len(data), replace=True)
        
        # Calculate estimate on bootstrap sample
        try:
            estimate = estimator_func(bootstrap_sample, **kwargs)
            bootstrap_estimates.append(estimate)
        except Exception as e:
            warnings.warn(f"Bootstrap iteration failed: {e}")
            continue
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.quantile(bootstrap_estimates, alpha / 2)
    ci_upper = np.quantile(bootstrap_estimates, 1 - alpha / 2)
    point_estimate = np.mean(bootstrap_estimates)
    
    return point_estimate, ci_lower, ci_upper


def permutation_test(treatment_effects_1: np.ndarray,
                    treatment_effects_2: np.ndarray,
                    n_permutations: int = 1000) -> float:
    """
    Permutation test for comparing two sets of treatment effects
    
    Args:
        treatment_effects_1: First set of treatment effects
        treatment_effects_2: Second set of treatment effects
        n_permutations: Number of permutations
    
    Returns:
        p-value from permutation test
    """
    # Observed difference
    observed_diff = np.mean(treatment_effects_1) - np.mean(treatment_effects_2)
    
    # Combine data
    combined = np.concatenate([treatment_effects_1, treatment_effects_2])
    n1 = len(treatment_effects_1)
    
    # Permutation test
    permuted_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_1 = combined[:n1]
        perm_2 = combined[n1:]
        permuted_diff = np.mean(perm_1) - np.mean(perm_2)
        permuted_diffs.append(permuted_diff)
    
    permuted_diffs = np.array(permuted_diffs)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return p_value


def sensitivity_analysis(model,
                       data: pd.DataFrame,
                       confounder_cols: List[str],
                       sensitivity_params: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Perform sensitivity analysis for unmeasured confounding
    
    Args:
        model: Causal inference model
        data: Input data
        confounder_cols: List of observed confounder columns
        sensitivity_params: Array of sensitivity parameters to test
    
    Returns:
        Dictionary with sensitivity analysis results
    """
    results = {
        'sensitivity_params': sensitivity_params,
        'ate_estimates': [],
        'ate_lower': [],
        'ate_upper': []
    }
    
    for param in sensitivity_params:
        # Simulate unmeasured confounding
        # This is a simplified example - actual implementation would depend on the specific sensitivity analysis method
        modified_data = data.copy()
        
        # Add simulated unmeasured confounder effect
        u = np.random.normal(0, param, len(data))
        modified_data['outcome'] += u * modified_data['treatment']
        
        # Re-estimate effect
        ate_estimate = model.estimate_ate(modified_data)
        
        results['ate_estimates'].append(ate_estimate)
    
    results['ate_estimates'] = np.array(results['ate_estimates'])
    
    return results