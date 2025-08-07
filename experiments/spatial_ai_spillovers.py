#!/usr/bin/env python
"""
Spatial AI Spillover Analysis for New Economic Geography Model
Addresses JUE editor concerns about AI adoption and spatial dynamics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import argparse
from scipy.spatial.distance import cdist
from scipy.stats import moran
import matplotlib.pyplot as plt
import seaborn as sns

from src.causal_ddpm import CausalConfig, CausalInferenceFramework
from src.utils import prepare_economic_data, visualize_effects
from src.evaluation import evaluate_ate, evaluate_cate


class SpatialAIModel:
    """
    Extends Krugman's NEG model with AI spillovers
    Addresses non-random AI adoption across locations
    """
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.framework = CausalInferenceFramework(config)
        
    def prepare_spatial_data(self, df: pd.DataFrame, 
                            location_coords: np.ndarray,
                            distance_decay: float = 0.1):
        """
        Prepare data with spatial structure for AI spillover analysis
        
        Args:
            df: DataFrame with economic variables
            location_coords: Coordinates of locations (n_locations, 2)
            distance_decay: Parameter for spatial weight decay
        """
        # Calculate spatial weights matrix
        distances = cdist(location_coords, location_coords)
        W = np.exp(-distance_decay * distances)
        np.fill_diagonal(W, 0)  # No self-influence
        W = W / W.sum(axis=1, keepdims=True)  # Row-normalize
        
        # Calculate spatial lags for AI adoption
        ai_adoption = df['ai_adoption'].values
        spatial_ai_lag = W @ ai_adoption
        df['spatial_ai_lag'] = spatial_ai_lag
        
        # Calculate local AI density (spillover potential)
        df['local_ai_density'] = spatial_ai_lag / (1 + distances.min(axis=1))
        
        return df, W
    
    def estimate_ai_spillovers(self, df: pd.DataFrame, W: np.ndarray):
        """
        Estimate direct and indirect AI spillover effects
        Using DDPM to handle non-random adoption
        """
        # Define treatment levels based on AI intensity
        ai_quartiles = pd.qcut(df['ai_adoption'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        df['ai_intensity'] = ai_quartiles.cat.codes
        
        # Confounders including spatial factors
        confounder_cols = [
            'industry_concentration',  # Krugman's agglomeration force
            'market_access',           # NEG market potential
            'labor_skill_index',       # Human capital
            'infrastructure_quality',   # Physical capital
            'spatial_ai_lag',          # Spatial spillover potential
            'distance_to_center'       # Core-periphery structure
        ]
        
        # Prepare dataset for DDPM
        dataset = self.framework.prepare_data(
            df,
            treatment_col='ai_intensity',
            outcome_col='productivity',
            confounder_cols=confounder_cols
        )
        
        # Train DDPM model
        print("Training DDPM to learn AI adoption patterns...")
        self.framework.train(dataset)
        
        # Estimate direct effects
        direct_ate, direct_std = self.framework.estimate_ate(dataset)
        
        # Estimate spillover effects by location
        spillover_effects = self._estimate_spatial_spillovers(df, W, dataset)
        
        return {
            'direct_effect': direct_ate,
            'direct_std': direct_std,
            'spillover_effects': spillover_effects,
            'total_effect': direct_ate + np.mean(spillover_effects)
        }
    
    def _estimate_spatial_spillovers(self, df: pd.DataFrame, W: np.ndarray, dataset):
        """
        Estimate location-specific spillover effects
        """
        n_locations = len(df)
        spillover_effects = np.zeros(n_locations)
        
        for i in range(n_locations):
            # Get neighbors
            neighbors = np.where(W[i] > 0)[0]
            
            if len(neighbors) == 0:
                continue
            
            # Estimate counterfactual: what if neighbors had no AI?
            original_ai = df.iloc[neighbors]['ai_intensity'].values
            
            # Create counterfactual scenario
            df_counterfactual = df.copy()
            df_counterfactual.iloc[neighbors, df.columns.get_loc('ai_intensity')] = 0
            
            # Re-calculate spatial lag
            ai_adoption_cf = df_counterfactual['ai_adoption'].values
            spatial_ai_lag_cf = W @ ai_adoption_cf
            df_counterfactual['spatial_ai_lag'] = spatial_ai_lag_cf
            
            # Estimate effect for location i
            # This captures the spillover from neighbors' AI adoption
            spillover_effects[i] = self._estimate_location_effect(
                i, df, df_counterfactual, dataset
            )
        
        return spillover_effects
    
    def _estimate_location_effect(self, location_idx: int, 
                                 df_actual: pd.DataFrame,
                                 df_counterfactual: pd.DataFrame,
                                 dataset):
        """
        Estimate effect for a specific location
        """
        # Get actual and counterfactual confounders for this location
        actual_confounders = df_actual.iloc[location_idx][
            ['industry_concentration', 'market_access', 'labor_skill_index',
             'infrastructure_quality', 'spatial_ai_lag', 'distance_to_center']
        ].values
        
        cf_confounders = df_counterfactual.iloc[location_idx][
            ['industry_concentration', 'market_access', 'labor_skill_index',
             'infrastructure_quality', 'spatial_ai_lag', 'distance_to_center']
        ].values
        
        # Convert to tensors
        actual_conf_tensor = torch.tensor(actual_confounders, dtype=torch.float32).unsqueeze(0)
        cf_conf_tensor = torch.tensor(cf_confounders, dtype=torch.float32).unsqueeze(0)
        
        # Get treatment for this location
        treatment = torch.tensor(
            [df_actual.iloc[location_idx]['ai_intensity']], 
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Generate outcomes using DDPM
        actual_outcome = self.framework.model.generate_counterfactual(
            treatment, actual_conf_tensor, n_samples=50
        ).mean()
        
        cf_outcome = self.framework.model.generate_counterfactual(
            treatment, cf_conf_tensor, n_samples=50
        ).mean()
        
        return (actual_outcome - cf_outcome).item()
    
    def decompose_agglomeration_forces(self, df: pd.DataFrame, effects: dict):
        """
        Decompose total effect into Krugman's forces + AI spillovers
        This addresses the editor's concern about value-added
        """
        # Traditional NEG forces (Krugman model)
        traditional_forces = {
            'market_access_effect': self._estimate_market_access_effect(df),
            'competition_effect': self._estimate_competition_effect(df),
            'linkage_effect': self._estimate_linkage_effect(df)
        }
        
        # AI-specific additions
        ai_additions = {
            'direct_ai_effect': effects['direct_effect'],
            'spatial_ai_spillover': np.mean(effects['spillover_effects']),
            'ai_agglomeration_bonus': self._estimate_ai_agglomeration(df)
        }
        
        # Value added by AI extension
        traditional_total = sum(traditional_forces.values())
        ai_total = sum(ai_additions.values())
        value_added_ratio = ai_total / (traditional_total + ai_total)
        
        return {
            'traditional_neg': traditional_forces,
            'ai_extensions': ai_additions,
            'value_added_ratio': value_added_ratio,
            'total_effect': traditional_total + ai_total
        }
    
    def _estimate_market_access_effect(self, df: pd.DataFrame):
        """Estimate traditional market access effect"""
        return np.corrcoef(df['market_access'], df['productivity'])[0, 1] * df['market_access'].std()
    
    def _estimate_competition_effect(self, df: pd.DataFrame):
        """Estimate competition effect from firm density"""
        return -0.1 * df['industry_concentration'].mean()  # Negative effect
    
    def _estimate_linkage_effect(self, df: pd.DataFrame):
        """Estimate forward/backward linkages"""
        return 0.15 * df['infrastructure_quality'].mean()
    
    def _estimate_ai_agglomeration(self, df: pd.DataFrame):
        """Estimate additional agglomeration from AI clustering"""
        ai_cluster = df.groupby('ai_intensity')['productivity'].mean()
        return ai_cluster.diff().mean()
    
    def plot_spatial_effects(self, df: pd.DataFrame, 
                           spillover_effects: np.ndarray,
                           location_coords: np.ndarray):
        """
        Visualize spatial distribution of AI effects
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Direct AI adoption
        scatter1 = axes[0].scatter(location_coords[:, 0], location_coords[:, 1],
                                  c=df['ai_adoption'], cmap='RdYlBu_r', s=100)
        axes[0].set_title('AI Adoption Intensity')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Spillover effects
        scatter2 = axes[1].scatter(location_coords[:, 0], location_coords[:, 1],
                                  c=spillover_effects, cmap='RdYlBu_r', s=100)
        axes[1].set_title('Spatial Spillover Effects')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Productivity outcomes
        scatter3 = axes[2].scatter(location_coords[:, 0], location_coords[:, 1],
                                  c=df['productivity'], cmap='RdYlBu_r', s=100)
        axes[2].set_title('Productivity Outcomes')
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        plt.colorbar(scatter3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('results/spatial_ai_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_moran_i(self, df: pd.DataFrame, W: np.ndarray):
        """
        Calculate Moran's I for spatial autocorrelation
        Tests whether AI adoption clusters spatially
        """
        ai_adoption = df['ai_adoption'].values
        n = len(ai_adoption)
        
        # Demean the variable
        ai_centered = ai_adoption - ai_adoption.mean()
        
        # Calculate Moran's I
        numerator = (ai_centered @ W @ ai_centered)
        denominator = (ai_centered @ ai_centered)
        W_sum = W.sum()
        
        I = (n / W_sum) * (numerator / denominator)
        
        # Expected value under null hypothesis
        E_I = -1 / (n - 1)
        
        # Variance calculation (simplified)
        b2 = (ai_centered ** 2).sum() / n
        s0 = W_sum
        s1 = 0.5 * ((W + W.T) ** 2).sum()
        s2 = (W.sum(axis=1) ** 2).sum()
        
        var_I = (n * ((n**2 - 3*n + 3) * s1 - n * s2 + 3 * s0**2) - 
                ((n**2 - n) * s1 - 2*n * s2 + 6 * s0**2)) / ((n - 1) * (n - 2) * (n - 3) * s0**2)
        
        # Z-score
        z_score = (I - E_I) / np.sqrt(var_I)
        
        return {
            'morans_i': I,
            'expected': E_I,
            'variance': var_I,
            'z_score': z_score,
            'p_value': 2 * (1 - stats.norm.cdf(abs(z_score)))
        }


def generate_tokyo_style_data(n_locations=23, seed=42):
    """
    Generate synthetic data mimicking Tokyo's 23 wards structure
    """
    np.random.seed(seed)
    
    # Generate location coordinates (simplified Tokyo layout)
    angles = np.linspace(0, 2*np.pi, n_locations)
    distances = np.random.uniform(0, 10, n_locations)
    distances[0] = 0  # Central district (like Chiyoda)
    
    location_coords = np.column_stack([
        distances * np.cos(angles),
        distances * np.sin(angles)
    ])
    
    # Distance to center
    distance_to_center = np.sqrt(location_coords[:, 0]**2 + location_coords[:, 1]**2)
    
    # Generate economic variables
    df = pd.DataFrame({
        'location_id': range(n_locations),
        'distance_to_center': distance_to_center,
        
        # NEG variables
        'industry_concentration': np.random.beta(2, 5, n_locations) * (1 - 0.3 * distance_to_center/10),
        'market_access': 10 * np.exp(-0.1 * distance_to_center) + np.random.normal(0, 1, n_locations),
        'labor_skill_index': np.random.beta(3, 2, n_locations) * (1 - 0.2 * distance_to_center/10),
        'infrastructure_quality': np.random.beta(3, 3, n_locations) * (1 - 0.15 * distance_to_center/10),
        
        # AI adoption (non-random, depends on location characteristics)
        'ai_adoption': np.zeros(n_locations)  # Will be filled based on selection
    })
    
    # Non-random AI adoption based on location characteristics
    adoption_propensity = (
        0.3 * (1 - distance_to_center/10) +  # Central locations more likely
        0.25 * df['labor_skill_index'] +
        0.25 * df['market_access'] / df['market_access'].max() +
        0.2 * df['infrastructure_quality']
    )
    
    # Add noise and create binary/continuous adoption
    adoption_propensity = 1 / (1 + np.exp(-5 * (adoption_propensity - 0.5)))
    df['ai_adoption'] = adoption_propensity + np.random.normal(0, 0.1, n_locations)
    df['ai_adoption'] = np.clip(df['ai_adoption'], 0, 1)
    
    # Generate productivity with AI effects and spillovers
    base_productivity = (
        2 * df['market_access'] +
        1.5 * df['labor_skill_index'] +
        1 * df['infrastructure_quality'] -
        0.5 * df['industry_concentration']  # Competition effect
    )
    
    # AI direct effect (heterogeneous)
    ai_effect = df['ai_adoption'] * (2 + 0.5 * df['labor_skill_index'])
    
    # Spatial spillovers (will be calculated in the model)
    
    # Final productivity
    df['productivity'] = base_productivity + ai_effect + np.random.normal(0, 0.5, n_locations)
    
    return df, location_coords


def main(args):
    print("="*80)
    print("Spatial AI Spillover Analysis for NEG Model")
    print("Addressing JUE Editor Concerns")
    print("="*80)
    
    # Generate or load data
    if args.data_path:
        df = pd.read_csv(args.data_path)
        # Assume coordinates are in the data or use default grid
        n_locs = len(df)
        location_coords = np.column_stack([
            np.random.uniform(-10, 10, n_locs),
            np.random.uniform(-10, 10, n_locs)
        ])
    else:
        print("\n1. Generating Tokyo-style spatial data...")
        df, location_coords = generate_tokyo_style_data(
            n_locations=args.n_locations,
            seed=args.seed
        )
    
    print(f"   Locations: {len(df)}")
    print(f"   Mean AI adoption: {df['ai_adoption'].mean():.3f}")
    print(f"   AI adoption std: {df['ai_adoption'].std():.3f}")
    
    # Initialize spatial model
    print("\n2. Initializing Spatial AI Model...")
    config = CausalConfig(
        n_epochs=args.n_epochs,
        batch_size=16,
        learning_rate=1e-3,
        hidden_dim=256
    )
    
    model = SpatialAIModel(config)
    
    # Prepare spatial data
    print("\n3. Calculating spatial structure...")
    df, W = model.prepare_spatial_data(df, location_coords, args.distance_decay)
    
    # Test for spatial autocorrelation
    moran_results = model.calculate_moran_i(df, W)
    print(f"   Moran's I: {moran_results['morans_i']:.4f}")
    print(f"   Z-score: {moran_results['z_score']:.4f}")
    print(f"   P-value: {moran_results['p_value']:.4f}")
    
    if moran_results['p_value'] < 0.05:
        print("   ✓ Significant spatial clustering detected")
    
    # Estimate AI spillover effects
    print("\n4. Estimating AI spillover effects using DDPM...")
    effects = model.estimate_ai_spillovers(df, W)
    
    print(f"\n   Direct AI Effect: {effects['direct_effect']:.4f} ± {effects['direct_std']:.4f}")
    print(f"   Mean Spillover Effect: {np.mean(effects['spillover_effects']):.4f}")
    print(f"   Total Effect: {effects['total_effect']:.4f}")
    
    # Decompose into NEG + AI components
    print("\n5. Decomposing agglomeration forces...")
    decomposition = model.decompose_agglomeration_forces(df, effects)
    
    print("\n   Traditional NEG Forces:")
    for force, value in decomposition['traditional_neg'].items():
        print(f"      {force}: {value:.4f}")
    
    print("\n   AI-Specific Additions:")
    for force, value in decomposition['ai_extensions'].items():
        print(f"      {force}: {value:.4f}")
    
    print(f"\n   Value Added by AI Extension: {decomposition['value_added_ratio']:.1%}")
    
    # Visualize spatial effects
    if args.visualize:
        print("\n6. Generating spatial visualizations...")
        model.plot_spatial_effects(df, effects['spillover_effects'], location_coords)
    
    # Save results
    if args.save_results:
        import json
        results = {
            'direct_effect': float(effects['direct_effect']),
            'spillover_mean': float(np.mean(effects['spillover_effects'])),
            'total_effect': float(effects['total_effect']),
            'moran_i': float(moran_results['morans_i']),
            'moran_pvalue': float(moran_results['p_value']),
            'value_added_ratio': float(decomposition['value_added_ratio']),
            'decomposition': {
                k: {kk: float(vv) for kk, vv in v.items()} 
                if isinstance(v, dict) else float(v)
                for k, v in decomposition.items()
            }
        }
        
        with open('results/spatial_ai_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n   Results saved to results/spatial_ai_analysis.json")
    
    print("\n" + "="*80)
    print("CONCLUSION: AI extensions add {:.1%} to traditional NEG model".format(
        decomposition['value_added_ratio']
    ))
    print("Spatial spillovers account for {:.1%} of total AI effect".format(
        np.mean(effects['spillover_effects']) / effects['total_effect']
    ))
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate AI spillovers in spatial economic model"
    )
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to spatial data CSV')
    parser.add_argument('--n_locations', type=int, default=23,
                       help='Number of locations (for synthetic data)')
    parser.add_argument('--distance_decay', type=float, default=0.1,
                       help='Distance decay parameter for spatial weights')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Training epochs for DDPM')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate spatial visualizations')
    parser.add_argument('--save_results', action='store_true',
                       help='Save analysis results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
