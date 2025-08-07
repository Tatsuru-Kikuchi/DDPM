"""
Main DDPM implementation for causal inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

@dataclass
class CausalConfig:
    """Configuration for causal DDPM model"""
    dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    diffusion_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NoiseSchedule:
    """Manages the noise schedule for diffusion process"""
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.betas = torch.linspace(
            config.beta_start, 
            config.beta_end, 
            config.diffusion_steps
        )
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-calculate useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def get_index_from_list(self, vals, t, x_shape):
        """Extract values from a list according to indices t"""
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class CausalDDPM(nn.Module):
    """Main DDPM model with causal structure"""
    
    def __init__(self, config: CausalConfig, data_dim: int, treatment_dim: int, confounder_dim: int):
        super().__init__()
        self.config = config
        self.data_dim = data_dim
        
        # Import models from models.py
        from .models import TreatmentEncoder, ConfounderEncoder, CausalAttention
        
        # Encoders
        self.treatment_encoder = TreatmentEncoder(treatment_dim, config.hidden_dim)
        self.confounder_encoder = ConfounderEncoder(confounder_dim, config.hidden_dim)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Main network
        self.input_proj = nn.Linear(data_dim, config.dim)
        
        # Transformer blocks with causal attention
        self.blocks = nn.ModuleList([
            CausalAttention(config.dim, config.n_heads)
            for _ in range(config.n_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(config.dim)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.dim + 2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, data_dim)
        )
        
        # Noise schedule
        self.noise_schedule = NoiseSchedule(config)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                treatment: torch.Tensor, confounders: torch.Tensor):
        """Forward pass of the model"""
        
        # Encode inputs
        t_emb = self.time_embedding(t.unsqueeze(-1).float() / self.config.diffusion_steps)
        treatment_emb = self.treatment_encoder(treatment)
        confounder_emb = self.confounder_encoder(confounders)
        
        # Project input
        h = self.input_proj(x)
        h = h.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer blocks
        for block, norm in zip(self.blocks, self.norm_layers):
            h = h + block(norm(h), treatment, confounders)
        
        # Combine embeddings
        h = h.squeeze(1)  # Remove sequence dimension
        combined = torch.cat([h, treatment_emb, confounder_emb], dim=-1)
        
        # Output noise prediction
        noise_pred = self.output_proj(combined)
        
        return noise_pred
    
    def compute_loss(self, x0: torch.Tensor, treatment: torch.Tensor, confounders: torch.Tensor):
        """Compute training loss"""
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=x0.device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Add noise to data
        sqrt_alphas_cumprod_t = self.noise_schedule.get_index_from_list(
            self.noise_schedule.sqrt_alphas_cumprod, t, x0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule.get_index_from_list(
            self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        
        x_noisy = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Predict noise
        noise_pred = self.forward(x_noisy, t, treatment, confounders)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def generate_counterfactual(self, treatment: torch.Tensor, confounders: torch.Tensor, 
                                n_samples: int = 1):
        """Generate counterfactual outcomes"""
        device = next(self.parameters()).device
        
        # Start from noise
        x = torch.randn(n_samples, self.data_dim, device=device)
        
        # Expand treatment and confounders if needed
        if treatment.shape[0] == 1:
            treatment = treatment.repeat(n_samples, 1)
        if confounders.shape[0] == 1:
            confounders = confounders.repeat(n_samples, 1)
        
        # Reverse diffusion process
        for i in reversed(range(self.config.diffusion_steps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x, t, treatment, confounders)
            
            # Denoise step
            alpha_t = self.noise_schedule.alphas[i]
            alpha_t_prev = self.noise_schedule.alphas_cumprod_prev[i]
            beta_t = self.noise_schedule.betas[i]
            
            # Compute x_{t-1}
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t_prev)) * noise_pred)
            
            # Add noise (except for last step)
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = x + sigma_t * noise
        
        return x


class CausalInferenceFramework:
    """Main framework for causal inference using DDPM"""
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        
    def prepare_data(self, df: pd.DataFrame, treatment_col: str, 
                     outcome_col: str, confounder_cols: List[str]):
        """Prepare data for causal inference"""
        
        # Extract components
        treatment = torch.tensor(df[treatment_col].values, dtype=torch.float32)
        outcome = torch.tensor(df[outcome_col].values, dtype=torch.float32)
        confounders = torch.tensor(df[confounder_cols].values, dtype=torch.float32)
        
        # Normalize
        outcome = (outcome - outcome.mean()) / outcome.std()
        confounders = (confounders - confounders.mean(dim=0)) / confounders.std(dim=0)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(outcome.unsqueeze(-1), 
                                                 treatment.unsqueeze(-1), 
                                                 confounders)
        
        return dataset
    
    def train(self, dataset: torch.utils.data.TensorDataset):
        """Train the causal DDPM model"""
        
        # Initialize model
        sample_outcome, sample_treatment, sample_confounders = dataset[0]
        data_dim = sample_outcome.shape[0]
        treatment_dim = sample_treatment.shape[0]
        confounder_dim = sample_confounders.shape[0]
        
        self.model = CausalDDPM(self.config, data_dim, treatment_dim, confounder_dim)
        self.model.to(self.config.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.n_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.n_epochs}"):
                outcomes, treatments, confounders = [b.to(self.config.device) for b in batch]
                
                # Compute loss
                loss = self.model.compute_loss(outcomes, treatments, confounders)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    def estimate_ate(self, dataset: torch.utils.data.TensorDataset, n_samples: int = 100):
        """Estimate Average Treatment Effect"""
        
        self.model.eval()
        treatment_effects = []
        
        with torch.no_grad():
            for outcome, treatment, confounders in dataset:
                outcome = outcome.to(self.config.device).unsqueeze(0)
                treatment = treatment.to(self.config.device).unsqueeze(0)
                confounders = confounders.to(self.config.device).unsqueeze(0)
                
                # Generate counterfactual with opposite treatment
                cf_treatment = 1 - treatment
                cf_outcome = self.model.generate_counterfactual(cf_treatment, confounders, n_samples)
                
                # Calculate individual treatment effect
                ite = cf_outcome.mean(dim=0) - outcome
                treatment_effects.append(ite.cpu())
        
        # Calculate ATE
        ate = torch.cat(treatment_effects).mean()
        ate_std = torch.cat(treatment_effects).std()
        
        return ate.item(), ate_std.item()
    
    def estimate_heterogeneous_effects(self, dataset: torch.utils.data.TensorDataset, 
                                      covariate_values: torch.Tensor):
        """Estimate Conditional Average Treatment Effects (CATE)"""
        
        self.model.eval()
        cate_estimates = []
        
        with torch.no_grad():
            for cov_val in covariate_values:
                # Create synthetic confounders
                synthetic_confounders = cov_val.unsqueeze(0).to(self.config.device)
                
                # Generate outcomes for both treatment states
                treatment_0 = torch.zeros(1, 1, device=self.config.device)
                treatment_1 = torch.ones(1, 1, device=self.config.device)
                
                outcome_0 = self.model.generate_counterfactual(treatment_0, synthetic_confounders, 50)
                outcome_1 = self.model.generate_counterfactual(treatment_1, synthetic_confounders, 50)
                
                # Calculate CATE
                cate = (outcome_1 - outcome_0).mean()
                cate_estimates.append(cate.cpu().item())
        
        return np.array(cate_estimates)