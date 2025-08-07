"""
Neural network components for causal DDPM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalAttention(nn.Module):
    """Attention mechanism that respects causal structure"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Causal mask parameters
        self.treatment_weight = nn.Parameter(torch.ones(1))
        self.confounder_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, treatment: torch.Tensor, confounders: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores with causal adjustment
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply causal weighting
        treatment_mask = treatment.unsqueeze(1).unsqueeze(2)
        confounder_mask = confounders.mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(2)
        
        scores = scores + self.treatment_weight * treatment_mask + self.confounder_weight * confounder_mask
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output


class TreatmentEncoder(nn.Module):
    """Encodes treatment information"""
    
    def __init__(self, treatment_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, treatment: torch.Tensor):
        return self.encoder(treatment)


class ConfounderEncoder(nn.Module):
    """Encodes confounder information"""
    
    def __init__(self, confounder_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(confounder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, confounders: torch.Tensor):
        return self.encoder(confounders)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor):
        return x + self.net(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 2
        embeddings = torch.exp(
            -torch.arange(half_dim, device=t.device) * 
            (np.log(self.max_period) / half_dim)
        )
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1), mode='constant')
            
        return embeddings