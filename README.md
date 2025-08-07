# DDPM-Based Causal Inference Framework for Economics

## Overview

This repository implements a novel framework for causal inference using Denoising Diffusion Probabilistic Models (DDPMs). The approach addresses fundamental challenges in economic causal inference, particularly when treatment assignment is non-random (e.g., AI adoption across industries).

### Key Features

- **Non-random treatment handling**: Learns treatment assignment mechanisms implicitly through diffusion
- **Small sample augmentation**: Generates high-quality synthetic counterfactuals
- **Flexible confounding control**: Handles high-dimensional confounders without parametric assumptions
- **Heterogeneous effects**: Automatically discovers treatment effect heterogeneity

## Background

Traditional causal inference methods struggle with:
- Selection bias in observational data
- Limited counterfactual observations
- High-dimensional confounding
- Small sample sizes in economic data

Our DDPM-based approach leverages the generative capabilities of diffusion models to address these challenges. For theoretical background, see:
- [Diffusion models for parameter inference](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.045302)
- [Our previous work on AI adoption](https://arxiv.org/abs/2507.19911)

## Installation

```bash
# Clone the repository
git clone https://github.com/Tatsuru-Kikuchi/DDPM.git
cd DDPM

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.causal_ddpm import CausalInferenceFramework, CausalConfig
import pandas as pd

# Load your data
df = pd.read_csv('data/economic_data.csv')

# Initialize framework
config = CausalConfig(n_epochs=100, batch_size=32)
framework = CausalInferenceFramework(config)

# Prepare data
dataset = framework.prepare_data(
    df, 
    treatment_col='ai_adoption',
    outcome_col='productivity', 
    confounder_cols=['industry_size', 'rd_intensity', 'education']
)

# Train model
framework.train(dataset)

# Estimate causal effects
ate, ate_std = framework.estimate_ate(dataset)
print(f"Average Treatment Effect: {ate:.3f} ± {ate_std:.3f}")
```

## Repository Structure

```
DDPM/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── src/                    # Source code
│   ├── __init__.py
│   ├── causal_ddpm.py      # Main DDPM implementation
│   ├── models.py           # Neural network architectures
│   ├── utils.py            # Utility functions
│   └── evaluation.py       # Evaluation metrics
├── notebooks/              # Jupyter notebooks
│   ├── 01_introduction.ipynb
│   ├── 02_ai_adoption_analysis.ipynb
│   └── 03_benchmarks.ipynb
├── data/                   # Data directory
│   ├── synthetic/          # Synthetic data for testing
│   └── economic/           # Economic datasets
├── experiments/            # Experiment scripts
│   ├── run_synthetic.py
│   └── run_economic.py
├── results/                # Results and figures
└── docs/                   # Documentation
    ├── theory.md          # Theoretical framework
    └── api.md             # API documentation
```

## Methodology

### 1. Modified Diffusion Process

We modify the standard DDPM forward process to incorporate treatment information:

```
x_t = α_t · x_0 + σ_t · (ε + λ·T·η)
```

Where:
- `x_0`: Original economic data
- `T`: Treatment indicator
- `λ`: Treatment effect strength
- `η`: Treatment-specific noise

### 2. Causal Attention Mechanism

Our model includes a custom attention mechanism that respects causal structure:
- Preserves treatment-outcome relationships
- Controls for confounding through learned representations
- Maintains economic equilibrium constraints

### 3. Counterfactual Generation

The reverse diffusion process generates counterfactual outcomes:
- Start from noise conditioned on confounders
- Reverse through learned denoising steps
- Produce synthetic outcomes under alternative treatment

## Applications

### AI Adoption Impact Analysis

Analyze the causal effect of AI adoption on productivity:

```python
python experiments/run_ai_adoption.py --data data/economic/ai_adoption.csv
```

### Policy Evaluation

Evaluate economic policies with synthetic control:

```python
python experiments/run_policy.py --policy minimum_wage --region US
```

## Benchmarks

We compare our method against:
- Propensity Score Matching (PSM)
- Instrumental Variables (IV)
- Double Machine Learning (DML)
- Synthetic Control Methods

See `notebooks/03_benchmarks.ipynb` for detailed comparisons.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{kikuchi2024ddpm,
  title={DDPM-Based Causal Inference for Economics},
  author={Kikuchi, Tatsuru},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

Tatsuru Kikuchi - [GitHub](https://github.com/Tatsuru-Kikuchi)

## Acknowledgments

- Inspired by recent advances in diffusion models and causal inference
- Thanks to the econometrics and machine learning communities