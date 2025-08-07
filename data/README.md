# Data Directory

This directory contains datasets for DDPM causal inference experiments.

## Structure

- `synthetic/`: Synthetic datasets with known ground truth
- `economic/`: Real economic datasets (when available)

## Synthetic Data Generation

To generate synthetic data for testing:

```python
from src.utils import generate_synthetic_data

# Generate data with known treatment effect
df = generate_synthetic_data(
    n_samples=1000,
    treatment_effect=3.0,
    heterogeneous=True,
    seed=42
)

df.to_csv('data/synthetic/test_data.csv', index=False)
```

## Economic Data Format

Economic datasets should have the following columns:
- `treatment`: Binary treatment indicator (0 or 1)
- `outcome`: Continuous outcome variable
- Confounder columns: Any relevant confounding variables

## Privacy Note

Please do not commit sensitive or proprietary data to this repository.