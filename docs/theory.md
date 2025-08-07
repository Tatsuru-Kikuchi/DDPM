# Theoretical Framework

## DDPM for Causal Inference

### Background

Causal inference in economics faces fundamental challenges:

1. **Selection Bias**: Treatment assignment is rarely random
2. **Confounding**: Unobserved variables affect both treatment and outcome
3. **Limited Data**: Economic data is often small and expensive to collect
4. **Heterogeneous Effects**: Treatment effects vary across units

### The DDPM Solution

Denoising Diffusion Probabilistic Models offer a novel approach to these challenges through their unique properties:

#### 1. Forward Diffusion Process

The forward process gradually adds noise to the data:

```
q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

We modify this to incorporate treatment information:

```
q(x_t|x_{t-1}, T, Z) = N(x_t; √(1-β_t)x_{t-1} + λ_t T, β_t I)
```

Where:
- `T` is the treatment indicator
- `Z` are observed confounders
- `λ_t` controls treatment influence at timestep t

#### 2. Reverse Process for Counterfactual Generation

The reverse process learns to denoise while preserving causal structure:

```
p_θ(x_{t-1}|x_t, T, Z) = N(x_{t-1}; μ_θ(x_t, t, T, Z), Σ_θ(x_t, t))
```

This allows generation of counterfactual outcomes by:
1. Starting from noise
2. Conditioning on opposite treatment status
3. Preserving confounder relationships

### Identification Strategy

#### Assumptions

1. **Conditional Independence (CIA)**:
   ```
   Y(1), Y(0) ⊥ T | Z
   ```
   Potential outcomes are independent of treatment given confounders.

2. **Overlap/Common Support**:
   ```
   0 < P(T=1|Z) < 1 for all Z
   ```
   Every unit has non-zero probability of receiving treatment.

3. **Consistency**:
   ```
   Y = T·Y(1) + (1-T)·Y(0)
   ```
   Observed outcome equals the potential outcome for received treatment.

#### How DDPM Satisfies These

1. **CIA**: The model learns conditional distributions p(Y|T,Z) directly
2. **Overlap**: Generation process works across full confounder space
3. **Consistency**: Enforced through training on observed data

### Advantages Over Traditional Methods

| Method | Limitation | DDPM Solution |
|--------|-----------|---------------|
| Propensity Score | Requires correct specification | Learns flexibly from data |
| Regression | Assumes linear relationships | Captures non-linear patterns |
| Matching | Limited to observed similar units | Generates synthetic matches |
| IV | Requires valid instruments | No instrument needed |

### Mathematical Derivation

#### Treatment Effect Estimation

The Average Treatment Effect (ATE) is:

```
τ = E[Y(1) - Y(0)]
```

Using DDPM, we estimate:

```
�^{DDPM} = 1/N Σ_i [Ê[Y|T=1,Z_i] - Ê[Y|T=0,Z_i]]
```

Where Ê[Y|T,Z] is estimated through reverse diffusion:

```
Ê[Y|T,Z] = ∫ y p_θ(y|T,Z) dy ≈ 1/M Σ_m y^(m)
```

With y^(m) sampled from the reverse process.

#### Variance Estimation

We use the sampling variance from multiple diffusion runs:

```
Var(τ^{DDPM}) = 1/N² Σ_i Var[Y^(m)|T=1,Z_i] + Var[Y^(m)|T=0,Z_i]]
```

### Heterogeneous Effects

Conditional Average Treatment Effects (CATE):

```
τ(z) = E[Y(1) - Y(0)|Z=z]
```

DDPM naturally captures heterogeneity through the conditional generation:

```
τ^{DDPM}(z) = Ê[Y|T=1,Z=z] - Ê[Y|T=0,Z=z]
```

## Connection to Economic Theory

### Structural Models

DDPM can be viewed as a flexible structural model where:
- The forward process represents economic shocks
- The reverse process recovers equilibrium outcomes
- Treatment effects emerge from structural differences

### Policy Evaluation

For policy analysis:
1. Train on pre-policy data
2. Generate counterfactual post-policy outcomes
3. Estimate policy effects across the distribution

### Market Equilibrium

The diffusion process can incorporate equilibrium constraints:
- Supply-demand balance
- Budget constraints
- Market clearing conditions

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models
2. Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters
3. Imbens, G. W., & Rubin, D. B. (2015). Causal inference in statistics, social, and biomedical sciences