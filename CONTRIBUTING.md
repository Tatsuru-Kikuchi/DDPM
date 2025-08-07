# Contributing to DDPM Causal Inference

We welcome contributions to improve and extend this framework!

## How to Contribute

### 1. Reporting Issues

If you find a bug or have a suggestion:
1. Check existing issues to avoid duplicates
2. Create a new issue with a clear title and description
3. Include minimal reproducible example if reporting a bug

### 2. Proposing Features

1. Open an issue describing the feature
2. Explain the motivation and use case
3. Discuss the design before implementation

### 3. Code Contributions

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/DDPM.git
cd DDPM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

#### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes:
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality

3. Test your changes:
   ```bash
   python -m pytest tests/
   ```

4. Commit with descriptive messages:
   ```bash
   git commit -m "Add: detailed description of changes"
   ```

5. Push and create pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### 4. Code Style Guidelines

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

#### Example:

```python
def estimate_treatment_effect(
    model: CausalDDPM,
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: List[str],
    n_samples: int = 100
) -> Tuple[float, float]:
    """
    Estimate average treatment effect using DDPM.
    
    Args:
        model: Trained DDPM model
        data: Input dataframe
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names
        n_samples: Number of MC samples for inference
    
    Returns:
        Tuple of (ATE estimate, standard error)
    """
    # Implementation here
    pass
```

### 5. Documentation

- Add docstrings to all functions and classes
- Update README.md if adding new features
- Include examples in docstrings
- Update theory.md for methodological contributions

### 6. Testing

All new code should include tests:

```python
# tests/test_causal_ddpm.py
import pytest
from src.causal_ddpm import CausalDDPM

def test_treatment_effect_estimation():
    """Test that ATE estimation works correctly"""
    # Setup
    model = CausalDDPM(...)
    
    # Test
    ate, std = model.estimate_ate(...)
    
    # Assert
    assert isinstance(ate, float)
    assert std > 0
```

## Areas for Contribution

### High Priority

1. **Benchmarking**: Compare against more causal inference methods
2. **Scalability**: Optimize for larger datasets
3. **Visualization**: Better plotting tools for effects
4. **Documentation**: More examples and tutorials

### Research Extensions

1. **Time-varying treatments**: Extend to panel data
2. **Multiple treatments**: Handle multi-valued treatments
3. **Mediation analysis**: Decompose treatment effects
4. **Sensitivity analysis**: Tools for assumption violations

### Engineering

1. **GPU optimization**: Better CUDA utilization
2. **Distributed training**: Multi-GPU support
3. **Model compression**: Faster inference
4. **API improvements**: Cleaner interfaces

## Review Process

1. All PRs require at least one review
2. Tests must pass
3. Documentation must be updated
4. Code should follow style guidelines

## Questions?

Feel free to:
- Open an issue for discussion
- Reach out to maintainers
- Join discussions in existing issues

## License

By contributing, you agree that your contributions will be licensed under MIT License.

## Acknowledgments

Thank you for contributing to advancing causal inference methods!