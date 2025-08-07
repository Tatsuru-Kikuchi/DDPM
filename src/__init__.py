"""
DDPM-Based Causal Inference Framework
"""

from .causal_ddpm import (
    CausalConfig,
    CausalDDPM,
    CausalInferenceFramework,
    NoiseSchedule
)

from .models import (
    CausalAttention,
    TreatmentEncoder,
    ConfounderEncoder
)

from .utils import (
    prepare_economic_data,
    generate_synthetic_data,
    visualize_effects
)

from .evaluation import (
    evaluate_ate,
    evaluate_cate,
    compare_methods
)

__version__ = "0.1.0"
__author__ = "Tatsuru Kikuchi"

__all__ = [
    "CausalConfig",
    "CausalDDPM",
    "CausalInferenceFramework",
    "NoiseSchedule",
    "CausalAttention",
    "TreatmentEncoder",
    "ConfounderEncoder",
    "prepare_economic_data",
    "generate_synthetic_data",
    "visualize_effects",
    "evaluate_ate",
    "evaluate_cate",
    "compare_methods"
]