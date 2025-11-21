"""Bias mitigation module.

Provides high-level pipeline for detecting and mitigating AI bias through
culturally-diverse multi-agent synthesis.
"""

from agorai.bias.core import mitigate_bias, BiasConfig
from agorai.bias.contexts import (
    BiasContext,
    HATE_SPEECH_DETECTION,
    CONTENT_MODERATION,
    FAIRNESS_ASSESSMENT,
    CULTURAL_SENSITIVITY,
)

__all__ = [
    "mitigate_bias",
    "BiasConfig",
    "BiasContext",
    "HATE_SPEECH_DETECTION",
    "CONTENT_MODERATION",
    "FAIRNESS_ASSESSMENT",
    "CULTURAL_SENSITIVITY",
]
