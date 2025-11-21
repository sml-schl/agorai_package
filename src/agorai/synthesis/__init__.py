"""LLM-based opinion synthesis module.

Provides multi-provider LLM integration for converting text opinions
into aggregated decisions using the agorai.aggregate module.
"""

from agorai.synthesis.core import synthesize, Agent, Council
from agorai.synthesis.providers import (
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    list_providers,
)

__all__ = [
    "synthesize",
    "Agent",
    "Council",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "list_providers",
]
