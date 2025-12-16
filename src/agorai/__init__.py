"""AgorAI: Democratic AI Through Multi-Agent Aggregation.

A Python library for building fair, unbiased AI systems through democratic
multi-agent opinion aggregation.
"""

__version__ = "0.2.0"

# Core aggregation
from agorai.aggregate import aggregate, list_methods

# Synthesis and agent coordination
from agorai.synthesis.core import Agent, Council, synthesize, synthesize_structured

# Validation utilities
from agorai.synthesis.validation import (
    StructuredResponse,
    ValidationResult,
    ResponseValidator,
    format_prompt_with_options
)

# Configuration and logging
from agorai.config import config, Config
from agorai.logging_config import setup_logging, get_logger

# Metrics and monitoring
from agorai.synthesis.metrics import SynthesisMetrics

# Circuit breaker for resilience
from agorai.synthesis.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState
)

# Optional research modules (imported on demand to avoid heavy dependencies)
# Import queue: from agorai import queue
# Import visualization: from agorai import visualization

__all__ = [
    # Aggregation
    "aggregate",
    "list_methods",
    # Synthesis
    "Agent",
    "Council",
    "synthesize",
    "synthesize_structured",
    # Validation
    "StructuredResponse",
    "ValidationResult",
    "ResponseValidator",
    "format_prompt_with_options",
    # Configuration
    "config",
    "Config",
    # Logging
    "setup_logging",
    "get_logger",
    # Metrics
    "SynthesisMetrics",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    # Version
    "__version__"
]

# Convenience: Check if research modules are available
def has_queue():
    """Check if queue module is available."""
    try:
        import agorai.queue
        return True
    except ImportError:
        return False

def has_visualization():
    """Check if visualization module is available."""
    try:
        import agorai.visualization
        return True
    except ImportError:
        return False
