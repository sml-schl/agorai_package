# Configuration Guide

Complete guide to configuring AgorAI for your environment and use case.

## Table of Contents

1. [Installation Options](#installation-options)
2. [Environment Variables](#environment-variables)
3. [LLM Provider Configuration](#llm-provider-configuration)
4. [Default Settings](#default-settings)
5. [Production Configuration](#production-configuration)
6. [Development Setup](#development-setup)

---

## Installation Options

### Minimal Installation

For pure mathematical aggregation only:

```bash
pip install agorai
```

**Includes:**
- Core aggregation methods (14+)
- No external dependencies beyond NumPy

### Research Installation

For queue processing and visualization:

```bash
pip install agorai[research]
```

**Adds:**
- Queue processing module
- Visualization capabilities (matplotlib)
- Metrics calculation

### Synthesis Installation

For LLM-based opinion synthesis:

```bash
pip install agorai[synthesis]
```

**Adds:**
- Multi-provider LLM support
- OpenAI, Anthropic, Ollama, Google integrations

### Bias Mitigation Installation

For cultural perspective diversity:

```bash
pip install agorai[bias]
```

**Adds:**
- Bias detection and mitigation
- Cultural perspective management

### Complete Installation

Everything included:

```bash
pip install agorai[all]
```

**Includes:** All modules and dependencies

### Development Installation

For contributing to AgorAI:

```bash
git clone https://github.com/yourusername/agorai-package.git
cd agorai-package
pip install -e ".[dev]"
```

**Adds:**
- Testing tools (pytest)
- Linting (flake8, black)
- Type checking (mypy)
- Documentation tools

---

## Environment Variables

### LLM API Keys

Set API keys for different providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google (Gemini)
export GOOGLE_API_KEY="..."

# Optional: Ollama (if not using default localhost)
export OLLAMA_HOST="http://localhost:11434"
```

### Using .env Files

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OLLAMA_HOST=http://localhost:11434
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()

# Now environment variables are available
from agorai.synthesis import Agent

agent = Agent(provider="openai", model="gpt-4")  # Uses OPENAI_API_KEY
```

### Package Configuration

Optional environment variables for AgorAI:

```bash
# Enable debug logging
export AGORAI_DEBUG=1

# Set default aggregation method
export AGORAI_DEFAULT_METHOD=atkinson

# Set default cache directory
export AGORAI_CACHE_DIR=/path/to/cache

# Disable warnings
export AGORAI_NO_WARNINGS=1
```

---

## LLM Provider Configuration

### OpenAI Configuration

#### Basic Setup

```python
from agorai.synthesis import Agent

agent = Agent(
    provider="openai",
    model="gpt-4",
    api_key="sk-..."  # or use environment variable
)
```

#### Available Models

```python
# GPT-4 (recommended)
Agent(provider="openai", model="gpt-4")
Agent(provider="openai", model="gpt-4-turbo-preview")
Agent(provider="openai", model="gpt-4-turbo")

# GPT-4o (multimodal)
Agent(provider="openai", model="gpt-4o")
Agent(provider="openai", model="gpt-4o-mini")

# GPT-3.5 (faster, cheaper)
Agent(provider="openai", model="gpt-3.5-turbo")
```

#### Advanced Parameters

```python
agent = Agent(
    provider="openai",
    model="gpt-4",

    # Generation parameters
    temperature=0.7,        # 0.0-2.0 (default: 1.0)
    max_tokens=1000,        # Max response length
    top_p=1.0,              # Nucleus sampling
    frequency_penalty=0.0,  # -2.0 to 2.0
    presence_penalty=0.0,   # -2.0 to 2.0

    # System configuration
    system_prompt="You are a helpful assistant.",

    # Timeout and retries
    timeout=30,             # seconds
    max_retries=3,
)
```

#### Custom Base URL (for proxies)

```python
agent = Agent(
    provider="openai",
    model="gpt-4",
    base_url="https://your-proxy.com/v1",  # Custom endpoint
    api_key="sk-..."
)
```

---

### Anthropic Configuration

#### Basic Setup

```python
from agorai.synthesis import Agent

agent = Agent(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-..."  # or use environment variable
)
```

#### Available Models

```python
# Claude 3.5 Sonnet (recommended)
Agent(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Claude 3 Opus (most capable)
Agent(provider="anthropic", model="claude-3-opus-20240229")

# Claude 3 Sonnet
Agent(provider="anthropic", model="claude-3-sonnet-20240229")

# Claude 3 Haiku (fastest)
Agent(provider="anthropic", model="claude-3-haiku-20240307")
```

#### Advanced Parameters

```python
agent = Agent(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",

    # Generation parameters
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,
    top_k=250,

    # System configuration
    system_prompt="You are a thoughtful analyst.",

    # Timeout
    timeout=30,
)
```

---

### Ollama Configuration

#### Basic Setup

```python
from agorai.synthesis import Agent

# Default (localhost:11434)
agent = Agent(
    provider="ollama",
    model="llama3.2"
)
```

#### Available Models

Install models first:

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull mixtral
ollama pull gemma2
ollama pull qwen2.5
```

Then use in Python:

```python
# Llama 3.2 (Meta)
Agent(provider="ollama", model="llama3.2")
Agent(provider="ollama", model="llama3.2:70b")  # Larger version

# Mistral
Agent(provider="ollama", model="mistral")
Agent(provider="ollama", model="mistral:7b-instruct")

# Mixtral (mixture of experts)
Agent(provider="ollama", model="mixtral")
Agent(provider="ollama", model="mixtral:8x7b")

# Gemma 2 (Google)
Agent(provider="ollama", model="gemma2")
Agent(provider="ollama", model="gemma2:27b")

# Qwen 2.5 (Alibaba)
Agent(provider="ollama", model="qwen2.5")
Agent(provider="ollama", model="qwen2.5:72b")
```

#### Custom Host

```python
# Remote Ollama server
agent = Agent(
    provider="ollama",
    model="llama3.2",
    base_url="http://192.168.1.100:11434"
)

# Different port
agent = Agent(
    provider="ollama",
    model="llama3.2",
    base_url="http://localhost:8080"
)
```

#### Advanced Parameters

```python
agent = Agent(
    provider="ollama",
    model="llama3.2",

    # Generation parameters
    temperature=0.8,
    num_predict=500,        # Max tokens to generate
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,

    # System configuration
    system_prompt="You are an expert evaluator.",

    # Timeout
    timeout=60,             # Ollama can be slow for large models
)
```

#### Model Management

```python
import requests

def list_ollama_models(host="http://localhost:11434"):
    """List installed Ollama models."""
    response = requests.get(f"{host}/api/tags")
    models = response.json()['models']

    print("Installed models:")
    for model in models:
        print(f"  - {model['name']} ({model['size'] / 1e9:.1f}GB)")

    return [m['name'] for m in models]

# Use it
available_models = list_ollama_models()
```

---

### Google Configuration

#### Basic Setup

```python
from agorai.synthesis import Agent

agent = Agent(
    provider="google",
    model="gemini-pro",
    api_key="..."  # or use GOOGLE_API_KEY environment variable
)
```

#### Available Models

```python
# Gemini Pro
Agent(provider="google", model="gemini-pro")

# Gemini Pro Vision (multimodal)
Agent(provider="google", model="gemini-pro-vision")
```

#### Advanced Parameters

```python
agent = Agent(
    provider="google",
    model="gemini-pro",

    # Generation parameters
    temperature=0.7,
    max_tokens=1000,
    top_p=0.95,
    top_k=40,

    # System configuration
    system_prompt="You are a helpful assistant.",
)
```

---

## Default Settings

### Changing Default Aggregation Method

```python
from agorai.aggregate import set_default_method

# Set default for all subsequent calls
set_default_method("atkinson", epsilon=1.0)

# Now aggregate() uses atkinson by default
from agorai.aggregate import aggregate
result = aggregate([[0.8, 0.2], [0.3, 0.7]])  # Uses atkinson
```

### Changing Default Metrics

```python
from agorai.queue import set_default_metrics

# Set default metrics for queue processing
set_default_metrics(['fairness', 'efficiency'])

# Now process_queue() uses these metrics by default
from agorai.queue import process_queue
result = process_queue("file.json", method="majority")  # Computes fairness and efficiency
```

---

## Production Configuration

### Recommended Production Setup

```python
import os
from agorai.synthesis import Agent, Council

# Load from environment
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

# Validate keys
if not OPENAI_KEY or not ANTHROPIC_KEY:
    raise ValueError("Missing API keys! Set OPENAI_API_KEY and ANTHROPIC_API_KEY")

# Create production agents with conservative settings
production_agents = [
    Agent(
        provider="openai",
        model="gpt-4",
        api_key=OPENAI_KEY,
        temperature=0.3,  # Lower temperature for consistent outputs
        max_tokens=500,
        timeout=30,
        max_retries=3
    ),
    Agent(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=ANTHROPIC_KEY,
        temperature=0.3,
        max_tokens=500,
        timeout=30
    )
]

# Create production council
production_council = Council(
    agents=production_agents,
    aggregation_method="majority",  # Simple, reliable method
    enable_caching=True
)
```

### Error Handling

```python
from agorai.aggregate import aggregate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_aggregate(utilities, method="majority", **params):
    """Production-ready aggregation with error handling."""
    try:
        result = aggregate(utilities, method=method, **params)
        logger.info(f"Aggregation successful: method={method}, winner={result['winner']}")
        return result

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        # Return default/fallback result
        return {
            'winner': 0,
            'scores': [1.0] + [0.0] * (len(utilities[0]) - 1),
            'method': 'fallback',
            'error': str(e)
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
```

### Caching

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def cached_aggregate(utilities_hash: str, method: str, **params):
    """Cached aggregation for repeated requests."""
    utilities = json.loads(utilities_hash)
    return aggregate(utilities, method=method, **params)

def aggregate_with_cache(utilities, method="majority", **params):
    """Aggregate with automatic caching."""
    # Create hash of utilities for cache key
    utilities_str = json.dumps(utilities, sort_keys=True)
    utilities_hash = hashlib.md5(utilities_str.encode()).hexdigest()

    return cached_aggregate(utilities_hash, method, **params)
```

### Rate Limiting

```python
from ratelimit import limits, sleep_and_retry
import time

# Allow 100 requests per minute
@sleep_and_retry
@limits(calls=100, period=60)
def rate_limited_synthesis(prompt, agents, **kwargs):
    """Synthesis with rate limiting."""
    from agorai.synthesis import synthesize
    return synthesize(prompt=prompt, agents=agents, **kwargs)
```

---

## Development Setup

### Development Dependencies

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or manually
pip install pytest pytest-cov black flake8 mypy ipython jupyter
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agorai --cov-report=html

# Run specific test
pytest tests/test_aggregate.py::test_majority

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type check
mypy src/
```

### Development Configuration

Create `dev_config.py`:

```python
# dev_config.py
import os

# Use test API keys
os.environ['OPENAI_API_KEY'] = "sk-test-..."
os.environ['ANTHROPIC_API_KEY'] = "sk-ant-test-..."

# Enable debug mode
os.environ['AGORAI_DEBUG'] = "1"

# Use local Ollama
os.environ['OLLAMA_HOST'] = "http://localhost:11434"

# Set test data directory
TEST_DATA_DIR = "tests/data"
```

Use in tests:

```python
import dev_config
from agorai.synthesis import Agent

# Uses test API keys from dev_config
agent = Agent(provider="openai", model="gpt-3.5-turbo")
```

### Jupyter Notebook Setup

```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '../src')  # Add package to path

from agorai.aggregate import aggregate
from agorai.queue import process_queue
from agorai.visualization import plot_utility_matrix

# Enable inline plots
%matplotlib inline
```

---

## Configuration Examples

### Example 1: Research Lab Setup

```python
# research_config.py

import os
from agorai.synthesis import Agent

# Use institutional API keys
OPENAI_KEY = os.getenv("RESEARCH_OPENAI_KEY")
ANTHROPIC_KEY = os.getenv("RESEARCH_ANTHROPIC_KEY")

# Create diverse research agents
research_agents = [
    Agent(provider="openai", model="gpt-4", api_key=OPENAI_KEY),
    Agent(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_KEY),
    Agent(provider="ollama", model="llama3.2"),
    Agent(provider="ollama", model="mistral"),
    Agent(provider="ollama", model="gemma2"),
]

# Default research methods
RESEARCH_METHODS = ["majority", "borda", "atkinson", "maximin", "nash_bargaining"]

# Default research metrics
RESEARCH_METRICS = ["fairness", "efficiency", "agreement"]
```

### Example 2: Production API Setup

```python
# production_config.py

import os
from agorai.synthesis import Agent

# Strict production configuration
OPENAI_KEY = os.getenv("PROD_OPENAI_KEY")
if not OPENAI_KEY:
    raise EnvironmentError("PROD_OPENAI_KEY not set!")

# Single reliable agent
production_agent = Agent(
    provider="openai",
    model="gpt-4",
    api_key=OPENAI_KEY,
    temperature=0.2,  # Low temperature for consistency
    max_tokens=300,
    timeout=20,
    max_retries=3,
    system_prompt="You are a production decision-making system. Be concise and consistent."
)

# Simple, fast aggregation
DEFAULT_METHOD = "majority"

# Minimal metrics for speed
DEFAULT_METRICS = ["efficiency"]
```

### Example 3: Local Development

```python
# local_config.py

import os

# Use Ollama exclusively (no API costs)
os.environ['OLLAMA_HOST'] = "http://localhost:11434"

from agorai.synthesis import Agent

# Free local models
dev_agents = [
    Agent(provider="ollama", model="llama3.2"),
    Agent(provider="ollama", model="mistral"),
]

# Fast methods for iteration
DEV_METHODS = ["majority", "borda"]

# Skip expensive metrics
DEV_METRICS = []
```

---

## Troubleshooting

### API Key Issues

```python
# Test API keys
from agorai.synthesis import Agent

try:
    agent = Agent(provider="openai", model="gpt-3.5-turbo", api_key="sk-test")
    print("✓ OpenAI key loaded")
except Exception as e:
    print(f"✗ OpenAI key failed: {e}")
```

### Ollama Connection Issues

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# List models
ollama list
```

### Import Issues

```python
# Check installation
try:
    import agorai
    print(f"✓ AgorAI version: {agorai.__version__}")
except ImportError:
    print("✗ AgorAI not installed: pip install agorai")

# Check optional modules
try:
    import agorai.queue
    print("✓ Queue module available")
except ImportError:
    print("✗ Queue module not available: pip install agorai[research]")

try:
    import agorai.visualization
    print("✓ Visualization module available")
except ImportError:
    print("✗ Visualization module not available: pip install agorai[research]")
```

---

## Summary

This guide covered:

✅ **Installation options** - Minimal, research, synthesis, complete
✅ **Environment variables** - API keys and configuration
✅ **LLM provider configuration** - OpenAI, Anthropic, Ollama, Google
✅ **Default settings** - Changing defaults for methods and metrics
✅ **Production configuration** - Error handling, caching, rate limiting
✅ **Development setup** - Testing, code quality, Jupyter notebooks

---

**Next steps:**
- [Extending Guide](extending.md) - Add custom methods and integrations
- [Queue Processing Guide](queue.md) - Batch processing from files
- [Visualization Guide](visualization.md) - Create plots and explanations
