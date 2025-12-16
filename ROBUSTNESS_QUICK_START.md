# AgorAI Robustness Features - Quick Start Guide

## Overview

The AgorAI package now includes comprehensive robustness features including timeout handling, logging, metrics collection, circuit breakers, and transaction safety. This guide shows you how to use these features.

---

## 🚀 Quick Start

### Basic Usage (No Changes Required)

Your existing code continues to work without any modifications:

```python
from agorai import Agent, synthesize_structured

agents = [
    Agent("ollama", "llama3.2", name="Agent 1"),
    Agent("ollama", "llama3.2", name="Agent 2")
]

result = synthesize_structured(
    question="Should we implement this feature?",
    options=["Yes, immediately", "No, reject", "Defer for later"],
    agents=agents
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2f}")

# NEW: Access metrics
print(f"\nMetrics:")
print(f"  Average response time: {result['metrics']['avg_response_time']:.2f}s")
print(f"  Total retries: {result['metrics']['total_retries']}")
print(f"  Timeouts: {result['metrics']['timeout_count']}")
```

---

## ⚙️ Configuration

### View Current Configuration

```python
from agorai import config

print(f"LLM Timeout: {config.LLM_TIMEOUT}s")
print(f"Max Retries: {config.MAX_RETRIES}")
print(f"Max Prompt Length: {config.MAX_PROMPT_LENGTH} chars")
print(f"Log Directory: {config.get_log_dir()}")
```

### Override Configuration via Environment Variables

```bash
# Set timeout to 60 seconds
export AGORAI_LLM_TIMEOUT=60

# Set max retries to 5
export AGORAI_MAX_RETRIES=5

# Set log level to DEBUG
export AGORAI_LOG_LEVEL=DEBUG

# Set custom log directory
export AGORAI_LOG_DIR=/path/to/logs
```

```python
# Or in Python
import os
os.environ['AGORAI_LLM_TIMEOUT'] = '60'
os.environ['AGORAI_LOG_LEVEL'] = 'DEBUG'

# Then import agorai
from agorai import Agent, config
```

### All Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_TIMEOUT` | 30.0 | LLM generation timeout (seconds) |
| `MAX_RETRIES` | 2 | Maximum retry attempts |
| `MAX_PROMPT_LENGTH` | 100,000 | Maximum prompt length (characters) |
| `MIN_OPTIONS` | 2 | Minimum options for structured synthesis |
| `MAX_OPTIONS` | 20 | Maximum options for structured synthesis |
| `MIN_TEMPERATURE` | 0.0 | Minimum LLM temperature |
| `MAX_TEMPERATURE` | 2.0 | Maximum LLM temperature |
| `CIRCUIT_BREAKER_FAIL_MAX` | 5 | Failures before circuit opens |
| `CIRCUIT_BREAKER_RESET_TIMEOUT` | 60 | Seconds before circuit reset |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `ENABLE_TOKEN_CACHE` | True | Enable tokenization caching |

---

## 📊 Metrics

Every `synthesize_structured()` call now returns detailed metrics:

```python
result = synthesize_structured(...)
metrics = result['metrics']

# Available metrics:
print(f"Agent response times: {metrics['agent_response_times']}")  # List[float]
print(f"Average response time: {metrics['avg_response_time']}")    # float
print(f"Total retries: {metrics['total_retries']}")                # int
print(f"Validation failures: {metrics['validation_failures']}")    # int
print(f"Aggregation time: {metrics['aggregation_time']}")          # float
print(f"Total time: {metrics['total_time']}")                      # float
print(f"Agent names: {metrics['agent_names']}")                    # List[str]
print(f"Timeout count: {metrics['timeout_count']}")                # int
print(f"Error count: {metrics['error_count']}")                    # int
print(f"Errors: {metrics['errors']}")                              # List[Dict]
```

---

## 📝 Logging

### Log Location

Logs are automatically written to:
- **Linux/Mac:** `~/.agorai/logs/agorai_YYYYMMDD.log`
- **Windows:** `%USERPROFILE%\.agorai\logs\agorai_YYYYMMDD.log`

### Log Format

```
[TIMESTAMP] [LEVEL] [module:function:line] MESSAGE
```

Example:
```
[2025-12-16 14:23:45.123] [INFO] [agorai.synthesis.core:generate_structured:247] Agent openai:gpt-4: Starting structured generation with 3 options
[2025-12-16 14:23:47.456] [WARNING] [agorai.synthesis.core:generate_structured:296] Agent openai:gpt-4: Validation failed - response not in valid range
[2025-12-16 14:23:49.789] [INFO] [agorai.synthesis.core:generate_structured:285] Agent openai:gpt-4: Validation successful (option 2)
```

### Access Logger in Your Code

```python
from agorai import get_logger

logger = get_logger(__name__)
logger.info("Starting my synthesis task")
logger.warning("Something unusual happened")
logger.error("An error occurred")
```

### View Logs

```bash
# View latest log file
tail -f ~/.agorai/logs/agorai_$(date +%Y%m%d).log

# Search for errors
grep ERROR ~/.agorai/logs/agorai_*.log

# Search for specific agent
grep "Agent openai:gpt-4" ~/.agorai/logs/agorai_*.log
```

---

## 🔧 Circuit Breaker

Circuit breakers automatically protect against cascading failures from unresponsive LLM APIs.

### How It Works

1. **CLOSED (Normal):** Requests pass through normally
2. **OPEN (Failing):** After 5 consecutive failures, circuit opens and blocks requests
3. **HALF_OPEN (Testing):** After 60 seconds, circuit attempts to close by testing one request

### Access Circuit Breaker State

```python
from agorai import Agent

agent = Agent("openai", "gpt-4", api_key="sk-...")

# Check circuit breaker state
print(f"Circuit breaker state: {agent._circuit_breaker.state}")  # CLOSED/OPEN/HALF_OPEN

# Manually reset if needed
agent._circuit_breaker.reset()
```

### Handle Circuit Breaker Errors

```python
from agorai import CircuitBreakerError

try:
    result = synthesize_structured(...)
except CircuitBreakerError as e:
    print(f"Circuit breaker is open: {e}")
    print("Service is temporarily unavailable, try again later")
```

---

## ⏱️ Timeout Handling

### How It Works

- Default timeout: 30 seconds per LLM generation
- Timeout applies to each generation attempt (including retries)
- Configurable via `AGORAI_LLM_TIMEOUT` environment variable

### Handle Timeouts

```python
try:
    result = synthesize_structured(...)
except TimeoutError as e:
    print(f"LLM generation timed out: {e}")
    print("Try again or increase timeout")
```

### Adjust Timeout

```python
import os
os.environ['AGORAI_LLM_TIMEOUT'] = '60'  # 60 seconds

# Or for a specific use case
from agorai import config
config.LLM_TIMEOUT = 60.0
```

---

## ✅ Validation Features

### Agent Configuration Validation

```python
from agorai import Agent

# Valid agent
agent = Agent("openai", "gpt-4", temperature=0.7)  # ✓

# Invalid temperature
try:
    agent = Agent("openai", "gpt-4", temperature=3.0)  # ✗
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: temperature must be between 0.0 and 2.0, got 3.0

# Empty provider
try:
    agent = Agent("", "gpt-4")  # ✗
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: provider must be a non-empty string
```

### Options Validation

```python
from agorai import synthesize_structured

# Valid options
result = synthesize_structured(
    question="Choose one:",
    options=["Option A", "Option B", "Option C"],  # ✓
    agents=agents
)

# Too few options
try:
    result = synthesize_structured(
        question="Choose one:",
        options=["Only one"],  # ✗ (minimum is 2)
        agents=agents
    )
except ValueError as e:
    print(f"Validation error: {e}")

# Duplicate options
try:
    result = synthesize_structured(
        question="Choose one:",
        options=["Option A", "Option A", "Option B"],  # ✗
        agents=agents
    )
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: Options must be unique

# Empty option
try:
    result = synthesize_structured(
        question="Choose one:",
        options=["Option A", "", "Option B"],  # ✗
        agents=agents
    )
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: Options cannot be empty or whitespace
```

### Prompt Length Validation

```python
# Automatically validated
long_prompt = "x" * 150000  # Exceeds default 100,000 limit

try:
    result = synthesize_structured(
        question=long_prompt,
        options=["A", "B"],
        agents=agents
    )
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: Prompt length 150000 exceeds maximum 100000 characters
```

---

## 🔒 Transaction Safety (Backend)

Benchmark creation now uses transaction safety to prevent data corruption:

```python
# Automatic - no code changes needed
# If benchmark creation fails, partial data is automatically cleaned up

try:
    benchmark = await benchmark_service.create_benchmark(upload_data)
    print(f"Benchmark created: {benchmark.benchmark_id}")
except Exception as e:
    # Rollback automatically happened
    print(f"Benchmark creation failed: {e}")
```

---

## 🎯 Complete Example

```python
import os
from agorai import Agent, synthesize_structured, get_logger, config

# Configure (optional)
os.environ['AGORAI_LLM_TIMEOUT'] = '45'
os.environ['AGORAI_LOG_LEVEL'] = 'INFO'

# Get logger
logger = get_logger(__name__)
logger.info("Starting synthesis task")

# Create agents
agents = [
    Agent("ollama", "llama3.2", name="Technical Lead", temperature=0.7),
    Agent("ollama", "llama3.2", name="Product Manager", temperature=0.8),
    Agent("ollama", "llama3.2", name="UX Designer", temperature=0.6),
]

# Run synthesis with error handling
try:
    result = synthesize_structured(
        question="Should we implement dark mode for the application?",
        options=[
            "Yes, implement immediately",
            "No, not a priority",
            "Yes, but after Q1",
            "Need more user research first"
        ],
        agents=agents,
        context="Budget: $50k, Timeline: Q1 2024, User requests: 127"
    )

    # Process results
    print(f"\n✓ Decision: {result['decision']}")
    print(f"✓ Confidence: {result['confidence']:.2%}")
    print(f"✓ Method: {result['method']}")

    # Show agent reasoning
    print("\nAgent Reasoning:")
    for output in result['agent_outputs']:
        print(f"  {output['agent']}: Option {output['response_number']}")
        print(f"    → {output['reasoning']}")

    # Show metrics
    metrics = result['metrics']
    print(f"\nMetrics:")
    print(f"  Total time: {metrics['total_time']:.2f}s")
    print(f"  Avg response time: {metrics['avg_response_time']:.2f}s")
    print(f"  Total retries: {metrics['total_retries']}")
    print(f"  Validation failures: {metrics['validation_failures']}")
    print(f"  Timeouts: {metrics['timeout_count']}")

    logger.info(f"Synthesis completed: {result['decision']}")

except TimeoutError as e:
    logger.error(f"Timeout error: {e}")
    print("⚠️  One or more agents timed out. Try increasing AGORAI_LLM_TIMEOUT.")

except ValueError as e:
    logger.error(f"Validation error: {e}")
    print(f"⚠️  Validation failed: {e}")

except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    print(f"✗ An error occurred: {e}")

finally:
    print(f"\nLogs written to: {config.get_log_dir()}")
```

---

## 📚 Additional Resources

- **Full Documentation:** [ROBUSTNESS_IMPLEMENTATION_SUMMARY.md](ROBUSTNESS_IMPLEMENTATION_SUMMARY.md)
- **Original Requirements:** [ROBUSTNESS_IMPROVEMENTS.md](ROBUSTNESS_IMPROVEMENTS.md)
- **Package Documentation:** [docs/STRUCTURED_SYNTHESIS.md](docs/STRUCTURED_SYNTHESIS.md)

---

## 🐛 Troubleshooting

### Logs Not Appearing
```bash
# Check log directory exists
ls -la ~/.agorai/logs/

# Check log level
python -c "from agorai import config; print(config.LOG_LEVEL)"

# Set to DEBUG to see all logs
export AGORAI_LOG_LEVEL=DEBUG
```

### Timeout Issues
```bash
# Check current timeout
python -c "from agorai import config; print(f'Timeout: {config.LLM_TIMEOUT}s')"

# Increase timeout
export AGORAI_LLM_TIMEOUT=120
```

### Circuit Breaker Always Open
```python
from agorai import Agent

agent = Agent(...)
print(f"Circuit state: {agent._circuit_breaker.state}")

# Manually reset
agent._circuit_breaker.reset()
```

### Import Errors
```bash
# Reinstall package
cd /path/to/agorai-package
pip install -e .

# Or ensure src is in Python path
export PYTHONPATH=/path/to/agorai-package/src:$PYTHONPATH
```

---

## ✨ Key Benefits

1. **Automatic Timeout Protection** - No more infinite hangs
2. **Detailed Logging** - Full observability of synthesis operations
3. **Performance Metrics** - Track and optimize synthesis performance
4. **Circuit Breaker** - Automatic protection from failing APIs
5. **Comprehensive Validation** - Catch errors early with clear messages
6. **Transaction Safety** - No data corruption in benchmark creation
7. **Zero Breaking Changes** - All existing code continues to work
8. **Configurable** - Fine-tune behavior via environment variables

---

**All robustness features are production-ready and fully backward compatible!**
