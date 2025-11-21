# Backend Migration Guide: Using AgorAI PyPI Package

This guide explains how to migrate your existing backend to use the `agorai` PyPI package instead of the local codebase.

## Overview

The AgorAI functionality has been extracted into three reusable modules:
- `agorai.aggregate` - Pure mathematical aggregation (no LLM dependencies)
- `agorai.synthesis` - LLM-based opinion synthesis
- `agorai.bias` - Full bias mitigation pipeline

## Migration Steps

### Step 1: Install AgorAI Package

```bash
# Install with all features
pip install agorai[all]

# Or install selectively
pip install agorai[synthesis]  # For LLM integration only
pip install agorai[bias]        # For bias mitigation only
```

### Step 2: Update Import Statements

#### Before (Local Package)

```python
# backend/services/llm_manager.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from agorai import AgentConfig, BaseAgent, OpenAIAgent, OllamaAgent, MockAgent
from agorai.aggregators import AGGREGATOR_REGISTRY, MajorityAggregator
from agorai.council import AgentCouncil, CouncilConfig
```

#### After (PyPI Package)

```python
# backend/services/llm_manager.py
# No sys.path manipulation needed!

from agorai.synthesis import Agent, Council
from agorai.aggregate import aggregate, list_methods
from agorai.bias import mitigate_bias, BiasConfig
```

### Step 3: Update Agent Creation Logic

The new package has a cleaner Agent API.

#### Before

```python
# Old way (local package)
from agorai import AgentConfig, OllamaAgent

agent_config = AgentConfig(
    name="test-agent",
    model="llama3.2",
    system_prompt="You are helpful"
)

agent = OllamaAgent(
    cfg=agent_config,
    host="127.0.0.1",
    port=11434
)

output = agent.generate(prompt="Hello")
```

#### After

```python
# New way (PyPI package)
from agorai.synthesis import Agent

agent = Agent(
    provider="ollama",
    model="llama3.2",
    base_url="http://localhost:11434",
    system_prompt="You are helpful",
    name="test-agent"
)

output = agent.generate(prompt="Hello")
```

### Step 4: Update Aggregation Logic

The aggregation API is now simpler and more consistent.

#### Before

```python
# Old way (local package)
from agorai.aggregators import AGGREGATOR_REGISTRY

# Create aggregator instance
aggregator = AGGREGATOR_REGISTRY["atkinson"]()

# Prepare agent outputs
candidates = ["option1", "option2"]
agent_outputs = [
    {"text": "I prefer option1", "scores": {"option1": 0.8, "option2": 0.2}},
    {"text": "I prefer option2", "scores": {"option1": 0.3, "option2": 0.7}},
]

# Aggregate
result = aggregator.aggregate(candidates, agent_outputs)
winner = result["winner"]
```

#### After

```python
# New way (PyPI package)
from agorai.aggregate import aggregate

# Simple utility matrix
utilities = [
    [0.8, 0.2],  # Agent 1's utilities
    [0.3, 0.7],  # Agent 2's utilities
]

# Aggregate
result = aggregate(utilities, method="atkinson", epsilon=1.0)
winner = result["winner"]  # Returns index (0 or 1)
```

### Step 5: Update Council/Multi-Agent Logic

The new synthesis module provides a cleaner multi-agent API.

#### Before

```python
# Old way (local package)
from agorai.council import AgentCouncil, CouncilConfig
from agorai import OllamaAgent, AgentConfig

# Create agents
agents = [
    OllamaAgent(cfg=AgentConfig(name="agent1", model="llama3.2"), host="127.0.0.1", port=11434),
    OllamaAgent(cfg=AgentConfig(name="agent2", model="llama3.2"), host="127.0.0.1", port=11434),
]

# Create council
config = CouncilConfig(
    name="test-council",
    aggregator="majority",
    max_rounds=1
)
council = AgentCouncil(agents=agents, config=config)

# Run decision
result = council.run_once(prompt="Should we approve?")
decision = result["result"]
```

#### After

```python
# New way (PyPI package)
from agorai.synthesis import Agent, Council

# Create agents (simpler!)
agents = [
    Agent("ollama", "llama3.2", name="agent1"),
    Agent("ollama", "llama3.2", name="agent2"),
]

# Create council (simpler!)
council = Council(
    agents=agents,
    aggregation_method="majority"
)

# Run decision
result = council.decide(
    prompt="Should we approve?",
    candidates=["approve", "reject"]
)
decision = result["decision"]
```

### Step 6: Update Bias Mitigation Code

The new bias module provides a high-level pipeline.

#### Before

```python
# Old way (complex manual setup)
from agorai.contexts import BIAS_MITIGATION, get_council_config
from agorai.council import AgentCouncil
from agorai import OllamaAgent, AgentConfig

# Manual cultural perspective setup
agents = []
for i, prompt in enumerate(cultural_prompts):
    agent = OllamaAgent(
        cfg=AgentConfig(
            name=f"perspective-{i}",
            model="llama3.2",
            system_prompt=prompt
        ),
        host="127.0.0.1",
        port=11434
    )
    agents.append(agent)

# Create council with bias mitigation config
config = get_council_config(BIAS_MITIGATION, aggregator="schulze_condorcet")
council = AgentCouncil(agents=agents, config=config)

# Run evaluation
result = council.run_once(prompt="Is this content hateful?")
```

#### After

```python
# New way (simple high-level API!)
from agorai.bias import mitigate_bias, BiasConfig

# One-liner with sensible defaults
result = mitigate_bias(
    input_text="Is this content hateful?",
    candidates=["hateful", "not_hateful"],
    config=BiasConfig(
        context="hate_speech_detection",
        providers=["ollama"],
        aggregation_method="schulze_condorcet",
        cultural_perspectives=5
    )
)

decision = result["decision"]
fairness = result["fairness_metrics"]
```

## Complete Backend Service Example

Here's how to refactor a complete backend service:

### Before (backend/services/evaluation.py)

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from agorai.council import AgentCouncil, CouncilConfig
from agorai import OllamaAgent, AgentConfig
from agorai.aggregators import AGGREGATOR_REGISTRY

class EvaluationService:
    def run_evaluation(self, prompt: str, agent_configs: List[Dict]):
        # Create agents
        agents = []
        for config in agent_configs:
            agent = OllamaAgent(
                cfg=AgentConfig(
                    name=config["name"],
                    model=config["model"],
                    system_prompt=config.get("system_prompt", "")
                ),
                host=config.get("host", "127.0.0.1"),
                port=config.get("port", 11434)
            )
            agents.append(agent)

        # Create council
        council_config = CouncilConfig(
            name="eval-council",
            aggregator=config.get("aggregator", "majority"),
            max_rounds=1
        )
        council = AgentCouncil(agents=agents, config=council_config)

        # Run evaluation
        result = council.run_once(prompt=prompt)
        return result["result"]
```

### After (backend/services/evaluation.py)

```python
from agorai.synthesis import Agent, synthesize
from agorai.aggregate import list_methods

class EvaluationService:
    def run_evaluation(self, prompt: str, agent_configs: List[Dict], candidates: List[str]):
        # Create agents (simpler!)
        agents = [
            Agent(
                provider=config["provider"],
                model=config["model"],
                base_url=config.get("base_url"),
                system_prompt=config.get("system_prompt", ""),
                name=config["name"]
            )
            for config in agent_configs
        ]

        # Run synthesis (one function call!)
        result = synthesize(
            prompt=prompt,
            agents=agents,
            candidates=candidates,
            aggregation_method=config.get("aggregator", "majority")
        )

        return result["decision"]

    def list_aggregation_methods(self):
        """New feature: easily list all available methods"""
        return list_methods()
```

## API Mapping Reference

### Agent Classes

| Old (Local) | New (PyPI) | Notes |
|-------------|------------|-------|
| `AgentConfig` | `Agent(...)` kwargs | Configuration is now kwargs |
| `OllamaAgent(cfg, host, port)` | `Agent("ollama", model, base_url)` | Unified interface |
| `OpenAIAgent(cfg)` | `Agent("openai", model, api_key)` | Now fully implemented |
| `MockAgent(cfg)` | N/A | No longer needed (use Ollama) |
| `BaseAgent.generate(prompt)` | `Agent.generate(prompt)` | Same interface |

### Aggregation

| Old (Local) | New (PyPI) | Notes |
|-------------|------------|-------|
| `AGGREGATOR_REGISTRY["name"]()` | `aggregate(..., method="name")` | Functional API |
| `aggregator.aggregate(candidates, outputs)` | `aggregate(utilities, method)` | Direct utilities |
| `aggregator.prepare(...)` | `prepare_utilities(...)` | Standalone function |

### Council/Multi-Agent

| Old (Local) | New (PyPI) | Notes |
|-------------|------------|-------|
| `AgentCouncil(agents, config)` | `Council(agents, aggregation_method)` | Simpler init |
| `council.run_once(prompt)` | `council.decide(prompt, candidates)` | Explicit candidates |
| `council.run_debate(prompt)` | Multiple `decide()` calls | More control |
| `CouncilConfig(...)` | kwargs to `Council(...)` | No separate config class |

## Benefits of Migration

### 1. Cleaner Code

**Before:**
```python
# 15+ lines of boilerplate
import sys, os
sys.path.insert(0, ...)
from agorai import AgentConfig, OllamaAgent
agent_config = AgentConfig(...)
agent = OllamaAgent(cfg=agent_config, host=..., port=...)
```

**After:**
```python
# 2 lines
from agorai.synthesis import Agent
agent = Agent("ollama", "llama3.2")
```

### 2. Better Separation of Concerns

- **aggregate**: Pure math, no LLM dependencies
- **synthesis**: LLM integration only
- **bias**: High-level bias mitigation

Install only what you need!

### 3. Improved Documentation

- Comprehensive docstrings
- Type hints
- Examples in every function
- Separate docs for each module

### 4. Easier Testing

```python
# Test aggregation without any LLM
from agorai.aggregate import aggregate
result = aggregate([[1,0],[1,0],[0,1]], method="majority")
assert result["winner"] == 0
```

### 5. Version Management

```bash
# Lock to specific version
pip install agorai==0.1.0

# Update when ready
pip install --upgrade agorai
```

## Backward Compatibility Notes

### Breaking Changes

1. **Agent creation**: No more `AgentConfig` dataclass
2. **Aggregation**: Now uses utility matrices directly
3. **Council**: Simplified to `decide()` method
4. **Imports**: Different module structure

### Non-Breaking

- Agent `generate()` method signature unchanged
- Aggregation method names unchanged
- Output formats mostly compatible

## Migration Checklist

- [ ] Install `agorai` package: `pip install agorai[all]`
- [ ] Remove `sys.path.insert()` hacks
- [ ] Update imports to use `agorai.synthesis`, `agorai.aggregate`, `agorai.bias`
- [ ] Replace `AgentConfig` + `OllamaAgent` with `Agent("ollama", ...)`
- [ ] Replace `AGGREGATOR_REGISTRY["name"]()` with `aggregate(..., method="name")`
- [ ] Replace `AgentCouncil` with `Council` or `synthesize()`
- [ ] Update tests to use new API
- [ ] Remove local `agorai/` package from `sys.path`
- [ ] Test all functionality end-to-end
- [ ] Update documentation

## Troubleshooting

### Issue: "No module named 'agorai'"

**Solution:** Install the package:
```bash
pip install agorai[all]
```

### Issue: "Agent object has no attribute 'cfg'"

**Solution:** The new API doesn't use `AgentConfig`. Update to:
```python
# Old
agent.cfg.name

# New
agent.name
```

### Issue: "aggregate() missing positional argument 'candidates'"

**Solution:** The new API uses utility matrices:
```python
# Old
aggregator.aggregate(candidates=["A", "B"], agent_outputs=[...])

# New
aggregate(utilities=[[0.8, 0.2], [0.3, 0.7]], method="majority")
```

### Issue: "Council has no method run_once()"

**Solution:** Use `decide()`:
```python
# Old
result = council.run_once(prompt="...")

# New
result = council.decide(prompt="...", candidates=["yes", "no"])
```

## Support

If you encounter issues during migration:

1. Check the [API documentation](docs/)
2. Review the [examples](examples/)
3. Open an issue on GitHub
4. Consult the [PyPI package page](https://pypi.org/project/agorai/)

## Next Steps

After migration:

1. Consider removing the local `agorai/` directory
2. Update your `requirements.txt`:
   ```
   agorai[all]>=0.1.0
   ```
3. Share feedback on the new API
4. Contribute improvements via pull requests
