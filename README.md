# AgorAI: Democratic AI Through Multi-Agent Aggregation

AgorAI is a Python library for building fair, unbiased AI systems through democratic multi-agent opinion aggregation. It combines social choice theory, welfare economics, and modern LLMs to enable collective decision-making with provable fairness guarantees.

## Features

- **Pure Mathematical Aggregation** (`agorai.aggregate`): 25+ aggregation methods from social choice theory, welfare economics, and game theory
- **LLM-Based Synthesis** (`agorai.synthesis`): Multi-provider LLM integration (OpenAI, Anthropic, Ollama, Google) with unified opinion synthesis
- **Bias Mitigation** (`agorai.bias`): Full pipeline for detecting and mitigating AI bias through cultural perspective diversity

## Installation

```bash
# Minimal installation (aggregation only)
pip install agorai

# With LLM synthesis support
pip install agorai[synthesis]

# With bias mitigation support
pip install agorai[bias]

# Full installation
pip install agorai[all]
```

## Quick Start

### 1. Pure Mathematical Aggregation

```python
from agorai.aggregate import aggregate

# Aggregate utilities from multiple agents
utilities = [
    [0.8, 0.2, 0.5],  # Agent 1's utilities for 3 candidates
    [0.3, 0.7, 0.4],  # Agent 2's utilities
    [0.6, 0.5, 0.9],  # Agent 3's utilities
]

result = aggregate(utilities, method="atkinson", epsilon=1.0)
print(result)
# {'winner': 2, 'scores': [0.54, 0.42, 0.58], 'method': 'atkinson'}
```

### 2. LLM-Based Opinion Synthesis

```python
from agorai.synthesis import synthesize, Agent

# Create diverse agents
agents = [
    Agent(provider="openai", model="gpt-4", api_key="sk-..."),
    Agent(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-..."),
    Agent(provider="ollama", model="llama3.2"),
]

# Synthesize opinions
result = synthesize(
    prompt="Should we approve this marketing campaign?",
    agents=agents,
    aggregation_method="majority"
)

print(result['decision'])  # The aggregated decision
print(result['confidence'])  # Confidence score
```

### 3. Bias Mitigation

```python
from agorai.bias import mitigate_bias, BiasConfig

# Configure bias mitigation
config = BiasConfig(
    context="hate_speech_detection",
    providers=["openai", "anthropic"],
    aggregation_method="schulze_condorcet",
    cultural_perspectives=5
)

# Mitigate bias in content moderation
result = mitigate_bias(
    input_text="Is this content appropriate?",
    config=config
)

print(result['decision'])  # Bias-mitigated decision
print(result['fairness_metrics'])  # Fairness analysis
```

## Available Aggregation Methods

- **Social Choice**: `majority`, `borda`, `schulze_condorcet`, `approval_voting`, `supermajority`
- **Welfare Economics**: `maximin`, `atkinson`, `nash_bargaining`
- **Machine Learning**: `centroid`, `robust_median`, `consensus`
- **Game Theory**: `quadratic_voting`, `veto_hybrid`

## Use Cases

### For AI Researchers
- Experiment with multi-agent architectures
- Integrate human feedback through democratic aggregation
- Compare fairness properties of different aggregation methods

### For ML Engineers
- Build bias-resistant content moderation systems
- Aggregate predictions from ensemble models with fairness guarantees
- Implement human-in-the-loop AI systems

### For Social Scientists
- Study collective decision-making in AI systems
- Analyze cultural bias in language models
- Evaluate fairness of algorithmic decisions

## Documentation

- **API Reference**: See [docs/aggregate.md](docs/aggregate.md) for complete aggregation API documentation
- **Migration Guide**: [BACKEND_MIGRATION_GUIDE.md](BACKEND_MIGRATION_GUIDE.md) - Migrate from local package
- **PyPI Upload Guide**: [PYPI_UPLOAD_GUIDE.md](PYPI_UPLOAD_GUIDE.md) - Publishing instructions
- **Examples**: [examples/backend_integration.py](examples/backend_integration.py) - Integration examples
- **Quick Start**: [QUICK_START.md](QUICK_START.md) - 5-minute tutorial

## Citation

If you use AgorAI in your research, please cite:

```bibtex
@software{agorai2025,
  title = {AgorAI: Democratic AI Through Multi-Agent Aggregation},
  author = {Schlenker, Samuel},
  year = {2025},
  url = {https://github.com/agorai/agorai}
}
```

## License

**Research and Non-Commercial License**

Copyright (c) 2025 Samuel Schlenker

This software is free to use for:
- ✅ Academic and scientific research
- ✅ Educational purposes
- ✅ Personal and private use
- ✅ Non-profit organizations

**Commercial use requires prior written agreement** with Samuel Schlenker.

**Full license terms**: The package includes a custom Research and Non-Commercial License. The complete license text is included in the `LICENSE` file within the package distribution.

For commercial licensing inquiries, please contact Samuel Schlenker.

## Contributing

Contributions for research and non-commercial purposes are welcome! Please see CONTRIBUTING.md for guidelines.
