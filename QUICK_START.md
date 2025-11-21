# AgorAI - Quick Start Guide

## Installation

```bash
# Install with all features
pip install agorai[all]

# Or install only what you need
pip install agorai              # Just aggregation
pip install agorai[synthesis]   # + LLM integration
pip install agorai[bias]        # + Bias mitigation
```

## 5-Minute Tutorial

### 1. Pure Mathematical Aggregation

No LLM required - just aggregate utilities!

```python
from agorai.aggregate import aggregate

# 3 agents rating 2 options
utilities = [
    [0.8, 0.2],  # Agent 1: prefers option 0
    [0.3, 0.7],  # Agent 2: prefers option 1
    [0.5, 0.5],  # Agent 3: neutral
]

# Majority voting
result = aggregate(utilities, method="majority")
print(f"Winner: Option {result['winner']}")  # Option 0

# Fairness-aware (Atkinson)
result = aggregate(utilities, method="atkinson", epsilon=1.0)
print(f"Winner: Option {result['winner']}")
```

### 2. LLM Opinion Synthesis

Aggregate opinions from multiple LLMs:

```python
from agorai.synthesis import Agent, synthesize

# Create agents from different providers
agents = [
    Agent("openai", "gpt-4", api_key="sk-..."),
    Agent("anthropic", "claude-3-5-sonnet-20241022", api_key="sk-ant-..."),
    Agent("ollama", "llama3.2"),  # Local
]

# Get collective decision
result = synthesize(
    prompt="Should we approve this feature?",
    agents=agents,
    candidates=["approve", "reject", "revise"],
    aggregation_method="majority"
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### 3. Bias Mitigation

Mitigate AI bias through cultural diversity:

```python
from agorai.bias import mitigate_bias, BiasConfig

# Configure bias mitigation
config = BiasConfig(
    context="hate_speech_detection",
    providers=["openai", "anthropic"],
    api_keys={"openai": "sk-...", "anthropic": "sk-ant-..."},
    cultural_perspectives=5,
    aggregation_method="atkinson"
)

# Analyze content with bias mitigation
result = mitigate_bias(
    input_text="Is this content appropriate?",
    config=config
)

print(f"Decision: {result['decision']}")
print(f"Fairness: {result['fairness_metrics']}")
```

## Available Methods

### Aggregation Methods (14 total)

```python
from agorai.aggregate import list_methods

for method in list_methods():
    print(f"{method['name']} ({method['category']})")
```

- **Social Choice**: majority, borda, schulze_condorcet, approval_voting, etc.
- **Welfare Economics**: maximin, atkinson
- **Machine Learning**: score_centroid, robust_median, consensus
- **Game Theory**: quadratic_voting, nash_bargaining, veto_hybrid

### LLM Providers (4 total)

- **OpenAI**: gpt-4, gpt-4o, gpt-3.5-turbo
- **Anthropic**: claude-3-5-sonnet, claude-3-opus
- **Ollama**: llama3.2, mistral, mixtral, gemma2
- **Google**: gemini-1.5-pro, gemini-1.5-flash

### Bias Contexts (4 presets)

- `hate_speech_detection`
- `content_moderation`
- `fairness_assessment`
- `cultural_sensitivity`

## Common Use Cases

### Ensemble Model Aggregation

```python
from agorai.aggregate import aggregate

# 5 model predictions for 3 classes
model_confidences = [
    [0.7, 0.2, 0.1],  # Model 1
    [0.6, 0.3, 0.1],  # Model 2
    [0.1, 0.8, 0.1],  # Model 3
    [0.5, 0.4, 0.1],  # Model 4
    [0.4, 0.5, 0.1],  # Model 5
]

result = aggregate(model_confidences, method="score_centroid")
print(f"Ensemble prediction: Class {result['winner']}")
```

### Content Moderation

```python
from agorai.bias import mitigate_bias, BiasConfig

config = BiasConfig(
    context="content_moderation",
    providers=["ollama"],  # Use local Ollama
)

result = mitigate_bias(
    input_text="User generated content here...",
    config=config
)

if result['decision'] == 'remove':
    print("⚠️ Content flagged for removal")
```

### Multi-Agent Deliberation

```python
from agorai.synthesis import Agent, Council

# Create specialized agents
agents = [
    Agent("openai", "gpt-4", system_prompt="You are a security expert", name="Security"),
    Agent("openai", "gpt-4", system_prompt="You are a UX expert", name="UX"),
    Agent("openai", "gpt-4", system_prompt="You are a business analyst", name="Business"),
]

council = Council(agents, aggregation_method="borda")

result = council.decide(
    prompt="Should we implement two-factor authentication?",
    candidates=["yes", "no", "partial"]
)

print(f"Council decision: {result['decision']}")
for output in result['agent_outputs']:
    print(f"  {output['agent']}: {output['text'][:100]}...")
```

## Documentation

- **Complete API Reference**: [docs/aggregate.md](docs/aggregate.md)
- **Migration Guide**: [BACKEND_MIGRATION_GUIDE.md](BACKEND_MIGRATION_GUIDE.md)
- **PyPI Upload**: [PYPI_UPLOAD_GUIDE.md](PYPI_UPLOAD_GUIDE.md)
- **Examples**: [examples/backend_integration.py](examples/backend_integration.py)

## Next Steps

1. **Explore examples**: `python examples/backend_integration.py`
2. **Read full docs**: Browse `docs/` directory
3. **Integrate into your app**: See migration guide
4. **Contribute**: Add custom aggregation methods or providers

## Support

- **GitHub Issues**: (Add URL after uploading)
- **PyPI Page**: https://pypi.org/project/agorai/
- **Documentation**: (Add ReadTheDocs URL if created)

---

**Ready to build fairer AI systems? Let's go!** 🚀
