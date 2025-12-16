# AgorAI: Democratic AI Through Multi-Agent Aggregation

[![PyPI version](https://badge.fury.io/py/agorai.svg)](https://badge.fury.io/py/agorai)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Custom](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)

AgorAI is a Python library for building fair, unbiased AI systems through democratic multi-agent opinion aggregation. It combines social choice theory, welfare economics, and modern LLMs to enable collective decision-making with provable fairness guarantees.

**Perfect for:** AI researchers, ML engineers, social scientists, and anyone building fair multi-agent systems.

## ✨ Features

### Core Modules

- **Pure Mathematical Aggregation** (`agorai.aggregate`): 14+ aggregation methods from social choice theory, welfare economics, and game theory
- **LLM-Based Synthesis** (`agorai.synthesis`): Multi-provider LLM integration (OpenAI, Anthropic, Ollama, Google) with unified opinion synthesis
- **Bias Mitigation** (`agorai.bias`): Full pipeline for detecting and mitigating AI bias through cultural perspective diversity

### 🆕 Research Modules (v0.2.0)

- **Queue Processing** (`agorai.queue`): Process multiple aggregation requests from files (production data, test datasets, benchmarks)
- **Visualization** (`agorai.visualization`): Generate publication-quality plots and natural language explanations
- **Property Verification** (`agorai.properties`): Coming soon - Formally verify axiom satisfaction

### 🛡️ Production-Ready Robustness Features

- **Timeout Handling**: Automatic timeout protection for LLM API calls (configurable, default 30s)
- **Circuit Breaker**: Automatic protection against cascading failures from unresponsive APIs
- **Comprehensive Logging**: Structured file-based logging with rotation (logs to `~/.agorai/logs/`)
- **Metrics Collection**: Detailed performance metrics for every synthesis operation
- **Input Validation**: Comprehensive validation of all inputs (temperature, options, prompt length, etc.)
- **Transaction Safety**: Atomic operations with automatic rollback for benchmark creation
- **Performance Optimization**: Tokenization caching for improved performance
- **Zero Breaking Changes**: All features are backward compatible

📚 **[Robustness Quick Start Guide](ROBUSTNESS_QUICK_START.md)** | **[Implementation Details](ROBUSTNESS_IMPLEMENTATION_SUMMARY.md)**

## 📦 Installation

### From PyPI (when published)

```bash
# Minimal installation (aggregation only)
pip install agorai

# With LLM synthesis support
pip install 'agorai[synthesis]'

# With bias mitigation support
pip install 'agorai[bias]'

# Full installation (all features)
pip install 'agorai[all]'
```

**Note for zsh users:** Always quote the brackets to prevent shell globbing: `'agorai[all]'`

### Local Development

```bash
# Clone and install in editable mode
cd path/to/agorai-package

# Minimal installation
pip install -e .

# With optional dependencies
pip install -e '.[synthesis]'
pip install -e '.[bias]'
pip install -e '.[all]'

# For development
pip install -e '.[dev]'
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

### 4. 🆕 Queue Processing (Batch Operations)

```python
from agorai.queue import process_queue, compare_methods_on_queue

# Process multiple requests from a file (production data, test datasets, etc.)
results = process_queue(
    requests_file="production_batch.json",
    method="atkinson",
    metrics=["fairness", "efficiency", "agreement"],
    epsilon=1.0
)

print(f"Processed: {results['num_requests']} requests")
print(f"Gini Coefficient: {results['summary']['fairness']['gini_coefficient']:.3f}")
print(f"Social Welfare: {results['summary']['efficiency']['social_welfare']:.2f}")

# Compare multiple methods on same queue
comparison = compare_methods_on_queue(
    requests_file="daily_decisions.json",
    methods=["majority", "atkinson", "maximin"],
    metrics=["fairness", "efficiency"]
)

print("Fairness Rankings:", comparison['rankings']['fairness_gini_coefficient'])
```

### 5. 🆕 Visualization & Explanations

```python
from agorai.visualization import plot_utility_matrix, explain_decision
from agorai.aggregate import aggregate

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

# Plot utility matrix
plot_utility_matrix(
    utilities,
    agent_labels=["Agent 1", "Agent 2", "Agent 3"],
    candidate_labels=["Option A", "Option B"],
    save_path="utilities.png"
)

# Get natural language explanation
result = aggregate(utilities, method="atkinson", epsilon=1.0)
explanation = explain_decision(
    utilities, "atkinson", result['winner'], result['scores'], epsilon=1.0
)
print(explanation)
# Output: "Candidate 0 won using Atkinson aggregation with ε=1.0 (geometric mean).
# How it works: Atkinson method computes the equally-distributed equivalent..."
```

## 📊 Available Aggregation Methods

### Social Choice Theory
- `majority` - One-agent-one-vote plurality
- `weighted_plurality` - Weighted voting
- `borda` - Borda count positional ranking
- `schulze_condorcet` - Condorcet-consistent ranking
- `approval_voting` - Multi-approval voting
- `supermajority` - Threshold-based consensus

### Welfare Economics
- `maximin` - Rawlsian fairness (maximize minimum utility)
- `atkinson` - Parameterizable inequality aversion

### Machine Learning
- `score_centroid` - Weighted average
- `robust_median` - Outlier-resistant median
- `consensus` - Agreement-focused aggregation

### Game Theory
- `quadratic_voting` - Intensity-aware voting with budget constraints
- `nash_bargaining` - Cooperative bargaining solution
- `veto_hybrid` - Minority protection via veto power

## 🎯 Use Cases

### For AI Researchers
- 🔬 **Benchmark aggregation methods** with scientific metrics (fairness, efficiency, agreement)
- 🧪 **Experiment with multi-agent architectures** (MARL, Constitutional AI, test-time compute)
- 📊 **Generate publication-quality figures** for papers
- ✅ **Compare fairness properties** with formal axiom verification (coming soon)

### For ML Engineers
- 🛡️ **Build bias-resistant systems** with cultural perspective diversity
- ⚖️ **Aggregate ensemble predictions** with provable fairness guarantees
- 🤝 **Implement human-in-the-loop AI** with democratic decision-making
- 📈 **Monitor fairness metrics** in production (Gini, Atkinson, etc.)

### For Social Scientists
- 🔍 **Study collective decision-making** in AI systems
- 🌍 **Analyze cultural bias** in language models
- 📐 **Evaluate algorithmic fairness** with rigorous metrics
- 🧮 **Bridge social choice theory** and modern AI

## 📚 Documentation

### Getting Started
- **[Documentation Index](docs/README.md)** - Complete documentation overview
- **[Configuration Guide](docs/configuration.md)** - Installation and setup
- **[Examples](examples/)** - Code examples and sample data
- **[Jupyter Notebooks](notebooks/)** - Interactive tutorials

### User Guides
- **[Aggregation API](docs/aggregate.md)** - All 14+ aggregation methods
- **[Queue Processing](docs/queue.md)** - Batch processing from files
- **[Visualization](docs/visualization.md)** - Plots and natural language explanations

### Developer Guides
- **[Extending AgorAI](docs/extending.md)** - Add custom methods, configure LLMs, integrate with your stack
- **[Configuration](docs/configuration.md)** - Environment setup, API keys, production config
- **[Backend Compatibility](BACKEND_COMPATIBILITY.md)** - Migration guide

## 🔬 Research & Papers

AgorAI bridges classical social choice theory with modern multi-agent AI systems:

**Connections to Recent AI:**
- **Energy-Based Models** - Democratic aggregation as energy minimization
- **Constitutional AI** - Formal methods for collective constitutional design (Anthropic)
- **Test-Time Compute** - Multi-round deliberation for complex decisions (OpenAI o1)
- **MARL** - Democratic reward aggregation for multi-agent reinforcement learning

**Key Properties:**
- ✅ Provable fairness guarantees (anonymity, monotonicity, Pareto efficiency)
- ✅ 14+ aggregation methods from social choice theory, welfare economics, game theory
- ✅ Scientific evaluation with metrics (Gini coefficient, Atkinson index, social welfare)
- ✅ Natural language explanations for interpretability

**Citing AgorAI:**
```bibtex
@software{agorai2025,
  author = {Schlenker, Samuel},
  title = {AgorAI: Democratic AI Through Multi-Agent Aggregation},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/yourusername/agorai}
}
```

## 🤝 Contributing

Contributions for research and non-commercial purposes are welcome!

**Areas where we'd love help:**
- 🧪 Additional benchmark datasets (PRISM, voting data, constitutional preferences)
- 📊 More visualization types (interactive plots, dashboards)
- ✅ Property verification module (formal axiom checking)
- 🏛️ Constitutional AI module (democratic constitution design)
- 📖 Documentation improvements and tutorials
- 🐛 Bug reports and feature requests

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🌟 Roadmap

### v0.3.0 (Next Release)
- [ ] Property verification module - Formally verify axiom satisfaction
- [ ] Additional benchmarks - PRISM dataset, voting data
- [ ] Interactive visualizations - Plotly/Dash dashboards
- [ ] Framework integrations - LangChain, HuggingFace, AutoGen

### v0.4.0 (Future)
- [ ] Constitutional AI module - Democratic constitution design
- [ ] MARL integration - Democratic reward aggregation
- [ ] Federated learning - Democratic model aggregation
- [ ] Advanced metrics - Shapley values, causal analysis

## 📄 License

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

## 🙏 Acknowledgments

AgorAI builds on decades of research in social choice theory, welfare economics, and game theory. We're grateful to the researchers who developed these foundational methods:

- Kenneth Arrow (Social Choice Theory)
- Anthony Atkinson (Inequality Measurement)
- Markus Schulze (Condorcet Methods)
- And many others in the fields of voting theory, mechanism design, and multi-agent systems

**Inspired by:**
- Anthropic's Constitutional AI and Collective Constitutional AI
- Stuart Russell's work on social choice for AI alignment
- Recent advances in multi-agent systems and MARL

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/agorai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agorai/discussions)
- **Email**: samuel.schlenker@example.com
- **Twitter**: [@yourusername](https://twitter.com/yourusername)

---

**Built with ❤️ for the democratic AI research community**