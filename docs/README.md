# AgorAI Documentation

Complete documentation for the AgorAI framework - democratic multi-agent opinion aggregation.

## 📚 Documentation Structure

### 🎯 Getting Started
- **[Main README](../README.md)** - Start here for overview and quick start
- **[Examples](../examples/)** - Code examples and use cases

### 🔧 Core Functionality
- **[Aggregation Methods](core/aggregation.md)** - Complete reference for all 14+ aggregation mechanisms
- **[Mechanism Aliases](core/aliases.md)** - Intuitive names for aggregation methods
- **[Property Analysis](core/properties.md)** - Theoretical guarantees and mechanism selection

### 🛡️ Applications
- **[Bias Mitigation](applications/bias_mitigation.md)** - Multi-perspective bias detection and mitigation
- **[Automatic Council Creation](applications/automatic_council.md)** - LLM-based perspective generation
- **[Counterfactual Testing](applications/counterfactual_testing.md)** - Causal robustness evaluation

### 🚀 Advanced Topics
- **[Queue Processing](advanced/queue_processing.md)** - Batch processing and production workflows
- **[Visualization](advanced/visualization.md)** - Plots and explanations
- **[Extending AgorAI](advanced/extending.md)** - Custom methods, providers, and integrations

### 📖 Reference
- **[API Reference](reference/api.md)** - Complete function signatures
- **[Configuration](reference/configuration.md)** - Settings and environment variables

---

## 🚀 Installation

```bash
pip install agorai[all]
```

See [Main README](../README.md#installation) for details.

---

## 🎓 Quick Start

### Use Case 1: Aggregate Opinions from Multiple Agents

```python
from agorai.aggregate import aggregate

utilities = [
    [0.8, 0.2, 0.5],  # Agent 1's utilities
    [0.3, 0.7, 0.4],  # Agent 2's utilities
    [0.6, 0.5, 0.9],  # Agent 3's utilities
]

result = aggregate(utilities, method="fair")  # Uses Atkinson mechanism
print(f"Winner: Candidate {result['winner']}")
```

→ See [Aggregation Methods](core/aggregation.md) for all 14+ methods

### Use Case 2: Mitigate Bias Through Multi-Perspective Analysis

```python
from agorai.bias import mitigate_bias

result = mitigate_bias(
    input_text="Is this job posting discriminatory?",
    aggregation_method="fair",
    num_perspectives=5
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")
```

→ See [Bias Mitigation](applications/bias_mitigation.md) for complete guide

---

## 🗺️ Quick Navigation by Use Case

| I want to... | Go to... |
|--------------|----------|
| **Aggregate opinions from multiple agents** | [Aggregation Methods](core/aggregation.md) |
| **Use intuitive method names like "fair" or "robust"** | [Mechanism Aliases](core/aliases.md) |
| **Detect and mitigate AI bias** | [Bias Mitigation](applications/bias_mitigation.md) |
| **Generate diverse perspectives automatically** | [Automatic Council Creation](applications/automatic_council.md) |
| **Test for spurious correlations** | [Counterfactual Testing](applications/counterfactual_testing.md) |
| **Choose the right aggregation method** | [Property Analysis](core/properties.md) |
| **Process large batches of decisions** | [Queue Processing](advanced/queue_processing.md) |
| **Visualize aggregation results** | [Visualization](advanced/visualization.md) |
| **Add custom aggregation methods** | [Extending AgorAI](advanced/extending.md) |
| **Configure LLM providers** | [Configuration](reference/configuration.md) |

---

## 👥 Quick Navigation by Role

### For Researchers
- [Aggregation Methods](core/aggregation.md) - Theory and implementation
- [Property Analysis](core/properties.md) - Theoretical guarantees
- [Counterfactual Testing](applications/counterfactual_testing.md) - Causal evaluation

### For ML Engineers
- [Bias Mitigation](applications/bias_mitigation.md) - Practical bias detection
- [Queue Processing](advanced/queue_processing.md) - Production workflows
- [API Reference](reference/api.md) - Complete function signatures

### For Social Scientists
- [Mechanism Aliases](core/aliases.md) - Accessible method selection
- [Automatic Council Creation](applications/automatic_council.md) - Perspective modeling
- [Visualization](advanced/visualization.md) - Result interpretation

---

## 📦 Core Modules

### Aggregation
The heart of AgorAI - multiple methods for combining agent utilities:
- **Social Choice Theory**: Majority, Borda, Schulze, Approval
- **Welfare Economics**: Maximin, Atkinson
- **Machine Learning**: Score Centroid, Robust Median
- **Game Theory**: Nash Bargaining, Quadratic Voting

→ Complete reference: [Aggregation Methods](core/aggregation.md)

### Aliases
Intuitive method selection without deep theoretical knowledge:
- `fair` → Atkinson (balances efficiency and equality)
- `minority-focused` → Maximin (protects worst-off)
- `robust` → Robust Median (outlier-resistant)
- `democratic` → Majority (simple plurality)

→ Complete guide: [Mechanism Aliases](core/aliases.md)

### Bias Mitigation
Multi-perspective analysis for bias detection:
- Automatic cultural perspective generation
- Multimodal support (text + image)
- Fairness metrics (demographic parity, equalized odds)
- Configurable aggregation strategies

→ Complete guide: [Bias Mitigation](applications/bias_mitigation.md)

### Counterfactual Testing
Causal robustness evaluation:
- Protected attribute manipulation
- Spurious correlation detection
- Edge case identification

→ Complete guide: [Counterfactual Testing](applications/counterfactual_testing.md)

---

## 📁 Documentation Folder Structure

```
docs/
├── README.md                                    # This file
├── core/
│   ├── aggregation.md                           # Complete method reference
│   ├── aliases.md                               # Intuitive method names
│   └── properties.md                            # Theoretical guarantees
├── applications/
│   ├── bias_mitigation.md                       # Bias detection guide
│   ├── automatic_council.md                     # Perspective generation
│   └── counterfactual_testing.md                # Causal evaluation
├── advanced/
│   ├── queue_processing.md                      # Batch workflows
│   ├── visualization.md                         # Plots and explanations
│   └── extending.md                             # Custom methods
└── reference/
    ├── api.md                                   # Complete API
    └── configuration.md                         # Settings
```

---

## ❓ FAQ

**Q: How do I choose the right aggregation method?**
A: Use [aliases](core/aliases.md) for intuitive selection, or consult [Property Analysis](core/properties.md) for theory-driven choice.

**Q: How do I detect bias in my AI system?**
A: See the [Bias Mitigation guide](applications/bias_mitigation.md) for multi-perspective analysis.

**Q: How do I add custom aggregation methods?**
A: See [Extending AgorAI](advanced/extending.md#custom-methods).

**Q: How do I process large batches of decisions?**
A: See [Queue Processing](advanced/queue_processing.md) for production workflows.

**Q: How do I configure LLM providers?**
A: See [Configuration](reference/configuration.md#llm-providers).

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📞 Support

- **GitHub Issues**: Report bugs and request features
- **Email**: Contact for research collaboration

---

## 📜 License

Custom Research and Non-Commercial License - Free for academic research, education, and personal use.

See [LICENSE](../LICENSE) for details.

---

**Happy aggregating! 🚀**
