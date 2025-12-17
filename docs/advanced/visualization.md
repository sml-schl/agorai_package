# Visualization Guide

The `agorai.visualization` module provides tools for creating publication-quality plots and natural language explanations of aggregation decisions.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Plotting Functions](#plotting-functions)
4. [Natural Language Explanations](#natural-language-explanations)
5. [Examples](#examples)
6. [Customization](#customization)
7. [Best Practices](#best-practices)

---

## Overview

The visualization module provides two main capabilities:

1. **Plotting** - Generate publication-quality figures
   - Utility matrix heatmaps
   - Method comparison charts
   - Fairness-efficiency tradeoff plots

2. **Explanations** - Natural language descriptions
   - Why a candidate won
   - How aggregation methods work
   - When to use each method

**Requirements:**
- matplotlib (optional, for plotting): `pip install matplotlib`

---

## Quick Start

### Plotting

```python
from agorai.visualization import plot_utility_matrix, plot_aggregation_comparison

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

# Plot utility matrix
plot_utility_matrix(
    utilities,
    agent_labels=["Agent 1", "Agent 2", "Agent 3"],
    candidate_labels=["Option A", "Option B"],
    save_path="utilities.png"
)

# Compare aggregation methods
plot_aggregation_comparison(
    utilities,
    methods=["majority", "atkinson", "maximin"],
    save_path="comparison.png"
)
```

### Explanations

```python
from agorai.visualization import explain_decision, explain_method
from agorai.aggregate import aggregate

# Get aggregation result
result = aggregate(utilities, method="atkinson", epsilon=1.0)

# Explain the decision
explanation = explain_decision(
    utilities, "atkinson", result['winner'], result['scores'], epsilon=1.0
)
print(explanation)

# Explain the method
guide = explain_method("atkinson")
print(guide)
```

---

## Plotting Functions

### plot_utility_matrix()

Create a heatmap visualization of the utility matrix.

**Parameters:**
- `utilities` (List[List[float]]): Utility matrix (n_agents × n_candidates)
- `agent_labels` (Optional[List[str]]): Labels for agents (default: "Agent 1", "Agent 2", ...)
- `candidate_labels` (Optional[List[str]]): Labels for candidates (default: "Candidate 0", "Candidate 1", ...)
- `save_path` (Optional[str]): Path to save plot (if None, displays interactively)

**Example:**

```python
from agorai.visualization import plot_utility_matrix

utilities = [
    [0.8, 0.2, 0.5],  # Agent 1
    [0.3, 0.7, 0.4],  # Agent 2
    [0.6, 0.5, 0.9],  # Agent 3
]

plot_utility_matrix(
    utilities,
    agent_labels=["Western Perspective", "Eastern Perspective", "Global South"],
    candidate_labels=["Approve", "Reject", "Neutral"],
    save_path="utility_heatmap.png"
)
```

**Output:**
- Color-coded heatmap (red=low, yellow=medium, green=high)
- Values displayed in each cell
- Suitable for papers and presentations

---

### plot_aggregation_comparison()

Compare multiple aggregation methods side-by-side.

**Parameters:**
- `utilities` (List[List[float]]): Utility matrix
- `methods` (List[str]): List of aggregation methods to compare
- `highlight_differences` (bool): Highlight if methods choose different winners (default: True)
- `save_path` (Optional[str]): Path to save plot
- `**method_params`: Parameters passed to all methods

**Example:**

```python
from agorai.visualization import plot_aggregation_comparison

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

plot_aggregation_comparison(
    utilities,
    methods=["majority", "atkinson", "maximin", "nash_bargaining"],
    highlight_differences=True,
    save_path="method_comparison.png"
)
```

**Output:**
- Bar charts for each method
- Winner highlighted in green
- Alert if methods choose different winners
- Useful for demonstrating method differences

---

### plot_fairness_tradeoffs()

Visualize fairness-efficiency tradeoffs across methods.

**Parameters:**
- `utilities` (List[List[float]]): Utility matrix
- `methods` (List[str]): List of methods to compare
- `x_axis` (str): Metric for x-axis (default: "social_welfare")
  - Options: "social_welfare", "utilitarian_welfare"
- `y_axis` (str): Metric for y-axis (default: "gini_coefficient")
  - Options: "gini_coefficient", "atkinson_index"
- `pareto_frontier` (bool): Draw Pareto frontier (default: False)
- `save_path` (Optional[str]): Path to save plot

**Example:**

```python
from agorai.visualization import plot_fairness_tradeoffs

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

plot_fairness_tradeoffs(
    utilities,
    methods=["majority", "borda", "atkinson", "maximin", "nash_bargaining"],
    x_axis="social_welfare",
    y_axis="gini_coefficient",
    save_path="tradeoffs.png"
)
```

**Output:**
- Scatter plot with methods positioned by fairness vs efficiency
- Ideal methods are in bottom-right (low Gini, high welfare)
- Shows tradeoff space clearly

---

## Natural Language Explanations

### explain_decision()

Generate natural language explanation for why a candidate won.

**Parameters:**
- `utilities` (List[List[float]]): Utility matrix
- `method` (str): Aggregation method used
- `winner` (int): Winning candidate index
- `scores` (List[float]): Aggregated scores
- `**method_params`: Parameters used

**Returns:**
- Markdown-formatted string explaining the decision

**Supported Methods:**
- `majority` - Plurality voting explanation
- `atkinson` - Inequality aversion and EDE
- `maximin` - Rawlsian fairness explanation
- `borda` - Ranking-based explanation
- `score_centroid` - Averaging explanation
- `nash_bargaining` - Game-theoretic explanation
- Generic fallback for other methods

**Example:**

```python
from agorai.visualization import explain_decision
from agorai.aggregate import aggregate

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

# Get result
result = aggregate(utilities, method="atkinson", epsilon=1.0)

# Explain
explanation = explain_decision(
    utilities,
    method="atkinson",
    winner=result['winner'],
    scores=result['scores'],
    epsilon=1.0
)

print(explanation)
```

**Output:**
```markdown
Candidate 0 won using **Atkinson aggregation** with ε=1.0 (geometric mean).

**How it works:** Atkinson method computes the "equally-distributed equivalent"
(EDE) utility for each candidate. The EDE represents the utility level that, if
given equally to all agents, would provide the same total welfare. Higher EDE =
more fair distribution.

**Inequality aversion (ε):**
- ε=0: No aversion (utilitarian - just maximize total)
- ε=1: Moderate aversion (geometric mean - balance equality and efficiency)
- ε>1: Strong aversion (heavily penalize inequality)

**This result:** Candidate 0 has the highest EDE score (0.520). This means its
utility distribution across agents is most equally distributed (given ε=1.0).

**Comparison:**
- Candidate 0: EDE = 0.520
- Candidate 1: EDE = 0.458
```

---

### explain_method()

Explain what an aggregation method does and when to use it.

**Parameters:**
- `method` (str): Aggregation method name

**Returns:**
- Markdown-formatted string with:
  - Description
  - Properties (axioms satisfied)
  - Parameters (if applicable)
  - When to use
  - When NOT to use

**Supported Methods:**
- `majority`, `atkinson`, `maximin`, `borda`, and more

**Example:**

```python
from agorai.visualization import explain_method

# Get method guide
guide = explain_method("maximin")
print(guide)
```

**Output:**
```markdown
**Maximin (Rawlsian) Aggregation**

**Description:** Choose candidate that maximizes the minimum utility
(helps worst-off agent most).

**Properties:**
- ✓ Strongly egalitarian
- ✓ Protects minorities
- ✗ Can be Pareto suboptimal
- ✓ Clear fairness interpretation

**Philosophy:**
Based on John Rawls' "veil of ignorance" - judge by worst outcome.

**When to use:**
- Fairness is paramount
- Minority protection required
- Bias mitigation (ensure no group harmed)
- High-stakes decisions

**When NOT to use:**
- Efficiency important
- Outlier agents present
- Need to balance multiple objectives
```

---

## Examples

### Example 1: Paper Figure

Create a figure for your research paper:

```python
from agorai.visualization import plot_utility_matrix, plot_aggregation_comparison
import matplotlib.pyplot as plt

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

# Create 2-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Panel 1: Utility matrix
plot_utility_matrix(utilities, save_path="temp_utils.png")

# Panel 2: Method comparison
plot_aggregation_comparison(
    utilities,
    methods=["majority", "atkinson", "maximin"],
    save_path="temp_comparison.png"
)

# Combine and save
fig.savefig("paper_figure.png", dpi=300, bbox_inches='tight')
```

### Example 2: Interactive Explanation

Create an interactive tool that explains decisions:

```python
from agorai.visualization import explain_decision
from agorai.aggregate import aggregate, list_methods

def explain_all_methods(utilities):
    """Explain how each method would decide."""
    methods = list_methods()

    for method in methods:
        try:
            result = aggregate(utilities, method=method)
            explanation = explain_decision(
                utilities, method, result['winner'], result['scores']
            )

            print(f"\n{'='*60}")
            print(f"METHOD: {method.upper()}")
            print('='*60)
            print(explanation)

        except Exception as e:
            print(f"✗ {method}: {e}")

# Use it
utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
explain_all_methods(utilities)
```

### Example 3: Fairness-Efficiency Analysis

Visualize the fairness-efficiency tradeoff:

```python
from agorai.visualization import plot_fairness_tradeoffs
from agorai.benchmarks import compare_methods

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

# Get detailed comparison
comparison = compare_methods(
    methods=["majority", "borda", "atkinson", "maximin", "nash_bargaining"],
    benchmark={"name": "custom", "items": [{"utilities": utilities}]}
)

# Plot tradeoffs
plot_fairness_tradeoffs(
    utilities,
    methods=["majority", "borda", "atkinson", "maximin", "nash_bargaining"],
    save_path="fairness_efficiency.png"
)

# Print analysis
print("\nFairness-Efficiency Analysis:")
for method_result in comparison['methods']:
    name = method_result['method']
    gini = method_result['summary']['fairness']['gini_coefficient']
    welfare = method_result['summary']['efficiency']['social_welfare']

    print(f"{name:20s} Gini={gini:.3f} (fairness) Welfare={welfare:.2f} (efficiency)")
```

---

## Customization

### Custom Colors

```python
from agorai.visualization import plot_utility_matrix
import matplotlib.pyplot as plt
import numpy as np

utilities = [[0.8, 0.2], [0.3, 0.7]]

# Custom colormap
plt.figure(figsize=(8, 6))
plt.imshow(utilities, cmap='viridis', aspect='auto')  # Use viridis instead of RdYlGn
plt.colorbar(label="Utility")
plt.title("Custom Color Scheme")
plt.savefig("custom_colors.png")
```

### Custom Labels and Annotations

```python
from agorai.visualization import plot_aggregation_comparison

utilities = [[0.8, 0.2], [0.3, 0.7]]

plot_aggregation_comparison(
    utilities,
    methods=["atkinson", "maximin"],
    save_path="annotated_comparison.png"
)

# Add custom annotations
import matplotlib.pyplot as plt
fig = plt.gcf()
ax = fig.axes[0]
ax.text(0.5, 0.95, "Fairness-Focused Methods",
        transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
plt.savefig("annotated_comparison.png")
```

---

## Best Practices

### 1. Always Save Plots for Papers

```python
# Save with high DPI for publications
plot_utility_matrix(
    utilities,
    save_path="figure1.png"
)

# Then manually increase DPI if needed
import matplotlib.pyplot as plt
plt.savefig("figure1_hires.png", dpi=300, bbox_inches='tight')
```

### 2. Use Explanations in Documentation

```python
from agorai.visualization import explain_method

# Generate method documentation
methods = ["majority", "atkinson", "maximin", "borda"]

with open("method_guide.md", "w") as f:
    f.write("# Aggregation Method Guide\n\n")

    for method in methods:
        explanation = explain_method(method)
        f.write(f"## {method.title()}\n\n")
        f.write(explanation)
        f.write("\n\n---\n\n")
```

### 3. Create Comparative Visualizations

```python
from agorai.visualization import plot_fairness_tradeoffs

# Compare fairness-efficiency for different epsilon values
utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

methods = [
    {"name": "atkinson", "params": {"epsilon": 0.5}},
    {"name": "atkinson", "params": {"epsilon": 1.0}},
    {"name": "atkinson", "params": {"epsilon": 1.5}},
    {"name": "atkinson", "params": {"epsilon": 2.0}},
]

# Show how epsilon affects fairness-efficiency tradeoff
plot_fairness_tradeoffs(
    utilities,
    methods=[f"atkinson_eps{m['params']['epsilon']}" for m in methods],
    save_path="epsilon_sweep.png"
)
```

---

## API Reference

### Plotting Functions

**`plot_utility_matrix(utilities, agent_labels, candidate_labels, save_path)`**
- Creates heatmap of utility matrix
- Returns: None (saves or displays plot)

**`plot_aggregation_comparison(utilities, methods, highlight_differences, save_path, **params)`**
- Compares methods side-by-side
- Returns: None (saves or displays plot)

**`plot_fairness_tradeoffs(utilities, methods, x_axis, y_axis, pareto_frontier, save_path)`**
- Plots fairness vs efficiency tradeoffs
- Returns: None (saves or displays plot)

### Explanation Functions

**`explain_decision(utilities, method, winner, scores, **params)`**
- Explains why candidate won
- Returns: Markdown string

**`explain_method(method)`**
- Explains method properties and usage
- Returns: Markdown string

---

## Troubleshooting

### Matplotlib Not Found

```python
try:
    from agorai.visualization import plot_utility_matrix
    plot_utility_matrix([[0.8, 0.2]], save_path="test.png")
except ImportError:
    print("Install matplotlib: pip install matplotlib")
```

### Plots Not Displaying

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from agorai.visualization import plot_utility_matrix
plot_utility_matrix([[0.8, 0.2]], save_path="test.png")  # Must save, won't display
```

### Explanations Too Verbose

```python
from agorai.visualization import explain_decision

explanation = explain_decision(utilities, method, winner, scores, **params)

# Get just the first paragraph
short_explanation = explanation.split('\n\n')[0]
print(short_explanation)
```

---

**For more examples, see the `examples/` directory and the [AgorAI Research Strategy Report](../AgorAI_Research_Strategy_Report.md).**
