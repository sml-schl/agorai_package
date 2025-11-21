# agorai.aggregate

Pure mathematical aggregation module for combining agent utilities into collective decisions.

## Overview

`agorai.aggregate` provides 14+ aggregation methods from social choice theory, welfare economics, and game theory. All methods operate on normalized utility matrices without any LLM integration.

## Installation

```bash
pip install agorai  # Minimal installation includes aggregate
```

## Quick Start

```python
from agorai.aggregate import aggregate

# Define utilities: utilities[i][j] = utility of agent i for candidate j
utilities = [
    [0.8, 0.2, 0.5],  # Agent 1's utilities for 3 candidates
    [0.3, 0.7, 0.4],  # Agent 2's utilities
    [0.6, 0.5, 0.9],  # Agent 3's utilities
]

# Aggregate using Atkinson method with inequality aversion
result = aggregate(utilities, method="atkinson", epsilon=1.0)

print(f"Winner: Candidate {result['winner']}")
print(f"Scores: {result['scores']}")
```

## API Reference

### aggregate()

```python
def aggregate(
    utilities: Union[List[List[float]], List[float]],
    method: str = "majority",
    weights: Optional[List[float]] = None,
    normalize: bool = True,
    **method_params
) -> Dict[str, Any]
```

Aggregate agent utilities into a collective decision.

**Parameters:**

- `utilities`: Utility matrix (n_agents × n_candidates) where `utilities[i][j]` is agent i's utility for candidate j
- `method`: Aggregation method name (see [Available Methods](#available-methods))
- `weights`: Optional per-agent weights (default: uniform)
- `normalize`: Whether to normalize utilities to [0, 1] per agent (default: True)
- `**method_params`: Method-specific parameters (e.g., `epsilon` for atkinson)

**Returns:**

Dictionary with:
- `winner`: Index of winning candidate (int)
- `scores`: Scores for all candidates (List[float])
- `method`: Name of aggregation method used (str)
- `metadata`: Additional method-specific information (Dict)

**Example:**

```python
# Majority voting
result = aggregate([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], method="majority")
# result['winner'] = 0

# Atkinson with strong inequality aversion
result = aggregate(
    [[0.9, 0.1], [0.6, 0.4], [0.7, 0.3]],
    method="atkinson",
    epsilon=2.0
)
```

### list_methods()

```python
def list_methods() -> List[Dict[str, str]]
```

List all available aggregation methods with descriptions.

**Returns:**

List of dictionaries containing:
- `name`: Method identifier
- `category`: Method category (Social Choice, Welfare Economics, etc.)
- `description`: Brief description

**Example:**

```python
from agorai.aggregate import list_methods

methods = list_methods()
for m in methods:
    print(f"{m['name']} ({m['category']}): {m['description']}")
```

## Available Methods

### Social Choice Theory

#### majority
One-agent-one-vote plurality (most votes wins).

- **Properties:** Strategy-proof, violates Condorcet criterion
- **Best for:** Simple binary decisions, democratic voting
- **Parameters:** None

```python
aggregate(utilities, method="majority")
```

#### weighted_plurality
Plurality with per-agent weights.

- **Properties:** Weighted voting
- **Best for:** Expertise-weighted decisions
- **Parameters:** `weights` (optional)

```python
aggregate(utilities, method="weighted_plurality", weights=[1.0, 2.0, 1.5])
```

#### borda
Borda count positional ranking aggregation.

- **Properties:** Monotonic, Pareto efficient, violates IIA
- **Best for:** Ranking aggregation, preference synthesis
- **Parameters:** None

```python
aggregate(utilities, method="borda")
```

#### schulze_condorcet
Schulze method for Condorcet-consistent ranking.

- **Properties:** Condorcet-consistent, clone-independent, monotonic
- **Best for:** Complex preference aggregation
- **Parameters:** None
- **Complexity:** O(n³)

```python
aggregate(utilities, method="schulze_condorcet")
```

#### approval_voting
Agents approve candidates above utility threshold.

- **Properties:** Strategy-proof, resistant to vote splitting
- **Best for:** Multi-winner selection, approval-based decisions
- **Parameters:** `threshold` (default: 0.5)

```python
aggregate(utilities, method="approval_voting", threshold=0.6)
```

#### supermajority
Requires threshold fraction of votes to win.

- **Properties:** Protects status quo, reduces close votes
- **Best for:** High-stakes decisions, constitutional changes
- **Parameters:** `threshold` (default: 0.66)

```python
aggregate(utilities, method="supermajority", threshold=0.75)
```

### Welfare Economics

#### maximin
Rawlsian maximin: maximize minimum utility across agents.

- **Properties:** Strongly egalitarian, protects worst-off agent
- **Best for:** Fairness-critical decisions, minority protection
- **Parameters:** None

```python
aggregate(utilities, method="maximin")
```

#### atkinson
Atkinson index with parameterizable inequality aversion.

- **Properties:** Monotonic, anonymous, continuous spectrum from utilitarian to maximin
- **Best for:** Tunable fairness-efficiency tradeoff
- **Parameters:** `epsilon` (default: 1.0)
  - `epsilon=0`: Utilitarian (sum of utilities)
  - `epsilon=1`: Geometric mean
  - `epsilon>1`: Increasing inequality aversion
  - `epsilon→∞`: Approaches maximin

```python
# Utilitarian
aggregate(utilities, method="atkinson", epsilon=0.0)

# Geometric mean (balanced)
aggregate(utilities, method="atkinson", epsilon=1.0)

# Strong inequality aversion
aggregate(utilities, method="atkinson", epsilon=2.0)
```

### Machine Learning / Statistics

#### score_centroid
Weighted average of normalized utilities.

- **Properties:** Simple, interpretable
- **Best for:** Ensemble aggregation, baseline comparisons
- **Parameters:** None

```python
aggregate(utilities, method="score_centroid")
```

#### robust_median
Per-candidate median utility (outlier-resistant).

- **Properties:** Robust to outliers, non-parametric
- **Best for:** Noisy agent outputs, outlier resistance
- **Parameters:** None

```python
aggregate(utilities, method="robust_median")
```

#### consensus
Balance mean utility and consensus (low disagreement).

- **Properties:** Detects agreement, measures coefficient of variation
- **Best for:** Detecting genuine consensus, confidence estimation
- **Parameters:** `cv_weight` (default: 0.5)

```python
aggregate(utilities, method="consensus", cv_weight=0.7)
```

### Game Theory

#### quadratic_voting
Budget-constrained voting with quadratic costs.

- **Properties:** Encourages intensity expression, resistant to strategic voting
- **Best for:** Intensity-aware voting, resource allocation
- **Parameters:** `budget` (default: 10.0)

```python
aggregate(utilities, method="quadratic_voting", budget=20.0)
```

#### nash_bargaining
Nash bargaining solution: maximize weighted product of utility gains.

- **Properties:** Pareto optimal, symmetric, scale-invariant
- **Best for:** Cooperative bargaining, compromise solutions
- **Parameters:** `disagreement` (default: "worst")
  - `"zero"`: No fallback (d_i = 0)
  - `"worst"`: Worst candidate for each agent
  - `"uniform"`: Equal split (d_i = 1/m)

```python
aggregate(utilities, method="nash_bargaining", disagreement="worst")
```

#### veto_hybrid
Veto-filtered centroid (minority protection).

- **Properties:** Combines deontological (veto) and utilitarian (centroid)
- **Best for:** Preventing unacceptable outcomes, minority protection
- **Parameters:** `veto_threshold` (default: 1)

```python
aggregate(utilities, method="veto_hybrid", veto_threshold=2)
```

## Notes

### Normalization

By default, utilities are normalized to [0, 1] per agent using min-max scaling. This ensures fairness across agents with different utility scales.

```python
# Disable normalization if utilities are already normalized
result = aggregate(utilities, method="majority", normalize=False)
```

### Weights

Most methods support optional per-agent weights for expertise-based or confidence-based aggregation:

```python
weights = [1.0, 2.0, 1.5]  # Agent 2 has twice the weight
result = aggregate(utilities, method="weighted_plurality", weights=weights)
```

### Input Formats

```python
# Standard format: List[List[float]]
utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

# Single agent format (converted automatically)
utilities = [0.8, 0.2]  # Treated as [[0.8, 0.2]]
```

## Examples

### Example 1: Fairness-Critical Decision

```python
from agorai.aggregate import aggregate

# 3 agents, 2 candidates
# Agent 1 strongly prefers candidate 0
# Agent 2 strongly prefers candidate 1
# Agent 3 is neutral
utilities = [
    [0.9, 0.1],
    [0.2, 0.8],
    [0.5, 0.5]
]

# Majority: Winner is tie-broken
result_majority = aggregate(utilities, method="majority")
print(f"Majority winner: {result_majority['winner']}")

# Maximin: Protects worst-off agent
result_maximin = aggregate(utilities, method="maximin")
print(f"Maximin winner: {result_maximin['winner']}")

# Atkinson: Balanced inequality aversion
result_atkinson = aggregate(utilities, method="atkinson", epsilon=1.0)
print(f"Atkinson winner: {result_atkinson['winner']}")
```

### Example 2: Ensemble Model Aggregation

```python
from agorai.aggregate import aggregate

# 5 model predictions for 3 classes
# Each model outputs confidence scores
model_confidences = [
    [0.7, 0.2, 0.1],  # Model 1 predicts class 0
    [0.6, 0.3, 0.1],  # Model 2 predicts class 0
    [0.1, 0.8, 0.1],  # Model 3 predicts class 1
    [0.5, 0.4, 0.1],  # Model 4 predicts class 0
    [0.4, 0.5, 0.1],  # Model 5 predicts class 1
]

# Simple average (centroid)
result = aggregate(model_confidences, method="score_centroid")
print(f"Ensemble prediction: Class {result['winner']}")
print(f"Class scores: {result['scores']}")
```

### Example 3: Stakeholder Preference Aggregation

```python
from agorai.aggregate import aggregate

# 4 stakeholder groups rating 3 policy options
stakeholder_preferences = [
    [0.9, 0.5, 0.2],  # Stakeholder 1
    [0.3, 0.8, 0.6],  # Stakeholder 2
    [0.7, 0.4, 0.9],  # Stakeholder 3
    [0.5, 0.6, 0.7],  # Stakeholder 4
]

# Schulze (Condorcet-consistent)
result_schulze = aggregate(stakeholder_preferences, method="schulze_condorcet")
print(f"Condorcet winner: Policy {result_schulze['winner']}")

# Borda (ranking-based)
result_borda = aggregate(stakeholder_preferences, method="borda")
print(f"Borda winner: Policy {result_borda['winner']}")
```

## References

1. Arrow, K. J. (1951). Social Choice and Individual Values.
2. Atkinson, A. B. (1970). On the measurement of inequality. Journal of Economic Theory.
3. Schulze, M. (2011). A new monotonic, clone-independent, reversal symmetric, and condorcet-consistent single-winner election method. Social Choice and Welfare.
4. Rawls, J. (1971). A Theory of Justice.
5. Nash, J. F. (1950). The bargaining problem. Econometrica.
