# Aggregation Methods

Complete reference for all aggregation methods in AgorAI.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Main Function](#main-function)
- [Available Methods](#available-methods)
  - [Social Choice Theory](#social-choice-theory)
  - [Welfare Economics](#welfare-economics)
  - [Machine Learning](#machine-learning)
  - [Game Theory](#game-theory)
- [Method Selection Guide](#method-selection-guide)
- [Examples](#examples)

---

## Basic Usage

```python
from agorai.aggregate import aggregate

utilities = [
    [0.8, 0.2, 0.5],  # Agent 1's utilities for 3 candidates
    [0.3, 0.7, 0.4],  # Agent 2's utilities
    [0.6, 0.5, 0.9],  # Agent 3's utilities
]

result = aggregate(utilities, method="fair")
print(result)
# {
#     'winner': 2,
#     'scores': [0.54, 0.42, 0.58],
#     'method': 'atkinson',
#     'metadata': {...}
# }
```

---

## Main Function

### `aggregate()`

Aggregate utilities from multiple agents using a specified method.

**Signature:**
```python
aggregate(
    utilities: List[List[float]],
    method: str = "majority",
    weights: Optional[List[float]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `utilities` | `List[List[float]]` | *required* | 2D list where `utilities[i][j]` is agent `i`'s utility for candidate `j`. Shape: `(n_agents, n_candidates)` |
| `method` | `str` | `"majority"` | Aggregation method to use. Can be technical name or alias (see [aliases](aliases.md)) |
| `weights` | `List[float]` | `None` | Optional weights for agents. If `None`, all agents weighted equally. Must sum to 1.0 or will be normalized |
| `**kwargs` | Various | - | Method-specific parameters (see below) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `winner` | `int` | Index of winning candidate (0-indexed) |
| `scores` | `List[float]` | Final scores for each candidate |
| `method` | `str` | Resolved method name (technical name, even if alias was used) |
| `metadata` | `Dict` | Method-specific metadata (voting details, intermediate results, etc.) |

**Raises:**

| Exception | When |
|-----------|------|
| `ValueError` | Invalid utilities shape, invalid method name, invalid parameters |
| `TypeError` | Wrong parameter types |

---

## Available Methods

### Social Choice Theory

Methods from voting theory and social choice.

#### `majority`

**Alias:** `democratic`

Simple plurality voting - candidate with most votes wins.

**Parameters:**
```python
aggregate(utilities, method="majority")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *(no additional parameters)* | - | - | - |

**How it works:**
1. Each agent votes for their highest-utility candidate
2. Candidate with most votes wins
3. Ties broken by total utility sum

**Properties:**
- ✅ Anonymity: All agents treated equally
- ✅ Neutrality: All candidates treated equally
- ✅ Monotonicity: Improving ranking can't hurt
- ❌ Minority protection: Winner-takes-all
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]
result = aggregate(utilities, method="majority")
# Winner: 0 (2 votes for candidate 0, 1 vote for candidate 1)
```

---

#### `weighted_plurality`

Weighted voting where agent weights matter.

**Parameters:**
```python
aggregate(utilities, method="weighted_plurality", weights=[0.5, 0.3, 0.2])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `List[float]` | `None` | Agent weights (required for this method) |

**How it works:**
1. Each agent's vote weighted by their weight parameter
2. Candidate with highest weighted vote count wins

**Properties:**
- ⚠️ Anonymity: Violated (agents have different weights)
- ✅ Neutrality: All candidates treated equally
- ✅ Monotonicity

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]
result = aggregate(utilities, method="weighted_plurality", weights=[0.2, 0.3, 0.5])
# Winner: 1 (weighted votes: 0=0.5, 1=0.5, tie broken by utility sum)
```

---

#### `borda`

**Alias:** `ranked-choice`

Borda count - considers full preference ranking.

**Parameters:**
```python
aggregate(utilities, method="borda", normalized=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalized` | `bool` | `True` | Whether to normalize Borda points |

**How it works:**
1. Each agent ranks candidates by utility
2. Points awarded: highest rank gets `m-1` points, second gets `m-2`, etc.
3. Candidate with most total points wins

**Properties:**
- ✅ Anonymity
- ✅ Neutrality
- ✅ Considers full rankings (not just top choice)
- ❌ Condorcet: May not select Condorcet winner
- Computational complexity: O(n×m log m)

**Example:**
```python
utilities = [[0.9, 0.7, 0.3], [0.8, 0.6, 0.4], [0.5, 0.9, 0.6]]
result = aggregate(utilities, method="borda")
# Ranks each agent's preferences and aggregates points
```

---

#### `schulze_condorcet`

**Alias:** `condorcet`

Schulze method - Condorcet-consistent ranking with cycle resolution.

**Parameters:**
```python
aggregate(utilities, method="schulze_condorcet")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *(no additional parameters)* | - | - | - |

**How it works:**
1. Constructs pairwise preference matrix
2. Computes strongest paths between all candidate pairs
3. Selects candidate with strongest path strengths

**Properties:**
- ✅ Condorcet consistency: Always selects Condorcet winner if one exists
- ✅ Clone independence
- ✅ Monotonicity
- ✅ Anonymity
- Computational complexity: O(m³)

**Example:**
```python
utilities = [[0.9, 0.5, 0.3], [0.6, 0.8, 0.4], [0.4, 0.7, 0.9]]
result = aggregate(utilities, method="schulze_condorcet")
# Metadata includes pairwise comparisons and path strengths
```

---

#### `approval_voting`

**Alias:** `strategy-proof`

Approval voting - agents approve multiple candidates.

**Parameters:**
```python
aggregate(utilities, method="approval_voting", threshold=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | Utility threshold for approval. Range: [0.0, 1.0] |

**How it works:**
1. Each agent approves all candidates with utility ≥ threshold
2. Candidate with most approvals wins

**Properties:**
- ✅ Strategy-proof under certain conditions
- ✅ Anonymity
- ✅ Neutrality
- ✅ Resistant to tactical voting
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.6, 0.3], [0.8, 0.7, 0.4], [0.5, 0.9, 0.4]]
result = aggregate(utilities, method="approval_voting", threshold=0.6)
# Candidates with utility ≥ 0.6 get approved
```

---

#### `supermajority`

Requires supermajority threshold to win.

**Parameters:**
```python
aggregate(utilities, method="supermajority", threshold=0.67)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.67` | Fraction of votes required to win. Range: [0.5, 1.0] |

**How it works:**
1. Each agent votes for highest-utility candidate
2. Candidate wins only if they get ≥ threshold fraction of votes
3. If no candidate meets threshold, highest vote-getter wins

**Properties:**
- ✅ Minority protection (requires broad consensus)
- ✅ Anonymity
- ⚠️ May not have winner if votes split

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
result = aggregate(utilities, method="supermajority", threshold=0.67)
# Candidate 0 wins with 3/3 = 100% > 67%
```

---

### Welfare Economics

Methods from welfare economics and inequality theory.

#### `maximin`

**Alias:** `minority-focused`

Rawlsian fairness - maximize the minimum utility.

**Parameters:**
```python
aggregate(utilities, method="maximin")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *(no additional parameters)* | - | - | - |

**How it works:**
1. For each candidate, find the minimum utility across all agents
2. Select candidate with highest minimum utility

**Properties:**
- ✅ Maximum minority protection
- ✅ Worst-off protection (Rawlsian justice)
- ✅ Outlier resistance
- ❌ Pareto efficiency: May select Pareto-dominated option
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.1, 0.5], [0.8, 0.2, 0.6], [0.3, 0.9, 0.4]]
result = aggregate(utilities, method="maximin")
# Winner: 2 (min utilities: [0.3, 0.1, 0.4] → max is 0.4)
```

---

#### `atkinson`

**Alias:** `fair`

Atkinson index - parameterizable inequality aversion.

**Parameters:**
```python
aggregate(utilities, method="atkinson", epsilon=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | `float` | `1.0` | Inequality aversion parameter. Range: [0.0, ∞) |

**Epsilon values:**
- `epsilon = 0`: No inequality aversion (equivalent to utilitarian sum)
- `epsilon = 1`: Geometric mean (balanced fairness)
- `epsilon > 1`: Strong inequality aversion (approaches maximin)
- `epsilon → ∞`: Equivalent to maximin

**How it works:**
1. Computes equally-distributed equivalent utility for each candidate
2. Formula: `EDE = (mean(u_i^(1-ε)))^(1/(1-ε))` for ε≠1
3. For ε=1: `EDE = exp(mean(log(u_i)))`  (geometric mean)

**Properties:**
- ✅ Tunable fairness-efficiency trade-off
- ✅ Minority protection (for ε > 0)
- ✅ Continuous spectrum from utilitarian to maximin
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]

# Balanced fairness
result = aggregate(utilities, method="atkinson", epsilon=1.0)

# Strong fairness (closer to maximin)
result = aggregate(utilities, method="atkinson", epsilon=2.0)

# Weak fairness (closer to utilitarian)
result = aggregate(utilities, method="atkinson", epsilon=0.5)
```

---

### Machine Learning

Methods from machine learning and statistics.

#### `score_centroid`

**Alias:** `efficient`

Weighted average - utilitarian sum.

**Parameters:**
```python
aggregate(utilities, method="score_centroid")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *(uses `weights` from main function)* | - | - | If provided, computes weighted average |

**How it works:**
1. Computes mean utility for each candidate across all agents
2. If weights provided, computes weighted mean
3. Candidate with highest mean utility wins

**Properties:**
- ✅ Pareto efficiency
- ✅ Continuous scores (not just rankings)
- ✅ Simple and interpretable
- ❌ No minority protection
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]
result = aggregate(utilities, method="score_centroid")
# Candidate 0: (0.9+0.8+0.3)/3 = 0.67
# Candidate 1: (0.1+0.2+0.7)/3 = 0.33
# Winner: 0
```

---

#### `robust_median`

**Alias:** `robust`

Median aggregation - highly outlier-resistant.

**Parameters:**
```python
aggregate(utilities, method="robust_median")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *(no additional parameters)* | - | - | - |

**How it works:**
1. For each candidate, computes median utility across all agents
2. Candidate with highest median utility wins

**Properties:**
- ✅ High outlier resistance (50% breakdown point)
- ✅ Robust to adversarial agents
- ✅ Simple and fast
- ⚠️ Loses information (ignores non-median values)
- Computational complexity: O(n×m log n)

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.05, 0.95], [0.7, 0.3]]
# With outlier:   agent 2 gives very different utilities
result = aggregate(utilities, method="robust_median")
# Uses median, so outlier (agent 2) has limited impact
```

---

#### `consensus`

**Alias:** `consensus-seeking`

Minimizes disagreement among agents.

**Parameters:**
```python
aggregate(utilities, method="consensus", metric="variance")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | `"variance"` | Disagreement metric. Options: `"variance"`, `"range"`, `"std"` |

**How it works:**
1. For each candidate, computes disagreement metric across agent utilities
2. Selects candidate with minimum disagreement
3. Breaks ties by highest mean utility

**Metrics:**
- `"variance"`: Variance of utilities (penalizes both high and low outliers)
- `"range"`: Max utility - min utility (simple spread)
- `"std"`: Standard deviation (similar to variance but different scale)

**Properties:**
- ✅ Promotes consensus
- ✅ Reduces polarization
- ⚠️ May select mediocre option everyone barely accepts
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.1, 0.5], [0.8, 0.2, 0.5], [0.7, 0.3, 0.5]]
result = aggregate(utilities, method="consensus")
# Candidate 2 has variance 0 (perfect agreement)
```

---

### Game Theory

Methods from game theory and mechanism design.

#### `quadratic_voting`

Intensity-aware voting with budget constraints.

**Parameters:**
```python
aggregate(utilities, method="quadratic_voting", budget=100)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget` | `float` | `100` | Vote budget per agent. Range: (0, ∞) |

**How it works:**
1. Each agent has a budget to allocate across candidates
2. Cost to cast `k` votes for a candidate is `k²`
3. Agents allocate budget to maximize utility
4. Candidate with most votes wins

**Properties:**
- ✅ Reveals preference intensity
- ✅ Efficient with strategic voters
- ✅ Prevents minority tyranny
- ⚠️ Assumes rational budget allocation
- Computational complexity: O(n×m²)

**Example:**
```python
utilities = [[0.9, 0.6, 0.3], [0.5, 0.8, 0.4]]
result = aggregate(utilities, method="quadratic_voting", budget=100)
# Agents allocate budget proportional to utility differences
```

---

#### `nash_bargaining`

Nash bargaining solution - cooperative game theory.

**Parameters:**
```python
aggregate(utilities, method="nash_bargaining", fallback=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fallback` | `float` or `List[float]` | `0.0` | Disagreement point utility. Can be single value or per-agent |

**How it works:**
1. Maximizes product of utilities above fallback point
2. Formula: `max_c ∏_i (u_ic - fallback_i)`
3. Represents cooperative bargaining outcome

**Properties:**
- ✅ Pareto efficiency
- ✅ Symmetry
- ✅ Independence of irrelevant alternatives
- ✅ Scale invariance
- Computational complexity: O(n×m)

**Example:**
```python
utilities = [[0.9, 0.6], [0.8, 0.7]]
result = aggregate(utilities, method="nash_bargaining", fallback=0.3)
# Maximizes (0.9-0.3)*(0.8-0.3) = 0.30 for candidate 0
#           (0.6-0.3)*(0.7-0.3) = 0.12 for candidate 1
# Winner: 0
```

---

#### `veto_hybrid`

**Alias:** `veto-protected`

Combines majority rule with minority veto power.

**Parameters:**
```python
aggregate(utilities, method="veto_hybrid", veto_threshold=0.2)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `veto_threshold` | `float` | `0.2` | Utility below which veto applies. Range: [0.0, 1.0] |

**How it works:**
1. Runs majority voting to find winner
2. Checks if any agent has utility < veto_threshold for winner
3. If veto triggered, eliminates winner and repeats
4. Continues until no-veto winner found or all eliminated

**Properties:**
- ✅ Strong minority protection (veto power)
- ✅ Democratic (uses majority when no veto)
- ⚠️ May have no winner if all options vetoed
- Computational complexity: O(n×m²) worst case

**Example:**
```python
utilities = [[0.9, 0.1], [0.8, 0.2], [0.15, 0.9]]
result = aggregate(utilities, method="veto_hybrid", veto_threshold=0.2)
# Candidate 0 wins majority but agent 2 has utility 0.15 < 0.2
# Veto triggered → Candidate 1 wins
```

---

## Method Selection Guide

### By Use Case

| Use Case | Recommended Method | Alias | Why |
|----------|-------------------|-------|-----|
| **Fairness & Equality** | Atkinson | `fair` | Balances efficiency with inequality aversion |
| **Protect Minorities** | Maximin | `minority-focused` | Maximizes worst-off utility |
| **Democratic Legitimacy** | Majority | `democratic` | Simple, transparent plurality |
| **Resist Outliers** | Robust Median | `robust` | Immune to extreme opinions |
| **Prevent Manipulation** | Approval Voting | `strategy-proof` | Hard to game strategically |
| **Build Consensus** | Consensus | `consensus-seeking` | Minimizes disagreement |
| **Ranked Preferences** | Schulze | `condorcet` | Respects pairwise preferences |
| **High Stakes Safety** | Veto Hybrid | `veto-protected` | Minority can block harmful options |

### By Properties

| Property | Methods |
|----------|---------|
| **Anonymity** (all agents equal) | Majority, Borda, Schulze, Approval, Maximin, Atkinson, Score Centroid, Robust Median |
| **Pareto Efficiency** | Score Centroid, Nash Bargaining |
| **Minority Protection** | Maximin, Atkinson (ε>0), Veto Hybrid |
| **Outlier Resistance** | Robust Median, Maximin, Atkinson (ε>0) |
| **Condorcet Consistency** | Schulze |
| **Strategy-Proofness** | Approval Voting (under certain conditions) |

### By Computational Complexity

| Complexity | Methods |
|------------|---------|
| **O(n×m)** | Majority, Weighted Plurality, Maximin, Atkinson, Score Centroid, Consensus, Veto Hybrid |
| **O(n×m log n)** | Robust Median |
| **O(n×m log m)** | Borda |
| **O(m³)** | Schulze |

---

## Examples

### Example 1: Comparing Methods

```python
from agorai.aggregate import aggregate

utilities = [
    [0.9, 0.3, 0.5],
    [0.8, 0.4, 0.6],
    [0.2, 0.9, 0.4],
]

methods = ["majority", "maximin", "atkinson", "robust_median"]

for method in methods:
    result = aggregate(utilities, method=method)
    print(f"{method}: Winner = {result['winner']}, Scores = {result['scores']}")

# Output:
# majority: Winner = 0, Scores = [1.0, 0.0, 0.0]
# maximin: Winner = 2, Scores = [0.2, 0.3, 0.4]
# atkinson: Winner = 0, Scores = [0.59, 0.51, 0.50]
# robust_median: Winner = 0, Scores = [0.8, 0.4, 0.5]
```

### Example 2: With Weights

```python
# Give experts more weight than novices
utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]
weights = [0.5, 0.3, 0.2]  # Expert, experienced, novice

result = aggregate(utilities, method="score_centroid", weights=weights)
# Weighted average favors expert opinions
```

### Example 3: Method-Specific Parameters

```python
# Atkinson with different inequality aversion
utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]

# Mild fairness
result1 = aggregate(utilities, method="atkinson", epsilon=0.5)

# Balanced
result2 = aggregate(utilities, method="atkinson", epsilon=1.0)

# Strong fairness
result3 = aggregate(utilities, method="atkinson", epsilon=2.0)
```

### Example 4: Using Aliases

```python
# These are equivalent:
result1 = aggregate(utilities, method="atkinson", epsilon=1.0)
result2 = aggregate(utilities, method="fair")  # Uses default epsilon=1.0

# These are equivalent:
result3 = aggregate(utilities, method="maximin")
result4 = aggregate(utilities, method="minority-focused")
```

---

## See Also

- [Mechanism Aliases](aliases.md) - Intuitive names for methods
- [Property Analysis](properties.md) - Select methods by theoretical properties
- [API Reference](../reference/api.md) - Complete API documentation
