# Aggregation Method Aliases

This directory contains an alias system for aggregation methods, making it easier to use methods based on their properties rather than memorizing technical names.

## Overview

Instead of remembering method names like `robust_median` or `veto_hybrid`, you can use intuitive aliases like `robust` or `minority-protection`.

## Usage

### Basic Usage

```python
from agorai.aggregate import aggregate

# Use alias instead of method name
result = aggregate(utilities, method="robust")  # → robust_median
result = aggregate(utilities, method="minority-focused")  # → maximin
result = aggregate(utilities, method="fair")  # → atkinson
```

### List All Aliases

```python
from agorai.aggregate import load_aliases, list_aliases

# Get all aliases
all_aliases = load_aliases()
print(all_aliases)
# {'robust': 'robust_median', 'minority-focused': 'maximin', ...}

# Get aliases grouped by method
by_method = list_aliases()
print(by_method['robust_median'])
# ['robust', 'outlier-resistant', 'stable']
```

### Find Aliases by Property

```python
from agorai.aggregate import get_aliases_by_property

# Find all minority-related methods
minority_aliases = get_aliases_by_property('minority')
print(minority_aliases)
# [('minority-focused', 'maximin'), ('minority-protection', 'veto_hybrid'), ...]
```

### Resolve Aliases

```python
from agorai.aggregate import get_method_from_alias

# Resolve alias to method name
method = get_method_from_alias("robust")
print(method)  # → 'robust_median'

# Already a method name? No problem
method = get_method_from_alias("majority")
print(method)  # → 'majority'
```

### Add Custom Aliases

```python
from agorai.aggregate import add_custom_alias

# Add a custom alias (session only)
add_custom_alias("my-favorite", "atkinson")

# Add and persist to YAML file
add_custom_alias("team-consensus", "consensus", persist=True)
```

## Available Alias Categories

### Robustness-Focused
- `robust` → robust_median
- `outlier-resistant` → robust_median
- `stable` → robust_median

### Minority-Protection
- `minority-focused` → maximin
- `minority-protection` → veto_hybrid
- `protect-minorities` → maximin
- `veto-based` → veto_hybrid
- `worst-off-protection` → maximin
- `rawlsian` → maximin

### Fairness-Focused
- `egalitarian` → atkinson
- `equality-focused` → atkinson
- `fair` → atkinson
- `inequality-averse` → atkinson

### Consensus-Focused
- `consensus-seeking` → consensus
- `agreement-based` → consensus
- `low-disagreement` → consensus
- `collaborative` → consensus

### Efficiency-Focused
- `utilitarian` → score_centroid
- `efficient` → score_centroid
- `average-based` → score_centroid

### Democratic Voting
- `democratic` → majority
- `simple-vote` → majority
- `one-person-one-vote` → majority
- `plurality` → weighted_plurality

### Intensity-Aware
- `intensity-based` → quadratic_voting
- `preference-strength` → quadratic_voting

### Ranking-Based
- `ranking-based` → borda
- `positional` → borda
- `condorcet` → schulze_condorcet

### Game-Theoretic
- `nash` → nash_bargaining
- `bargaining` → nash_bargaining
- `cooperative` → nash_bargaining

### Composite Properties
- `robust-and-fair` → atkinson
- `democratic-with-protection` → veto_hybrid
- `balanced` → consensus

## File Structure

```
aggregate/
├── aliases.yaml          # Main alias configuration file
├── alias_loader.py       # Python loader for aliases
├── ALIASES_README.md     # This file
└── ...
```

## Configuration File

The `aliases.yaml` file uses a simple mapping format:

```yaml
# Comment lines start with #
alias-name: method_name
robust: robust_median
minority-focused: maximin
```

## Method Selection Guide

Choose aliases based on your priorities:

| Priority | Suggested Alias | Method | Key Property |
|----------|----------------|---------|--------------|
| Protect minorities | `minority-focused` | maximin | Maximizes worst-off utility |
| Robustness to outliers | `robust` | robust_median | Uses median instead of mean |
| Fairness & equality | `fair` | atkinson | Parameterizable inequality aversion |
| Consensus building | `consensus-seeking` | consensus | Balances utility and agreement |
| Democratic simplicity | `democratic` | majority | One agent, one vote |
| Intensity expression | `intensity-based` | quadratic_voting | Budget-constrained voting |
| Minority veto power | `minority-protection` | veto_hybrid | Excludes unacceptable options |

## Advanced: Theoretical Properties

Each method satisfies different theoretical properties. See the main documentation for details on:

- **Pareto Efficiency**: No alternative makes someone better off without making someone worse off
- **Strategy-Proofness**: Agents cannot benefit from misreporting preferences
- **Monotonicity**: Improving a candidate's rank shouldn't hurt them
- **Independence of Irrelevant Alternatives**: Removing a non-winner shouldn't change the winner
- **Condorcet Consistency**: Selects the candidate that beats all others in pairwise comparisons

## Contributing

To add new aliases:

1. Edit `aliases.yaml` directly, or
2. Use `add_custom_alias(..., persist=True)` in Python

Please keep aliases:
- Intuitive and self-explanatory
- Consistent with existing naming patterns
- Well-documented with use cases

## References

For more information on aggregation methods and their properties, see:
- Arrow, K. J. (1951). *Social Choice and Individual Values*
- Atkinson, A. B. (1970). *On the measurement of inequality*
- Sen, A. (1970). *Collective Choice and Social Welfare*
