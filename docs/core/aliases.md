# Mechanism Aliases

Intuitive names for aggregation mechanisms.

## Overview

Instead of remembering technical names like `schulze_condorcet` or `atkinson_inequality`, you can use intuitive aliases that describe what the mechanism does.

```python
from agorai.aggregate import aggregate

# These are equivalent:
result = aggregate(utilities, method="atkinson", epsilon=1.0)
result = aggregate(utilities, method="fair")

# These are equivalent:
result = aggregate(utilities, method="maximin")
result = aggregate(utilities, method="minority-focused")
```

## Available Aliases

| Alias | Resolves To | Description | Use When |
|-------|-------------|-------------|----------|
| `fair` | `atkinson` | Balances efficiency with inequality aversion | You want fairness without sacrificing too much efficiency |
| `minority-focused` | `maximin` | Protects worst-off group | Minority protection is paramount |
| `robust` | `robust_median` | Resistant to outliers | You have adversarial or unreliable agents |
| `democratic` | `majority` | Simple plurality voting | You want democratic legitimacy |
| `consensus-seeking` | `consensus` | Minimizes disagreement | Building agreement is important |
| `efficient` | `score_centroid` | Optimized for total welfare | Efficiency matters most |
| `condorcet` | `schulze_condorcet` | Respects pairwise preferences | You have ranked preferences |
| `ranked-choice` | `borda` | Considers full rankings | You want to use all ranking information |
| `strategy-proof` | `approval_voting` | Hard to manipulate | Strategic voting is a concern |
| `veto-protected` | `veto_hybrid` | Minority veto power | Need to block harmful outcomes |

## Usage

### Basic Usage

```python
from agorai.aggregate import aggregate

utilities = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]

# Use intuitive alias
result = aggregate(utilities, method="fair")

# Or use technical name (equivalent)
result = aggregate(utilities, method="atkinson", epsilon=1.0)
```

### With Method Parameters

Aliases work with all method-specific parameters:

```python
# Alias with parameters
result = aggregate(utilities, method="fair", epsilon=2.0)

# Equivalent to:
result = aggregate(utilities, method="atkinson", epsilon=2.0)
```

### Listing Available Aliases

```python
from agorai.aggregate import list_aliases

# Get all aliases
aliases = list_aliases()
print(aliases)
# {'fair': 'atkinson', 'robust': 'robust_median', ...}

# Get aliases for a specific property
from agorai.aggregate import get_aliases_by_property
fair_aliases = get_aliases_by_property("fairness")
# ['fair', 'minority-focused', ...]
```

## Custom Aliases

You can add your own aliases at runtime:

```python
from agorai.aggregate import add_custom_alias, aggregate

# Add custom alias
add_custom_alias("my-favorite", "schulze_condorcet")

# Use it
result = aggregate(utilities, method="my-favorite")
```

### Persistent Custom Aliases

To make custom aliases permanent, add them to `src/agorai/aggregate/aliases.yaml`:

```yaml
# aliases.yaml
my-favorite: schulze_condorcet
safety-first: maximin
profit-maximizing: score_centroid
```

## Alias Resolution

The system resolves aliases as follows:

1. **Check if direct match**: If method name matches a technical method, use it directly
2. **Check aliases**: If method name is an alias, resolve to technical method
3. **Check custom aliases**: If method name is a custom alias, resolve it
4. **Error**: If no match found, raise `ValueError`

```python
# Direct match - no resolution needed
aggregate(utilities, method="maximin")  # ✓ Technical name

# Alias resolution
aggregate(utilities, method="minority-focused")  # ✓ Resolves to maximin

# Custom alias (if defined)
aggregate(utilities, method="my-favorite")  # ✓ Resolves to configured method

# Invalid
aggregate(utilities, method="nonexistent")  # ✗ ValueError
```

## When to Use Aliases vs Technical Names

### Use Aliases When:
- ✅ Writing quick prototypes
- ✅ Teaching/demos
- ✅ You care about what the method does, not its name
- ✅ Code readability for non-experts

### Use Technical Names When:
- ✅ Writing research code that will be published
- ✅ You need specific method parameters
- ✅ Documenting exact methods used
- ✅ Comparing to literature

## Examples

### Example 1: Quick Prototyping

```python
# Try different fairness levels
for fairness in ["efficient", "fair", "minority-focused"]:
    result = aggregate(utilities, method=fairness)
    print(f"{fairness}: {result['winner']}")

# Output:
# efficient: 0 (utilitarian)
# fair: 0 (balanced)
# minority-focused: 2 (protects minorities)
```

### Example 2: Method Comparison

```python
methods = {
    "Democratic": "democratic",
    "Fair": "fair",
    "Robust": "robust",
    "Consensus": "consensus-seeking",
}

for name, alias in methods.items():
    result = aggregate(utilities, method=alias)
    print(f"{name}: Winner={result['winner']}")
```

### Example 3: Domain-Specific Aliases

```python
# For content moderation
add_custom_alias("safety-first", "maximin")
add_custom_alias("balanced-moderation", "atkinson")

# For recommendation systems
add_custom_alias("popularity", "majority")
add_custom_alias("personalized", "score_centroid")
```

## Alias Categories

### By Priority

**Fairness-Focused:**
- `fair` (Atkinson)
- `minority-focused` (Maximin)
- `veto-protected` (Veto Hybrid)

**Efficiency-Focused:**
- `efficient` (Score Centroid)
- `democratic` (Majority)

**Robustness-Focused:**
- `robust` (Robust Median)
- `strategy-proof` (Approval Voting)

**Consensus-Focused:**
- `consensus-seeking` (Consensus)
- `ranked-choice` (Borda)
- `condorcet` (Schulze)

## Future Extensions

The alias system is designed to support future enhancements:

1. **Context-Aware Aliases**: Aliases that resolve differently based on context
2. **Parameterized Aliases**: Aliases that include default parameters
3. **Composite Aliases**: Aliases that trigger multiple methods
4. **Domain-Specific Bundles**: Pre-configured alias sets for specific domains

## See Also

- [Aggregation Methods](aggregation.md) - Complete method documentation
- [Property Analysis](properties.md) - Select methods by properties
- [Configuration](../reference/configuration.md) - Configure aliases globally
