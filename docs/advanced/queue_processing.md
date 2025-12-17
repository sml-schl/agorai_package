# Queue Processing Guide

The `agorai.queue` module provides a framework for processing multiple aggregation requests from files, enabling batch operations on production data, test datasets, benchmarks, or any collection of decision-making scenarios.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [File Format](#file-format)
4. [Processing Requests](#processing-requests)
5. [Comparing Methods](#comparing-methods)
6. [Metrics](#metrics)
7. [Examples](#examples)
8. [Best Practices](#best-practices)

---

## Overview

The queue module enables:

- **Batch processing** of aggregation requests from JSON files
- **Production use cases**: Daily decisions, content moderation, automated systems
- **Testing & evaluation**: Process multiple test cases, compare methods scientifically
- **Metrics calculation**: Fairness, efficiency, and agreement metrics for analysis

**Key Functions:**
- `load_requests_from_file()` - Load requests from JSON
- `process_single_request()` - Process one request
- `process_queue()` - Process entire queue with one method
- `compare_methods_on_queue()` - Compare multiple methods on same queue

---

## Quick Start

```python
from agorai.queue import process_queue, compare_methods_on_queue

# Process a queue of requests
results = process_queue(
    requests_file="production_batch.json",
    method="atkinson",
    metrics=["fairness", "efficiency"],
    epsilon=1.0
)

print(f"Processed {results['num_requests']} requests")
print(f"Gini: {results['summary']['fairness']['gini_coefficient']:.3f}")
print(f"Social Welfare: {results['summary']['efficiency']['social_welfare']:.2f}")

# Compare multiple methods
comparison = compare_methods_on_queue(
    requests_file="production_batch.json",
    methods=["majority", "atkinson", "maximin"],
    metrics=["fairness"]
)

print("Fairness Rankings:", comparison['rankings']['fairness_gini_coefficient'])
```

---

## File Format

Request files are JSON documents with the following structure:

```json
{
  "name": "content_moderation_daily_batch",
  "description": "Daily content moderation decisions",
  "items": [
    {
      "id": "item_001",
      "utilities": [
        [0.8, 0.2, 0.5],
        [0.3, 0.7, 0.4],
        [0.6, 0.5, 0.9]
      ],
      "ground_truth": 0,
      "metadata": {
        "timestamp": "2025-11-24T08:15:00Z",
        "content_type": "text"
      }
    }
  ],
  "metadata": {
    "date": "2025-11-24",
    "source": "production",
    "num_candidates": 3,
    "candidate_labels": ["Approve", "Reject", "Escalate"]
  }
}
```

### Required Fields

- **`items`**: Array of aggregation requests
  - **`utilities`**: 2D array (n_agents × n_candidates)

### Optional Fields

- **`name`**: Name of the request set
- **`description`**: Description of what these requests represent
- **`id`**: Unique identifier for each request
- **`ground_truth`**: Correct answer (for accuracy calculation)
- **`metadata`**: Any additional information

---

## Processing Requests

### Single Request

Process one request programmatically:

```python
from agorai.queue import process_single_request

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]

result = process_single_request(
    utilities,
    method="atkinson",
    metrics=["fairness", "efficiency"],
    epsilon=1.0
)

print(f"Winner: Candidate {result['winner']}")
print(f"Gini: {result['metrics']['fairness']['gini_coefficient']:.3f}")
```

### Queue from File

Process multiple requests from a file:

```python
from agorai.queue import process_queue

results = process_queue(
    requests_file="daily_decisions.json",
    method="atkinson",
    metrics=["fairness", "efficiency", "agreement"],
    epsilon=1.0
)

# Access results
print(f"Source: {results['source_name']}")
print(f"Processed: {results['num_requests']} requests")

# Summary statistics
print("\nSummary:")
print(f"  Gini: {results['summary']['fairness']['gini_coefficient']:.3f}")
print(f"  Social Welfare: {results['summary']['efficiency']['social_welfare']:.2f}")
print(f"  Consensus: {results['summary']['agreement']['consensus_score']:.2%}")

# If ground truth provided
if 'accuracy' in results['summary']:
    print(f"  Accuracy: {results['summary']['accuracy']:.1%}")

# Per-item results
for item_result in results['results']:
    print(f"\nItem {item_result['item_id']}:")
    print(f"  Winner: {item_result['winner']}")
    print(f"  Correct: {item_result.get('is_correct', 'N/A')}")
```

### With Different Parameters

```python
# Compare different epsilon values for Atkinson
for epsilon in [0.5, 1.0, 1.5, 2.0]:
    results = process_queue(
        "requests.json",
        method="atkinson",
        metrics=["fairness"],
        epsilon=epsilon
    )
    gini = results['summary']['fairness']['gini_coefficient']
    print(f"ε={epsilon}: Gini={gini:.3f}")
```

---

## Comparing Methods

Compare multiple aggregation methods on the same queue:

```python
from agorai.queue import compare_methods_on_queue

comparison = compare_methods_on_queue(
    requests_file="production_batch.json",
    methods=["majority", "atkinson", "maximin", "nash_bargaining"],
    metrics=["fairness", "efficiency", "agreement"]
)

# View rankings
print("Fairness (Gini) Rankings:")
for rank, method in enumerate(comparison['rankings']['fairness_gini_coefficient'], 1):
    print(f"  {rank}. {method}")

print("\nEfficiency (Social Welfare) Rankings:")
for rank, method in enumerate(comparison['rankings']['efficiency_social_welfare'], 1):
    print(f"  {rank}. {method}")

# Compare specific metrics
print("\nDetailed Comparison:")
for method_result in comparison['methods']:
    name = method_result['method']
    gini = method_result['summary']['fairness']['gini_coefficient']
    welfare = method_result['summary']['efficiency']['social_welfare']
    print(f"{name:20s} Gini={gini:.3f}  Welfare={welfare:.2f}")
```

### With Method Parameters

```python
# Compare methods with different configurations
comparison = compare_methods_on_queue(
    requests_file="requests.json",
    methods=[
        "majority",
        {"name": "atkinson", "params": {"epsilon": 0.5}},
        {"name": "atkinson", "params": {"epsilon": 1.0}},
        {"name": "atkinson", "params": {"epsilon": 2.0}},
        "maximin"
    ],
    metrics=["fairness", "efficiency"]
)
```

---

## Metrics

The queue module computes three categories of metrics:

### 1. Fairness Metrics

Measure how equally utilities are distributed.

**`gini_coefficient`** (0-1, lower is better)
- 0 = perfect equality
- 1 = perfect inequality
- Classic measure from welfare economics

**`atkinson_index`** (0-1, lower is better)
- Parameterizable inequality measure
- Based on equally-distributed equivalent utility

**`variance`** (lower is better)
- Statistical variance of utilities

**`coefficient_of_variation`** (lower is better)
- Normalized variance (CV = σ/μ)

### 2. Efficiency Metrics

Measure total welfare and Pareto optimality.

**`social_welfare`** (higher is better)
- Sum of utilities for winning candidate

**`utilitarian_welfare`** (higher is better)
- Mean utility for winning candidate

**`pareto_efficiency`** (1.0 or 0.0)
- 1.0 if no Pareto-dominated alternative exists

### 3. Agreement Metrics

Measure consensus and preference alignment.

**`consensus_score`** (0-1, higher is better)
- Fraction of agents who prefer the winner

**`average_support`** (0-1, higher is better)
- Mean utility for winner across agents

**`minimum_support`** (0-1, higher is better)
- Minimum utility for winner (worst-case satisfaction)

---

## Examples

### Example 1: Production Content Moderation

```python
from agorai.queue import process_queue

# Process daily content moderation decisions
results = process_queue(
    requests_file="daily_moderation.json",
    method="atkinson",
    metrics=["fairness", "efficiency"],
    epsilon=1.0
)

print(f"Processed {results['num_requests']} moderation decisions")
print(f"Fairness (Gini): {results['summary']['fairness']['gini_coefficient']:.3f}")

# Check decisions
for item in results['results']:
    action = ["Approve", "Reject", "Escalate"][item['winner']]
    print(f"Content {item['item_id']}: {action}")
```

### Example 2: Testing with Ground Truth

```python
from agorai.queue import process_queue

# Test on labeled dataset
results = process_queue(
    requests_file="test_dataset.json",
    method="schulze_condorcet",
    metrics=["fairness", "efficiency", "agreement"]
)

print(f"Accuracy: {results['summary']['accuracy']:.1%}")
print(f"Correct: {results['summary']['num_with_ground_truth']} items")

# Find incorrect predictions
incorrect = [r for r in results['results'] if not r.get('is_correct', True)]
print(f"\nIncorrect predictions: {len(incorrect)}")
for item in incorrect:
    print(f"  {item['item_id']}: predicted {item['winner']}, actual {item['ground_truth']}")
```

### Example 3: Method Selection

```python
from agorai.queue import compare_methods_on_queue

# Find best method for your use case
comparison = compare_methods_on_queue(
    requests_file="validation_set.json",
    methods=["majority", "borda", "atkinson", "maximin", "nash_bargaining"],
    metrics=["fairness", "efficiency", "agreement"]
)

# Find most fair method
fairest = comparison['rankings']['fairness_gini_coefficient'][0]
print(f"Most fair method: {fairest}")

# Find most efficient method
most_efficient = comparison['rankings']['efficiency_social_welfare'][0]
print(f"Most efficient method: {most_efficient}")

# Find best balance
for method_result in comparison['methods']:
    name = method_result['method']
    gini = method_result['summary']['fairness']['gini_coefficient']
    welfare = method_result['summary']['efficiency']['social_welfare']

    # Custom scoring (lower Gini + higher welfare = better)
    score = welfare - gini
    print(f"{name}: score={score:.2f} (Gini={gini:.3f}, Welfare={welfare:.2f})")
```

### Example 4: Parameter Sweep

```python
from agorai.queue import process_queue
import matplotlib.pyplot as plt

# Sweep epsilon parameter for Atkinson method
epsilons = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
gini_values = []
welfare_values = []

for epsilon in epsilons:
    results = process_queue(
        "requests.json",
        method="atkinson",
        metrics=["fairness", "efficiency"],
        epsilon=epsilon
    )
    gini_values.append(results['summary']['fairness']['gini_coefficient'])
    welfare_values.append(results['summary']['efficiency']['social_welfare'])

# Plot tradeoff
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epsilons, gini_values, marker='o')
plt.xlabel('Epsilon (ε)')
plt.ylabel('Gini Coefficient')
plt.title('Fairness vs Inequality Aversion')

plt.subplot(1, 2, 2)
plt.plot(epsilons, welfare_values, marker='o')
plt.xlabel('Epsilon (ε)')
plt.ylabel('Social Welfare')
plt.title('Efficiency vs Inequality Aversion')

plt.tight_layout()
plt.savefig('parameter_sweep.png')
```

---

## Best Practices

### 1. File Organization

```bash
# Organize request files by purpose
data/
├── production/
│   ├── daily_2025-11-24.json
│   ├── daily_2025-11-25.json
│   └── ...
├── testing/
│   ├── validation_set.json
│   ├── test_set.json
│   └── ...
└── benchmarks/
    ├── simple_voting.json
    ├── fairness_test.json
    └── ...
```

### 2. Include Ground Truth When Available

```json
{
  "items": [
    {
      "id": "item_001",
      "utilities": [[0.8, 0.2], [0.3, 0.7]],
      "ground_truth": 0,  // Include for accuracy calculation
      "metadata": {
        "human_decision": 0,
        "confidence": 0.9
      }
    }
  ]
}
```

### 3. Add Metadata for Analysis

```json
{
  "items": [
    {
      "id": "mod_20251124_001",
      "utilities": [[0.8, 0.2], [0.3, 0.7]],
      "metadata": {
        "timestamp": "2025-11-24T08:15:00Z",
        "content_type": "text",
        "reported_by": 12,
        "urgency": "high"
      }
    }
  ]
}
```

### 4. Choose Appropriate Metrics

```python
# For fairness-critical applications
results = process_queue(
    "requests.json",
    method="maximin",
    metrics=["fairness", "agreement"]  // Skip efficiency
)

# For efficiency-focused applications
results = process_queue(
    "requests.json",
    method="majority",
    metrics=["efficiency", "agreement"]  // Skip fairness
)
```

### 5. Validate Request Files

```python
from agorai.queue import load_requests_from_file

try:
    requests = load_requests_from_file("my_requests.json")
    print(f"✓ Loaded {len(requests['items'])} requests")

    # Validate utilities shape
    for i, item in enumerate(requests['items']):
        utilities = item['utilities']
        if not utilities:
            print(f"✗ Item {i}: Empty utilities")
        elif len(set(len(row) for row in utilities)) > 1:
            print(f"✗ Item {i}: Inconsistent number of candidates")
        else:
            print(f"✓ Item {i}: {len(utilities)} agents × {len(utilities[0])} candidates")

except FileNotFoundError:
    print("✗ File not found")
except ValueError as e:
    print(f"✗ Invalid file format: {e}")
```

---

## API Reference

### Functions

**`load_requests_from_file(file_path)`**
- Load requests from JSON file
- Returns: Dict with request data

**`process_single_request(utilities, method, metrics, **method_params)`**
- Process one aggregation request
- Returns: Dict with winner, scores, metrics

**`process_queue(requests_file, method, metrics, **method_params)`**
- Process multiple requests from file
- Returns: Dict with results and summary

**`compare_methods_on_queue(requests_file, methods, metrics)`**
- Compare multiple methods on same queue
- Returns: Dict with comparison and rankings

### Metrics Functions

**`calculate_fairness_metrics(utilities, scores)`**
- Returns: Dict with Gini, Atkinson, variance, CV

**`calculate_efficiency_metrics(utilities, winner)`**
- Returns: Dict with social welfare, utilitarian welfare, Pareto efficiency

**`calculate_agreement_metrics(utilities, winner)`**
- Returns: Dict with consensus score, average support, minimum support

---

## Use Cases

### Production Batch Processing
- Daily content moderation decisions
- Automated recommendation systems
- Resource allocation at scale
- Multi-stakeholder decision-making

### Testing & Validation
- Evaluate methods on labeled datasets
- Compare accuracy across methods
- Validate fairness properties
- Regression testing

### Research & Analysis
- Empirical studies on aggregation methods
- Parameter sensitivity analysis
- Cross-cultural preference studies
- Social choice experiments

---

**For more examples, see the `examples/` directory and the [AgorAI Research Strategy Report](../AgorAI_Research_Strategy_Report.md).**
