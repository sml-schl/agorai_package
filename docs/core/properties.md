# Aggregation Mechanism Property Mapping

**Version:** 1.0
**Date:** 2025-12-16
**Purpose:** Systematically map theoretical properties to aggregation mechanisms

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Properties Taxonomy](#theoretical-properties-taxonomy)
3. [Mechanism-Property Matrix](#mechanism-property-matrix)
4. [Property Verification Methods](#property-verification-methods)
5. [Implementation Guide](#implementation-guide)

---

## Overview

This document establishes a systematic mapping between theoretical properties (e.g., robustness, minority protection, fairness) and aggregation mechanisms (e.g., majority, maximin, atkinson). The goal is to enable:

1. **Evidence-based mechanism selection** - Choose mechanisms based on desired properties
2. **Property verification** - Test whether a mechanism exhibits claimed properties
3. **Comparative analysis** - Compare mechanisms on specific dimensions
4. **Transparent decision-making** - Justify mechanism choices with formal criteria

### Key Insight

No single aggregation mechanism is universally "best". Each mechanism embodies different value tradeoffs:
- **Efficiency** vs **Fairness**
- **Majority preference** vs **Minority protection**
- **Simplicity** vs **Strategic robustness**

---

## Theoretical Properties Taxonomy

### 1. Fairness Properties

#### 1.1 Equality-Focused
- **Equal Treatment** - All agents have equal influence
- **Anonymity** - Agent identity doesn't affect outcome
- **Neutrality** - Candidate labels don't affect process

#### 1.2 Equity-Focused
- **Minority Protection** - Safeguards against tyranny of majority
- **Worst-Off Protection** - Maximizes utility of least advantaged
- **Veto Power** - Minorities can block unacceptable outcomes

### 2. Efficiency Properties

#### 2.1 Utilitarian Efficiency
- **Sum Maximization** - Maximizes total utility across all agents
- **Pareto Efficiency** - No agent can improve without harming another
- **Average Utility** - Maximizes mean agent satisfaction

#### 2.2 Computational Efficiency
- **Time Complexity** - O(n) vs O(n²) vs O(n³)
- **Space Complexity** - Memory requirements
- **Parallelizability** - Can computations be parallelized?

### 3. Robustness Properties

#### 3.1 Statistical Robustness
- **Outlier Resistance** - Not dominated by extreme opinions
- **Noise Tolerance** - Stable under measurement error
- **Breakdown Point** - Fraction of bad data tolerable

#### 3.2 Strategic Robustness
- **Strategy-Proofness** - Truth-telling is optimal
- **Manipulation Resistance** - Hard to game the system
- **Collusion Resistance** - Coordinated lying doesn't help

### 4. Social Choice Axioms

#### 4.1 Arrow's Conditions
- **Unrestricted Domain** - Works for any preferences
- **Non-Dictatorship** - No single agent determines outcome
- **Independence of Irrelevant Alternatives** (IIA) - Removing non-winner doesn't change winner
- **Pareto Efficiency** (see above)

#### 4.2 Condorcet Properties
- **Condorcet Consistency** - Selects Condorcet winner when exists
- **Smith Set** - Winner is from the Smith set
- **Clone Independence** - Adding similar candidates doesn't hurt them

### 5. Consensus Properties

- **Agreement Detection** - Identifies genuine consensus
- **Disagreement Minimization** - Reduces conflict
- **Compromise Quality** - Acceptable to most agents

---

## Mechanism-Property Matrix

### Matrix Legend
- ✅ **Strong** - Mechanism strongly satisfies property (provable)
- ⚠️ **Weak** - Partially satisfies or context-dependent
- ❌ **Fails** - Violates property
- ❓ **Unknown** - Needs empirical verification

### Fairness Properties

| Mechanism | Equal Treatment | Anonymity | Minority Protection | Worst-Off Protection | Veto Power |
|-----------|----------------|-----------|---------------------|---------------------|------------|
| **majority** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **weighted_plurality** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **borda** | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| **schulze_condorcet** | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| **approval_voting** | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| **supermajority** | ✅ | ✅ | ✅ | ⚠️ | ⚠️ |
| **maximin** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **atkinson** | ✅ | ✅ | ✅ | ⚠️ | ❌ |
| **score_centroid** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **robust_median** | ✅ | ✅ | ⚠️ | ⚠️ | ❌ |
| **consensus** | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| **quadratic_voting** | ⚠️ | ✅ | ⚠️ | ❌ | ❌ |
| **nash_bargaining** | ⚠️ | ✅ | ⚠️ | ⚠️ | ❌ |
| **veto_hybrid** | ✅ | ✅ | ✅ | ✅ | ✅ |

### Efficiency Properties

| Mechanism | Pareto Efficiency | Sum Maximization | Time Complexity | Space Complexity |
|-----------|-------------------|------------------|-----------------|------------------|
| **majority** | ❌ | ❌ | O(nm) | O(m) |
| **weighted_plurality** | ❌ | ❌ | O(nm) | O(m) |
| **borda** | ⚠️ | ⚠️ | O(nm log m) | O(m) |
| **schulze_condorcet** | ✅ | ❌ | O(m³) | O(m²) |
| **approval_voting** | ⚠️ | ⚠️ | O(nm) | O(m) |
| **supermajority** | ❌ | ❌ | O(nm) | O(m) |
| **maximin** | ❌ | ❌ | O(nm) | O(m) |
| **atkinson** | ⚠️ | ⚠️ | O(nm) | O(m) |
| **score_centroid** | ⚠️ | ✅ | O(nm) | O(m) |
| **robust_median** | ❌ | ❌ | O(nm log n) | O(nm) |
| **consensus** | ⚠️ | ⚠️ | O(nm) | O(m) |
| **quadratic_voting** | ⚠️ | ⚠️ | O(nm) | O(m) |
| **nash_bargaining** | ✅ | ❌ | O(nm) | O(m) |
| **veto_hybrid** | ❌ | ❌ | O(nm) | O(m) |

*Where n = number of agents, m = number of candidates*

### Robustness Properties

| Mechanism | Outlier Resistance | Strategy-Proofness | Manipulation Resistance | Collusion Resistance |
|-----------|--------------------|--------------------|------------------------|---------------------|
| **majority** | ❌ | ⚠️ | ❌ | ❌ |
| **weighted_plurality** | ❌ | ❌ | ❌ | ❌ |
| **borda** | ❌ | ❌ | ❌ | ❌ |
| **schulze_condorcet** | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **approval_voting** | ⚠️ | ✅ | ⚠️ | ⚠️ |
| **supermajority** | ❌ | ⚠️ | ⚠️ | ⚠️ |
| **maximin** | ✅ | ⚠️ | ⚠️ | ❌ |
| **atkinson** | ✅ | ⚠️ | ⚠️ | ❌ |
| **score_centroid** | ❌ | ❌ | ❌ | ❌ |
| **robust_median** | ✅ | ⚠️ | ✅ | ⚠️ |
| **consensus** | ⚠️ | ❌ | ❌ | ❌ |
| **quadratic_voting** | ⚠️ | ⚠️ | ✅ | ✅ |
| **nash_bargaining** | ⚠️ | ❌ | ❌ | ❌ |
| **veto_hybrid** | ⚠️ | ❌ | ⚠️ | ⚠️ |

### Social Choice Axioms

| Mechanism | Condorcet Consistent | IIA | Clone Independent | Monotonic |
|-----------|---------------------|-----|-------------------|-----------|
| **majority** | ❌ | ❌ | ❌ | ✅ |
| **weighted_plurality** | ❌ | ❌ | ❌ | ✅ |
| **borda** | ❌ | ❌ | ❌ | ✅ |
| **schulze_condorcet** | ✅ | ⚠️ | ✅ | ✅ |
| **approval_voting** | ❌ | ❌ | ⚠️ | ✅ |
| **supermajority** | ❌ | ❌ | ❌ | ✅ |
| **maximin** | ❌ | ✅ | ✅ | ✅ |
| **atkinson** | ❌ | ✅ | ✅ | ✅ |
| **score_centroid** | ❌ | ✅ | ✅ | ✅ |
| **robust_median** | ❌ | ✅ | ✅ | ✅ |
| **consensus** | ❌ | ⚠️ | ⚠️ | ✅ |
| **quadratic_voting** | ❌ | ❌ | ❌ | ✅ |
| **nash_bargaining** | ❌ | ✅ | ✅ | ✅ |
| **veto_hybrid** | ❌ | ❌ | ❌ | ✅ |

---

## Property Verification Methods

### 1. Axiomatic Verification (Formal Proofs)

For properties with mathematical definitions, verify through formal proofs.

**Example: Anonymity for Majority Voting**

```
Theorem: majority() is anonymous

Proof:
Let π be any permutation of agents.
Let U be utility matrix, U' = π(U) (permuted rows).

For majority():
  vote_counts[c] = Σ_{i: argmax_j(U[i][j]) = c} 1
  vote_counts'[c] = Σ_{i: argmax_j(U'[i][j]) = c} 1

Since π only permutes rows but argmax is row-wise:
  {i: argmax(U[i]) = c} = {π(i): argmax(U'[π(i)]) = c}

Therefore vote_counts = vote_counts', QED.
```

**Formalization Template:**

```python
def verify_anonymity(mechanism, utilities_original, permutation):
    """Test if mechanism satisfies anonymity axiom."""
    utilities_permuted = permute_rows(utilities_original, permutation)

    result_original = mechanism(utilities_original)
    result_permuted = mechanism(utilities_permuted)

    return result_original['winner'] == result_permuted['winner']
```

### 2. Empirical Verification (Monte Carlo)

For properties without clean axiomatization, use statistical tests.

**Example: Outlier Resistance**

```python
def measure_outlier_resistance(mechanism, utilities, n_trials=1000):
    """Measure breakdown point: fraction of outliers tolerable."""
    n_agents, n_candidates = len(utilities), len(utilities[0])
    baseline_winner = mechanism(utilities)['winner']

    breakdown_points = []
    for _ in range(n_trials):
        # Inject increasing fractions of outliers
        for outlier_frac in np.linspace(0, 0.5, 20):
            n_outliers = int(outlier_frac * n_agents)
            corrupted = inject_outliers(utilities, n_outliers)
            new_winner = mechanism(corrupted)['winner']

            if new_winner != baseline_winner:
                breakdown_points.append(outlier_frac)
                break

    return np.mean(breakdown_points), np.std(breakdown_points)
```

**Metrics:**
- **Breakdown Point** - Mean outlier fraction before winner changes
- **Stability** - Std deviation of breakdown point across trials
- **Recovery** - Can mechanism return to original winner after outlier removal?

### 3. Adversarial Testing

Actively try to manipulate or break the mechanism.

**Example: Strategic Manipulation**

```python
def test_strategic_manipulation(mechanism, true_utilities):
    """Try to find beneficial misreport for each agent."""
    n_agents = len(true_utilities)
    manipulation_success = []

    for agent_i in range(n_agents):
        # Agent i's true preference
        true_pref = true_utilities[agent_i]
        sincere_winner = mechanism(true_utilities)['winner']
        sincere_utility = true_pref[sincere_winner]

        # Try all possible misreports
        best_gain = 0
        for misreport in generate_all_rankings():
            utilities_with_lie = true_utilities.copy()
            utilities_with_lie[agent_i] = misreport

            new_winner = mechanism(utilities_with_lie)['winner']
            new_utility = true_pref[new_winner]

            gain = new_utility - sincere_utility
            if gain > best_gain:
                best_gain = gain

        manipulation_success.append(best_gain > 0)

    return sum(manipulation_success) / n_agents  # Fraction of agents who can benefit from lying
```

### 4. Comparative Analysis

Compare mechanisms on standardized test scenarios.

**Scenarios:**

1. **Tyranny Test** - 90% agents prefer A, 10% strongly prefer B (utility = 1.0 vs 0.01)
2. **Polarization Test** - 50/50 split with no compromise candidate
3. **Compromise Test** - Majority prefers A (utility 0.6), minority prefers B (utility 1.0), compromise C exists (all 0.7)
4. **Outlier Test** - One agent has extreme opposite preferences
5. **Clone Test** - Add identical copy of candidate A, does A's support get split?

**Expected Outcomes:**

| Test | maximin | majority | atkinson | score_centroid |
|------|---------|----------|----------|----------------|
| Tyranny | Protects minority (B) | Majority wins (A) | Depends on ε | Majority wins (A) |
| Polarization | Either A or B | Either A or B | Depends on ε | Either A or B |
| Compromise | Compromise (C) | Majority (A) | Compromise (C) | Compromise (C) |
| Outlier | Ignores outlier | Ignores outlier | Robust | Influenced |
| Clone | Unaffected | A loses (split) | Unaffected | Unaffected |

---

## Implementation Guide

### Step 1: Property Test Suite

Create a comprehensive test suite in `agorai-package/tests/test_mechanism_properties.py`:

```python
import pytest
from agorai.aggregate import aggregate

class TestFairnessProperties:
    def test_anonymity_all_mechanisms(self):
        """Test anonymity for all mechanisms."""
        utilities = [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [0.5, 1.0, 0.5]]

        for method in ['majority', 'borda', 'maximin', 'atkinson', ...]:
            for perm in generate_all_permutations(3):
                assert verify_anonymity(method, utilities, perm)

    def test_minority_protection_maximin(self):
        """Maximin should protect worst-off agent."""
        # 9 agents prefer A (util 0.9), 1 agent prefers B (util 1.0)
        utilities = [[0.9, 0.1]] * 9 + [[0.1, 1.0]]
        result = aggregate(utilities, method='maximin')

        # Maximin should choose B (min utility 0.1 for B vs 0.1 for A)
        # Actually ties, but let's make it clearer:
        utilities = [[0.9, 0.05]] * 9 + [[0.05, 1.0]]
        result = aggregate(utilities, method='maximin')
        assert result['winner'] == 1  # Protects minority

class TestRobustnessProperties:
    def test_outlier_resistance_median(self):
        """Median should resist outliers."""
        utilities = [[0.8, 0.2]] * 10  # 10 agents prefer A
        baseline = aggregate(utilities, method='robust_median')

        # Add 3 extreme outliers preferring B
        utilities_outliers = utilities + [[0.0, 1.0]] * 3
        result = aggregate(utilities_outliers, method='robust_median')

        assert result['winner'] == baseline['winner']  # Still A

class TestEfficiencyProperties:
    def test_pareto_efficiency_schulze(self):
        """Schulze should be Pareto efficient."""
        utilities = [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]
        result = aggregate(utilities, method='schulze_condorcet')

        # Candidate 0 Pareto dominates 1 and 2
        assert result['winner'] == 0
```

### Step 2: Property Measurement Dashboard

Create visualization tool to compare mechanisms:

```python
# backend/services/mechanism_property_analyzer.py

class MechanismPropertyAnalyzer:
    def analyze_all_mechanisms(self, test_scenarios):
        """Analyze all mechanisms on all test scenarios."""
        results = []

        for scenario in test_scenarios:
            for method in AGGREGATOR_REGISTRY.keys():
                # Run mechanism on scenario
                result = aggregate(scenario.utilities, method=method)

                # Measure properties
                properties = {
                    'fairness': self.measure_fairness(method, scenario),
                    'efficiency': self.measure_efficiency(method, scenario),
                    'robustness': self.measure_robustness(method, scenario),
                    'runtime': self.measure_runtime(method, scenario)
                }

                results.append({
                    'mechanism': method,
                    'scenario': scenario.name,
                    'winner': result['winner'],
                    'properties': properties
                })

        return pd.DataFrame(results)
```

### Step 3: Mechanism Recommendation Engine

Given desired properties, recommend suitable mechanisms:

```python
def recommend_mechanism(
    priority_properties: List[str],
    constraints: Dict[str, Any] = None
) -> List[Tuple[str, float]]:
    """Recommend mechanisms based on desired properties.

    Parameters
    ----------
    priority_properties : List[str]
        Properties in order of importance, e.g.:
        ['minority_protection', 'robustness', 'efficiency']
    constraints : Dict[str, Any]
        Hard constraints, e.g.:
        {'max_time_complexity': 'O(nm)', 'min_pareto_efficiency': 0.8}

    Returns
    -------
    List[Tuple[str, float]]
        Ranked list of (mechanism_name, compatibility_score)
    """
    scores = []

    for mechanism in AGGREGATOR_REGISTRY.keys():
        # Compute compatibility score
        score = 0.0
        for i, prop in enumerate(priority_properties):
            weight = 1.0 / (i + 1)  # Diminishing weights
            score += weight * PROPERTY_MATRIX[mechanism][prop]

        # Check constraints
        if constraints and not satisfies_constraints(mechanism, constraints):
            continue

        scores.append((mechanism, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

## References

- Arrow, K. J. (1951). *Social Choice and Individual Values*
- Sen, A. (1970). *Collective Choice and Social Welfare*
- Moulin, H. (1988). *Axioms of Cooperative Decision Making*
- Rawls, J. (1971). *A Theory of Justice*
- Atkinson, A. B. (1970). *On the measurement of inequality*
- Schulze, M. (2011). *A new monotonic, clone-independent, reversal symmetric, and condorcet-consistent single-winner election method*
