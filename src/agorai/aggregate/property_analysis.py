"""Mechanism property analysis and verification.

This module provides tools to analyze and verify theoretical properties
of aggregation mechanisms, enabling evidence-based mechanism selection.
"""

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
from itertools import permutations
import time

from agorai.aggregate.core import aggregate, AGGREGATOR_REGISTRY


# Property Matrix: Mechanism -> Property -> Score (0.0-1.0)
# Based on theoretical analysis and empirical validation
PROPERTY_MATRIX = {
    "majority": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.0,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.0,
        "outlier_resistance": 0.0,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "weighted_plurality": {
        "equal_treatment": 0.0,
        "anonymity": 0.0,
        "minority_protection": 0.0,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.0,
        "outlier_resistance": 0.0,
        "strategy_proofness": 0.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "borda": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.4,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.6,
        "outlier_resistance": 0.0,
        "strategy_proofness": 0.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "schulze_condorcet": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.5,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 1.0,
        "outlier_resistance": 0.5,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 1.0,
        "monotonic": 1.0,
    },
    "approval_voting": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.4,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.5,
        "outlier_resistance": 0.5,
        "strategy_proofness": 1.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "supermajority": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.8,
        "worst_off_protection": 0.4,
        "veto_power": 0.4,
        "pareto_efficiency": 0.0,
        "outlier_resistance": 0.0,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "maximin": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 1.0,
        "worst_off_protection": 1.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.0,
        "outlier_resistance": 1.0,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "atkinson": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.9,
        "worst_off_protection": 0.7,
        "veto_power": 0.0,
        "pareto_efficiency": 0.6,
        "outlier_resistance": 0.9,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "score_centroid": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.0,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.5,
        "outlier_resistance": 0.0,
        "strategy_proofness": 0.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "robust_median": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.5,
        "worst_off_protection": 0.5,
        "veto_power": 0.0,
        "pareto_efficiency": 0.0,
        "outlier_resistance": 1.0,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "consensus": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 0.4,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.5,
        "outlier_resistance": 0.4,
        "strategy_proofness": 0.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "quadratic_voting": {
        "equal_treatment": 0.5,
        "anonymity": 1.0,
        "minority_protection": 0.5,
        "worst_off_protection": 0.0,
        "veto_power": 0.0,
        "pareto_efficiency": 0.5,
        "outlier_resistance": 0.5,
        "strategy_proofness": 0.5,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "nash_bargaining": {
        "equal_treatment": 0.5,
        "anonymity": 1.0,
        "minority_protection": 0.5,
        "worst_off_protection": 0.5,
        "veto_power": 0.0,
        "pareto_efficiency": 1.0,
        "outlier_resistance": 0.5,
        "strategy_proofness": 0.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
    "veto_hybrid": {
        "equal_treatment": 1.0,
        "anonymity": 1.0,
        "minority_protection": 1.0,
        "worst_off_protection": 1.0,
        "veto_power": 1.0,
        "pareto_efficiency": 0.0,
        "outlier_resistance": 0.5,
        "strategy_proofness": 0.0,
        "condorcet_consistent": 0.0,
        "monotonic": 1.0,
    },
}


def recommend_mechanism(
    priority_properties: List[str],
    weights: List[float] = None,
    min_score: float = 0.0
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Recommend mechanisms based on desired properties.

    Parameters
    ----------
    priority_properties : List[str]
        Properties in order of importance, e.g.:
        ['minority_protection', 'outlier_resistance', 'anonymity']
    weights : List[float], optional
        Explicit weights for each property (default: diminishing 1/i)
    min_score : float
        Minimum compatibility score to include (default: 0.0)

    Returns
    -------
    List[Tuple[str, float, Dict[str, float]]]
        Ranked list of (mechanism_name, total_score, property_scores)

    Examples
    --------
    >>> recommendations = recommend_mechanism(
    ...     ['minority_protection', 'outlier_resistance'],
    ...     weights=[0.7, 0.3]
    ... )
    >>> print(recommendations[0])
    ('maximin', 1.0, {'minority_protection': 1.0, 'outlier_resistance': 1.0})
    """
    if weights is None:
        # Default: diminishing weights 1, 1/2, 1/3, ...
        weights = [1.0 / (i + 1) for i in range(len(priority_properties))]

    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    results = []

    for mechanism in AGGREGATOR_REGISTRY.keys():
        if mechanism not in PROPERTY_MATRIX:
            continue

        # Compute weighted score
        total_score = 0.0
        property_scores = {}

        for prop, weight in zip(priority_properties, weights):
            if prop in PROPERTY_MATRIX[mechanism]:
                prop_score = PROPERTY_MATRIX[mechanism][prop]
                total_score += weight * prop_score
                property_scores[prop] = prop_score

        if total_score >= min_score:
            results.append((mechanism, total_score, property_scores))

    # Sort by total score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def verify_anonymity(method: str, utilities: List[List[float]], n_tests: int = 10) -> float:
    """Verify anonymity property through permutation tests.

    Parameters
    ----------
    method : str
        Aggregation method name
    utilities : List[List[float]]
        Test utility matrix
    n_tests : int
        Number of random permutations to test

    Returns
    -------
    float
        Fraction of tests passed (1.0 = perfect anonymity)
    """
    n_agents = len(utilities)
    weights = [1.0] * n_agents

    baseline = aggregate(utilities, method=method, weights=weights)
    baseline_winner = baseline['winner']

    passed = 0

    for _ in range(n_tests):
        # Random permutation
        perm = np.random.permutation(n_agents)
        utilities_perm = [utilities[i] for i in perm]

        result = aggregate(utilities_perm, method=method, weights=weights)

        if result['winner'] == baseline_winner:
            passed += 1

    return passed / n_tests


def measure_outlier_resistance(
    method: str,
    utilities: List[List[float]],
    n_trials: int = 100
) -> Dict[str, float]:
    """Measure outlier resistance (breakdown point).

    Parameters
    ----------
    method : str
        Aggregation method name
    utilities : List[List[float]]
        Baseline utility matrix
    n_trials : int
        Number of Monte Carlo trials

    Returns
    -------
    Dict[str, float]
        {
            'breakdown_point_mean': float,    # Mean outlier fraction tolerable
            'breakdown_point_std': float,     # Std deviation
            'robustness_score': float         # 0-1 score (higher is more robust)
        }
    """
    n_agents, n_candidates = len(utilities), len(utilities[0])
    weights = [1.0] * n_agents

    baseline_result = aggregate(utilities, method=method, weights=weights)
    baseline_winner = baseline_result['winner']

    breakdown_points = []

    for _ in range(n_trials):
        # Try increasing outlier fractions
        for outlier_frac in np.linspace(0.0, 0.5, 20):
            n_outliers = int(outlier_frac * n_agents)

            if n_outliers == 0:
                continue

            # Inject outliers (flip preferences)
            corrupted = utilities.copy()
            outlier_indices = np.random.choice(n_agents, n_outliers, replace=False)

            for idx in outlier_indices:
                # Reverse preference order
                corrupted[idx] = [1.0 - u for u in utilities[idx]]

            result = aggregate(corrupted, method=method, weights=weights + [1.0] * 0)

            if result['winner'] != baseline_winner:
                breakdown_points.append(outlier_frac)
                break

    if not breakdown_points:
        # Never broke down
        return {
            'breakdown_point_mean': 0.5,
            'breakdown_point_std': 0.0,
            'robustness_score': 1.0
        }

    mean_bp = np.mean(breakdown_points)
    std_bp = np.std(breakdown_points)

    # Robustness score: higher breakdown point = more robust
    robustness_score = mean_bp / 0.5  # Normalize to [0, 1]

    return {
        'breakdown_point_mean': mean_bp,
        'breakdown_point_std': std_bp,
        'robustness_score': robustness_score
    }


def compare_mechanisms_on_scenario(
    scenario_name: str,
    utilities: List[List[float]],
    methods: List[str] = None
) -> Dict[str, Any]:
    """Compare multiple mechanisms on a single test scenario.

    Parameters
    ----------
    scenario_name : str
        Descriptive name for the scenario
    utilities : List[List[float]]
        Test utility matrix
    methods : List[str], optional
        Methods to compare (default: all registered methods)

    Returns
    -------
    Dict[str, Any]
        {
            'scenario': str,
            'results': List[Dict],  # Per-method results
            'analysis': str         # Natural language summary
        }
    """
    if methods is None:
        methods = list(AGGREGATOR_REGISTRY.keys())

    weights = [1.0] * len(utilities)
    results = []

    for method in methods:
        start_time = time.time()
        result = aggregate(utilities, method=method, weights=weights)
        runtime = time.time() - start_time

        # Get winner utility distribution
        winner_idx = result['winner']
        winner_utilities = [u[winner_idx] for u in utilities]

        results.append({
            'method': method,
            'winner': winner_idx,
            'scores': result['scores'],
            'runtime_ms': runtime * 1000,
            'winner_mean_utility': np.mean(winner_utilities),
            'winner_min_utility': np.min(winner_utilities),
            'winner_max_utility': np.max(winner_utilities),
            'winner_std_utility': np.std(winner_utilities)
        })

    # Analyze diversity of winners
    unique_winners = len(set(r['winner'] for r in results))
    winner_counts = {}
    for r in results:
        winner_counts[r['winner']] = winner_counts.get(r['winner'], 0) + 1

    most_common_winner = max(winner_counts, key=winner_counts.get)
    consensus_level = winner_counts[most_common_winner] / len(results)

    analysis = (
        f"Scenario: {scenario_name}\n"
        f"Mechanisms tested: {len(results)}\n"
        f"Unique winners: {unique_winners}/{len(utilities[0])}\n"
        f"Most common winner: Candidate {most_common_winner} ({consensus_level:.1%} of methods)\n"
    )

    if consensus_level >= 0.8:
        analysis += "High consensus - most mechanisms agree"
    elif consensus_level >= 0.5:
        analysis += "Moderate consensus - some disagreement"
    else:
        analysis += "Low consensus - significant disagreement across mechanisms"

    return {
        'scenario': scenario_name,
        'results': results,
        'consensus_level': consensus_level,
        'unique_winners': unique_winners,
        'analysis': analysis
    }


def get_mechanism_properties(method: str) -> Dict[str, float]:
    """Get all property scores for a mechanism.

    Parameters
    ----------
    method : str
        Mechanism name

    Returns
    -------
    Dict[str, float]
        Property name -> score (0.0-1.0)
    """
    if method not in PROPERTY_MATRIX:
        return {}

    return PROPERTY_MATRIX[method].copy()


def compare_mechanisms_properties(
    methods: List[str],
    properties: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """Compare multiple mechanisms across specified properties.

    Parameters
    ----------
    methods : List[str]
        Methods to compare
    properties : List[str], optional
        Properties to compare (default: all properties)

    Returns
    -------
    Dict[str, Dict[str, float]]
        method -> (property -> score)
    """
    if properties is None:
        # Get all unique properties
        all_props = set()
        for method in methods:
            if method in PROPERTY_MATRIX:
                all_props.update(PROPERTY_MATRIX[method].keys())
        properties = sorted(all_props)

    comparison = {}
    for method in methods:
        if method in PROPERTY_MATRIX:
            comparison[method] = {
                prop: PROPERTY_MATRIX[method].get(prop, 0.0)
                for prop in properties
            }

    return comparison
