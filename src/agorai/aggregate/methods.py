"""Aggregation method implementations.

All methods follow a consistent signature:
    func(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]

Returns:
    {
        'winner': int,              # Index of winning candidate
        'scores': List[float],      # Scores for all candidates
        'metadata': Dict[str, Any]  # Method-specific info
    }
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math
import statistics

from agorai.aggregate.core import register_aggregator


# ============================================================================
# SOCIAL CHOICE THEORY
# ============================================================================

@register_aggregator("majority")
def majority(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """One-agent-one-vote plurality (most votes wins).

    Each agent votes for their highest-utility candidate. Winner is the
    candidate with the most votes.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights (currently ignored for true majority)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with most votes
        scores: Vote counts per candidate
        metadata: Voting details

    Properties
    ----------
    - Strategy-proof under sincere voting
    - Violates Condorcet criterion
    - Risk of tyranny of majority
    - Simple and intuitive

    Examples
    --------
    >>> utilities = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    >>> result = majority(utilities, [1.0, 1.0, 1.0])
    >>> result['winner']
    0
    """
    n_candidates = len(utilities[0])
    vote_counts = [0] * n_candidates

    for u in utilities:
        vote = max(range(n_candidates), key=lambda i: u[i])
        vote_counts[vote] += 1

    winner = max(range(n_candidates), key=lambda i: vote_counts[i])

    return {
        'winner': winner,
        'scores': vote_counts,
        'metadata': {
            'vote_counts': vote_counts,
            'total_votes': sum(vote_counts)
        }
    }


@register_aggregator("weighted_plurality")
def weighted_plurality(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """Plurality with per-agent weights.

    Like majority, but each agent's vote is weighted.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest weighted votes
        scores: Weighted vote counts per candidate
    """
    n_candidates = len(utilities[0])
    weighted_votes = [0.0] * n_candidates

    for u, w in zip(utilities, weights):
        vote = max(range(n_candidates), key=lambda i: u[i])
        weighted_votes[vote] += w

    winner = max(range(n_candidates), key=lambda i: weighted_votes[i])

    return {
        'winner': winner,
        'scores': weighted_votes,
        'metadata': {
            'weighted_votes': weighted_votes,
            'total_weight': sum(weights)
        }
    }


@register_aggregator("borda")
def borda(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """Borda count positional ranking aggregation.

    Candidates receive points based on their ranking position.
    Higher-ranked candidates get more points.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest Borda score
        scores: Borda scores per candidate

    Properties
    ----------
    - Monotonic and Pareto efficient
    - Violates independence of irrelevant alternatives
    - Vulnerable to strategic voting
    - Good for ranking aggregation

    Examples
    --------
    >>> utilities = [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]
    >>> result = borda(utilities, [1.0, 1.0])
    >>> result['scores']  # Candidate 1 ranks highest overall
    [2.0, 4.0, 2.0]
    """
    n_candidates = len(utilities[0])
    borda_scores = [0.0] * n_candidates

    for u, w in zip(utilities, weights):
        # Rank candidates by utility (descending)
        ranked = sorted(range(n_candidates), key=lambda i: -u[i])
        # Assign Borda points (m points for 1st, m-1 for 2nd, ..., 1 for last)
        for rank_pos, idx in enumerate(ranked):
            borda_scores[idx] += (n_candidates - rank_pos) * w

    winner = max(range(n_candidates), key=lambda i: borda_scores[i])

    return {
        'winner': winner,
        'scores': borda_scores,
        'metadata': {
            'borda_scores': borda_scores
        }
    }


@register_aggregator("approval_voting")
def approval_voting(
    utilities: List[List[float]],
    weights: List[float],
    threshold: float = 0.5,
    **params
) -> Dict[str, Any]:
    """Approval voting: agents approve candidates above utility threshold.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    threshold : float
        Utility threshold for approval (0.0-1.0, default: 0.5)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with most approvals
        scores: Weighted approval counts per candidate

    Properties
    ----------
    - Simple and intuitive
    - Resistant to vote splitting
    - Encourages honest voting
    - Strategy-proof under certain conditions
    - No ranking needed

    Examples
    --------
    >>> utilities = [[0.8, 0.3], [0.6, 0.9], [0.9, 0.2]]
    >>> result = approval_voting(utilities, [1.0, 1.0, 1.0], threshold=0.5)
    >>> result['winner']  # Candidate 0 gets 3 approvals, candidate 1 gets 1
    0
    """
    n_candidates = len(utilities[0])
    approval_scores = [0.0] * n_candidates

    for u, w in zip(utilities, weights):
        for i in range(n_candidates):
            if u[i] >= threshold:
                approval_scores[i] += w

    # Fallback if no approvals
    if sum(approval_scores) == 0:
        # Use highest average utility
        avg_utils = [sum(u[i] for u in utilities) / len(utilities) for i in range(n_candidates)]
        winner = max(range(n_candidates), key=lambda i: avg_utils[i])
        approval_scores = avg_utils
    else:
        winner = max(range(n_candidates), key=lambda i: approval_scores[i])

    return {
        'winner': winner,
        'scores': approval_scores,
        'metadata': {
            'threshold': threshold,
            'approval_counts': approval_scores
        }
    }


@register_aggregator("supermajority")
def supermajority(
    utilities: List[List[float]],
    weights: List[float],
    threshold: float = 0.66,
    **params
) -> Dict[str, Any]:
    """Supermajority voting: requires threshold fraction of votes to win.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    threshold : float
        Vote share required to win (0.5-1.0, default: 0.66)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate meeting supermajority threshold
        scores: Vote shares per candidate

    Properties
    ----------
    - Protects status quo
    - Reduces frequency of close votes
    - May result in no winner (falls back to plurality)
    - Configurable threshold (2/3 common)
    - Useful for high-stakes decisions
    """
    n_candidates = len(utilities[0])
    vote_weights = [0.0] * n_candidates
    total_weight = sum(weights)

    for u, w in zip(utilities, weights):
        vote = max(range(n_candidates), key=lambda i: u[i])
        vote_weights[vote] += w

    vote_shares = [v / total_weight for v in vote_weights]
    winner = None
    winner_share = 0.0

    # Check for supermajority winner
    for i in range(n_candidates):
        if vote_shares[i] >= threshold:
            if winner is None or vote_weights[i] > vote_weights[winner]:
                winner = i
                winner_share = vote_shares[i]

    # Fallback to plurality if no supermajority
    if winner is None:
        winner = max(range(n_candidates), key=lambda i: vote_weights[i])
        winner_share = vote_shares[winner]

    return {
        'winner': winner,
        'scores': vote_shares,
        'metadata': {
            'threshold': threshold,
            'vote_shares': vote_shares,
            'threshold_met': winner_share >= threshold,
            'winner_share': winner_share
        }
    }


@register_aggregator("schulze_condorcet")
def schulze_condorcet(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """Schulze method for Condorcet-consistent ranking aggregation.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights

    Returns
    -------
    Dict[str, Any]
        winner: Index of Condorcet winner (or None)
        scores: Empty dict (Schulze doesn't produce cardinal scores)

    Properties
    ----------
    - Condorcet-consistent (selects Condorcet winner when exists)
    - Clone-independent
    - Monotonic
    - Computationally expensive (O(n³))

    Notes
    -----
    Uses Floyd-Warshall algorithm to compute strongest paths between candidates.
    """
    n_candidates = len(utilities[0])

    # Build pairwise preference matrix from utilities
    pairwise = defaultdict(int)
    for u in utilities:
        # Create ranking from utilities
        ranked = sorted(range(n_candidates), key=lambda i: -u[i])
        # Count pairwise preferences
        for i, a in enumerate(ranked):
            for b in ranked[i+1:]:
                pairwise[(a, b)] += 1

    # Compute Schulze path strengths
    p = {i: {j: 0 for j in range(n_candidates)} for i in range(n_candidates)}

    # Initialize with direct comparisons
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i != j:
                if pairwise.get((i, j), 0) > pairwise.get((j, i), 0):
                    p[i][j] = pairwise.get((i, j), 0)

    # Floyd-Warshall to find strongest paths
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i == j:
                continue
            for k in range(n_candidates):
                if i == k or j == k:
                    continue
                p[j][k] = max(p[j][k], min(p[j][i], p[i][k]))

    # Find Condorcet winner(s)
    winners = []
    for a in range(n_candidates):
        is_winner = True
        for b in range(n_candidates):
            if a == b:
                continue
            if p[a][b] < p[b][a]:
                is_winner = False
                break
        if is_winner:
            winners.append(a)

    winner = winners[0] if winners else None

    return {
        'winner': winner,
        'scores': [0.0] * n_candidates,  # Schulze doesn't produce cardinal scores
        'metadata': {
            'pairwise_matrix': dict(pairwise),
            'path_strengths': {f"{i},{j}": p[i][j] for i in range(n_candidates) for j in range(n_candidates)},
            'condorcet_winners': winners
        }
    }


# ============================================================================
# WELFARE ECONOMICS
# ============================================================================

@register_aggregator("maximin")
def maximin(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """Rawlsian maximin: maximize the minimum utility across agents.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights (currently ignored for true maximin)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest minimum utility
        scores: Minimum utilities per candidate

    Properties
    ----------
    - Strongly egalitarian
    - Protects worst-off agent
    - May sacrifice efficiency for fairness
    - Pareto suboptimal in some cases
    - Clear fairness interpretation

    Examples
    --------
    >>> utilities = [[0.9, 0.1], [0.6, 0.8], [0.7, 0.2]]
    >>> result = maximin(utilities, [1.0, 1.0, 1.0])
    >>> result['winner']  # Candidate 1 has min utility 0.1, candidate 0 has 0.6
    0
    """
    n_candidates = len(utilities[0])
    min_utilities = []

    for j in range(n_candidates):
        candidate_utilities = [u[j] for u in utilities]
        min_utilities.append(min(candidate_utilities))

    winner = max(range(n_candidates), key=lambda i: min_utilities[i])

    return {
        'winner': winner,
        'scores': min_utilities,
        'metadata': {
            'min_utilities': min_utilities,
            'worst_off_agent': [
                min(range(len(utilities)), key=lambda i: utilities[i][j])
                for j in range(n_candidates)
            ]
        }
    }


@register_aggregator("atkinson")
def atkinson(
    utilities: List[List[float]],
    weights: List[float],
    epsilon: float = 1.0,
    **params
) -> Dict[str, Any]:
    """Atkinson index-based aggregation: parameterizable inequality aversion.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    epsilon : float
        Inequality aversion parameter (0 to infinity, default: 1.0)
        - 0: utilitarian (sum of utilities)
        - 1: geometric mean
        - >1: increasing inequality aversion
        - →∞: approaches maximin

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest EDE utility
        scores: Equally-distributed equivalent utilities per candidate

    Properties
    ----------
    - Parameterizable inequality aversion
    - Satisfies monotonicity and anonymity
    - Continuous spectrum from utilitarian to maximin
    - Based on welfare economics theory

    Examples
    --------
    >>> utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
    >>> # Geometric mean (epsilon=1)
    >>> result = atkinson(utilities, [1.0, 1.0, 1.0], epsilon=1.0)
    """
    def ede(utilities_list: List[float]) -> float:
        """Calculate equally-distributed equivalent utility."""
        n = len(utilities_list)
        if n == 0:
            return 0.0

        if epsilon == 0.0:
            # Utilitarian: simple mean
            return sum(utilities_list) / n

        if epsilon == 1.0:
            # Geometric mean
            product = 1.0
            for u in utilities_list:
                product *= max(u, 1e-10)  # Avoid log(0)
            return product ** (1.0 / n)

        # General case: power mean
        total = sum(max(u, 1e-10) ** (1 - epsilon) for u in utilities_list)
        return (total / n) ** (1 / (1 - epsilon))

    n_candidates = len(utilities[0])
    ede_scores = []

    for j in range(n_candidates):
        candidate_utilities = [u[j] for u in utilities]
        ede_scores.append(ede(candidate_utilities))

    winner = max(range(n_candidates), key=lambda i: ede_scores[i])

    interpretation = (
        "utilitarian (sum of utilities)" if epsilon == 0.0
        else "geometric mean" if epsilon == 1.0
        else f"inequality aversion level {epsilon}"
    )

    return {
        'winner': winner,
        'scores': ede_scores,
        'metadata': {
            'epsilon': epsilon,
            'ede_scores': ede_scores,
            'interpretation': interpretation
        }
    }


# ============================================================================
# MACHINE LEARNING / STATISTICS
# ============================================================================

@register_aggregator("score_centroid")
def score_centroid(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """Weighted average of normalized utilities.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest average utility
        scores: Weighted average utilities per candidate

    Examples
    --------
    >>> utilities = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    >>> result = score_centroid(utilities, [1.0, 1.0, 1.0])
    >>> result['scores']
    [0.5, 0.5]
    """
    n_candidates = len(utilities[0])
    weighted_sums = [0.0] * n_candidates
    total_weight = sum(weights)

    for u, w in zip(utilities, weights):
        for i in range(n_candidates):
            weighted_sums[i] += u[i] * w

    scores = [s / total_weight for s in weighted_sums]
    winner = max(range(n_candidates), key=lambda i: scores[i])

    return {
        'winner': winner,
        'scores': scores,
        'metadata': {
            'weighted_average': scores
        }
    }


@register_aggregator("robust_median")
def robust_median(utilities: List[List[float]], weights: List[float], **params) -> Dict[str, Any]:
    """Per-candidate median utility (outlier-resistant).

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights (currently ignored for true median)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest median utility
        scores: Median utilities per candidate

    Properties
    ----------
    - Resistant to outliers
    - Non-parametric
    - Ignores extreme opinions
    - Simple and robust
    """
    n_candidates = len(utilities[0])
    medians = []

    for j in range(n_candidates):
        candidate_utilities = [u[j] for u in utilities]
        medians.append(statistics.median(candidate_utilities))

    winner = max(range(n_candidates), key=lambda i: medians[i])

    return {
        'winner': winner,
        'scores': medians,
        'metadata': {
            'medians': medians
        }
    }


@register_aggregator("consensus")
def consensus(
    utilities: List[List[float]],
    weights: List[float],
    cv_weight: float = 0.5,
    **params
) -> Dict[str, Any]:
    """Balance mean utility and consensus (low disagreement).

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    cv_weight : float
        Weight for coefficient of variation penalty (0-1, default: 0.5)
        - 0: ignore consensus, just maximize mean utility
        - 1: only minimize disagreement, ignore mean
        - 0.5: balance both

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with best consensus score
        scores: Combined consensus scores per candidate

    Properties
    ----------
    - Measures utility variance/coefficient of variation
    - Low variance = high consensus
    - Can detect genuine agreement
    - Useful as meta-aggregator or confidence metric
    """
    n_candidates = len(utilities[0])
    consensus_scores = []
    mean_utilities = []
    cv_values = []

    for j in range(n_candidates):
        candidate_utilities = [u[j] for u in utilities]

        mean_u = statistics.mean(candidate_utilities)
        mean_utilities.append(mean_u)

        if len(candidate_utilities) > 1 and mean_u > 1e-6:
            stdev_u = statistics.stdev(candidate_utilities)
            cv = stdev_u / mean_u
        else:
            cv = 0.0

        cv_values.append(cv)

        # Combined score: balance mean utility and consensus
        score = mean_u * (1 - cv_weight) + (1 - cv) * cv_weight
        consensus_scores.append(score)

    winner = max(range(n_candidates), key=lambda i: consensus_scores[i])

    return {
        'winner': winner,
        'scores': consensus_scores,
        'metadata': {
            'cv_weight': cv_weight,
            'mean_utilities': mean_utilities,
            'coefficient_of_variation': cv_values,
            'has_consensus': [cv < 0.3 for cv in cv_values]
        }
    }


# ============================================================================
# GAME THEORY
# ============================================================================

@register_aggregator("quadratic_voting")
def quadratic_voting(
    utilities: List[List[float]],
    weights: List[float],
    budget: float = 10.0,
    **params
) -> Dict[str, Any]:
    """Budget-constrained voting with quadratic costs.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    budget : float
        Vote budget per agent (default: 10.0)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate with highest QV score
        scores: Quadratic voting scores per candidate

    Properties
    ----------
    - Encourages intensity expression
    - Resistant to strategic voting
    - Budget constraint prevents dominance
    - Square-root dampening of extreme preferences
    """
    n_candidates = len(utilities[0])
    total_alloc = [0.0] * n_candidates

    for u, w in zip(utilities, weights):
        # Allocate budget proportional to utilities
        u_sum = sum(u)
        if u_sum > 1e-9:
            for i in range(n_candidates):
                alloc = (u[i] / u_sum) * budget * w
                total_alloc[i] += alloc

    # Apply square root (quadratic voting effect)
    effective_scores = [math.sqrt(max(0.0, x)) for x in total_alloc]
    winner = max(range(n_candidates), key=lambda i: effective_scores[i])

    return {
        'winner': winner,
        'scores': effective_scores,
        'metadata': {
            'budget': budget,
            'total_allocations': total_alloc,
            'effective_scores': effective_scores
        }
    }


@register_aggregator("nash_bargaining")
def nash_bargaining(
    utilities: List[List[float]],
    weights: List[float],
    disagreement: str = "worst",
    **params
) -> Dict[str, Any]:
    """Nash bargaining solution: maximize weighted product of utility gains.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    disagreement : str
        Disagreement point type (default: "worst")
        - "zero": no fallback (d_i = 0)
        - "worst": worst candidate for each agent
        - "uniform": equal split (d_i = 1/m)

    Returns
    -------
    Dict[str, Any]
        winner: Index of candidate maximizing Nash product
        scores: Nash products per candidate

    Properties
    ----------
    - Satisfies Pareto optimality
    - Symmetric and scale-invariant
    - Independence of irrelevant alternatives
    - Requires disagreement point (fallback)
    - Multiplicative rather than additive
    """
    n_candidates = len(utilities[0])
    n_agents = len(utilities)

    # Calculate disagreement point
    if disagreement == "zero":
        d = [0.0] * n_agents
    elif disagreement == "uniform":
        d = [1.0 / n_candidates] * n_agents
    else:  # "worst"
        d = [min(u) for u in utilities]

    # Calculate Nash product for each candidate
    nash_products = []
    for j in range(n_candidates):
        product = 1.0
        for i in range(n_agents):
            gain = utilities[i][j] - d[i]
            if gain <= 0:
                # No gain - heavily penalize
                product *= 1e-10
            else:
                # Weighted Nash product
                product *= gain ** weights[i]
        nash_products.append(product)

    winner = max(range(n_candidates), key=lambda i: nash_products[i])

    return {
        'winner': winner,
        'scores': nash_products,
        'metadata': {
            'disagreement_point': d,
            'disagreement_type': disagreement,
            'nash_products': nash_products
        }
    }


@register_aggregator("veto_hybrid")
def veto_hybrid(
    utilities: List[List[float]],
    weights: List[float],
    veto_threshold: int = 1,
    **params
) -> Dict[str, Any]:
    """Veto-filtered centroid (minority protection).

    Candidates with too many vetoes are excluded, then centroid on remainder.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    weights : List[float]
        Agent weights
    veto_threshold : int
        Number of vetoes required to exclude candidate (default: 1)

    Returns
    -------
    Dict[str, Any]
        winner: Index of non-vetoed candidate with highest average utility
        scores: Average utilities for non-vetoed candidates

    Properties
    ----------
    - Minority protection via veto power
    - Prevents unacceptable outcomes
    - Combines deontological (veto) and utilitarian (centroid)
    - May result in no winner if all vetoed

    Notes
    -----
    A candidate is vetoed by an agent if their utility is below 0.1 (threshold).
    """
    n_candidates = len(utilities[0])
    veto_counts = [0] * n_candidates

    # Count vetoes (low utility = veto)
    VETO_UTILITY_THRESHOLD = 0.1
    for u in utilities:
        for i in range(n_candidates):
            if u[i] < VETO_UTILITY_THRESHOLD:
                veto_counts[i] += 1

    # Filter candidates by veto threshold
    allowed = [i for i in range(n_candidates) if veto_counts[i] < veto_threshold]

    if not allowed:
        # All candidates vetoed - fall back to majority
        return majority(utilities, weights)

    # Centroid on allowed candidates
    weighted_sums = [0.0] * len(allowed)
    total_weight = sum(weights)

    for u, w in zip(utilities, weights):
        for j, i in enumerate(allowed):
            weighted_sums[j] += u[i] * w

    scores = [s / total_weight for s in weighted_sums]
    winner_j = max(range(len(allowed)), key=lambda j: scores[j])
    winner = allowed[winner_j]

    # Full scores array (non-allowed get -1)
    full_scores = [-1.0 if i not in allowed else scores[allowed.index(i)] for i in range(n_candidates)]

    return {
        'winner': winner,
        'scores': full_scores,
        'metadata': {
            'veto_threshold': veto_threshold,
            'veto_counts': veto_counts,
            'allowed_candidates': allowed,
            'filtered_scores': scores
        }
    }
