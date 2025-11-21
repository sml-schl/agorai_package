"""Core aggregation functionality and API.

Provides the main `aggregate()` function and utility normalization logic.
"""

from typing import List, Dict, Optional, Any, Union
import statistics

# Global registry for aggregation methods
AGGREGATOR_REGISTRY: Dict[str, Any] = {}


def register_aggregator(name: str):
    """Decorator to register an aggregation method."""
    def _wrap(func):
        AGGREGATOR_REGISTRY[name] = func
        return func
    return _wrap


def prepare_utilities(
    utilities: Union[List[List[float]], List[float]],
    weights: Optional[List[float]] = None,
    normalize: bool = True
) -> Dict[str, Any]:
    """Prepare and normalize utility matrix for aggregation.

    Parameters
    ----------
    utilities : List[List[float]] or List[float]
        Utility matrix where utilities[i][j] is agent i's utility for candidate j.
        Can also be a 1D list for single agent case.
    weights : Optional[List[float]]
        Per-agent weights (default: uniform weights).
    normalize : bool
        Whether to normalize utilities to [0, 1] range per agent (default: True).

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'utilities': Normalized utility matrix (n_agents × n_candidates)
        - 'weights': Agent weights
        - 'n_agents': Number of agents
        - 'n_candidates': Number of candidates

    Notes
    -----
    Normalization uses min-max scaling per agent to ensure utilities are in [0, 1].
    This prevents agents with extreme utilities from dominating the aggregation.

    Examples
    --------
    >>> utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
    >>> data = prepare_utilities(utilities)
    >>> data['utilities']
    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    """
    # Handle single agent case
    if utilities and not isinstance(utilities[0], (list, tuple)):
        utilities = [utilities]

    n_agents = len(utilities)
    n_candidates = len(utilities[0]) if utilities else 0

    # Validate input
    if n_agents == 0 or n_candidates == 0:
        raise ValueError("utilities must be non-empty")

    for i, u in enumerate(utilities):
        if len(u) != n_candidates:
            raise ValueError(f"Agent {i} has {len(u)} utilities but expected {n_candidates}")

    # Normalize utilities per agent
    if normalize:
        normalized = []
        for u in utilities:
            minv = min(u)
            maxv = max(u)
            if maxv - minv > 1e-12:
                u_norm = [(x - minv) / (maxv - minv) for x in u]
            else:
                # All utilities equal - set to 0.5 (neutral)
                u_norm = [0.5 for _ in u]
            normalized.append(u_norm)
        utilities = normalized

    # Set default weights
    if weights is None:
        weights = [1.0] * n_agents
    elif len(weights) != n_agents:
        raise ValueError(f"weights length {len(weights)} doesn't match n_agents {n_agents}")

    return {
        'utilities': utilities,
        'weights': weights,
        'n_agents': n_agents,
        'n_candidates': n_candidates
    }


def aggregate(
    utilities: Union[List[List[float]], List[float]],
    method: str = "majority",
    weights: Optional[List[float]] = None,
    normalize: bool = True,
    **method_params
) -> Dict[str, Any]:
    """Aggregate agent utilities into a collective decision.

    Parameters
    ----------
    utilities : List[List[float]] or List[float]
        Utility matrix where utilities[i][j] is agent i's utility for candidate j.
        Can also be a 1D list for single agent case.
    method : str
        Aggregation method name (default: "majority").
        See `list_methods()` for available options.
    weights : Optional[List[float]]
        Per-agent weights for weighted aggregation (default: uniform).
    normalize : bool
        Whether to normalize utilities to [0, 1] per agent (default: True).
    **method_params
        Additional parameters specific to the aggregation method.
        Examples: epsilon for atkinson, threshold for approval_voting, etc.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with:
        - 'winner': Index of winning candidate (int)
        - 'scores': Scores for all candidates (List[float])
        - 'method': Name of aggregation method used (str)
        - 'metadata': Additional method-specific information (Dict)

    Raises
    ------
    ValueError
        If method is not registered or utilities are invalid.

    Notes
    -----
    All aggregation methods operate on normalized utilities in [0, 1] by default.
    This ensures fairness across agents with different utility scales.

    The winner is determined by the specific aggregation method's logic:
    - majority: most votes
    - atkinson: highest equally-distributed equivalent utility
    - maximin: highest minimum utility across agents
    - etc.

    Examples
    --------
    >>> # Simple majority voting
    >>> utilities = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    >>> result = aggregate(utilities, method="majority")
    >>> result['winner']
    0

    >>> # Atkinson with inequality aversion
    >>> utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
    >>> result = aggregate(utilities, method="atkinson", epsilon=1.0)
    >>> result['winner']
    1

    References
    ----------
    .. [1] Arrow, K. J. (1951). Social Choice and Individual Values.
    .. [2] Atkinson, A. B. (1970). On the measurement of inequality.
    .. [3] Schulze, M. (2011). A new monotonic, clone-independent, reversal
           symmetric, and condorcet-consistent single-winner election method.
    """
    if method not in AGGREGATOR_REGISTRY:
        available = ", ".join(sorted(AGGREGATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available methods: {available}"
        )

    # Prepare utilities
    data = prepare_utilities(utilities, weights, normalize)

    # Call aggregation method
    aggregator_func = AGGREGATOR_REGISTRY[method]
    result = aggregator_func(
        data['utilities'],
        data['weights'],
        **method_params
    )

    # Ensure consistent return format
    if 'method' not in result:
        result['method'] = method

    return result


def list_methods() -> List[Dict[str, str]]:
    """List all available aggregation methods with descriptions.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing method information:
        - 'name': Method identifier
        - 'category': Method category (Social Choice, Welfare Economics, etc.)
        - 'description': Brief description

    Examples
    --------
    >>> methods = list_methods()
    >>> for m in methods:
    ...     print(f"{m['name']}: {m['description']}")
    """
    # Method metadata
    METHODS = [
        # Social Choice Theory
        {
            'name': 'majority',
            'category': 'Social Choice',
            'description': 'One-agent-one-vote plurality (most votes wins)'
        },
        {
            'name': 'weighted_plurality',
            'category': 'Social Choice',
            'description': 'Plurality with per-agent weights'
        },
        {
            'name': 'borda',
            'category': 'Social Choice',
            'description': 'Borda count positional ranking aggregation'
        },
        {
            'name': 'schulze_condorcet',
            'category': 'Social Choice',
            'description': 'Schulze method for Condorcet-consistent ranking'
        },
        {
            'name': 'approval_voting',
            'category': 'Social Choice',
            'description': 'Agents approve candidates above utility threshold'
        },
        {
            'name': 'supermajority',
            'category': 'Social Choice',
            'description': 'Requires threshold fraction of votes to win'
        },
        # Welfare Economics
        {
            'name': 'maximin',
            'category': 'Welfare Economics',
            'description': 'Rawlsian maximin (maximize minimum utility)'
        },
        {
            'name': 'atkinson',
            'category': 'Welfare Economics',
            'description': 'Atkinson index with parameterizable inequality aversion'
        },
        # Machine Learning/Statistics
        {
            'name': 'score_centroid',
            'category': 'Machine Learning',
            'description': 'Weighted average of normalized utilities'
        },
        {
            'name': 'robust_median',
            'category': 'Machine Learning',
            'description': 'Per-candidate median utility (outlier-resistant)'
        },
        {
            'name': 'consensus',
            'category': 'Machine Learning',
            'description': 'Balance mean utility and consensus (low disagreement)'
        },
        # Game Theory
        {
            'name': 'quadratic_voting',
            'category': 'Game Theory',
            'description': 'Budget-constrained voting with quadratic costs'
        },
        {
            'name': 'nash_bargaining',
            'category': 'Game Theory',
            'description': 'Nash bargaining solution (maximize utility product)'
        },
        {
            'name': 'veto_hybrid',
            'category': 'Game Theory',
            'description': 'Veto-filtered centroid (minority protection)'
        },
    ]

    # Filter to only registered methods
    return [m for m in METHODS if m['name'] in AGGREGATOR_REGISTRY]
