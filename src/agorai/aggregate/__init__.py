"""Pure mathematical aggregation module.

Provides aggregation methods from social choice theory, welfare economics,
and game theory for combining agent utilities into collective decisions.
"""

from agorai.aggregate.core import aggregate, list_methods, prepare_utilities, AGGREGATOR_REGISTRY
from agorai.aggregate.methods import (
    # Social Choice
    majority,
    weighted_plurality,
    borda,
    schulze_condorcet,
    approval_voting,
    supermajority,
    # Welfare Economics
    maximin,
    atkinson,
    # Machine Learning/Statistics
    score_centroid,
    robust_median,
    consensus,
    # Game Theory
    quadratic_voting,
    nash_bargaining,
    veto_hybrid,
)

__all__ = [
    # Main API
    "aggregate",
    "list_methods",
    "prepare_utilities",
    "AGGREGATOR_REGISTRY",
    # Individual methods
    "majority",
    "weighted_plurality",
    "borda",
    "schulze_condorcet",
    "approval_voting",
    "supermajority",
    "maximin",
    "atkinson",
    "score_centroid",
    "robust_median",
    "consensus",
    "quadratic_voting",
    "nash_bargaining",
    "veto_hybrid",
]
