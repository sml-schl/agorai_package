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
from agorai.aggregate.alias_loader import (
    get_method_from_alias,
    load_aliases,
    list_aliases,
    add_custom_alias,
    get_aliases_by_property,
)
from agorai.aggregate.property_analysis import (
    recommend_mechanism,
    verify_anonymity,
    measure_outlier_resistance,
    compare_mechanisms_on_scenario,
    get_mechanism_properties,
    compare_mechanisms_properties,
    PROPERTY_MATRIX,
)

__all__ = [
    # Main API
    "aggregate",
    "list_methods",
    "prepare_utilities",
    "AGGREGATOR_REGISTRY",
    # Alias API
    "get_method_from_alias",
    "load_aliases",
    "list_aliases",
    "add_custom_alias",
    "get_aliases_by_property",
    # Property Analysis API
    "recommend_mechanism",
    "verify_anonymity",
    "measure_outlier_resistance",
    "compare_mechanisms_on_scenario",
    "get_mechanism_properties",
    "compare_mechanisms_properties",
    "PROPERTY_MATRIX",
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
