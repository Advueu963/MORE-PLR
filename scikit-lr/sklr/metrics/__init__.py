"""
The :mod:`sklr.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from .label_ranking import kendall_distance, tau_score
from .partial_label_ranking import penalized_kendall_distance, tau_x_score


# =============================================================================
# Module public objects
# =============================================================================
__all__ = ["kendall_distance", "tau_score", "penalized_kendall_distance", "tau_x_score"]  # noqa
