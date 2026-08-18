"""
Statistical analytics and trajectory evaluation for SLD handwriting data.
"""

from src.analysis.stats import (
    compute_category_stats,
    compute_demographics_summary,
    compute_feature_stats,
    compute_label_distribution,
    get_subject_trajectory,
)

__all__ = [
    "compute_feature_stats",
    "compute_label_distribution",
    "compute_category_stats",
    "compute_demographics_summary",
    "get_subject_trajectory",
]
