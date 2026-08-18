"""
SLD-slimGSGP: Research framework for online handwriting kinematics and SLIM-GSGP Genetic Programming.
"""

# Core schema & types
from src.core.schema import (
    ALL_COLUMNS,
    Cohort,
    CohortInfo,
    DatasetSummary,
    DEMOGRAPHIC_COLUMNS,
    EVENT_FEATURES,
    FEATURE_CATEGORIES_MAP,
    FeatureCategory,
    KINEMATIC_FEATURES,
    KINEMATICS_FEATURES,
    LABEL_DESCRIPTION,
    METADATA_COLUMNS,
    PRESSURE_FEATURES,
    PreparedTensors,
    SPATIAL_FEATURES,
    TARGET_COLUMN,
    TaskData,
    TEMPORAL_FEATURES,
    TILT_FEATURES,
)

# Data loading & preprocessing
from src.data.loader import (
    get_available_cohorts,
    get_available_tasks,
    get_dataset_summary,
    get_default_data_dir,
    get_task_data,
    load_all,
    load_cohort,
    load_task,
)
from src.data.preprocessor import (
    clean_and_impute_demographics,
    encode_demographics,
    prepare_tensors,
    train_val_test_split,
)

# Analytics & Statistics
from src.analysis.stats import (
    compute_category_stats,
    compute_demographics_summary,
    compute_feature_stats,
    compute_label_distribution,
    get_subject_trajectory,
)

# Visualization
from src.visualization.viewer import (
    render_dataset_overview,
    render_feature_categories,
    render_label_distribution,
    render_subject_profile,
    render_task_summary,
    render_tensor_summary,
)

# Genetic Programming
from src.genetic.runner import (
    SlimConfig,
    SlimResult,
    run_slim_experiment,
)

__all__ = [
    # Schema
    "Cohort",
    "FeatureCategory",
    "CohortInfo",
    "DatasetSummary",
    "TaskData",
    "PreparedTensors",
    "KINEMATIC_FEATURES",
    "PRESSURE_FEATURES",
    "TILT_FEATURES",
    "TEMPORAL_FEATURES",
    "SPATIAL_FEATURES",
    "KINEMATICS_FEATURES",
    "EVENT_FEATURES",
    "DEMOGRAPHIC_COLUMNS",
    "METADATA_COLUMNS",
    "ALL_COLUMNS",
    "TARGET_COLUMN",
    "LABEL_DESCRIPTION",
    "FEATURE_CATEGORIES_MAP",
    # Loader
    "get_default_data_dir",
    "get_available_cohorts",
    "get_available_tasks",
    "load_task",
    "load_cohort",
    "load_all",
    "get_task_data",
    "get_dataset_summary",
    # Preprocessor
    "clean_and_impute_demographics",
    "encode_demographics",
    "prepare_tensors",
    "train_val_test_split",
    # Stats
    "compute_feature_stats",
    "compute_label_distribution",
    "compute_category_stats",
    "compute_demographics_summary",
    "get_subject_trajectory",
    # Viewer
    "render_dataset_overview",
    "render_label_distribution",
    "render_feature_categories",
    "render_task_summary",
    "render_subject_profile",
    "render_tensor_summary",
    # Genetic
    "SlimConfig",
    "SlimResult",
    "run_slim_experiment",
]
