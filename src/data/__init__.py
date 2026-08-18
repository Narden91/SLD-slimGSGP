"""
Data loading and preprocessing package for SLD handwriting kinematics.
"""

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

__all__ = [
    "get_default_data_dir",
    "get_available_cohorts",
    "get_available_tasks",
    "load_task",
    "load_cohort",
    "load_all",
    "get_task_data",
    "get_dataset_summary",
    "clean_and_impute_demographics",
    "encode_demographics",
    "prepare_tensors",
    "train_val_test_split",
]
