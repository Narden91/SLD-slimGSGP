"""
Schema definitions, feature taxonomy, and data models for SLD (Specific Learning Disorder) online handwriting dataset.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import torch


class Cohort(str, Enum):
    """Primary school grade cohorts."""
    YEAR_1 = "1year"
    YEAR_2 = "2year"
    YEAR_3 = "3year"


class FeatureCategory(str, Enum):
    """Kinematic and dynamic feature categories."""
    PRESSURE = "pressure"
    TILT = "tilt"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    KINEMATICS = "kinematics"
    EVENTS = "events"


# --- Feature Definitions ---
PRESSURE_FEATURES: List[str] = [
    "mean:pressure",
    "std:pressure",
    "iqr:pressure",
    "number_of_changes_in_pressure_profile",
]

TILT_FEATURES: List[str] = [
    "mean:tilt(on-surface)",
    "std:tilt(on-surface)",
    "iqr:tilt(on-surface)",
]

TEMPORAL_FEATURES: List[str] = [
    "writing_duration(on-surface)",
    "writing_duration_overall",
    "mean:stroke_duration(on-surface)",
    "in_air_movement_time",
    "in_air_on_paper_ratio",
]

SPATIAL_FEATURES: List[str] = [
    "writing_width(on-surface)",
    "writing_height(on-surface)",
    "mean:stroke_width(on-surface)",
    "std:stroke_width(on-surface)",
    "iqr:stroke_width(on-surface)",
    "mean:stroke_height(on-surface)",
    "std:stroke_height(on-surface)",
    "iqr:stroke_height(on-surface)",
    "mean:stroke_length(on-surface)",
    "number_of_changes_in_x_profile",
    "number_of_changes_in_y_profile",
]

KINEMATICS_FEATURES: List[str] = [
    "mean:velocity:axis-xy(on-surface)",
    "std:velocity:axis-xy(on-surface)",
    "mean:acceleration:axis-xy(on-surface)",
    "std:acceleration:axis-xy(on-surface)",
    "mean:jerk:axis-xy(on-surface)",
    "average_normalized_jerk",
]

EVENT_FEATURES: List[str] = [
    "number_of_interruptions",
]

# All 30 pure numerical kinematic features extracted from stylus
KINEMATIC_FEATURES: List[str] = (
    PRESSURE_FEATURES
    + TILT_FEATURES
    + TEMPORAL_FEATURES
    + SPATIAL_FEATURES
    + KINEMATICS_FEATURES
    + EVENT_FEATURES
)

DEMOGRAPHIC_COLUMNS: List[str] = [
    "Gender",
    "Age",
    "Dominant_Hand",
]

METADATA_COLUMNS: List[str] = [
    "Id",
    "Task",
]

TARGET_COLUMN: str = "Label"

ALL_COLUMNS: List[str] = (
    ["Id"]
    + KINEMATIC_FEATURES
    + DEMOGRAPHIC_COLUMNS
    + ["Label", "Task"]
)

FEATURE_CATEGORIES_MAP: Dict[FeatureCategory, List[str]] = {
    FeatureCategory.PRESSURE: PRESSURE_FEATURES,
    FeatureCategory.TILT: TILT_FEATURES,
    FeatureCategory.TEMPORAL: TEMPORAL_FEATURES,
    FeatureCategory.SPATIAL: SPATIAL_FEATURES,
    FeatureCategory.KINEMATICS: KINEMATICS_FEATURES,
    FeatureCategory.EVENTS: EVENT_FEATURES,
}

# Task Performance Labels: 0 best, 2 worst; -1 indicates subject not present
LABEL_DESCRIPTION: Dict[int, str] = {
    0: "Best Performance (Score 0)",
    1: "Intermediate Performance (Score 1)",
    2: "Worst Performance (Score 2)",
    -1: "Absent / Not Present",
}


@dataclass
class CohortInfo:
    """Metadata summary of a single grade cohort."""
    cohort: Cohort
    task_count: int
    subject_count: int
    total_records: int
    tasks: List[int] = field(default_factory=list)
    label_distribution: Dict[int, int] = field(default_factory=dict)


@dataclass
class DatasetSummary:
    """High-level summary of the entire dataset directory."""
    data_dir: Path
    cohorts: Dict[str, CohortInfo]
    total_files: int
    total_records: int
    num_kinematic_features: int
    unique_total_subjects: int


@dataclass
class TaskData:
    """Container for a loaded task dataset."""
    cohort: str
    task_id: int
    df: pd.DataFrame
    num_samples: int
    num_features: int
    label_counts: Dict[int, int]

    def __repr__(self) -> str:
        return f"<TaskData cohort={self.cohort} task={self.task_id} samples={self.num_samples} features={self.num_features}>"


@dataclass
class PreparedTensors:
    """Container for model-ready PyTorch tensors."""
    X: torch.Tensor
    y: torch.Tensor
    feature_names: List[str]
    subject_ids: torch.Tensor
    task_ids: torch.Tensor
    cohorts: List[str]

    @property
    def shape(self) -> Tuple[torch.Size, torch.Size]:
        """Return (X.shape, y.shape)."""
        return self.X.shape, self.y.shape

    def __repr__(self) -> str:
        return f"<PreparedTensors X={tuple(self.X.shape)} y={tuple(self.y.shape)} dtype={self.X.dtype}>"
