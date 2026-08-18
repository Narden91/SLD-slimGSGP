"""
Data loader module for the SLD (Specific Learning Disorder) online handwriting dataset.
Fast, typed, and structured loading for individual tasks, cohorts, or the entire dataset.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union
import glob
import os
import re
import pandas as pd

from src.core.schema import (
    ALL_COLUMNS,
    Cohort,
    CohortInfo,
    DatasetSummary,
    KINEMATIC_FEATURES,
    TaskData,
    TARGET_COLUMN,
)


def get_default_data_dir() -> Path:
    """Resolve the default dataset directory relative to project root."""
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if (parent / "pyproject.toml").is_file():
            candidate = parent / "data"
            if candidate.is_dir():
                return candidate
        candidate = parent / "data"
        if candidate.is_dir() and (candidate / "1year").is_dir():
            return candidate
    return Path("data").resolve()


def get_available_cohorts(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Return sorted list of available cohort directory names (e.g. ['1year', '2year', '3year'])."""
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found at: {base_dir}")
    cohorts = [
        d.name for d in base_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(cohorts)


def _extract_task_id(filename: str) -> int:
    """Extract integer task ID from a file name like 'Task_1.csv' or 'Task_21.csv'."""
    match = re.search(r"Task_(\d+)\.csv", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def get_available_tasks(cohort: str, data_dir: Optional[Union[str, Path]] = None) -> List[int]:
    """Return sorted list of task IDs present in the given cohort directory."""
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()
    cohort_dir = base_dir / cohort
    if not cohort_dir.is_dir():
        raise FileNotFoundError(f"Cohort directory not found: {cohort_dir}")

    files = list(cohort_dir.glob("Task_*.csv"))
    tasks = [_extract_task_id(f.name) for f in files if _extract_task_id(f.name) > 0]
    return sorted(tasks)


def load_task(
    cohort: Union[str, Cohort],
    task: int,
    data_dir: Optional[Union[str, Path]] = None,
    filter_absent: bool = False,
) -> pd.DataFrame:
    """
    Load a single task CSV dataset.

    Args:
        cohort: Cohort string (e.g. '1year', '2year', '3year')
        task: Integer task number (e.g. 1 to 21)
        data_dir: Optional custom base data directory
        filter_absent: If True, exclude records where Label == -1 (subject absent)

    Returns:
        pd.DataFrame: Loaded task dataset with 'Cohort' metadata column.
    """
    cohort_str = cohort.value if isinstance(cohort, Cohort) else str(cohort)
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()
    file_path = base_dir / cohort_str / f"Task_{task}.csv"

    if not file_path.is_file():
        raise FileNotFoundError(f"Task file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["Cohort"] = cohort_str

    if filter_absent:
        df = df[df[TARGET_COLUMN] != -1].reset_index(drop=True)

    return df


def load_cohort(
    cohort: Union[str, Cohort],
    data_dir: Optional[Union[str, Path]] = None,
    filter_absent: bool = False,
) -> pd.DataFrame:
    """
    Load and concatenate all task datasets for a specific school cohort.

    Args:
        cohort: Cohort name ('1year', '2year', or '3year')
        data_dir: Optional custom base data directory
        filter_absent: If True, exclude absent subjects (Label == -1)

    Returns:
        pd.DataFrame: Combined DataFrame across all tasks for the cohort.
    """
    cohort_str = cohort.value if isinstance(cohort, Cohort) else str(cohort)
    tasks = get_available_tasks(cohort_str, data_dir=data_dir)
    if not tasks:
        raise ValueError(f"No task files found in cohort: {cohort_str}")

    dfs = [load_task(cohort_str, t, data_dir=data_dir, filter_absent=filter_absent) for t in tasks]
    return pd.concat(dfs, ignore_index=True)


def load_all(
    data_dir: Optional[Union[str, Path]] = None,
    filter_absent: bool = False,
) -> pd.DataFrame:
    """
    Load and concatenate all datasets across all cohorts and tasks.

    Args:
        data_dir: Optional custom base data directory
        filter_absent: If True, exclude absent subjects (Label == -1)

    Returns:
        pd.DataFrame: Unified DataFrame containing all cohorts and tasks.
    """
    cohorts = get_available_cohorts(data_dir=data_dir)
    dfs = [load_cohort(c, data_dir=data_dir, filter_absent=filter_absent) for c in cohorts]
    return pd.concat(dfs, ignore_index=True)


def get_task_data(
    cohort: Union[str, Cohort],
    task: int,
    data_dir: Optional[Union[str, Path]] = None,
    filter_absent: bool = False,
) -> TaskData:
    """Return a typed TaskData container for a given cohort and task."""
    cohort_str = cohort.value if isinstance(cohort, Cohort) else str(cohort)
    df = load_task(cohort_str, task, data_dir=data_dir, filter_absent=filter_absent)
    label_counts = df[TARGET_COLUMN].value_counts().to_dict()

    return TaskData(
        cohort=cohort_str,
        task_id=task,
        df=df,
        num_samples=len(df),
        num_features=len(KINEMATIC_FEATURES),
        label_counts=label_counts,
    )


def get_dataset_summary(data_dir: Optional[Union[str, Path]] = None) -> DatasetSummary:
    """
    Generate a comprehensive structural summary of the entire dataset directory.

    Returns:
        DatasetSummary: Aggregated metadata, cohort info, file counts, and subject numbers.
    """
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()
    cohort_names = get_available_cohorts(base_dir)

    cohort_summaries: Dict[str, CohortInfo] = {}
    total_files = 0
    total_records = 0
    all_subject_ids = set()

    for c_name in cohort_names:
        tasks = get_available_tasks(c_name, base_dir)
        task_count = len(tasks)
        total_files += task_count

        # Load cohort data to calculate records and label distribution
        df_cohort = load_cohort(c_name, base_dir, filter_absent=False)
        rec_count = len(df_cohort)
        total_records += rec_count

        subjects = set(df_cohort["Id"].unique())
        all_subject_ids.update(f"{c_name}_{s}" for s in subjects)

        label_dist = df_cohort[TARGET_COLUMN].value_counts().to_dict()

        cohort_enum = Cohort(c_name) if c_name in [e.value for e in Cohort] else None
        cohort_summaries[c_name] = CohortInfo(
            cohort=cohort_enum or c_name,
            task_count=task_count,
            subject_count=len(subjects),
            total_records=rec_count,
            tasks=tasks,
            label_distribution=label_dist,
        )

    return DatasetSummary(
        data_dir=base_dir,
        cohorts=cohort_summaries,
        total_files=total_files,
        total_records=total_records,
        num_kinematic_features=len(KINEMATIC_FEATURES),
        unique_total_subjects=len(all_subject_ids),
    )
