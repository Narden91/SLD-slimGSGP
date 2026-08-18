"""
Unit tests for data loader module.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import (
    get_available_cohorts,
    get_available_tasks,
    get_dataset_summary,
    load_all,
    load_cohort,
    load_task,
)
from src.core.schema import KINEMATIC_FEATURES, TARGET_COLUMN


def test_get_available_cohorts():
    cohorts = get_available_cohorts()
    assert "1year" in cohorts
    assert "2year" in cohorts
    assert "3year" in cohorts
    assert len(cohorts) == 3


def test_get_available_tasks():
    tasks_1y = get_available_tasks("1year")
    tasks_2y = get_available_tasks("2year")
    tasks_3y = get_available_tasks("3year")

    assert len(tasks_1y) == 21
    assert len(tasks_2y) == 21
    assert len(tasks_3y) == 18
    assert tasks_1y[0] == 1
    assert tasks_1y[-1] == 21


def test_load_single_task():
    df = load_task("1year", 1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 140
    assert "Cohort" in df.columns
    assert df["Cohort"].iloc[0] == "1year"
    for feat in KINEMATIC_FEATURES:
        assert feat in df.columns
    assert TARGET_COLUMN in df.columns


def test_load_single_task_filter_absent():
    df_raw = load_task("1year", 1, filter_absent=False)
    df_filtered = load_task("1year", 1, filter_absent=True)
    assert len(df_filtered) < len(df_raw)
    assert (df_filtered[TARGET_COLUMN] == -1).sum() == 0


def test_load_cohort():
    df_1y = load_cohort("1year")
    assert len(df_1y) == 2935
    assert df_1y["Task"].nunique() == 21


def test_load_all():
    df_all = load_all()
    # 2935 (1year) + 3045 (2year) + 2610 (3year) = 8590
    assert len(df_all) == 8590
    assert set(df_all["Cohort"].unique()) == {"1year", "2year", "3year"}


def test_get_dataset_summary():
    summary = get_dataset_summary()
    assert summary.total_files == 60
    assert summary.total_records == 8590
    assert summary.num_kinematic_features == 30
    assert "1year" in summary.cohorts
    assert "2year" in summary.cohorts
    assert "3year" in summary.cohorts
