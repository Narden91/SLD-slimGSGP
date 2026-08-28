"""
Unit tests for analytics and stats module.
"""

import pytest
import pandas as pd
from src.data.loader import load_all, load_cohort, load_task
from src.analysis.stats import (
    compute_category_stats,
    compute_demographics_summary,
    compute_feature_stats,
    compute_label_distribution,
    get_subject_trajectory,
)


def test_compute_feature_stats():
    df = load_task("1year", 1)
    stats_df = compute_feature_stats(df)

    assert len(stats_df) == 30
    assert "mean" in stats_df.columns
    assert "std" in stats_df.columns
    assert "median" in stats_df.columns
    assert not stats_df["mean"].isna().all()


def test_compute_label_distribution():
    df = load_cohort("1year")
    dist_df = compute_label_distribution(df)

    assert set(dist_df["label"].unique()) == {-1, 0, 1, 2}
    assert pytest.approx(dist_df["percentage"].sum(), 0.01) == 100.0


def test_compute_category_stats():
    df = load_task("1year", 1)
    cat_stats = compute_category_stats(df)

    assert "pressure" in cat_stats
    assert "tilt" in cat_stats
    assert "temporal" in cat_stats
    assert "spatial" in cat_stats
    assert "kinematics" in cat_stats
    assert "events" in cat_stats


def test_get_subject_trajectory():
    df_1y = load_cohort("1year")
    traj = get_subject_trajectory(df_1y, subject_id=1)

    assert len(traj) == 21  # 21 tasks in 1year
    assert "Task" in traj.columns
    assert "Label" in traj.columns
    assert "Performance" in traj.columns


def test_subject_trajectory_is_cohort_scoped():
    # Subject IDs are cohort-local: without scoping, this would mix
    # different children from 1year/2year/3year together.
    traj = get_subject_trajectory(load_all(), subject_id=1, cohort="1year")
    assert len(traj) == 21
    assert set(traj["Cohort"].unique()) == {"1year"}
