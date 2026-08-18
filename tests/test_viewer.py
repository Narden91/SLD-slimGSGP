"""
Unit tests for sld_data.viewer module.
"""

import pytest
from sld_data.loader import get_dataset_summary, load_cohort, load_task
from sld_data.preprocessor import prepare_tensors
from sld_data.viewer import (
    render_dataset_overview,
    render_feature_categories,
    render_label_distribution,
    render_subject_profile,
    render_task_summary,
    render_tensor_summary,
)


def test_render_dataset_overview():
    summary = get_dataset_summary()
    render_dataset_overview(summary)


def test_render_label_distribution():
    df = load_task("1year", 1)
    render_label_distribution(df)


def test_render_feature_categories():
    df = load_task("1year", 1)
    render_feature_categories(df=df)
    render_feature_categories(df=None)


def test_render_task_summary():
    df = load_task("1year", 1)
    render_task_summary("1year", 1, df)


def test_render_subject_profile():
    df = load_cohort("1year")
    render_subject_profile(df, subject_id=1)


def test_render_tensor_summary():
    df = load_task("1year", 1, filter_absent=True)
    tensors = prepare_tensors(df, feature_set="kinematic")
    render_tensor_summary(tensors)
