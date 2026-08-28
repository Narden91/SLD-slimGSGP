"""
Statistical aggregation, feature metrics, and distribution computations for SLD handwriting data.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.core.schema import (
    AGE_MAX_VALID,
    AGE_MIN_VALID,
    FEATURE_CATEGORIES_MAP,
    FeatureCategory,
    KINEMATIC_FEATURES,
    LABEL_DESCRIPTION,
    TARGET_COLUMN,
)


def compute_feature_stats(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for kinematic features.

    Returns:
        pd.DataFrame: Table with columns [Feature, Mean, Std, Min, Median, Max, IQR, Missing].
    """
    feature_list = features if features is not None else [f for f in KINEMATIC_FEATURES if f in df.columns]
    records = []

    for feat in feature_list:
        series = pd.to_numeric(df[feat], errors="coerce")
        missing_count = series.isna().sum()
        valid = series.dropna()

        if len(valid) == 0:
            records.append({
                "feature": feat,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "median": np.nan,
                "max": np.nan,
                "iqr": np.nan,
                "missing": missing_count,
            })
            continue

        q25 = valid.quantile(0.25)
        q75 = valid.quantile(0.75)
        iqr = q75 - q25

        records.append({
            "feature": feat,
            "mean": valid.mean(),
            "std": valid.std(),
            "min": valid.min(),
            "median": valid.median(),
            "max": valid.max(),
            "iqr": iqr,
            "missing": missing_count,
        })

    stats_df = pd.DataFrame(records)
    return stats_df


def compute_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute label counts, percentages, and descriptive performance meanings.

    Returns:
        pd.DataFrame: Table with [Label, Description, Count, Percentage].
    """
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame(columns=["label", "description", "count", "percentage"])

    counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    total = len(df)
    records = []

    for label_val, count in counts.items():
        desc = LABEL_DESCRIPTION.get(int(label_val), "Unknown")
        records.append({
            "label": int(label_val),
            "description": desc,
            "count": int(count),
            "percentage": (count / total) * 100.0,
        })

    return pd.DataFrame(records)


def compute_category_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute summary stats grouped by feature categories (Pressure, Tilt, Temporal, etc.).
    """
    category_tables: Dict[str, pd.DataFrame] = {}
    for cat_enum, feat_names in FEATURE_CATEGORIES_MAP.items():
        present_feats = [f for f in feat_names if f in df.columns]
        if present_feats:
            category_tables[cat_enum.value] = compute_feature_stats(df, present_feats)
    return category_tables


def compute_demographics_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute demographics summary (Gender, Dominant Hand, Age).
    """
    summary: Dict[str, Any] = {}

    if "Gender" in df.columns:
        summary["gender_counts"] = df["Gender"].value_counts().to_dict()

    if "Dominant_Hand" in df.columns:
        summary["dominant_hand_counts"] = df["Dominant_Hand"].value_counts().to_dict()

    if "Age" in df.columns:
        valid_age = df["Age"][df["Age"].between(AGE_MIN_VALID, AGE_MAX_VALID, inclusive="neither")]
        if len(valid_age) > 0:
            summary["age_stats"] = {
                "min": float(valid_age.min()),
                "median": float(valid_age.median()),
                "mean": float(valid_age.mean()),
                "max": float(valid_age.max()),
                "valid_count": int(len(valid_age)),
            }

    return summary


def get_subject_trajectory(
    df: pd.DataFrame,
    subject_id: int,
    cohort: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract task-by-task performance trajectory for a specific subject ID.

    Subject IDs are cohort-local (not globally unique), so passing `cohort`
    is required whenever `df` may span more than one cohort — otherwise
    rows from different children can be mixed together. If `cohort` is
    omitted and `df` spans multiple cohorts, results are sorted by
    [Cohort, Task] instead of [Task] alone to keep each cohort contiguous.

    Returns:
        pd.DataFrame: Columns [Task, Cohort, Label, Performance, Duration, Pressure].
    """
    sub_df = df[df["Id"] == subject_id]
    if cohort is not None and "Cohort" in sub_df.columns:
        sub_df = sub_df[sub_df["Cohort"] == cohort]

    multi_cohort = "Cohort" in sub_df.columns and sub_df["Cohort"].nunique() > 1
    sort_cols = ["Cohort", "Task"] if multi_cohort else ["Task"]
    sub_df = sub_df.sort_values(sort_cols).copy()
    if sub_df.empty:
        return pd.DataFrame()

    sub_df["Performance"] = sub_df[TARGET_COLUMN].map(LABEL_DESCRIPTION)
    cols = ["Task", "Cohort", TARGET_COLUMN, "Performance", "writing_duration_overall", "mean:pressure"]
    available_cols = [c for c in cols if c in sub_df.columns]
    return sub_df[available_cols].reset_index(drop=True)
