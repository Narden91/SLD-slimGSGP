"""Robust measurement descriptions; no imputation, scaling, or row deletion."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from src.core.schema import (
    EDA_DURATION_RELATIONSHIPS,
    EDA_EXTREME_IQR_MULTIPLIER,
    EDA_NEAR_CONSTANT_FRACTION,
    EDA_PERCENTILES,
    EDA_SCALE_RATIO_WARNING,
)
from src.eda.io import TaskTable, measurement_masks, numerical_features, valid_labels


@dataclass
class FeatureResult:
    quality: pd.DataFrame
    statistics: pd.DataFrame
    measurements: pd.DataFrame
    decisions: pd.DataFrame


def feature_quality(
    tables: list[TaskTable],
    near_constant_fraction: float = EDA_NEAR_CONSTANT_FRACTION,
    extreme_multiplier: float = EDA_EXTREME_IQR_MULTIPLIER,
) -> FeatureResult:
    """Summarize both raw rows and the provisional valid-label subset.

    Percentiles exclude nonfinite values, which are counted separately. An outer
    IQR fence is descriptive evidence for review, not a measurement error rule.
    """
    if not 0 < near_constant_fraction <= 1 or extreme_multiplier <= 0:
        raise ValueError(
            "Require 0 < near_constant_fraction <= 1 and a positive IQR multiplier."
        )
    quality, statistics, measurements, decisions = [], [], [], []
    for table in tables:
        df = table.frame
        values = numerical_features(df)
        eligible = valid_labels(df["Label"]).notna()
        base = dict(cohort=table.cohort, task=table.task)
        masks = measurement_masks(values)
        for issue, mask in masks.items():
            measurements.append(
                dict(
                    **base,
                    scope="raw",
                    issue=issue,
                    rows=int(mask.sum()),
                    valid_label_rows=int((mask & eligible).sum()),
                    evaluable_rows=len(df),
                    detail="",
                )
            )
        for overall, surface in EDA_DURATION_RELATIONSHIPS:
            if {overall, surface}.issubset(values):
                evaluable = np.isfinite(values[[overall, surface]]).all(axis=1)
                violation = evaluable & values[overall].lt(values[surface])
                measurements.append(
                    dict(
                        **base,
                        scope="raw",
                        issue="overall_duration_below_on_surface",
                        rows=int(violation.sum()),
                        valid_label_rows=int((violation & eligible).sum()),
                        evaluable_rows=int(evaluable.sum()),
                        detail=f"{overall} >= {surface}",
                    )
                )
            else:
                measurements.append(
                    dict(
                        **base,
                        scope="raw",
                        issue="duration_check_unavailable",
                        rows=0,
                        valid_label_rows=0,
                        evaluable_rows=0,
                        detail=f"Required: {overall}, {surface}",
                    )
                )
        for scope, subset in (
            ("raw", values),
            ("valid_labels_only", values.loc[eligible]),
        ):
            finite = subset.where(np.isfinite(subset))
            quantiles = finite.quantile(EDA_PERCENTILES)
            raw_columns = {column: df.loc[subset.index, column] for column in subset}
            scope_quality = []
            for column in subset:
                series = finite[column].dropna()
                missing = int(raw_columns[column].isna().sum())
                invalid = int(
                    (raw_columns[column].notna() & subset[column].isna()).sum()
                )
                mode_fraction = (
                    float(series.value_counts(normalize=True).max())
                    if len(series)
                    else np.nan
                )
                q25, median, q75 = (quantiles.loc[q, column] for q in (0.25, 0.5, 0.75))
                iqr = q75 - q25
                extreme = series.lt(q25 - extreme_multiplier * iqr) | series.gt(
                    q75 + extreme_multiplier * iqr
                )
                item = dict(
                    **base,
                    scope=scope,
                    feature=column,
                    rows=len(subset),
                    finite_values=len(series),
                    missing=missing,
                    invalid_numeric=invalid,
                    infinite=int(np.isinf(subset[column]).sum()),
                    zero=int(subset[column].eq(0).sum()),
                    zero_fraction=float(subset[column].eq(0).mean())
                    if len(subset)
                    else np.nan,
                    constant=bool(len(series) and series.nunique() == 1),
                    near_constant=bool(
                        len(series) and mode_fraction >= near_constant_fraction
                    ),
                    modal_fraction=mode_fraction,
                    near_constant_threshold=near_constant_fraction,
                    potential_extreme_values=int(extreme.sum()),
                    iqr_fence_multiplier=extreme_multiplier,
                    zero_iqr=bool(iqr == 0),
                )
                quality.append(item)
                scope_quality.append(item)
                statistics.append(
                    dict(
                        **base,
                        scope=scope,
                        feature=column,
                        minimum=series.min(),
                        maximum=series.max(),
                        median=median,
                        iqr=iqr,
                        **{
                            f"p{round(q * 100):02d}": quantiles.loc[q, column]
                            for q in EDA_PERCENTILES
                        },
                    )
                )
            for left, right in combinations(subset.columns, 2):
                # Compare raw tokens so two malformed numeric columns cannot look identical after coercion.
                if len(subset) and raw_columns[left].equals(raw_columns[right]):
                    measurements.append(
                        dict(
                            **base,
                            issue="duplicate_feature_columns",
                            scope=scope,
                            rows=len(subset),
                            valid_label_rows=int(eligible.loc[subset.index].sum()),
                            evaluable_rows=len(subset),
                            detail=f"{left} == {right}",
                        )
                    )
            magnitudes = finite.abs().median()
            nonzero = magnitudes[magnitudes.gt(0)]
            ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) else np.nan
            decisions.append(
                dict(
                    **base,
                    scope=scope,
                    nonfinite_or_invalid_cells=sum(
                        r["missing"] + r["invalid_numeric"] + r["infinite"]
                        for r in scope_quality
                    ),
                    constant_features=sum(r["constant"] for r in scope_quality),
                    near_constant_features=sum(
                        r["near_constant"] for r in scope_quality
                    ),
                    finite_median_magnitude_ratio=ratio,
                    median_imputation_may_be_needed=any(
                        r["missing"] + r["invalid_numeric"] + r["infinite"]
                        for r in scope_quality
                    ),
                    robust_scaling_worth_testing=bool(
                        ratio >= EDA_SCALE_RATIO_WARNING
                        or any(r["potential_extreme_values"] for r in scope_quality)
                    ),
                    scale_ratio_warning_threshold=EDA_SCALE_RATIO_WARNING,
                    manual_review_rows=int(
                        (
                            masks["all_zero_measurements"]
                            | masks["no_finite_measurements"]
                        )
                        .loc[subset.index]
                        .sum()
                    ),
                )
            )
    return FeatureResult(
        pd.DataFrame(quality),
        pd.DataFrame(statistics),
        pd.DataFrame(measurements),
        pd.DataFrame(decisions),
    )
