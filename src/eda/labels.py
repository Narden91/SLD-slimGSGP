"""Nominal label summaries and conservative subject joins."""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from src.core.schema import (
    EDA_RARE_CLASS_THRESHOLD,
    EDA_UNKNOWN_LABEL,
    EDA_VALID_LABELS,
)
from src.eda.io import TaskTable, valid_labels


def label_counts(tables: list[TaskTable]) -> pd.DataFrame:
    """Include absent valid classes, -1, unexpected tokens, and missing labels."""
    rows = []
    for table in tables:
        raw = table.frame["Label"]
        numeric = pd.to_numeric(raw, errors="coerce")
        eligible = int(valid_labels(raw).notna().sum())
        known = (*EDA_VALID_LABELS, EDA_UNKNOWN_LABEL)
        for label in known:
            count = int(numeric.eq(label).sum())
            rows.append(
                dict(
                    cohort=table.cohort,
                    task=table.task,
                    label=str(label),
                    label_kind="valid" if label in EDA_VALID_LABELS else "unresolved",
                    count=count,
                    raw_total=len(raw),
                    eligible_total=eligible,
                    raw_proportion=count / len(raw) if len(raw) else np.nan,
                    eligible_proportion=count / eligible
                    if eligible and label in EDA_VALID_LABELS
                    else np.nan,
                )
            )
        unexpected = raw[~numeric.isin(known)]
        for label, count in unexpected.value_counts(dropna=False).items():
            rows.append(
                dict(
                    cohort=table.cohort,
                    task=table.task,
                    label="<missing>" if pd.isna(label) else str(label),
                    label_kind="unexpected",
                    count=int(count),
                    raw_total=len(raw),
                    eligible_total=eligible,
                    raw_proportion=count / len(raw),
                    eligible_proportion=np.nan,
                )
            )
    return pd.DataFrame(rows)


def rare_class_flags(
    counts: pd.DataFrame, threshold: int = EDA_RARE_CLASS_THRESHOLD
) -> pd.DataFrame:
    valid = counts[counts.label_kind.eq("valid")].copy()
    grouped = valid.groupby(["cohort", "task"])
    valid["smallest_class_count"] = grouped["count"].transform("min")
    valid["majority_prevalence"] = grouped["eligible_proportion"].transform("max")
    valid["support"] = np.select(
        [valid["count"].eq(0), valid["count"].eq(1), valid["count"].lt(threshold)],
        ["absent", "singleton", "very_few"],
        default="at_least_threshold",
    )
    valid["warning_threshold"] = threshold
    return valid[
        [
            "cohort",
            "task",
            "label",
            "count",
            "support",
            "smallest_class_count",
            "majority_prevalence",
            "warning_threshold",
        ]
    ]


def subject_label_matrix(tables: list[TaskTable]) -> pd.DataFrame:
    """Return one cohort's outer join; refuse ambiguous or missing identifiers.

    Matching IDs and demographics only support identity linkage. Researcher
    confirmation of acquisition identifiers is still needed before final folds.
    """
    if not tables or len({t.cohort for t in tables}) != 1:
        raise ValueError("A subject label matrix requires exactly one cohort.")
    columns = {}
    for table in tables:
        ids = table.frame["Id"].astype("string")
        if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError(
                f"Missing or duplicate IDs prevent a safe join in {table.task}."
            )
        if table.task in columns:
            raise ValueError("Duplicate source task keys prevent a safe join.")
        columns[table.task] = pd.Series(
            table.frame.Label.to_numpy(), index=ids, name=table.task
        )
    matrix = pd.concat(columns, axis=1)
    matrix.index.name = "Id"
    return matrix.sort_index()


def subject_label_patterns(matrix: pd.DataFrame) -> pd.DataFrame:
    valid = matrix.apply(valid_labels)
    distinct = valid.nunique(axis=1)
    return pd.DataFrame(
        {
            "distinct_valid_labels": range(len(EDA_VALID_LABELS) + 1),
            "subjects": [
                int(distinct.eq(n).sum()) for n in range(len(EDA_VALID_LABELS) + 1)
            ],
        }
    )


def task_label_agreement(matrix: pd.DataFrame) -> pd.DataFrame:
    """Exact agreement; pairwise denominators exclude all unknown labels."""
    valid = matrix.apply(valid_labels)
    rows = []
    for left, right in combinations_with_replacement(valid.columns, 2):
        eligible = valid[left].notna() & valid[right].notna()
        n = int(eligible.sum())
        agree = int(valid.loc[eligible, left].eq(valid.loc[eligible, right]).sum())
        rows.append(
            dict(
                task_a=left,
                task_b=right,
                eligible_subjects=n,
                agreeing_subjects=agree,
                exact_agreement=agree / n if n else np.nan,
            )
        )
    return pd.DataFrame(rows)
