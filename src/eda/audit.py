"""Aggregate schema, identity, coverage, and measurement checks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.schema import (
    AGE_MAX_VALID,
    AGE_MIN_VALID,
    CSV_DTYPES,
    DEMOGRAPHIC_COLUMNS,
    DOMINANT_HAND_VALUES,
    EDA_EXPECTED_TASKS,
    EDA_UNKNOWN_LABEL,
    EDA_VALID_LABELS,
    GENDER_VALUES,
)
from src.eda.io import (
    TaskTable,
    cohort_tables,
    filename_task_id,
    measurement_masks,
    numerical_features,
)


@dataclass
class AuditResult:
    inventory: pd.DataFrame
    schema: pd.DataFrame
    flags: pd.DataFrame
    coverage: pd.DataFrame
    cohorts: pd.DataFrame


def audit_tasks(tables: list[TaskTable]) -> AuditResult:
    """Keep flags aggregate: no subject identifiers or raw records are exported."""
    inventory, schema, flags, coverage, summaries = [], [], [], [], []

    def flag(cohort: str, task: str, issue: str, count: int, detail: str = "") -> None:
        if count:
            flags.append(
                dict(
                    cohort=cohort,
                    task=task,
                    issue=issue,
                    count=int(count),
                    detail=detail,
                )
            )

    for cohort, group in cohort_tables(tables).items():
        sets = []
        for table in group:
            ids = table.frame.get("Id", pd.Series(dtype="string")).dropna()
            sets.append(set(ids[ids.astype("string").str.strip().ne("")]))
        union = set().union(*sets)
        common = set.intersection(*sets)
        expected = EDA_EXPECTED_TASKS.get(cohort)
        if expected is not None and len(group) != expected:
            flag(
                cohort,
                "all",
                "unexpected_task_count",
                1,
                f"Observed {len(group)} files; repository expectation is {expected}.",
            )
        declarations: dict[float, list[str]] = {}
        for table, subjects in zip(group, sets):
            df, task = table.frame, table.task
            values = numerical_features(df)
            masks = measurement_masks(values)
            ids = df.get("Id", pd.Series(pd.NA, index=df.index, dtype="string"))
            blank_ids = ids.isna() | ids.astype("string").str.strip().eq("").fillna(
                False
            )
            duplicates = ids[~blank_ids].duplicated(keep=False).sum()
            inventory.append(
                dict(
                    cohort=cohort,
                    task=task,
                    rows=len(df),
                    unique_subjects=len(subjects),
                    columns=len(df.columns),
                    features=values.shape[1],
                    missing_ids=int(blank_ids.sum()),
                    duplicate_id_rows=int(duplicates),
                    nan_cells=int(df.isna().sum().sum()),
                    infinite_feature_cells=int(np.isinf(values.to_numpy()).sum()),
                    all_zero_rows=int(masks["all_zero_measurements"].sum()),
                    no_finite_measurement_rows=int(
                        masks["no_finite_measurements"].sum()
                    ),
                )
            )
            coverage.append(
                dict(
                    cohort=cohort,
                    task=task,
                    subjects=len(subjects),
                    union_subjects=len(union),
                    common_subjects=len(common),
                    missing_from_union=len(union - subjects),
                    matches_first_task=subjects == sets[0],
                )
            )
            flag(cohort, task, "missing_id", blank_ids.sum())
            flag(cohort, task, "duplicate_id_rows", duplicates)
            flag(cohort, task, "incomplete_subject_coverage", len(union - subjects))
            flag(cohort, task, "nan_cells", df.isna().sum().sum())
            flag(
                cohort,
                task,
                "infinite_feature_cells",
                np.isinf(values.to_numpy()).sum(),
            )
            for issue, mask in masks.items():
                flag(cohort, task, issue, mask.sum())
            for column in sorted(set(CSV_DTYPES) | set(df.columns)):
                present = column in df
                expected_type = CSV_DTYPES.get(column, "unrecognized_column")
                invalid = 0
                if present and expected_type in ("int64", "float64"):
                    parsed = pd.to_numeric(df[column], errors="coerce")
                    invalid = (df[column].notna() & parsed.isna()).sum()
                    if expected_type == "int64":
                        invalid += (
                            parsed.notna()
                            & (~np.isfinite(parsed) | parsed.mod(1).ne(0))
                        ).sum()
                schema.append(
                    dict(
                        cohort=cohort,
                        task=task,
                        column=column,
                        present=present,
                        expected_dtype=expected_type,
                        invalid_numeric_cells=int(invalid),
                    )
                )
                flag(cohort, task, "missing_column", int(not present), column)
                flag(
                    cohort,
                    task,
                    "unrecognized_column",
                    int(column not in CSV_DTYPES),
                    column,
                )
                flag(cohort, task, "invalid_numeric_cells", invalid, column)
            task_values = pd.to_numeric(
                df.get("Task", pd.Series(dtype=float)), errors="coerce"
            )
            declared = task_values.dropna().unique()
            for value in declared:
                declarations.setdefault(value, []).append(task)
            filename_id = filename_task_id(table.source)
            flag(cohort, task, "unrecognized_task_filename", int(filename_id is None))
            flag(cohort, task, "multiple_task_values", int(len(declared) > 1))
            if filename_id is not None and "Task" in df:
                flag(
                    cohort,
                    task,
                    "task_filename_mismatch",
                    (task_values.ne(filename_id) | task_values.isna()).sum(),
                )
            if {"Id", "Task"}.issubset(df):
                flag(
                    cohort,
                    task,
                    "duplicate_subject_task_rows",
                    df.loc[~blank_ids, ["Id", "Task"]].duplicated(keep=False).sum(),
                )
            if "Label" in df:
                labels = pd.to_numeric(df.Label, errors="coerce")
                flag(
                    cohort,
                    task,
                    "unknown_label_minus_one",
                    labels.eq(EDA_UNKNOWN_LABEL).sum(),
                    "Meaning unresolved; retained separately from classification labels.",
                )
                flag(
                    cohort,
                    task,
                    "unexpected_or_missing_label",
                    (~labels.isin((*EDA_VALID_LABELS, EDA_UNKNOWN_LABEL))).sum(),
                )
            if "Age" in df:
                age = pd.to_numeric(df.Age, errors="coerce")
                invalid = ~age.between(
                    AGE_MIN_VALID, AGE_MAX_VALID, inclusive="neither"
                )
                for value, count in (
                    df.loc[invalid, "Age"].value_counts(dropna=False).items()
                ):
                    flag(
                        cohort,
                        task,
                        "implausible_or_missing_age",
                        count,
                        f"Age={value!r}",
                    )
            for column, allowed in (
                ("Gender", GENDER_VALUES),
                ("Dominant_Hand", DOMINANT_HAND_VALUES),
            ):
                if column in df:
                    invalid = ~df[column].isin(allowed)
                    for value, count in (
                        df.loc[invalid, column].value_counts(dropna=False).items()
                    ):
                        flag(
                            cohort,
                            task,
                            "unrecognized_or_missing_category",
                            count,
                            f"{column}={value!r}",
                        )
        for value, sources in declarations.items():
            flag(
                cohort,
                "all",
                "task_identifier_in_multiple_files",
                int(len(sources) > 1),
                f"Declared task {value:g} occurs in {len(sources)} files.",
            )
        combined = pd.concat(
            [t.frame.assign(_source=t.source) for t in group], ignore_index=True
        )
        if {"Id", "Task"}.issubset(combined):
            source_counts = combined.groupby(["Id", "Task"])["_source"].transform(
                "nunique"
            )
            flag(
                cohort,
                "all",
                "cross_file_duplicate_subject_task_rows",
                source_counts.gt(1).sum(),
            )
        conflicts = {}
        for column in DEMOGRAPHIC_COLUMNS:
            count = 0
            if {"Id", column}.issubset(combined):
                count = int(
                    combined.groupby("Id")[column].nunique(dropna=False).gt(1).sum()
                )
            conflicts[column] = count
            flag(cohort, "all", "cross_task_demographic_conflict", count, column)
        summaries.append(
            dict(
                cohort=cohort,
                task_files=len(group),
                expected_tasks=expected,
                rows=sum(len(t.frame) for t in group),
                subjects=len(union),
                subjects_in_every_task=len(common),
                identical_subject_sets=all(s == sets[0] for s in sets),
                **{f"{c}_conflicting_subjects": n for c, n in conflicts.items()},
            )
        )
    return AuditResult(
        pd.DataFrame(inventory),
        pd.DataFrame(schema),
        pd.DataFrame(flags, columns=["cohort", "task", "issue", "count", "detail"]),
        pd.DataFrame(coverage),
        pd.DataFrame(summaries),
    )
