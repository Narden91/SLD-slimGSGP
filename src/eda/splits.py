"""Provisional common-subject allocation and diagnostics, never final CV."""

import numpy as np
import pandas as pd

from src.core.schema import (
    DEFAULT_SEED,
    EDA_CANDIDATE_SEEDS,
    EDA_FOLD_COUNTS,
    EDA_MIN_FOLD_SUPPORT,
    EDA_VALID_LABELS,
)
from src.eda.io import TaskTable, cohort_tables, valid_labels
from src.eda.labels import subject_label_matrix


def theoretical_feasibility(
    tables: list[TaskTable],
    fold_counts: tuple[int, ...] = EDA_FOLD_COUNTS,
    min_support: int = EDA_MIN_FOLD_SUPPORT,
) -> pd.DataFrame:
    """Necessary support conditions use distinct subjects, never duplicate rows."""
    rows = []
    for cohort, group in cohort_tables(tables).items():
        matrix = subject_label_matrix(group).apply(valid_labels)
        for task in matrix:
            for k in fold_counts:
                if k < 2:
                    raise ValueError("Fold counts must be at least two.")
                for label in EDA_VALID_LABELS:
                    count = int(matrix[task].eq(label).sum())
                    rows.append(
                        dict(
                            cohort=cohort,
                            task=task,
                            folds=k,
                            label=label,
                            subject_count=count,
                            class_absent=count == 0,
                            every_test_fold_coverage_possible=count >= k,
                            unavoidable_missing_test_folds=max(0, k - count),
                            balanced_test_support_floor=count // k,
                            balanced_training_support_floor=count
                            - int(np.ceil(count / k)),
                            tiny_test_support=count // k < min_support,
                            tiny_training_support=count - int(np.ceil(count / k))
                            < min_support,
                            support_warning_threshold=min_support,
                        )
                    )
    return pd.DataFrame(rows)


def candidate_assignment(
    matrix: pd.DataFrame, folds: int, seed: int = DEFAULT_SEED
) -> pd.Series:
    """Greedy iterative multilabel allocation with hard balanced size capacities.

    Subjects carry binary task-class indicators. Allocate a remaining rare
    indicator first, then prefer the fold with greatest unmet indicator demand,
    breaking ties by aggregate demand, capacity, and seeded random choice.
    This heuristic does not certify joint coverage or statistical reliability.
    """
    if matrix.index.has_duplicates or matrix.index.isna().any():
        raise ValueError("Assignments require unique, nonmissing subject IDs.")
    if not 2 <= folds <= len(matrix):
        raise ValueError("Require 2 <= folds <= number of subjects.")
    valid = matrix.apply(valid_labels)
    indicators = np.column_stack(
        [
            valid[task].eq(label).fillna(False).to_numpy(dtype=bool)
            for task in valid
            for label in EDA_VALID_LABELS
        ]
    )
    rng = np.random.default_rng(seed)
    capacities = np.full(folds, len(matrix) // folds, dtype=int)
    capacities[rng.permutation(folds)[: len(matrix) % folds]] += 1
    demand = capacities[:, None] / len(matrix) * indicators.sum(axis=0)[None, :]
    remaining = np.ones(len(matrix), dtype=bool)
    assignment = np.full(len(matrix), -1, dtype=int)
    while remaining.any():
        counts = indicators[remaining].sum(axis=0)
        positive = np.flatnonzero(counts > 0)
        if len(positive):
            rare = positive[counts[positive] == counts[positive].min()]
            indicator = int(rng.choice(rare))
            subjects = np.flatnonzero(remaining & indicators[:, indicator])
        else:
            indicator = None
            subjects = np.flatnonzero(remaining)
        for subject in rng.permutation(subjects):
            choices = np.flatnonzero(capacities > 0)
            if indicator is not None:
                choices = choices[
                    demand[choices, indicator] == demand[choices, indicator].max()
                ]
                score = demand[choices][:, indicators[subject]].sum(axis=1)
                choices = choices[np.isclose(score, score.max())]
            choices = choices[capacities[choices] == capacities[choices].max()]
            fold = int(rng.choice(choices))
            assignment[subject] = fold
            capacities[fold] -= 1
            demand[fold] -= indicators[subject]
            remaining[subject] = False
    return pd.Series(assignment, index=matrix.index, name="provisional_fold")


def split_diagnostics(
    matrix: pd.DataFrame,
    assignment: pd.Series,
    folds: int,
    min_support: int = EDA_MIN_FOLD_SUPPORT,
) -> pd.DataFrame:
    """Validate an assignment and report every task/fold/class, including zeros."""
    if (
        matrix.index.has_duplicates
        or matrix.index.isna().any()
        or assignment.index.has_duplicates
    ):
        raise ValueError("Subject identifiers must be unique and nonmissing.")
    if len(assignment) != len(matrix) or set(assignment.index) != set(matrix.index):
        raise ValueError("Assignment must contain exactly the subject union.")
    aligned = assignment.reindex(matrix.index)
    if aligned.isna().any() or not aligned.isin(range(folds)).all():
        raise ValueError("Each subject must have one valid fold number.")
    if folds < 2 or set(aligned) != set(range(folds)):
        raise ValueError("Every candidate fold must contain subjects.")
    valid = matrix.apply(valid_labels)
    rows = []
    for fold in range(folds):
        test_mask = aligned.eq(fold)
        train_ids, test_ids = (
            set(matrix.index[~test_mask]),
            set(matrix.index[test_mask]),
        )
        disjoint = train_ids.isdisjoint(test_ids)
        for task in matrix:
            test = valid.loc[test_mask, task]
            train = valid.loc[~test_mask, task]
            total = int(valid[task].notna().sum())
            n_test, n_train = int(test.notna().sum()), int(train.notna().sum())
            for label in EDA_VALID_LABELS:
                test_count, train_count = (
                    int(test.eq(label).sum()),
                    int(train.eq(label).sum()),
                )
                prevalence = (
                    int(valid[task].eq(label).sum()) / total if total else np.nan
                )
                test_proportion = test_count / n_test if n_test else np.nan
                train_proportion = train_count / n_train if n_train else np.nan
                rows.append(
                    dict(
                        task=task,
                        folds=folds,
                        fold=fold,
                        label=label,
                        provisional=True,
                        subject_disjoint=disjoint,
                        train_subjects=len(train_ids),
                        test_subjects=len(test_ids),
                        eligible_train=n_train,
                        eligible_test=n_test,
                        train_count=train_count,
                        test_count=test_count,
                        train_class_missing=train_count == 0,
                        test_class_missing=test_count == 0,
                        tiny_train_support=train_count < min_support,
                        tiny_test_support=test_count < min_support,
                        support_warning_threshold=min_support,
                        full_eligible_proportion=prevalence,
                        test_proportion=test_proportion,
                        train_proportion=train_proportion,
                        test_proportion_difference=test_proportion - prevalence,
                        train_proportion_difference=train_proportion - prevalence,
                    )
                )
    return pd.DataFrame(rows)


def evaluate_candidates(
    tables: list[TaskTable],
    fold_counts: tuple[int, ...] = EDA_FOLD_COUNTS,
    seeds: tuple[int, ...] = EDA_CANDIDATE_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate diagnostics only; keep all candidate assignments local."""
    results, blocked = [], []
    for cohort, group in cohort_tables(tables).items():
        try:
            matrix = subject_label_matrix(group)
        except (ValueError, KeyError) as error:
            blocked.append(dict(cohort=cohort, reason=str(error)))
            continue
        for folds in fold_counts:
            if folds > len(matrix):
                blocked.append(
                    dict(
                        cohort=cohort, reason=f"{folds} folds exceed the subject count."
                    )
                )
                continue
            for seed in seeds:
                assignment = candidate_assignment(matrix, folds, seed)
                diagnostics = split_diagnostics(matrix, assignment, folds)
                diagnostics.insert(0, "cohort", cohort)
                diagnostics.insert(1, "seed", seed)
                results.append(diagnostics)
    return (
        pd.concat(results, ignore_index=True) if results else pd.DataFrame(),
        pd.DataFrame(blocked, columns=["cohort", "reason"]),
    )


def candidate_comparison(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    result = diagnostics.groupby(["cohort", "folds", "seed"], sort=False).agg(
        subject_disjoint=("subject_disjoint", "all"),
        missing_test_task_class_folds=("test_class_missing", "sum"),
        missing_train_task_class_folds=("train_class_missing", "sum"),
        worst_test_class_support=("test_count", "min"),
        worst_train_class_support=("train_count", "min"),
        test_subjects_min=("test_subjects", "min"),
        test_subjects_max=("test_subjects", "max"),
    )
    differences = (
        diagnostics.assign(
            absolute_test_difference=diagnostics.test_proportion_difference.abs(),
            absolute_train_difference=diagnostics.train_proportion_difference.abs(),
        )
        .groupby(["cohort", "folds", "seed"], sort=False)[
            ["absolute_test_difference", "absolute_train_difference"]
        ]
        .max()
    )
    return result.join(differences).reset_index()
