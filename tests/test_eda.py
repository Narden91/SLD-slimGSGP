"""Small synthetic fixtures protect EDA semantics without exposing research data."""

import numpy as np
import pandas as pd
import pytest

from src.core.schema import (
    CSV_DTYPES,
    DOMINANT_HAND_VALUES,
    EDA_DURATION_RELATIONSHIPS,
    EDA_VALID_LABELS,
    GENDER_VALUES,
    KINEMATIC_FEATURES,
)
from src.eda.audit import audit_tasks
from src.eda.features import feature_quality
from src.eda.io import TaskTable, load_tasks, measurement_masks, numerical_features
from src.eda.labels import (
    label_counts,
    rare_class_flags,
    subject_label_matrix,
    subject_label_patterns,
    task_label_agreement,
)
from src.eda.splits import (
    candidate_assignment,
    candidate_comparison,
    evaluate_candidates,
    split_diagnostics,
    theoretical_feasibility,
)


def task_table(labels, ids=None, task=1, cohort="synthetic"):
    ids = list(range(len(labels))) if ids is None else ids
    df = pd.DataFrame({column: [1.0] * len(labels) for column in CSV_DTYPES})
    df["Id"], df["Label"], df["Task"] = ids, labels, task
    df["Gender"], df["Dominant_Hand"], df["Age"] = (
        GENDER_VALUES[0],
        DOMINANT_HAND_VALUES[0],
        7.0,
    )
    return TaskTable(cohort, f"Task_{task}.csv", df.astype("string"))


def test_discovery_preserves_raw_tokens_and_natural_order(tmp_path):
    directory = tmp_path / "synthetic"
    directory.mkdir()
    for task in (10, 2, 1):
        table = task_table([0, "unresolved"], ids=["001", None], task=task)
        table.frame.to_csv(directory / table.source, index=False)
    before = {path: path.read_bytes() for path in directory.iterdir()}
    loaded = load_tasks(tmp_path)
    assert [t.task for t in loaded] == ["Task_1", "Task_2", "Task_10"]
    assert loaded[0].frame.Id.iloc[0] == "001"
    assert pd.isna(loaded[0].frame.Id.iloc[1])
    assert loaded[0].frame.Label.iloc[1] == "unresolved"
    assert all(path.read_bytes() == content for path, content in before.items())


def test_discovery_does_not_silently_accept_empty_or_repeated_selection(tmp_path):
    with pytest.raises(ValueError, match="No cohorts"):
        load_tasks(tmp_path, cohorts=[])
    with pytest.raises(ValueError, match="only once"):
        load_tasks(tmp_path, cohorts=["synthetic", "synthetic"])
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path, cohorts=["missing"])


def test_label_counts_keep_unknown_and_absent_classes():
    table = task_table([0, 2, 2, -1, "bad", None])
    counts = label_counts([table]).set_index("label")
    assert counts.loc["1", "count"] == 0
    assert counts.loc["-1", "label_kind"] == "unresolved"
    assert counts.loc["bad", "count"] == counts.loc["<missing>", "count"] == 1
    assert counts["count"].sum() == 6
    assert counts.loc["2", "raw_proportion"] == pytest.approx(2 / 6)
    assert counts.loc["2", "eligible_proportion"] == pytest.approx(2 / 3)
    assert pd.isna(counts.loc["-1", "eligible_proportion"])
    rare = rare_class_flags(counts.reset_index()).set_index("label")
    assert rare.loc["1", "support"] == "absent"
    assert rare.loc["0", "support"] == "singleton"


def test_outer_join_and_nominal_agreement_exclude_unknown_pairwise():
    left = task_table([0, 1, -1], ids=[1, 2, 3])
    right = task_table([0, -1, 2], ids=[1, 2, 4], task=2)
    matrix = subject_label_matrix([left, right])
    assert len(matrix) == 4
    assert pd.isna(matrix.loc["3", "Task_2"])
    agreement = task_label_agreement(matrix)
    pair = agreement.loc[agreement.task_a.ne(agreement.task_b)].iloc[0]
    assert pair.eligible_subjects == 1
    assert pair.exact_agreement == 1


@pytest.mark.parametrize("ids", [[1, 1], [1, None], [1, " "]])
def test_ambiguous_subject_joins_fail(ids):
    with pytest.raises(ValueError, match="Missing or duplicate"):
        subject_label_matrix([task_table([0, 1], ids=ids)])


def test_cross_cohort_ids_never_join():
    with pytest.raises(ValueError, match="exactly one cohort"):
        subject_label_matrix([task_table([0], cohort="a"), task_table([2], cohort="b")])


def test_label_patterns_preserve_task_specific_categories():
    tables = [
        task_table(labels, task=index + 1)
        for index, labels in enumerate([[0, 0, 0, -1], [0, 1, 1, -1], [0, 1, 2, -1]])
    ]
    patterns = subject_label_patterns(subject_label_matrix(tables)).set_index(
        "distinct_valid_labels"
    )
    assert patterns.subjects.to_dict() == {0: 1, 1: 1, 2: 1, 3: 1}


def test_agreement_with_no_eligible_pair_is_undefined():
    matrix = subject_label_matrix([task_table([-1, 0]), task_table([1, -1], task=2)])
    pair = task_label_agreement(matrix).query("task_a != task_b").iloc[0]
    assert pair.eligible_subjects == 0 and pd.isna(pair.exact_agreement)


def test_audit_coverage_demographic_conflicts_and_duplicate_ids():
    left = task_table([0, 1, -1], ids=[1, 2, 2])
    right = task_table([2, 1], ids=[1, 3], task=2)
    right.frame.loc[0, "Gender"] = GENDER_VALUES[1]
    audit = audit_tasks([left, right])
    assert audit.cohorts.subjects.iloc[0] == 3
    assert audit.cohorts.subjects_in_every_task.iloc[0] == 1
    assert not audit.cohorts.identical_subject_sets.iloc[0]
    assert audit.cohorts.Gender_conflicting_subjects.iloc[0] == 1
    assert audit.flags.query("issue == 'duplicate_id_rows'")["count"].iloc[0] == 2
    assert "Id" not in audit.flags and "Id" not in audit.coverage


def test_audit_schema_task_and_invalid_metadata_flags():
    table = task_table(["bad", -1], ids=[None, 1])
    table.frame.loc[0, ["Age", "Gender"]] = ["0", "unrecognized"]
    table.frame["Task"] = "2"
    table.frame = table.frame.drop(columns=[KINEMATIC_FEATURES[0]])
    audit = audit_tasks([table])
    issues = set(audit.flags.issue)
    assert {
        "missing_id",
        "missing_column",
        "task_filename_mismatch",
        "implausible_or_missing_age",
        "unrecognized_or_missing_category",
        "unexpected_or_missing_label",
    } <= issues
    duplicate = TaskTable(table.cohort, "nested/Task_2.csv", table.frame.copy())
    repeated_issues = set(audit_tasks([table, duplicate]).flags.issue)
    assert {
        "task_identifier_in_multiple_files",
        "cross_file_duplicate_subject_task_rows",
    } <= repeated_issues


def test_measurements_distinguish_zero_only_missing_and_partly_zero():
    values = pd.DataFrame([[0, 0], [0, np.nan], [np.nan, np.inf], [1, 0]])
    masks = measurement_masks(values)
    assert masks["all_zero_measurements"].tolist() == [True, False, False, False]
    assert masks["no_finite_measurements"].tolist() == [False, False, True, False]


def test_feature_summary_excludes_metadata_and_keeps_original_data():
    table = task_table([0, -1, 2, 1])
    first, second = KINEMATIC_FEATURES[:2]
    table.frame.loc[:, first] = pd.Series(["0", "bad", "inf", "4"], dtype="string")
    table.frame.loc[:, second] = pd.Series(["0", None, "0", "4"], dtype="string")
    original = table.frame.copy(deep=True)
    result = feature_quality([table])
    assert numerical_features(table.frame).shape[1] == len(KINEMATIC_FEATURES)
    raw = result.quality.query("scope == 'raw'").set_index("feature")
    assert raw.loc[first, "invalid_numeric"] == 1
    assert raw.loc[first, "infinite"] == 1
    assert raw.loc[second, "missing"] == 1
    statistics = result.statistics.query("scope == 'raw'").set_index("feature")
    assert statistics.loc[first, "median"] == 2
    assert set(result.quality.scope) == {"raw", "valid_labels_only"}
    pd.testing.assert_frame_equal(original, table.frame)


def test_feature_zero_rows_duration_violations_and_duplicate_columns():
    table = task_table([0, 1])
    table.frame.loc[0, KINEMATIC_FEATURES] = "0"
    overall, surface = EDA_DURATION_RELATIONSHIPS[0]
    table.frame.loc[1, [overall, surface]] = ["1", "2"]
    result = feature_quality([table])
    measurements = result.measurements
    assert measurements.query("issue == 'all_zero_measurements'").rows.iloc[0] == 1
    duration = measurements.query("issue == 'overall_duration_below_on_surface'").iloc[
        0
    ]
    assert duration.rows == 1 and duration.evaluable_rows == 2
    assert "duplicate_feature_columns" in set(measurements.issue)


def test_near_constant_and_zero_iqr_extremes_are_descriptive():
    table = task_table([0] * 20)
    table.frame.loc[19, KINEMATIC_FEATURES[0]] = "1000"
    result = feature_quality([table])
    record = result.quality.query("scope == 'raw'").iloc[0]
    assert record.near_constant and not record.constant
    assert record.modal_fraction == 0.95
    assert record.zero_iqr and record.potential_extreme_values == 1


def test_measurement_schema_is_stable_when_no_columns_are_duplicated():
    table = task_table([0, 1])
    for index, feature in enumerate(KINEMATIC_FEATURES):
        table.frame[feature] = str(index + 1)
    measurements = feature_quality([table]).measurements
    assert "duplicate_feature_columns" not in set(measurements.issue)
    assert "scope" in measurements


def test_feasibility_absent_and_rare_classes_use_subject_counts():
    table = task_table([0, 0, 2, 2, 2, -1])
    feasibility = theoretical_feasibility([table], fold_counts=(3,))
    by_class = feasibility.set_index("label")
    assert by_class.loc[1, "class_absent"]
    assert by_class.loc[1, "unavoidable_missing_test_folds"] == 3
    assert not by_class.loc[0, "every_test_fold_coverage_possible"]
    assert by_class.loc[2, "every_test_fold_coverage_possible"]
    assert by_class.loc[2, "tiny_test_support"]
    with pytest.raises(ValueError, match="duplicate"):
        theoretical_feasibility([task_table([0, 0], ids=[1, 1])])


@pytest.mark.parametrize("folds", [2, 3, 4, 5])
def test_common_assignments_are_deterministic_balanced_and_diagnosable(folds):
    tables = [
        task_table([0, 1, 2, 0, 1, 2, -1, -1, 2, 2, 1]),
        task_table([2, 1, 0, 1, -1, 2, 1, 0, 1, 2], ids=list(range(1, 11)), task=2),
    ]
    matrix = subject_label_matrix(tables)
    assignment = candidate_assignment(matrix, folds)
    pd.testing.assert_series_equal(assignment, candidate_assignment(matrix, folds))
    sizes = assignment.value_counts()
    assert sizes.max() - sizes.min() <= 1
    diagnostics = split_diagnostics(matrix, assignment, folds)
    assert diagnostics.subject_disjoint.all()
    assert len(diagnostics) == 2 * folds * len(EDA_VALID_LABELS)
    eligible = matrix.apply(
        lambda column: pd.to_numeric(column, errors="coerce").isin(EDA_VALID_LABELS)
    )
    for row in diagnostics.itertuples():
        label = pd.to_numeric(matrix[row.task], errors="coerce")
        assert row.test_count == int(
            (label.eq(row.label) & assignment.eq(row.fold)).sum()
        )
        assert row.train_count + row.test_count == int(label.eq(row.label).sum())
        assert row.eligible_train + row.eligible_test == int(eligible[row.task].sum())


def test_joint_allocation_can_spread_a_feasible_rare_class():
    matrix = subject_label_matrix([task_table([0] * 3 + [1] * 3 + [2] * 3)])
    diagnostics = split_diagnostics(matrix, candidate_assignment(matrix, 3), 3)
    assert diagnostics.test_count.eq(1).all()


def test_all_unknown_candidates_still_assign_every_subject():
    matrix = subject_label_matrix([task_table([-1] * 7)])
    assignment = candidate_assignment(matrix, 3)
    diagnostics = split_diagnostics(matrix, assignment, 3)
    assert diagnostics.test_count.eq(0).all()
    assert diagnostics.test_proportion.isna().all()


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "invalid_fold"])
def test_diagnostics_reject_invalid_assignments(kind):
    matrix = subject_label_matrix([task_table([0, 1, 2, 0])])
    assignment = candidate_assignment(matrix, 2)
    if kind == "missing":
        assignment = assignment.iloc[:-1]
    elif kind == "extra":
        assignment.loc["extra"] = 0
    elif kind == "duplicate":
        assignment = pd.concat([assignment, assignment.iloc[:1]])
    else:
        assignment.iloc[0] = 9
    with pytest.raises(ValueError):
        split_diagnostics(matrix, assignment, 2)


def test_candidate_evaluation_separates_cohorts_and_reports_blocked_joins():
    tables = [
        task_table([0, 1, 2, -1], cohort="a"),
        task_table([2, 2, 2, 2], cohort="b"),
        task_table([0, 1], ids=[1, 1], cohort="bad"),
    ]
    diagnostics, blocked = evaluate_candidates(tables, fold_counts=(2,), seeds=(42,))
    assert set(diagnostics.cohort) == {"a", "b"}
    assert blocked.cohort.tolist() == ["bad"]
    assert diagnostics.test_subjects.eq(2).all()
    comparison = candidate_comparison(diagnostics)
    assert comparison.subject_disjoint.all()
    assert comparison.set_index("cohort").loc["b", "worst_test_class_support"] == 0
