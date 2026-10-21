"""Small, aggregate figures; importing analysis modules does not require plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.schema import EDA_VALID_LABELS, FEATURE_CATEGORIES_MAP
from src.eda.io import TaskTable, cohort_tables, numerical_features


def plot_label_distributions(counts: pd.DataFrame) -> plt.Figure:
    cohorts = counts.cohort.unique()
    figure, axes = plt.subplots(
        len(cohorts),
        1,
        figsize=(13, 3 * len(cohorts)),
        squeeze=False,
        layout="constrained",
    )
    for ax, cohort in zip(axes.flat, cohorts):
        subset = counts[counts.cohort.eq(cohort)]
        proportions = subset.pivot(
            index="task", columns="label", values="raw_proportion"
        ).fillna(0)
        proportions = proportions.reindex(subset.task.drop_duplicates())
        proportions.plot.bar(stacked=True, ax=ax, width=0.85, colormap="tab10")
        ax.set(
            title=f"{cohort}: raw labels (−1 unresolved; categories nominal)",
            xlabel="Task",
            ylabel="Fraction of raw rows",
            ylim=(0, 1),
        )
        ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    return figure


def plot_class_counts(counts: pd.DataFrame) -> plt.Figure:
    cohorts = counts.cohort.unique()
    figure, axes = plt.subplots(
        1,
        len(cohorts),
        figsize=(5 * len(cohorts), 7),
        squeeze=False,
        layout="constrained",
    )
    for ax, cohort in zip(axes.flat, cohorts):
        subset = counts[counts.cohort.eq(cohort) & counts.label_kind.eq("valid")]
        matrix = subset.pivot(index="task", columns="label", values="count")
        matrix = matrix.reindex(
            index=subset.task.drop_duplicates(),
            columns=[str(x) for x in EDA_VALID_LABELS],
        )
        heatmap = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
        ax.set(
            xticks=range(len(matrix.columns)),
            xticklabels=matrix.columns,
            yticks=range(len(matrix)),
            yticklabels=matrix.index,
            title=f"{cohort}: valid class counts",
            xlabel="Nominal label",
        )
        for row, column in np.ndindex(matrix.shape):
            ax.text(
                column,
                row,
                str(int(matrix.iloc[row, column])),
                ha="center",
                va="center",
                color="white"
                if matrix.iloc[row, column] > np.nanmax(matrix.to_numpy()) * 0.6
                else "black",
            )
        figure.colorbar(heatmap, ax=ax, label="Rows")
    return figure


def plot_agreement(agreements: pd.DataFrame) -> plt.Figure:
    """Show exact agreement and a separate pairwise eligible-N heatmap."""
    cohorts = agreements.cohort.unique()
    figure, axes = plt.subplots(
        len(cohorts),
        2,
        figsize=(14, 5 * len(cohorts)),
        squeeze=False,
        layout="constrained",
    )
    for row, cohort in enumerate(cohorts):
        subset = agreements[agreements.cohort.eq(cohort)]
        tasks = list(dict.fromkeys([*subset.task_a, *subset.task_b]))
        for column, metric in enumerate(("exact_agreement", "eligible_subjects")):
            matrix = pd.DataFrame(np.nan, index=tasks, columns=tasks)
            for record in subset.itertuples():
                matrix.loc[record.task_a, record.task_b] = getattr(record, metric)
                matrix.loc[record.task_b, record.task_a] = getattr(record, metric)
            ax = axes[row, column]
            kwargs = (
                dict(vmin=0, vmax=1) if metric == "exact_agreement" else dict(vmin=0)
            )
            heatmap = ax.imshow(matrix, cmap="viridis", **kwargs)
            ax.set(
                title=f"{cohort}: {metric}",
                xticks=range(len(tasks)),
                yticks=range(len(tasks)),
                xticklabels=tasks,
                yticklabels=tasks,
            )
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            figure.colorbar(heatmap, ax=ax)
    return figure


def plot_representative_distributions(
    tables: list[TaskTable], task_index: int = 0
) -> plt.Figure:
    """Raw finite values for one explicitly identified task per cohort, four features."""
    groups = cohort_tables(tables)
    figure, axes = plt.subplots(
        len(groups),
        4,
        figsize=(14, 3 * len(groups)),
        squeeze=False,
        layout="constrained",
    )
    features = [columns[0] for columns in FEATURE_CATEGORIES_MAP.values()][:4]
    for row, (cohort, group) in enumerate(groups.items()):
        table = group[task_index]
        values = numerical_features(table.frame)
        for ax, feature in zip(axes[row], features):
            if feature in values:
                finite = values.loc[np.isfinite(values[feature]), feature]
                ax.hist(finite, bins="auto", color="#367c9c", edgecolor="white")
            ax.set(title=f"{cohort} / {table.task}", xlabel=feature, ylabel="Raw rows")
    return figure
