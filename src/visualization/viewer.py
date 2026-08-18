"""
Rich terminal visualization and display module for the SLD handwriting dataset.
Provides colorized tables, histograms, feature breakdowns, and subject trajectories.
"""

from typing import Dict, List, Optional
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.core.schema import (
    DatasetSummary,
    FEATURE_CATEGORIES_MAP,
    FeatureCategory,
    KINEMATIC_FEATURES,
    LABEL_DESCRIPTION,
    PreparedTensors,
    TARGET_COLUMN,
)
from src.analysis.stats import (
    compute_category_stats,
    compute_demographics_summary,
    compute_feature_stats,
    compute_label_distribution,
    get_subject_trajectory,
)

console = Console()


def _get_label_color(label: int) -> str:
    """Return color string for a score label."""
    if label == 0:
        return "green"  # Best performance
    elif label == 1:
        return "yellow"  # Intermediate performance
    elif label == 2:
        return "red"  # Worst performance
    elif label == -1:
        return "dim"  # Absent
    return "white"


def render_dataset_overview(summary: DatasetSummary) -> None:
    """Display high-level dataset overview table."""
    table = Table(
        title="[bold cyan]SLD Online Handwriting Dataset Overview[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Cohort / Grade", style="cyan", justify="left")
    table.add_column("Tasks", justify="right")
    table.add_column("Subjects", justify="right")
    table.add_column("Total Rows", justify="right")
    table.add_column("Score 0 (Best)", justify="right", style="green")
    table.add_column("Score 1 (Mid)", justify="right", style="yellow")
    table.add_column("Score 2 (Worst)", justify="right", style="red")
    table.add_column("Absent (-1)", justify="right", style="dim")

    for c_name, c_info in summary.cohorts.items():
        dist = c_info.label_distribution
        s0 = dist.get(0, 0)
        s1 = dist.get(1, 0)
        s2 = dist.get(2, 0)
        s_abs = dist.get(-1, 0)
        table.add_row(
            c_name,
            str(c_info.task_count),
            str(c_info.subject_count),
            f"{c_info.total_records:,}",
            f"{s0:,}",
            f"{s1:,}",
            f"{s2:,}",
            f"{s_abs:,}",
        )

    summary_panel = Panel(
        Text.from_markup(
            f"[bold]Root Directory:[/bold] {summary.data_dir}\n"
            f"[bold]Total Files:[/bold] {summary.total_files} CSVs | "
            f"[bold]Total Records:[/bold] {summary.total_records:,} | "
            f"[bold]Kinematic Features:[/bold] {summary.num_kinematic_features} per record"
        ),
        title="[bold]Dataset Metadata[/bold]",
        border_style="blue",
    )

    console.print()
    console.print(summary_panel)
    console.print(table)
    console.print()


def render_label_distribution(df: pd.DataFrame, title: str = "Performance Label Distribution") -> None:
    """Display performance label breakdown with ASCII bar representation."""
    dist_df = compute_label_distribution(df)
    if dist_df.empty:
        console.print("[yellow]No label data found.[/yellow]")
        return

    table = Table(title=f"[bold cyan]{title}[/bold cyan]", box=box.ROUNDED, header_style="bold blue")
    table.add_column("Score", justify="center")
    table.add_column("Meaning / Status", justify="left")
    table.add_column("Count", justify="right")
    table.add_column("Percent", justify="right")
    table.add_column("Distribution", justify="left")

    for _, row in dist_df.iterrows():
        lbl = int(row["label"])
        color = _get_label_color(lbl)
        bar_len = int(row["percentage"] / 2.5)  # Scale to ~40 chars max
        bar_str = f"[{color}]{'=' * bar_len}[/{color}]"

        table.add_row(
            f"[{color}]{lbl}[/{color}]",
            row["description"],
            f"{int(row['count']):,}",
            f"{row['percentage']:.1f}%",
            bar_str,
        )

    console.print(table)


def render_feature_categories(df: Optional[pd.DataFrame] = None) -> None:
    """Display categorized list of the 30 kinematic features with optional statistics."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]30 Kinematic & Dynamic Handwriting Features Extracted from Stylus[/bold cyan]\n"
            "Features are computed during online tablet handwriting tasks.",
            border_style="cyan",
        )
    )

    for cat, feat_names in FEATURE_CATEGORIES_MAP.items():
        table = Table(
            title=f"[bold yellow]Category: {cat.value.upper()}[/bold yellow] ({len(feat_names)} features)",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Feature Name", justify="left", style="bold")

        if df is not None:
            table.add_column("Mean +/- Std", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("Median", justify="right")
            table.add_column("Max", justify="right")

            for idx, feat in enumerate(feat_names, 1):
                if feat in df.columns:
                    s = pd.to_numeric(df[feat], errors="coerce").dropna()
                    if len(s) > 0:
                        mean_std = f"{s.mean():.3f} +/- {s.std():.3f}"
                        min_v = f"{s.min():.3f}"
                        med_v = f"{s.median():.3f}"
                        max_v = f"{s.max():.3f}"
                    else:
                        mean_std, min_v, med_v, max_v = "N/A", "N/A", "N/A", "N/A"
                    table.add_row(str(idx), feat, mean_std, min_v, med_v, max_v)
                else:
                    table.add_row(str(idx), feat, "N/A", "N/A", "N/A", "N/A")
        else:
            for idx, feat in enumerate(feat_names, 1):
                table.add_row(str(idx), feat)

        console.print(table)


def render_task_summary(cohort: str, task_id: int, df: pd.DataFrame) -> None:
    """Display comprehensive summary for a single task."""
    console.print()
    header_panel = Panel(
        Text.from_markup(
            f"[bold]Cohort:[/bold] [cyan]{cohort}[/cyan] | "
            f"[bold]Task ID:[/bold] [yellow]Task {task_id}[/yellow] | "
            f"[bold]Total Subjects in Task:[/bold] [green]{len(df)}[/green]"
        ),
        title="[bold green]Task Inspection[/bold green]",
        border_style="green",
    )
    console.print(header_panel)

    # 1. Label Distribution
    render_label_distribution(df, title=f"Task {task_id} Performance Scores")

    # 2. Key Kinematic Feature Stats
    key_features = [
        "writing_duration(on-surface)",
        "writing_duration_overall",
        "mean:pressure",
        "mean:velocity:axis-xy(on-surface)",
        "mean:stroke_duration(on-surface)",
        "in_air_movement_time",
        "average_normalized_jerk",
        "number_of_interruptions",
    ]
    present_keys = [f for f in key_features if f in df.columns]
    if present_keys:
        stats_df = compute_feature_stats(df, present_keys)
        table = Table(title="[bold cyan]Key Kinematic Indicators[/bold cyan]", box=box.ROUNDED)
        table.add_column("Feature", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Max", justify="right")

        for _, row in stats_df.iterrows():
            table.add_row(
                row["feature"],
                f"{row['mean']:.3f}",
                f"{row['std']:.3f}",
                f"{row['min']:.3f}",
                f"{row['median']:.3f}",
                f"{row['max']:.3f}",
            )
        console.print()
        console.print(table)


def render_subject_profile(df: pd.DataFrame, subject_id: int) -> None:
    """Display subject longitudinal performance across all tasks."""
    traj = get_subject_trajectory(df, subject_id)
    if traj.empty:
        console.print(f"[red]Subject ID {subject_id} not found in the loaded dataset.[/red]")
        return

    table = Table(
        title=f"[bold cyan]Subject ID {subject_id} Performance Profile Across Tasks[/bold cyan]",
        box=box.ROUNDED,
    )
    table.add_column("Task", justify="right")
    table.add_column("Cohort", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Performance Description", justify="left")
    if "writing_duration_overall" in traj.columns:
        table.add_column("Duration (s)", justify="right")
    if "mean:pressure" in traj.columns:
        table.add_column("Mean Pressure", justify="right")

    for _, row in traj.iterrows():
        lbl = int(row[TARGET_COLUMN])
        color = _get_label_color(lbl)
        dur = f"{row['writing_duration_overall']:.2f}" if "writing_duration_overall" in row else "-"
        press = f"{row['mean:pressure']:.1f}" if "mean:pressure" in row else "-"
        table.add_row(
            str(row["Task"]),
            str(row["Cohort"]),
            f"[{color}]{lbl}[/{color}]",
            f"[{color}]{row['Performance']}[/{color}]",
            dur,
            press,
        )

    console.print()
    console.print(table)


def render_tensor_summary(tensors: PreparedTensors) -> None:
    """Display PyTorch PreparedTensors shape, dtype, and class breakdown."""
    X, y = tensors.X, tensors.y
    panel_text = (
        f"[bold]X Tensor Shape:[/bold] [green]{tuple(X.shape)}[/green] (N={X.shape[0]}, D={X.shape[1]})\n"
        f"[bold]y Tensor Shape:[/bold] [green]{tuple(y.shape)}[/green] (N={y.shape[0]})\n"
        f"[bold]Data Type (dtype):[/bold] {X.dtype}\n"
        f"[bold]Features ({len(tensors.feature_names)}):[/bold] {', '.join(tensors.feature_names[:6])}...\n"
        f"[bold]Unique Subjects:[/bold] {len(torch.unique(tensors.subject_ids))} | "
        f"[bold]Unique Tasks:[/bold] {len(torch.unique(tensors.task_ids))}"
    )

    console.print()
    console.print(Panel(Text.from_markup(panel_text), title="[bold cyan]Prepared PyTorch Tensors for SLIM-GSGP[/bold cyan]", border_style="magenta"))
