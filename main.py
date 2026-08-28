"""
SLD-slimGSGP: Main Entry Point.
Provides CLI interface for dataset exploration, feature inspection, tensor preparation,
and integration with SLIM-GSGP genetic programming algorithms.
"""

import argparse
import sys
from typing import List, Optional, Union

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.data.loader import (
    get_available_cohorts,
    get_available_tasks,
    get_dataset_summary,
    load_task,
)
from src.data.preprocessor import prepare_tensors, train_val_test_split
from src.visualization.viewer import (
    render_dataset_overview,
    render_feature_categories,
    render_label_distribution,
    render_subject_profile,
    render_task_summary,
    render_tensor_summary,
)
from src.genetic.runner import SlimConfig, run_slim_experiment

console = Console()

ALL = "all"


def _resolve_cohorts(selection: Optional[List[str]], data_dir: Optional[str]) -> List[str]:
    """Resolve a --cohort selection ('all', one, or several names) to a cohort list."""
    if not selection or selection == [ALL]:
        return get_available_cohorts(data_dir=data_dir)
    return selection


def _resolve_tasks(cohort: str, selection: Optional[List[Union[int, str]]], data_dir: Optional[str]) -> List[int]:
    """Resolve a --task selection ('all', one, or several IDs) to a task list for one cohort."""
    if not selection or selection == [ALL]:
        return get_available_tasks(cohort, data_dir=data_dir)
    return [int(t) for t in selection]


def load_selection(
    cohorts: List[str],
    tasks: Optional[List[Union[int, str]]],
    data_dir: Optional[str] = None,
    filter_absent: bool = False,
) -> pd.DataFrame:
    """
    Load and concatenate every (cohort, task) combination in the selection.

    `tasks` is resolved independently per cohort since task IDs and their
    counts differ across cohorts (e.g. '3year' only has 18, not 21).
    """
    frames = [
        load_task(cohort, task, data_dir=data_dir, filter_absent=filter_absent)
        for cohort in cohorts
        for task in _resolve_tasks(cohort, tasks, data_dir)
    ]
    if not frames:
        raise ValueError("No matching (cohort, task) files found for the given selection.")
    return pd.concat(frames, ignore_index=True)


def run_full_dashboard(data_dir: Optional[str] = None) -> None:
    """Run default end-to-end data overview, statistics, and tensor demo."""
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]SLD-slimGSGP[/bold cyan]: [white]Online Handwriting Kinematics for Specific Learning Disorder Screening[/white]\n"
                "[dim]Built with first-principles data loading, robust demographic imputation, and PyTorch tensor integration.[/dim]"
            ),
            title="[bold yellow]Welcome[/bold yellow]",
            border_style="cyan",
        )
    )

    # 1. Dataset Overview
    summary = get_dataset_summary(data_dir=data_dir)
    render_dataset_overview(summary)

    # 2. Performance Label Distribution across all cohorts
    all_cohorts = get_available_cohorts(data_dir=data_dir)
    df_all = load_selection(all_cohorts, tasks=None, data_dir=data_dir, filter_absent=False)
    render_label_distribution(df_all, title="Overall Performance Score Distribution (All Cohorts & Tasks)")

    # 3. Feature Categories Overview
    render_feature_categories(df=df_all)

    # 4. PyTorch Tensor Preparation Demo
    console.print("\n[bold cyan]--- Preparing PyTorch Tensors for SLIM-GSGP (Task 1, 1year) ---[/bold cyan]")
    df_task1 = load_task("1year", 1, data_dir=data_dir, filter_absent=True)
    tensors = prepare_tensors(df_task1, feature_set="kinematic", standardize=True)
    render_tensor_summary(tensors)

    train, val, test = train_val_test_split(tensors)
    console.print(
        f"[green][OK] Train split:[/green] X={tuple(train[0].shape)}, y={tuple(train[1].shape)} | "
        f"[yellow]Val split:[/yellow] X={tuple(val[0].shape)}, y={tuple(val[1].shape)} | "
        f"[magenta]Test split:[/magenta] X={tuple(test[0].shape)}, y={tuple(test[1].shape)}"
    )


def run_slim_demo(df: pd.DataFrame, label: str, iterations: int = 15, pop_size: int = 50) -> None:
    """Run a fast SLIM-GSGP genetic programming demonstration on a loaded task selection."""
    console.print(f"\n[bold magenta]Running SLIM-GSGP demo on {label}...[/bold magenta]")
    try:
        tensors = prepare_tensors(df, feature_set="kinematic", standardize=True)
        config = SlimConfig(pop_size=pop_size, n_iter=iterations, slim_version="SLIM+SIG2")

        console.print(f"Dataset prepared: {len(tensors.X)} total samples, 30 features. Evolving SLIM-GSGP tree...")
        result = run_slim_experiment(tensors, config=config, dataset_name=f"sld_{label}")

        console.print(f"[bold green][OK] SLIM-GSGP run finished successfully in {result.elapsed_seconds:.2f}s![/bold green]")
        console.print(f"Test RMSE: [bold yellow]{result.test_rmse:.4f}[/bold yellow] | Test MAE: [bold yellow]{result.test_mae:.4f}[/bold yellow]")

    except ImportError as e:
        console.print(f"[red]Error loading slim-gsgp library:[/red] {e}")


def main() -> None:
    """CLI entry point for SLD data explorer and SLIM-GSGP integration."""
    parser = argparse.ArgumentParser(
        description="SLD-slimGSGP: Online Handwriting Data Module & Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, default=None, help="Path to base data directory (default: ./data)"
    )
    parser.add_argument(
        "--overview", "-o", action="store_true", help="Display dataset high-level summary table"
    )
    parser.add_argument(
        "--cohort", "-c", type=str, nargs="+", default=None, metavar="COHORT",
        help="One or more cohorts (e.g. '1year' '2year'), or 'all' for every cohort (default: all cohorts)",
    )
    parser.add_argument(
        "--task", "-t", type=str, nargs="+", default=None, metavar="TASK",
        help="One or more task IDs (e.g. 1 5 9), or 'all' for every task in the selected cohort(s) (default: all tasks)",
    )
    parser.add_argument(
        "--features", "-f", action="store_true", help="Display detailed 30 kinematic features breakdown"
    )
    parser.add_argument(
        "--subject", "-s", type=int, default=None, help="Inspect longitudinal performance for a subject ID"
    )
    parser.add_argument(
        "--tensor-demo", action="store_true", help="Demonstrate PyTorch tensor preparation"
    )
    parser.add_argument(
        "--run-slim", action="store_true", help="Execute quick SLIM-GSGP symbolic regression on loaded task(s)"
    )

    args = parser.parse_args()

    # If no specific flags provided, run full interactive dashboard
    if len(sys.argv) == 1:
        run_full_dashboard(data_dir=args.data_dir)
        return

    # Specific routing
    try:
        if args.overview:
            summary = get_dataset_summary(data_dir=args.data_dir)
            render_dataset_overview(summary)

        if (args.cohort or args.task) and not (args.subject is not None or args.tensor_demo or args.run_slim):
            cohorts = _resolve_cohorts(args.cohort, args.data_dir)
            for cohort in cohorts:
                for task_id in _resolve_tasks(cohort, args.task, args.data_dir):
                    df_task = load_task(cohort, task_id, data_dir=args.data_dir, filter_absent=False)
                    render_task_summary(cohort, task_id, df_task)

        if args.features:
            cohorts = _resolve_cohorts(args.cohort, args.data_dir)
            df_selection = load_selection(cohorts, args.task, data_dir=args.data_dir, filter_absent=False)
            render_feature_categories(df=df_selection)

        if args.subject is not None:
            # Subject IDs are cohort-local, so each selected cohort is rendered separately.
            for cohort in _resolve_cohorts(args.cohort, args.data_dir):
                df_cohort = load_selection([cohort], tasks=None, data_dir=args.data_dir, filter_absent=False)
                render_subject_profile(df_cohort, subject_id=args.subject)

        if args.tensor_demo:
            cohorts = _resolve_cohorts(args.cohort, args.data_dir)
            df_selection = load_selection(cohorts, args.task, data_dir=args.data_dir, filter_absent=True)
            tensors = prepare_tensors(df_selection, feature_set="kinematic", standardize=True)
            render_tensor_summary(tensors)
            train, val, test = train_val_test_split(tensors)
            console.print(
                f"[green][OK] Train:[/green] X={tuple(train[0].shape)}, y={tuple(train[1].shape)} | "
                f"[yellow]Val:[/yellow] X={tuple(val[0].shape)}, y={tuple(val[1].shape)} | "
                f"[magenta]Test:[/magenta] X={tuple(test[0].shape)}, y={tuple(test[1].shape)}"
            )

        if args.run_slim:
            cohorts = _resolve_cohorts(args.cohort, args.data_dir)
            df_selection = load_selection(cohorts, args.task, data_dir=args.data_dir, filter_absent=True)
            run_slim_demo(df_selection, label="+".join(cohorts))

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
