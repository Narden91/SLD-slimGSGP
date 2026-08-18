"""
SLD-slimGSGP: Main Entry Point.
Provides CLI interface for dataset exploration, feature inspection, tensor preparation,
and integration with SLIM-GSGP genetic programming algorithms.
"""

import argparse
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import sld_data as sld

console = Console()


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
    summary = sld.get_dataset_summary(data_dir=data_dir)
    sld.render_dataset_overview(summary)

    # 2. Performance Label Distribution across all cohorts
    df_all = sld.load_all(data_dir=data_dir, filter_absent=False)
    sld.render_label_distribution(df_all, title="Overall Performance Score Distribution (All Cohorts & Tasks)")

    # 3. Feature Categories Overview
    sld.render_feature_categories(df=df_all)

    # 4. PyTorch Tensor Preparation Demo
    console.print("\n[bold cyan]--- Preparing PyTorch Tensors for SLIM-GSGP (Task 1, 1year) ---[/bold cyan]")
    df_task1 = sld.load_task("1year", 1, data_dir=data_dir, filter_absent=True)
    tensors = sld.prepare_tensors(df_task1, feature_set="kinematic", standardize=True)
    sld.render_tensor_summary(tensors)

    train, val, test = sld.train_val_test_split(tensors, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    console.print(
        f"[green][OK] Train split:[/green] X={tuple(train[0].shape)}, y={tuple(train[1].shape)} | "
        f"[yellow]Val split:[/yellow] X={tuple(val[0].shape)}, y={tuple(val[1].shape)} | "
        f"[magenta]Test split:[/magenta] X={tuple(test[0].shape)}, y={tuple(test[1].shape)}"
    )


def run_slim_demo(cohort: str = "1year", task_id: int = 1, iterations: int = 15, pop_size: int = 50) -> None:
    """Run a fast SLIM-GSGP genetic programming demonstration on an SLD task."""
    console.print(f"\n[bold magenta]Running SLIM-GSGP demo on {cohort} - Task {task_id}...[/bold magenta]")
    try:
        from slim_gsgp.main_slim import slim
        from slim_gsgp.evaluators.fitness_functions import rmse

        # Load task and prepare standardized tensors (excluding absent subjects)
        df = sld.load_task(cohort, task_id, filter_absent=True)
        tensors = sld.prepare_tensors(df, feature_set="kinematic", standardize=True)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = sld.train_val_test_split(
            tensors, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        console.print(f"Dataset prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples. Features: {X_train.shape[1]}")
        console.print(f"Evolving SLIM-GSGP tree (pop_size={pop_size}, n_iter={iterations})...")

        final_tree = slim(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            dataset_name=f"sld_{cohort}_t{task_id}",
            slim_version="SLIM+SIG2",
            pop_size=pop_size,
            n_iter=iterations,
            ms_lower=0,
            ms_upper=1,
            p_inflate=0.5,
            reconstruct=True,
            verbose=False,
        )

        preds = final_tree.predict(X_test)
        test_rmse = float(rmse(y_true=y_test, y_pred=preds))

        console.print(f"[bold green][OK] SLIM-GSGP run finished successfully![/bold green]")
        console.print(f"Test RMSE on Task {task_id}: [bold yellow]{test_rmse:.4f}[/bold yellow]")

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
        "--cohort", "-c", type=str, default=None, help="Inspect a specific cohort (e.g. '1year', '2year', '3year')"
    )
    parser.add_argument(
        "--task", "-t", type=int, default=None, help="Inspect a specific task ID (e.g. 1 to 21)"
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
        "--run-slim", action="store_true", help="Execute quick SLIM-GSGP symbolic regression on loaded task"
    )

    args = parser.parse_args()

    # If no specific flags provided, run full interactive dashboard
    if len(sys.argv) == 1:
        run_full_dashboard(data_dir=args.data_dir)
        return

    # Specific routing
    if args.overview:
        summary = sld.get_dataset_summary(data_dir=args.data_dir)
        sld.render_dataset_overview(summary)

    if args.cohort and not args.task:
        df_cohort = sld.load_cohort(args.cohort, data_dir=args.data_dir, filter_absent=False)
        sld.render_label_distribution(df_cohort, title=f"Performance Distribution in Cohort {args.cohort}")
        sld.render_feature_categories(df=df_cohort)

    if args.task:
        cohort = args.cohort if args.cohort else "1year"
        df_task = sld.load_task(cohort, args.task, data_dir=args.data_dir, filter_absent=False)
        sld.render_task_summary(cohort, args.task, df_task)

    if args.features:
        df_all = sld.load_all(data_dir=args.data_dir, filter_absent=False)
        sld.render_feature_categories(df=df_all)

    if args.subject is not None:
        cohort = args.cohort if args.cohort else "1year"
        df_cohort = sld.load_cohort(cohort, data_dir=args.data_dir, filter_absent=False)
        sld.render_subject_profile(df_cohort, subject_id=args.subject)

    if args.tensor_demo:
        cohort = args.cohort if args.cohort else "1year"
        task_id = args.task if args.task else 1
        df_task = sld.load_task(cohort, task_id, data_dir=args.data_dir, filter_absent=True)
        tensors = sld.prepare_tensors(df_task, feature_set="kinematic", standardize=True)
        sld.render_tensor_summary(tensors)
        train, val, test = sld.train_val_test_split(tensors)
        console.print(
            f"[green][OK] Train:[/green] X={tuple(train[0].shape)}, y={tuple(train[1].shape)} | "
            f"[yellow]Val:[/yellow] X={tuple(val[0].shape)}, y={tuple(val[1].shape)} | "
            f"[magenta]Test:[/magenta] X={tuple(test[0].shape)}, y={tuple(test[1].shape)}"
        )

    if args.run_slim:
        cohort = args.cohort if args.cohort else "1year"
        task_id = args.task if args.task else 1
        run_slim_demo(cohort=cohort, task_id=task_id)


if __name__ == "__main__":
    main()
