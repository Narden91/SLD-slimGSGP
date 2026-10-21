"""Deterministic raw ingestion without demographic cleaning or label filtering."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.core.schema import CSV_DTYPES, EDA_CSV_DTYPES, EDA_VALID_LABELS
from src.data.loader import get_default_data_dir


@dataclass
class TaskTable:
    """One source file; source is relative to its cohort directory."""

    cohort: str
    source: str
    frame: pd.DataFrame

    @property
    def task(self) -> str:
        return str(Path(self.source).with_suffix(""))


def natural_key(value: str) -> tuple:
    """Sort numeric filename components numerically, with deterministic ties."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    ) + ((0, value),)


def filename_task_id(source: str) -> int | None:
    match = re.fullmatch(r"Task_(\d+)", Path(source).stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_tasks(
    data_dir: Path | None = None,
    cohorts: Iterable[str] | None = None,
) -> list[TaskTable]:
    """Discover all CSVs in selected cohorts; preserve original cell tokens.

    The modeling loader uses strict numeric dtypes. Auditing instead uses the
    explicit string schema so a missing ID or malformed label remains auditable.
    """
    root = Path(data_dir) if data_dir is not None else get_default_data_dir()
    selected = (
        list(cohorts)
        if cohorts is not None
        else [
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
    )
    if len(selected) != len(set(selected)):
        raise ValueError("Select each cohort only once.")
    tables = []
    for cohort in sorted(selected, key=natural_key):
        directory = root / cohort
        if not directory.is_dir():
            raise FileNotFoundError(f"Cohort directory does not exist: {cohort}")
        files = sorted(
            (
                p
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() == ".csv"
            ),
            key=lambda p: natural_key(p.relative_to(directory).as_posix()),
        )
        if not files:
            raise FileNotFoundError(f"No task CSVs found in cohort: {cohort}")
        for path in files:
            frame = pd.read_csv(path, dtype=EDA_CSV_DTYPES)
            # Explicit typing also covers previously unseen feature columns.
            frame = frame.astype("string")
            tables.append(
                TaskTable(cohort, path.relative_to(directory).as_posix(), frame)
            )
    if not tables:
        raise ValueError("No cohorts selected.")
    return tables


def cohort_tables(tables: list[TaskTable]) -> dict[str, list[TaskTable]]:
    result: dict[str, list[TaskTable]] = {}
    for table in tables:
        result.setdefault(table.cohort, []).append(table)
    return result


def numerical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Include known measurements plus numerical extra columns; exclude metadata."""
    metadata = set(CSV_DTYPES) - {
        c for c, dtype in CSV_DTYPES.items() if dtype == "float64" and c != "Age"
    }
    columns = []
    for column in frame.columns:
        if column in metadata or column == "Cohort":
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if column in CSV_DTYPES or converted.notna().any():
            columns.append(column)
    return frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)


def valid_labels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin(EDA_VALID_LABELS))


def measurement_masks(values: pd.DataFrame) -> dict[str, pd.Series]:
    """Zero-only requires every measurement to be present and exactly zero."""
    finite = pd.DataFrame(
        np.isfinite(values), index=values.index, columns=values.columns
    )
    if values.shape[1] == 0:
        return {
            name: pd.Series(name == "no_finite_measurements", index=values.index)
            for name in ("all_zero_measurements", "no_finite_measurements")
        }
    return {
        "all_zero_measurements": values.eq(0).all(axis=1) & finite.all(axis=1),
        "no_finite_measurements": ~finite.any(axis=1),
    }
