# Commands

## Setup

```bash
uv sync              # installs deps, including the slim-gsgp submodule (slim_v2/)
pytest tests          # 24 tests, run against real data/ CSVs
```

## CLI (`main.py`)

```bash
python main.py        # no flags: full dashboard (overview, label distribution,
                       # feature breakdown, tensor prep demo on 1year/Task 1)
```

### Selecting cohorts and tasks

`--cohort`/`-c` and `--task`/`-t` both accept one value, several space-separated values, or
the literal `all`. **Omitting either means "all"** — this is symmetric and deliberate.

```bash
python main.py --cohort 1year --task 1              # single cohort, single task
python main.py --cohort 1year 2year --task 1 5 9     # 2 cohorts x 3 tasks
python main.py --cohort all --task 1                 # every cohort, task 1 only
python main.py --task all                            # every cohort (default), every task
python main.py --tensor-demo                         # no --cohort/--task: ALL data by default
```

Task IDs are resolved per cohort (`3year` only goes up to 18, not 21), so `--task all` with
multiple cohorts correctly loads a different task range per cohort.

### Flags

| Flag | Effect |
|---|---|
| `--overview`, `-o` | Dataset summary table (files, records, per-cohort label distribution) |
| `--cohort NAME...`, `-c` | Select cohort(s): one, several, or `all` (default: all) |
| `--task N...`, `-t` | Select task(s): one, several, or `all` (default: all) |
| `--features`, `-f` | 30 kinematic features, grouped by category, with stats over the selection |
| `--subject N`, `-s` | Longitudinal trajectory for subject `N`, rendered once per selected cohort (subject IDs are cohort-local — see [DATA.md](DATA.md)) |
| `--tensor-demo` | Build PyTorch tensors from the selection, show shape/dtype, run a train/val/test split |
| `--run-slim` | Run a fast SLIM-GSGP demo on the selection (small pop/iterations, for a quick check — not a real experiment) |
| `--data-dir PATH`, `-d` | Override the data directory (default: `./data`) |

Combining `--cohort`/`--task` with a task-summary view (no other action flag) prints one
table per resolved `(cohort, task)` pair:

```bash
python main.py --cohort 1year 2year --task 1 2   # prints 4 task-summary tables
```

Bad selections (unknown cohort, missing task file) print `Error: ...` and exit 1, instead of
a raw traceback.

## Python API

Import from the real module — there are no `src`-level re-exports (see [CLAUDE.md](../CLAUDE.md)).

### Load data

```python
from src.data.loader import load_task, load_cohort, load_all

df_task   = load_task("1year", 1, filter_absent=True)   # one cohort, one task
df_cohort = load_cohort("2year")                         # one cohort, all its tasks
df_all    = load_all()                                   # every cohort, every task
```

All three accept `data_dir=` (override) and `filter_absent=` (drop `Label == -1` rows,
default `False`). `load_task`/`load_cohort` also accept a `Cohort` enum member instead of a
string.

```python
from src.data.loader import get_available_cohorts, get_available_tasks

get_available_cohorts()          # ['1year', '2year', '3year']
get_available_tasks("3year")     # [1, 2, ..., 18]
```

### Clean and prepare tensors

```python
from src.data.preprocessor import clean_and_impute_demographics, prepare_tensors, train_val_test_split

df_clean = clean_and_impute_demographics(df_task)   # fixes Age/Gender/Dominant_Hand anomalies

tensors = prepare_tensors(
    df_task,
    feature_set="kinematic",   # "kinematic" (30 features) | "all" (+ Age, encoded demographics) | list[str]
    filter_absent=True,
    standardize=True,           # z-score normalize X
)
# tensors.X: (N, 30) float32, tensors.y: (N,) float32

(X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(tensors)
# ratios/seed default to DEFAULT_TRAIN_RATIO / DEFAULT_VAL_RATIO / DEFAULT_TEST_RATIO / DEFAULT_SEED
```

### Stats and rendering

```python
from src.analysis.stats import compute_feature_stats, get_subject_trajectory
from src.visualization.viewer import render_task_summary, render_subject_profile

stats = compute_feature_stats(df_task)          # mean/std/min/median/max/iqr per feature

traj = get_subject_trajectory(df_all, subject_id=1, cohort="1year")   # pass cohort explicitly
render_subject_profile(df_cohort, subject_id=1)                        # rich table
```

### Run SLIM-GSGP

```python
from src.genetic.runner import SlimConfig, run_slim_experiment

config = SlimConfig(pop_size=50, n_iter=20, slim_version="SLIM+SIG2")
result = run_slim_experiment(tensors, config=config)
print(result.test_rmse, result.test_mae)
```

`run_slim_experiment` imports the `slim_gsgp` package lazily and raises `ImportError` if the
`slim_v2/` submodule isn't installed — everything else in `src/` works without it.
