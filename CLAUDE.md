# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repo.

## What this is

Research framework analyzing online handwriting kinematics (digital stylus/tablet data from
primary-school children) and applying SLIM-GSGP (Semantic Linear Initialization & Mutation
Geometric Semantic Genetic Programming) to screen for Specific Learning Disorders (SLD) /
dysgraphia. See [docs/DATA.md](docs/DATA.md) for the dataset and [docs/COMMANDS.md](docs/COMMANDS.md)
for CLI/API usage.

## Architecture

```
src/core/schema.py        # single source of truth: feature lists, enums, dataclasses,
                           # all named constants (thresholds, ratios, seed, dtypes)
src/data/loader.py        # CSV ingestion: load_task / load_cohort / load_all
src/data/preprocessor.py  # demographic cleaning/imputation, tensor prep, train/val/test split
src/analysis/stats.py     # descriptive stats, label distribution, subject trajectories
src/visualization/viewer.py  # rich terminal tables/panels
src/genetic/runner.py     # SlimConfig / run_slim_experiment wrapping the slim_gsgp library
main.py                   # CLI entry point
slim_v2/                  # git submodule: the slim-gsgp library itself (editable install)
```

Every `__init__.py` in `src/` is a docstring only — **no re-export facades**. Import from the
real module: `from src.data.loader import load_task`, not `from src import load_task` or
`from src.data import load_task`. This was a deliberate cleanup; do not reintroduce facades.

## Hard rules

1. **No hardcoded domain values outside `src/core/schema.py`.** Category strings
   (`"Femminile"`/`"Maschile"`/`"Destra"`/`"Sinistra"`), age validity bounds, split ratios,
   the RNG seed, CSV dtypes — all live as named constants in `schema.py` and get imported
   everywhere else. If you need a new one, add it there, not inline.
2. **`pd.read_csv` always passes `dtype=CSV_DTYPES`.** Never let pandas sniff dtypes on this
   dataset — with 60 files, per-file inference risks silent dtype drift between them.
3. **Subject `Id` is cohort-local, not globally unique.** ~140 of 145 IDs collide across
   `1year`/`2year`/`3year`. Any function that filters or groups by `Id` across a multi-cohort
   DataFrame must also scope by `Cohort`, or it will silently mix different children's records.
   `get_subject_trajectory()` takes an explicit `cohort` param for this reason — use it.
4. **Don't build new re-export layers.** If a symbol needs to be reachable from a shorter
   path, that's a signal to import it directly at the call site, not to add another `__init__.py`
   export list.
5. **Validate CSV shape on ingest.** `load_task` raises `ValueError` if a file is missing an
   expected column. Keep that guard — it's what stands between a malformed data drop and a
   silently corrupted tensor.

## Testing

```bash
pytest tests
```

24 tests, all read real CSVs from `data/` (no mocked I/O — this repo is small enough that
hitting the actual files is faster and more honest than mocking). Tests pin exact row counts
(140 / 2935 / 8590), 30 kinematic features, cohort names, and rendering-without-raising for
every `render_*` function. If you change ingestion or imputation logic, diff the cleaned
DataFrame against a baseline before trusting a refactor — see the pattern used historically:
pickle `clean_and_impute_demographics(load_all())` before, compare `.equals()` after.

## Conventions

- Ratios/seed default to `DEFAULT_TRAIN_RATIO` / `DEFAULT_VAL_RATIO` / `DEFAULT_TEST_RATIO` /
  `DEFAULT_SEED` from `schema.py`. Don't re-hardcode `0.7, 0.15, 0.15` or `42` at a call site —
  import the constant, even if the value happens to match the function's own default.
- CLI flags `--cohort`/`--task` accept one value, several (`nargs="+"`), or the literal `all`.
  Omitting either means "all" (see [docs/COMMANDS.md](docs/COMMANDS.md)) — this is a deliberate,
  symmetric default; don't special-case one axis to mean something narrower.
- Prefer vectorized pandas/numpy over `.apply(lambda ...)` for anything that runs per-row across
  the full dataset — `CSV_DTYPES` already guarantees column dtypes, so `isinstance` checks on
  ingested columns are dead code.
- No comments explaining *what* code does. A comment is only for a non-obvious *why*
  (a data quirk, a workaround, an invariant that would surprise a reader).

## Known constraints

- `slim_v2/` is a git submodule (the `slim-gsgp` package, editable-installed via
  `[tool.uv.sources]` in `pyproject.toml`). `src/genetic/runner.py` imports it lazily inside
  the function body, not at module load time, so `import src...` never requires it to be
  present — `run_slim_experiment` raises `ImportError` cleanly if it's missing.
- `log/*.csv` are run logs written by the `slim_gsgp` library itself during
  `run_slim_experiment` — not something this codebase writes directly, and not meaningful
  to read outside of a debugging session on the GP run.
