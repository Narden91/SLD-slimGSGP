# Dataset

Kinematic data from digital stylus/tablet handwriting tasks, recorded from primary-school
children (grades 1–3), used to screen for Specific Learning Disorders (SLD) / dysgraphia.

## Layout

```
data/
  1year/Task_1.csv .. Task_21.csv   # grade 1, 21 tasks
  2year/Task_1.csv .. Task_21.csv   # grade 2, 21 tasks
  3year/Task_1.csv .. Task_18.csv   # grade 3, 18 tasks
```

| Cohort | Grade | Tasks | Unique subjects | Total rows |
|---|---|---|---|---|
| `1year` | 1 | 21 | 140 | 2,935 |
| `2year` | 2 | 21 | 145 | 3,045 |
| `3year` | 3 | 18 | 145 | 2,610 |
| **Total** | | **60 files** | ~145/cohort | **8,590** |

Each CSV is one task; one row per subject who attempted it.

## ⚠️ Subject IDs are cohort-local

`Id` is **not** globally unique — it resets per cohort. Of ~145 subjects per cohort, ~140 IDs
collide across `1year`/`2year`/`3year`. **`Id` alone never identifies a child across cohorts —
always pair it with `Cohort`.** Code that loads multiple cohorts and needs to track one
subject must filter/group by `(Cohort, Id)`, not `Id` alone.

## Columns (36 total)

| Group | Columns |
|---|---|
| Subject ID | `Id` |
| Pressure (4) | `mean:pressure`, `std:pressure`, `iqr:pressure`, `number_of_changes_in_pressure_profile` |
| Tilt (3) | `mean:tilt(on-surface)`, `std:tilt(on-surface)`, `iqr:tilt(on-surface)` |
| Temporal (5) | `writing_duration(on-surface)`, `writing_duration_overall`, `mean:stroke_duration(on-surface)`, `in_air_movement_time`, `in_air_on_paper_ratio` |
| Spatial (11) | `writing_width(on-surface)`, `writing_height(on-surface)`, `mean:stroke_width(on-surface)`, `std:stroke_width(on-surface)`, `iqr:stroke_width(on-surface)`, `mean:stroke_height(on-surface)`, `std:stroke_height(on-surface)`, `iqr:stroke_height(on-surface)`, `mean:stroke_length(on-surface)`, `number_of_changes_in_x_profile`, `number_of_changes_in_y_profile` |
| Kinematics (6) | `mean:velocity:axis-xy(on-surface)`, `std:velocity:axis-xy(on-surface)`, `mean:acceleration:axis-xy(on-surface)`, `std:acceleration:axis-xy(on-surface)`, `mean:jerk:axis-xy(on-surface)`, `average_normalized_jerk` |
| Events (1) | `number_of_interruptions` |
| Demographics (3) | `Gender`, `Age`, `Dominant_Hand` |
| Metadata (1) | `Task` |
| Target (1) | `Label` |

The 30 feature columns (Pressure through Events) are `KINEMATIC_FEATURES` in
[`src/core/schema.py`](../src/core/schema.py) — that file is the source of truth for the exact
list and grouping (`FEATURE_CATEGORIES_MAP`).

### Demographics

- `Gender`: `"Femminile"` or `"Maschile"`. Raw files also contain the literal string `"0"`
  meaning unrecorded.
- `Dominant_Hand`: `"Destra"` or `"Sinistra"`. Same `"0"`-means-unrecorded quirk.
- `Age`: float, years. Raw files contain out-of-range junk values (e.g. `-4992.0`, `0.0`) from
  the acquisition device. Valid range is `3 < Age < 20` (`AGE_MIN_VALID`/`AGE_MAX_VALID` in
  `schema.py`).

None of this is cleaned in the raw CSVs — call `clean_and_impute_demographics()`
(see [COMMANDS.md](COMMANDS.md)) before using `Gender`/`Age`/`Dominant_Hand`.

### Target: `Label`

| Value | Meaning |
|---|---|
| `0` | Best performance |
| `1` | Intermediate performance |
| `2` | Worst performance (struggling) |
| `-1` | Absent — subject did not attempt this task |

`-1` rows are real rows (the subject exists in the cohort but skipped this task), not missing
data. Loader functions take `filter_absent=True` to drop them; default is `False`.

## Ingestion contract

- Every CSV is read with an explicit `dtype=` map (`CSV_DTYPES` in `schema.py`) — no per-file
  type sniffing. `Id`/`Label`/`Task` are `int64`, all 30 kinematic features + `Age` are
  `float64`, `Gender`/`Dominant_Hand` are string.
- `load_task()` raises `ValueError` if a file is missing any expected column — a malformed
  file fails loudly instead of silently producing a frame with a missing feature.
