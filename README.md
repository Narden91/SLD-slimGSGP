# SLD-slimGSGP

**SLD-slimGSGP** is a research framework for analyzing online handwriting kinematics and applying Semantic Linear Initialization & Mutation Geometric Semantic Genetic Programming (**SLIM-GSGP**) to screen and evaluate Specific Learning Disorders (SLD) / dysgraphia in primary school children.

---

## 1. Dataset Organization

The `data/` folder contains kinematic data extracted from digital stylus/tablet recordings across 3 school cohorts:

| Cohort | Grade / Year | Tasks | Unique Subjects | Total Rows |
| :--- | :--- | :---: | :---: | :---: |
| `1year/` | Grade 1 | 21 (`Task_1.csv` .. `Task_21.csv`) | 140 | 2,935 |
| `2year/` | Grade 2 | 21 (`Task_1.csv` .. `Task_21.csv`) | 145 | 3,045 |
| `3year/` | Grade 3 | 18 (`Task_1.csv` .. `Task_18.csv`) | 145 | 2,610 |
| **Total** | | **60 CSVs** | **~145 / cohort** | **8,590** |

### Data Columns & Feature Taxonomy
Each CSV file contains **36 columns**:
- **Subject Identifier (1)**: `Id`
- **30 Kinematic Features**:
  - *Pressure (4)*: `mean:pressure`, `std:pressure`, `iqr:pressure`, `number_of_changes_in_pressure_profile`
  - *Tilt (3)*: `mean:tilt(on-surface)`, `std:tilt(on-surface)`, `iqr:tilt(on-surface)`
  - *Temporal / Duration (5)*: `writing_duration(on-surface)`, `writing_duration_overall`, `mean:stroke_duration(on-surface)`, `in_air_movement_time`, `in_air_on_paper_ratio`
  - *Spatial / Geometry (11)*: `writing_width(on-surface)`, `writing_height(on-surface)`, `mean:stroke_width(on-surface)`, `std:stroke_width(on-surface)`, `iqr:stroke_width(on-surface)`, `mean:stroke_height(on-surface)`, `std:stroke_height(on-surface)`, `iqr:stroke_height(on-surface)`, `mean:stroke_length(on-surface)`, `number_of_changes_in_x_profile`, `number_of_changes_in_y_profile`
  - *Kinematics & Dynamics (6)*: `mean:velocity:axis-xy(on-surface)`, `std:velocity:axis-xy(on-surface)`, `mean:acceleration:axis-xy(on-surface)`, `std:acceleration:axis-xy(on-surface)`, `mean:jerk:axis-xy(on-surface)`, `average_normalized_jerk`
  - *Events (1)*: `number_of_interruptions`
- **Demographics & Metadata (4)**: `Gender`, `Age` (anomalous values cleaned & imputed), `Dominant_Hand`, `Task`
- **Target Performance Score (1)**: `Label`
  - `0`: Best Performance
  - `1`: Intermediate Performance
  - `2`: Worst Performance (Struggling)
  - `-1`: Absent / Not present for the task

---

## 2. CLI Usage (`main.py`)

Run the full interactive dashboard:
```bash
python main.py
```

### Command-Line Options
```bash
# Dataset overview table across cohorts
python main.py --overview

# Inspect specific cohort
python main.py --cohort 1year

# Inspect specific task with key indicators & score distributions
python main.py --cohort 1year --task 1

# Display 30 kinematic features breakdown
python main.py --features

# View subject longitudinal trajectory across tasks
python main.py --cohort 1year --subject 1

# Demonstrate PyTorch tensor extraction and train/val/test splits
python main.py --tensor-demo

# Run end-to-end SLIM-GSGP symbolic regression benchmark on SLD task
python main.py --run-slim
```

---

## 3. Python API Quickstart

```python
import sld_data as sld

# 1. Load single task or entire cohort
df_task = sld.load_task(cohort="1year", task=1, filter_absent=True)
df_cohort = sld.load_cohort(cohort="2year")
df_all = sld.load_all()

# 2. Extract standardized PyTorch Tensors ready for SLIM-GSGP
tensors = sld.prepare_tensors(df_task, feature_set="kinematic", standardize=True)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = sld.train_val_test_split(tensors)

print("X_train shape:", X_train.shape) # (N_train, 30)
print("y_train shape:", y_train.shape) # (N_train,)
```

---

## 4. Testing

Run the test suite with `pytest`:
```bash
pytest tests
```
