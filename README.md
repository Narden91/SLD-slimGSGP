# SLD-slimGSGP

**SLD-slimGSGP** is a research framework for analyzing online handwriting kinematics and applying Semantic Linear Initialization & Mutation Geometric Semantic Genetic Programming (**SLIM-GSGP**) to screen and evaluate Specific Learning Disorders (SLD) / dysgraphia in primary school children.

## Docs

- [docs/DATA.md](docs/DATA.md) — dataset layout, columns, demographics, target labels
- [docs/COMMANDS.md](docs/COMMANDS.md) — CLI flags and Python API
- [CLAUDE.md](CLAUDE.md) — architecture and conventions for contributors

## Quickstart

```bash
uv sync
python main.py            # full dashboard
pytest tests               # run the test suite
```

```python
from src.data.loader import load_task
from src.data.preprocessor import prepare_tensors

df = load_task("1year", 1, filter_absent=True)
tensors = prepare_tensors(df, feature_set="kinematic", standardize=True)
```

See [docs/COMMANDS.md](docs/COMMANDS.md) for the full CLI and API reference.
