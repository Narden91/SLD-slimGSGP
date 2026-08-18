"""
Unit tests for genetic programming runner and SLIM-GSGP integration.
"""

import pytest
import torch
from src.data.loader import load_task
from src.data.preprocessor import prepare_tensors
from src.genetic.runner import SlimConfig, SlimResult, run_slim_experiment


def test_slim_experiment_execution():
    df = load_task("1year", 1, filter_absent=True)
    tensors = prepare_tensors(df, feature_set="kinematic", standardize=True)

    config = SlimConfig(
        pop_size=10,
        n_iter=2,
        slim_version="SLIM+SIG2",
        seed=42,
        verbose=False,
    )

    result = run_slim_experiment(tensors, config=config, dataset_name="test_run")

    assert isinstance(result, SlimResult)
    assert result.train_rmse >= 0.0
    assert result.val_rmse >= 0.0
    assert result.test_rmse >= 0.0
    assert result.test_mae >= 0.0
    assert len(result.test_predictions) == result.metadata["n_test"]
    assert result.elapsed_seconds > 0.0
