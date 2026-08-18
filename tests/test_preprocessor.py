"""
Unit tests for data preprocessor module.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.data.loader import load_cohort, load_task
from src.data.preprocessor import (
    clean_and_impute_demographics,
    encode_demographics,
    prepare_tensors,
    train_val_test_split,
)


def test_clean_and_impute_demographics():
    # Load 3year which contains -4992.0 in age and '0' in demographics
    df_3y = load_cohort("3year")
    cleaned = clean_and_impute_demographics(df_3y)

    assert (cleaned["Age"] <= 0).sum() == 0
    assert (cleaned["Age"] < 3).sum() == 0
    assert (cleaned["Gender"] == "0").sum() == 0
    assert (cleaned["Dominant_Hand"] == "0").sum() == 0
    assert set(cleaned["Gender"].unique()).issubset({"Femminile", "Maschile"})
    assert set(cleaned["Dominant_Hand"].unique()).issubset({"Destra", "Sinistra"})


def test_encode_demographics():
    df = load_task("1year", 1)
    cleaned = clean_and_impute_demographics(df)
    encoded = encode_demographics(cleaned)

    assert "Gender_encoded" in encoded.columns
    assert "Hand_encoded" in encoded.columns
    assert set(encoded["Gender_encoded"].unique()).issubset({0.0, 1.0})
    assert set(encoded["Hand_encoded"].unique()).issubset({0.0, 1.0})


def test_prepare_tensors_kinematic():
    df = load_task("1year", 1, filter_absent=True)
    tensors = prepare_tensors(df, feature_set="kinematic", standardize=False)

    assert isinstance(tensors.X, torch.Tensor)
    assert isinstance(tensors.y, torch.Tensor)
    assert tensors.X.shape[0] == len(df)
    assert tensors.X.shape[1] == 30  # 30 kinematic features
    assert tensors.y.shape[0] == len(df)
    assert (tensors.y == -1).sum() == 0


def test_prepare_tensors_standardize():
    df = load_task("1year", 1, filter_absent=True)
    tensors = prepare_tensors(df, feature_set="kinematic", standardize=True)

    X_np = tensors.X.numpy()
    means = np.mean(X_np, axis=0)
    stds = np.std(X_np, axis=0)

    # Tolerances for mean approx 0 and std approx 1 (or 0 for constant features)
    assert np.allclose(means, 0.0, atol=1e-4)
    for s in stds:
        assert np.isclose(s, 1.0, atol=1e-4) or np.isclose(s, 0.0, atol=1e-4)


def test_train_val_test_split():
    df = load_task("1year", 1, filter_absent=True)
    tensors = prepare_tensors(df, feature_set="kinematic")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(
        tensors, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    total_samples = len(df)
    assert len(X_tr) + len(X_va) + len(X_te) == total_samples
    assert len(X_tr) == len(y_tr)
    assert len(X_va) == len(y_va)
    assert len(X_te) == len(y_te)
