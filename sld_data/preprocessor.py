"""
Data preprocessor and PyTorch tensor preparation module for SLD handwriting data.
Provides anomaly imputation, demographic encoding, feature scaling, and tensor conversions.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from sld_data.schema import (
    DEMOGRAPHIC_COLUMNS,
    KINEMATIC_FEATURES,
    PreparedTensors,
    TARGET_COLUMN,
)


def clean_and_impute_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and impute demographic columns (Age, Gender, Dominant_Hand).
    
    1. Subject-level propagation: If a subject has valid demographic records in some tasks,
       propagate them to tasks where they were unrecorded ('0' or <= 0).
    2. Cohort-level imputation: For any remaining unrecorded values, impute cohort median Age
       and mode Gender / Dominant_Hand.
    
    Returns:
        pd.DataFrame: A copy of the DataFrame with cleaned and imputed demographic columns.
    """
    df = df.copy()

    # Determine cohort group if available, otherwise entire dataframe
    group_cols = ["Cohort", "Id"] if "Cohort" in df.columns else ["Id"]

    # 1. Clean Age anomalies (negative numbers like -4992.0 or 0.0)
    df["Age_valid"] = df["Age"].apply(lambda x: x if (isinstance(x, (int, float)) and x > 3 and x < 20) else np.nan)
    
    # Propagate valid age per subject
    df["Age"] = df.groupby(group_cols)["Age_valid"].transform(lambda s: s.ffill().bfill())
    
    # Impute remaining missing age with cohort median (or overall median, fallback 7.0)
    if "Cohort" in df.columns:
        cohort_age_median = df.groupby("Cohort")["Age"].transform("median")
        df["Age"] = df["Age"].fillna(cohort_age_median).fillna(7.0)
    else:
        df["Age"] = df["Age"].fillna(df["Age"].median()).fillna(7.0)
    
    df.drop(columns=["Age_valid"], inplace=True)

    # 2. Clean Gender ('Femminile', 'Maschile', '0')
    df["Gender_valid"] = df["Gender"].apply(lambda x: str(x).strip() if str(x).strip() in ["Femminile", "Maschile"] else np.nan)
    df["Gender"] = df.groupby(group_cols)["Gender_valid"].transform(lambda s: s.ffill().bfill())
    df["Gender"] = df["Gender"].fillna("Femminile")  # Mode
    df.drop(columns=["Gender_valid"], inplace=True)

    # 3. Clean Dominant_Hand ('Destra', 'Sinistra', '0')
    df["Hand_valid"] = df["Dominant_Hand"].apply(lambda x: str(x).strip() if str(x).strip() in ["Destra", "Sinistra"] else np.nan)
    df["Dominant_Hand"] = df.groupby(group_cols)["Hand_valid"].transform(lambda s: s.ffill().bfill())
    df["Dominant_Hand"] = df["Dominant_Hand"].fillna("Destra")  # Mode
    df.drop(columns=["Hand_valid"], inplace=True)

    return df


def encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical demographic features into numeric values:
    - Gender: 'Femminile' -> 0.0, 'Maschile' -> 1.0
    - Dominant_Hand: 'Destra' -> 0.0, 'Sinistra' -> 1.0
    """
    df = df.copy()
    if "Gender" in df.columns:
        df["Gender_encoded"] = df["Gender"].map({"Femminile": 0.0, "Maschile": 1.0}).fillna(0.0).astype(float)
    if "Dominant_Hand" in df.columns:
        df["Hand_encoded"] = df["Dominant_Hand"].map({"Destra": 0.0, "Sinistra": 1.0}).fillna(0.0).astype(float)
    return df


def prepare_tensors(
    df: pd.DataFrame,
    feature_set: Union[str, List[str]] = "kinematic",
    filter_absent: bool = True,
    standardize: bool = False,
    target_col: str = TARGET_COLUMN,
    dtype: torch.dtype = torch.float32,
) -> PreparedTensors:
    """
    Convert a DataFrame into clean PyTorch Tensors ready for slim-gsgp or standard ML models.

    Args:
        df: Input DataFrame containing task data
        feature_set: 'kinematic' (30 kinematic features), 'all' (kinematics + Age + encoded demographics),
                     or a custom List[str] of feature column names.
        filter_absent: If True (default), filter out absent subjects (Label == -1).
        standardize: If True, apply z-score normalization (mean=0, std=1) to feature matrix X.
        target_col: Target column name (default 'Label').
        dtype: PyTorch tensor data type (default torch.float32).

    Returns:
        PreparedTensors: Dataclass containing X tensor (N, D), y tensor (N,), and metadata.
    """
    # 1. Clean and impute demographics
    cleaned_df = clean_and_impute_demographics(df)
    encoded_df = encode_demographics(cleaned_df)

    # 2. Filter absent subjects (Label == -1) if requested
    if filter_absent and target_col in encoded_df.columns:
        data = encoded_df[encoded_df[target_col] != -1].reset_index(drop=True)
    else:
        data = encoded_df.reset_index(drop=True)

    # 3. Determine feature columns
    if isinstance(feature_set, list):
        feature_names = feature_set
    elif feature_set == "kinematic":
        feature_names = [f for f in KINEMATIC_FEATURES if f in data.columns]
    elif feature_set == "all":
        demo_encoded = ["Age", "Gender_encoded", "Hand_encoded"]
        feature_names = [f for f in KINEMATIC_FEATURES if f in data.columns] + [
            f for f in demo_encoded if f in data.columns
        ]
    else:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Use 'kinematic', 'all', or list of column names.")

    # 4. Extract feature matrix X and target y
    X_mat = data[feature_names].to_numpy(dtype=np.float32)
    
    # Handle NaNs in X if any by column mean imputation
    if np.isnan(X_mat).any():
        col_means = np.nanmean(X_mat, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(np.isnan(X_mat))
        X_mat[inds] = np.take(col_means, inds[1])

    # 5. Optional Z-score standardization
    if standardize:
        mean = np.mean(X_mat, axis=0)
        std = np.std(X_mat, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero for constant features
        X_mat = (X_mat - mean) / std

    # 6. Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(np.array(X_mat, copy=True)).to(dtype)
    y_tensor = (
        torch.from_numpy(np.array(data[target_col].to_numpy(dtype=np.float32), copy=True)).to(dtype)
        if target_col in data.columns
        else torch.empty(0)
    )

    subject_ids = (
        torch.from_numpy(np.array(data["Id"].to_numpy(dtype=np.int64), copy=True))
        if "Id" in data.columns
        else torch.empty(0, dtype=torch.int64)
    )

    task_ids = (
        torch.from_numpy(np.array(data["Task"].to_numpy(dtype=np.int64), copy=True))
        if "Task" in data.columns
        else torch.empty(0, dtype=torch.int64)
    )

    cohorts = data["Cohort"].tolist() if "Cohort" in data.columns else []

    return PreparedTensors(
        X=X_tensor,
        y=y_tensor,
        feature_names=feature_names,
        subject_ids=subject_ids,
        task_ids=task_ids,
        cohorts=cohorts,
    )


def train_val_test_split(
    tensors: PreparedTensors,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Split PreparedTensors into Train, Validation, and Test sets (X, y) tuples.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    N = tensors.X.shape[0]

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    indices = torch.randperm(N) if shuffle else torch.arange(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_set = (tensors.X[train_idx], tensors.y[train_idx])
    val_set = (tensors.X[val_idx], tensors.y[val_idx])
    test_set = (tensors.X[test_idx], tensors.y[test_idx])

    return train_set, val_set, test_set
