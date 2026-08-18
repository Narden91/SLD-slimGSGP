"""
Genetic programming execution engine and SLIM-GSGP runner for SLD handwriting prediction.
Provides typed configuration, training execution, model evaluation, and result tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import time
import torch

from src.core.schema import PreparedTensors
from src.data.preprocessor import train_val_test_split


@dataclass
class SlimConfig:
    """Hyperparameter configuration for SLIM-GSGP genetic programming runs."""
    pop_size: int = 50
    n_iter: int = 20
    slim_version: str = "SLIM+SIG2"  # 'SLIM+SIG2', 'SLIM*SIG2', 'SLIM+SIG1', etc.
    p_inflate: float = 0.5
    ms_lower: float = 0.0
    ms_upper: float = 1.0
    reconstruct: bool = True
    seed: int = 42
    verbose: int = 0


@dataclass
class SlimResult:
    """Container for SLIM-GSGP training and evaluation results."""
    model: Any
    train_rmse: float
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_predictions: torch.Tensor
    elapsed_seconds: float
    config: SlimConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<SlimResult test_rmse={self.test_rmse:.4f} test_mae={self.test_mae:.4f} "
            f"elapsed={self.elapsed_seconds:.2f}s pop_size={self.config.pop_size}>"
        )


def run_slim_experiment(
    data: Union[PreparedTensors, Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    config: Optional[SlimConfig] = None,
    dataset_name: str = "sld_task",
) -> SlimResult:
    """
    Execute a full SLIM-GSGP symbolic regression / classification experiment.

    Args:
        data: Either a PreparedTensors instance or a pre-split ((X_train, y_train), (X_val, y_val), (X_test, y_test)) tuple.
        config: SlimConfig instance containing GP hyperparameters.
        dataset_name: Identifier name for the run.

    Returns:
        SlimResult: Complete metrics, trained symbolic tree model, predictions, and elapsed time.
    """
    if config is None:
        config = SlimConfig()

    # Unpack or split data
    if isinstance(data, PreparedTensors):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
            data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=config.seed
        )
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

    from slim_gsgp.main_slim import slim
    from slim_gsgp.evaluators.fitness_functions import rmse, mae

    start_time = time.perf_counter()

    final_tree = slim(
        X_train=X_train,
        y_train=y_train,
        X_test=X_val,
        y_test=y_val,
        dataset_name=dataset_name,
        slim_version=config.slim_version,
        pop_size=config.pop_size,
        n_iter=config.n_iter,
        p_inflate=config.p_inflate,
        ms_lower=config.ms_lower,
        ms_upper=config.ms_upper,
        reconstruct=config.reconstruct,
        seed=config.seed,
        verbose=config.verbose,
    )

    elapsed = time.perf_counter() - start_time

    # Evaluate on Train, Val, and Test
    train_preds = final_tree.predict(X_train)
    val_preds = final_tree.predict(X_val)
    test_preds = final_tree.predict(X_test)

    train_loss = float(rmse(y_true=y_train, y_pred=train_preds))
    val_loss = float(rmse(y_true=y_val, y_pred=val_preds))
    test_loss = float(rmse(y_true=y_test, y_pred=test_preds))
    test_mae_val = float(mae(y_true=y_test, y_pred=test_preds))

    return SlimResult(
        model=final_tree,
        train_rmse=train_loss,
        val_rmse=val_loss,
        test_rmse=test_loss,
        test_mae=test_mae_val,
        test_predictions=test_preds,
        elapsed_seconds=elapsed,
        config=config,
        metadata={
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "dataset_name": dataset_name,
        },
    )
