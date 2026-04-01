from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..config.training import TrainingConfig


MatrixTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]


def standardize_for_linear_models(
    matrices: MatrixTriplet,
    training_config: "TrainingConfig",
    context: dict[str, Any],
) -> MatrixTriplet:
    """Return z-score standardized train/dev/test matrices.

    This transform is intended for linear models and is configured through
    TrainingConfig.preprocess_methods.
    """
    _ = context
    if not training_config.standardize_linear_models:
        return matrices

    x_train, x_dev, x_test = matrices
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_dev - mean) / std, (x_test - mean) / std
