"""This module contains funtions for training ML models."""

import time
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold


def calculate_class_weights(n_samples: int, total_samples: int, n_classes: int) -> float:
    """
    Calculate class weights for imbalanced datasets.

    Parameters
    ----------
    n_samples : int
        Number of samples in a specific class.
    total_samples : int
        Total number of samples in the dataset.
    n_classes : int
        Number of classes in the dataset.

    Returns
    -------
    float
        The calculated class weight, rounded to 4 decimal places.
    """
    return np.round(total_samples / (n_samples * n_classes), 4)


def train_model_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_splits: int = 5,
) -> tuple[Any, list[float], float, float]:
    """
    Train a model using cross-validation and return performance metrics.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target labels array of shape (n_samples,).
    estimator : Any
        The machine learning model to be trained and evaluated.
    n_splits : int, optional
        Number of splits for cross-validation, by default 5.

    Returns
    -------
    tuple[Any, list[float], float, float]
        A tuple containing:
        - The trained estimator
        - List of accuracy scores for each fold
        - Mean accuracy across all folds
        - Standard deviation of accuracy across all folds
    """
    start_time: float = time.time()
    kfold: StratifiedKFold = StratifiedKFold(n_splits=n_splits).split(X, y)

    scores: list[float] = []

    for k, (train, test) in enumerate(kfold):
        estimator.fit(X[train], y[train])
        score: float = estimator.score(X[test], y[test])
        scores.append(score)
        print(f"Fold: {k+1:2d} | Class dist.: {np.bincount(y[train])} | Acc: {score:.3f}")  # noqa

    mean_accuracy: float = np.mean(scores)
    std_accuracy: float = np.std(scores)
    stop_time: float = time.time()
    print(f"\nCV accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")  # noqa
    print(f"\nTime taken: {stop_time - start_time:.3f} seconds")  # noqa

    return estimator, scores, mean_accuracy, std_accuracy
