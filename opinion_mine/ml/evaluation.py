"""This model contains utility functions for evaluating/plotting machine learning pipelines."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import learning_curve


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], cmap: str = "Set3"
) -> None:
    """
    Create and plot a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples,)
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,)
    labels : list[str]
        List of label names
    cmap : str, optional
        Colormap for the heatmap, by default "Set3"

    Returns
    -------
    None

    Notes
    -----
    This function creates a confusion matrix from the true and predicted labels,
    and then plots it as a heatmap using seaborn.
    """
    # Create the confusion matrix.
    cm: np.ndarray = confusion_matrix(y_true, y_pred)

    # Plot confusion_matrix.
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(cm, annot=True, cmap=cmap, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_yticklabels(labels, rotation=0)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def calculate_precision_recall_curves(
    train_class: np.ndarray,
    test_class: np.ndarray,
    y_proba_train: np.ndarray,
    y_proba_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Calculate precision-recall curves and average precision scores for train and test data.

    Parameters
    ----------
    train_class : np.ndarray
        True labels for training data, shape (n_samples,)
    test_class : np.ndarray
        True labels for test data, shape (n_samples,)
    y_proba_train : np.ndarray
        Predicted probabilities for training data, shape (n_samples,)
    y_proba_test : np.ndarray
        Predicted probabilities for test data, shape (n_samples,)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]
        Precision and recall values for train and test data, and their average precision scores
        (precision_train, recall_train, precision_test, recall_test, ap_train, ap_test)
    """
    # Calculate precision-recall curves
    precision_train, recall_train, _ = precision_recall_curve(train_class, y_proba_train)
    precision_test, recall_test, _ = precision_recall_curve(test_class, y_proba_test)

    # Calculate average precision scores
    ap_train: float = average_precision_score(train_class, y_proba_train)
    ap_test: float = average_precision_score(test_class, y_proba_test)

    return precision_train, recall_train, precision_test, recall_test, ap_train, ap_test


def plot_precision_recall_curves(
    recall_train: np.ndarray,
    precision_train: np.ndarray,
    recall_test: np.ndarray,
    precision_test: np.ndarray,
    ap_train: float,
    ap_test: float,
) -> None:
    """
    Plot precision-recall curves for train and test data.

    Parameters
    ----------
    recall_train : np.ndarray
        Recall values for training data, shape (n_thresholds,)
    precision_train : np.ndarray
        Precision values for training data, shape (n_thresholds,)
    recall_test : np.ndarray
        Recall values for test data, shape (n_thresholds,)
    precision_test : np.ndarray
        Precision values for test data, shape (n_thresholds,)
    ap_train : float
        Average precision score for training data
    ap_test : float
        Average precision score for test data

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 8))
    plt.plot(
        recall_train,
        precision_train,
        color="blue",
        lw=2,
        label=f"Train (Avg Precision = {ap_train:.2f})",
    )
    plt.plot(
        recall_test,
        precision_test,
        color="darkorange",
        lw=2,
        label=f"Test (Avg Precision = {ap_test:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def plot_roc_curves(
    y_train: np.ndarray,
    y_train_proba: np.ndarray,
    y_test: np.ndarray,
    y_test_proba: np.ndarray,
) -> None:
    """
    Plot ROC curves for train and test sets.

    Parameters
    ----------
    y_train : np.ndarray
        True labels for train set.
    y_train_proba : np.ndarray
        Predicted probabilities for train set.
    y_test : np.ndarray
        True labels for test set.
    y_test_proba : np.ndarray
        Predicted probabilities for test set.

    Returns
    -------
    None
    """
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, y_train_proba)
    roc_auc_train = metrics.roc_auc_score(y_train, y_train_proba)

    fpr_test, tpr_test, _ = metrics.roc_curve(y_test, y_test_proba)
    roc_auc_test = metrics.roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr_train,
        tpr_train,
        color="blue",
        lw=2,
        label=f"Train ROC curve (AUC = {roc_auc_train:.2f})",
    )
    plt.plot(
        fpr_test,
        tpr_test,
        color="darkorange",
        lw=2,
        label=f"Test ROC curve (AUC = {roc_auc_test:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curve(
    estimator: object,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    cv: int = 5,
) -> None:
    """
    Plot the learning curve for a given estimator.

    Parameters
    ----------
    estimator : object
        The machine learning model to evaluate.
    X : np.ndarray of shape (n_samples, n_features)
        The input samples.
    y : np.ndarray of shape (n_samples,)
        The target values.
    train_sizes : np.ndarray of shape (n_points,), default=np.linspace(0.1, 1.0, 10)
        The points of the learning curve to evaluate.
    cv : int, default=5
        The number of folds in cross-validation.

    Returns
    -------
    None
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        # n_jobs=1,
    )

    train_mean: np.ndarray = np.mean(train_scores, axis=1)
    train_std: np.ndarray = np.std(train_scores, axis=1)
    test_mean: np.ndarray = np.mean(test_scores, axis=1)
    test_std: np.ndarray = np.std(test_scores, axis=1)

    plt.figure(figsize=(6, 6))
    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training accuracy",
    )

    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation accuracy",
    )

    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.5, 1.03])
    plt.tight_layout()
    plt.show()


def plot_cumulative_variance(estimator: TruncatedSVD) -> None:
    """
    Plot the cumulative variance percentage for a TruncatedSVD estimator.

    Parameters
    ----------
    estimator : TruncatedSVD
        The fitted TruncatedSVD estimator.

    Returns
    -------
    None
        This function doesn't return anything, it displays a plot.

    Notes
    -----
    The plot shows the cumulative explained variance ratio as a function
    of the number of components.
    """
    # Plot the cumulative variance percentage.
    explained: np.ndarray = estimator.explained_variance_ratio_.cumsum()  # shape: (n_components,)

    plt.figure(figsize=(6, 6))
    plt.plot(explained, ".-", ms=6, color="b")
    plt.xlabel("Num of components", fontsize=14)
    plt.ylabel("Cumulative variance percentage", fontsize=12)
    plt.title("Cumulative variance percentage", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
