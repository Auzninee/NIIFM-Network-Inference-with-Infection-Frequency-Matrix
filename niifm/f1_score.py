import numpy as np


def _binarize_adjacency(A: np.ndarray, threshold: float = 0.0, diag_zero: bool = True) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    if threshold == 0.0:
        B = (A != 0).astype(np.int8)
    else:
        B = (A > threshold).astype(np.int8)

    if diag_zero:
        np.fill_diagonal(B, 0)
    return B


def F1Score_undirected_threshold(A_true: np.ndarray, Q_pred: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute F1-score (undirected) at a fixed threshold.
    - A_true: nonzero treated as 1
    - Q_pred: > threshold treated as 1
    """
    A_true_bin = _binarize_adjacency(A_true, threshold=0.0, diag_zero=True)
    A_pred_bin = _binarize_adjacency(Q_pred, threshold=threshold, diag_zero=True)

    TP = np.sum((A_true_bin + A_pred_bin) == 2) / 2.0
    FP = np.sum((A_true_bin - A_pred_bin) == -1) / 2.0
    FN = np.sum((A_true_bin - A_pred_bin) == 1) / 2.0

    if TP + FP == 0 or TP + FN == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def F1Score_undirected_best_threshold(
    A_true: np.ndarray,
    Q_pred: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Sweep thresholds to find the best F1-score (undirected).

    Returns dict:
      best_threshold, best_f1, thresholds, f1_scores
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    thresholds = np.asarray(thresholds, dtype=float)

    f1_scores = np.zeros_like(thresholds, dtype=float)
    for i, thr in enumerate(thresholds):
        f1_scores[i] = F1Score_undirected_threshold(A_true, Q_pred, threshold=float(thr))

    best_idx = int(np.argmax(f1_scores))
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "thresholds": thresholds,
        "f1_scores": f1_scores,
    }
