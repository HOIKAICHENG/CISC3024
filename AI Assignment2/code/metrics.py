import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.clip(probs, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def classification_metrics(
    probs: np.ndarray, targets: np.ndarray, bins: int = 15
) -> dict[str, float]:
    safe = np.clip(probs, 1e-12, 1.0)
    predictions = safe.argmax(axis=1)
    confidence = safe.max(axis=1)
    accuracy = (predictions == targets).mean()
    nll = -np.log(safe[np.arange(len(targets)), targets]).mean()
    one_hot = np.eye(safe.shape[1])[targets]
    brier = np.square(safe - one_hot).sum(axis=1).mean()

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_accuracy = (predictions[mask] == targets[mask]).mean()
            ece += mask.mean() * abs(bin_accuracy - confidence[mask].mean())

    return {
        "accuracy": float(accuracy),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
    }


def ood_metrics(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> dict[str, float]:
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    indices = np.flatnonzero(tpr >= 0.95)
    fpr95 = fpr[indices[0]] if len(indices) else 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": float(fpr95),
    }
