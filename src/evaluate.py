import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    fbeta_score,
    confusion_matrix,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)

from src.config import setup_logger

logger = setup_logger(__name__)


def find_best_threshold(y_true, y_proba, beta=1.0) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # dropping the last element as the precision_recall_curve returns N+1 
    # precisions/recalls for N thresholds
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # computing f-beta score for each threshold
    numerator = (1 + beta**2) * precisions * recalls
    denominator = (beta**2  * precisions) + recalls
    f_scores = np.divide(
                    numerator, 
                    denominator, 
                    out=np.zeros_like(denominator, dtype=float), 
                    where=denominator > 0
                )

    # finding the best threshold
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_score = f_scores[best_idx]

    logger.info(f"Best threshold: {best_threshold:.3f} | "
                f"F-beta: {best_score:.3f}")

    return float(best_threshold), float(f_scores[best_idx])


def compute_metrics(y_true, y_proba, threshold) -> dict[str, float]:
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # finding predictions based on threshold
    y_pred = (y_proba >= threshold).astype(int)

    # computing metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    logger.info(f"Precision: {precision:.3f} | Recall: {recall:.3f} | "
                f"F1: {f1:.3f} | F2: {f2:.3f}")

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    logger.info(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    return {
        "threshold": float(threshold),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_pr_curve(y_true, y_proba, title):
    fig, ax = plt.subplots(figsize=(10,10))
    display = PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"{title} - Precision-Recall Curve", fontsize=16)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(10,10))
    display = ConfusionMatrixDisplay.from_predictions(
                    y_true, 
                    y_pred, 
                    display_labels=["Legit", "Fraud"],
                    ax=ax
                )
    ax.set_title("Confusion Matrix", fontsize=16)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    y_true = np.array([0,1,1,0,1])
    y_proba = np.array([0.05, 0.20, 0.40, 0.35, 0.90])
    threshold, score = find_best_threshold(y_true, y_proba)
    metrics = compute_metrics(y_true, y_proba, threshold)
    pr_fig = plot_pr_curve(y_true, y_proba, "Test")
    cm_fig = plot_confusion_matrix(y_true, y_proba, threshold)
    print(metrics)
    plt.show()
    plt.close('all')
