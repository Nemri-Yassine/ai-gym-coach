"""Shared metrics: classification scores, confusion plots, rep-count errors."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_f1(y_true, y_pred, labels) -> dict:
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {str(label): float(score) for label, score in zip(labels, scores)}


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion(cm, labels, path, title=""):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    thresh = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rep_count_errors(true_counts, pred_counts) -> dict:
    err = np.abs(np.asarray(pred_counts, dtype=np.float64) - np.asarray(true_counts, dtype=np.float64))
    if len(err) == 0:
        return {"n": 0, "mae": None, "exact": None, "within_1": None}
    return {
        "n": int(len(err)),
        "mae": float(err.mean()),
        "exact": float(np.mean(err == 0)),
        "within_1": float(np.mean(err <= 1)),
    }
