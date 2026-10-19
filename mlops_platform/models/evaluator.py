"""
Model Evaluation & Artifact Generation
======================================
Computes comprehensive classification metrics and generates evaluation plots
(ROC Curve, Precision-Recall Curve, Confusion Matrix, Feature Importance).
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    """Encapsulates all evaluation metrics for a model split."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Computes production evaluation metrics and generates visual evaluation artifacts."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> EvaluationResult:
        """Calculate complete suite of classification metrics."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, y_prob)),
            log_loss=float(log_loss(y_true, y_prob)),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_positives=int(tp),
        )

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str = "ROC Curve"):
        """Generate and save ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
        """Generate and save Precision-Recall curve."""
        prec, rec, _ = precision_recall_curve(y_true, y_prob)

        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, color="#059669", lw=2, label="Precision-Recall Curve")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm_dict: dict[str, int], output_path: Path):
        """Generate formatted confusion matrix heatmap."""
        matrix = np.array([
            [cm_dict["true_negatives"], cm_dict["false_positives"]],
            [cm_dict["false_negatives"], cm_dict["true_positives"]],
        ])

        fig, ax = plt.subplots(figsize=(5, 4.5))
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="black", fontsize=13, fontweight="bold")

        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Retain (0)", "Pred: Churn (1)"], fontsize=10)
        ax.set_yticklabels(["Actual: Retain (0)", "Actual: Churn (1)"], fontsize=10)
        plt.title("Confusion Matrix", fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_feature_importance(feature_names: list[str], importances: np.ndarray, output_path: Path, top_n: int = 10):
        """Generate horizontal bar chart of top N feature importances."""
        indices = np.argsort(importances)[-top_n:]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(7, 5))
        plt.barh(range(len(sorted_features)), sorted_importances, color="#6366F1", align="center")
        plt.yticks(range(len(sorted_features)), sorted_features, fontsize=10)
        plt.xlabel("Relative Importance", fontsize=11)
        plt.title(f"Top {top_n} Feature Importances", fontsize=12, fontweight="bold")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
