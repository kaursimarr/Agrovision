"""Evaluation metrics (scikit-learn / numpy only)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    k = min(k, probs.shape[1])
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y in row for y, row in zip(y_true, top_k)]))


def compute_metrics(y_true, y_pred, class_names, probs=None) -> dict:
    """Overall + per-class metrics and the confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "num_test_samples": int(len(y_true)),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probs is not None:
        result["top3_accuracy"] = top_k_accuracy(np.asarray(probs), y_true, 3)
    return result
