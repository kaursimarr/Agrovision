"""Matplotlib figures: training curves, confusion matrices, model comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from .labels import pretty_label

COLORS = {"cnn": "#d95f02", "mobilenetv2": "#1b9e77", "resnet50": "#7570b3", "vit": "#e7298a"}


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, path, title: str) -> None:
    """Accuracy and loss (train vs validation). Dashed line marks the start of fine-tuning."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    boundary = history.get("phase_boundary")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, history[key], marker="o", label=f"Train {label.lower()}")
        ax.plot(epochs, history[f"val_{key}"], marker="s", label=f"Validation {label.lower()}")
        if boundary:
            ax.axvline(boundary + 0.5, ls="--", color="gray", label="Fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{title}: {label}")
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)


def plot_history_comparison(histories: dict, path, display_names: dict | None = None) -> None:
    """Validation accuracy / loss of all models on the same axes."""
    display_names = display_names or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        epochs = np.arange(1, len(h["loss"]) + 1)
        for ax, key in ((axes[0], "val_accuracy"), (axes[1], "val_loss")):
            ax.plot(epochs, h[key], marker="o", color=COLORS.get(name), label=display_names.get(name, name))
    axes[0].set_title("Validation accuracy")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch (fine-tuning epochs continue the same axis)")
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)


def plot_confusion_matrix(cm, class_names, path, normalize: bool = False, title: str = "Confusion matrix") -> None:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    size = max(6.0, 0.42 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [pretty_label(c) for c in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=max(5, 9 - n // 8))
    ax.set_yticklabels(labels, fontsize=max(5, 9 - n // 8))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    if n <= 20:  # annotate cells only when readable
        thresh = cm.max() / 2 if cm.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")
    _save(fig, path)


def plot_metrics_comparison(table: dict, path, display_names: dict | None = None) -> None:
    """Grouped bars: accuracy / precision / recall / F1 for every model.

    ``table`` maps model name -> metrics dict (as produced by ``compute_metrics``).
    """
    display_names = display_names or {}
    keys = [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
            ("recall_macro", "Recall"), ("f1_macro", "F1 (macro)")]
    names = list(table)
    x = np.arange(len(keys))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(names):
        vals = [table[name][k] for k, _ in keys]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, color=COLORS.get(name),
                      label=display_names.get(name, name))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys])
    lo = min(min(table[n][k] for k, _ in keys) for n in names)
    ax.set_ylim(max(0.0, lo - 0.1), 1.03)
    ax.set_title("Test-set metrics: CNN vs transfer learning")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, path)
