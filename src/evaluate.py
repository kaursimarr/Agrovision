"""Evaluate trained models on the held-out test set and compare them.

    python -m src.evaluate                      # every model that has been trained
    python -m src.evaluate --models cnn resnet50

Outputs (under outputs/):
    <model>/metrics.json, classification_report.csv,
    <model>/confusion_matrix.png, <model>/confusion_matrix_normalized.png
    comparison/metrics_comparison.csv / .png, comparison/training_curves_comparison.png
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import keras

from . import config as C
from .data_loader import make_dataset
from .metrics import compute_metrics
from .models import (AddPositionEmbs,  # noqa: F401  (registers the custom layer for loading)
                    ResNetPreprocess)
from .plots import (plot_confusion_matrix, plot_history_comparison, plot_metrics_comparison)


def evaluate_model(name: str, split: dict, batch_size: int) -> dict:
    class_names = split["class_names"]
    model = keras.models.load_model(C.model_path(name), compile=False)
    test = split["test"]
    test_ds = make_dataset(split["meta"]["root"], test["paths"], test["labels"], batch_size, training=False)

    start = time.time()
    probs = model.predict(test_ds, verbose=0)
    seconds = time.time() - start
    y_true = np.asarray(test["labels"])
    y_pred = probs.argmax(axis=1)

    result = compute_metrics(y_true, y_pred, class_names, probs)
    result.update(
        model=name,
        params_millions=model.count_params() / 1e6,
        size_mb=C.model_path(name).stat().st_size / 1e6,
        ms_per_image=1000 * seconds / len(y_true),   # includes image decoding + resizing
    )

    out_dir = C.OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["report"]).T.to_csv(out_dir / "classification_report.csv")
    slim = {k: v for k, v in result.items() if k not in ("report", "confusion_matrix")}
    (out_dir / "metrics.json").write_text(json.dumps(slim, indent=2))
    title = C.DISPLAY_NAMES[name]
    plot_confusion_matrix(result["confusion_matrix"], class_names, out_dir / "confusion_matrix.png",
                          normalize=False, title=title)
    plot_confusion_matrix(result["confusion_matrix"], class_names, out_dir / "confusion_matrix_normalized.png",
                          normalize=True, title=title)
    print(f"[eval] {title}: acc={result['accuracy']:.4f} f1_macro={result['f1_macro']:.4f} "
          f"top3={result.get('top3_accuracy', float('nan')):.4f}")
    return result


def build_comparison(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        hist_path = C.OUTPUTS_DIR / name / "history.json"
        hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
        rows.append({
            "Model": C.DISPLAY_NAMES[name],
            "Test accuracy": round(r["accuracy"], 4),
            "Precision (macro)": round(r["precision_macro"], 4),
            "Recall (macro)": round(r["recall_macro"], 4),
            "F1 (macro)": round(r["f1_macro"], 4),
            "F1 (weighted)": round(r["f1_weighted"], 4),
            "Top-3 accuracy": round(r.get("top3_accuracy", float("nan")), 4),
            "Params (M)": round(r["params_millions"], 2),
            "Size (MB)": round(r["size_mb"], 1),
            "Inference (ms/img)": round(r["ms_per_image"], 1),
            "Epochs run": hist.get("epochs_run"),
            "Train time (min)": round(hist["train_time_sec"] / 60, 1) if "train_time_sec" in hist else None,
        })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(C.MODEL_NAMES), choices=C.MODEL_NAMES)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    args = p.parse_args(argv)

    if not C.SPLIT_PATH.exists():
        raise SystemExit("outputs/split.json not found - run `python -m src.train` first.")
    split = json.loads(C.SPLIT_PATH.read_text())

    results = {}
    for name in args.models:
        if not C.model_path(name).exists():
            print(f"[eval] skipping {name}: {C.model_path(name)} not found")
            continue
        results[name] = evaluate_model(name, split, args.batch_size)
    if not results:
        raise SystemExit("No trained models found in models/.")

    C.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    table = build_comparison(results)
    table.to_csv(C.COMPARISON_DIR / "metrics_comparison.csv", index=False)
    plot_metrics_comparison(results, C.COMPARISON_DIR / "metrics_comparison.png", C.DISPLAY_NAMES)

    histories = {n: json.loads((C.OUTPUTS_DIR / n / "history.json").read_text())
                 for n in results if (C.OUTPUTS_DIR / n / "history.json").exists()}
    if histories:
        plot_history_comparison(histories, C.COMPARISON_DIR / "training_curves_comparison.png", C.DISPLAY_NAMES)

    print("\n" + table.to_string(index=False))
    print(f"\nSaved comparison to {C.COMPARISON_DIR}")


if __name__ == "__main__":
    main()
