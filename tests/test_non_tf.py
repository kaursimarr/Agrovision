"""Tests for the TensorFlow-free parts. Run from the project root:  python tests/test_non_tf.py"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_utils as du  # noqa: E402
from src.labels import parse_label  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.plots import (plot_confusion_matrix, plot_history, plot_history_comparison,  # noqa: E402
                       plot_metrics_comparison)


def test_labels():
    assert parse_label("Tomato___Early_blight") == ("Tomato", "Early blight", False)
    assert parse_label("Tomato_healthy") == ("Tomato", "Healthy", True)
    assert parse_label("Pepper__bell___Bacterial_spot") == ("Pepper bell", "Bacterial spot", False)
    assert parse_label("Tomato__Target_Spot") == ("Tomato", "Target Spot", False)
    assert parse_label("Corn_(maize)___Common_rust_") == ("Corn (maize)", "Common rust", False)
    assert parse_label("Apple___healthy")[2] is True


def test_data_pipeline(tmp: Path):
    nested = tmp / "PlantVillage"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_dataset.py"), "--out",
                    str(nested / "PlantVillage"), "--per-class", "40"], check=True, capture_output=True)
    root = du.find_dataset_root(nested)                    # must descend through the wrapper folder
    assert root.name == "PlantVillage" and root != nested
    paths, labels, classes = du.list_images(root)
    assert len(classes) == 4 and len(paths) == 160

    split = du.get_or_create_split(root, tmp / "split.json", (0.7, 0.15, 0.15), seed=1)
    n = {k: len(split[k]["paths"]) for k in ("train", "val", "test")}
    assert sum(n.values()) == 160 and n["train"] == 112, n
    sets = [set(split[k]["paths"]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for k in ("train", "val", "test"):                      # stratified: every class in every split
        assert set(split[k]["labels"]) == {0, 1, 2, 3}
    again = du.get_or_create_split(root, tmp / "split.json", (0.7, 0.15, 0.15), seed=1)
    assert again == split                                   # reloaded, not regenerated

    small = du.get_or_create_split(root, tmp / "split2.json", (0.7, 0.15, 0.15), seed=1, max_per_class=20)
    assert sum(len(small[k]["paths"]) for k in ("train", "val", "test")) == 80

    cw = du.compute_class_weights([0] * 90 + [1] * 10, 2)
    assert cw[1] > cw[0]


def test_metrics_and_plots(tmp: Path):
    rng = np.random.default_rng(0)
    classes = ["Tomato___healthy", "Tomato___Early_blight", "Potato___healthy"]
    y_true = rng.integers(0, 3, 200)
    y_pred = np.where(rng.random(200) < 0.85, y_true, rng.integers(0, 3, 200))
    probs = np.eye(3)[y_pred] * 0.8 + 0.1
    m = compute_metrics(y_true, y_pred, classes, probs)
    assert 0.8 < m["accuracy"] <= 1.0 and 0 <= m["f1_macro"] <= 1 and m["top3_accuracy"] == 1.0
    assert np.asarray(m["confusion_matrix"]).sum() == 200

    plot_confusion_matrix(m["confusion_matrix"], classes, tmp / "cm.png")
    plot_confusion_matrix(m["confusion_matrix"], classes, tmp / "cmn.png", normalize=True)
    big = [f"Plant{i}___Disease_{i}" for i in range(38)]       # 38 classes: no cell annotations
    plot_confusion_matrix(np.eye(38) * 5, big, tmp / "cm38.png", normalize=True)
    hist = {"accuracy": [.5, .7, .8, .9], "val_accuracy": [.4, .6, .75, .85],
            "loss": [1.2, .8, .5, .3], "val_loss": [1.3, .9, .6, .4], "phase_boundary": 2}
    plot_history(hist, tmp / "h.png", "Demo")
    plot_history_comparison({"cnn": hist, "resnet50": hist}, tmp / "hc.png")
    plot_metrics_comparison({"cnn": m, "resnet50": m}, tmp / "mc.png")
    for f in ("cm", "cmn", "cm38", "h", "hc", "mc"):
        assert (tmp / f"{f}.png").stat().st_size > 5_000, f


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        test_labels()
        test_data_pipeline(d)
        test_metrics_and_plots(d)
    print("All TensorFlow-free tests passed.")
