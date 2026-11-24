"""Unit and integration tests for the Vision Transformer (ViT) implementation.

Run from project root:
    python tests/test_vit.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
import keras

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_vit, build_model, AddPositionEmbs  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src import config as C  # noqa: E402


def test_vit_architecture():
    print("Testing ViT architecture and dimensions...")
    model = build_vit(num_classes=38)
    assert model.name == "vit"
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 38)

    # Check forward pass with dummy batch
    dummy = tf.random.uniform((2, 224, 224, 3), 0, 255)
    preds = model(dummy, training=False)
    assert preds.shape == (2, 38)
    # Output probabilities should sum to ~1.0
    row_sums = tf.reduce_sum(preds, axis=-1).numpy()
    np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)
    print("  ViT architecture: OK")


def test_vit_serialization(tmp_path: Path):
    print("Testing ViT serialization (.keras saving and loading)...")
    model = build_vit(num_classes=38)
    save_file = tmp_path / "vit_test.keras"
    model.save(save_file)
    assert save_file.exists()

    reloaded = keras.models.load_model(save_file)
    dummy = tf.random.uniform((2, 224, 224, 3), 0, 255)
    p1 = model(dummy, training=False).numpy()
    p2 = reloaded(dummy, training=False).numpy()
    np.testing.assert_allclose(p1, p2, atol=1e-5)
    print("  ViT serialization: OK")


def test_vit_factory():
    print("Testing build_model factory with 'vit'...")
    model, backbone = build_model("vit", num_classes=38)
    assert backbone is None
    assert model.output_shape == (None, 38)
    print("  ViT factory: OK")


def test_vit_predictor(tmp_path: Path):
    print("Testing Predictor integration with ViT model...")
    model = build_vit(num_classes=38)
    save_file = tmp_path / "vit_pred.keras"
    model.save(save_file)

    class_names = [f"Class_{i}" for i in range(38)]
    predictor = Predictor(save_file, class_names=class_names)
    test_img = Image.new("RGB", (224, 224), (80, 160, 80))
    res = predictor.predict(test_img, top_k=3)
    assert len(res) == 3
    assert all("confidence" in r and "class_name" in r for r in res)
    assert res[0]["confidence"] >= res[1]["confidence"] >= res[2]["confidence"]
    print("  ViT Predictor integration: OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_vit_architecture()
        test_vit_serialization(p)
        test_vit_factory()
        test_vit_predictor(p)
    print("All Vision Transformer tests passed successfully!")
